"""Run a small deterministic real-DEAM to frozen-MERT sanity check.

By default this script processes four fixed DEAM excerpts chosen to exercise
stereo, mono, and multiple source sample-rate paths. Explicit sample IDs can be
provided for smaller inspections. It saves only shapes, labels, numerical
summaries, runtime, and GPU-memory observations—not embeddings.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mert_emotion_probing.audio import (  # noqa: E402
    convert_to_mono,
    load_audio,
    resample_mono,
)
from mert_emotion_probing.mert import (  # noqa: E402
    extract_hidden_states,
    load_frozen_mert,
    prepare_mert_inputs,
    temporal_mean_pool,
)


TARGET_SAMPLE_RATE = 24_000
SELECTED_SAMPLES = {
    10: "ordinary 44.1 kHz stereo excerpt",
    1198: "48 kHz stereo excerpt exercising non-44.1-kHz resampling",
    811: "44.1 kHz mono excerpt exercising the mono passthrough branch",
    1024: "16 kHz stereo excerpt exercising 16-to-24-kHz resampling",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("data/raw/deam/audio"),
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path(
            "data/raw/deam/verification/deam_static_annotations_unified.csv"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "data/raw/deam/verification/deam_mert_e2e_sanity.json"
        ),
    )
    parser.add_argument(
        "--sample-ids",
        type=int,
        nargs="+",
        default=list(SELECTED_SAMPLES),
        help="DEAM song IDs to inspect (the Stage B1 run should use only 1-3).",
    )
    parser.add_argument(
        "--model-source",
        default="m-a-p/MERT-v1-95M",
        help=(
            "Canonical model ID or a local snapshot of that model. A local snapshot "
            "can be used for a reproducible offline run."
        ),
    )
    return parser.parse_args()


def find_audio(audio_root: Path, song_id: int) -> Path:
    matches = list(audio_root.rglob(f"{song_id}.mp3"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one MP3 for ID {song_id}, found {len(matches)}"
        )
    return matches[0]


def load_label_lookup(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    required = {"song_id", "valence_mean", "arousal_mean"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Label table is missing columns: {sorted(missing)}")
    if labels["song_id"].duplicated().any():
        raise ValueError("Label table contains duplicate song IDs")
    return labels.set_index("song_id")


def layer_name(index: int) -> str:
    if index == 0:
        return "pre_transformer_representation"
    return f"transformer_layer_{index}"


def bytes_to_mib(value: int) -> float:
    return value / (1024.0**2)


def tensor_statistics(vector: torch.Tensor) -> dict[str, Any]:
    finite = bool(torch.isfinite(vector).all().item())
    return {
        "finite": finite,
        "contains_nan": bool(torch.isnan(vector).any().item()),
        "contains_inf": bool(torch.isinf(vector).any().item()),
        "all_zero": bool(torch.count_nonzero(vector).item() == 0),
        "mean": float(vector.mean().item()),
        "std": float(vector.std(unbiased=False).item()),
        "l2_norm": float(torch.linalg.vector_norm(vector).item()),
    }


def main() -> None:
    args = parse_args()
    if not 1 <= len(args.sample_ids) <= 4:
        raise ValueError("This sanity check accepts between one and four samples")
    if len(set(args.sample_ids)) != len(args.sample_ids):
        raise ValueError("--sample-ids must not contain duplicates")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda")

    labels = load_label_lookup(args.labels_csv)
    feature_extractor, model = load_frozen_mert(
        model_id=args.model_source, device=device
    )
    model_dtype = next(model.parameters()).dtype
    all_parameters_frozen = all(not parameter.requires_grad for parameter in model.parameters())
    encoder_layer_count = len(model.encoder.layers)
    model_forward_source = inspect.getsource(model.__class__.forward)
    encoder_forward_source = inspect.getsource(model.encoder.__class__.forward)
    implementation_mapping_evidence = {
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "encoder_class": (
            f"{model.encoder.__class__.__module__}."
            f"{model.encoder.__class__.__name__}"
        ),
        "config_num_hidden_layers": int(model.config.num_hidden_layers),
        "config_hidden_size": int(model.config.hidden_size),
        "encoder_layer_module_count": encoder_layer_count,
        "model_forward_passes_feature_projection_output_to_encoder": (
            "hidden_states = self.feature_projection(extract_features)"
            in model_forward_source
            and "encoder_outputs = self.encoder(" in model_forward_source
        ),
        "encoder_records_state_before_each_layer": (
            "for layer in self.layers:" in encoder_forward_source
            and "all_hidden_states = all_hidden_states + (hidden_states,)"
            in encoder_forward_source
        ),
        "encoder_appends_final_state_after_layer_loop": (
            "all_hidden_states = all_hidden_states + (hidden_states,)"
            in encoder_forward_source
        ),
        "interpretation": (
            "The encoder applies positional convolution, layer normalization, and "
            "dropout, then appends the current state immediately before each of its "
            "12 Transformer layers and appends the final state after the loop. "
            "Therefore returned hidden_states[0] is the input to Transformer Layer 1 "
            "(the project's pre-Transformer representation), and hidden_states[i] "
            "for i=1..12 is the output of Transformer Layer i."
        ),
    }

    results: list[dict[str, Any]] = []
    pooled_by_id: dict[int, torch.Tensor] = {}
    mapping_hook_verification: dict[str, Any] | None = None

    for sample_index, song_id in enumerate(args.sample_ids):
        selection_reason = SELECTED_SAMPLES.get(song_id, "explicit CLI selection")
        audio_path = find_audio(args.audio_root, song_id)
        loaded = load_audio(audio_path)
        mono = convert_to_mono(loaded.waveform)
        mono_passthrough_equal = (
            bool(torch.equal(mono, loaded.waveform[0]))
            if loaded.channels == 1
            else None
        )
        resampled = resample_mono(
            mono, loaded.sample_rate, target_rate=TARGET_SAMPLE_RATE
        )

        duration_before = loaded.samples / loaded.sample_rate
        duration_after = resampled.numel() / TARGET_SAMPLE_RATE

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model_inputs = prepare_mert_inputs(
            resampled,
            feature_extractor,
            sample_rate=TARGET_SAMPLE_RATE,
            device=device,
        )
        torch.cuda.synchronize()
        forward_started = time.perf_counter()

        first_layer_input: list[torch.Tensor] = []
        layer_outputs: list[torch.Tensor | None] = [None] * encoder_layer_count
        execution_context: dict[str, bool] = {}
        hook_handles: list[Any] = []
        if sample_index == 0:
            def capture_first_layer_input(
                _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
            ) -> None:
                first_layer_input.append(inputs[0])
                execution_context["grad_enabled_inside_encoder"] = torch.is_grad_enabled()
                execution_context["inference_mode_inside_encoder"] = (
                    torch.is_inference_mode_enabled()
                )

            hook_handles.append(
                model.encoder.layers[0].register_forward_pre_hook(
                    capture_first_layer_input
                )
            )
            for layer_index, layer in enumerate(model.encoder.layers):
                def capture_layer_output(
                    _module: torch.nn.Module,
                    _inputs: tuple[torch.Tensor, ...],
                    output: tuple[torch.Tensor, ...],
                    index: int = layer_index,
                ) -> None:
                    layer_outputs[index] = output[0]

                hook_handles.append(layer.register_forward_hook(capture_layer_output))

        hidden_states = extract_hidden_states(model, model_inputs)
        for handle in hook_handles:
            handle.remove()
        pooled = temporal_mean_pool(hidden_states)
        input_attention_mask = model_inputs.get("attention_mask")
        if input_attention_mask is None:
            feature_attention_mask = torch.ones(
                hidden_states[0].shape[:2], dtype=torch.bool, device=device
            )
        else:
            feature_attention_mask = model._get_feature_vector_attention_mask(
                hidden_states[0].shape[1], input_attention_mask
            )
        valid_counts = feature_attention_mask.sum(dim=1)
        valid_mask_pooled = torch.stack(
            [
                (state * feature_attention_mask.unsqueeze(-1)).sum(dim=1)
                / valid_counts.unsqueeze(-1)
                for state in hidden_states
            ],
            dim=1,
        )
        torch.cuda.synchronize()
        finished = time.perf_counter()

        if sample_index == 0:
            if len(first_layer_input) != 1 or any(
                output is None for output in layer_outputs
            ):
                raise RuntimeError("Layer-mapping hooks did not capture all states")
            pre_transformer_matches = torch.equal(
                hidden_states[0], first_layer_input[0]
            )
            layer_output_matches = [
                torch.equal(hidden_states[index + 1], output)
                for index, output in enumerate(layer_outputs)
                if output is not None
            ]
            mapping_hook_verification = {
                "sample_id": song_id,
                "hidden_state_0_equals_input_to_transformer_layer_1": (
                    pre_transformer_matches
                ),
                "hidden_states_1_to_12_equal_corresponding_layer_outputs": (
                    layer_output_matches
                ),
                "all_mapping_checks_passed": (
                    pre_transformer_matches and all(layer_output_matches)
                ),
                **execution_context,
            }

        item_matrix = pooled[0].detach().cpu()
        pooled_by_id[song_id] = item_matrix
        layer_statistics = {
            str(index): {
                "name": layer_name(index),
                **tensor_statistics(item_matrix[index]),
            }
            for index in range(item_matrix.shape[0])
        }

        label = labels.loc[song_id]
        representation_shapes = [list(state.shape) for state in hidden_states]
        item_representation_shapes = [list(state[0].shape) for state in hidden_states]
        pooled_shapes = [list(pooled[:, index, :].shape) for index in range(13)]
        pooled_item_shapes = [list(item_matrix[index].shape) for index in range(13)]
        representation_dtypes = [str(state.dtype) for state in hidden_states]
        representation_devices = [str(state.device) for state in hidden_states]
        representation_requires_grad = [
            bool(state.requires_grad) for state in hidden_states
        ]
        hidden_state_numerical_checks = [
            {
                "contains_nan": bool(torch.isnan(state).any().item()),
                "contains_inf": bool(torch.isinf(state).any().item()),
                "finite": bool(torch.isfinite(state).all().item()),
            }
            for state in hidden_states
        ]
        result = {
            "song_id": song_id,
            "selection_reason": selection_reason,
            "filepath": audio_path.as_posix(),
            "label": {
                "valence_mean": float(label["valence_mean"]),
                "arousal_mean": float(label["arousal_mean"]),
            },
            "audio_loading": {
                "waveform_shape_channels_samples": list(loaded.waveform.shape),
                "axis_0": "channels",
                "axis_1": "source_samples",
                "dtype": str(loaded.waveform.dtype),
                "source_sample_rate": loaded.sample_rate,
                "channels": loaded.channels,
                "source_samples": loaded.samples,
                "duration_seconds": loaded.duration_seconds,
            },
            "channel_handling": {
                "before_shape": list(loaded.waveform.shape),
                "method": "squeeze single channel"
                if loaded.channels == 1
                else "arithmetic mean across channel dimension 0",
                "after_shape": list(mono.shape),
                "mono_passthrough_values_exactly_equal": mono_passthrough_equal,
            },
            "resampling": {
                "source_sample_rate": loaded.sample_rate,
                "source_samples": loaded.samples,
                "target_sample_rate": TARGET_SAMPLE_RATE,
                "target_samples": int(resampled.numel()),
                "duration_before_seconds": duration_before,
                "duration_after_seconds": duration_after,
                "duration_delta_milliseconds": (duration_after - duration_before)
                * 1000.0,
            },
            "mert_input": {
                "input_values_shape_batch_samples": list(
                    model_inputs["input_values"].shape
                ),
                "batch_axis": "one audio item",
                "sample_axis": "24 kHz waveform samples",
                "dtype": str(model_inputs["input_values"].dtype),
                "device": str(model_inputs["input_values"].device),
                "sampling_rate": TARGET_SAMPLE_RATE,
                "official_feature_extractor_do_normalize": bool(
                    feature_extractor.do_normalize
                ),
                "extra_normalization_applied": False,
                "attention_mask_present": "attention_mask" in model_inputs,
                "attention_mask_shape": (
                    list(input_attention_mask.shape)
                    if input_attention_mask is not None
                    else None
                ),
                "attention_mask_all_ones": (
                    bool(input_attention_mask.bool().all().item())
                    if input_attention_mask is not None
                    else None
                ),
                "valid_input_samples": (
                    int(input_attention_mask.sum().item())
                    if input_attention_mask is not None
                    else int(model_inputs["input_values"].shape[1])
                ),
                "padding_applied": False,
            },
            "representations": {
                "count": len(hidden_states),
                "index_mapping": {
                    str(index): layer_name(index) for index in range(13)
                },
                "shapes": representation_shapes,
                "item_shapes_without_batch_axis": item_representation_shapes,
                "temporal_frames": int(hidden_states[0].shape[1]),
                "valid_temporal_positions": int(hidden_states[0].shape[1]),
                "valid_position_basis": (
                    "Single-item unpadded inference; every decoded/resampled input "
                    "sample is real audio. The input attention mask contains only ones, "
                    "and its model-derived feature mask marks all returned temporal "
                    "positions as valid."
                ),
                "hidden_dimension": int(hidden_states[0].shape[2]),
                "dtypes": representation_dtypes,
                "devices": representation_devices,
                "requires_grad": representation_requires_grad,
                "numerical_checks": hidden_state_numerical_checks,
            },
            "pooling": {
                "operation": "mean over temporal dimension 1",
                "per_representation_shapes": pooled_shapes,
                "per_item_representation_shapes": pooled_item_shapes,
                "stacked_batch_shape": list(pooled.shape),
                "final_item_matrix_shape": list(item_matrix.shape),
                "valid_mask_reference_shape": list(valid_mask_pooled.shape),
                "simple_mean_equals_valid_mask_mean": bool(
                    torch.equal(pooled, valid_mask_pooled)
                ),
                "simple_mean_allclose_to_valid_mask_mean": bool(
                    torch.allclose(
                        pooled, valid_mask_pooled, rtol=1e-6, atol=1e-7
                    )
                ),
                "max_absolute_difference_from_valid_mask_mean": float(
                    (pooled - valid_mask_pooled).abs().max().item()
                ),
            },
            "numerical_checks_by_representation": layer_statistics,
            "runtime_and_gpu": {
                "input_preparation_seconds": forward_started - started,
                "forward_and_pooling_seconds": finished - forward_started,
                "total_seconds": finished - started,
                "peak_allocated_mib": bytes_to_mib(
                    torch.cuda.max_memory_allocated()
                ),
                "peak_reserved_mib": bytes_to_mib(torch.cuda.max_memory_reserved()),
            },
        }
        results.append(result)

        del (
            hidden_states,
            pooled,
            valid_mask_pooled,
            feature_attention_mask,
            model_inputs,
            resampled,
            mono,
            loaded,
        )
        torch.cuda.empty_cache()

    cross_sample_checks: list[dict[str, Any]] = []
    for first_id, second_id in combinations(args.sample_ids, 2):
        first = pooled_by_id[first_id].reshape(-1)
        second = pooled_by_id[second_id].reshape(-1)
        distance = float(torch.linalg.vector_norm(first - second).item())
        cosine = float(
            torch.nn.functional.cosine_similarity(first, second, dim=0).item()
        )
        cross_sample_checks.append(
            {
                "song_ids": [first_id, second_id],
                "flattened_13x768_l2_distance": distance,
                "flattened_13x768_cosine_similarity": cosine,
                "exactly_equal": bool(torch.equal(first, second)),
            }
        )

    output = {
        "model": {
            "model_id": "m-a-p/MERT-v1-95M",
            "loaded_from": str(args.model_source),
            "device": str(device),
            "dtype": str(model_dtype),
            "evaluation_mode": not model.training,
            "all_parameters_frozen": all_parameters_frozen,
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "hidden_state_layer_mapping_evidence": implementation_mapping_evidence,
        "hidden_state_layer_mapping_hook_verification": mapping_hook_verification,
        "selected_sample_ids": list(args.sample_ids),
        "samples": results,
        "cross_sample_checks": cross_sample_checks,
        "embeddings_saved": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
