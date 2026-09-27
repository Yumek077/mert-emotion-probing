"""Run a small deterministic real-DEAM to frozen-MERT sanity check.

This script processes four fixed DEAM excerpts chosen to exercise stereo,
mono, and multiple source sample-rate paths. It saves only shapes, labels,
numerical summaries, runtime, and GPU-memory observations—not embeddings.
"""

from __future__ import annotations

import argparse
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
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda")

    labels = load_label_lookup(args.labels_csv)
    feature_extractor, model = load_frozen_mert(device=device)
    model_dtype = next(model.parameters()).dtype
    all_parameters_frozen = all(not parameter.requires_grad for parameter in model.parameters())

    results: list[dict[str, Any]] = []
    pooled_by_id: dict[int, torch.Tensor] = {}

    for song_id, selection_reason in SELECTED_SAMPLES.items():
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
        hidden_states = extract_hidden_states(model, model_inputs)
        pooled = temporal_mean_pool(hidden_states)
        torch.cuda.synchronize()
        finished = time.perf_counter()

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
        pooled_shapes = [list(pooled[:, index, :].shape) for index in range(13)]
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
            },
            "representations": {
                "count": len(hidden_states),
                "index_mapping": {
                    str(index): layer_name(index) for index in range(13)
                },
                "shapes": representation_shapes,
                "temporal_frames": int(hidden_states[0].shape[1]),
                "hidden_dimension": int(hidden_states[0].shape[2]),
            },
            "pooling": {
                "operation": "mean over temporal dimension 1",
                "per_representation_shapes": pooled_shapes,
                "stacked_batch_shape": list(pooled.shape),
                "final_item_matrix_shape": list(item_matrix.shape),
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

        del hidden_states, pooled, model_inputs, resampled, mono, loaded
        torch.cuda.empty_cache()

    cross_sample_checks: list[dict[str, Any]] = []
    for first_id, second_id in combinations(SELECTED_SAMPLES, 2):
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
            "device": str(device),
            "dtype": str(model_dtype),
            "evaluation_mode": not model.training,
            "all_parameters_frozen": all_parameters_frozen,
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "selected_sample_ids": list(SELECTED_SAMPLES),
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
