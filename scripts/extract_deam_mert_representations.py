"""Extract and verify the canonical frozen-MERT cache for primary DEAM excerpts.

The primary path is deliberately single-item inference. Each successful sample
is saved atomically as a small intermediate part so an interrupted run can
resume without recomputing completed samples. The final cache is consolidated
only after all 1,744 primary samples pass validation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
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
    EXPECTED_HIDDEN_DIMENSION,
    EXPECTED_REPRESENTATIONS,
    extract_hidden_states,
    feature_frame_mask,
    load_frozen_mert,
    masked_temporal_mean_pool,
    prepare_mert_inputs,
)
from mert_emotion_probing.splits import (  # noqa: E402
    ALLOWED_SPLITS,
    PRIMARY_SAMPLE_COUNT,
)


TARGET_SAMPLE_RATE = 24_000
FRESH_VERIFICATION_IDS = (10, 811, 1640)
FRESH_RTOL = 1e-6
FRESH_ATOL = 1e-7


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-root", type=Path, default=Path("data/raw/deam/audio")
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=Path("data/raw/deam/verification/deam_item_mapping.csv"),
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=Path("data/metadata/deam_primary_split_seed42.csv"),
    )
    parser.add_argument(
        "--output-cache",
        type=Path,
        default=Path(
            "outputs/embeddings/deam_mert_v1_95m_meanpool_all_layers.pt"
        ),
    )
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=None,
        help="Resume parts directory; defaults to <output stem>.parts.",
    )
    parser.add_argument(
        "--model-source",
        default="m-a-p/MERT-v1-95M",
        help="Canonical model ID or a local snapshot of that same checkpoint.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_protocol_tables(
    mapping_path: Path, split_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    mapping = pd.read_csv(mapping_path)
    required_mapping = {
        "song_id",
        "is_full_song",
        "valence_mean",
        "arousal_mean",
    }
    missing_columns = required_mapping - set(mapping.columns)
    if missing_columns:
        raise ValueError(
            f"Mapping table is missing columns: {sorted(missing_columns)}"
        )
    if mapping["song_id"].duplicated().any():
        raise ValueError("Mapping table contains duplicate song IDs")
    if mapping[["valence_mean", "arousal_mean"]].isna().any().any():
        raise ValueError("Mapping table contains missing primary targets")

    primary = mapping.loc[~mapping["is_full_song"].astype(bool)].copy()
    primary["song_id"] = primary["song_id"].astype(int)
    primary_ids = sorted(primary["song_id"].tolist())
    if len(primary_ids) != PRIMARY_SAMPLE_COUNT or len(set(primary_ids)) != len(
        primary_ids
    ):
        raise ValueError(
            f"Expected {PRIMARY_SAMPLE_COUNT} unique primary IDs, received "
            f"{len(primary_ids)} rows and {len(set(primary_ids))} unique IDs"
        )

    split = pd.read_csv(split_path)
    if list(split.columns) != ["sample_id", "split"]:
        raise ValueError("Split artifact must contain only sample_id,split")
    split["sample_id"] = split["sample_id"].astype(int)
    if len(split) != PRIMARY_SAMPLE_COUNT:
        raise ValueError(f"Split artifact has {len(split)} rows")
    if split["sample_id"].duplicated().any():
        raise ValueError("Split artifact contains duplicate sample IDs")
    if set(split["sample_id"]) != set(primary_ids):
        raise ValueError("Split ID set does not equal the primary DEAM population")
    invalid_splits = sorted(set(split["split"]) - set(ALLOWED_SPLITS))
    if invalid_splits:
        raise ValueError(f"Invalid split values: {invalid_splits}")
    expected_counts = {"train": 1221, "validation": 262, "test": 261}
    observed_counts = {
        name: int((split["split"] == name).sum()) for name in ALLOWED_SPLITS
    }
    if observed_counts != expected_counts:
        raise ValueError(
            f"Split counts {observed_counts} do not match {expected_counts}"
        )
    return primary.set_index("song_id"), split.set_index("sample_id"), primary_ids


def index_audio(audio_root: Path, expected_ids: list[int]) -> dict[int, Path]:
    expected = set(expected_ids)
    indexed: dict[int, Path] = {}
    duplicates: list[int] = []
    for path in audio_root.rglob("*.mp3"):
        try:
            sample_id = int(path.stem)
        except ValueError:
            continue
        if sample_id not in expected:
            continue
        if sample_id in indexed:
            duplicates.append(sample_id)
        indexed[sample_id] = path
    missing = sorted(expected - set(indexed))
    if duplicates or missing:
        raise ValueError(
            f"Audio index failure: duplicate IDs={sorted(set(duplicates))}, "
            f"missing IDs={missing}"
        )
    return indexed


def representation_is_valid(representation: torch.Tensor) -> bool:
    return bool(
        representation.shape
        == (EXPECTED_REPRESENTATIONS, EXPECTED_HIDDEN_DIMENSION)
        and representation.dtype == torch.float32
        and torch.isfinite(representation).all().item()
        and torch.count_nonzero(representation).item() > 0
    )


def load_valid_part(path: Path, expected_id: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("sample_id") != expected_id:
        raise RuntimeError(f"Invalid sample identity in resume part {path}")
    representation = payload.get("representation")
    if not isinstance(representation, torch.Tensor) or not representation_is_valid(
        representation
    ):
        raise RuntimeError(f"Invalid representation in resume part {path}")
    return representation


def extract_one(
    sample_id: int,
    audio_path: Path,
    feature_extractor: Any,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    loaded = load_audio(audio_path)
    mono = convert_to_mono(loaded.waveform)
    resampled = resample_mono(
        mono, loaded.sample_rate, target_rate=TARGET_SAMPLE_RATE
    )
    model_inputs = prepare_mert_inputs(
        resampled,
        feature_extractor,
        sample_rate=TARGET_SAMPLE_RATE,
        device=device,
    )
    input_mask = model_inputs.get("attention_mask")
    if input_mask is None or not bool(input_mask.bool().all().item()):
        raise RuntimeError(
            f"Single-item input mask for {sample_id} is absent or contains padding"
        )
    hidden_states = extract_hidden_states(model, model_inputs)
    frame_mask = feature_frame_mask(model, input_mask, hidden_states[0].shape[1])
    if not bool(frame_mask.all().item()):
        raise RuntimeError(
            f"Single-item feature mask for {sample_id} contains invalid frames"
        )
    pooled = masked_temporal_mean_pool(hidden_states, frame_mask)
    representation = pooled[0].detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not representation_is_valid(representation):
        raise RuntimeError(f"Invalid pooled representation for {sample_id}")
    observation = {
        "source_sample_rate": loaded.sample_rate,
        "source_channels": loaded.channels,
        "source_samples": loaded.samples,
        "decoded_duration_seconds": loaded.duration_seconds,
        "resampled_samples": int(resampled.numel()),
        "valid_feature_frames": int(frame_mask.sum().item()),
    }
    return representation, observation


def checkpoint_identity(model_source: str) -> str | None:
    name = Path(model_source).name
    if len(name) == 40 and all(character in "0123456789abcdef" for character in name):
        return name
    return None


def build_metadata(
    args: argparse.Namespace,
    model: torch.nn.Module,
    split: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": "DEAM",
        "population": "1,744 approximately 45-second primary excerpts",
        "sample_count": PRIMARY_SAMPLE_COUNT,
        "identity_key": "sample_id",
        "ordering": "ascending sample_id",
        "model_name": "m-a-p/MERT-v1-95M",
        "checkpoint_snapshot": checkpoint_identity(str(args.model_source)),
        "loaded_from": str(args.model_source),
        "target_sample_rate_hz": TARGET_SAMPLE_RATE,
        "mono_conversion": "arithmetic channel mean; mono values unchanged",
        "decoded_duration_policy": (
            "preserve the actually decoded excerpt; no forced 45-second crop or pad"
        ),
        "extraction_mode": "single-item inference",
        "batch_size": 1,
        "cross_sample_raw_waveform_padding": False,
        "model_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "model_evaluation_mode": not model.training,
        "inference_mode": True,
        "representation_levels": ["pre_transformer"]
        + [f"transformer_layer_{index}" for index in range(1, 13)],
        "hidden_dimension": EXPECTED_HIDDEN_DIMENSION,
        "pooling": "temporal mean over valid feature frames",
        "dtype": "torch.float32",
        "split_artifact": str(args.split_csv),
        "split_counts": {
            name: int((split["split"] == name).sum()) for name in ALLOWED_SPLITS
        },
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "created_utc": utc_now(),
    }


def validate_final_cache(
    cache: dict[str, Any], expected_ids: list[int]
) -> dict[str, Any]:
    representations = cache.get("representations")
    sample_ids = cache.get("sample_ids")
    metadata = cache.get("metadata")
    if not isinstance(representations, torch.Tensor):
        raise RuntimeError("Cache representations are not a tensor")
    if not isinstance(sample_ids, torch.Tensor):
        raise RuntimeError("Cache sample_ids are not a tensor")
    if not isinstance(metadata, dict):
        raise RuntimeError("Cache metadata is not a dictionary")

    expected_shape = (
        PRIMARY_SAMPLE_COUNT,
        EXPECTED_REPRESENTATIONS,
        EXPECTED_HIDDEN_DIMENSION,
    )
    ids = [int(value) for value in sample_ids.tolist()]
    all_zero_rows = (
        torch.count_nonzero(representations.reshape(PRIMARY_SAMPLE_COUNT, -1), dim=1)
        == 0
    )
    checks = {
        "representation_shape": list(representations.shape),
        "expected_representation_shape": list(expected_shape),
        "sample_ids_shape": list(sample_ids.shape),
        "expected_sample_ids_shape": [PRIMARY_SAMPLE_COUNT],
        "dtype": str(representations.dtype),
        "sample_id_dtype": str(sample_ids.dtype),
        "sample_ids_unique": len(set(ids)) == PRIMARY_SAMPLE_COUNT,
        "sample_ids_ascending": ids == sorted(ids),
        "missing_primary_ids": sorted(set(expected_ids) - set(ids)),
        "extra_ids": sorted(set(ids) - set(expected_ids)),
        "contains_nan": bool(torch.isnan(representations).any().item()),
        "contains_inf": bool(torch.isinf(representations).any().item()),
        "all_zero_sample_ids": [
            ids[index]
            for index in torch.nonzero(all_zero_rows, as_tuple=False).flatten().tolist()
        ],
    }
    checks["passed"] = all(
        [
            tuple(representations.shape) == expected_shape,
            tuple(sample_ids.shape) == (PRIMARY_SAMPLE_COUNT,),
            representations.dtype == torch.float32,
            sample_ids.dtype == torch.int64,
            checks["sample_ids_unique"],
            checks["sample_ids_ascending"],
            not checks["missing_primary_ids"],
            not checks["extra_ids"],
            not checks["contains_nan"],
            not checks["contains_inf"],
            not checks["all_zero_sample_ids"],
        ]
    )
    if not checks["passed"]:
        raise RuntimeError(f"Final cache integrity failed: {checks}")
    return checks


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    primary, split, primary_ids = load_protocol_tables(
        args.mapping_csv, args.split_csv
    )
    audio_paths = index_audio(args.audio_root, primary_ids)
    output_dir = args.output_cache.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    write_test = output_dir / ".stage3_write_test.tmp"
    write_test.write_text("ok", encoding="utf-8")
    write_test.unlink()

    expected_tensor_bytes = (
        PRIMARY_SAMPLE_COUNT
        * EXPECTED_REPRESENTATIONS
        * EXPECTED_HIDDEN_DIMENSION
        * torch.tensor([], dtype=torch.float32).element_size()
    )
    disk = shutil.disk_usage(output_dir)
    minimum_reasonable_free_bytes = max(expected_tensor_bytes * 4, 1 << 30)
    if disk.free < minimum_reasonable_free_bytes:
        raise RuntimeError(
            f"Insufficient free space: {disk.free} bytes available, "
            f"minimum {minimum_reasonable_free_bytes}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda")
    feature_extractor, model = load_frozen_mert(
        model_id=args.model_source, device=device
    )
    result = {
        "passed": True,
        "primary_sample_count": len(primary_ids),
        "primary_ids_unique": len(set(primary_ids)) == len(primary_ids),
        "audio_file_count": len(audio_paths),
        "labels_complete": bool(
            primary[["valence_mean", "arousal_mean"]].notna().all().all()
        ),
        "split_count": len(split),
        "split_counts": {
            name: int((split["split"] == name).sum()) for name in ALLOWED_SPLITS
        },
        "model_loaded": True,
        "model_evaluation_mode": not model.training,
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "model_dtype": str(next(model.parameters()).dtype),
        "cuda_device": torch.cuda.get_device_name(0),
        "feature_extractor_sampling_rate": int(feature_extractor.sampling_rate),
        "output_directory": str(output_dir),
        "output_directory_writable": True,
        "disk_free_bytes": disk.free,
        "expected_final_tensor_bytes": expected_tensor_bytes,
        "minimum_reasonable_free_bytes": minimum_reasonable_free_bytes,
    }
    del model, feature_extractor
    torch.cuda.empty_cache()
    return result


def main() -> None:
    stage_started = time.perf_counter()
    args = parse_args()
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    if args.parts_dir is None:
        args.parts_dir = Path(str(args.output_cache.with_suffix("")) + ".parts")

    preflight = run_preflight(args)
    print(json.dumps({"preflight": preflight}, indent=2), flush=True)
    if args.preflight_only:
        return

    primary, split, primary_ids = load_protocol_tables(
        args.mapping_csv, args.split_csv
    )
    audio_paths = index_audio(args.audio_root, primary_ids)
    args.parts_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.parts_dir / "state.json"
    manifest_path = args.output_cache.with_suffix(".manifest.json")

    unexpected_parts = []
    for path in args.parts_dir.glob("*.pt"):
        try:
            part_id = int(path.stem)
        except ValueError:
            unexpected_parts.append(path.name)
            continue
        if part_id not in set(primary_ids):
            unexpected_parts.append(path.name)
    if unexpected_parts:
        raise RuntimeError(f"Unexpected resume parts: {unexpected_parts}")

    valid_existing_ids = []
    for sample_id in primary_ids:
        part_path = args.parts_dir / f"{sample_id}.pt"
        if part_path.exists():
            load_valid_part(part_path, sample_id)
            valid_existing_ids.append(sample_id)

    previous_state: dict[str, Any] = {}
    if state_path.exists():
        previous_state = json.loads(state_path.read_text(encoding="utf-8"))
    cumulative_seconds = float(previous_state.get("cumulative_extraction_seconds", 0.0))
    initial_started_utc = previous_state.get("initial_started_utc", utc_now())
    resumed = bool(valid_existing_ids)

    device = torch.device("cuda")
    feature_extractor, model = load_frozen_mert(
        model_id=args.model_source, device=device
    )
    run_started = time.perf_counter()
    newly_extracted = 0
    failures: dict[str, str] = {}
    observations: dict[str, Any] = {}

    for position, sample_id in enumerate(primary_ids, start=1):
        part_path = args.parts_dir / f"{sample_id}.pt"
        if part_path.exists():
            continue
        sample_started = time.perf_counter()
        try:
            representation, observation = extract_one(
                sample_id,
                audio_paths[sample_id],
                feature_extractor,
                model,
                device,
            )
            atomic_torch_save(
                part_path,
                {
                    "sample_id": sample_id,
                    "representation": representation,
                },
            )
            observations[str(sample_id)] = observation
            newly_extracted += 1
        except Exception as error:  # Preserve progress and report every failure.
            failures[str(sample_id)] = f"{type(error).__name__}: {error}"

        completed = len(valid_existing_ids) + newly_extracted
        elapsed_this_run = time.perf_counter() - run_started
        average = elapsed_this_run / newly_extracted if newly_extracted else None
        remaining = PRIMARY_SAMPLE_COUNT - completed
        eta = average * remaining if average is not None else None
        if (
            newly_extracted % args.progress_every == 0
            or failures
            or completed == PRIMARY_SAMPLE_COUNT
        ):
            current_state = {
                "initial_started_utc": initial_started_utc,
                "last_updated_utc": utc_now(),
                "completed": completed,
                "total": PRIMARY_SAMPLE_COUNT,
                "newly_extracted_this_run": newly_extracted,
                "resumed": resumed,
                "cumulative_extraction_seconds": cumulative_seconds
                + elapsed_this_run,
                "failures": failures,
            }
            atomic_write_json(state_path, current_state)
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": PRIMARY_SAMPLE_COUNT,
                        "current_sample_id": sample_id,
                        "sample_seconds": time.perf_counter() - sample_started,
                        "elapsed_this_run_seconds": elapsed_this_run,
                        "estimated_remaining_seconds": eta,
                        "failure_count": len(failures),
                    }
                ),
                flush=True,
            )
        if failures:
            break

    elapsed_this_run = time.perf_counter() - run_started
    cumulative_seconds += elapsed_this_run
    completed_ids = [
        sample_id
        for sample_id in primary_ids
        if (args.parts_dir / f"{sample_id}.pt").exists()
    ]
    final_state = {
        "initial_started_utc": initial_started_utc,
        "last_updated_utc": utc_now(),
        "completed": len(completed_ids),
        "total": PRIMARY_SAMPLE_COUNT,
        "newly_extracted_this_run": newly_extracted,
        "resumed": resumed,
        "cumulative_extraction_seconds": cumulative_seconds,
        "failures": failures,
    }
    atomic_write_json(state_path, final_state)
    if failures:
        raise RuntimeError(
            f"Extraction stopped after {len(completed_ids)} completed samples: "
            f"{failures}"
        )
    if len(completed_ids) != PRIMARY_SAMPLE_COUNT:
        raise RuntimeError(
            f"Extraction incomplete: {len(completed_ids)}/{PRIMARY_SAMPLE_COUNT}"
        )

    representations = torch.empty(
        (
            PRIMARY_SAMPLE_COUNT,
            EXPECTED_REPRESENTATIONS,
            EXPECTED_HIDDEN_DIMENSION,
        ),
        dtype=torch.float32,
    )
    for index, sample_id in enumerate(primary_ids):
        representations[index] = load_valid_part(
            args.parts_dir / f"{sample_id}.pt", sample_id
        )
    sample_ids_tensor = torch.tensor(primary_ids, dtype=torch.int64)
    metadata = build_metadata(args, model, split)
    cache = {
        "representations": representations,
        "sample_ids": sample_ids_tensor,
        "metadata": metadata,
    }
    integrity = validate_final_cache(cache, primary_ids)
    atomic_torch_save(args.output_cache, cache)

    loaded_cache = torch.load(
        args.output_cache, map_location="cpu", weights_only=False
    )
    serialization_integrity = validate_final_cache(loaded_cache, primary_ids)
    id_to_index = {
        int(sample_id): index
        for index, sample_id in enumerate(loaded_cache["sample_ids"].tolist())
    }

    fresh_checks = []
    mapping_by_id = primary
    split_by_id = split
    traceability_rows = []
    for sample_id in FRESH_VERIFICATION_IDS:
        fresh, _ = extract_one(
            sample_id,
            audio_paths[sample_id],
            feature_extractor,
            model,
            device,
        )
        cached = loaded_cache["representations"][id_to_index[sample_id]]
        difference = (fresh - cached).abs()
        fresh_checks.append(
            {
                "sample_id": sample_id,
                "cache_shape": list(cached.shape),
                "fresh_shape": list(fresh.shape),
                "max_absolute_difference": float(difference.max().item()),
                "allclose": bool(
                    torch.allclose(
                        fresh, cached, rtol=FRESH_RTOL, atol=FRESH_ATOL
                    )
                ),
                "rtol": FRESH_RTOL,
                "atol": FRESH_ATOL,
            }
        )
        traceability_rows.append(
            {
                "sample_id": sample_id,
                "cache_index": id_to_index[sample_id],
                "representation_shape": list(cached.shape),
                "valence": float(mapping_by_id.loc[sample_id, "valence_mean"]),
                "arousal": float(mapping_by_id.loc[sample_id, "arousal_mean"]),
                "split": str(split_by_id.loc[sample_id, "split"]),
            }
        )

    fresh_verification_passed = all(item["allclose"] for item in fresh_checks)
    traceability_passed = (
        {row["split"] for row in traceability_rows}
        == {"train", "validation", "test"}
        and all(row["representation_shape"] == [13, 768] for row in traceability_rows)
    )
    cache_size_bytes = args.output_cache.stat().st_size
    intermediate_size_bytes = sum(
        path.stat().st_size for path in args.parts_dir.rglob("*") if path.is_file()
    )
    stage_passed = all(
        [
            integrity["passed"],
            serialization_integrity["passed"],
            fresh_verification_passed,
            traceability_passed,
            not failures,
        ]
    )
    manifest = {
        "stage_passed": stage_passed,
        "preflight": preflight,
        "resume": final_state,
        "canonical_cache": str(args.output_cache),
        "cache_size_bytes": cache_size_bytes,
        "intermediate_parts_directory": str(args.parts_dir),
        "intermediate_size_bytes": intermediate_size_bytes,
        "cache_integrity": integrity,
        "serialization_integrity": serialization_integrity,
        "fresh_reextraction": fresh_checks,
        "fresh_reextraction_passed": fresh_verification_passed,
        "traceability": traceability_rows,
        "traceability_passed": traceability_passed,
        "runtime": {
            "initial_started_utc": initial_started_utc,
            "completed_utc": utc_now(),
            "cumulative_extraction_seconds": cumulative_seconds,
            "average_seconds_per_sample": cumulative_seconds
            / PRIMARY_SAMPLE_COUNT,
            "newly_extracted_this_run": newly_extracted,
            "resumed": resumed,
            "total_seconds_this_invocation": time.perf_counter() - stage_started,
        },
        "failed_samples": failures,
        "embeddings_git_tracked": False,
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    if not stage_passed:
        raise RuntimeError("Stage 3 verification did not pass")


if __name__ == "__main__":
    main()
