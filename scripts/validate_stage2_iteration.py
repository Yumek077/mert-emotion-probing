"""Validate the fixed split, single-item MERT path, and ID-based alignment."""

from __future__ import annotations

import argparse
import json
import sys
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
    feature_frame_mask,
    load_frozen_mert,
    masked_temporal_mean_pool,
    prepare_mert_inputs,
)
from mert_emotion_probing.splits import (  # noqa: E402
    ALLOWED_SPLITS,
    PRIMARY_SAMPLE_COUNT,
    SPLIT_SEED,
    build_primary_split,
    primary_split_counts,
)


TARGET_SAMPLE_RATE = 24_000
VALIDATION_SAMPLE_IDS = (10, 811, 1640)


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
        "--model-source",
        default="m-a-p/MERT-v1-95M",
        help="Canonical model ID or a local snapshot of that same checkpoint.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "data/raw/deam/verification/module_b_stage2_iteration_validation.json"
        ),
    )
    return parser.parse_args()


def find_audio(audio_root: Path, sample_id: int) -> Path:
    matches = list(audio_root.rglob(f"{sample_id}.mp3"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one MP3 for ID {sample_id}, found {len(matches)}"
        )
    return matches[0]


def describe_target(series: pd.Series) -> dict[str, float | int]:
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "standard_deviation": float(series.std(ddof=1)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def main() -> None:
    args = parse_args()
    mapping = pd.read_csv(args.mapping_csv)
    required_mapping = {
        "song_id",
        "is_full_song",
        "valence_mean",
        "arousal_mean",
    }
    missing_mapping_columns = required_mapping - set(mapping.columns)
    if missing_mapping_columns:
        raise ValueError(
            f"Mapping table is missing columns: {sorted(missing_mapping_columns)}"
        )
    if mapping["song_id"].duplicated().any():
        raise ValueError("Mapping table contains duplicate song IDs")

    primary = mapping.loc[~mapping["is_full_song"].astype(bool)].copy()
    primary["song_id"] = primary["song_id"].astype(int)
    primary_ids = primary["song_id"].tolist()
    if len(primary_ids) != PRIMARY_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {PRIMARY_SAMPLE_COUNT} primary excerpts, received "
            f"{len(primary_ids)}"
        )

    split = pd.read_csv(args.split_csv)
    required_split_columns = ["sample_id", "split"]
    if list(split.columns) != required_split_columns:
        raise ValueError(
            f"Split columns must be {required_split_columns}, received "
            f"{list(split.columns)}"
        )
    split["sample_id"] = split["sample_id"].astype(int)
    expected = build_primary_split(primary_ids, seed=SPLIT_SEED)
    normalized_split = split.sort_values("sample_id").reset_index(drop=True)

    primary_id_set = set(primary_ids)
    split_id_set = set(normalized_split["sample_id"].tolist())
    duplicate_ids = sorted(
        normalized_split.loc[
            normalized_split["sample_id"].duplicated(keep=False), "sample_id"
        ].unique().tolist()
    )
    invalid_split_values = sorted(
        set(normalized_split["split"].astype(str)) - set(ALLOWED_SPLITS)
    )
    observed_counts = {
        name: int((normalized_split["split"] == name).sum())
        for name in ALLOWED_SPLITS
    }
    expected_counts = primary_split_counts(PRIMARY_SAMPLE_COUNT)
    split_integrity = {
        "total_rows": int(len(normalized_split)),
        "expected_total": PRIMARY_SAMPLE_COUNT,
        "unique_sample_ids": int(normalized_split["sample_id"].nunique()),
        "duplicate_sample_ids": duplicate_ids,
        "invalid_split_values": invalid_split_values,
        "missing_primary_ids": sorted(primary_id_set - split_id_set),
        "extra_ids": sorted(split_id_set - primary_id_set),
        "observed_counts": observed_counts,
        "expected_counts": expected_counts,
        "mutually_exclusive": not duplicate_ids,
        "union_equals_primary_population": split_id_set == primary_id_set,
        "every_sample_exactly_once": (
            len(normalized_split) == PRIMARY_SAMPLE_COUNT
            and normalized_split["sample_id"].nunique() == PRIMARY_SAMPLE_COUNT
        ),
        "seed": SPLIT_SEED,
        "reproduces_generation_method_exactly": normalized_split.equals(expected),
    }
    split_integrity_passed = all(
        [
            split_integrity["total_rows"] == PRIMARY_SAMPLE_COUNT,
            split_integrity["unique_sample_ids"] == PRIMARY_SAMPLE_COUNT,
            not duplicate_ids,
            not invalid_split_values,
            not split_integrity["missing_primary_ids"],
            not split_integrity["extra_ids"],
            observed_counts == expected_counts,
            split_integrity["mutually_exclusive"],
            split_integrity["union_equals_primary_population"],
            split_integrity["every_sample_exactly_once"],
            split_integrity["reproduces_generation_method_exactly"],
        ]
    )

    joined = primary.merge(
        normalized_split,
        left_on="song_id",
        right_on="sample_id",
        how="left",
        validate="one_to_one",
    )
    distributions: dict[str, Any] = {}
    for split_name in ALLOWED_SPLITS:
        subset = joined.loc[joined["split"] == split_name]
        distributions[split_name] = {
            "valence": describe_target(subset["valence_mean"]),
            "arousal": describe_target(subset["arousal_mean"]),
        }

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda")
    feature_extractor, model = load_frozen_mert(
        model_id=args.model_source, device=device
    )
    model_state = {
        "model_id": "m-a-p/MERT-v1-95M",
        "loaded_from": str(args.model_source),
        "evaluation_mode": not model.training,
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }

    mapping_by_id = mapping.set_index("song_id", verify_integrity=True)
    split_by_id = normalized_split.set_index("sample_id", verify_integrity=True)
    representations: dict[int, torch.Tensor] = {}
    extraction_rows: list[dict[str, Any]] = []
    for sample_id in VALIDATION_SAMPLE_IDS:
        loaded = load_audio(find_audio(args.audio_root, sample_id))
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
        hidden_states = extract_hidden_states(model, model_inputs)
        input_mask = model_inputs["attention_mask"]
        frame_mask = feature_frame_mask(model, input_mask, hidden_states[0].shape[1])
        pooled = masked_temporal_mean_pool(hidden_states, frame_mask)
        item = pooled[0].detach().cpu()
        representations[sample_id] = item
        extraction_rows.append(
            {
                "sample_id": sample_id,
                "source_waveform_shape": list(loaded.waveform.shape),
                "source_sample_rate": loaded.sample_rate,
                "decoded_duration_seconds": loaded.duration_seconds,
                "resampled_length": int(resampled.numel()),
                "model_input_shape": list(model_inputs["input_values"].shape),
                "input_attention_mask_shape": list(input_mask.shape),
                "input_mask_all_valid": bool(input_mask.bool().all().item()),
                "hidden_state_count": len(hidden_states),
                "each_hidden_state_shape": list(hidden_states[0].shape),
                "valid_feature_frames": int(frame_mask.sum().item()),
                "feature_frame_mask_shape": list(frame_mask.shape),
                "pooled_batch_shape": list(pooled.shape),
                "item_shape": list(item.shape),
                "dtype": str(item.dtype),
                "finite": bool(torch.isfinite(item).all().item()),
                "contains_nan": bool(torch.isnan(item).any().item()),
                "contains_inf": bool(torch.isinf(item).any().item()),
                "all_zero": bool(torch.count_nonzero(item).item() == 0),
                "requires_grad": bool(item.requires_grad),
            }
        )
        del loaded, mono, resampled, model_inputs, hidden_states, frame_mask, pooled
        torch.cuda.empty_cache()

    alignment_rows = []
    for sample_id in VALIDATION_SAMPLE_IDS:
        label_row = mapping_by_id.loc[sample_id]
        split_value = str(split_by_id.loc[sample_id, "split"])
        alignment_rows.append(
            {
                "sample_id": sample_id,
                "representation_shape": list(representations[sample_id].shape),
                "valence": float(label_row["valence_mean"]),
                "arousal": float(label_row["arousal_mean"]),
                "split": split_value,
            }
        )
    alignment_checks = {
        "sample_id_is_authoritative_key": True,
        "selected_ids_unique": len(set(VALIDATION_SAMPLE_IDS))
        == len(VALIDATION_SAMPLE_IDS),
        "representation_ids_match": set(representations)
        == set(VALIDATION_SAMPLE_IDS),
        "no_missing_valence": all(
            not pd.isna(row["valence"]) for row in alignment_rows
        ),
        "no_missing_arousal": all(
            not pd.isna(row["arousal"]) for row in alignment_rows
        ),
        "no_missing_split": all(row["split"] in ALLOWED_SPLITS for row in alignment_rows),
        "row_order_assumed": False,
    }
    extraction_passed = all(
        row["input_mask_all_valid"]
        and row["hidden_state_count"] == 13
        and row["item_shape"] == [13, 768]
        and row["finite"]
        and not row["contains_nan"]
        and not row["contains_inf"]
        and not row["all_zero"]
        and not row["requires_grad"]
        for row in extraction_rows
    )
    alignment_passed = all(
        [
            alignment_checks["sample_id_is_authoritative_key"],
            alignment_checks["selected_ids_unique"],
            alignment_checks["representation_ids_match"],
            alignment_checks["no_missing_valence"],
            alignment_checks["no_missing_arousal"],
            alignment_checks["no_missing_split"],
            not alignment_checks["row_order_assumed"],
        ]
    )
    stage_iteration_passed = all(
        [
            split_integrity_passed,
            model_state["evaluation_mode"],
            model_state["all_parameters_frozen"],
            extraction_passed,
            alignment_passed,
        ]
    )

    output = {
        "stage": "Module B Stage 2 iteration",
        "stage_iteration_passed": stage_iteration_passed,
        "approved_primary_extraction": "single-item inference (batch size 1)",
        "split_artifact": str(args.split_csv),
        "split_integrity": split_integrity,
        "split_integrity_passed": split_integrity_passed,
        "target_distribution_sanity": distributions,
        "model": model_state,
        "single_item_extraction": extraction_rows,
        "single_item_extraction_passed": extraction_passed,
        "alignment_rows": alignment_rows,
        "alignment_checks": alignment_checks,
        "alignment_passed": alignment_passed,
        "embeddings_saved": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
