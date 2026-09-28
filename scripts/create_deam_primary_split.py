"""Create or verify the fixed seed-42 split for 1,744 primary DEAM excerpts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mert_emotion_probing.splits import (  # noqa: E402
    PRIMARY_SAMPLE_COUNT,
    SPLIT_SEED,
    build_primary_split,
    primary_split_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=Path("data/raw/deam/verification/deam_item_mapping.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/metadata/deam_primary_split_seed42.csv"),
    )
    return parser.parse_args()


def primary_ids_from_mapping(path: Path) -> list[int]:
    mapping = pd.read_csv(path)
    required = {"song_id", "is_full_song"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Mapping table is missing columns: {sorted(missing)}")
    if mapping["song_id"].duplicated().any():
        raise ValueError("Mapping table contains duplicate song IDs")
    primary = mapping.loc[~mapping["is_full_song"].astype(bool), "song_id"]
    ids = primary.astype(int).tolist()
    if len(ids) != PRIMARY_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {PRIMARY_SAMPLE_COUNT} primary excerpts, received {len(ids)}"
        )
    return ids


def main() -> None:
    args = parse_args()
    primary_ids = primary_ids_from_mapping(args.mapping_csv)
    expected = build_primary_split(primary_ids, seed=SPLIT_SEED)

    action = "created"
    if args.output_csv.exists():
        existing = pd.read_csv(args.output_csv)
        expected_columns = ["sample_id", "split"]
        if list(existing.columns) != expected_columns:
            raise RuntimeError(
                f"Existing split has columns {list(existing.columns)}, "
                f"expected {expected_columns}; refusing to overwrite"
            )
        existing = existing.sort_values("sample_id").reset_index(drop=True)
        if not existing.equals(expected):
            raise RuntimeError(
                "Existing split differs from the deterministic seed-42 result; "
                "refusing to overwrite"
            )
        action = "verified_existing"
    else:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        expected.to_csv(args.output_csv, index=False, lineterminator="\n")

    summary = {
        "action": action,
        "output_csv": str(args.output_csv),
        "population": "1,744 verified primary DEAM excerpts",
        "identity_key": "sample_id",
        "method": (
            "Sort unique primary sample IDs, permute once with "
            "numpy.random.default_rng(42), assign contiguous train/validation/test "
            "segments, then sort the saved artifact by sample_id."
        ),
        "rounding": (
            "Round train and validation targets half-up; assign the exact remaining "
            "samples to test."
        ),
        "seed": SPLIT_SEED,
        "counts": primary_split_counts(PRIMARY_SAMPLE_COUNT),
        "total": len(expected),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
