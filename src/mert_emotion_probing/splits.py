"""Deterministic sample-level split utilities for the primary DEAM excerpts."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


PRIMARY_SAMPLE_COUNT = 1_744
SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
ALLOWED_SPLITS = ("train", "validation", "test")


def round_half_up(value: float) -> int:
    """Round a non-negative value to the nearest integer with halves upward."""

    if value < 0:
        raise ValueError("Split counts must be non-negative")
    return int(math.floor(value + 0.5))


def primary_split_counts(total: int) -> dict[str, int]:
    """Return deterministic 70/15/15 counts whose sum is exactly ``total``."""

    if total <= 0:
        raise ValueError("Split population must be positive")
    train = round_half_up(total * TRAIN_RATIO)
    validation = round_half_up(total * VALIDATION_RATIO)
    test = total - train - validation
    if test <= 0:
        raise ValueError("Rounding produced a non-positive test split")
    return {"train": train, "validation": validation, "test": test}


def build_primary_split(
    sample_ids: Sequence[int], seed: int = SPLIT_SEED
) -> pd.DataFrame:
    """Assign sorted unique sample IDs to one fixed random sample-level split."""

    ids = [int(sample_id) for sample_id in sample_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("Primary sample IDs must be unique")
    if len(ids) != PRIMARY_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {PRIMARY_SAMPLE_COUNT} primary IDs, received {len(ids)}"
        )

    sorted_ids = np.asarray(sorted(ids), dtype=np.int64)
    shuffled_ids = np.random.default_rng(seed).permutation(sorted_ids)
    counts = primary_split_counts(len(ids))
    train_end = counts["train"]
    validation_end = train_end + counts["validation"]

    assignments = pd.DataFrame(
        {
            "sample_id": shuffled_ids,
            "split": (
                ["train"] * counts["train"]
                + ["validation"] * counts["validation"]
                + ["test"] * counts["test"]
            ),
        }
    )
    if validation_end + counts["test"] != len(assignments):
        raise RuntimeError("Split boundaries do not cover the complete population")
    return assignments.sort_values("sample_id").reset_index(drop=True)
