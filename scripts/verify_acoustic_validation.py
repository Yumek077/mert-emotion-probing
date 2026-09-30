"""Independently verify Module E Stage 2 artifacts without evaluating Test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mert_emotion_probing.probing import (  # noqa: E402
    ALPHA_GRID, TARGET_COLUMNS, regression_metrics,
    select_candidate_by_validation_r2,
)


def main() -> None:
    result = json.loads(Path("outputs/results/module_e_stage2_validation.json").read_text(encoding="utf-8"))
    gate = json.loads(Path("outputs/results/module_e_stage2_pre_test_verification.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv("outputs/results/module_e_stage2_validation_predictions.csv")
    split = pd.read_csv("data/metadata/deam_primary_split_seed42.csv")
    labels = pd.read_csv("data/raw/deam/verification/deam_item_mapping.csv", usecols=["song_id", "is_full_song", "valence_mean", "arousal_mean"])
    validation_ids = set(split.loc[split.split == "validation", "sample_id"].astype(int))
    test_ids = set(split.loc[split.split == "test", "sample_id"].astype(int))
    train_ids = set(split.loc[split.split == "train", "sample_id"].astype(int))
    if len(predictions) != 524 or set(predictions.split) != {"validation"}:
        raise AssertionError("Wrong Validation prediction rows")
    if set(predictions.sample_id) != validation_ids or set(predictions.sample_id) & test_ids:
        raise AssertionError("Prediction IDs are not exactly Validation")
    if predictions.duplicated(["sample_id", "target"]).any():
        raise AssertionError("Duplicate prediction keys")
    if not gate["passed"] or not all(gate["checks"].values()) or gate["test_evaluation_performed"]:
        raise AssertionError("Pre-Test gate is not closed and passing")
    if result["protocol"]["mert_comparator"] != "transformer_layer_12":
        raise AssertionError("MERT comparator changed")
    primary = labels.loc[~labels.is_full_song.astype(bool)].set_index("song_id")
    for target, column in TARGET_COLUMNS.items():
        saved = result["targets"][target]
        rows = saved["candidate_validation_r2"]
        complete = saved["candidate_validation_metrics"]
        if tuple(row["alpha"] for row in rows) != ALPHA_GRID or tuple(row["alpha"] for row in complete) != ALPHA_GRID:
            raise AssertionError(f"{target} alpha grid changed")
        for row, detailed in zip(rows, complete):
            if not np.isclose(row["validation_r2"], detailed["r2"], rtol=0, atol=1e-15):
                raise AssertionError(f"{target} candidate metrics mismatch")
        selected, best, exact_ties = select_candidate_by_validation_r2(rows)
        if selected["alpha"] != saved["selected_alpha"] or best != saved["best_validation_r2"] or len(exact_ties) != saved["exact_tie_count"]:
            raise AssertionError(f"{target} selection rule mismatch")
        frame = predictions.loc[predictions.target == target].set_index("sample_id")
        if set(frame.index) != validation_ids:
            raise AssertionError(f"{target} Validation IDs differ")
        expected_y = primary.loc[frame.index, column].to_numpy(dtype=np.float64)
        if not np.allclose(frame.y_true, expected_y, rtol=0, atol=1e-14):
            raise AssertionError(f"{target} labels misaligned")
        train_mean = float(primary.loc[sorted(train_ids), column].mean())
        if not np.isclose(train_mean, saved["baseline"]["train_target_mean"], rtol=0, atol=1e-14):
            raise AssertionError(f"{target} Train mean mismatch")
        if not np.allclose(frame.baseline_prediction, train_mean, rtol=0, atol=1e-14):
            raise AssertionError(f"{target} baseline prediction mismatch")
        for observed, expected in ((regression_metrics(expected_y, frame.prediction.to_numpy()), saved["selected_validation_metrics"]), (regression_metrics(expected_y, frame.baseline_prediction.to_numpy()), saved["baseline"]["validation_metrics"])):
            for metric in ("mae", "r2", "pearson_r"):
                if observed[metric] is None:
                    if expected[metric] is not None:
                        raise AssertionError(f"{target} {metric} null mismatch")
                elif not np.isclose(observed[metric], expected[metric], rtol=0, atol=1e-12):
                    raise AssertionError(f"{target} {metric} mismatch")
        if saved["standardization"]["fit_split"] != "train" or saved["ridge"]["fit_split"] != "train":
            raise AssertionError(f"{target} fit split changed")
    print(json.dumps({"passed": True, "validation_prediction_rows": 524, "targets_verified": list(TARGET_COLUMNS), "complete_alpha_grid_verified": True, "test_predictions_or_metrics": False}, indent=2))


if __name__ == "__main__":
    main()
