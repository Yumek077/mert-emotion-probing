"""Independently verify Module C Stage C3 pre-Test result artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mert_emotion_probing.probing import (  # noqa: E402
    ALPHA_GRID,
    EXPECTED_SPLIT_COUNTS,
    TARGET_COLUMNS,
    regression_metrics,
    select_candidate_by_validation_r2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("outputs/results/module_c_stage3_validation.json"),
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("outputs/results/module_c_stage3_validation_predictions.csv"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("data/metadata/deam_primary_split_seed42.csv"),
    )
    return parser.parse_args()


def assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=1e-12):
        raise AssertionError(f"{label}: {actual} != {expected}")


def main() -> None:
    args = parse_args()
    payload = json.loads(args.results_json.read_text(encoding="utf-8"))
    predictions = pd.read_csv(args.predictions_csv)
    split = pd.read_csv(args.split)

    expected_columns = {
        "sample_id",
        "split",
        "target",
        "representation_level",
        "representation_index",
        "selected_alpha",
        "y_true",
        "prediction",
        "baseline_prediction",
    }
    if set(predictions.columns) != expected_columns:
        raise AssertionError("Validation prediction columns differ from the expected schema")
    if set(predictions["split"]) != {"validation"}:
        raise AssertionError("Prediction artifact contains a non-Validation row")
    if len(predictions) != EXPECTED_SPLIT_COUNTS["validation"] * len(TARGET_COLUMNS):
        raise AssertionError("Prediction artifact has the wrong row count")
    if predictions.duplicated(["sample_id", "target"]).any():
        raise AssertionError("Prediction artifact contains duplicate sample-target rows")

    validation_ids = set(
        split.loc[split["split"] == "validation", "sample_id"].astype(int).tolist()
    )
    test_ids = set(split.loc[split["split"] == "test", "sample_id"].astype(int).tolist())
    prediction_ids = set(predictions["sample_id"].astype(int).tolist())
    if prediction_ids != validation_ids:
        raise AssertionError("Prediction IDs do not equal the frozen Validation IDs")
    if prediction_ids & test_ids:
        raise AssertionError("Prediction artifact contains a Test ID")

    for target in TARGET_COLUMNS:
        target_payload = payload["targets"][target]
        rows = target_payload["candidate_validation_r2"]
        if tuple(float(row["alpha"]) for row in rows) != ALPHA_GRID:
            raise AssertionError(f"{target} alpha grid differs from the frozen grid")
        selected, best_r2, exact_ties = select_candidate_by_validation_r2(rows)
        if float(selected["alpha"]) != target_payload["selected_alpha"]:
            raise AssertionError(f"{target} selected alpha does not follow the rule")
        assert_close(best_r2, target_payload["best_validation_r2"], f"{target} best R2")
        if len(exact_ties) != target_payload["exact_tie_count"]:
            raise AssertionError(f"{target} exact-tie count is inconsistent")

        target_predictions = predictions.loc[predictions["target"] == target].copy()
        metrics = regression_metrics(
            target_predictions["y_true"].to_numpy(dtype=np.float64),
            target_predictions["prediction"].to_numpy(dtype=np.float64),
        )
        saved_metrics = target_payload["selected_validation_metrics"]
        for metric in ("mae", "r2", "pearson_r"):
            assert_close(metrics[metric], saved_metrics[metric], f"{target} {metric}")

        baseline_metrics = regression_metrics(
            target_predictions["y_true"].to_numpy(dtype=np.float64),
            target_predictions["baseline_prediction"].to_numpy(dtype=np.float64),
        )
        saved_baseline = target_payload["baseline"]
        if not np.all(
            target_predictions["baseline_prediction"].to_numpy(dtype=np.float64)
            == saved_baseline["train_target_mean"]
        ):
            raise AssertionError(f"{target} baseline is not the saved Train mean")
        for metric in ("mae", "r2"):
            assert_close(
                baseline_metrics[metric],
                saved_baseline["validation_metrics"][metric],
                f"{target} baseline {metric}",
            )
        if baseline_metrics["pearson_r"] is not None:
            raise AssertionError(f"{target} constant baseline Pearson r must be undefined")
        if saved_baseline["validation_metrics"]["pearson_r"] is not None:
            raise AssertionError(f"{target} saved baseline Pearson r must be null")
        if not payload["determinism"][target]["passed"]:
            raise AssertionError(f"{target} deterministic repeat did not pass")

    exact_tie_rows = [
        {"alpha": 1.0, "validation_r2": 0.5},
        {"alpha": 10.0, "validation_r2": 0.5},
    ]
    selected, _, tied = select_candidate_by_validation_r2(exact_tie_rows)
    if selected["alpha"] != 10.0 or len(tied) != 2:
        raise AssertionError("Exact-tie rule did not choose the larger alpha")
    non_tie_rows = [
        {"alpha": 1.0, "validation_r2": 0.5},
        {"alpha": 10.0, "validation_r2": np.nextafter(0.5, -np.inf)},
    ]
    selected, _, tied = select_candidate_by_validation_r2(non_tie_rows)
    if selected["alpha"] != 1.0 or len(tied) != 1:
        raise AssertionError("Non-equal scores were incorrectly treated as a tie")

    gate = payload["pre_test_gate"]
    if any(
        [
            gate["test_predictions_generated"],
            gate["test_performance_computed"],
            gate["test_performance_inspected"],
        ]
    ):
        raise AssertionError("Pre-Test artifact indicates that the Test gate was opened")

    summary = {
        "passed": True,
        "validation_prediction_rows": int(len(predictions)),
        "validation_ids": int(len(prediction_ids)),
        "contains_test_ids": False,
        "target_metrics_recomputed": list(TARGET_COLUMNS),
        "mean_baseline_recomputed": list(TARGET_COLUMNS),
        "exact_tie_rule_verified": True,
        "near_but_not_exact_tie_rule_verified": True,
        "test_predictions_or_metrics_inspected": False,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
