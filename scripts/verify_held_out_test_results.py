"""Independently verify frozen Module C Stage C4 Test artifacts."""

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
    EXPECTED_SPLIT_COUNTS,
    TARGET_COLUMNS,
    regression_metrics,
)


FROZEN_ALPHA = {"valence": 1000.0, "arousal": 1000.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("outputs/results/module_c_stage4_test.json"),
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("outputs/results/module_c_stage4_test_predictions.csv"),
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
        "frozen_alpha",
        "y_true",
        "prediction",
        "baseline_prediction",
    }
    if set(predictions.columns) != expected_columns:
        raise AssertionError("Test prediction columns differ from the expected schema")
    if len(predictions) != EXPECTED_SPLIT_COUNTS["test"] * len(TARGET_COLUMNS):
        raise AssertionError("Test prediction artifact has the wrong row count")
    if set(predictions["split"]) != {"test"}:
        raise AssertionError("Test prediction artifact contains a non-Test row")
    if predictions.duplicated(["sample_id", "target"]).any():
        raise AssertionError("Test prediction artifact contains duplicate rows")

    test_ids = set(split.loc[split["split"] == "test", "sample_id"].astype(int))
    train_validation_ids = set(
        split.loc[split["split"] != "test", "sample_id"].astype(int)
    )
    prediction_ids = set(predictions["sample_id"].astype(int))
    if prediction_ids != test_ids or prediction_ids & train_validation_ids:
        raise AssertionError("Prediction IDs do not equal the isolated frozen Test set")

    for target in TARGET_COLUMNS:
        rows = predictions.loc[predictions["target"] == target].copy()
        if len(rows) != EXPECTED_SPLIT_COUNTS["test"]:
            raise AssertionError(f"{target} does not have exactly 261 Test rows")
        if set(rows["frozen_alpha"].astype(float)) != {FROZEN_ALPHA[target]}:
            raise AssertionError(f"{target} predictions use the wrong frozen alpha")
        saved = payload["targets"][target]
        if saved["frozen_alpha"] != FROZEN_ALPHA[target]:
            raise AssertionError(f"{target} JSON uses the wrong frozen alpha")

        metrics = regression_metrics(
            rows["y_true"].to_numpy(dtype=np.float64),
            rows["prediction"].to_numpy(dtype=np.float64),
        )
        for metric in ("mae", "r2", "pearson_r"):
            assert_close(metrics[metric], saved["test_metrics"][metric], f"{target} {metric}")

        baseline_metrics = regression_metrics(
            rows["y_true"].to_numpy(dtype=np.float64),
            rows["baseline_prediction"].to_numpy(dtype=np.float64),
        )
        baseline = saved["baseline"]
        if not np.all(
            rows["baseline_prediction"].to_numpy(dtype=np.float64)
            == baseline["train_target_mean"]
        ):
            raise AssertionError(f"{target} baseline is not the frozen Train mean")
        for metric in ("mae", "r2"):
            assert_close(
                baseline_metrics[metric],
                baseline["test_metrics"][metric],
                f"{target} baseline {metric}",
            )
        if baseline_metrics["pearson_r"] is not None:
            raise AssertionError(f"{target} constant baseline Pearson r must be undefined")
        if baseline["test_metrics"]["pearson_r"] is not None:
            raise AssertionError(f"{target} saved baseline Pearson r must be null")
        if not payload["determinism"][target]["passed"]:
            raise AssertionError(f"{target} deterministic rerun did not pass")

    controls = payload["leakage_and_selection_controls"]
    if controls["post_test_tuning_performed"] is not False:
        raise AssertionError("Artifact indicates post-Test tuning")
    if controls["validation_in_final_refit"] is not False:
        raise AssertionError("Artifact indicates Validation entered final refit")

    summary = {
        "passed": True,
        "test_prediction_rows": int(len(predictions)),
        "unique_test_ids": int(len(prediction_ids)),
        "contains_train_or_validation_ids": False,
        "test_metrics_recomputed": list(TARGET_COLUMNS),
        "mean_baseline_recomputed": list(TARGET_COLUMNS),
        "frozen_alphas_verified": FROZEN_ALPHA,
        "deterministic_reruns_verified": True,
        "post_test_tuning_performed": False,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
