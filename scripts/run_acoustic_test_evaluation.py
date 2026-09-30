"""Run Module E's frozen Train-only acoustic Test evaluation exactly once.

The MERT comparator is read from authoritative Module C artifacts. No alpha
search, MERT fitting, acoustic extraction, or RQ4 analysis is performed here.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mert_emotion_probing.probing import (  # noqa: E402
    ALPHA_GRID, EXPECTED_SPLIT_COUNTS, PRIMARY_LEVEL_NAME, TARGET_COLUMNS,
    TrainTestData, _load_labels, _load_split, fit_train_evaluate_test,
    regression_metrics,
)
from run_acoustic_validation import (  # noqa: E402
    DIMENSION, identity, load_acoustic_cache, tracked_file_unchanged,
    write_csv, write_json,
)

RESULTS = Path("outputs/results/module_e_stage3_test.json")
PREDICTIONS = Path("outputs/results/module_e_stage3_test_predictions.csv")
COMPARISON = Path("outputs/results/module_e_stage3_rq3_comparison.csv")
STARTED = Path("outputs/results/module_e_stage3_test_started.json")
STAGE2 = Path("outputs/results/module_e_stage2_validation.json")
STAGE2_GATE = Path("outputs/results/module_e_stage2_pre_test_verification.json")
STAGE2_PREDICTIONS = Path("outputs/results/module_e_stage2_validation_predictions.csv")
MERT_RESULTS = Path("outputs/results/module_c_stage4_test.json")
MERT_PREDICTIONS = Path("outputs/results/module_c_stage4_test_predictions.csv")


def close_metrics(observed: dict, saved: dict) -> None:
    for metric in ("mae", "r2", "pearson_r"):
        if observed[metric] is None or saved[metric] is None:
            if observed[metric] is not saved[metric]:
                raise ValueError(f"Undefined {metric} disagrees")
        elif not np.isclose(observed[metric], saved[metric], rtol=0, atol=1e-12):
            raise ValueError(f"Saved {metric} disagrees with predictions")


def preflight() -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str], dict]:
    stage2 = json.loads(STAGE2.read_text(encoding="utf-8"))
    gate = json.loads(STAGE2_GATE.read_text(encoding="utf-8"))
    if stage2["stage"] != "Module E Stage 2" or not gate["passed"] or not all(gate["checks"].values()):
        raise ValueError("Stage 2 identity or pre-Test checks failed")
    if gate["test_evaluation_performed"] is not False or stage2["validation_prediction_artifact"]["contains_test_ids"] is not False:
        raise ValueError("Stage 2 did not preserve the Test boundary")
    protocol = stage2["protocol"]
    expected = {
        "representation": "frozen_conventional_acoustic_51d", "input_dimension": DIMENSION,
        "mert_comparator": PRIMARY_LEVEL_NAME, "probe": "Ridge Regression",
        "ridge_solver": "cholesky", "fit_intercept": True,
        "alpha_grid": list(ALPHA_GRID), "input_scaler": "StandardScaler fit on Train only",
        "targets_standardized": False, "targets": TARGET_COLUMNS,
        "selection": "maximum_validation_r2_exact_tie_larger_alpha",
        "final_fit_policy": "Train only; deferred to Test stage",
    }
    for key, value in expected.items():
        if protocol[key] != value:
            raise ValueError(f"Frozen Stage 2 protocol differs: {key}")
    alphas = {target: float(stage2["targets"][target]["selected_alpha"]) for target in TARGET_COLUMNS}
    if alphas != {"valence": 100.0, "arousal": 10.0}:
        raise ValueError("Stage 2 selected alphas differ from the approved configuration")
    current_inputs = {}
    for name, recorded in stage2["inputs"].items():
        current = identity(Path(recorded["path"]))
        if current != recorded or gate["inputs"][name] != recorded:
            raise ValueError(f"Frozen input identity changed: {name}")
        current_inputs[name] = current
    for path in (STAGE2, STAGE2_GATE, STAGE2_PREDICTIONS, MERT_RESULTS, MERT_PREDICTIONS):
        if not tracked_file_unchanged(path):
            raise ValueError(f"Authoritative artifact is not unchanged: {path}")

    cache_ids, features, names = load_acoustic_cache(
        Path(current_inputs["acoustic_cache"]["path"]),
        Path(current_inputs["stage1_diagnostics"]["path"]),
    )
    labels, _ = _load_labels(Path(current_inputs["labels"]["path"]))
    split, split_checks = _load_split(Path(current_inputs["fixed_split"]["path"]))
    if set(cache_ids) != set(labels.song_id) or set(cache_ids) != set(split.sample_id):
        raise ValueError("Cache, primary population, and frozen split IDs differ")
    role_ids = {role: set(split.loc[split.split == role, "sample_id"]) for role in EXPECTED_SPLIT_COUNTS}
    stage2_predictions = pd.read_csv(STAGE2_PREDICTIONS, float_precision="round_trip")
    if len(stage2_predictions) != 524 or set(stage2_predictions.split) != {"validation"} or stage2_predictions.duplicated(["sample_id", "target"]).any():
        raise ValueError("Stage 2 predictions are not a valid Validation-only artifact")
    for target in TARGET_COLUMNS:
        saved = stage2["targets"][target]
        rows = stage2_predictions.loc[stage2_predictions.target == target]
        if set(rows.sample_id) != role_ids["validation"] or set(rows.selected_alpha) != {alphas[target]}:
            raise ValueError("Stage 2 target or selected alpha provenance mismatch")
        close_metrics(regression_metrics(rows.y_true.to_numpy(), rows.prediction.to_numpy()), saved["selected_validation_metrics"])

    mert = json.loads(MERT_RESULTS.read_text(encoding="utf-8"))
    mert_config = mert["frozen_configuration"]
    for key, value in {
        "representation_level": PRIMARY_LEVEL_NAME, "resolved_representation_index": 12,
        "probe": "Ridge Regression", "ridge_solver": "cholesky",
        "input_scaler": "StandardScaler fit on Train only", "target_standardization": False,
        "final_refit_training_split": "train_only", "validation_in_final_refit": False,
    }.items():
        if mert_config[key] != value:
            raise ValueError(f"Module C comparator protocol mismatch: {key}")
    for name in ("labels", "fixed_split"):
        if mert["provenance"]["inputs"][name] != current_inputs[name]:
            raise ValueError("Module C did not use the same targets/split")
    mert_predictions = pd.read_csv(MERT_PREDICTIONS, float_precision="round_trip")
    if len(mert_predictions) != 522 or set(mert_predictions.split) != {"test"} or mert_predictions.duplicated(["sample_id", "target"]).any():
        raise ValueError("Module C Test predictions have invalid rows")
    by_id = labels.set_index("song_id", verify_integrity=True)
    for target, column in TARGET_COLUMNS.items():
        rows = mert_predictions.loc[mert_predictions.target == target]
        if set(rows.sample_id) != role_ids["test"] or set(rows.representation_level) != {PRIMARY_LEVEL_NAME} or set(rows.representation_index) != {12}:
            raise ValueError("Module C Test IDs or representation differ")
        saved = mert["targets"][target]
        if set(rows.frozen_alpha) != {saved["frozen_alpha"]} or saved["frozen_alpha"] != mert_config["alphas"][target]:
            raise ValueError("Module C alpha provenance mismatch")
        expected_y = by_id.loc[rows.sample_id.to_numpy(), column].to_numpy(dtype=np.float64)
        if not np.array_equal(expected_y, rows.y_true.to_numpy()):
            raise ValueError("Module C Test targets are not ID-aligned")
        close_metrics(regression_metrics(expected_y, rows.prediction.to_numpy()), saved["test_metrics"])
        close_metrics(regression_metrics(expected_y, rows.baseline_prediction.to_numpy()), saved["baseline"]["test_metrics"])
        train_mean = float(by_id.loc[sorted(role_ids["train"]), column].mean())
        if train_mean != saved["baseline"]["train_target_mean"] or not np.all(rows.baseline_prediction.to_numpy() == train_mean):
            raise ValueError("Module C baseline is not the matching Train mean")

    checked = {
        "stage2_review_authorized_by_researcher": True,
        "stage2_artifacts_unchanged": True, "stage2_input_hashes_and_sizes_unchanged": True,
        "frozen_alphas_read_from_stage2": alphas,
        "cache_shape": list(features.shape), "split_counts": split_checks["split_counts"],
        "test_ids_exact_unique": len(role_ids["test"]), "split_overlap_count": 0,
        "module_c_metrics_verified_from_saved_predictions": True,
        "module_c_artifacts_unchanged": True,
        "prior_acoustic_test_artifacts_present": any(path.exists() for path in (STARTED, RESULTS, PREDICTIONS, COMPARISON)),
        "passed": True,
    }
    return stage2, mert, labels, split, cache_ids, features, names, checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    stage2, mert, labels, split, cache_ids, features, names, checked = preflight()
    if args.preflight_only:
        print(json.dumps(checked, indent=2))
        return
    if checked["prior_acoustic_test_artifacts_present"]:
        raise RuntimeError("Official acoustic Test run has already started; verify saved artifacts instead of rerunning")

    ordered_ids = np.asarray(sorted(cache_ids.tolist()), dtype=np.int64)
    positions = {int(sample_id): index for index, sample_id in enumerate(cache_ids)}
    ordered_features = features[[positions[int(sample_id)] for sample_id in ordered_ids]].astype(np.float64)
    roles = split.set_index("sample_id").loc[ordered_ids, "split"].to_numpy(dtype=str)
    labels_by_id = labels.set_index("song_id", verify_integrity=True)
    train_mask, test_mask = roles == "train", roles == "test"
    train_ids, test_ids = ordered_ids[train_mask], ordered_ids[test_mask]
    alphas = checked["frozen_alphas_read_from_stage2"]
    provenance = {
        "stage2_results": identity(STAGE2), "stage2_validation_predictions": identity(STAGE2_PREDICTIONS),
        "stage2_pre_test_verification": identity(STAGE2_GATE),
        "module_c_authoritative_results": identity(MERT_RESULTS),
        "module_c_authoritative_predictions": identity(MERT_PREDICTIONS),
        "inputs": stage2["inputs"],
        "evaluation_code": [identity(Path("scripts/run_acoustic_test_evaluation.py")), identity(Path("src/mert_emotion_probing/probing.py"))],
    }
    started = {"stage": "Module E Stage 3 Test gate opened", "created_utc": datetime.now(timezone.utc).isoformat(),
               "pre_test_checks": checked, "provenance": provenance,
               "frozen_alphas": alphas, "official_evaluation_calls_per_target": 1}
    # Exclusive marker prevents accidentally replacing the first formal evaluation.
    with STARTED.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(started, indent=2, allow_nan=False) + "\n")

    target_results, prediction_frames, comparison_rows = {}, [], []
    for target, column in TARGET_COLUMNS.items():
        # Only Train and Test arrays enter the frozen fitting routine.
        view = TrainTestData(
            train_ids=train_ids.copy(), test_ids=test_ids.copy(),
            x_train=ordered_features[train_mask].copy(), x_test=ordered_features[test_mask].copy(),
            y_train=labels_by_id.loc[train_ids, column].to_numpy(dtype=np.float64),
            y_test=labels_by_id.loc[test_ids, column].to_numpy(dtype=np.float64),
        )
        result, prediction, baseline = fit_train_evaluate_test(
            view, alpha=alphas[target], expected_dimension=DIMENSION,
            include_scaler_statistics=True,
        )
        target_results[target] = result
        prediction_frames.append(pd.DataFrame({
            "sample_id": test_ids, "split": "test", "target": target,
            "representation": "conventional_acoustic_51d", "frozen_alpha": alphas[target],
            "y_true": view.y_test, "prediction": prediction, "baseline_prediction": baseline,
        }))
        for representation, dimension, saved, reused in (
            ("MERT Layer 12", 768, mert["targets"][target], True),
            ("Conventional Acoustic 51-D", DIMENSION, result, False),
        ):
            comparison_rows.append({"target": target, "representation": representation, "input_dimension": dimension,
                                    "selected_alpha": saved["frozen_alpha"],
                                    "test_mae": saved["test_metrics"]["mae"], "test_r2": saved["test_metrics"]["r2"],
                                    "test_pearson_r": saved["test_metrics"]["pearson_r"],
                                    "reused_module_c_evidence": reused})

    predictions = pd.concat(prediction_frames, ignore_index=True)
    if len(predictions) != 522 or predictions.duplicated(["sample_id", "target"]).any() or set(predictions.sample_id) != set(test_ids):
        raise RuntimeError("Official Test predictions failed identity validation")
    for name, recorded in stage2["inputs"].items():
        if identity(Path(recorded["path"])) != recorded:
            raise RuntimeError(f"Input changed during Test evaluation: {name}")
    payload = {
        "stage": "Module E Stage 3", "created_utc": started["created_utc"],
        "frozen_configuration": {"representation": "conventional_acoustic_51d", "input_dimension": DIMENSION,
                                 "feature_names": names, "alphas": alphas, "probe": "Ridge Regression",
                                 "ridge_solver": "cholesky", "fit_intercept": True,
                                 "input_scaler": "StandardScaler fit on Train only", "target_standardization": False,
                                 "final_refit_training_split": "train_only", "validation_in_final_refit": False,
                                 "metrics": ["mae", "r2", "pearson_r"], "mert_comparator": PRIMARY_LEVEL_NAME},
        "provenance": provenance, "pre_test_gate": checked, "targets": target_results,
        "comparison": comparison_rows,
        "test_prediction_artifact": {"path": PREDICTIONS.as_posix(), "rows": 522, "unique_test_ids": 261, "splits_present": ["test"]},
        "controls": {"scaler_fit_split": "train", "ridge_fit_split": "train", "baseline_mean_source": "train_target_only",
                     "validation_in_final_refit": False, "test_used_for_fitting_or_selection": False,
                     "post_test_tuning_performed": False, "alpha_selection_rerun": False,
                     "acoustic_extraction_rerun": False, "mert_evaluation_rerun": False, "rq4_analysis_performed": False,
                     "official_evaluation_calls_per_target": 1},
        "software_environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                                 "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
    }
    write_csv(PREDICTIONS, predictions)
    write_csv(COMPARISON, pd.DataFrame(comparison_rows))
    write_json(RESULTS, payload)
    print(json.dumps({"targets": {target: result["test_metrics"] for target, result in target_results.items()},
                      "official_evaluation_calls_per_target": 1, "post_test_tuning_performed": False}, indent=2))


if __name__ == "__main__":
    main()
