"""Run the frozen Module C Stage C4 held-out Test evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mert_emotion_probing.probing import (  # noqa: E402
    ALPHA_GRID,
    EXPECTED_SPLIT_COUNTS,
    PRIMARY_LEVEL_NAME,
    TARGET_COLUMNS,
    assemble_probing_dataset,
    fit_train_evaluate_test,
    train_test_view,
)


FROZEN_ALPHA = {"valence": 1000.0, "arousal": 1000.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("outputs/embeddings/deam_mert_v1_95m_meanpool_all_layers.pt"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/raw/deam/verification/deam_item_mapping.csv"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("data/metadata/deam_primary_split_seed42.csv"),
    )
    parser.add_argument(
        "--stage-c3-json",
        type=Path,
        default=Path("outputs/results/module_c_stage3_validation.json"),
    )
    parser.add_argument(
        "--stage-c3-predictions",
        type=Path,
        default=Path("outputs/results/module_c_stage3_validation_predictions.csv"),
    )
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
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False, lineterminator="\n", float_format="%.17g")
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_stage_c3_gate(
    stage_c3: dict[str, Any], input_identities: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    failures: list[str] = []
    protocol = stage_c3.get("protocol", {})
    checks = stage_c3.get("data_assembly_verification", {})
    gate = stage_c3.get("pre_test_gate", {})
    leakage = stage_c3.get("leakage_controls", {})

    if stage_c3.get("stage") != "Module C Stage C3":
        failures.append("Stage C3 result identity is missing or incorrect")
    if protocol.get("representation_level") != PRIMARY_LEVEL_NAME:
        failures.append("Stage C3 representation is not transformer_layer_12")
    if protocol.get("resolved_representation_index") != 12:
        failures.append("Stage C3 Layer-12 index is not 12")
    if protocol.get("probe") != "Ridge Regression":
        failures.append("Stage C3 probe is not Ridge Regression")
    if protocol.get("ridge_solver") != "cholesky":
        failures.append("Stage C3 Ridge solver is not cholesky")
    if tuple(protocol.get("alpha_grid", [])) != ALPHA_GRID:
        failures.append("Stage C3 alpha grid differs from the frozen grid")
    if protocol.get("input_scaler") != "StandardScaler fit on Train only":
        failures.append("Stage C3 scaler policy is not Train-only StandardScaler")
    if protocol.get("target_standardization") is not False:
        failures.append("Stage C3 targets were not recorded as unstandardized")
    if checks.get("split_counts") != EXPECTED_SPLIT_COUNTS:
        failures.append("Stage C3 split counts differ from the frozen counts")
    for target, expected_alpha in FROZEN_ALPHA.items():
        actual = stage_c3.get("targets", {}).get(target, {}).get("selected_alpha")
        if actual != expected_alpha:
            failures.append(f"Stage C3 {target} alpha {actual} != {expected_alpha}")
        if not stage_c3.get("determinism", {}).get(target, {}).get("passed", False):
            failures.append(f"Stage C3 {target} determinism check did not pass")
    if gate.get("ready_for_researcher_review_before_stage_c4") is not True:
        failures.append("Stage C3 did not mark the pre-Test gate ready")
    for key in (
        "test_predictions_generated",
        "test_performance_computed",
        "test_performance_inspected",
    ):
        if gate.get(key) is not False:
            failures.append(f"Stage C3 pre-Test gate has unexpected {key}={gate.get(key)}")
    if leakage.get("test_used_for_fitting") is not False:
        failures.append("Stage C3 indicates Test was used for fitting")
    if leakage.get("test_used_for_selection") is not False:
        failures.append("Stage C3 indicates Test was used for selection")

    c3_inputs = stage_c3.get("inputs", {})
    for name, current in input_identities.items():
        recorded = c3_inputs.get(name, {})
        if recorded.get("sha256") != current["sha256"]:
            failures.append(f"{name} SHA-256 differs from Stage C3")
        if recorded.get("size_bytes") != current["size_bytes"]:
            failures.append(f"{name} size differs from Stage C3")
    if failures:
        raise RuntimeError("Pre-Test gate failed: " + "; ".join(failures))

    return {
        "passed": True,
        "stage_c3_identity_verified": True,
        "frozen_representation_verified": True,
        "frozen_alphas_verified": True,
        "train_only_scaler_policy_verified": True,
        "targets_unstandardized_verified": True,
        "split_counts_verified": True,
        "prior_test_predictions_or_metrics": False,
        "input_hashes_and_sizes_unchanged": True,
    }


def main() -> None:
    args = parse_args()
    stage_c3 = json.loads(args.stage_c3_json.read_text(encoding="utf-8"))
    input_identities = {
        "canonical_cache": file_identity(args.cache),
        "labels": file_identity(args.labels),
        "fixed_split": file_identity(args.split),
    }
    stage_c3_gate = verify_stage_c3_gate(stage_c3, input_identities)
    dataset = assemble_probing_dataset(args.cache, args.labels, args.split)
    if dataset.representation_level != PRIMARY_LEVEL_NAME or dataset.representation_index != 12:
        raise RuntimeError("Current Layer-12 resolution differs from the frozen gate")

    train_ids = set(dataset.sample_ids[dataset.split == "train"].tolist())
    validation_ids = set(dataset.sample_ids[dataset.split == "validation"].tolist())
    test_ids = set(dataset.sample_ids[dataset.split == "test"].tolist())
    if len(test_ids) != EXPECTED_SPLIT_COUNTS["test"]:
        raise RuntimeError("Test does not contain exactly 261 unique IDs")
    if train_ids & test_ids or validation_ids & test_ids:
        raise RuntimeError("Test IDs overlap Train or Validation IDs")

    target_results: dict[str, Any] = {}
    determinism: dict[str, Any] = {}
    prediction_tables: list[pd.DataFrame] = []
    for target in TARGET_COLUMNS:
        data = train_test_view(dataset, target)
        alpha = FROZEN_ALPHA[target]
        result, prediction, baseline_prediction = fit_train_evaluate_test(
            data, alpha=alpha
        )
        repeated_result, repeated_prediction, repeated_baseline = (
            fit_train_evaluate_test(data, alpha=alpha)
        )
        repeat_checks = {
            "test_metrics_equal": result["test_metrics"]
            == repeated_result["test_metrics"],
            "baseline_equal": result["baseline"] == repeated_result["baseline"],
            "predictions_exactly_equal": bool(
                np.array_equal(prediction, repeated_prediction)
            ),
            "baseline_predictions_exactly_equal": bool(
                np.array_equal(baseline_prediction, repeated_baseline)
            ),
        }
        repeat_checks["passed"] = all(repeat_checks.values())
        if not repeat_checks["passed"]:
            raise RuntimeError(f"{target} deterministic Test repeat failed")
        determinism[target] = repeat_checks
        target_results[target] = result
        prediction_tables.append(
            pd.DataFrame(
                {
                    "sample_id": data.test_ids,
                    "split": "test",
                    "target": target,
                    "representation_level": dataset.representation_level,
                    "representation_index": dataset.representation_index,
                    "frozen_alpha": alpha,
                    "y_true": data.y_test,
                    "prediction": prediction,
                    "baseline_prediction": baseline_prediction,
                }
            )
        )

    predictions = pd.concat(prediction_tables, ignore_index=True)
    expected_rows = EXPECTED_SPLIT_COUNTS["test"] * len(TARGET_COLUMNS)
    if len(predictions) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} Test prediction rows")
    if set(predictions["split"]) != {"test"}:
        raise RuntimeError("Stage C4 prediction artifact contains a non-Test row")
    if predictions.duplicated(["sample_id", "target"]).any():
        raise RuntimeError("Stage C4 predictions contain duplicate sample-target rows")
    if set(predictions["sample_id"].astype(int)) != test_ids:
        raise RuntimeError("Stage C4 prediction IDs do not equal the frozen Test IDs")

    validation_context = {
        target: stage_c3["targets"][target]["selected_validation_metrics"]
        for target in TARGET_COLUMNS
    }
    payload = {
        "stage": "Module C Stage C4",
        "stage_objective": "Frozen Train-only refit and held-out Test evaluation",
        "created_utc": utc_now(),
        "frozen_configuration": {
            "representation_level": dataset.representation_level,
            "resolved_representation_index": dataset.representation_index,
            "probe": "Ridge Regression",
            "ridge_solver": "cholesky",
            "alphas": FROZEN_ALPHA,
            "input_scaler": "StandardScaler fit on Train only",
            "target_standardization": False,
            "final_refit_training_split": "train_only",
            "validation_in_final_refit": False,
            "metrics": ["mae", "r2", "pearson_r"],
            "baseline": "constant prediction from Train target mean",
        },
        "provenance": {
            "stage_c3_results": file_identity(args.stage_c3_json),
            "stage_c3_validation_predictions": file_identity(
                args.stage_c3_predictions
            ),
            "inputs": input_identities,
        },
        "pre_test_gate": {
            **stage_c3_gate,
            "current_data_assembly": dataset.verification,
            "test_unique_ids": len(test_ids),
            "train_test_overlap": 0,
            "validation_test_overlap": 0,
            "opened_only_after_all_checks_passed": True,
        },
        "targets": target_results,
        "validation_context_from_stage_c3": validation_context,
        "test_prediction_artifact": {
            "path": args.predictions_csv.as_posix(),
            "rows": int(len(predictions)),
            "unique_test_ids": int(predictions["sample_id"].nunique()),
            "splits_present": ["test"],
            "sample_id_aligned": True,
            "contains_train_or_validation_predictions": False,
        },
        "determinism": determinism,
        "leakage_and_selection_controls": {
            "scaler_fit_split": "train",
            "ridge_fit_split": "train",
            "baseline_mean_source": "train_target_only",
            "validation_in_final_refit": False,
            "test_used_for_fitting": False,
            "test_used_for_baseline_construction": False,
            "test_used_for_model_selection": False,
            "post_test_tuning_performed": False,
        },
        "software_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    atomic_write_csv(args.predictions_csv, predictions)
    atomic_write_json(args.results_json, payload)
    if not args.quiet:
        print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
