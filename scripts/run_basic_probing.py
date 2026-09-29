"""Run Module C Stage C3 Train/Validation probing without opening the Test gate."""

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
    TARGET_COLUMNS,
    assemble_probing_dataset,
    fit_select_validate,
    train_validation_view,
    verify_deterministic_repeat,
)


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
        "--quiet",
        action="store_true",
        help="Write artifacts without printing the complete result JSON.",
    )
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


def main() -> None:
    args = parse_args()
    dataset = assemble_probing_dataset(args.cache, args.labels, args.split)

    target_results: dict[str, Any] = {}
    prediction_tables: list[pd.DataFrame] = []
    determinism: dict[str, Any] = {}

    for target in TARGET_COLUMNS:
        train_validation = train_validation_view(dataset, target)
        result, prediction, baseline_prediction = fit_select_validate(
            train_validation
        )
        repeated_result, repeated_prediction, _ = fit_select_validate(
            train_validation
        )
        determinism[target] = verify_deterministic_repeat(
            result, prediction, repeated_result, repeated_prediction
        )
        target_results[target] = result
        prediction_tables.append(
            pd.DataFrame(
                {
                    "sample_id": train_validation.validation_ids,
                    "split": "validation",
                    "target": target,
                    "representation_level": dataset.representation_level,
                    "representation_index": dataset.representation_index,
                    "selected_alpha": result["selected_alpha"],
                    "y_true": train_validation.y_validation,
                    "prediction": prediction,
                    "baseline_prediction": baseline_prediction,
                }
            )
        )

    predictions = pd.concat(prediction_tables, ignore_index=True)
    expected_prediction_rows = (
        EXPECTED_SPLIT_COUNTS["validation"] * len(TARGET_COLUMNS)
    )
    if len(predictions) != expected_prediction_rows:
        raise RuntimeError(
            f"Expected {expected_prediction_rows} Validation prediction rows, "
            f"received {len(predictions)}"
        )
    if set(predictions["split"]) != {"validation"}:
        raise RuntimeError("Prediction artifact contains a non-Validation split")
    if predictions.duplicated(["sample_id", "target"]).any():
        raise RuntimeError("Prediction artifact contains duplicate sample-target rows")

    result_payload = {
        "stage": "Module C Stage C3",
        "stage_objective": (
            "Train/Validation Layer-12 Ridge selection and pre-Test verification"
        ),
        "created_utc": utc_now(),
        "protocol": {
            "research_question": (
                "Can continuous Valence and Arousal be decoded from frozen MERT "
                "representations?"
            ),
            "interpretation": "linear_decodability_not_human_like_understanding",
            "representation_level": dataset.representation_level,
            "resolved_representation_index": dataset.representation_index,
            "input_dimension": int(dataset.features.shape[1]),
            "targets": list(TARGET_COLUMNS),
            "target_columns": TARGET_COLUMNS,
            "probe": "Ridge Regression",
            "ridge_solver": "cholesky",
            "alpha_grid": list(ALPHA_GRID),
            "selection_metric": "validation_r2",
            "exact_tie_rule": "choose_larger_alpha_on_actual_exact_numerical_tie",
            "input_scaler": "StandardScaler fit on Train only",
            "target_standardization": False,
            "baseline": "constant prediction from Train target mean",
            "test_policy": "membership/alignment verification only; no predictions or metrics",
        },
        "inputs": {
            "canonical_cache": file_identity(args.cache),
            "labels": file_identity(args.labels),
            "fixed_split": file_identity(args.split),
        },
        "data_assembly_verification": dataset.verification,
        "targets": target_results,
        "validation_prediction_artifact": {
            "path": args.predictions_csv.as_posix(),
            "rows": int(len(predictions)),
            "splits_present": ["validation"],
            "sample_id_aligned": True,
            "contains_test_predictions": False,
        },
        "determinism": determinism,
        "leakage_controls": {
            "scaler_fit_split": "train",
            "ridge_fit_split": "train",
            "validation_role": "alpha_selection_and_evaluation",
            "selection_uses_only_validation_r2": True,
            "baseline_mean_source": "train_target_only",
            "targets_standardized": False,
            "test_used_for_fitting": False,
            "test_used_for_selection": False,
            "test_predictions_generated": False,
            "test_metrics_computed": False,
        },
        "pre_test_gate": {
            "population_and_membership_verified": True,
            "test_id_count": EXPECTED_SPLIT_COUNTS["test"],
            "test_ids_used_only_for_integrity_and_alignment": True,
            "test_predictions_generated": False,
            "test_performance_computed": False,
            "test_performance_inspected": False,
            "ready_for_researcher_review_before_stage_c4": True,
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
    atomic_write_json(args.results_json, result_payload)
    if not args.quiet:
        print(json.dumps(result_payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
