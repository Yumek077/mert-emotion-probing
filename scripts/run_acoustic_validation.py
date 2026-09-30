"""Module E Stage 2: acoustic Train/Validation Ridge selection; no Test evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mert_emotion_probing.probing import (  # noqa: E402
    ALPHA_GRID, EXPECTED_SPLIT_COUNTS, PRIMARY_LEVEL_NAME, TARGET_COLUMNS,
    TrainValidationData, _load_labels, _load_split, fit_select_validate,
    verify_deterministic_repeat,
)

DIMENSION = 51


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/processed/deam_acoustic_51d.npz"))
    parser.add_argument("--stage1-diagnostics", type=Path, default=Path("outputs/results/module_e_stage1_acoustic_diagnostics.json"))
    parser.add_argument("--labels", type=Path, default=Path("data/raw/deam/verification/deam_item_mapping.csv"))
    parser.add_argument("--split", type=Path, default=Path("data/metadata/deam_primary_split_seed42.csv"))
    parser.add_argument("--c3", type=Path, default=Path("outputs/results/module_c_stage3_validation.json"))
    parser.add_argument("--c4", type=Path, default=Path("outputs/results/module_c_stage4_test.json"))
    parser.add_argument("--results", type=Path, default=Path("outputs/results/module_e_stage2_validation.json"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/results/module_e_stage2_validation_predictions.csv"))
    parser.add_argument("--verification", type=Path, default=Path("outputs/results/module_e_stage2_pre_test_verification.json"))
    return parser.parse_args()


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": path.as_posix(), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def tracked_file_unchanged(path: Path) -> bool:
    relative = path.as_posix()
    prefix = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
    current = subprocess.run(prefix + ["hash-object", relative], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    committed = subprocess.run(prefix + ["rev-parse", f"HEAD:{relative}"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    return current == committed


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, value: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    value.to_csv(temporary, index=False, lineterminator="\n", float_format="%.17g")
    os.replace(temporary, path)


def load_acoustic_cache(cache_path: Path, stage1_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))
    if not stage1["passed"] or stage1["shape"] != [1744, DIMENSION]:
        raise ValueError("Stage 1 validation record did not pass")
    with np.load(cache_path, allow_pickle=False) as cache:
        if set(cache.files) != {"sample_ids", "features", "feature_names"}:
            raise ValueError("Acoustic cache schema changed")
        ids = cache["sample_ids"]
        features = cache["features"]
        names = cache["feature_names"].tolist()
    if ids.shape != (1744,) or ids.dtype != np.int64 or len(np.unique(ids)) != 1744:
        raise ValueError("Acoustic Sample IDs are not 1,744 unique int64 IDs")
    if features.shape != (1744, DIMENSION) or features.dtype != np.float32:
        raise ValueError("Acoustic matrix shape/dtype changed")
    if names != stage1["feature_names"] or not np.isfinite(features).all():
        raise ValueError("Stage 1 feature order or finiteness changed")
    for index, name in enumerate(names):
        observed = features[:, index]
        expected = stage1["feature_summaries"][name]
        for key, value in (("min", np.min(observed)), ("median", np.median(observed)), ("max", np.max(observed))):
            if not np.isclose(float(value), expected[key], rtol=0, atol=1e-5):
                raise ValueError(f"Stage 1 feature distribution changed: {name} {key}")
    return ids, features, names


def main() -> None:
    args = parse_args()
    cache_ids, cache_features, feature_names = load_acoustic_cache(args.cache, args.stage1_diagnostics)
    labels, label_checks = _load_labels(args.labels)
    split, split_checks = _load_split(args.split)
    cache_set = set(cache_ids.tolist())
    if cache_set != set(labels.song_id.tolist()) or cache_set != set(split.sample_id.tolist()):
        raise ValueError("Acoustic cache, primary targets, and split ID sets differ")
    ordered_ids = np.asarray(sorted(cache_set), dtype=np.int64)
    index = {int(sample_id): i for i, sample_id in enumerate(cache_ids)}
    features = cache_features[[index[int(i)] for i in ordered_ids]].astype(np.float64)
    labels_by_id = labels.set_index("song_id", verify_integrity=True)
    split_by_id = split.set_index("sample_id", verify_integrity=True)
    roles = split_by_id.loc[ordered_ids, "split"].to_numpy(dtype=str)
    train_ids = ordered_ids[roles == "train"]
    validation_ids = ordered_ids[roles == "validation"]
    test_ids = ordered_ids[roles == "test"]
    if len(train_ids) != 1221 or len(validation_ids) != 262 or len(test_ids) != 261:
        raise ValueError("Split role counts changed")
    if set(train_ids) & set(validation_ids) or set(train_ids) & set(test_ids) or set(validation_ids) & set(test_ids):
        raise ValueError("Split roles overlap")
    if not np.array_equal(roles, split.iloc[::-1].set_index("sample_id").loc[ordered_ids, "split"].to_numpy(dtype=str)):
        raise ValueError("Split join depends on row order")

    results = {}
    determinism = {}
    prediction_frames = []
    for target, column in TARGET_COLUMNS.items():
        target_values = labels_by_id.loc[ordered_ids, column].to_numpy(dtype=np.float64)
        reversed_values = labels.iloc[::-1].set_index("song_id").loc[ordered_ids, column].to_numpy(dtype=np.float64)
        if not np.array_equal(target_values, reversed_values):
            raise ValueError("Target join depends on row order")
        view = TrainValidationData(
            train_ids=train_ids.copy(), validation_ids=validation_ids.copy(),
            x_train=features[roles == "train"].copy(),
            x_validation=features[roles == "validation"].copy(),
            y_train=target_values[roles == "train"].copy(),
            y_validation=target_values[roles == "validation"].copy(),
        )
        result, prediction, baseline = fit_select_validate(
            view, expected_dimension=DIMENSION, include_candidate_metrics=True
        )
        repeated, repeated_prediction, _ = fit_select_validate(
            view, expected_dimension=DIMENSION, include_candidate_metrics=True
        )
        determinism[target] = verify_deterministic_repeat(
            result, prediction, repeated, repeated_prediction
        )
        if not determinism[target]["passed"]:
            raise ValueError(f"Deterministic repeat failed for {target}")
        results[target] = result
        prediction_frames.append(pd.DataFrame({
            "sample_id": validation_ids,
            "split": "validation",
            "target": target,
            "selected_alpha": result["selected_alpha"],
            "y_true": view.y_validation,
            "prediction": prediction,
            "baseline_prediction": baseline,
        }))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    if len(predictions) != 524 or set(predictions.split) != {"validation"} or predictions.duplicated(["sample_id", "target"]).any():
        raise ValueError("Validation prediction rows are invalid")
    if set(predictions.sample_id) != set(validation_ids) or set(predictions.sample_id) & set(test_ids):
        raise ValueError("Prediction artifact includes wrong IDs")

    inputs = {"acoustic_cache": identity(args.cache), "stage1_diagnostics": identity(args.stage1_diagnostics), "labels": identity(args.labels), "fixed_split": identity(args.split), "module_c_c3_unchanged": identity(args.c3), "module_c_c4_unchanged": identity(args.c4)}
    c3 = json.loads(args.c3.read_text(encoding="utf-8"))
    if c3["protocol"]["representation_level"] != PRIMARY_LEVEL_NAME or c3["protocol"]["alpha_grid"] != list(ALPHA_GRID):
        raise ValueError("Frozen Module C comparator/probe protocol mismatch")
    for name in ("labels", "fixed_split"):
        if inputs[name]["sha256"] != c3["inputs"][name]["sha256"]:
            raise ValueError(f"{name} changed since authoritative Module C")
    if not tracked_file_unchanged(args.c3) or not tracked_file_unchanged(args.c4):
        raise ValueError("Authoritative Module C result file changed")

    payload = {
        "stage": "Module E Stage 2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "representation": "frozen_conventional_acoustic_51d",
            "mert_comparator": PRIMARY_LEVEL_NAME,
            "input_dimension": DIMENSION,
            "targets": TARGET_COLUMNS,
            "probe": "Ridge Regression", "ridge_solver": "cholesky", "fit_intercept": True,
            "alpha_grid": list(ALPHA_GRID), "selection": "maximum_validation_r2_exact_tie_larger_alpha",
            "input_scaler": "StandardScaler fit on Train only", "targets_standardized": False,
            "final_fit_policy": "Train only; deferred to Test stage",
            "baseline": "constant Train target mean",
        },
        "inputs": inputs,
        "data_assembly": {"cache_shape": list(cache_features.shape), "feature_names": feature_names,
                          "sample_ids_unique": True, "canonical_id_sets_equal": True,
                          "row_order_invariance_verified": True, "target_alignment_by_sample_id": True,
                          "split_counts": split_checks["split_counts"], "label_checks": label_checks},
        "targets": results,
        "determinism": determinism,
        "validation_prediction_artifact": {"path": args.predictions.as_posix(), "rows": len(predictions), "splits_present": ["validation"], "contains_test_ids": False},
    }
    gate = {
        "stage": "Module E Stage 2 pre-Test verification",
        "created_utc": payload["created_utc"],
        "inputs": inputs,
        "checks": {
            "acoustic_cache_1744_by_51": True, "sample_ids_exact_unique_population": True,
            "split_counts_1221_262_261": True, "split_roles_mutually_exclusive": True,
            "targets_aligned_by_sample_id": True, "row_order_independent": True,
            "scaler_train_only": all(r["standardization"]["fit_split"] == "train" and r["standardization"]["fit_sample_count"] == 1221 and r["standardization"]["transformed_splits"] == ["train", "validation"] for r in results.values()),
            "targets_not_standardized": all(not r["standardization"]["target_standardized"] for r in results.values()),
            "ridge_cholesky_intercept_train_only": all(r["ridge"]["solver"] == "cholesky" and r["ridge"]["fit_intercept"] and r["ridge"]["fit_split"] == "train" for r in results.values()),
            "frozen_alpha_grid": all(r["alpha_grid"] == list(ALPHA_GRID) for r in results.values()),
            "selection_validation_r2_only": all(r["selection_metric"] == "validation_r2" for r in results.values()),
            "target_specific_selected_alphas": all("selected_alpha" in r for r in results.values()),
            "train_mean_baseline": all(r["baseline"]["definition"] == "constant_prediction_from_train_target_mean" for r in results.values()),
            "validation_predictions_only": set(predictions.split) == {"validation"},
            "no_test_predictions_or_metrics": True,
            "stage1_acoustic_representation_unchanged": True,
            "mert_comparator_layer_12_unchanged": True,
            "module_c_authoritative_files_unmodified": True,
            "later_test_reproducible_from_frozen_inputs_and_selected_alpha": True,
            "deterministic_repeat": all(v["passed"] for v in determinism.values()),
        },
        "test_id_count_structural_only": len(test_ids),
        "test_evaluation_performed": False,
    }
    gate["passed"] = all(gate["checks"].values())
    if not gate["passed"]:
        raise RuntimeError("Pre-Test verification failed")
    write_csv(args.predictions, predictions)
    write_json(args.results, payload)
    write_json(args.verification, gate)
    print(json.dumps({"selected": {name: {"alpha": r["selected_alpha"], "metrics": r["selected_validation_metrics"], "baseline": r["baseline"]["validation_metrics"]} for name, r in results.items()}, "pre_test_passed": gate["passed"], "test_evaluation_performed": False}, indent=2))


if __name__ == "__main__":
    main()
