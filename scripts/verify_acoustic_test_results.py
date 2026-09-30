"""Verify Module E Test predictions and the RQ3 table without refitting models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mert_emotion_probing.probing import TARGET_COLUMNS, regression_metrics  # noqa: E402
from run_acoustic_validation import identity, tracked_file_unchanged, write_json  # noqa: E402


def require(condition: bool, explanation: str) -> None:
    if not condition:
        raise AssertionError(explanation)


def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    result_path = Path("outputs/results/module_e_stage3_test.json")
    prediction_path = Path("outputs/results/module_e_stage3_test_predictions.csv")
    comparison_path = Path("outputs/results/module_e_stage3_rq3_comparison.csv")
    marker_path = Path("outputs/results/module_e_stage3_test_started.json")
    result = read_json(result_path.as_posix())
    marker = read_json(marker_path.as_posix())
    stage2 = read_json("outputs/results/module_e_stage2_validation.json")
    gate = read_json("outputs/results/module_e_stage2_pre_test_verification.json")
    mert = read_json("outputs/results/module_c_stage4_test.json")
    predictions = pd.read_csv(prediction_path, float_precision="round_trip")
    mert_predictions = pd.read_csv("outputs/results/module_c_stage4_test_predictions.csv", float_precision="round_trip")
    comparison = pd.read_csv(comparison_path, float_precision="round_trip")
    provenance = result["provenance"]
    for name, recorded in provenance["inputs"].items():
        require(identity(Path(recorded["path"])) == recorded == stage2["inputs"][name] == gate["inputs"][name], f"Frozen input changed: {name}")
    for name in ("stage2_results", "stage2_validation_predictions", "stage2_pre_test_verification", "module_c_authoritative_results", "module_c_authoritative_predictions"):
        recorded = provenance[name]
        path = Path(recorded["path"])
        require(identity(path) == recorded and tracked_file_unchanged(path), f"Authoritative artifact changed: {name}")
    for recorded in provenance["evaluation_code"]:
        require(identity(Path(recorded["path"])) == recorded, "Evaluation code changed after Test")
    require(marker["provenance"] == provenance, "Test-start provenance differs")
    require(not marker["pre_test_checks"]["prior_acoustic_test_artifacts_present"], "Prior official Test artifacts existed")
    require(marker["official_evaluation_calls_per_target"] == 1, "Wrong official evaluation count")

    with np.load(provenance["inputs"]["acoustic_cache"]["path"], allow_pickle=False) as cache:
        ids, features, names = cache["sample_ids"], cache["features"], cache["feature_names"].tolist()
    require(ids.shape == (1744,) and len(set(ids.tolist())) == 1744 and features.shape == (1744, 51), "Wrong acoustic shape or IDs")
    require(np.isfinite(features).all() and names == stage2["data_assembly"]["feature_names"], "Feature order or values changed")
    split = pd.read_csv(provenance["inputs"]["fixed_split"]["path"])
    require(len(split) == 1744 and not split.sample_id.duplicated().any(), "Wrong split IDs")
    role_ids = {role: set(split.loc[split.split == role, "sample_id"].astype(int)) for role in ("train", "validation", "test")}
    require({role: len(values) for role, values in role_ids.items()} == {"train": 1221, "validation": 262, "test": 261}, "Wrong split counts")
    require(not (role_ids["train"] & role_ids["validation"] or role_ids["train"] & role_ids["test"] or role_ids["validation"] & role_ids["test"]), "Split roles overlap")
    labels = pd.read_csv(provenance["inputs"]["labels"]["path"])
    primary = labels.loc[~labels.is_full_song.astype(bool)].set_index("song_id", verify_integrity=True)
    require(set(ids) == set(primary.index) == set(split.sample_id), "Frozen population mismatch")
    expected_columns = {"sample_id", "split", "target", "representation", "frozen_alpha", "y_true", "prediction", "baseline_prediction"}
    require(set(predictions.columns) == expected_columns and len(predictions) == 522 and set(predictions.split) == {"test"}, "Test prediction schema/count mismatch")
    require(not predictions.duplicated(["sample_id", "target"]).any() and set(predictions.sample_id) == role_ids["test"], "Test prediction IDs mismatch")
    require(set(predictions.target) == set(TARGET_COLUMNS) and set(predictions.representation) == {"conventional_acoustic_51d"}, "Wrong target or representation")
    require(len(comparison) == 4 and not comparison.duplicated(["target", "representation"]).any(), "Comparison keys mismatch")
    require({(row.target, row.representation) for row in comparison.itertuples()} == {(target, representation) for target in TARGET_COLUMNS for representation in ("MERT Layer 12", "Conventional Acoustic 51-D")}, "Comparison representations mismatch")
    require(result["frozen_configuration"]["mert_comparator"] == "transformer_layer_12", "MERT comparator changed")
    cache_by_id = {int(sample_id): index for index, sample_id in enumerate(ids)}
    train_features = features[[cache_by_id[int(sample_id)] for sample_id in sorted(role_ids["train"])]].astype(np.float64)
    independent_numeric_differences = {}
    for target, column in TARGET_COLUMNS.items():
        saved = result["targets"][target]
        alpha = stage2["targets"][target]["selected_alpha"]
        require(alpha == {"valence": 100.0, "arousal": 10.0}[target] == saved["frozen_alpha"] == result["frozen_configuration"]["alphas"][target], "Frozen alpha changed")
        rows = predictions.loc[predictions.target == target]
        require(len(rows) == 261 and set(rows.sample_id) == role_ids["test"] and set(rows.frozen_alpha) == {alpha}, "Target Test rows/alpha mismatch")
        y = primary.loc[rows.sample_id.to_numpy(), column].to_numpy(dtype=np.float64)
        prediction = rows.prediction.to_numpy()
        require(np.array_equal(y, rows.y_true.to_numpy()), "Test targets are not Sample-ID aligned")
        metrics = regression_metrics(y, prediction)
        require(metrics == saved["test_metrics"], "Test metrics do not exactly reproduce from round-trip predictions")
        independent = {
            "mae": float(np.mean(np.abs(y - prediction))),
            "r2": float(1.0 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2)),
            "pearson_r": float(np.corrcoef(y, prediction)[0, 1]),
        }
        require(all(np.isclose(independent[key], metrics[key], rtol=0, atol=1e-12) for key in independent), "Independent metric formula mismatch")
        independent_numeric_differences[target] = {key: abs(independent[key] - metrics[key]) for key in independent}
        train_mean = float(primary.loc[sorted(role_ids["train"]), column].to_numpy(dtype=np.float64).mean())
        baseline = saved["baseline"]
        require(train_mean == baseline["train_target_mean"] and np.all(rows.baseline_prediction.to_numpy() == train_mean), "Baseline is not the Train target mean")
        require(regression_metrics(y, rows.baseline_prediction.to_numpy()) == baseline["test_metrics"] and baseline["test_metrics"]["pearson_r"] is None, "Baseline metrics/null do not reproduce")
        scaling, ridge = saved["standardization"], saved["ridge"]
        require(scaling["fit_split"] == "train" and scaling["fit_sample_count"] == 1221 and scaling["transformed_splits"] == ["train", "test"] and not scaling["target_standardized"] and not scaling["validation_in_final_refit"], "Scaler policy changed")
        require(np.allclose(scaling["fitted_feature_mean"], np.mean(train_features, axis=0), rtol=0, atol=1e-12) and np.allclose(scaling["fitted_feature_variance"], np.var(train_features, axis=0), rtol=0, atol=1e-12), "Fitted scaler statistics are not Train-only")
        require(ridge["fit_split"] == "train" and ridge["fit_sample_count"] == 1221 and ridge["solver"] == "cholesky" and ridge["fit_intercept"], "Final Ridge protocol changed")
        c_rows = mert_predictions.loc[mert_predictions.target == target].set_index("sample_id")
        require(set(c_rows.index) == role_ids["test"], "MERT Test IDs differ")
        require(np.array_equal(c_rows.loc[rows.sample_id, "y_true"].to_numpy(), y), "Acoustic/MERT Test targets differ")
        require(regression_metrics(c_rows.y_true.to_numpy(), c_rows.prediction.to_numpy()) == mert["targets"][target]["test_metrics"], "Authoritative MERT metrics do not reproduce")
        require(mert["targets"][target]["baseline"] == baseline, "Acoustic/MERT Train-mean references differ")
        for representation, dimension, source, reused in (("MERT Layer 12", 768, mert["targets"][target], True), ("Conventional Acoustic 51-D", 51, saved, False)):
            table = comparison.loc[(comparison.target == target) & (comparison.representation == representation)].iloc[0]
            require(table.selected_alpha == source["frozen_alpha"] and table.input_dimension == dimension and bool(table.reused_module_c_evidence) == reused, "Comparison configuration/provenance mismatch")
            require(all(table[f"test_{metric}"] == source["test_metrics"][metric] for metric in ("mae", "r2", "pearson_r")), "Comparison metrics differ from source")
            embedded = [row for row in result["comparison"] if row["target"] == target and row["representation"] == representation][0]
            require(all(embedded[f"test_{metric}"] == source["test_metrics"][metric] for metric in ("mae", "r2", "pearson_r")), "Embedded comparison differs from source")

    controls = result["controls"]
    require(all(controls[key] is False for key in ("validation_in_final_refit", "test_used_for_fitting_or_selection", "post_test_tuning_performed", "alpha_selection_rerun", "acoustic_extraction_rerun", "mert_evaluation_rerun", "rq4_analysis_performed")), "Forbidden work was recorded")
    require(controls["official_evaluation_calls_per_target"] == 1, "Official evaluation count changed")
    report = {
        "stage": "Module E Stage 3 final verification", "passed": True,
        "checks": {
            "cache_identity_shape_and_feature_order_unchanged": True, "population_ids_exact_unique": True,
            "frozen_split_exact_disjoint_1221_262_261": True, "test_ids_exact_261_per_target": True,
            "targets_aligned_by_sample_id": True, "fitted_scaler_statistics_match_train_only": True,
            "final_ridge_train_only_cholesky_intercept": True, "frozen_stage2_alphas_100_and_10": True,
            "targets_original_scale": True, "test_metrics_exactly_recomputed_from_saved_predictions": True,
            "independent_metric_formula_check": True, "train_mean_reference_exactly_recomputed": True,
            "module_c_authoritative_artifacts_unchanged": True, "mert_metrics_exactly_match_authoritative_artifacts": True,
            "comparison_table_exactly_matches_sources": True, "stage2_artifacts_unchanged": True,
            "evaluation_code_unchanged_since_gate_opening": True, "no_post_test_tuning": True,
            "no_rq4_analysis": True,
        },
        "test_prediction_rows": 522, "unique_test_ids": 261,
        "independent_formula_absolute_differences": independent_numeric_differences,
        "verification_refitted_models": False,
        "inputs_verified": provenance,
        "output_identities": [identity(result_path), identity(prediction_path), identity(comparison_path), identity(marker_path)],
    }
    write_json(Path("outputs/results/module_e_stage3_final_verification.json"), report)
    print(json.dumps({"passed": True, "checks_passed": len(report["checks"]), "test_prediction_rows": 522, "unique_test_ids": 261, "verification_refitted_models": False}, indent=2))


if __name__ == "__main__":
    main()
