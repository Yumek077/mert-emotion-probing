"""Module F Stage 2: authorized frozen Test acoustic associations and comparison.

Uses the exact Stage 1 correlate/object definitions and Pearson implementation.
All predictions are reused; no model fitting, feature extraction or selection.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Importing the Stage 1 module only exposes definitions; its main is not run.
import run_acoustic_association_validation as frozen
import matplotlib.pyplot as plt

ROOT = frozen.ROOT
PREFIX = "module_f_stage2"
STAGE1_GATE = "outputs/results/module_f_stage1_pre_test_verification.json"
VERIFICATION = f"outputs/results/{PREFIX}_final_verification.json"
NEW_INPUTS = {
    "mert_test": "outputs/results/module_c_stage4_test.json",
    "mert_test_predictions": "outputs/results/module_c_stage4_test_predictions.csv",
    "acoustic_test": "outputs/results/module_e_stage3_test.json",
    "acoustic_test_predictions": "outputs/results/module_e_stage3_test_predictions.csv",
}
STAGE_FILES = {
    "scripts/run_acoustic_association_test.py", "scripts/verify_acoustic_association_test.py",
    "docs/codex_reports/module_f_stage2_frozen_test_acoustic_association_analysis.md",
    VERIFICATION,
    *[f"outputs/results/{PREFIX}_{name}.csv" for name in
      ("test_analysis_data", "test_associations", "validation_test_comparison")],
    *[f"outputs/figures/{PREFIX}_validation_test_associations.{ext}" for ext in ("png", "svg")],
}


def frozen_files_unchanged() -> bool:
    return not (set(frozen.git("diff", "HEAD", "--name-only").splitlines()) - STAGE_FILES)


def json_input(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def direction(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "undefined"
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def load_test_labels(test_ids: list[int], population: set[int]) -> pd.DataFrame:
    wanted, primary, seen, rows = set(test_ids), set(), set(), []
    with (ROOT / frozen.INPUTS["labels"]).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = int(row["song_id"])
            frozen.require(sample_id not in seen, "Duplicate mapping ID")
            seen.add(sample_id)
            frozen.require(row["is_full_song"] in {"True", "False"}, "Unknown full-song flag")
            if row["is_full_song"] == "False":
                primary.add(sample_id)
            # Train target values never enter an analysis. Parse only Test here.
            if sample_id in wanted:
                frozen.require(row["is_full_song"] == "False", "Test contains a full song")
                rows.append({"sample_id": sample_id, **{t: float(row[c]) for t, c in frozen.TARGETS.items()}})
    frozen.require(primary == population, "Frozen mapping population mismatch")
    labels = pd.DataFrame(rows).set_index("sample_id", verify_integrity=True)
    frozen.require(set(labels.index) == wanted, "Missing Test labels")
    return labels.loc[test_ids]


def load_test_predictions(name: str, result: dict, ids: list[int]) -> pd.DataFrame:
    rows = pd.read_csv(ROOT / NEW_INPUTS[name], float_precision="round_trip")
    frozen.require(len(rows) == 522 and set(rows.split) == {"test"}, "Wrong Test prediction rows")
    frozen.require(set(rows.target) == set(frozen.TARGETS), "Prediction targets changed")
    frozen.require(not rows.duplicated(["sample_id", "target"]).any(), "Duplicate Sample-ID/target keys")
    for target in frozen.TARGETS:
        part = rows.loc[rows.target == target]
        frozen.require(len(part) == 261 and set(part.sample_id) == set(ids), "Wrong Test ID set")
        frozen.require((part.frozen_alpha == result["frozen_configuration"]["alphas"][target]).all(), "Frozen alpha provenance mismatch")
    if name == "mert_test_predictions":
        frozen.require(set(rows.representation_level) == {"transformer_layer_12"}
                       and set(rows.representation_index) == {12}, "MERT must remain Layer 12/index 12")
    else:
        frozen.require(set(rows.representation) == {"conventional_acoustic_51d"}, "Acoustic representation changed")
    frozen.require(np.isfinite(rows[["y_true", "prediction"]].to_numpy()).all(), "Nonfinite source predictions")
    return rows.set_index(["sample_id", "target"], verify_integrity=True)


def compare(validation: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    keys = ["target", "analysis_object", "correlate"]
    valid = validation.set_index(keys, verify_integrity=True)
    records = []
    for _, row in test.iterrows():
        source = valid.loc[tuple(row[key] for key in keys)]
        for column in ("analysis_layer", "feature_name", "association_type",
                       "residual_definition", "positive_residual_meaning"):
            frozen.require(source[column] == row[column], "Validation/Test schema meaning changed")
        vd, td = direction(source.pearson_r), direction(row.pearson_r)
        relation = "undefined" if "undefined" in (vd, td) else "same_direction" if vd == td else "different_direction"
        records.append({"target": row.target, "analysis_layer": row.analysis_layer,
            "analysis_object": row.analysis_object, "correlate": row.correlate,
            "feature_name": row.feature_name, "association_type": row.association_type,
            "validation_partition": "validation", "test_partition": "test",
            "validation_pearson_r": source.pearson_r, "test_pearson_r": row.pearson_r,
            "validation_sample_count": 262, "test_sample_count": 261,
            "validation_direction": vd, "test_direction": td, "direction_comparison": relation,
            "validation_statistic_status": source.statistic_status, "test_statistic_status": row.statistic_status,
            "residual_definition": row.residual_definition, "positive_residual_meaning": row.positive_residual_meaning})
    return pd.DataFrame(records)


def plot_comparison(validation: pd.DataFrame, test: pd.DataFrame) -> None:
    labels = ["A  True target", "B  MERT L12 prediction", "B  Acoustic 51-D prediction",
              "C  MERT L12 residual", "C  Acoustic 51-D residual"]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.4), sharey=True)
    for row_index, (partition, table, count) in enumerate((("Validation", validation, 262), ("Test", test, 261))):
        for col_index, target in enumerate(frozen.TARGETS):
            ax = axes[row_index, col_index]
            matrix = np.array([[table.loc[(table.target == target) & (table.analysis_object == obj)
                               & (table.correlate == correlate), "pearson_r"].iloc[0]
                               for correlate in frozen.CORRELATES] for obj in frozen.OBJECTS], dtype=float)
            mesh = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            ax.set_title(f"{target.capitalize()} · {partition} (n={count})", fontsize=12, fontweight="bold", pad=12)
            ax.set_xticks([0, 1], ["Tempo\n(tempo_bpm)", "Energy\n(rms_mean)"])
            ax.set_yticks(range(5), labels)
            ax.tick_params(length=0, pad=7, labelsize=10)
            for (i, j), value in np.ndenumerate(matrix):
                label = ("undefined" if not np.isfinite(value) else
                         f"{value:+.1e}" if 0 < abs(value) < 0.0005 else f"{value:+.3f}")
                ax.text(j, i, label, ha="center", va="center", fontsize=11,
                        color="white" if abs(value) > 0.6 else "#17202a")
            for boundary in (0.5, 2.5):
                ax.axhline(boundary, color="white", linewidth=2)
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.subplots_adjust(left=0.275, right=0.86, bottom=0.145, top=0.83, hspace=0.52, wspace=0.16)
    cax = fig.add_axes([0.89, 0.22, 0.018, 0.53])
    fig.colorbar(mesh, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1], label="Pearson r")
    fig.suptitle("Module F / RQ4 — Validation and Test acoustic associations", y=0.955, fontsize=14)
    fig.text(0.5, 0.905, "Frozen design · MERT Layer 12 primary · acoustic baseline supporting", ha="center", fontsize=10)
    fig.text(0.5, 0.035, "Residual = true target − prediction; positive residual means under-prediction.\n"
             "Test was viewed in Modules C–E: descriptive consistency check, not untouched confirmation.\n"
             "Pearson associations do not establish causality or a fraction of performance explained.", ha="center", fontsize=9)
    for suffix in ("png", "svg"):
        fig.savefig(ROOT / f"outputs/figures/{PREFIX}_validation_test_associations.{suffix}", dpi=200)
    plt.close(fig)


def main() -> None:
    frozen.require(len(sys.argv) == 1, "Stage 2 accepts no arguments; frozen Test analysis only")
    frozen.require(frozen_files_unchanged(), "A pre-existing tracked file changed")
    stage1 = json_input(STAGE1_GATE)
    frozen.require(stage1["passed"] and stage1["independent_verification"]["passed"]
                   and all(stage1["checks"].values())
                   and all(stage1["independent_verification"]["checks"].values()), "Stage 1 verification did not pass")
    frozen.require(stage1["pre_test_gate"]["status"] == "closed"
                   and not stage1["pre_test_gate"]["stage2_authorized"], "Historical Stage 1 gate was altered")
    frozen.require(stage1["protocol"] == {"partition": "validation", "statistic": "Pearson r",
        "correlates": frozen.CORRELATES, "mert_representation": "transformer_layer_12",
        "supporting_comparator": "frozen_conventional_acoustic_51d", "targets": frozen.TARGETS,
        "residual": "y_true - prediction", "positive_residual": "model under-predicts the target",
        "association_is_causality": False}, "Frozen Stage 1 protocol conflict")
    for section in ("inputs", "outputs", "code"):
        for item in stage1[section].values():
            frozen.require(frozen.identity(item["path"]) == item, f"Stage 1 identity changed: {item['path']}")
    snapshot = {p: frozen.identity(p) for p in sorted(frozen.STAGE_FILES)}
    inputs = {name: frozen.identity(path) for name, path in {**frozen.INPUTS, **NEW_INPUTS}.items()}
    protocol = {**stage1["protocol"], "partition": "test"}
    payload = {"stage": "Module F Stage 2", "started_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"branch": frozen.git("branch", "--show-current"), "head": frozen.git("rev-parse", "HEAD")},
        "protocol": protocol, "inputs": inputs, "stage1_snapshot": snapshot,
        "authorization": {"stage1_researcher_chatgpt_review_passed": True, "stage2_authorized": True,
            "source": "researcher Stage 2 prompt", "client_date": "2026-10-01",
            "statement": "Module F Stage 1 review passed; formally authorized to open Test gate",
            "historical_stage1_gate": "closed; preserved unchanged", "module_f_test_gate": "opened_under_researcher_authorization"},
        "prior_test_exposure": {"viewed_in_modules": ["C", "D", "E"], "untouched_confirmatory_evidence": False,
            "independent_replication": False, "purpose": "frozen descriptive consistency check"},
        "execution": {"test_association_analysis_performed": False, "train_association_analysis_performed": False,
            "model_fits": 0, "audio_extractions": 0, "mert_inference_calls": 0, "alpha_or_layer_selections": 0},
        "independent_verification": {"passed": False, "status": "pending"}, "passed": False}
    # Preserve the researcher authorization before starting Test calculations.
    frozen.write_json(ROOT / VERIFICATION, payload)
    c4, e3 = json_input(NEW_INPUTS["mert_test"]), json_input(NEW_INPUTS["acoustic_test"])
    for name in ("fixed_split", "labels"):
        frozen.require(inputs[name]["sha256"] == c4["provenance"]["inputs"][name]["sha256"]
                       == e3["provenance"]["inputs"][name]["sha256"], "C/E frozen input identity mismatch")
    for name in ("acoustic_cache", "stage1_diagnostics"):
        frozen.require(inputs[name]["sha256"] == e3["provenance"]["inputs"][name]["sha256"], "E3 frozen input identity mismatch")
    for key, name in (("module_c_authoritative_results", "mert_test"),
                      ("module_c_authoritative_predictions", "mert_test_predictions"),
                      ("stage2_results", "acoustic_validation"), ("stage2_validation_predictions", "acoustic_predictions")):
        frozen.require(inputs[name]["sha256"] == e3["provenance"][key]["sha256"], "E3 prediction provenance mismatch")
    for key, name in (("stage_c3_results", "mert_validation"), ("stage_c3_validation_predictions", "mert_predictions")):
        frozen.require(inputs[name]["sha256"] == c4["provenance"][key]["sha256"], "C4 prediction provenance mismatch")
    config_c, config_e = c4["frozen_configuration"], e3["frozen_configuration"]
    frozen.require(config_c["representation_level"] == "transformer_layer_12"
                   and config_c["resolved_representation_index"] == 12, "MERT representation changed")
    frozen.require(config_e["representation"] == "conventional_acoustic_51d"
                   and config_e["input_dimension"] == 51, "Acoustic representation changed")
    for result, alphas in ((c4, {"valence": 1000, "arousal": 1000}), (e3, {"valence": 100, "arousal": 10})):
        cfg = result["frozen_configuration"]
        frozen.require(cfg["alphas"] == alphas and not cfg["target_standardization"]
                       and cfg["final_refit_training_split"] == "train_only"
                       and not cfg["validation_in_final_refit"], "Frozen fitting provenance mismatch")
        for target, alpha in alphas.items():
            r = result["targets"][target]
            frozen.require(r["frozen_alpha"] == alpha and r["ridge"]["fit_split"] == "train"
                           and r["standardization"]["fit_split"] == "train"
                           and not r["standardization"]["target_standardized"], "Saved prediction provenance mismatch")

    split = pd.read_csv(ROOT / frozen.INPUTS["fixed_split"])
    frozen.require(len(split) == 1744 and not split.sample_id.duplicated().any()
                   and split.split.value_counts().to_dict() == {"train": 1221, "validation": 262, "test": 261}, "Frozen split changed")
    test_ids = sorted(split.loc[split.split == "test", "sample_id"].tolist())
    frozen.require(set(test_ids).isdisjoint(stage1["alignment"]["validation_sample_ids"]), "Test overlaps Validation")
    labels = load_test_labels(test_ids, set(split.sample_id))
    with np.load(ROOT / frozen.INPUTS["acoustic_cache"], allow_pickle=False) as cache:
        frozen.require(set(cache.files) == {"sample_ids", "features", "feature_names"}, "Cache schema changed")
        ids, features, names = cache["sample_ids"], cache["features"], cache["feature_names"].tolist()
    frozen.require(ids.shape == (1744,) and ids.dtype == np.int64 and len(set(ids)) == 1744
                   and set(ids) == set(split.sample_id) and features.shape == (1744, 51)
                   and features.dtype == np.float32, "Cache identity/shape changed")
    diag = json_input(frozen.INPUTS["stage1_diagnostics"])
    frozen.require(diag["passed"] and len(set(names)) == 51 and names == diag["feature_names"]
                   == config_e["feature_names"], "Frozen feature names changed")
    indices = {name: names.index(name) for name in frozen.CORRELATES.values()}
    frozen.require(indices == stage1["alignment"]["feature_indices_zero_based"], "Feature selection changed")
    by_id = {int(i): j for j, i in enumerate(ids)}
    acoustic = {name: features[[by_id[i] for i in test_ids], col].astype(np.float64) for name, col in indices.items()}
    mert = load_test_predictions("mert_test_predictions", c4, test_ids)
    conventional = load_test_predictions("acoustic_test_predictions", e3, test_ids)
    frames = []
    for target in frozen.TARGETS:
        keys = pd.MultiIndex.from_product([test_ids, [target]])
        m, a = mert.loc[keys], conventional.loc[keys]
        y = labels[target].to_numpy(dtype=np.float64)
        frozen.require(np.array_equal(m.y_true.to_numpy(), y) and np.array_equal(a.y_true.to_numpy(), y), "ID-aligned targets differ")
        frames.append(pd.DataFrame({"sample_id": test_ids, "partition": "test", "target": target,
            **acoustic, "y_true": y, "mert_layer12_prediction": m.prediction.to_numpy(),
            "acoustic51_prediction": a.prediction.to_numpy(), "mert_layer12_residual": y - m.prediction.to_numpy(),
            "acoustic51_residual": y - a.prediction.to_numpy()}))
    data = pd.concat(frames, ignore_index=True)
    frozen.require(np.isfinite(data.select_dtypes(include="number").to_numpy()).all(), "Nonfinite analysis values")
    records = []
    for target in frozen.TARGETS:
        part = data.loc[data.target == target]
        for obj, (layer, kind, column) in frozen.OBJECTS.items():
            for correlate, feature in frozen.CORRELATES.items():
                r, status = frozen.pearson(part[feature].to_numpy(), part[column].to_numpy())
                records.append({"partition": "test", "target": target, "analysis_layer": layer,
                    "analysis_object": obj, "correlate": correlate, "feature_name": feature,
                    "association_type": kind, "pearson_r": r, "sample_count": 261, "statistic_status": status,
                    "residual_definition": "y_true - prediction", "positive_residual_meaning": "model under-predicts the target"})
    associations = pd.DataFrame(records)
    validation = pd.read_csv(ROOT / "outputs/results/module_f_stage1_validation_associations.csv", float_precision="round_trip")
    frozen.require(list(associations.columns) == list(validation.columns), "Stage 1 association schema changed")
    frozen.require(len(validation) == 20 and set(validation.partition) == {"validation"}
                   and set(validation.sample_count) == {262}, "Wrong Stage 1 association population")
    comparison = compare(validation, associations)
    for name, frame in (("test_analysis_data", data), ("test_associations", associations), ("validation_test_comparison", comparison)):
        frame.to_csv(ROOT / f"outputs/results/{PREFIX}_{name}.csv", index=False, float_format="%.17g", lineterminator="\n")
    plot_comparison(validation, associations)
    frozen.require(snapshot == {p: frozen.identity(p) for p in snapshot}, "Stage 1 file changed during execution")
    frozen.require(inputs == {name: frozen.identity(item["path"]) for name, item in inputs.items()}, "C/E authoritative input changed")
    frozen.require(frozen_files_unchanged(), "A pre-existing tracked file changed")
    paths = [f"outputs/results/{PREFIX}_{name}.csv" for name in ("test_analysis_data", "test_associations", "validation_test_comparison")]
    paths += [f"outputs/figures/{PREFIX}_validation_test_associations.{ext}" for ext in ("png", "svg")]
    payload.update({"completed_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {p: frozen.identity(p) for p in paths},
        "code": {"runner": frozen.identity("scripts/run_acoustic_association_test.py"),
                 "verifier": frozen.identity("scripts/verify_acoustic_association_test.py")},
        "alignment": {"test_sample_ids": test_ids, "sample_count_per_target": 261,
            "analysis_rows": len(data), "association_rows": len(associations), "comparison_rows": len(comparison),
            "feature_indices_zero_based": indices},
        "checks": {name: True for name in (
            "stage1_verification_passed", "researcher_stage2_authorization_recorded",
            "stage1_design_reused_unchanged", "stage1_all_eight_files_unchanged",
            "frozen_split_identity_and_counts", "exact_261_test_ids_per_target", "test_validation_disjoint",
            "composite_prediction_keys_unique", "mert_layer12_and_acoustic51_provenance",
            "frozen_alpha_train_only_original_target_scale", "sample_id_alignment",
            "y_true_exactly_matches_frozen_labels", "named_feature_columns_match_stage1",
            "finite_analysis_values", "residual_y_true_minus_prediction", "comparison_exact_source_values",
            "inputs_unchanged_before_after", "existing_tracked_a_to_e_files_unchanged",
            "no_fitting_extraction_inference_or_selection", "no_train_association_analysis")},
        "undefined_associations": [r for r in records if r["pearson_r"] is None],
        "evidence_provenance": {"mert": "reused Module C Test evidence; no new MERT evaluation",
            "acoustic": "reused Module E frozen 51-D Test predictions",
            "numeric_label_parsing": "Test membership checked before parsing targets",
            "cache_access": "Monolithic NPZ loaded; only Test rows and two named columns enter new associations",
            "comparison_direction_labels": "exact signs only; no threshold or replication/pass-fail classification",
            "known_uncertain_tempo": "ID 437 remains Train, cached estimate retained; absent from both analyzed partitions"},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                     "matplotlib": frozen.matplotlib.__version__}})
    payload["execution"]["test_association_analysis_performed"] = True
    frozen.write_json(ROOT / VERIFICATION, payload)
    print(json.dumps({"test_samples": 261, "test_associations": len(associations),
        "comparison_rows": len(comparison), "stage1_unchanged": True, "independent_verification": "pending"}, indent=2))


if __name__ == "__main__":
    main()
