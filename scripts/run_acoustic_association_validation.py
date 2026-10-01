"""Module F Stage 1: frozen Validation-only descriptive acoustic associations.

Reuses saved predictions and cache values. No model fitting or audio processing.
There is deliberately no partition argument or Test implementation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "module_f_stage1"
INPUTS = {
    "fixed_split": "data/metadata/deam_primary_split_seed42.csv",
    "acoustic_cache": "data/processed/deam_acoustic_51d.npz",
    "labels": "data/raw/deam/verification/deam_item_mapping.csv",
    "stage1_diagnostics": "outputs/results/module_e_stage1_acoustic_diagnostics.json",
    "sample_diagnostics": "data/processed/deam_acoustic_sample_diagnostics.csv",
    "mert_validation": "outputs/results/module_c_stage3_validation.json",
    "mert_predictions": "outputs/results/module_c_stage3_validation_predictions.csv",
    "acoustic_validation": "outputs/results/module_e_stage2_validation.json",
    "acoustic_predictions": "outputs/results/module_e_stage2_validation_predictions.csv",
}
TARGETS = {"valence": "valence_mean", "arousal": "arousal_mean"}
CORRELATES = {"Tempo": "tempo_bpm", "Energy": "rms_mean"}
OBJECTS = {
    "true_target": ("A", "correlate_to_true_target", "y_true"),
    "mert_layer12_prediction": ("B", "correlate_to_prediction", "mert_layer12_prediction"),
    "acoustic51_prediction": ("B", "correlate_to_prediction", "acoustic51_prediction"),
    "mert_layer12_residual": ("C", "correlate_to_residual", "mert_layer12_residual"),
    "acoustic51_residual": ("C", "correlate_to_residual", "acoustic51_residual"),
}
STAGE_FILES = {
    "scripts/run_acoustic_association_validation.py",
    "scripts/verify_acoustic_association_validation.py",
    "docs/codex_reports/module_f_stage1_validation_acoustic_association_analysis.md",
    *[f"outputs/results/{PREFIX}_{name}.csv" for name in ("validation_analysis_data", "validation_associations")],
    f"outputs/results/{PREFIX}_pre_test_verification.json",
    *[f"outputs/figures/{PREFIX}_validation_associations.{ext}" for ext in ("png", "svg")],
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def identity(relative: str) -> dict:
    path = ROOT / relative
    return {"path": relative, "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def frozen_files_unchanged() -> bool:
    # Permit reproducing these Stage artifacts after they are committed, while
    # still requiring every pre-existing tracked file to match HEAD.
    return not (set(git("diff", "HEAD", "--name-only").splitlines()) - STAGE_FILES)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_validation_labels(validation_ids: list[int], population: set[int]) -> pd.DataFrame:
    # Read identity fields from the monolithic mapping, parse numeric targets only
    # after membership in Validation is established.
    wanted = set(validation_ids)
    primary_ids, seen, rows = set(), set(), []
    with (ROOT / INPUTS["labels"]).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = int(row["song_id"])
            require(sample_id not in seen, "Duplicate mapping ID")
            seen.add(sample_id)
            require(row["is_full_song"] in {"True", "False"}, "Unknown full-song flag")
            if row["is_full_song"] == "False":
                primary_ids.add(sample_id)
            if sample_id in wanted:
                require(row["is_full_song"] == "False", "Validation contains a full song")
                rows.append({"sample_id": sample_id,
                             **{target: float(row[column]) for target, column in TARGETS.items()}})
    require(primary_ids == population, "Mapping primary IDs disagree with frozen split")
    labels = pd.DataFrame(rows).set_index("sample_id", verify_integrity=True)
    require(set(labels.index) == wanted, "Missing Validation labels")
    return labels.loc[validation_ids]


def load_predictions(name: str, result: dict, validation_ids: list[int]) -> pd.DataFrame:
    rows = pd.read_csv(ROOT / INPUTS[name], float_precision="round_trip")
    require(len(rows) == 524 and set(rows.split) == {"validation"}, "Predictions must be Validation only")
    require(set(rows.target) == set(TARGETS), "Prediction target schema changed")
    require(not rows.duplicated(["sample_id", "target"]).any(), "Duplicate Sample-ID/target prediction key")
    for target in TARGETS:
        selected = rows.loc[rows.target == target]
        require(len(selected) == 262 and set(selected.sample_id) == set(validation_ids), "Wrong prediction ID set")
        require((selected.selected_alpha == result["targets"][target]["selected_alpha"]).all(), "Saved alpha provenance mismatch")
    if name == "mert_predictions":
        require(set(rows.representation_level) == {"transformer_layer_12"}
                and set(rows.representation_index) == {12}, "MERT must remain Layer 12")
    require(np.isfinite(rows[["y_true", "prediction"]].to_numpy()).all(), "Nonfinite prediction or target")
    return rows.set_index(["sample_id", "target"], verify_integrity=True)


def pearson(x: np.ndarray, y: np.ndarray) -> tuple[float | None, str]:
    require(np.isfinite(x).all() and np.isfinite(y).all(), "Nonfinite correlation input")
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None, "undefined_constant_input"
    return float(np.corrcoef(x, y)[0, 1]), "defined"


def plot_summary(associations: pd.DataFrame) -> None:
    labels = ["A  True target", "B  MERT L12 prediction", "B  Acoustic 51-D prediction",
              "C  MERT L12 residual", "C  Acoustic 51-D residual"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), sharey=True)
    for ax, target in zip(axes, TARGETS):
        matrix = np.array([[associations.loc[(associations.target == target)
                          & (associations.analysis_object == obj)
                          & (associations.correlate == correlate), "pearson_r"].iloc[0]
                          for correlate in CORRELATES] for obj in OBJECTS], dtype=float)
        mesh = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks([0, 1], ["Tempo\n(tempo_bpm)", "Energy\n(rms_mean)"])
        ax.set_yticks(range(5), labels)
        ax.set_title(target.capitalize(), fontsize=13, fontweight="bold")
        ax.tick_params(length=0, pad=8)
        for (row, col), value in np.ndenumerate(matrix):
            label = ("undefined" if not np.isfinite(value) else
                     f"{value:+.1e}" if 0 < abs(value) < 0.0005 else f"{value:+.3f}")
            ax.text(col, row, label, ha="center", va="center", fontsize=11,
                    color="white" if abs(value) > 0.6 else "#17202a")
        for boundary in (0.5, 2.5):
            ax.axhline(boundary, color="white", linewidth=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.28, right=0.86, bottom=0.21, top=0.80, wspace=0.15)
    cax = fig.add_axes([0.89, 0.24, 0.018, 0.53])
    fig.colorbar(mesh, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1], label="Pearson r")
    fig.suptitle("Module F / RQ4 — Validation acoustic associations", y=0.95, fontsize=15)
    fig.text(0.5, 0.86, "Same 262 samples in every cell · MERT Layer 12 primary · acoustic baseline supporting",
             ha="center", fontsize=9)
    fig.text(0.5, 0.055, "Residual = true target − prediction; positive residual means under-prediction.\n"
             "Descriptive linear associations only; association does not establish causality. Test gate closed.",
             ha="center", fontsize=9)
    for suffix in ("png", "svg"):
        fig.savefig(ROOT / f"outputs/figures/{PREFIX}_validation_associations.{suffix}", dpi=200)
    plt.close(fig)


def main() -> None:
    require(len(sys.argv) == 1, "Stage 1 accepts no arguments; Validation only")
    # Existing tracked A–E artifacts must be unchanged. Untracked unrelated files
    # are not read, edited, staged, or removed.
    require(frozen_files_unchanged(), "Existing tracked files changed; review conflict before running")
    inputs = {name: identity(path) for name, path in INPUTS.items()}
    c3 = json.loads((ROOT / INPUTS["mert_validation"]).read_text())
    e2 = json.loads((ROOT / INPUTS["acoustic_validation"]).read_text())
    diagnostics = json.loads((ROOT / INPUTS["stage1_diagnostics"]).read_text())
    for name in ("fixed_split", "labels"):
        require(inputs[name]["sha256"] == c3["inputs"][name]["sha256"]
                == e2["inputs"][name]["sha256"], f"Frozen {name} identity changed")
    for name in ("acoustic_cache", "stage1_diagnostics"):
        require(inputs[name]["sha256"] == e2["inputs"][name]["sha256"], f"Frozen {name} identity changed")
    require(inputs["mert_validation"]["sha256"] == e2["inputs"]["module_c_c3_unchanged"]["sha256"], "C3 identity changed")
    require(c3["protocol"]["representation_level"] == "transformer_layer_12"
            and c3["protocol"]["resolved_representation_index"] == 12, "C3 representation changed")
    require(e2["protocol"]["representation"] == "frozen_conventional_acoustic_51d"
            and e2["protocol"]["input_dimension"] == 51, "E2 representation changed")
    require(not c3["protocol"]["target_standardization"]
            and not e2["protocol"]["targets_standardized"], "Targets must retain original scale")
    require(c3["protocol"]["target_columns"] == TARGETS == e2["protocol"]["targets"], "Frozen target mapping changed")
    for result, expected in ((c3, {"valence": 1000, "arousal": 1000}),
                             (e2, {"valence": 100, "arousal": 10})):
        for target, alpha in expected.items():
            require(result["targets"][target]["selected_alpha"] == alpha, "Frozen alpha changed")
            require(result["targets"][target]["ridge"]["fit_split"] == "train"
                    and result["targets"][target]["standardization"]["fit_split"] == "train", "Prediction fitting provenance changed")

    split = pd.read_csv(ROOT / INPUTS["fixed_split"])
    require(len(split) == 1744 and not split.sample_id.duplicated().any(), "Split IDs invalid")
    require(split.split.value_counts().to_dict() == {"train": 1221, "validation": 262, "test": 261}, "Split counts changed")
    population = set(split.sample_id)
    validation_ids = sorted(split.loc[split.split == "validation", "sample_id"].tolist())
    labels = load_validation_labels(validation_ids, population)
    with np.load(ROOT / INPUTS["acoustic_cache"], allow_pickle=False) as cache:
        require(set(cache.files) == {"sample_ids", "features", "feature_names"}, "Cache schema changed")
        ids, features, names = cache["sample_ids"], cache["features"], cache["feature_names"].tolist()
    require(ids.shape == (1744,) and ids.dtype == np.int64 and len(set(ids)) == 1744
            and set(ids) == population, "Cache IDs do not match frozen population")
    require(features.shape == (1744, 51) and features.dtype == np.float32, "Cache shape/dtype changed")
    require(diagnostics["passed"] and names == diagnostics["feature_names"]
            == e2["data_assembly"]["feature_names"] and len(set(names)) == 51, "Frozen feature names changed")
    indices = {name: names.index(name) for name in CORRELATES.values()}
    by_id = {int(sample_id): row for row, sample_id in enumerate(ids)}
    acoustic = pd.DataFrame({"sample_id": validation_ids,
        **{name: features[[by_id[i] for i in validation_ids], index].astype(np.float64)
           for name, index in indices.items()}}).set_index("sample_id", verify_integrity=True)
    mert = load_predictions("mert_predictions", c3, validation_ids)
    conventional = load_predictions("acoustic_predictions", e2, validation_ids)
    frames = []
    for target in TARGETS:
        keys = pd.MultiIndex.from_product([validation_ids, [target]])
        m, a = mert.loc[keys], conventional.loc[keys]
        y = labels[target].to_numpy(dtype=np.float64)
        require(np.array_equal(m.y_true.to_numpy(), y) and np.array_equal(a.y_true.to_numpy(), y), "ID-aligned y_true differs from frozen labels")
        frames.append(pd.DataFrame({"sample_id": validation_ids, "partition": "validation", "target": target,
            **{name: acoustic[name].to_numpy() for name in CORRELATES.values()}, "y_true": y,
            "mert_layer12_prediction": m.prediction.to_numpy(),
            "acoustic51_prediction": a.prediction.to_numpy(),
            "mert_layer12_residual": y - m.prediction.to_numpy(),
            "acoustic51_residual": y - a.prediction.to_numpy()}))
    data = pd.concat(frames, ignore_index=True)
    require(np.isfinite(data.select_dtypes(include="number").to_numpy()).all(), "Nonfinite analysis data")
    rows = []
    for target in TARGETS:
        frame = data.loc[data.target == target]
        for obj, (layer, association_type, column) in OBJECTS.items():
            for correlate, feature in CORRELATES.items():
                r, status = pearson(frame[feature].to_numpy(), frame[column].to_numpy())
                rows.append({"partition": "validation", "target": target, "analysis_layer": layer,
                    "analysis_object": obj, "correlate": correlate, "feature_name": feature,
                    "association_type": association_type, "pearson_r": r, "sample_count": 262,
                    "statistic_status": status, "residual_definition": "y_true - prediction",
                    "positive_residual_meaning": "model under-predicts the target"})
    associations = pd.DataFrame(rows)
    result_dir = ROOT / "outputs/results"
    figure_dir = ROOT / "outputs/figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("validation_analysis_data", data), ("validation_associations", associations)):
        frame.to_csv(result_dir / f"{PREFIX}_{name}.csv", index=False, float_format="%.17g", lineterminator="\n")
    plot_summary(associations)
    require(inputs == {name: identity(path) for name, path in INPUTS.items()}, "An authoritative input changed during execution")
    require(frozen_files_unchanged(), "Existing tracked artifacts changed during execution")
    uncertain = split.loc[split.sample_id == 437, "split"].item()
    with (ROOT / INPUTS["sample_diagnostics"]).open(newline="", encoding="utf-8") as handle:
        uncertain_rows = [row for row in csv.DictReader(handle) if int(row["sample_id"]) == 437]
    require(len(uncertain_rows) == 1 and int(uncertain_rows[0]["beat_count"]) == 1
            and float(uncertain_rows[0]["tempo_raw_bpm"]) == 117.1875, "Known ID 437 diagnostic changed")
    checks = {name: True for name in (
        "frozen_input_hashes_match_c3_e2", "frozen_split_1744_unique_ids_counts_1221_262_261",
        "validation_exact_262_ids_per_target", "prediction_composite_keys_unique",
        "sample_id_alignment", "both_y_true_exactly_match_frozen_labels",
        "cache_feature_names_exact", "finite_analysis_values", "residual_y_true_minus_prediction",
        "prediction_provenance_layer12_and_frozen_alphas", "original_target_scale",
        "inputs_unchanged_before_after", "existing_tracked_a_to_e_files_unchanged",
        "no_refit_reextraction_or_mert_inference", "no_test_predictions_read_or_associations_computed")}
    output_paths = [f"outputs/results/{PREFIX}_{name}.csv" for name in
                    ("validation_analysis_data", "validation_associations")]
    output_paths += [f"outputs/figures/{PREFIX}_validation_associations.{ext}" for ext in ("png", "svg")]
    payload = {"stage": "Module F Stage 1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD")},
        "protocol": {"partition": "validation", "statistic": "Pearson r", "correlates": CORRELATES,
            "mert_representation": "transformer_layer_12", "supporting_comparator": "frozen_conventional_acoustic_51d",
            "targets": TARGETS, "residual": "y_true - prediction",
            "positive_residual": "model under-predicts the target", "association_is_causality": False},
        "inputs": inputs, "outputs": {p: identity(p) for p in output_paths},
        "code": {"runner": identity("scripts/run_acoustic_association_validation.py"),
                 "verifier": identity("scripts/verify_acoustic_association_validation.py")},
        "alignment": {"validation_sample_ids": validation_ids, "sample_count_per_target": 262,
            "analysis_rows": 524, "association_rows": 20, "feature_indices_zero_based": indices},
        "checks": checks, "independent_verification": {"passed": False, "status": "pending"},
        "undefined_associations": [row for row in rows if row["pearson_r"] is None],
        "known_measurement_uncertainty": {"sample_id": 437, "partition": uncertain,
            "tempo_bpm": 117.1875, "beat_count": 1, "estimate_retained": True,
            "included_in_validation": 437 in validation_ids},
        "scope_evidence": {"numeric_label_parsing": "Validation membership checked before parsing targets",
            "cache_access": "Monolithic NPZ loaded; only Validation rows and the two named columns enter analysis",
            "test_prediction_files_opened": [], "prediction_files_opened": [INPUTS["mert_predictions"], INPUTS["acoustic_predictions"]],
            "model_fits": 0, "audio_extractions": 0, "mert_inference_calls": 0},
        "pre_test_gate": {"status": "closed", "researcher_approval_required": True,
            "stage2_authorized": False, "test_association_analysis_performed": False,
            "future_test_provenance": "Test was viewed in Modules C–E; future frozen descriptive consistency check is not untouched confirmatory evidence"},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__}, "passed": False}
    write_json(result_dir / f"{PREFIX}_pre_test_verification.json", payload)
    print(json.dumps({"validation_samples": 262, "association_rows": 20,
                      "independent_verification": "pending", "test_gate": "closed"}, indent=2))


if __name__ == "__main__":
    main()
