"""Independently reopen and verify F1 artifacts; no imports from its runner.

Uses csv ID dictionaries and math.fsum Pearson calculations, separately from
the runner's pandas alignment and numpy.corrcoef. Validation only; no model fit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "module_f_stage1"
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
        raise AssertionError(message)


def read_rows(relative: str) -> list[dict]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def independent_pearson(x: list[float], y: list[float]) -> float | None:
    require(len(x) == len(y) == 262, "Wrong Pearson population")
    require(all(math.isfinite(v) for v in x + y), "Nonfinite input")
    if len(set(x)) == 1 or len(set(y)) == 1:
        return None
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    covariance = math.fsum(a * b for a, b in zip(dx, dy))
    denominator = math.sqrt(math.fsum(v * v for v in dx) * math.fsum(v * v for v in dy))
    return covariance / denominator


def main() -> None:
    require(len(sys.argv) == 1, "Stage 1 verifier accepts no arguments")
    gate_path = ROOT / f"outputs/results/{PREFIX}_pre_test_verification.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    # Invalidate a previous pass before checking; a failed rerun cannot leave a
    # stale successful gate. The Test gate remains closed regardless of checks.
    gate["passed"] = False
    gate["independent_verification"] = {"passed": False, "status": "in_progress"}
    gate_path.write_text(json.dumps(gate, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    expected_inputs = {
        "data/metadata/deam_primary_split_seed42.csv", "data/processed/deam_acoustic_51d.npz",
        "data/raw/deam/verification/deam_item_mapping.csv", "data/processed/deam_acoustic_sample_diagnostics.csv",
        "outputs/results/module_e_stage1_acoustic_diagnostics.json", "outputs/results/module_c_stage3_validation.json",
        "outputs/results/module_c_stage3_validation_predictions.csv", "outputs/results/module_e_stage2_validation.json",
        "outputs/results/module_e_stage2_validation_predictions.csv"}
    require({v["path"] for v in gate["inputs"].values()} == expected_inputs, "Unexpected input scope")
    for item in list(gate["inputs"].values()) + list(gate["outputs"].values()) + list(gate["code"].values()):
        require(sha256(item["path"]) == item["sha256"], f"Changed artifact: {item['path']}")
    diff = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True).strip()
    require(not (set(diff.splitlines()) - STAGE_FILES), "Existing tracked A–E files modified")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(head == gate["git"]["head"], "HEAD changed since analysis")
    require(gate["protocol"] == {
        "partition": "validation", "statistic": "Pearson r",
        "correlates": {"Tempo": "tempo_bpm", "Energy": "rms_mean"},
        "mert_representation": "transformer_layer_12",
        "supporting_comparator": "frozen_conventional_acoustic_51d",
        "targets": {"valence": "valence_mean", "arousal": "arousal_mean"},
        "residual": "y_true - prediction", "positive_residual": "model under-predicts the target",
        "association_is_causality": False}, "Frozen Module F design changed")
    require(gate["scope_evidence"]["test_prediction_files_opened"] == []
            and all(gate["scope_evidence"][key] == 0 for key in
                    ("model_fits", "audio_extractions", "mert_inference_calls")), "Execution scope changed")

    split = read_rows("data/metadata/deam_primary_split_seed42.csv")
    roles = {int(row["sample_id"]): row["split"] for row in reversed(split)}
    require(len(split) == len(roles) == 1744, "Duplicate split IDs")
    require({role: list(roles.values()).count(role) for role in set(roles.values())}
            == {"train": 1221, "validation": 262, "test": 261}, "Wrong split counts")
    ids = sorted(i for i, role in roles.items() if role == "validation")
    require(len(ids) == 262 and ids == gate["alignment"]["validation_sample_ids"], "Wrong Validation identity")
    c3 = json.loads((ROOT / "outputs/results/module_c_stage3_validation.json").read_text())
    e2 = json.loads((ROOT / "outputs/results/module_e_stage2_validation.json").read_text())
    for name in ("fixed_split", "labels"):
        require(gate["inputs"][name]["sha256"] == c3["inputs"][name]["sha256"]
                == e2["inputs"][name]["sha256"], "Frozen split/labels mismatch")
    require(gate["inputs"]["acoustic_cache"]["sha256"] == e2["inputs"]["acoustic_cache"]["sha256"], "Frozen cache mismatch")
    require(c3["protocol"]["resolved_representation_index"] == 12
            and e2["protocol"]["input_dimension"] == 51, "Representation provenance mismatch")
    require({t: c3["targets"][t]["selected_alpha"] for t in ("valence", "arousal")}
            == {"valence": 1000, "arousal": 1000}
            and {t: e2["targets"][t]["selected_alpha"] for t in ("valence", "arousal")}
            == {"valence": 100, "arousal": 10}, "Frozen alpha provenance changed")
    wanted = set(ids)
    labels, primary_ids, mapping_seen = {}, set(), set()
    with (ROOT / "data/raw/deam/verification/deam_item_mapping.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            i = int(row["song_id"])
            require(i not in mapping_seen, "Duplicate mapping ID")
            mapping_seen.add(i)
            if row["is_full_song"] == "False":
                primary_ids.add(i)
            if i in wanted:
                labels[i] = {t: float(row[t + "_mean"]) for t in ("valence", "arousal")}
    require(primary_ids == set(roles) and set(labels) == wanted, "Mapping identity mismatch")
    with np.load(ROOT / "data/processed/deam_acoustic_51d.npz", allow_pickle=False) as cache:
        names, cache_ids, features = cache["feature_names"].tolist(), cache["sample_ids"], cache["features"]
    diag = json.loads((ROOT / "outputs/results/module_e_stage1_acoustic_diagnostics.json").read_text())
    require(names == diag["feature_names"] == e2["data_assembly"]["feature_names"], "Wrong feature order")
    require(len(set(cache_ids)) == 1744 and set(cache_ids) == set(roles)
            and features.shape == (1744, 51), "Cache population mismatch")
    columns = {name: names.index(name) for name in ("tempo_bpm", "rms_mean")}
    require(columns == gate["alignment"]["feature_indices_zero_based"], "Wrong feature selection")
    cache_by_id = {int(cache_ids[j]): {name: float(features[j, col]) for name, col in columns.items()}
                   for j in reversed(range(len(cache_ids))) if int(cache_ids[j]) in wanted}

    predictions = {}
    for model, path, result in (("mert_layer12", "outputs/results/module_c_stage3_validation_predictions.csv", c3),
                                ("acoustic51", "outputs/results/module_e_stage2_validation_predictions.csv", e2)):
        rows = read_rows(path)
        keyed = {}
        for row in reversed(rows):
            key = (int(row["sample_id"]), row["target"])
            require(key not in keyed and row["split"] == "validation", "Duplicate/wrong prediction key")
            require(float(row["selected_alpha"]) == result["targets"][row["target"]]["selected_alpha"], "Wrong saved alpha")
            if model == "mert_layer12":
                require(row["representation_level"] == "transformer_layer_12"
                        and int(row["representation_index"]) == 12, "Wrong MERT layer")
            keyed[key] = row
        require(len(rows) == 524 and set(keyed) == {(i, t) for i in ids for t in ("valence", "arousal")}, "Wrong prediction population")
        predictions[model] = keyed

    analysis_rows = read_rows(f"outputs/results/{PREFIX}_validation_analysis_data.csv")
    data = {}
    for row in reversed(analysis_rows):
        key = (int(row["sample_id"]), row["target"])
        require(key not in data and row["partition"] == "validation", "Wrong analysis key/partition")
        i, target = key
        require(i in wanted, "Non-Validation analysis row")
        numbers = {name: float(value) for name, value in row.items()
                   if name not in {"sample_id", "target", "partition"}}
        require(all(math.isfinite(v) for v in numbers.values()), "Nonfinite saved value")
        require(numbers["y_true"] == labels[i][target], "Saved target mismatch")
        for name in columns:
            require(numbers[name] == cache_by_id[i][name], "Cache feature value mismatch")
        for model in predictions:
            source = predictions[model][key]
            require(float(source["y_true"]) == numbers["y_true"], "Prediction label mismatch")
            require(float(source["prediction"]) == numbers[model + "_prediction"], "Saved prediction mismatch")
            require(numbers[model + "_residual"] == numbers["y_true"] - float(source["prediction"]), "Residual does not reproduce exactly")
        data[key] = numbers
    require(len(analysis_rows) == 524 and set(data) == set(predictions["mert_layer12"]), "Wrong saved analysis population")

    objects = {"true_target": ("A", "correlate_to_true_target", "y_true"),
               "mert_layer12_prediction": ("B", "correlate_to_prediction", "mert_layer12_prediction"),
               "acoustic51_prediction": ("B", "correlate_to_prediction", "acoustic51_prediction"),
               "mert_layer12_residual": ("C", "correlate_to_residual", "mert_layer12_residual"),
               "acoustic51_residual": ("C", "correlate_to_residual", "acoustic51_residual")}
    correlates = {"Tempo": "tempo_bpm", "Energy": "rms_mean"}
    associations = read_rows(f"outputs/results/{PREFIX}_validation_associations.csv")
    keys, differences, undefined = set(), [], 0
    for row in associations:
        key = (row["target"], row["analysis_object"], row["correlate"])
        require(key not in keys, "Duplicate association")
        keys.add(key)
        target, obj, correlate = key
        layer, kind, column = objects[obj]
        feature = correlates[correlate]
        require(row["partition"] == "validation" and int(row["sample_count"]) == 262
                and row["analysis_layer"] == layer and row["association_type"] == kind
                and row["feature_name"] == feature, "Wrong association schema")
        require(row["residual_definition"] == "y_true - prediction"
                and row["positive_residual_meaning"] == "model under-predicts the target", "Wrong residual sign metadata")
        r = independent_pearson([data[(i, target)][feature] for i in ids],
                                [data[(i, target)][column] for i in ids])
        if r is None:
            require(row["pearson_r"] == "" and row["statistic_status"] == "undefined_constant_input", "Undefined Pearson not preserved")
            undefined += 1
        else:
            saved = float(row["pearson_r"])
            require(math.isfinite(saved) and -1 <= saved <= 1 and row["statistic_status"] == "defined", "Invalid Pearson")
            differences.append(abs(r - saved))
            require(abs(r - saved) <= 1e-12, "Independent Pearson mismatch")
    require(len(associations) == 20 and keys == {(t, o, c) for t in ("valence", "arousal")
                                               for o in objects for c in correlates}, "Incomplete association table")
    require(undefined == len(gate["undefined_associations"]), "Undefined statistic record mismatch")
    require(roles[437] == "train" and 437 not in wanted
            and gate["known_measurement_uncertainty"]["estimate_retained"], "ID 437 treatment changed")
    require(gate["pre_test_gate"]["status"] == "closed"
            and gate["pre_test_gate"]["researcher_approval_required"]
            and not gate["pre_test_gate"]["stage2_authorized"]
            and not gate["pre_test_gate"]["test_association_analysis_performed"], "Test gate changed")
    checks = {name: True for name in (
        "input_output_code_hashes", "existing_tracked_a_to_e_files_unchanged", "frozen_split_identity_and_counts",
        "prediction_provenance", "composite_prediction_keys", "exact_validation_id_sets",
        "reversed_input_order_independent_alignment", "feature_names_and_exact_selected_values",
        "saved_targets_exactly_match_labels_and_both_prediction_files", "saved_predictions_exactly_match_sources",
        "residuals_reproduce_exactly", "all_saved_values_finite", "all_20_pearsons_independently_recomputed",
        "undefined_statistics_preserved", "known_uncertain_tempo_retained", "test_gate_closed",
        "frozen_module_f_design_and_execution_scope")}
    gate["independent_verification"] = {"passed": True, "status": "passed", "checks": checks,
        "method": "stdlib csv ID dictionaries (reversed source order), float residual subtraction, math.fsum centered Pearson formula",
        "pearson_absolute_tolerance": 1e-12, "maximum_absolute_difference": max(differences, default=0),
        "association_count": 20, "undefined_count": undefined}
    gate["passed"] = all(gate["checks"].values()) and all(checks.values())
    temporary = gate_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(gate, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(gate_path)
    print(json.dumps({"passed": gate["passed"], "independent_checks": len(checks),
                      "maximum_pearson_difference": max(differences, default=0), "test_gate": "closed"}, indent=2))


if __name__ == "__main__":
    main()
