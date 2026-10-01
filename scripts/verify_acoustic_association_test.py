"""Independently verify the frozen Module F Test analysis and comparison.

No runner is imported. CSV dictionaries use reversed source order; Pearson r
uses centered Python math.fsum calculations rather than numpy.corrcoef. The
only writable artifact is this Stage's combined final verification JSON.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "module_f_stage2"
STAGE1_PREFIX = "module_f_stage1"
TARGETS = ("valence", "arousal")
CORRELATES = {"Tempo": "tempo_bpm", "Energy": "rms_mean"}
OBJECTS = {
    "true_target": ("A", "correlate_to_true_target", "y_true"),
    "mert_layer12_prediction": ("B", "correlate_to_prediction", "mert_layer12_prediction"),
    "acoustic51_prediction": ("B", "correlate_to_prediction", "acoustic51_prediction"),
    "mert_layer12_residual": ("C", "correlate_to_residual", "mert_layer12_residual"),
    "acoustic51_residual": ("C", "correlate_to_residual", "acoustic51_residual"),
}
ANALYSIS_COLUMNS = [
    "sample_id", "partition", "target", "tempo_bpm", "rms_mean", "y_true",
    "mert_layer12_prediction", "acoustic51_prediction", "mert_layer12_residual", "acoustic51_residual",
]
ASSOCIATION_COLUMNS = [
    "partition", "target", "analysis_layer", "analysis_object", "correlate", "feature_name",
    "association_type", "pearson_r", "sample_count", "statistic_status", "residual_definition",
    "positive_residual_meaning",
]
COMPARISON_COLUMNS = [
    "target", "analysis_layer", "analysis_object", "correlate", "feature_name", "association_type",
    "validation_partition", "test_partition", "validation_pearson_r", "test_pearson_r",
    "validation_sample_count", "test_sample_count", "validation_direction", "test_direction",
    "direction_comparison", "validation_statistic_status", "test_statistic_status",
    "residual_definition", "positive_residual_meaning",
]
STAGE1_FILES = {
    "scripts/run_acoustic_association_validation.py",
    "scripts/verify_acoustic_association_validation.py",
    "docs/codex_reports/module_f_stage1_validation_acoustic_association_analysis.md",
    f"outputs/results/{STAGE1_PREFIX}_pre_test_verification.json",
    *[f"outputs/results/{STAGE1_PREFIX}_{name}.csv" for name in
      ("validation_analysis_data", "validation_associations")],
    *[f"outputs/figures/{STAGE1_PREFIX}_validation_associations.{ext}" for ext in ("png", "svg")],
}
OUTPUT_FILES = {
    *[f"outputs/results/{PREFIX}_{name}.csv" for name in
      ("test_analysis_data", "test_associations", "validation_test_comparison")],
    *[f"outputs/figures/{PREFIX}_validation_test_associations.{ext}" for ext in ("png", "svg")],
}
CODE_FILES = {"scripts/run_acoustic_association_test.py", "scripts/verify_acoustic_association_test.py"}
STAGE2_FILES = {
    *CODE_FILES, *OUTPUT_FILES,
    f"outputs/results/{PREFIX}_final_verification.json",
    "docs/codex_reports/module_f_stage2_frozen_test_acoustic_association_analysis.md",
}
EXPECTED_INPUTS = {
    "data/metadata/deam_primary_split_seed42.csv", "data/processed/deam_acoustic_51d.npz",
    "data/raw/deam/verification/deam_item_mapping.csv", "data/processed/deam_acoustic_sample_diagnostics.csv",
    "outputs/results/module_e_stage1_acoustic_diagnostics.json", "outputs/results/module_c_stage3_validation.json",
    "outputs/results/module_c_stage3_validation_predictions.csv", "outputs/results/module_e_stage2_validation.json",
    "outputs/results/module_e_stage2_validation_predictions.csv", "outputs/results/module_c_stage4_test.json",
    "outputs/results/module_c_stage4_test_predictions.csv", "outputs/results/module_e_stage3_test.json",
    "outputs/results/module_e_stage3_test_predictions.csv",
}
PEARSON_TOLERANCE = 1e-12  # Numerical implementation agreement only.


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def write_gate(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_rows(relative: str, columns: list[str] | None = None) -> list[dict]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames),
                f"Missing or duplicate header: {relative}")
        if columns is not None:
            require(reader.fieldnames == columns, f"Changed frozen CSV schema: {relative}")
        rows = list(reader)
    require(all(None not in row and all(value is not None for value in row.values()) for row in rows),
            f"Malformed CSV rows: {relative}")
    return rows


def identity(relative: str) -> dict:
    path = ROOT / relative
    return {"path": relative, "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify_identity(item: dict) -> None:
    require(item == identity(item["path"]), f"Changed artifact identity: {item['path']}")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def independent_pearson(x: list[float], y: list[float], count: int) -> float | None:
    require(len(x) == len(y) == count, "Wrong Pearson population")
    require(all(math.isfinite(v) for v in x + y), "Nonfinite Pearson input")
    if len(set(x)) == 1 or len(set(y)) == 1:
        return None
    mx, my = math.fsum(x) / count, math.fsum(y) / count
    dx, dy = [v - mx for v in x], [v - my for v in y]
    covariance = math.fsum(a * b for a, b in zip(dx, dy))
    denominator = math.sqrt(math.fsum(v * v for v in dx) * math.fsum(v * v for v in dy))
    require(denominator > 0, "Nonconstant input has invalid centered variance")
    return covariance / denominator


def load_predictions(partition: str, ids: list[int], c: dict, e: dict) -> dict:
    wanted_keys = {(i, target) for i in ids for target in TARGETS}
    stages = ("module_c_stage4_test", "module_e_stage3_test") if partition == "test" else (
        "module_c_stage3_validation", "module_e_stage2_validation")
    predictions = {}
    for model, stage, result in zip(("mert_layer12", "acoustic51"), stages, (c, e)):
        rows = read_rows(f"outputs/results/{stage}_predictions.csv")
        keyed = {}
        alpha_field = "frozen_alpha" if partition == "test" else "selected_alpha"
        for row in reversed(rows):
            key = (int(row["sample_id"]), row["target"])
            require(key in wanted_keys and key not in keyed and row["split"] == partition,
                    f"Duplicate, extra, or wrong partition prediction key: {stage}")
            target = key[1]
            require(float(row[alpha_field]) == result["targets"][target][alpha_field], "Saved alpha mismatch")
            require(all(math.isfinite(float(row[name])) for name in ("y_true", "prediction")),
                    "Nonfinite source prediction or label")
            if model == "mert_layer12":
                require(row["representation_level"] == "transformer_layer_12"
                        and int(row["representation_index"]) == 12, "Saved MERT layer mismatch")
            elif partition == "test":
                require(row["representation"] == "conventional_acoustic_51d", "Wrong saved acoustic representation")
            keyed[key] = row
        require(len(rows) == len(wanted_keys) and set(keyed) == wanted_keys, "Missing prediction IDs/targets")
        predictions[model] = keyed
    return predictions


def load_analysis(partition: str, ids: list[int], labels: dict, cache_by_id: dict,
                  predictions: dict) -> dict:
    prefix = PREFIX if partition == "test" else STAGE1_PREFIX
    rows = read_rows(f"outputs/results/{prefix}_{partition}_analysis_data.csv", ANALYSIS_COLUMNS)
    wanted_keys = {(i, target) for i in ids for target in TARGETS}
    data = {}
    for row in reversed(rows):
        key = (int(row["sample_id"]), row["target"])
        require(key in wanted_keys and key not in data and row["partition"] == partition,
                "Duplicate, extra, or wrong analysis key/partition")
        sample_id, target = key
        numbers = {name: float(row[name]) for name in ANALYSIS_COLUMNS[3:]}
        require(all(math.isfinite(v) for v in numbers.values()), "Nonfinite saved analysis value")
        require(numbers["y_true"] == labels[sample_id][target], "Saved target differs from frozen original-scale label")
        for feature in CORRELATES.values():
            require(numbers[feature] == cache_by_id[sample_id][feature], "Saved named-cache value mismatch")
        for model, keyed in predictions.items():
            source = keyed[key]
            require(float(source["y_true"]) == numbers["y_true"], "Saved prediction target mismatch")
            require(float(source["prediction"]) == numbers[model + "_prediction"], "Saved prediction differs from source")
            require(numbers[model + "_residual"] == numbers["y_true"] - float(source["prediction"]),
                    "Residual fails exact y_true - prediction reproduction")
        data[key] = numbers
    require(len(rows) == len(wanted_keys) and set(data) == wanted_keys, "Wrong aligned analysis population")
    return data


def verify_associations(partition: str, ids: list[int], data: dict) -> tuple[dict, list[float], list[dict]]:
    prefix = PREFIX if partition == "test" else STAGE1_PREFIX
    rows = read_rows(f"outputs/results/{prefix}_{partition}_associations.csv", ASSOCIATION_COLUMNS)
    wanted_keys = {(target, obj, correlate) for target in TARGETS for obj in OBJECTS for correlate in CORRELATES}
    keyed, differences, undefined = {}, [], []
    for row in reversed(rows):
        key = (row["target"], row["analysis_object"], row["correlate"])
        require(key in wanted_keys and key not in keyed, "Unexpected or duplicate association key")
        target, obj, correlate = key
        layer, kind, column = OBJECTS[obj]
        feature = CORRELATES[correlate]
        require(row["partition"] == partition and int(row["sample_count"]) == len(ids)
                and row["analysis_layer"] == layer and row["association_type"] == kind
                and row["feature_name"] == feature, "Association frozen design/schema mismatch")
        require(row["residual_definition"] == "y_true - prediction"
                and row["positive_residual_meaning"] == "model under-predicts the target", "Wrong residual sign metadata")
        r = independent_pearson([data[(i, target)][feature] for i in reversed(ids)],
                                [data[(i, target)][column] for i in reversed(ids)], len(ids))
        if r is None:
            require(row["pearson_r"] == "" and row["statistic_status"] == "undefined_constant_input",
                    "Frozen undefined/constant-input policy changed")
            undefined.append({**row, "pearson_r": None, "sample_count": int(row["sample_count"])})
        else:
            saved = float(row["pearson_r"])
            require(math.isfinite(saved) and -1 <= saved <= 1 and row["statistic_status"] == "defined",
                    "Invalid saved Pearson")
            difference = abs(r - saved)
            require(difference <= PEARSON_TOLERANCE, "Independent Pearson mismatch")
            differences.append(difference)
        keyed[key] = row
    require(len(rows) == 20 and set(keyed) == wanted_keys, "Incomplete 20-association population")
    return keyed, differences, undefined


def direction(row: dict) -> str:
    if row["statistic_status"] != "defined":
        return "undefined"
    r = float(row["pearson_r"])
    return "positive" if r > 0 else "negative" if r < 0 else "zero"


def verify_comparison(validation: dict, test: dict) -> None:
    rows = read_rows(f"outputs/results/{PREFIX}_validation_test_comparison.csv", COMPARISON_COLUMNS)
    seen = set()
    for row in reversed(rows):
        key = (row["target"], row["analysis_object"], row["correlate"])
        require(key in validation and key in test and key not in seen, "Unexpected/duplicate comparison key")
        seen.add(key)
        v, t = validation[key], test[key]
        for column in ("target", "analysis_layer", "analysis_object", "correlate", "feature_name",
                       "association_type", "residual_definition", "positive_residual_meaning"):
            require(row[column] == v[column] == t[column], "Comparison metadata differs from source associations")
        for name, source in (("validation", v), ("test", t)):
            require(row[name + "_partition"] == source["partition"]
                    and row[name + "_sample_count"] == source["sample_count"]
                    and row[name + "_statistic_status"] == source["statistic_status"], "Wrong comparison source metadata")
            if source["pearson_r"] == "":
                require(row[name + "_pearson_r"] == "", "Undefined comparison statistic changed")
            else:
                require(float(row[name + "_pearson_r"]) == float(source["pearson_r"]),
                        "Comparison Pearson does not exactly equal saved source value")
            require(row[name + "_direction"] == direction(source), "Direction does not reproduce exact sign")
        vd, td = direction(v), direction(t)
        expected = "undefined" if "undefined" in (vd, td) else (
            "same_direction" if vd == td else "different_direction")
        require(row["direction_comparison"] == expected, "Comparison sign-only description mismatch")
    require(len(rows) == 20 and seen == set(validation) == set(test), "Missing comparison rows")


def verify_execution_code() -> None:
    # Static scope evidence complements recorded execution counters; no runner
    # import or execution is used by this verifier.
    frozen_attributes = {"ROOT", "INPUTS", "STAGE_FILES", "TARGETS", "CORRELATES", "OBJECTS",
                         "identity", "git", "require", "write_json", "pearson", "matplotlib"}
    for relative in CODE_FILES | {"scripts/run_acoustic_association_validation.py"}:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            imports = ([alias.name for alias in node.names] if isinstance(node, ast.Import) else
                       [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            require(not any(name.split(".")[0] in {"sklearn", "torch", "torchaudio", "librosa", "transformers", "scipy"}
                            or name.startswith("verify_acoustic_association_validation")
                            for name in imports), "Stage 2 code imports fitting/inference/audio or another runner")
            for name in imports:
                if name.startswith("run_acoustic_association"):
                    require(relative == "scripts/run_acoustic_association_test.py"
                            and name == "run_acoustic_association_validation" and isinstance(node, ast.Import)
                            and len(node.names) == 1 and node.names[0].asname == "frozen",
                            "Unexpected runner import")
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "frozen":
                require(node.attr in frozen_attributes, "Runner accesses unfrozen Stage 1 execution/helper")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                require(node.func.attr not in {"fit", "fit_transform", "predict", "beat_track", "tempo", "from_pretrained"},
                        "Stage 2 code contains fitting, prediction, or extraction calls")


def main() -> None:
    require(len(sys.argv) == 1, "Frozen Stage 2 verifier accepts no arguments")
    gate_path = ROOT / f"outputs/results/{PREFIX}_final_verification.json"
    gate = read_json(f"outputs/results/{PREFIX}_final_verification.json")
    gate["passed"] = False
    gate["independent_verification"] = {"passed": False, "status": "in_progress"}
    write_gate(gate_path, gate)  # A failing rerun cannot preserve a stale pass.

    inputs = {item["path"]: item for item in gate["inputs"].values()}
    require(len(inputs) == len(gate["inputs"]) and set(inputs) == EXPECTED_INPUTS, "Unexpected frozen input scope")
    require(set(gate["stage1_snapshot"]) == STAGE1_FILES, "Incomplete unchanged Stage 1 snapshot")
    require(set(gate["outputs"]) == OUTPUT_FILES and {item["path"] for item in gate["code"].values()} == CODE_FILES,
            "Unexpected Stage 2 output/code scope")
    for collection in (gate["inputs"], gate["stage1_snapshot"], gate["outputs"], gate["code"]):
        for item in collection.values():
            verify_identity(item)
    for path, item in {**gate["stage1_snapshot"], **gate["outputs"]}.items():
        require(path == item["path"], "Artifact dictionary key/path mismatch")
    require(not (set(git("diff", "HEAD", "--name-only").splitlines()) - STAGE2_FILES),
            "Existing tracked artifacts changed outside exact Stage 2 file set")
    require(git("rev-parse", "HEAD") == gate["git"]["head"]
            and git("branch", "--show-current") == gate["git"]["branch"], "Branch/HEAD changed since analysis")

    stage1 = read_json(f"outputs/results/{STAGE1_PREFIX}_pre_test_verification.json")
    require(stage1["passed"] is True and stage1["independent_verification"]["passed"] is True
            and all(value is True for value in stage1["checks"].values())
            and all(value is True for value in stage1["independent_verification"]["checks"].values()),
            "Stage 1 lacks passed implementation and independent verification")
    for collection in (stage1["inputs"], stage1["outputs"], stage1["code"]):
        for item in collection.values():
            verify_identity(item)
    require(stage1["pre_test_gate"]["status"] == "closed"
            and stage1["pre_test_gate"]["stage2_authorized"] is False
            and stage1["pre_test_gate"]["test_association_analysis_performed"] is False,
            "Historical Stage 1 closed gate changed")
    protocol = {"partition": "validation", "statistic": "Pearson r", "correlates": CORRELATES,
                "mert_representation": "transformer_layer_12", "supporting_comparator": "frozen_conventional_acoustic_51d",
                "targets": {"valence": "valence_mean", "arousal": "arousal_mean"}, "residual": "y_true - prediction",
                "positive_residual": "model under-predicts the target", "association_is_causality": False}
    require(stage1["protocol"] == protocol and gate["protocol"] == {**protocol, "partition": "test"},
            "Frozen Stage 1 design changed")
    authorization = {
        "stage1_researcher_chatgpt_review_passed": True, "stage2_authorized": True,
        "source": "researcher Stage 2 prompt", "client_date": "2026-10-01",
        "statement": "Module F Stage 1 review passed; formally authorized to open Test gate",
        "historical_stage1_gate": "closed; preserved unchanged", "module_f_test_gate": "opened_under_researcher_authorization"}
    require(gate["authorization"] == authorization, "Stage 2 authorization provenance changed")
    require(gate["prior_test_exposure"] == {"viewed_in_modules": ["C", "D", "E"],
        "untouched_confirmatory_evidence": False, "independent_replication": False,
        "purpose": "frozen descriptive consistency check"}, "Prior Test exposure lost")
    require(gate["execution"] == {
        "test_association_analysis_performed": True, "model_fits": 0, "audio_extractions": 0,
        "mert_inference_calls": 0, "alpha_or_layer_selections": 0, "train_association_analysis_performed": False},
        "Frozen execution scope changed")
    verify_execution_code()

    split = read_rows("data/metadata/deam_primary_split_seed42.csv")
    roles = {}
    for row in reversed(split):
        sample_id = int(row["sample_id"])
        require(sample_id not in roles, "Duplicate split ID")
        roles[sample_id] = row["split"]
    require(len(split) == len(roles) == 1744 and {
        role: list(roles.values()).count(role) for role in set(roles.values())
    } == {"train": 1221, "validation": 262, "test": 261}, "Frozen split counts changed")
    ids = {partition: sorted(i for i, role in roles.items() if role == partition) for partition in ("validation", "test")}
    require(ids["test"] == gate["alignment"]["test_sample_ids"] and len(ids["test"]) == 261
            and ids["validation"] == stage1["alignment"]["validation_sample_ids"]
            and len(ids["validation"]) == 262, "Frozen Test/Validation identities changed")
    require(gate["alignment"]["sample_count_per_target"] == 261 and gate["alignment"]["analysis_rows"] == 522
            and gate["alignment"]["association_rows"] == gate["alignment"]["comparison_rows"] == 20,
            "Stage 2 alignment counts changed")
    wanted = set(ids["validation"]) | set(ids["test"])
    labels, primary_ids, mapping_seen = {}, set(), set()
    for row in reversed(read_rows("data/raw/deam/verification/deam_item_mapping.csv")):
        sample_id = int(row["song_id"])
        require(sample_id not in mapping_seen and row["is_full_song"] in {"True", "False"}, "Invalid label mapping identity")
        mapping_seen.add(sample_id)
        if row["is_full_song"] == "False":
            primary_ids.add(sample_id)
        if sample_id in wanted:
            require(row["is_full_song"] == "False", "Full-song label in primary analysis")
            labels[sample_id] = {target: float(row[target + "_mean"]) for target in TARGETS}
    require(primary_ids == set(roles) and set(labels) == wanted, "Frozen label population mismatch")

    c3, e2 = read_json("outputs/results/module_c_stage3_validation.json"), read_json("outputs/results/module_e_stage2_validation.json")
    c4, e3 = read_json("outputs/results/module_c_stage4_test.json"), read_json("outputs/results/module_e_stage3_test.json")
    for name in ("fixed_split", "labels"):
        require(inputs[stage1["inputs"][name]["path"]] == stage1["inputs"][name]
                == c3["inputs"][name] == e2["inputs"][name]
                == c4["provenance"]["inputs"][name] == e3["provenance"]["inputs"][name],
                "Frozen split/label provenance mismatch")
    require(stage1["inputs"]["acoustic_cache"] == e2["inputs"]["acoustic_cache"]
            == e3["provenance"]["inputs"]["acoustic_cache"], "Frozen acoustic cache provenance mismatch")
    for item in (c4["provenance"]["stage_c3_results"], c4["provenance"]["stage_c3_validation_predictions"],
                 e3["provenance"]["stage2_results"], e3["provenance"]["stage2_validation_predictions"],
                 e3["provenance"]["module_c_authoritative_results"], e3["provenance"]["module_c_authoritative_predictions"]):
        verify_identity(item)
    cf, ef = c4["frozen_configuration"], e3["frozen_configuration"]
    require(cf["representation_level"] == "transformer_layer_12" and cf["resolved_representation_index"] == 12
            and cf["alphas"] == {"valence": 1000, "arousal": 1000}
            and ef["representation"] == "conventional_acoustic_51d" and ef["input_dimension"] == 51
            and ef["alphas"] == {"valence": 100, "arousal": 10}, "Frozen representation/alpha provenance mismatch")
    for result in (c4, e3):
        frozen = result["frozen_configuration"]
        require(frozen["target_standardization"] is False and frozen["final_refit_training_split"] == "train_only"
                and frozen["validation_in_final_refit"] is False, "Frozen fitting/scale provenance mismatch")
        for target in TARGETS:
            require(result["targets"][target]["frozen_alpha"] == frozen["alphas"][target], "Target frozen alpha mismatch")

    with np.load(ROOT / "data/processed/deam_acoustic_51d.npz", allow_pickle=False) as cache:
        require(set(cache.files) == {"sample_ids", "features", "feature_names"}, "Saved acoustic cache schema changed")
        cache_ids, features, names = cache["sample_ids"], cache["features"], cache["feature_names"].tolist()
    diagnostics = read_json("outputs/results/module_e_stage1_acoustic_diagnostics.json")
    require(diagnostics["passed"] is True and names == diagnostics["feature_names"]
            == e2["data_assembly"]["feature_names"] == ef["feature_names"] and len(set(names)) == 51,
            "Frozen ordered feature names changed")
    require(cache_ids.shape == (1744,) and cache_ids.dtype == np.int64 and len(set(cache_ids)) == 1744
            and set(cache_ids) == set(roles) and features.shape == (1744, 51) and features.dtype == np.float32,
            "Frozen acoustic cache population/dtype changed")
    columns = {name: names.index(name) for name in CORRELATES.values()}
    require(columns == gate["alignment"]["feature_indices_zero_based"]
            == stage1["alignment"]["feature_indices_zero_based"], "Wrong named acoustic feature resolution")
    cache_by_id = {int(cache_ids[j]): {name: float(features[j, col]) for name, col in columns.items()}
                   for j in reversed(range(len(cache_ids))) if int(cache_ids[j]) in wanted}
    associations, differences, undefined = {}, {}, {}
    for partition, primary, supporting in (("validation", c3, e2), ("test", c4, e3)):
        predictions = load_predictions(partition, ids[partition], primary, supporting)
        data = load_analysis(partition, ids[partition], labels, cache_by_id, predictions)
        associations[partition], differences[partition], undefined[partition] = verify_associations(partition, ids[partition], data)
    for partition, saved in (("validation", stage1), ("test", gate)):
        normalized = sorted(undefined[partition], key=lambda row: (row["target"], row["analysis_object"], row["correlate"]))
        recorded = sorted(saved["undefined_associations"], key=lambda row: (row["target"], row["analysis_object"], row["correlate"]))
        require(normalized == recorded, "Undefined-association JSON record mismatch")
    verify_comparison(associations["validation"], associations["test"])
    require(roles[437] == "train" and 437 not in wanted
            and stage1["known_measurement_uncertainty"]["estimate_retained"] is True,
            "Known uncertain Tempo sample treatment changed")
    # Repeat identities after independent recomputation; protect untracked Stage
    # 1 files as well as authoritative cached/source artifacts.
    for collection in (gate["inputs"], gate["stage1_snapshot"], gate["outputs"], gate["code"]):
        for item in collection.values():
            verify_identity(item)
    require(bool(gate["checks"]) and all(value is True for value in gate["checks"].values()), "Runner check failed")
    checks = {name: True for name in (
        "input_output_code_hashes_and_sizes", "stage1_all_eight_files_unchanged",
        "existing_tracked_files_outside_stage2_unchanged", "branch_and_head_unchanged",
        "stage1_passed_and_historical_gate_unchanged", "stage2_prompt_authorization",
        "frozen_stage1_design_reused_exactly", "prior_test_exposure_preserved",
        "frozen_split_identity_and_counts", "exact_261_test_ids_and_262_validation_ids",
        "composite_prediction_keys_and_partition_membership", "reversed_source_order_independent_alignment",
        "named_features_and_exact_cache_values", "original_scale_labels_and_prediction_y_true_exact",
        "saved_predictions_exactly_match_c_and_e", "residual_formula_exact",
        "all_analyzed_values_finite", "reused_c4_e3_prediction_and_fit_provenance",
        "authoritative_c_and_e_artifacts_unchanged", "all_20_test_pearsons_independently_recomputed",
        "all_20_stage1_pearsons_independently_recomputed", "frozen_undefined_constant_input_policy",
        "comparison_schema_and_source_statistics_exact", "comparison_directions_use_exact_sign_only",
        "known_uncertain_tempo_retained", "execution_scope_and_static_code_audit",
    )}
    gate["independent_verification"] = {
        "passed": True, "status": "passed", "checks": checks,
        "method": "stdlib csv keyed dictionaries (reversed source order), exact float residual subtraction, math.fsum centered Pearson",
        "pearson_absolute_tolerance": PEARSON_TOLERANCE,
        "maximum_absolute_difference": max(differences["test"], default=0),
        "stage1_maximum_absolute_difference": max(differences["validation"], default=0),
        "association_count": 20, "stage1_association_count": 20,
        "undefined_count": len(undefined["test"]), "stage1_undefined_count": len(undefined["validation"]),
        "comparison_rows": 20,
        "direction_labels_meaning": "Exact sign only; no magnitude cutoff, inference, replication, or binary statistical consistency claim",
    }
    gate["passed"] = all(gate["checks"].values()) and all(checks.values())
    write_gate(gate_path, gate)
    print(json.dumps({"passed": gate["passed"], "independent_checks": len(checks),
        "maximum_test_pearson_difference": max(differences["test"], default=0),
        "maximum_stage1_pearson_difference": max(differences["validation"], default=0),
        "test_associations": 20, "validation_associations": 20, "comparison_rows": 20}, indent=2))


if __name__ == "__main__":
    main()
