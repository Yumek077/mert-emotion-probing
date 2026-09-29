"""Validated data assembly and Train/Validation probing for Module C."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from .mert import EXPECTED_HIDDEN_DIMENSION, EXPECTED_REPRESENTATIONS
from .splits import ALLOWED_SPLITS, PRIMARY_SAMPLE_COUNT


EXPECTED_LEVEL_NAMES = ("pre_transformer",) + tuple(
    f"transformer_layer_{index}" for index in range(1, 13)
)
PRIMARY_LEVEL_NAME = "transformer_layer_12"
EXPECTED_SPLIT_COUNTS = {"train": 1_221, "validation": 262, "test": 261}
ALPHA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
TARGET_COLUMNS = {"valence": "valence_mean", "arousal": "arousal_mean"}


class ProtocolViolation(RuntimeError):
    """Raised when repository data conflicts with the frozen research protocol."""


@dataclass(frozen=True)
class AssembledProbingDataset:
    """ID-aligned Layer-12 features, targets, split roles, and verification."""

    sample_ids: np.ndarray
    features: np.ndarray
    targets: dict[str, np.ndarray]
    split: np.ndarray
    representation_level: str
    representation_index: int
    verification: dict[str, Any]


@dataclass(frozen=True)
class TrainValidationData:
    """Only the arrays permitted to enter Stage C3 fitting and evaluation."""

    train_ids: np.ndarray
    validation_ids: np.ndarray
    x_train: np.ndarray
    x_validation: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray


@dataclass(frozen=True)
class TrainTestData:
    """Only the arrays permitted to enter the frozen Stage C4 evaluation."""

    train_ids: np.ndarray
    test_ids: np.ndarray
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def _integer_ids(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ProtocolViolation(f"{name} must contain finite integer identities")
    return numeric.astype(np.int64)


def _boolean_series(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="raise")
        if not set(numeric.unique()).issubset({0, 1}):
            raise ProtocolViolation(f"{name} contains values other than 0/1")
        return numeric.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ProtocolViolation(f"{name} contains unrecognized values: {unknown}")
    return normalized.map(mapping).astype(bool)


def _load_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    if not cache_path.is_file():
        raise FileNotFoundError(f"Canonical cache not found: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ProtocolViolation("Canonical cache payload must be a dictionary")

    representations = payload.get("representations")
    sample_ids = payload.get("sample_ids")
    metadata = payload.get("metadata")
    if not isinstance(representations, torch.Tensor):
        raise ProtocolViolation("Cache representations must be a tensor")
    if not isinstance(sample_ids, torch.Tensor):
        raise ProtocolViolation("Cache sample_ids must be a tensor")
    if not isinstance(metadata, dict):
        raise ProtocolViolation("Cache metadata must be a dictionary")

    expected_shape = (
        PRIMARY_SAMPLE_COUNT,
        EXPECTED_REPRESENTATIONS,
        EXPECTED_HIDDEN_DIMENSION,
    )
    if tuple(representations.shape) != expected_shape:
        raise ProtocolViolation(
            f"Cache shape {tuple(representations.shape)} does not match {expected_shape}"
        )
    if representations.dtype != torch.float32:
        raise ProtocolViolation(
            f"Cache representation dtype must be float32, got {representations.dtype}"
        )
    if tuple(sample_ids.shape) != (PRIMARY_SAMPLE_COUNT,):
        raise ProtocolViolation(f"Cache sample_ids shape is {tuple(sample_ids.shape)}")
    if sample_ids.dtype != torch.int64:
        raise ProtocolViolation(f"Cache sample_ids dtype must be int64, got {sample_ids.dtype}")
    if not bool(torch.isfinite(representations).all().item()):
        raise ProtocolViolation("Cache contains NaN or Inf")

    ids = sample_ids.numpy().astype(np.int64, copy=True)
    if len(np.unique(ids)) != PRIMARY_SAMPLE_COUNT:
        raise ProtocolViolation("Cache sample IDs are not unique")

    levels = metadata.get("representation_levels")
    if not isinstance(levels, list) or tuple(levels) != EXPECTED_LEVEL_NAMES:
        raise ProtocolViolation(
            "Cache representation_levels conflict with the frozen Module B mapping"
        )
    resolved_index = levels.index(PRIMARY_LEVEL_NAME)
    if resolved_index != 12:
        raise ProtocolViolation(
            f"{PRIMARY_LEVEL_NAME} resolved to index {resolved_index}, expected 12"
        )

    selected = representations[:, resolved_index, :].numpy().copy()
    if selected.shape != (PRIMARY_SAMPLE_COUNT, EXPECTED_HIDDEN_DIMENSION):
        raise ProtocolViolation(f"Selected Layer-12 shape is {selected.shape}")
    if not np.isfinite(selected).all():
        raise ProtocolViolation("Selected Layer-12 features contain NaN or Inf")
    all_zero_rows = np.flatnonzero(np.count_nonzero(selected, axis=1) == 0)
    if len(all_zero_rows):
        raise ProtocolViolation(
            f"Selected Layer-12 features contain all-zero rows: {all_zero_rows.tolist()}"
        )

    verification = {
        "cache_shape": list(representations.shape),
        "cache_dtype": str(representations.dtype),
        "cache_sample_id_count": int(len(ids)),
        "cache_sample_ids_unique": int(len(np.unique(ids))),
        "metadata_identity_key": metadata.get("identity_key"),
        "metadata_ordering": metadata.get("ordering"),
        "metadata_representation_levels": levels,
        "resolved_representation_level": PRIMARY_LEVEL_NAME,
        "resolved_representation_index": resolved_index,
        "selected_feature_shape": list(selected.shape),
        "selected_features_finite": True,
        "selected_all_zero_row_count": 0,
    }
    return ids, selected, resolved_index, verification


def _load_labels(labels_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not labels_path.is_file():
        raise FileNotFoundError(f"DEAM mapping table not found: {labels_path}")
    labels = pd.read_csv(labels_path)
    required = {"song_id", "is_full_song", *TARGET_COLUMNS.values()}
    missing = required - set(labels.columns)
    if missing:
        raise ProtocolViolation(f"Label table is missing columns: {sorted(missing)}")
    labels = labels.copy()
    labels["song_id"] = _integer_ids(labels["song_id"], "labels.song_id")
    if labels["song_id"].duplicated().any():
        duplicates = sorted(
            labels.loc[labels["song_id"].duplicated(keep=False), "song_id"]
            .unique()
            .tolist()
        )
        raise ProtocolViolation(f"Label table contains duplicate IDs: {duplicates}")
    labels["is_full_song"] = _boolean_series(
        labels["is_full_song"], "labels.is_full_song"
    )
    for column in TARGET_COLUMNS.values():
        labels[column] = pd.to_numeric(labels[column], errors="raise")

    primary = labels.loc[~labels["is_full_song"]].copy()
    if len(primary) != PRIMARY_SAMPLE_COUNT:
        raise ProtocolViolation(
            f"Expected {PRIMARY_SAMPLE_COUNT} primary label rows, got {len(primary)}"
        )
    if primary[list(TARGET_COLUMNS.values())].isna().any().any():
        raise ProtocolViolation("Primary labels contain missing targets")
    target_values = primary[list(TARGET_COLUMNS.values())].to_numpy(dtype=np.float64)
    if not np.isfinite(target_values).all():
        raise ProtocolViolation("Primary labels contain non-finite targets")

    verification = {
        "label_table_rows": int(len(labels)),
        "label_table_unique_ids": int(labels["song_id"].nunique()),
        "primary_label_rows": int(len(primary)),
        "excluded_full_song_rows": int(labels["is_full_song"].sum()),
        "target_columns": list(TARGET_COLUMNS.values()),
        "primary_targets_complete_and_finite": True,
    }
    return primary, verification


def _load_split(split_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not split_path.is_file():
        raise FileNotFoundError(f"Frozen split not found: {split_path}")
    split = pd.read_csv(split_path)
    if list(split.columns) != ["sample_id", "split"]:
        raise ProtocolViolation("Split must contain exactly sample_id,split")
    split = split.copy()
    split["sample_id"] = _integer_ids(split["sample_id"], "split.sample_id")
    if len(split) != PRIMARY_SAMPLE_COUNT:
        raise ProtocolViolation(f"Split contains {len(split)} rows")
    if split["sample_id"].duplicated().any():
        raise ProtocolViolation("Split contains duplicate sample IDs")
    invalid = sorted(set(split["split"]) - set(ALLOWED_SPLITS))
    if invalid:
        raise ProtocolViolation(f"Split contains invalid roles: {invalid}")
    counts = {name: int((split["split"] == name).sum()) for name in ALLOWED_SPLITS}
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ProtocolViolation(
            f"Split counts {counts} do not match {EXPECTED_SPLIT_COUNTS}"
        )
    role_sets = {
        name: set(split.loc[split["split"] == name, "sample_id"].tolist())
        for name in ALLOWED_SPLITS
    }
    overlap = (
        (role_sets["train"] & role_sets["validation"])
        | (role_sets["train"] & role_sets["test"])
        | (role_sets["validation"] & role_sets["test"])
    )
    if overlap:
        raise ProtocolViolation(f"Split roles overlap for IDs: {sorted(overlap)}")

    verification = {
        "split_rows": int(len(split)),
        "split_unique_ids": int(split["sample_id"].nunique()),
        "split_counts": counts,
        "allowed_split_values": list(ALLOWED_SPLITS),
        "invalid_split_values": [],
        "cross_split_overlap_count": 0,
    }
    return split, verification


def assemble_probing_dataset(
    cache_path: Path, labels_path: Path, split_path: Path
) -> AssembledProbingDataset:
    """Load and join frozen artifacts strictly by authoritative sample identity."""

    cache_ids, selected_features, level_index, cache_checks = _load_cache(cache_path)
    primary_labels, label_checks = _load_labels(labels_path)
    split_table, split_checks = _load_split(split_path)

    cache_id_set = set(cache_ids.tolist())
    label_id_set = set(primary_labels["song_id"].tolist())
    split_id_set = set(split_table["sample_id"].tolist())
    if cache_id_set != label_id_set or cache_id_set != split_id_set:
        raise ProtocolViolation(
            "Cache, primary-label, and split ID sets do not match exactly"
        )

    assembly_ids = np.asarray(sorted(cache_id_set), dtype=np.int64)
    cache_index = {int(sample_id): index for index, sample_id in enumerate(cache_ids)}
    feature_indices = [cache_index[int(sample_id)] for sample_id in assembly_ids]
    features = selected_features[feature_indices].astype(np.float64, copy=False)

    labels_by_id = primary_labels.set_index("song_id", verify_integrity=True)
    split_by_id = split_table.set_index("sample_id", verify_integrity=True)
    ordered_labels = labels_by_id.loc[assembly_ids]
    ordered_split = split_by_id.loc[assembly_ids, "split"]

    targets = {
        name: ordered_labels[column].to_numpy(dtype=np.float64, copy=True)
        for name, column in TARGET_COLUMNS.items()
    }
    split_values = ordered_split.to_numpy(dtype=str, copy=True)

    reversed_labels = primary_labels.iloc[::-1].set_index(
        "song_id", verify_integrity=True
    )
    reversed_split = split_table.iloc[::-1].set_index(
        "sample_id", verify_integrity=True
    )
    row_order_independent = all(
        np.array_equal(
            targets[name],
            reversed_labels.loc[assembly_ids, column].to_numpy(dtype=np.float64),
        )
        for name, column in TARGET_COLUMNS.items()
    ) and np.array_equal(
        split_values,
        reversed_split.loc[assembly_ids, "split"].to_numpy(dtype=str),
    )
    if not row_order_independent:
        raise ProtocolViolation("ID-based assembly changed when source rows were reversed")

    if features.shape != (PRIMARY_SAMPLE_COUNT, EXPECTED_HIDDEN_DIMENSION):
        raise ProtocolViolation(f"Assembled features have shape {features.shape}")
    if not np.isfinite(features).all():
        raise ProtocolViolation("Assembled features contain NaN or Inf")
    if any(not np.isfinite(values).all() for values in targets.values()):
        raise ProtocolViolation("Assembled targets contain NaN or Inf")

    joined_counts = {
        name: int(np.count_nonzero(split_values == name)) for name in ALLOWED_SPLITS
    }
    verification = {
        **cache_checks,
        **label_checks,
        **split_checks,
        "cache_label_split_id_sets_equal": True,
        "assembled_sample_count": int(len(assembly_ids)),
        "assembled_sample_ids_unique": int(len(np.unique(assembly_ids))),
        "assembled_feature_shape": list(features.shape),
        "assembled_targets_finite": True,
        "assembled_features_finite": True,
        "joined_split_counts": joined_counts,
        "id_based_join_row_order_invariance_verified": True,
        "excluded_full_songs_in_join": 0,
    }
    return AssembledProbingDataset(
        sample_ids=assembly_ids,
        features=features,
        targets=targets,
        split=split_values,
        representation_level=PRIMARY_LEVEL_NAME,
        representation_index=level_index,
        verification=verification,
    )


def train_validation_view(
    dataset: AssembledProbingDataset, target: str
) -> TrainValidationData:
    """Return only Train/Validation arrays; Test arrays never enter model selection."""

    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unknown target: {target}")
    train_mask = dataset.split == "train"
    validation_mask = dataset.split == "validation"
    return TrainValidationData(
        train_ids=dataset.sample_ids[train_mask].copy(),
        validation_ids=dataset.sample_ids[validation_mask].copy(),
        x_train=dataset.features[train_mask].copy(),
        x_validation=dataset.features[validation_mask].copy(),
        y_train=dataset.targets[target][train_mask].copy(),
        y_validation=dataset.targets[target][validation_mask].copy(),
    )


def train_test_view(dataset: AssembledProbingDataset, target: str) -> TrainTestData:
    """Return Train/Test arrays while excluding Validation from the final refit."""

    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unknown target: {target}")
    train_mask = dataset.split == "train"
    test_mask = dataset.split == "test"
    return TrainTestData(
        train_ids=dataset.sample_ids[train_mask].copy(),
        test_ids=dataset.sample_ids[test_mask].copy(),
        x_train=dataset.features[train_mask].copy(),
        x_test=dataset.features[test_mask].copy(),
        y_train=dataset.targets[target][train_mask].copy(),
        y_test=dataset.targets[target][test_mask].copy(),
    )


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Compute protocol metrics while representing undefined Pearson r explicitly."""

    y_true = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if y_true.ndim != 1 or prediction.shape != y_true.shape:
        raise ValueError("Metric inputs must be same-shaped one-dimensional arrays")
    if not np.isfinite(y_true).all() or not np.isfinite(prediction).all():
        raise ValueError("Metric inputs must be finite")

    pearson_value: float | None
    pearson_status: str
    if np.all(y_true == y_true[0]):
        pearson_value = None
        pearson_status = "undefined_constant_target"
    elif np.all(prediction == prediction[0]):
        pearson_value = None
        pearson_status = "undefined_constant_prediction"
    else:
        value = float(pearsonr(y_true, prediction).statistic)
        if np.isfinite(value):
            pearson_value = value
            pearson_status = "defined"
        else:
            pearson_value = None
            pearson_status = "undefined_nonfinite_result"

    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "r2": float(r2_score(y_true, prediction)),
        "pearson_r": pearson_value,
        "pearson_r_status": pearson_status,
    }


def select_candidate_by_validation_r2(
    candidate_rows: list[dict[str, float]],
) -> tuple[dict[str, float], float, list[dict[str, float]]]:
    """Select maximum Validation R-squared; exact ties prefer larger alpha."""

    if not candidate_rows:
        raise ValueError("At least one candidate result is required")
    alphas = [float(row["alpha"]) for row in candidate_rows]
    scores = [float(row["validation_r2"]) for row in candidate_rows]
    if len(set(alphas)) != len(alphas):
        raise ValueError("Candidate alphas must be unique")
    if not np.isfinite(alphas).all() or not np.isfinite(scores).all():
        raise ValueError("Candidate alphas and Validation R-squared values must be finite")
    best_r2 = max(scores)
    exact_ties = [
        row for row in candidate_rows if float(row["validation_r2"]) == best_r2
    ]
    selected = max(exact_ties, key=lambda row: float(row["alpha"]))
    return selected, best_r2, exact_ties


def fit_select_validate(
    data: TrainValidationData,
    *,
    alpha_grid: tuple[float, ...] = ALPHA_GRID,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Fit Train-only Ridge candidates and select by Validation R-squared."""

    if tuple(alpha_grid) != ALPHA_GRID:
        raise ProtocolViolation("Alpha grid differs from the frozen Stage C2 grid")
    expected_shapes = {
        "x_train": (EXPECTED_SPLIT_COUNTS["train"], EXPECTED_HIDDEN_DIMENSION),
        "x_validation": (
            EXPECTED_SPLIT_COUNTS["validation"],
            EXPECTED_HIDDEN_DIMENSION,
        ),
        "y_train": (EXPECTED_SPLIT_COUNTS["train"],),
        "y_validation": (EXPECTED_SPLIT_COUNTS["validation"],),
    }
    observed = {
        "x_train": data.x_train.shape,
        "x_validation": data.x_validation.shape,
        "y_train": data.y_train.shape,
        "y_validation": data.y_validation.shape,
    }
    if observed != expected_shapes:
        raise ProtocolViolation(f"Train/Validation shapes {observed} != {expected_shapes}")
    if any(
        not np.isfinite(array).all()
        for array in (data.x_train, data.x_validation, data.y_train, data.y_validation)
    ):
        raise ProtocolViolation("Train/Validation arrays must be finite")
    if np.all(data.y_train == data.y_train[0]) or np.all(
        data.y_validation == data.y_validation[0]
    ):
        raise ProtocolViolation("Train and Validation targets must be non-constant")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(data.x_train)
    x_validation_scaled = scaler.transform(data.x_validation)
    scaler_seen = int(np.asarray(scaler.n_samples_seen_).max())
    if scaler_seen != EXPECTED_SPLIT_COUNTS["train"]:
        raise RuntimeError(f"Scaler saw {scaler_seen} samples instead of Train only")
    if not np.allclose(scaler.mean_, np.mean(data.x_train, axis=0), rtol=0, atol=1e-12):
        raise RuntimeError("Scaler mean does not match the Train-only feature mean")
    if not np.allclose(scaler.var_, np.var(data.x_train, axis=0), rtol=0, atol=1e-12):
        raise RuntimeError("Scaler variance does not match the Train-only feature variance")

    candidate_rows: list[dict[str, float]] = []
    predictions_by_alpha: dict[float, np.ndarray] = {}
    for alpha in alpha_grid:
        model = Ridge(alpha=float(alpha), solver="cholesky")
        model.fit(x_train_scaled, data.y_train)
        prediction = model.predict(x_validation_scaled).astype(np.float64, copy=False)
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Non-finite Validation prediction for alpha={alpha}")
        score = float(r2_score(data.y_validation, prediction))
        candidate_rows.append({"alpha": float(alpha), "validation_r2": score})
        predictions_by_alpha[float(alpha)] = prediction.copy()

    selected_row, best_r2, exact_ties = select_candidate_by_validation_r2(
        candidate_rows
    )
    selected_alpha = float(selected_row["alpha"])
    selected_prediction = predictions_by_alpha[selected_alpha]
    selected_metrics = regression_metrics(data.y_validation, selected_prediction)

    train_target_mean = float(np.mean(data.y_train))
    baseline_prediction = np.full_like(
        data.y_validation, fill_value=train_target_mean, dtype=np.float64
    )
    baseline_metrics = regression_metrics(data.y_validation, baseline_prediction)
    if baseline_metrics["pearson_r"] is not None:
        raise RuntimeError("Constant mean baseline unexpectedly has defined Pearson r")

    result = {
        "alpha_grid": [float(alpha) for alpha in alpha_grid],
        "selection_metric": "validation_r2",
        "exact_tie_rule": "choose_larger_alpha",
        "candidate_validation_r2": candidate_rows,
        "best_validation_r2": best_r2,
        "exact_tie_count": len(exact_ties),
        "exact_tied_alphas": [float(row["alpha"]) for row in exact_ties],
        "selected_alpha": selected_alpha,
        "selected_validation_metrics": selected_metrics,
        "baseline": {
            "definition": "constant_prediction_from_train_target_mean",
            "train_target_mean": train_target_mean,
            "validation_metrics": baseline_metrics,
            "pearson_r_explanation": (
                "Undefined because the mean baseline has zero prediction variance; "
                "serialized as null rather than 0."
            ),
        },
        "standardization": {
            "input_scaler": "sklearn.preprocessing.StandardScaler",
            "fit_split": "train",
            "fit_sample_count": scaler_seen,
            "transformed_splits": ["train", "validation"],
            "target_standardized": False,
            "train_mean_verified": True,
            "train_variance_verified": True,
        },
        "ridge": {
            "implementation": "sklearn.linear_model.Ridge",
            "solver": "cholesky",
            "fit_intercept": True,
            "fit_split": "train",
            "fit_sample_count_per_candidate": int(len(data.train_ids)),
        },
    }
    return result, selected_prediction.copy(), baseline_prediction


def fit_train_evaluate_test(
    data: TrainTestData,
    *,
    alpha: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Fit the frozen Train-only Ridge configuration and evaluate held-out Test."""

    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Frozen Ridge alpha must be finite and positive")
    expected_shapes = {
        "x_train": (EXPECTED_SPLIT_COUNTS["train"], EXPECTED_HIDDEN_DIMENSION),
        "x_test": (EXPECTED_SPLIT_COUNTS["test"], EXPECTED_HIDDEN_DIMENSION),
        "y_train": (EXPECTED_SPLIT_COUNTS["train"],),
        "y_test": (EXPECTED_SPLIT_COUNTS["test"],),
    }
    observed = {
        "x_train": data.x_train.shape,
        "x_test": data.x_test.shape,
        "y_train": data.y_train.shape,
        "y_test": data.y_test.shape,
    }
    if observed != expected_shapes:
        raise ProtocolViolation(f"Train/Test shapes {observed} != {expected_shapes}")
    if any(
        not np.isfinite(array).all()
        for array in (data.x_train, data.x_test, data.y_train, data.y_test)
    ):
        raise ProtocolViolation("Train/Test arrays must be finite")
    if np.all(data.y_train == data.y_train[0]) or np.all(
        data.y_test == data.y_test[0]
    ):
        raise ProtocolViolation("Train and Test targets must be non-constant")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(data.x_train)
    x_test_scaled = scaler.transform(data.x_test)
    scaler_seen = int(np.asarray(scaler.n_samples_seen_).max())
    if scaler_seen != EXPECTED_SPLIT_COUNTS["train"]:
        raise RuntimeError(f"Scaler saw {scaler_seen} samples instead of Train only")
    if not np.allclose(scaler.mean_, np.mean(data.x_train, axis=0), rtol=0, atol=1e-12):
        raise RuntimeError("Scaler mean does not match the Train-only feature mean")
    if not np.allclose(scaler.var_, np.var(data.x_train, axis=0), rtol=0, atol=1e-12):
        raise RuntimeError("Scaler variance does not match the Train-only feature variance")

    model = Ridge(alpha=float(alpha), solver="cholesky")
    model.fit(x_train_scaled, data.y_train)
    prediction = model.predict(x_test_scaled).astype(np.float64, copy=False)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Frozen Ridge produced non-finite Test predictions")
    metrics = regression_metrics(data.y_test, prediction)

    train_target_mean = float(np.mean(data.y_train))
    baseline_prediction = np.full_like(
        data.y_test, fill_value=train_target_mean, dtype=np.float64
    )
    baseline_metrics = regression_metrics(data.y_test, baseline_prediction)
    if baseline_metrics["pearson_r"] is not None:
        raise RuntimeError("Constant mean baseline unexpectedly has defined Pearson r")

    result = {
        "frozen_alpha": float(alpha),
        "test_metrics": metrics,
        "baseline": {
            "definition": "constant_prediction_from_train_target_mean",
            "train_target_mean": train_target_mean,
            "test_metrics": baseline_metrics,
            "pearson_r_explanation": (
                "Undefined because the mean baseline has zero prediction variance; "
                "serialized as null rather than 0."
            ),
        },
        "standardization": {
            "input_scaler": "sklearn.preprocessing.StandardScaler",
            "fit_split": "train",
            "fit_sample_count": scaler_seen,
            "transformed_splits": ["train", "test"],
            "validation_in_final_refit": False,
            "target_standardized": False,
            "train_mean_verified": True,
            "train_variance_verified": True,
        },
        "ridge": {
            "implementation": "sklearn.linear_model.Ridge",
            "solver": "cholesky",
            "fit_intercept": True,
            "fit_split": "train",
            "fit_sample_count": int(len(data.train_ids)),
        },
    }
    return result, prediction.copy(), baseline_prediction


def verify_deterministic_repeat(
    first_result: dict[str, Any],
    first_prediction: np.ndarray,
    repeated_result: dict[str, Any],
    repeated_prediction: np.ndarray,
) -> dict[str, Any]:
    """Require an exact repeat for the deterministic Train/Validation procedure."""

    selected_alpha_equal = (
        first_result["selected_alpha"] == repeated_result["selected_alpha"]
    )
    candidate_results_equal = (
        first_result["candidate_validation_r2"]
        == repeated_result["candidate_validation_r2"]
    )
    selected_metrics_equal = (
        first_result["selected_validation_metrics"]
        == repeated_result["selected_validation_metrics"]
    )
    predictions_exactly_equal = bool(
        np.array_equal(first_prediction, repeated_prediction)
    )
    passed = all(
        [
            selected_alpha_equal,
            candidate_results_equal,
            selected_metrics_equal,
            predictions_exactly_equal,
        ]
    )
    if not passed:
        raise RuntimeError("Repeated deterministic probing run did not match exactly")
    return {
        "passed": True,
        "selected_alpha_equal": selected_alpha_equal,
        "candidate_results_equal": candidate_results_equal,
        "selected_metrics_equal": selected_metrics_equal,
        "predictions_exactly_equal": predictions_exactly_equal,
    }
