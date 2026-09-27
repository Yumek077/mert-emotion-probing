"""Verify a locally extracted official DEAM dataset without running MERT.

The script inventories every MP3, combines the two official static averaged
annotation tables, checks audio-label integrity, maps IDs to the three official
metadata files, and writes descriptive summaries and inspection figures.

All default input and tabular output paths are repository-relative. Temporary
CSV/JSON outputs are written under ``data/raw`` so they remain Git-ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/deam"),
        help="Root containing extracted audio, annotations, and metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/deam/verification"),
        help="Git-ignored directory for temporary CSV and JSON outputs.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("docs/codex_reports/figures"),
        help="Directory for temporary research-inspection figures.",
    )
    parser.add_argument("--excerpt-min", type=float, default=44.0)
    parser.add_argument("--excerpt-max", type=float, default=46.0)
    parser.add_argument("--long-threshold", type=float, default=60.0)
    return parser.parse_args()


def numeric_stem(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def inventory_audio(audio_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(audio_root.rglob("*.mp3"), key=lambda p: (numeric_stem(p) is None, numeric_stem(p) or 0)):
        row: dict[str, Any] = {
            "song_id": numeric_stem(path),
            "filename": path.name,
            "relative_path": path.relative_to(audio_root).as_posix(),
            "readable": False,
            "duration_seconds": np.nan,
            "sample_rate": np.nan,
            "channels": np.nan,
            "format": None,
            "subtype": None,
            "error": None,
        }
        try:
            info = sf.info(path)
            row.update(
                {
                    "readable": True,
                    "duration_seconds": float(info.duration),
                    "sample_rate": int(info.samplerate),
                    "channels": int(info.channels),
                    "format": info.format,
                    "subtype": info.subtype,
                }
            )
        except Exception as exc:  # Keep all failures in the integrity report.
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def load_static_annotations(annotation_root: Path) -> tuple[pd.DataFrame, list[str]]:
    paths = sorted(annotation_root.rglob("static_annotations_averaged_songs_*.csv"))
    if not paths:
        raise FileNotFoundError("No official static averaged annotation CSVs found")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame.columns = [str(column).strip() for column in frame.columns]
        frame["source_file"] = path.name
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    required = ["song_id", "valence_mean", "arousal_mean"]
    missing = [column for column in required if column not in combined.columns]
    if missing:
        raise ValueError(f"Missing required static columns: {missing}")

    combined["song_id"] = pd.to_numeric(combined["song_id"], errors="coerce").astype("Int64")
    for column in combined.columns:
        if column not in {"source_file", "song_id"}:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")
    return combined.sort_values("song_id").reset_index(drop=True), [str(path) for path in paths]


def read_first_column_ids(path: Path) -> list[int]:
    ids: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            try:
                ids.append(int(row[0].strip()))
            except ValueError:
                continue
    return ids


def load_metadata_mapping(metadata_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    files = sorted(metadata_root.rglob("metadata_*.csv"))
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for path in files:
        year = path.stem.removeprefix("metadata_")
        ids = read_first_column_ids(path)
        counts[year] = len(ids)
        rows.extend(
            {"song_id": song_id, "year_or_subset": year, "metadata_file": path.name}
            for song_id in ids
        )

    mapping = pd.DataFrame(rows)
    duplicate_ids = (
        mapping.loc[mapping.duplicated("song_id", keep=False), "song_id"]
        .drop_duplicates()
        .astype(int)
        .tolist()
        if not mapping.empty
        else []
    )
    return mapping, {
        "files": [str(path) for path in files],
        "counts_by_file_year": counts,
        "duplicate_ids_across_metadata_files": duplicate_ids,
    }


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def finite_float(value: Any) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def describe_series(series: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(series, errors="coerce")
    return {
        "count": int(numeric.count()),
        "missing": int(numeric.isna().sum()),
        "min": finite_float(numeric.min()),
        "max": finite_float(numeric.max()),
        "mean": finite_float(numeric.mean()),
        "std": finite_float(numeric.std(ddof=1)),
        "median": finite_float(numeric.median()),
    }


def make_figures(labels: pd.DataFrame, figures_dir: Path) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    group_colors = {False: "#3366cc", True: "#dc3912"}
    group_names = {False: "45-second excerpts", True: "2015 full songs"}

    for target, title in (("valence_mean", "Static Valence"), ("arousal_mean", "Static Arousal")):
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        bins = np.linspace(1.0, 9.0, 17)
        for is_full_song in (False, True):
            values = labels.loc[labels["is_full_song"] == is_full_song, target].dropna()
            ax.hist(
                values,
                bins=bins,
                alpha=0.65,
                label=f"{group_names[is_full_song]} (n={len(values)})",
                color=group_colors[is_full_song],
            )
        ax.set_title(f"DEAM {title} Distribution")
        ax.set_xlabel(f"{title} mean")
        ax.set_ylabel("Item count")
        ax.set_xlim(1, 9)
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = figures_dir / f"deam_{target}_histogram.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(str(path))

    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    for is_full_song in (False, True):
        subset = labels.loc[labels["is_full_song"] == is_full_song]
        ax.scatter(
            subset["valence_mean"],
            subset["arousal_mean"],
            s=18 if not is_full_song else 34,
            alpha=0.48 if not is_full_song else 0.8,
            label=f"{group_names[is_full_song]} (n={len(subset)})",
            color=group_colors[is_full_song],
            edgecolors="none",
        )
    ax.set_title("DEAM Static Valence vs Arousal")
    ax.set_xlabel("Valence mean")
    ax.set_ylabel("Arousal mean")
    ax.set_xlim(1, 9)
    ax.set_ylim(1, 9)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = figures_dir / "deam_valence_arousal_scatter.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(str(path))
    return outputs


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    audio_root = data_root / "audio"
    annotation_root = data_root / "annotations"
    metadata_root = data_root / "metadata"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audio = inventory_audio(audio_root)
    static, static_source_files = load_static_annotations(annotation_root)
    metadata, metadata_summary = load_metadata_mapping(metadata_root)

    audio["is_approximately_45_seconds"] = audio["duration_seconds"].between(
        args.excerpt_min, args.excerpt_max, inclusive="both"
    )
    audio["is_substantially_longer"] = audio["duration_seconds"] > args.long_threshold

    metadata_for_merge = metadata.drop_duplicates("song_id", keep=False)
    unified = audio.merge(metadata_for_merge, on="song_id", how="left")
    unified["year_or_subset"] = unified["year_or_subset"].fillna("unknown")
    unified["is_full_song"] = unified["year_or_subset"].eq("2015")
    unified = unified.merge(static, on="song_id", how="left", suffixes=("", "_label"))

    audio_ids = set(audio["song_id"].dropna().astype(int))
    label_ids = set(static["song_id"].dropna().astype(int))
    audio_duplicate_ids = sorted(
        audio.loc[audio["song_id"].duplicated(keep=False), "song_id"].dropna().astype(int).unique().tolist()
    )
    label_duplicate_ids = sorted(
        static.loc[static["song_id"].duplicated(keep=False), "song_id"].dropna().astype(int).unique().tolist()
    )
    malformed_audio = sorted(audio.loc[audio["song_id"].isna(), "filename"].tolist())
    unreadable_audio = audio.loc[~audio["readable"], ["filename", "error"]].to_dict("records")

    matched_ids = sorted(audio_ids & label_ids)
    audio_without_label = sorted(audio_ids - label_ids)
    label_without_audio = sorted(label_ids - audio_ids)

    excerpt_labels = unified.loc[~unified["is_full_song"]]
    full_song_labels = unified.loc[unified["is_full_song"]]
    correlation = unified[["valence_mean", "arousal_mean"]].corr(method="pearson").iloc[0, 1]
    excerpt_correlation = excerpt_labels[["valence_mean", "arousal_mean"]].corr(method="pearson").iloc[0, 1]
    full_song_correlation = full_song_labels[["valence_mean", "arousal_mean"]].corr(method="pearson").iloc[0, 1]

    nonstandard_rates = unified.loc[
        unified["sample_rate"].ne(44_100),
        ["song_id", "filename", "year_or_subset", "sample_rate", "duration_seconds"],
    ].to_dict("records")
    mono_items = unified.loc[
        unified["channels"].eq(1),
        ["song_id", "filename", "year_or_subset", "sample_rate", "duration_seconds"],
    ].to_dict("records")
    short_full_songs = unified.loc[
        unified["is_full_song"] & ~unified["is_substantially_longer"],
        ["song_id", "filename", "duration_seconds", "sample_rate", "channels"],
    ].to_dict("records")

    figures = make_figures(unified, args.figures_dir)

    inventory_path = args.output_dir / "deam_audio_inventory.csv"
    labels_path = args.output_dir / "deam_static_annotations_unified.csv"
    mapping_path = args.output_dir / "deam_item_mapping.csv"
    summary_path = args.output_dir / "deam_verification_summary.json"
    audio.to_csv(inventory_path, index=False)
    static.to_csv(labels_path, index=False)
    unified.to_csv(mapping_path, index=False)

    readable_durations = audio.loc[audio["readable"], "duration_seconds"]
    summary: dict[str, Any] = {
        "paths": {
            "data_root": str(data_root),
            "audio_inventory_csv": str(inventory_path),
            "static_annotations_csv": str(labels_path),
            "item_mapping_csv": str(mapping_path),
            "figures": figures,
            "static_source_files": static_source_files,
            "metadata_source_files": metadata_summary["files"],
        },
        "classification_rules": {
            "approximately_45_seconds": [args.excerpt_min, args.excerpt_max],
            "substantially_longer_than_seconds": args.long_threshold,
            "full_song": "year_or_subset == '2015' from official metadata filename",
        },
        "audio": {
            "mp3_count": int(len(audio)),
            "readable_count": int(audio["readable"].sum()),
            "unreadable": unreadable_audio,
            "malformed_filenames": malformed_audio,
            "duplicate_song_ids": audio_duplicate_ids,
            "duration_seconds": describe_series(readable_durations),
            "approximately_45_second_count": int(audio["is_approximately_45_seconds"].sum()),
            "substantially_longer_count": int(audio["is_substantially_longer"].sum()),
            "sample_rate_counts": value_counts_dict(audio["sample_rate"]),
            "channel_counts": value_counts_dict(audio["channels"]),
            "non_44100_hz_items": nonstandard_rates,
            "mono_items": mono_items,
            "shortest_item": unified.loc[
                unified["duration_seconds"].idxmin(),
                ["song_id", "filename", "year_or_subset", "duration_seconds"],
            ].to_dict(),
            "longest_item": unified.loc[
                unified["duration_seconds"].idxmax(),
                ["song_id", "filename", "year_or_subset", "duration_seconds"],
            ].to_dict(),
        },
        "static_annotations": {
            "item_count": int(len(static)),
            "unique_song_ids": int(static["song_id"].nunique()),
            "duplicate_song_ids": label_duplicate_ids,
            "valence": describe_series(static["valence_mean"]),
            "arousal": describe_series(static["arousal_mean"]),
            "valence_std": describe_series(static["valence_std"]) if "valence_std" in static else None,
            "arousal_std": describe_series(static["arousal_std"]) if "arousal_std" in static else None,
            "pearson_valence_arousal": finite_float(correlation),
        },
        "integrity": {
            "matched_id_count": len(matched_ids),
            "audio_without_static_label": audio_without_label,
            "static_label_without_audio": label_without_audio,
        },
        "metadata": {
            **metadata_summary,
            "mapped_audio_counts": value_counts_dict(unified["year_or_subset"]),
            "full_song_count_from_2015_metadata": int(unified["is_full_song"].sum()),
            "2015_full_songs_not_over_long_threshold": short_full_songs,
        },
        "group_label_descriptives": {
            "non_2015_items": {
                "count": int(len(excerpt_labels)),
                "valence": describe_series(excerpt_labels["valence_mean"]),
                "arousal": describe_series(excerpt_labels["arousal_mean"]),
                "duration_seconds": describe_series(excerpt_labels["duration_seconds"]),
                "pearson_valence_arousal": finite_float(excerpt_correlation),
                "sample_rate_counts": value_counts_dict(excerpt_labels["sample_rate"]),
                "channel_counts": value_counts_dict(excerpt_labels["channels"]),
            },
            "2015_full_songs": {
                "count": int(len(full_song_labels)),
                "valence": describe_series(full_song_labels["valence_mean"]),
                "arousal": describe_series(full_song_labels["arousal_mean"]),
                "duration_seconds": describe_series(full_song_labels["duration_seconds"]),
                "pearson_valence_arousal": finite_float(full_song_correlation),
                "sample_rate_counts": value_counts_dict(full_song_labels["sample_rate"]),
                "channel_counts": value_counts_dict(full_song_labels["channels"]),
            },
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
