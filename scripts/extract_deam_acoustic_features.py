"""Extract the frozen 51-D conventional acoustic baseline for DEAM excerpts.

This script reads audio and Sample-ID/split identity only. It never reads targets,
MERT representations, or probe results. Per-sample parts make interrupted runs
resumable; the final cache is written only after complete integrity validation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mert_emotion_probing.audio import convert_to_mono, load_audio, resample_mono  # noqa: E402
from mert_emotion_probing.splits import PRIMARY_SAMPLE_COUNT  # noqa: E402


SR = 24_000
N_FFT = 2048
HOP = 512
FEATURE_NAMES = (
    ["tempo_bpm", "rms_mean", "rms_std"]
    + [f"mfcc_{i}_{stat}" for i in range(1, 21) for stat in ("mean", "std")]
    + [f"{name}_{stat}" for name in ("spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "zero_crossing_rate") for stat in ("mean", "std")]
)
assert len(FEATURE_NAMES) == 51


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, default=Path("data/raw/deam/audio"))
    parser.add_argument("--mapping-csv", type=Path, default=Path("data/raw/deam/verification/deam_item_mapping.csv"))
    parser.add_argument("--split-csv", type=Path, default=Path("data/metadata/deam_primary_split_seed42.csv"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed/deam_acoustic_51d.npz"))
    parser.add_argument("--diagnostics", type=Path, default=Path("outputs/results/module_e_stage1_acoustic_diagnostics.json"))
    parser.add_argument("--sample-diagnostics", type=Path, default=Path("data/processed/deam_acoustic_sample_diagnostics.csv"))
    parser.add_argument("--parts-dir", type=Path, default=Path("data/processed/deam_acoustic_51d_parts"))
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def protocol_ids(mapping_path: Path, split_path: Path) -> list[int]:
    # Intentionally read identity/subset only. Labels never enter extraction.
    mapping = pd.read_csv(mapping_path, usecols=["song_id", "is_full_song"])
    if mapping.song_id.isna().any() or mapping.song_id.duplicated().any():
        raise ValueError("Invalid or duplicate canonical mapping IDs")
    primary = mapping.loc[~mapping.is_full_song.astype(bool), "song_id"].astype(int)
    ids = sorted(primary.tolist())
    if len(ids) != PRIMARY_SAMPLE_COUNT or len(set(ids)) != PRIMARY_SAMPLE_COUNT:
        raise ValueError("Canonical primary population is not 1,744 unique IDs")
    split = pd.read_csv(split_path)
    if list(split.columns) != ["sample_id", "split"] or split.sample_id.duplicated().any():
        raise ValueError("Invalid frozen split schema or duplicate IDs")
    counts = split["split"].value_counts().to_dict()
    if counts != {"train": 1221, "validation": 262, "test": 261}:
        raise ValueError(f"Frozen split counts changed: {counts}")
    if set(split.sample_id.astype(int)) != set(ids):
        raise ValueError("Frozen split does not exactly match canonical population")
    return ids


def audio_index(root: Path, ids: list[int]) -> dict[int, Path]:
    expected = set(ids)
    found: dict[int, Path] = {}
    for path in root.rglob("*.mp3"):
        try:
            sample_id = int(path.stem)
        except ValueError:
            continue
        if sample_id in expected:
            if sample_id in found:
                raise ValueError(f"Duplicate audio ID {sample_id}")
            found[sample_id] = path
    if set(found) != expected:
        raise ValueError(f"Missing audio IDs: {sorted(expected - set(found))}")
    return found


def stats(frames: np.ndarray) -> list[float]:
    values = np.asarray(frames, dtype=np.float64).reshape(-1)
    return [float(np.mean(values)), float(np.std(values, ddof=0))]


def extract_one(path: Path) -> tuple[np.ndarray, dict]:
    loaded = load_audio(path)
    mono = convert_to_mono(loaded.waveform)
    resampled = resample_mono(mono, loaded.sample_rate, SR)
    y = np.asarray(resampled.numpy(), dtype=np.float32)
    if y.size == 0 or not np.isfinite(y).all():
        raise ValueError("Empty or non-finite decoded waveform")

    # Direct decoded-amplitude path: no MERT feature-extractor normalization.
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP, center=True)[0]
    spectrum = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window="hann", center=True))
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=21, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window="hann", center=True)
    centroid = librosa.feature.spectral_centroid(S=spectrum, sr=SR)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=spectrum, sr=SR, centroid=centroid.reshape(1, -1))[0]
    rolloff = librosa.feature.spectral_rolloff(S=spectrum, sr=SR, roll_percent=0.85)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP, center=True)[0]
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, n_fft=N_FFT)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=SR, hop_length=HOP, trim=True)
    tempo = float(np.asarray(tempo).reshape(-1)[0])
    valid_tempo = bool(np.isfinite(tempo) and tempo > 0)
    # Preserve undefined tempo as NaN, never as factual BPM=0.
    values = [tempo if valid_tempo else np.nan] + stats(rms)
    for coefficient in mfcc[1:21]:
        values.extend(stats(coefficient))
    for frames in (centroid, bandwidth, rolloff, zcr):
        values.extend(stats(frames))
    vector = np.asarray(values, dtype=np.float32)
    diagnostic = {
        "source_audio": path.as_posix(),
        "source_sample_rate": loaded.sample_rate,
        "source_channels": loaded.channels,
        "decoded_duration_seconds": loaded.duration_seconds,
        "resampled_samples": int(y.size),
        "peak_absolute_amplitude": float(np.max(np.abs(y))),
        "whole_waveform_rms": float(np.sqrt(np.mean(np.square(y.astype(np.float64))))),
        "tempo_raw_bpm": tempo if np.isfinite(tempo) else None,
        "tempo_valid": valid_tempo,
        "beat_count": int(len(beats)),
        "onset_peak": float(np.max(onset)) if onset.size else 0.0,
        "nan_count": int(np.isnan(vector).sum()),
        "inf_count": int(np.isinf(vector).sum()),
    }
    return vector, diagnostic


def summarize(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"count": 0}
    return {"count": int(finite.size), "min": float(np.min(finite)), "p01": float(np.percentile(finite, 1)), "median": float(np.median(finite)), "p99": float(np.percentile(finite, 99)), "max": float(np.max(finite))}


def main() -> None:
    args = arguments()
    ids = protocol_ids(args.mapping_csv, args.split_csv)
    paths = audio_index(args.audio_root, ids)
    args.parts_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    vectors = []
    failures = []
    for index, sample_id in enumerate(ids, 1):
        part = args.parts_dir / f"{sample_id}.npz"
        if part.exists():
            with np.load(part, allow_pickle=False) as saved:
                if int(saved["sample_id"]) != sample_id:
                    raise ValueError(f"Wrong ID in {part}")
                vector = saved["features"]
                diagnostic = json.loads(str(saved["diagnostic_json"]))
        else:
            try:
                vector, diagnostic = extract_one(paths[sample_id])
                diagnostic["sample_id"] = sample_id
                diagnostic["source_audio"] = paths[sample_id].resolve().relative_to(ROOT).as_posix()
                temporary = part.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, sample_id=sample_id, features=vector, diagnostic_json=json.dumps(diagnostic))
                os.replace(temporary, part)
            except Exception as exc:
                failures.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
                print(f"Extraction failed for {sample_id}: {type(exc).__name__}: {exc}", flush=True)
                continue
        if vector.shape != (51,):
            failures.append({"sample_id": sample_id, "error": f"Wrong vector shape {vector.shape}"})
            continue
        vectors.append(vector)
        rows.append(diagnostic)
        if index % args.progress_every == 0:
            print(f"{index}/{len(ids)} processed", flush=True)

    df = pd.DataFrame(rows)
    matrix = np.stack(vectors) if vectors else np.empty((0, 51), dtype=np.float32)
    actual_ids = df.sample_id.astype(int).tolist() if len(df) else []
    tempo = matrix[:, 0] if len(matrix) else np.array([])
    report = {
        "protocol": {"sample_rate_hz": SR, "mono": "arithmetic channel mean", "duration": "actual decoded recording", "frame_length": N_FFT, "hop_length": HOP, "window": "Hann where applicable", "rolloff_fraction": 0.85, "tempo_estimator": "librosa.beat.beat_track on librosa onset strength", "mfcc": "librosa MFCC 1-20 excluding 0", "aggregation": "population temporal mean and standard deviation (ddof=0)", "energy_path": "decoded waveform -> mono -> 24kHz resample; no MERT normalization"},
        "feature_names": FEATURE_NAMES,
        "shape": list(matrix.shape),
        "sample_ids_unique": len(set(actual_ids)) == len(actual_ids),
        "exact_canonical_alignment": actual_ids == ids,
        "split_unchanged": True,
        "extraction_failures": failures,
        "nan_count": int(np.isnan(matrix).sum()),
        "inf_count": int(np.isinf(matrix).sum()),
        "tempo_valid_count": int(np.isfinite(tempo).sum()),
        "tempo_invalid_ids": [int(actual_ids[i]) for i in np.flatnonzero(~np.isfinite(tempo))],
        "tempo_bpm": summarize(tempo),
        "tempo_extreme_ids_below_40_or_above_240": [int(actual_ids[i]) for i in np.flatnonzero(np.isfinite(tempo) & ((tempo < 40) | (tempo > 240)))],
        "tempo_low_beat_count_ids_below_2": df.loc[df.beat_count < 2, "sample_id"].astype(int).tolist() if len(df) else [],
        "rms_mean": summarize(matrix[:, 1]) if len(matrix) else {"count": 0},
        "whole_waveform_rms": summarize(df.whole_waveform_rms.to_numpy()) if len(df) else {"count": 0},
        "near_silence_ids_peak_below_1e-4": df.loc[df.peak_absolute_amplitude < 1e-4, "sample_id"].astype(int).tolist() if len(df) else [],
        "near_silence_ids_rms_below_1e-5": df.loc[df.whole_waveform_rms < 1e-5, "sample_id"].astype(int).tolist() if len(df) else [],
        "feature_summaries": {name: summarize(matrix[:, i]) for i, name in enumerate(FEATURE_NAMES)},
        "constant_feature_names": [name for i, name in enumerate(FEATURE_NAMES) if len(matrix) and np.isfinite(matrix[:, i]).all() and np.min(matrix[:, i]) == np.max(matrix[:, i])],
        "versions": {"librosa": librosa.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }
    report["passed"] = bool(matrix.shape == (PRIMARY_SAMPLE_COUNT, 51) and report["sample_ids_unique"] and report["exact_canonical_alignment"] and not failures and report["nan_count"] == 0 and report["inf_count"] == 0 and not report["constant_feature_names"])
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.sample_diagnostics.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.sample_diagnostics, index=False)
    if report["passed"]:
        with args.cache.open("wb") as handle:
            np.savez_compressed(handle, sample_ids=np.asarray(ids, dtype=np.int64), features=matrix, feature_names=np.asarray(FEATURE_NAMES))
    print(json.dumps({key: report[key] for key in ("shape", "extraction_failures", "nan_count", "inf_count", "tempo_valid_count", "tempo_invalid_ids", "constant_feature_names", "passed")}, indent=2))
    if not report["passed"]:
        raise RuntimeError("Stage 1 validation failed; no final cache written")


if __name__ == "__main__":
    main()
