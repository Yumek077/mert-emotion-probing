# Module E Stage 1 — Acoustic Feature Extraction and Data-Quality Validation

## Objective and status

Construct the frozen conventional acoustic representation for RQ3 without probing, target-based repair, Validation comparison, or Test evaluation. **Stage 1 passed. Ready for Module E probing design/review.** RQ3 is not complete.

## Repository context and protected decisions

The project workflow, post-Module-D roadmap, README, and Module A–D logs were read before implementation. The same 1,744 primary DEAM excerpt IDs and the existing 1,221/262/261 Train/Validation/Test split were used. The split file was read only to assert its identity and counts. The script reads only `song_id` and `is_full_song` from the canonical mapping; it does not read targets. No MERT extraction, prior result, probe, or held-out evaluation was run. MERT Layer 12 remains the RQ3 comparator for a later Stage.

## Frozen feature definition and ordering

Each decoded recording is kept at its actual duration, converted to mono by arithmetic channel mean, then resampled to 24 kHz. Feature frames use 2,048 samples and a 512-sample hop. The spectral transform and MFCC analysis use a Hann window; spectral rolloff uses 0.85. Frame statistics use temporal population mean and standard deviation (`ddof=0`). Librosa 0.11.0 supplies the acoustic estimators.

| Column indices | Feature order | Dimensions |
|---|---|---:|
| 0 | estimated global `tempo_bpm` from onset strength and `librosa.beat.beat_track` | 1 |
| 1–2 | `rms_mean`, `rms_std` | 2 |
| 3–42 | `mfcc_1_mean`, `mfcc_1_std`, …, `mfcc_20_mean`, `mfcc_20_std`; coefficient 0 excluded | 40 |
| 43–44 | `spectral_centroid_mean`, `spectral_centroid_std` | 2 |
| 45–46 | `spectral_bandwidth_mean`, `spectral_bandwidth_std` | 2 |
| 47–48 | `spectral_rolloff_mean`, `spectral_rolloff_std` | 2 |
| 49–50 | `zero_crossing_rate_mean`, `zero_crossing_rate_std` | 2 |

The cache and tracked diagnostics contain the complete ordered `feature_names` array. `librosa.feature.mfcc` uses its default log-mel/DCT implementation with `n_mfcc=21`, and only rows 1–20 are retained. There was no feature or parameter search. RMS comes directly from the decoded-amplitude → mono → resampled waveform path; no MERT feature-extractor normalization is involved. RMS is a signal-level amplitude descriptor, neither calibrated perceptual loudness nor the Arousal label.

## Implementation and artifacts

- `scripts/extract_deam_acoustic_features.py` indexes MP3s and joins only by authoritative Sample ID; duplicate or missing IDs fail before extraction.
- `data/processed/deam_acoustic_51d.npz` is the local, Git-ignored final cache: `sample_ids` `[1744]`, `features` `[1744, 51]`, and ordered `feature_names` `[51]`. Rows are ascending Sample ID.
- `data/processed/deam_acoustic_sample_diagnostics.csv` is the local, Git-ignored trace from Sample ID to relative source audio path and decoding/extraction diagnostics. Per-sample resumable parts are also local and ignored.
- `outputs/results/module_e_stage1_acoustic_diagnostics.json` is the tracked aggregate validation and feature-range record, containing no labels or raw audio.

The first run exposed an ordinary path bug in writing a relative source path; it was fixed by resolving the audio path before converting it to repository-relative form. A subsequent first full pass marked ID 437 invalid because it had fewer than two tracked beats. Review showed that the estimator returned a finite nonzero 117.1875 BPM. Beat count is a reliability diagnostic, not a frozen condition for undefined BPM. The code was corrected to classify only non-finite or non-positive BPM as invalid, and ID 437 was freshly re-extracted. No label or predictive result guided this correction.

## Validation results

| Check | Result |
|---|---|
| Exact population and cache shape | 1,744 unique IDs × 51 dimensions |
| Canonical alignment | Exact ascending Sample-ID equality; no missing, extra, duplicate, or dropped sample |
| Frozen split | Exact same ID set and 1,221/262/261 counts; no split write |
| Extraction exceptions | 0 |
| NaN / Inf | 0 / 0 across the final matrix |
| Constant features | 0 |
| Tempo finite and positive | 1,744/1,744; undefined/zero cases 0 |
| Tempo suspicious extremes (<40 or >240 BPM) | 0 |
| Near silence (peak <1e-4 or whole-waveform RMS <1e-5) | 0 / 0 |
| Independent fresh recalculation | IDs 2, 437, and 2000 exactly equal their cached vectors |

Tempo estimates range from 50.223 to 200.893 BPM (1st percentile 80.357, median 122.283, 99th percentile 175.781). ID 437 has 117.1875 BPM but only one tracked beat, so its estimated global tempo is uncertain; it is retained without manual correction. Half/double-tempo ambiguity and other beat-tracking errors remain general limitations. The observed failure rate for undefined BPM is 0/1,744.

RMS temporal means range from 0.008709 to 0.559333 (median 0.094508); whole-waveform RMS ranges from 0.010051 to 0.565082. No near-silent samples met the documented diagnostic thresholds. Selected numerical sanity ranges: MFCC1 mean 5.573–246.623; MFCC20 mean −18.540–16.352; centroid mean 254.171–4,663.475 Hz; bandwidth mean 357.490–3,479.130 Hz; rolloff mean 282.066–8,631.601 Hz; zero-crossing-rate mean 0.01264–0.38111. Every feature's detailed min, 1st percentile, median, 99th percentile, and max is in the tracked JSON. Finite values and plausible ranges do not certify ground-truth musical tempo or perceptual validity.

## Commands and environment

Run from the repository root with the project's `mert-emotion` Python environment:

```powershell
$env:NUMBA_CACHE_DIR=(Join-Path (Get-Location) 'data\processed\numba_cache')
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMBA_NUM_THREADS='1'
& 'C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\python.exe' -m py_compile scripts/extract_deam_acoustic_features.py
& 'C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\python.exe' scripts/extract_deam_acoustic_features.py --progress-every 100
```

The local writable Numba cache avoids startup cache access delays in this environment and does not alter feature values. The final run succeeded. An independent Python check reopened the final NPZ, verified its shape, IDs, feature names, finiteness and basic physical bounds, then freshly decoded/extracted IDs 2, 437 and 2000 and obtained exact `np.array_equal` equality to cached rows.

## What I should now be able to explain

- Why this Stage establishes an acoustic input table, not RQ3 predictive performance.
- Why Sample ID is the join key, and why all 1,744 excerpts must remain present.
- How the 51 values arise from Tempo, RMS, MFCC1–20 and four shape descriptors.
- Why RMS must precede MERT amplitude normalization, and why RMS is not Arousal or calibrated loudness.
- Why a finite BPM with one tracked beat is retained but explicitly marked uncertain.
- Why Train/Validation/Test selection and MERT Layer-12 comparison belong to a later, separately reviewed probing Stage.

No research-level issue requires changing the frozen protocol. **Ready for Module E probing design/review.**
