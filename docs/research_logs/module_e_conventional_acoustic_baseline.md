# Module E — Conventional Acoustic Baseline / RQ3 (in progress)

## Research question

How does Valence and Arousal decodability from the pre-specified MERT Layer-12 representation compare with a conventional low-level acoustic representation under the same linear probing framework?

The Module D Test trajectory does not select a new MERT comparator. Layer 12 remains frozen. The intended controlled change is representation type; the same 1,744 primary DEAM excerpts, Sample IDs, targets, split, Train-only scaling and Ridge framework will be used in a later probing Stage.

## Stage 1 — Acoustic extraction and validation (complete)

The frozen 51-dimensional baseline contains one estimated global BPM; temporal mean and standard deviation of frame RMS; those statistics for MFCC1–20 excluding coefficient 0; and those statistics for spectral centroid, bandwidth, 0.85 rolloff, and zero-crossing rate. Audio is decoded at actual recording duration, converted to mono, and resampled to 24 kHz. Frames use length 2,048 and hop 512, with Hann window where applicable. RMS uses the decoded-amplitude path before any MERT input normalization.

The local cache `data/processed/deam_acoustic_51d.npz` contains 1,744 Sample-ID-aligned rows and 51 ordered columns. Validation found no dropped samples, duplicate IDs, extraction failures, NaN, Inf, constant features, or undefined/non-positive Tempo values. ID 437 returned finite 117.1875 BPM with only one tracked beat; this is retained as an uncertain estimate. No near-silent excerpt met the diagnostic thresholds. Detailed implementation, numerical distributions, commands, and the researcher checkpoint are in [the Stage 1 report](../codex_reports/module_e_stage1_acoustic_feature_extraction_and_validation.md); aggregate machine-readable diagnostics are in `outputs/results/module_e_stage1_acoustic_diagnostics.json`.

**Stage 1 passed. Ready for Module E probing design/review.** No probe has been fit, no alpha selected, and no Validation or Test performance examined for the acoustic representation. RQ3 remains open; this is a current Module log, not a completed Module conclusion.
