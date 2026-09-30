# Module E — Conventional Acoustic Baseline / RQ3 (in progress)

## Research question

How does Valence and Arousal decodability from the pre-specified MERT Layer-12 representation compare with a conventional low-level acoustic representation under the same linear probing framework?

The Module D Test trajectory does not select a new MERT comparator. Layer 12 remains frozen. The intended controlled change is representation type; the same 1,744 primary DEAM excerpts, Sample IDs, targets, split, Train-only scaling and Ridge framework will be used in a later probing Stage.

## Stage 1 — Acoustic extraction and validation (complete)

The frozen 51-dimensional baseline contains one estimated global BPM; temporal mean and standard deviation of frame RMS; those statistics for MFCC1–20 excluding coefficient 0; and those statistics for spectral centroid, bandwidth, 0.85 rolloff, and zero-crossing rate. Audio is decoded at actual recording duration, converted to mono, and resampled to 24 kHz. Frames use length 2,048 and hop 512, with Hann window where applicable. RMS uses the decoded-amplitude path before any MERT input normalization.

The local cache `data/processed/deam_acoustic_51d.npz` contains 1,744 Sample-ID-aligned rows and 51 ordered columns. Validation found no dropped samples, duplicate IDs, extraction failures, NaN, Inf, constant features, or undefined/non-positive Tempo values. ID 437 returned finite 117.1875 BPM with only one tracked beat; this is retained as an uncertain estimate. No near-silent excerpt met the diagnostic thresholds. Detailed implementation, numerical distributions, commands, and the researcher checkpoint are in [the Stage 1 report](../codex_reports/module_e_stage1_acoustic_feature_extraction_and_validation.md); aggregate machine-readable diagnostics are in `outputs/results/module_e_stage1_acoustic_diagnostics.json`.

**Stage 1 passed.** At its close, no probe had been fit, no alpha selected, and no Validation or Test performance examined for the acoustic representation. This remains a current Module log, not a completed Module conclusion.

## Stage 2 — Acoustic Ridge Validation and pre-Test verification (complete)

The Stage 1 representation was kept fixed. Sample-ID alignment and the frozen 1,221/262/261 split were verified. The Module C Train-only StandardScaler and Ridge framework was reused with the same nine alpha candidates, maximum Validation R² selection, larger-alpha exact-tie rule, and constant Train-target-mean reference. Valence selected alpha 100 (Validation MAE 0.721685451, R² 0.383931208, Pearson r 0.620778217); Arousal selected alpha 10 (MAE 0.794242460, R² 0.330553239, r 0.584901982). Their Train-mean reference Validation MAE/R² values were 0.924825572/−0.000046064 and 1.012681884/−0.000018369, with undefined Pearson r.

The complete alpha-grid metrics, selected Validation predictions, input identities, independent verification, and closed-Test gate are detailed in [the Stage 2 report](../codex_reports/module_e_stage2_acoustic_ridge_validation_and_pre_test_verification.md). Only Validation predictions were generated. **Stage 2 passed. Ready for Module E Test gate review.** These are selection results, not a final RQ3 comparison. Module E and RQ3 remain in progress; no Test evaluation was performed in Stage 2.
