# Module E — Conventional Acoustic Baseline / RQ3

## Research question

How does Valence and Arousal decodability from the pre-specified MERT Layer-12 representation compare with a conventional low-level acoustic representation under the same linear probing framework?

The Module D Test trajectory does not select a new MERT comparator. Layer 12 remains frozen. The controlled change is representation type; both branches use the same 1,744 primary DEAM excerpts, Sample IDs, original-scale static targets, 1,221/262/261 Train/Validation/Test split, Train-only scaling and Ridge framework. The 58 full songs remain excluded. Identity joins never depend on row order.

**Module E / operational RQ3 is finalized under the frozen protocol.** All three implementation Stages passed researcher review. The researcher learning checkpoint material and final interpretation review were completed on 2026-10-01. The review confirmed that the reported numbers match the authoritative artifacts and that the conclusion, reused evidence provenance, prior Test exposure and interpretation boundaries are accurate. The Chinese learning note remains private/local and excluded from Git. No experiment, frozen protocol or result changed. Module F / RQ4 has not started; its design is the next research task.

> **Historical-status note:** The paragraph above records Module E's closure before F. [Module F / RQ4 is now complete](module_f_acoustic_correlate_and_error_analysis.md); the current project phase is Final Synthesis. The original closure statement is preserved.

## Stage 1 — Acoustic extraction and validation (complete)

The frozen 51-dimensional baseline contains one estimated global BPM; temporal mean and standard deviation of frame RMS; those statistics for MFCC1–20 excluding coefficient 0; and those statistics for spectral centroid, bandwidth, 0.85 rolloff, and zero-crossing rate. Audio is decoded at actual recording duration, converted to mono, and resampled to 24 kHz. Frames use length 2,048 and hop 512, with Hann window where applicable. RMS uses the decoded-amplitude path before any MERT input normalization.

The local cache `data/processed/deam_acoustic_51d.npz` contains 1,744 Sample-ID-aligned rows and 51 ordered columns. Validation found no dropped samples, duplicate IDs, extraction failures, NaN, Inf, constant features, or undefined/non-positive Tempo values. ID 437 returned finite 117.1875 BPM with only one tracked beat; this is retained as an uncertain estimate. No near-silent excerpt met the diagnostic thresholds. Detailed implementation, numerical distributions, commands, and the researcher checkpoint are in [the Stage 1 report](../codex_reports/module_e_stage1_acoustic_feature_extraction_and_validation.md); aggregate machine-readable diagnostics are in `outputs/results/module_e_stage1_acoustic_diagnostics.json`.

**Stage 1 passed.** At its close, no probe had been fit, no alpha selected, and no Validation or Test performance examined for the acoustic representation.

## Stage 2 — Acoustic Ridge Validation and pre-Test verification (complete)

The Stage 1 representation was kept fixed. Sample-ID alignment and the frozen 1,221/262/261 split were verified. The Module C Train-only StandardScaler and Ridge framework was reused with the same nine alpha candidates, maximum Validation R² selection, larger-alpha exact-tie rule, and constant Train-target-mean reference. Valence selected alpha 100 (Validation MAE 0.721685451, R² 0.383931208, Pearson r 0.620778217); Arousal selected alpha 10 (MAE 0.794242460, R² 0.330553239, r 0.584901982). Their Train-mean reference Validation MAE/R² values were 0.924825572/−0.000046064 and 1.012681884/−0.000018369, with undefined Pearson r.

The complete alpha-grid metrics, selected Validation predictions, input identities, independent verification, and closed-Test gate are detailed in [the Stage 2 report](../codex_reports/module_e_stage2_acoustic_ridge_validation_and_pre_test_verification.md). Only Validation predictions were generated. **Stage 2 passed.** These were selection results; no Test evaluation was performed in Stage 2. The researcher subsequently approved this gate.

## Stage 3 — Official Test evaluation and RQ3 comparison

Stage 3 read the alphas and input identities from the authoritative Stage 2 artifacts. Each acoustic scaler and frozen Ridge configuration was fit using the 1,221 Train excerpts only, then evaluated once on the 261 Test excerpts. No Test-informed tuning or feature change occurred. MERT was not refit or reevaluated; the existing Module C Test JSON and predictions supplied its authoritative results.

Both branches retain Ridge with intercept and Cholesky solver, the nine-alpha grid (`1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000`), maximum Validation R² selection with larger alpha on an actual exact tie, Train-only StandardScaler, original-scale targets, and final Train-only fitting. Validation was not combined with Train in the final fit. No acoustic extraction was repeated. The common framework does not imply equal dimension or effective capacity: MERT Layer 12 has 768 features and the acoustic baseline 51.

| Target | Representation | Selected alpha | Test MAE | Test R² | Test Pearson r |
|---|---|---:|---:|---:|---:|
| Valence | MERT Layer 12 (reused Module C) | 1000 | 0.635144565 | 0.579780312 | 0.768855324 |
| Valence | Conventional Acoustic 51-D | 100 | 0.770499050 | 0.368149823 | 0.609708274 |
| Arousal | MERT Layer 12 (reused Module C) | 1000 | 0.744556180 | 0.511421706 | 0.715509311 |
| Arousal | Conventional Acoustic 51-D | 10 | 0.860429090 | 0.374161643 | 0.611792750 |

| Shared Train-mean reference | Train mean | Test MAE | Test R² | Pearson r |
|---|---:|---:|---:|---|
| Valence | 4.894692875 | 1.002563190 | −0.003113807 | undefined (`null`) |
| Arousal | 4.816953317 | 1.131995632 | −0.000320133 | undefined (`null`) |

MERT has lower held-out MAE and higher R² and Pearson r for both targets. The acoustic baseline also achieves positive held-out R² and lower MAE than its reference, so conventional features support substantial linear decodability within this setup. References are identical across representations because the targets and split are identical. Slight negative reference R² is valid because the constant comes from Train rather than the Test mean.

## Verification and provenance

All 19 final verification checks passed. The saved acoustic Test CSV contains 522 unique Sample-ID/target rows, exactly 261 Test IDs per target. With round-trip float parsing, metrics and references exactly reproduce the full-precision JSON; independent numerical formulas also agree. The actual fitted scaler means/variances match the Train features. Frozen alpha, Train-only Ridge configuration, target alignment, unchanged cache/split/Stage 2 artifacts, and no post-Test tuning were confirmed.

The Module C results and prediction artifacts remain unchanged. Their metrics were verified from their saved predictions, and comparison rows exactly match both source artifacts. The MERT rows reuse the accepted RQ1 evidence; they are not a new MERT experiment or an independent replication. The fixed Test set had already been examined in C/D, and that prior exposure remains part of the comparison's provenance. It did not select the MERT layer or change the frozen acoustic design.

Detailed commands, identities, checks and files are in the [Stage 3 report](../codex_reports/module_e_stage3_frozen_test_evaluation_and_rq3_comparison.md). Primary evidence is the [RQ3 comparison CSV](../../outputs/results/module_e_stage3_rq3_comparison.csv), [acoustic Test JSON](../../outputs/results/module_e_stage3_test.json), and [saved acoustic Test predictions](../../outputs/results/module_e_stage3_test_predictions.csv); MERT values reuse the [authoritative C4 Test JSON](../../outputs/results/module_c_stage4_test.json). The result, predictions, comparison table, start record and verification use `outputs/results/module_e_stage3_*`. Raw audio and both representation caches remain local and excluded from Git.

## RQ3 conclusion and evidence boundary

**Under the fixed linear probing framework on the frozen DEAM split, the pre-specified MERT Layer-12 representation shows stronger held-out Valence/Arousal decodability than the selected conventional acoustic baseline.**

This conclusion is descriptive and specific to these representations, this probe, and this split. No uncertainty interval or significance test was added. The comparison does not establish unique/additional emotion information, causality, statistically significant superiority, independence from acoustic correlates, human-like understanding, universal superiority, or cross-dataset generalization. A 51-dimensional compact recipe does not exhaust all possible conventional acoustic information, and equal probing rules do not guarantee equal effective capacity.

RMS remains a decoded-amplitude signal descriptor rather than calibrated perceptual loudness or Arousal. Recording gain and production are possible influences. Automated Tempo remains subject to half/double-tempo ambiguity, weak onsets and beat-tracking uncertainty. The uncertain ID 437 estimate was retained with its Stage 1 diagnostic; no BPM was manually corrected or repaired using emotion labels.

## What I should now be able to explain

- Why Layer 12 remained the pre-specified comparator after Module D and why representation type was the controlled change.
- How the 51 features are defined, and why RMS and Tempo interpretation boundaries matter.
- Why Validation selected alpha 100/10 while final fitting still used only Train.
- What MAE, R² and Pearson r jointly show for acoustic and MERT decodability.
- Why reused Module C evidence and prior Test exposure must remain explicit.
- Why the performance gap does not identify unique information, causal mechanisms, or statistically significant superiority.
- Why Module F requires its own design and cannot be completed by this representation comparison.

## Inputs to future RQ4

> **Historical handoff:** The paragraph below records the next step at E closure. See the [completed Module F log](module_f_acoustic_correlate_and_error_analysis.md) for the subsequent RQ4 evidence and Final Synthesis handoff.

Module E supplies the frozen acoustic table, extraction diagnostics, acoustic/MERT predictions and descriptive RQ3 comparison. Future Module F will examine how observed decoding is associated with simple acoustic correlates, particularly Tempo and Energy, after its own design review. No Tempo/Energy-target, prediction, residual or confound association was performed in Module E or during its final learning checkpoint. **RQ3 is finalized; RQ4 has not started. The next step is Module F design, not experiment execution.**
