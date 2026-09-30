# Module E Stage 3 — Frozen Test Evaluation and RQ3 Comparison

## Objective and outcome

Evaluate the frozen conventional acoustic Ridge configuration on the 261 Test excerpts and compare its held-out point estimates with the authoritative Module C MERT Layer-12 results. **Stage 3 passed. Module E / operational RQ3 is complete under the frozen protocol.** The next step is the researcher learning checkpoint and interpretation review, followed by Module F design. Module F / RQ4 has not started.

Under the fixed linear probing framework on the frozen DEAM split, the pre-specified MERT Layer-12 representation shows stronger held-out Valence and Arousal decodability than the selected 51-dimensional conventional acoustic baseline. This is a descriptive comparison; it does not establish statistically significant superiority or unique emotion information.

## Context recovery and Test authorization

Before implementation, the workflow, frozen post-Module-D roadmap, README, Module C/D/E logs, and Module E Stage 1/2 reports were read. The repository was on `main`, with Stage 1 commit `8aae770fc624589439e71c47929924596baef400` and Stage 2 commit `2435a7e3842725593eeeeabb2d339455be3bd811`. The only pre-existing untracked state was `docs/.obsidian/`, which was not inspected or included.

The researcher explicitly confirmed Stage 1/2 review, approved the Stage 2 Test gate, and authorized the first formal acoustic Test evaluation. Preflight checks verified all Stage 2 input hashes and sizes, its passing closed-Test artifact, unchanged committed Stage 2 results and predictions, the 1,744 × 51 cache, exact split, and artifact-derived selected alphas. No acoustic Stage 3 result/start artifact was present. An exclusively created `module_e_stage3_test_started.json` records the approved configuration and provenance immediately before prediction; the runner refuses a subsequent official run while those artifacts exist.

## Frozen evaluation implementation

The existing `fit_train_evaluate_test` function was reused with a 51-dimensional shape parameter; its default remains 768 dimensions for Module C. Optional fitted scaler statistics were added for an independent Train-only audit. No Ridge methodology changed.

The same 1,744 primary excerpt IDs, static targets, and 1,221/262/261 split were joined by Sample ID. Each target used its alpha read from the authoritative Stage 2 artifact: Valence 100 and Arousal 10. A fresh StandardScaler was fit on the 1,221 Train rows and transformed Train and Test only. Ridge used an intercept and Cholesky solver and fit on Train only. Targets remained on their original scale. Validation was not added to the final fit, and no alpha search was performed. Each target's acoustic Test prediction was generated once. No acoustic extraction or MERT model fitting/evaluation was rerun.

The conventional representation remained frozen: estimated global Tempo; temporal mean/std of RMS, MFCC1–20 excluding coefficient 0, spectral centroid, bandwidth, 0.85 rolloff, and zero-crossing rate. Mono, 24 kHz, actual decoded duration, frame 2,048, hop 512, and Hann where applicable remain unchanged. RMS retains the decoded-recording amplitude path. The acoustic NPZ SHA-256 remains `a737863c8c2bfb623448fe1d2d770e27206b9f42bee57faacf86ef739da0b6c4`.

## Official held-out comparison

| Target | Representation | Selected alpha | Test MAE | Test R² | Test Pearson r |
|---|---|---:|---:|---:|---:|
| Valence | MERT Layer 12 (reused Module C) | 1000 | 0.635144565 | 0.579780312 | 0.768855324 |
| Valence | Conventional Acoustic 51-D | 100 | 0.770499050 | 0.368149823 | 0.609708274 |
| Arousal | MERT Layer 12 (reused Module C) | 1000 | 0.744556180 | 0.511421706 | 0.715509311 |
| Arousal | Conventional Acoustic 51-D | 10 | 0.860429090 | 0.374161643 | 0.611792750 |

Full precision is retained in JSON and the comparison CSV. MERT has lower MAE and higher R²/r for both targets. Both acoustic probes nevertheless have positive held-out R² and lower MAE than their Train-mean references, supporting conventional acoustic linear decodability in this setup.

| Reference | Train target mean | Test MAE | Test R² | Test Pearson r |
|---|---:|---:|---:|---|
| Valence Train mean | 4.894692875 | 1.002563190 | −0.003113807 | undefined (`null`) |
| Arousal Train mean | 4.816953317 | 1.131995632 | −0.000320133 | undefined (`null`) |

The reference values exactly match Module C because both comparisons use the same targets and split. The slight negative R² values are valid: the constant comes from Train rather than the Test mean.

## MERT provenance and prior Test exposure

The comparator is `outputs/results/module_c_stage4_test.json` with its saved Test predictions. Before the acoustic evaluation, the Module C verifier passed, Git blob checks confirmed unchanged authoritative files, and their saved metrics were recomputed from Sample-ID-aligned predictions. The copied comparison values exactly equal Module C's full-precision JSON values.

These MERT values reuse the accepted RQ1 evaluation. They are not a new MERT experiment or independent replication. The same Test set was previously examined in Modules C/D. That prior exposure remains explicit; it did not reselect the RQ3 layer or tune this frozen acoustic experiment. Module D maxima were not used as new comparators.

## Final verification

All 19 final checks passed. The verifier fits no model and performs no selection. It confirms unchanged cache/input and evaluation-code hashes; exact population, feature order and split membership; 261 unique Test IDs per target and no cross-split overlap; Sample-ID target alignment; frozen alphas; fitted scaler mean/variance matching the Train cache; recorded Train-only Ridge/intercept/Cholesky fitting; and original target scale. The saved CSV contains exactly 522 Test prediction rows.

Using round-trip float parsing, MAE, R², Pearson r, and reference metrics exactly reproduce the saved JSON. Independent NumPy formulas also agree within `1e-12`. The Train target means recompute exactly. Module C artifacts remain unchanged, their metrics reproduce, and every comparison-table metric matches its source exactly. The start record and controls confirm one official prediction call per target, no post-Test tuning, no refitting with Validation, no MERT rerun, and no RQ4 analysis.

No implementation failure, invalid value, or unresolved research-level change occurred during Stage 3. No acoustic feature, preprocessing parameter, target, split, selection criterion, alpha, or comparator was revised after Test.

## Files and commands

- `src/mert_emotion_probing/probing.py`: optional input dimension and scaler-statistic recording in the existing frozen Test routine.
- `scripts/run_acoustic_test_evaluation.py`: preflight, official evaluation guard, acoustic predictions, and authoritative comparison.
- `scripts/verify_acoustic_test_results.py`: independent saved-artifact verification without refitting.
- `outputs/results/module_e_stage3_test_started.json`: frozen gate-opening record.
- `outputs/results/module_e_stage3_test.json`: full-precision acoustic metrics, references, fit audit, comparison, input/code identities, and controls.
- `outputs/results/module_e_stage3_test_predictions.csv`: 522 Sample-ID-aligned acoustic Test rows.
- `outputs/results/module_e_stage3_rq3_comparison.csv`: four descriptive comparison rows with reused-evidence flags.
- `outputs/results/module_e_stage3_final_verification.json`: passing verification and output identities.
- Updated Module E research log and README reflect RQ3 completion and the future RQ4 boundary.

Commands were run from the repository root:

```powershell
& 'C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\python.exe' -m py_compile scripts/run_acoustic_test_evaluation.py scripts/verify_acoustic_test_results.py src/mert_emotion_probing/probing.py
& 'C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\python.exe' scripts/verify_acoustic_validation.py
& 'C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\python.exe' scripts/verify_held_out_test_results.py
& 'C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\python.exe' scripts/run_acoustic_test_evaluation.py --preflight-only
& 'C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\python.exe' scripts/run_acoustic_test_evaluation.py
& 'C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\python.exe' scripts/verify_acoustic_test_results.py
```

The official acoustic evaluation command was invoked once. The existing Module C verifier reads its artifacts and recomputes their metrics; it does not train or run MERT. Dependency versions are recorded in the Test JSON.

## Evidence boundaries and future RQ4

This comparison uses one frozen DEAM split, one specified conventional recipe, and a restricted linear probe. Matching the methodology does not match feature dimensionality or effective capacity: MERT has 768 dimensions and acoustics 51. The outcome does not identify unique/additional emotion information, causality, statistical superiority, universal ranking, generalization beyond this setting, or human-like understanding. It does not establish that conventional acoustic correlates are absent from MERT or that a different acoustic recipe could not perform differently.

The completed table, Tempo reliability diagnostics, and decoded-amplitude RMS features provide inputs for future Module F design. No Tempo/Energy-target, prediction, residual, or confound association was calculated here.

## What I should now be able to explain

- Why RQ3 compares two representation types using a matched Ridge framework and why Layer 12 remains pre-specified.
- Why alpha 100/10 is inherited from Validation and the final scaler/Ridge still use only Train.
- How the acoustic Test MAE/R²/r and the shared Train-mean references support decodability.
- Why all three metrics favor MERT here while the comparison remains descriptive and does not establish unique information.
- Why the MERT rows retain their Module C provenance and prior Test exposure must remain visible.
- Which association questions remain for Module F and why completing RQ3 does not complete RQ4.

**Stage 3 passed. Module E / RQ3 complete under the frozen protocol.** Researcher learning and interpretation review are next; no human checkpoint completion is claimed by this implementation report.
