# Module E Stage 2 — Acoustic Ridge Validation and Pre-Test Verification

## Objective and status

Use the frozen 51-dimensional conventional acoustic cache for Train/Validation Ridge selection under the Module C framework, then verify the protected Test gate. **Stage 2 passed. Ready for Module E Test gate review.** No acoustic Test prediction or metric was computed; RQ3 remains in progress.

## Inputs and frozen protocol

Repository context was restored from the project workflow, post-Module-D roadmap, README, Module C/D/E logs, and the Stage 1 report. The Stage 1 cache is `data/processed/deam_acoustic_51d.npz`; its 1,744 × 51 shape, unique IDs, ordered feature names, finiteness, and per-feature distributions were checked against the tracked Stage 1 diagnostics. The authoritative mapping and split were joined by Sample ID. The exact split remains 1,221 Train, 262 Validation, and 261 Test, with mutually exclusive roles. Module C C3's recorded label and split hashes still match, and authoritative C3/C4 result files match their committed Git blobs. MERT Layer 12 remains the later primary comparator.

The reusable Module C `fit_select_validate` routine was parameterized to accept 51 input dimensions while retaining its 768-dimensional default for Module C. It also optionally records MAE, R², and Pearson r for each candidate. No MERT experiment or A–D artifact was rerun or edited.

For each target independently, a `StandardScaler` fit on Train features alone transforms Train and Validation. The target stays on its original scale. Each of nine alpha candidates (`1e-4`, `1e-3`, `1e-2`, `1e-1`, `1`, `10`, `100`, `1000`, `10000`) uses Ridge with intercept and Cholesky solver, fit only on Train. Maximum Validation R² selects alpha; an actual exact tie would choose the larger alpha. No tie occurred. The constant reference predicts that target's Train mean on Validation. Final Train-only fitting is deferred to a later Test Stage.

## Complete Validation grid

The tracked JSON contains full precision MAE, R², and Pearson r for every candidate. Rounded R² values are shown here to make the selection path visible:

| Alpha | Valence Validation R² | Arousal Validation R² |
|---:|---:|---:|
| 0.0001 | 0.372391288 | 0.327096629 |
| 0.001 | 0.372392242 | 0.327097827 |
| 0.01 | 0.372401757 | 0.327109777 |
| 0.1 | 0.372494770 | 0.327226170 |
| 1 | 0.373259192 | 0.328146925 |
| 10 | 0.376571964 | **0.330553239** |
| 100 | **0.383931208** | 0.324064290 |
| 1000 | 0.373219964 | 0.306021073 |
| 10000 | 0.253864571 | 0.221169858 |

| Target | Selected alpha | Validation MAE | Validation R² | Validation Pearson r |
|---|---:|---:|---:|---:|
| Valence | 100 | 0.721685451 | 0.383931208 | 0.620778217 |
| Arousal | 10 | 0.794242460 | 0.330553239 | 0.584901982 |

| Train-mean reference | Train mean | Validation MAE | Validation R² | Pearson r |
|---|---:|---:|---:|---|
| Valence | 4.894692875 | 0.924825572 | −0.000046064 | undefined (`null`) |
| Arousal | 4.816953317 | 1.012681884 | −0.000018369 | undefined (`null`) |

These are Validation observations for alpha selection, not the final RQ3 comparison. Negative reference R² reflects use of the Train target mean rather than the Validation mean. Constant-reference Pearson r is undefined and is serialized as JSON `null`.

## Pre-Test verification

The pre-Test artifact records passing checks for exact cache shape/population, Sample-ID uniqueness and row-order-independent joins, split membership/disjointness, target alignment, Train-only scaler fit and model fit, unstandardized targets, Ridge configuration, frozen alpha grid and R² selection rule, target-specific selected alphas, Train-mean references, Stage 1 representation consistency, unchanged MERT comparator, and unchanged authoritative Module C artifacts. Only the 261 Test IDs were used for structural membership verification. The saved prediction CSV has exactly 524 rows: two targets × 262 Validation IDs, with no Test ID. Deterministic repeat runs matched exactly for both targets.

The independent `verify_acoustic_validation.py` check reopened the artifacts, recomputed selected and reference metrics from saved Validation predictions, checked targets against the Sample-ID mapping, recomputed Train target means, re-applied the alpha selection rule, confirmed all nine candidate metric records and Test-ID absence, and passed. No Test features were transformed by the scaler, and no Test predictions or metrics were generated.

## Files and exact commands

- `src/mert_emotion_probing/probing.py`: optional 51-D dimension and full candidate metric reporting while preserving Module C defaults.
- `scripts/run_acoustic_validation.py`: Stage 2 runner, ID alignment, input identities, Train/Validation selection, and pre-Test gate.
- `scripts/verify_acoustic_validation.py`: independent artifact and metric verification.
- `outputs/results/module_e_stage2_validation.json`: protocol, input hashes, complete grid, selected metrics and references.
- `outputs/results/module_e_stage2_validation_predictions.csv`: selected Validation predictions only.
- `outputs/results/module_e_stage2_pre_test_verification.json`: explicit closed-Test gate.

```powershell
& 'C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\python.exe' -m py_compile scripts/run_acoustic_validation.py scripts/verify_acoustic_validation.py src/mert_emotion_probing/probing.py
& 'C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\python.exe' scripts/run_acoustic_validation.py
& 'C:\Users\Rinshinozaki\miniconda3\envs\mert-emotion\python.exe' scripts/verify_acoustic_validation.py
```

## What I should now be able to explain

- Why changing from MERT Layer 12 to the 51-D acoustic representation retains the same target, split, Ridge family, scaler rule, alpha grid, and selection criterion.
- Why StandardScaler learns statistics only from 1,221 Train rows, while Validation selects alpha and Test remains closed.
- Why Valence and Arousal may choose different alphas, and why these choices came only from Validation R².
- Why a Train-mean reference has undefined Pearson r and can have slightly negative Validation R².
- Why these Validation numbers establish a frozen acoustic configuration, not an RQ3 performance conclusion or a unique-information claim.

There is no unresolved research-level change. **Ready for Module E Test gate review.**
