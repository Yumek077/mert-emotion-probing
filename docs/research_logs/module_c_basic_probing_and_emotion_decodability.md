# Module C — Basic Probing and Emotion Decodability

## Module Objective

Module C tested whether continuous musical-emotion targets can be decoded from the frozen MERT representations constructed and validated in Module B. It established a deliberately restricted linear probing protocol, selected its regularization strength using Validation only, protected the Test split until all choices were frozen, and produced the first held-out evidence for emotion decodability in this project.

## Research Question

**RQ1: Can continuous Valence and Arousal be decoded from frozen MERT representations?**

The operational question was narrower than whether MERT “understands” emotion: can a regularized linear model learn a mapping from the specified frozen representation to each continuous target and generalize to unseen samples?

## Starting Point from Module B

Module B supplied the canonical cache:

```text
outputs/embeddings/deam_mert_v1_95m_meanpool_all_layers.pt
representations: [1744, 13, 768]
sample_ids:      [1744]
```

The 1,744 samples are the frozen primary DEAM excerpt population. Each sample has a mean-pooled 768-dimensional vector for the Pre-Transformer representation and each of Transformer Layers 1–12. Module C read this cache directly and did not rerun or update MERT.

Module C also reused the frozen Sample-ID-based split in `data/metadata/deam_primary_split_seed42.csv`: 1,221 Train, 262 Validation, and 261 Test samples. Representations, targets, and split roles were joined by authoritative Sample ID rather than row position.

## Probing Protocol

Two independent supervised regressions were defined:

```text
frozen MERT Layer-12 representation -> continuous Valence
frozen MERT Layer-12 representation -> continuous Arousal
```

The primary representation was pre-specified as `transformer_layer_12`, resolved from cache metadata to representation index 12. Each probe received one 768-dimensional mean-pooled vector per sample. The probe family was Ridge Regression with an intercept and deterministic Cholesky solver. Input features were standardized with a `StandardScaler` fitted on Train only; targets remained on their original scale.

The predeclared Ridge alpha grid was:

```text
1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000
```

Alpha was selected separately for Valence and Arousal by maximum Validation R². An exact numerical tie would select the larger alpha. Performance was reported with MAE, R², and Pearson `r`. A constant Train-target-mean predictor supplied the minimal baseline.

## Why a Linear Ridge Probe?

A low-capacity linear probe asks whether target information is available in a simply readable form without allowing a powerful downstream model to dominate the result. Ridge preserves this linear interpretation while L2 regularization limits coefficient magnitude and improves stability with 768 potentially correlated features. Standardizing inputs makes one regularization strength act more comparably across feature dimensions.

This design tests linear decodability; it does not assume that every relationship between MERT and emotion is linear. Poor linear performance would not prove that no emotion information exists, while good performance does not imply human-like understanding.

## Data and Representation

- Population: 1,744 primary DEAM excerpts; the 58 full songs remain excluded under the frozen Module A definition.
- Targets: static continuous `valence_mean` and `arousal_mean`, modeled independently.
- Representation: frozen, temporally mean-pooled MERT Transformer Layer 12.
- Resolved cache index: 12.
- Input dimension: 768.
- Identity key: Sample ID.
- MERT parameters: unchanged; probing used the cached representations only.

## Train / Validation / Test Roles

Train learned the scaler statistics, Ridge coefficients, intercept, and Train-mean baseline value. Validation compared the nine alpha candidates and selected one alpha per target. Test remained outside preprocessing estimation, fitting, baseline construction, hyperparameter selection, and protocol design until Stage C4.

For the final held-out evaluation, each frozen model was recreated using only the 1,221 Train samples. Validation was not added to the final refit. The already selected configuration was then evaluated once on the 261 Test samples, with no post-Test tuning.

## Model Selection

Validation R² selected `alpha = 1000` independently for both targets. The matching choices were an empirical outcome rather than a constraint that the targets share an alpha. No exact tie occurred.

## Validation Results

These values supported model selection and pre-Test pipeline assessment; they are not the official held-out results.

| Target | Selected alpha | Validation MAE | Validation R² | Validation Pearson r |
|---|---:|---:|---:|---:|
| Valence | 1000 | 0.633006612 | 0.523872177 | 0.725022828 |
| Arousal | 1000 | 0.631327600 | 0.565970174 | 0.753173069 |

## Held-out Test Evaluation

After the configuration, input identities, data alignment, leakage controls, determinism, and absence of earlier Test predictions were verified, the protected Test gate was opened. The official Test results are:

| Target | Frozen alpha | Test MAE | Test R² | Test Pearson r |
|---|---:|---:|---:|---:|
| Valence | 1000 | 0.635144565 | 0.579780312 | 0.768855324 |
| Arousal | 1000 | 0.744556180 | 0.511421706 | 0.715509311 |

Valence had nearly unchanged absolute error and somewhat higher Test R² and correlation than on Validation. Arousal was weaker on Test than on Validation, with higher MAE and lower R² and correlation. This difference was retained as ordinary held-out variation and was not used to revise the protocol.

## Baseline Comparison

The Test baseline predicted each target's Train mean for every Test sample:

| Target | Train mean | Baseline Test MAE | Baseline Test R² | Pearson r |
|---|---:|---:|---:|---|
| Valence | 4.894692875 | 1.002563190 | -0.003113807 | undefined |
| Arousal | 4.816953317 | 1.131995632 | -0.000320133 | undefined |

Both Ridge probes had lower MAE and substantially higher R² than their Train-mean baselines. The baseline correlations are undefined because constant predictions have zero variance; the result artifacts record them as JSON `null`, not zero. Slightly negative baseline R² is valid because the constants came from Train rather than the Test target mean.

## Main Finding

Continuous Valence and Arousal are linearly decodable from the tested frozen mean-pooled MERT Layer-12 representations on the held-out DEAM Test split under the fixed Ridge probing protocol. Both targets achieved positive held-out R², positive Pearson correlation, and clear improvement over their Train-mean baselines.

## Interpretation

The result supports RQ1 within the defined population, representation, split, and probe. Because Test did not participate in scaling, fitting, alpha selection, or protocol design, the Test metrics provide stronger generalization evidence than the Validation observations alone. The lower Arousal Test performance relative to Validation should be remembered when judging stability, but it does not reverse the positive held-out result.

The finding concerns access to information through a restricted learned mapping. It does not by itself reveal why the representation supports the targets, whether the probe relies on emotion-specific structure or correlated acoustic cues, or whether a different representation level or feature family would perform better.

## Evidence Boundary

Module C does **not** establish that:

- MERT understands emotion in a human-like sense;
- Layer 12 is the best MERT representation level;
- all emotion information in MERT is linear;
- MERT representations outperform conventional audio features;
- the result generalizes to other datasets, annotation protocols, or full songs;
- the representation is free from acoustic confounds; or
- MERT contains a causal emotion mechanism.

These claims require separate research questions and controlled protocols. In particular, systematic representation-level comparison belongs to later RQ2 work and was not started in Module C.

## Reproducibility and Traceability

The implementation verifies exact split counts, unique and complete Sample-ID sets, metadata-based Layer-12 resolution, finite feature and target values, Train-only scaler statistics, Train-only model fitting, Validation-only alpha selection, Test isolation, and deterministic reruns. Saved predictions permit independent metric recomputation.

The Stage C3 artifact records the canonical cache, label table, and split identities, complete candidate Validation R² values, selected models, baselines, and pre-Test gate. The Stage C4 artifact records the frozen configuration, successful gate checks, official Test metrics, baselines, determinism, and confirmation that no post-Test tuning occurred.

## Frozen Module C Decisions

- Primary RQ1 representation: `transformer_layer_12`, cache index 12.
- Input: one 768-dimensional mean-pooled frozen representation per sample.
- Targets: continuous Valence and Arousal, modeled separately without target standardization.
- Probe: Ridge Regression with intercept and deterministic Cholesky solver.
- Input preprocessing: `StandardScaler` fitted on Train only.
- Split: the frozen 1,221/262/261 Train/Validation/Test assignment.
- Alpha grid: `1e-4` through `1e4` as predeclared above.
- Selection: maximum Validation R², with larger alpha on an exact tie.
- Selected alpha: 1000 for Valence and 1000 for Arousal.
- Metrics: MAE, R², and Pearson `r`.
- Baseline: constant prediction from the Train target mean.
- Final evaluation: Train-only refit followed by one held-out Test evaluation; no post-Test tuning.

## Artifacts and Source Files

- Stage records: `docs/codex_reports/module_c_stage1_probing_and_regression_foundations.md` through `module_c_stage4_held_out_test_evaluation.md`.
- Reusable probing implementation: `src/mert_emotion_probing/probing.py`.
- Stage C3 runner and verifier: `scripts/run_basic_probing.py`, `scripts/verify_basic_probing_results.py`.
- Stage C4 runner and verifier: `scripts/run_held_out_test_evaluation.py`, `scripts/verify_held_out_test_results.py`.
- Validation results: `outputs/results/module_c_stage3_validation.json` and `outputs/results/module_c_stage3_validation_predictions.csv`.
- Test results: `outputs/results/module_c_stage4_test.json` and `outputs/results/module_c_stage4_test_predictions.csv`.
- Frozen split: `data/metadata/deam_primary_split_seed42.csv`.
- Canonical cache: `outputs/embeddings/deam_mert_v1_95m_meanpool_all_layers.pt` (generated locally and excluded from Git).

## Module Outcome

**Module C passed and is complete.** RQ1 received positive held-out evidence under the frozen Layer-12 Ridge protocol. C1–C4 remain the detailed conceptual, protocol, implementation, validation, and Test history; this log records the coherent research conclusion. No new experiment was run during Module closure, no frozen Module A/B/C decision changed, and RQ2 was not started.
