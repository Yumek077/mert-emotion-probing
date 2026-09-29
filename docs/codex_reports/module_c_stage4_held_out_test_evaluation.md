# Module C — Stage C4: Held-out Test Evaluation

## Stage Objective

Open the protected Test gate exactly once under the frozen Stage C3 configuration, refit the two selected Layer-12 Ridge probes using Train only, evaluate them on the 261 previously untouched Test samples, compare them with the frozen Train-mean baselines, save Sample-ID-aligned Test artifacts, and interpret the results without any post-Test tuning.

The primary research question is:

> Can continuous Valence and Arousal be decoded from frozen MERT representations?

## Repository Context Recovered

Before the Test gate was opened, the current repository state was recovered from:

- `docs/project_workflow.md`;
- the Module B final research log;
- Module C Stage C1 and C2 records;
- the Stage C3 implementation and validation report;
- `src/mert_emotion_probing/probing.py` and the Stage C3 runner/verifier;
- `outputs/results/module_c_stage3_validation.json` and its Validation prediction artifact;
- the canonical representation cache, DEAM label mapping, fixed split, environment specification, and Git status.

The authoritative records agreed: Stage C3 passed, the primary representation was Transformer Layer 12 at resolved index 12, the probe was Ridge Regression with the deterministic Cholesky solver, both targets selected `alpha = 1000`, inputs were standardized using a Train-only `StandardScaler`, targets were unstandardized, the frozen split was 1,221/262/261, and Test performance had not previously been generated or inspected.

## Frozen Configuration

Stage C4 used exactly:

| Component | Frozen value |
|---|---|
| Representation | `transformer_layer_12` |
| Resolved cache index | 12 |
| Per-sample input | 768 dimensions |
| Probe | Ridge Regression |
| Solver | `cholesky` |
| Valence alpha | 1000 |
| Arousal alpha | 1000 |
| Input scaling | `StandardScaler`, fit on Train only |
| Target scaling | None |
| Final fit population | 1,221 Train samples only |
| Test population | 261 Test samples |
| Metrics | MAE, R², Pearson `r` |
| Baseline | Constant Train-target mean |

Validation was not added to the final refit. No configuration was selected, revised, or expanded after Test results became available.

## Pre-Test Gate Verification

The gate was evaluated before any Test prediction. All checks passed:

- the Stage C3 artifact identified itself as Module C Stage C3;
- Stage C3's representation name/index, probe, solver, alpha grid, selected alphas, scaling policy, target policy, and split counts matched the frozen C4 configuration;
- Stage C3 determinism checks had passed for both targets;
- Stage C3 explicitly recorded no prior Test predictions, metrics, or inspection;
- the current cache, label mapping, and split sizes and SHA-256 hashes exactly matched those recorded by Stage C3;
- the current data assembly again produced `[1744, 768]` finite Layer-12 features aligned to 1,744 unique IDs;
- the split contained exactly 1,221 Train, 262 Validation, and 261 Test IDs;
- Test contained exactly 261 unique IDs;
- Train/Test and Validation/Test overlaps were both zero;
- Layer 12 again resolved from metadata to index 12;
- all primary targets were complete and finite.

The Test gate opened only after these checks completed successfully.

## Final Train-only Refit

For each target independently:

1. The frozen 1,221 Train rows were selected by Sample ID.
2. `StandardScaler` was fitted only on Train `X`.
3. The fitted scaler transformed Train `X` and Test `X`.
4. Ridge with frozen `alpha = 1000`, `solver="cholesky"`, and an intercept was fitted only on Train `X/y`.
5. Predictions were generated for the 261 Test rows.
6. MAE, R², and Pearson `r` were calculated once under the frozen protocol.

Scaler verification again confirmed `n_samples_seen_ = 1221` and exact agreement of its stored mean and variance with direct Train-feature calculations. Validation rows were not included in scaler or Ridge fitting. Test targets were used only after prediction to calculate the authorized final metrics.

## Held-out Test Results

| Target | Frozen alpha | Test MAE | Test R² | Test Pearson r |
|---|---:|---:|---:|---:|
| Valence | 1000 | 0.635144565 | 0.579780312 | 0.768855324 |
| Arousal | 1000 | 0.744556180 | 0.511421706 | 0.715509311 |

Both targets produced positive held-out R² and positive Pearson correlation under the frozen Layer-12 linear probe. These are the primary held-out results for RQ1.

## Mean Baseline Test Results

The frozen baseline used each target's Train mean as a constant Test prediction:

| Target | Train target mean | Baseline Test MAE | Baseline Test R² | Pearson r |
|---|---:|---:|---:|---|
| Valence | 4.894692875 | 1.002563190 | -0.003113807 | undefined (`null`) |
| Arousal | 4.816953317 | 1.131995632 | -0.000320133 | undefined (`null`) |

The baseline's small negative R² values are valid because its constant is the Train mean rather than the Test mean. Its Pearson `r` is mathematically undefined because the predictions have zero variance; it is serialized as JSON `null` with an explicit status and explanation.

For both targets, the frozen Ridge probe had lower MAE and substantially higher R² than the Train-mean baseline on Test.

## Validation vs Test Context

| Target | Split | MAE | R² | Pearson r |
|---|---|---:|---:|---:|
| Valence | Validation | 0.633006612 | 0.523872177 | 0.725022828 |
| Valence | Test | 0.635144565 | 0.579780312 | 0.768855324 |
| Arousal | Validation | 0.631327600 | 0.565970174 | 0.753173069 |
| Arousal | Test | 0.744556180 | 0.511421706 | 0.715509311 |

Valence was highly consistent in absolute error, with slightly higher Test R² and correlation. Arousal was weaker on Test than on Validation: MAE increased and both R² and correlation decreased. This difference is recorded as ordinary held-out variation, not used as a reason to revise alpha or any other protocol choice.

At the research-question level, Validation and Test are broadly consistent: both targets retain positive R², positive correlation, and clear improvement over their Train-mean baselines. The Arousal decrease is a limitation to acknowledge, not a tuning signal.

## Output Artifacts

Created:

- `outputs/results/module_c_stage4_test.json`
- `outputs/results/module_c_stage4_test_predictions.csv`

The JSON contains:

- the frozen configuration;
- Stage C3 and input artifact paths, sizes, and SHA-256 provenance;
- the complete pre-Test gate record;
- Test metrics and Train-mean baseline metrics for each target;
- copied Stage C3 Validation metrics for context;
- leakage, no-selection, and no-post-Test-tuning confirmations;
- deterministic rerun checks and software versions.

The CSV contains exactly 522 rows: 261 frozen Test IDs for each of two targets. Each row records Sample ID, Test role, target, representation name/index, frozen alpha, true Test target, Ridge prediction, and Train-mean baseline prediction. No Train or Validation prediction row is present.

Strict JSON serialization represents undefined correlations as standards-compliant `null`; non-standard JSON NaN values are not emitted.

## Verification and Determinism

The following checks passed after evaluation:

- source compilation;
- exactly 522 prediction rows and 261 unique Test IDs;
- exact equality between prediction IDs and the frozen Test ID set;
- zero Train/Validation IDs in the Test prediction artifact;
- one unique row per Sample-ID/target pair;
- frozen alpha 1000 in every row for both targets;
- independent recomputation of Test MAE, R², and Pearson `r` from the saved CSV;
- independent recomputation of baseline Test MAE and R²;
- constant-baseline Pearson stored as `null` rather than zero;
- exact deterministic repetition of each Train-only scaler/refit, Test prediction vector, Test metrics, baseline vector, and baseline metrics.

The independent artifact verifier completed with `passed: true`.

## Protocol Consistency

No frozen research decision changed. Stage C4 did not:

- retune alpha;
- change layer, scaler, probe, solver, metrics, baseline, target, sample, or split;
- include Validation in final fitting;
- remove any sample;
- inspect another MERT level;
- add another model, baseline, confidence interval, significance test, seed, or cross-validation procedure; or
- use Test as a model-selection signal.

No post-Test tuning occurred.

## Interpretation

The held-out results provide evidence that continuous static Valence and Arousal are linearly decodable from the frozen MERT Layer-12 item representations under this fixed Ridge protocol. The evidence is stronger than a Validation-only observation because the Test samples did not participate in preprocessing estimation, fitting, alpha selection, or protocol design.

Both probes improve materially over a trivial Train-mean predictor, and both have positive held-out R² and correlations. This supports RQ1 for the defined DEAM excerpt population, fixed split, frozen Layer-12 mean-pooled representations, and selected linear-probe protocol.

The results do not establish that:

- Layer 12 is the best MERT representation level;
- all emotion information in MERT is linear;
- MERT representations outperform conventional audio features;
- the relationship generalizes to full songs, another dataset, or another annotation protocol;
- the predictions are free from acoustic confounds;
- MERT has a causal or explicit emotion mechanism; or
- MERT understands emotion in a human-like sense.

Those claims require separate research questions and protocols.

## What I Should Now Be Able to Explain

- C3 had to complete model selection, leakage checks, identity alignment, Layer-12 verification, artifact generation, and deterministic validation before Test could be opened without contaminating the final evidence.
- Validation was used to choose alpha; Test was held back to evaluate the already frozen choice. Test results therefore cannot be fed back into alpha, layer, preprocessing, or protocol selection.
- The C4 models learned from exactly the 1,221 Train samples. `StandardScaler` and Ridge were both fitted on Train only; Validation was not included in the final refit.
- Test MAE measures average absolute error on the original rating scale, Test R² measures improvement relative to a Test-mean reference and can be negative, and Test Pearson `r` measures aligned variation without guaranteeing small errors or calibration.
- The Train-mean baseline asks whether the learned representation-to-target mapping outperforms a trivial constant that uses no Test target information.
- Validation/Test differences are expected sampling variation unless there is evidence of an implementation problem. They should be reported and interpreted, not used for post-Test tuning.
- The positive held-out R² values, correlations, and baseline improvements support linear decodability of Valence and Arousal from the specified frozen Layer-12 representations.
- Even strong held-out performance would not prove best-layer status, causal emotion encoding, robustness to other populations, superiority to other features, or human-like understanding.

## Stage Outcome

**Stage C4 passed.** The protected Test gate was opened only after all frozen-configuration and artifact-identity checks passed. Train-only Ridge probes with frozen `alpha = 1000` produced the official held-out Test results for Valence and Arousal, both outperforming the Train-mean baselines. The Test artifacts were independently recomputed and deterministic reruns matched exactly.

No protocol conflict and no post-Test tuning occurred. Stage C4 is complete. No layer-wise probing, RQ2/RQ3 work, Module C final research log, personal note, commit, or push was started.
