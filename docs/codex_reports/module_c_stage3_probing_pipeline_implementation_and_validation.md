# Module C — Stage C3: Probing Pipeline Implementation and Validation

## Stage Objective

Implement and validate the complete pre-Test pipeline for the primary RQ1 experiment: assemble the frozen Module B cache with DEAM static targets and the fixed split by authoritative Sample ID, resolve Transformer Layer 12 from cache metadata, fit Train-only standardized Ridge candidates, select alpha independently for Valence and Arousal using Validation R², evaluate the selected models and Train-mean baselines on Validation, and save reproducible artifacts without generating or inspecting held-out Test predictions or performance.

## Repository Context Recovered

Before implementation, the following repository context was inspected:

- `docs/project_workflow.md`, including the coherent-stage, minimal-sufficient-rigor, decision-locking, and technical-report principles;
- the Module B final research log and its frozen cache, representation, split, and identity decisions;
- Module C Stage C1's probing and generalization foundations;
- Module C Stage C2's frozen Ridge, scaling, selection, metric, baseline, and Test policies;
- the existing `src/mert_emotion_probing/` and `scripts/` organization;
- the Module B extraction script's cache and protocol-table validation logic;
- the generated DEAM item mapping, frozen split artifact, canonical cache metadata, output directories, `.gitignore`, environment definition, and Git status.

The existing source organization was sufficient. No new package hierarchy, experiment framework, dependency, representation extraction, or split generation was introduced.

## Frozen Protocol Implemented

The implementation follows the frozen Stage C2 protocol:

- Research target: linear decodability of continuous static Valence and Arousal, not human-like understanding.
- Primary representation: `transformer_layer_12`, resolved from cache metadata.
- Separate Valence and Arousal Ridge regressions.
- `StandardScaler` fitted only on Train features; targets remain on the original rating scale.
- Alpha grid: `{1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000}`.
- Selection metric: Validation R².
- Exact-tie rule: when actual computed Validation R² values are exactly equal, choose the larger alpha; no tolerance-based tie is used.
- Metrics: MAE, R², and Pearson `r`.
- Baseline: constant prediction from the Train target mean.
- Fixed split: 1,221 Train, 262 Validation, and 261 Test samples.
- Test boundary: Test membership may be checked for integrity, but Stage C3 generates no Test predictions or performance.

## Files Created or Modified

Created:

- `src/mert_emotion_probing/probing.py`
- `scripts/run_basic_probing.py`
- `scripts/verify_basic_probing_results.py`
- `outputs/results/module_c_stage3_validation.json`
- `outputs/results/module_c_stage3_validation_predictions.csv`
- `docs/codex_reports/module_c_stage3_probing_pipeline_implementation_and_validation.md`

No Module A/B research log or historical Stage report, Module C Stage C1/C2 protocol record, split artifact, canonical cache, dependency definition, or MERT extraction source was changed.

## Data Assembly

The canonical inputs were:

- cache: `outputs/embeddings/deam_mert_v1_95m_meanpool_all_layers.pt`;
- labels: `data/raw/deam/verification/deam_item_mapping.csv`;
- split: `data/metadata/deam_primary_split_seed42.csv`.

The cache was loaded on CPU and required to contain a dictionary with `representations`, `sample_ids`, and `metadata`. Validation confirmed:

- representation shape `[1744, 13, 768]`;
- representation dtype `torch.float32`;
- sample-ID shape `[1744]` and dtype `torch.int64`;
- 1,744 unique cache IDs;
- finite cache values;
- selected Layer-12 shape `[1744, 768]`;
- finite Layer-12 values and no all-zero sample row.

The label table contained 1,802 unique DEAM items. Its actual schema included `song_id`, `is_full_song`, `valence_mean`, and `arousal_mean`. The implementation normalized `song_id` to the common integer identity type, explicitly excluded the 58 `is_full_song` rows, and required the remaining 1,744 primary IDs to equal the cache ID set exactly. It did not select the first 1,744 rows or depend on label-table order. Both target columns were complete and finite for the primary population.

The split table was required to contain exactly `sample_id,split`. It contained 1,744 unique IDs, only the allowed roles, the exact frozen counts, and no cross-role overlap. Its ID set equaled both the cache and primary-label ID sets.

## Sample-ID Alignment

`sample_id` in the cache and split and `song_id` in the label mapping were treated as the same authoritative DEAM item identity after explicit integer normalization. Assembly used ID-indexed lookups and a deterministic sorted ID sequence; it never assumed that source row positions matched.

As a direct row-order-invariance check, labels and split rows were reversed and reassembled by ID. The targets and split memberships were unchanged. The final joined dataset contained exactly 1,744 unique primary samples, no excluded full songs, no missing targets, and the expected 1,221/262/261 roles.

## Layer 12 Resolution

The cache metadata contained this ordered representation list:

```text
pre_transformer
transformer_layer_1
...
transformer_layer_12
```

The implementation required the entire list to match the frozen Module B mapping, looked up `transformer_layer_12` by name, and resolved it to index `12`. It then asserted the selected matrix shape `[1744, 768]`. An absent, reordered, renamed, or otherwise inconsistent metadata mapping raises a protocol violation rather than silently falling back to a hard-coded index.

## Probe Implementation

Reusable assembly, metric, selection, and verification functions were added under the existing package. The CLI orchestrator loads the frozen inputs, assembles the dataset, creates target-specific Train/Validation views, runs the fixed alpha grid, saves Validation-only artifacts, and records provenance.

Each candidate uses `sklearn.linear_model.Ridge` with:

- `solver="cholesky"`;
- `fit_intercept=True`;
- only the 1,221 Train rows for fitting.

The Cholesky solver was chosen as a deterministic dense linear-algebra implementation suitable for this fixed 1,221-by-768 Train design and strictly positive alpha grid. No random seed or multi-seed loop is needed for this deterministic procedure.

Trained model and scaler binaries were not serialized. The selected alphas, exact protocol, input hashes, software versions, and validation artifacts are saved; Stage C4 can deterministically recreate the same Train-only scaler and selected Ridge fit before the protected Test evaluation.

## Standardization and Leakage Controls

For each target, `StandardScaler` was fitted on Train `X` only and then used unchanged to transform Train and Validation. Verification required:

- `n_samples_seen_ == 1221`;
- learned scaler means equal the direct Train-feature means;
- learned scaler variances equal the direct Train-feature variances;
- finite transformed Train and Validation arrays;
- no target standardization.

The model-selection function receives only explicit Train and Validation arrays. Test features and targets are not passed to scaling, Ridge fitting, prediction, metric, baseline, or selection functions. Every Ridge candidate fits only Train `X/y`; Validation is used only for candidate evaluation and selection. The baseline constant is computed only from Train `y`.

## Alpha Search and Selection

All nine frozen alpha values were evaluated independently for each target. Candidate Validation R² values were:

| Alpha | Valence Validation R² | Arousal Validation R² |
|---:|---:|---:|
| 0.0001 | -0.151212671 | -0.276281084 |
| 0.001 | -0.151038805 | -0.276065113 |
| 0.01 | -0.149309376 | -0.273915088 |
| 0.1 | -0.132879241 | -0.253329665 |
| 1 | -0.021220246 | -0.107488433 |
| 10 | 0.256374086 | 0.255105386 |
| 100 | 0.461313941 | 0.516536938 |
| 1000 | **0.523872177** | **0.565970174** |
| 10000 | 0.457924946 | 0.475422375 |

Both targets selected `alpha = 1000`. They did so independently; the matching values are an observed outcome, not a shared-alpha constraint. Neither target had an exact tie in the real Validation search.

The tie rule was verified separately with synthetic candidate-score records. Equal computed R² values selected the larger alpha, while values differing by the smallest tested floating-point step were not treated as tied.

## Validation Results

Selected-model Validation results were:

| Target | Selected alpha | MAE | R² | Pearson r |
|---|---:|---:|---:|---:|
| Valence | 1000 | 0.633006612 | 0.523872177 | 0.725022828 |
| Arousal | 1000 | 0.631327600 | 0.565970174 | 0.753173069 |

These are Validation results used for model selection and pipeline verification. They are not held-out Test results and are not yet final generalization evidence.

## Mean Baseline

Each target's baseline predicted its Train target mean for every Validation sample:

| Target | Train target mean | Validation MAE | Validation R² | Pearson r |
|---|---:|---:|---:|---|
| Valence | 4.894692875 | 0.924825572 | -0.000046064 | undefined (`null`) |
| Arousal | 4.816953317 | 1.012681884 | -0.000018369 | undefined (`null`) |

The small negative R² values are valid because the baseline uses the Train mean rather than the Validation mean. Pearson `r` is mathematically undefined for a constant prediction with zero variance. It is serialized as JSON `null` with an explicit status and explanation, never fabricated as zero.

## Verification and Sanity Checks

The following checks passed:

- source files compiled in the project environment;
- cache schema, shape, dtype, IDs, finiteness, and Layer-12 rows;
- label schema, primary-population filtering, uniqueness, completeness, and finiteness;
- split schema, values, counts, uniqueness, non-overlap, and complete set equality;
- ID-based assembly and reversed-source-row invariance;
- exact `[1744, 768]` Layer-12 assembly;
- exact Train/Validation shapes and finite inputs/targets;
- Train-only scaler sample count, mean, and variance;
- exact frozen alpha grid;
- Validation-R²-only selection and exact-tie rule;
- independent target selection;
- Train-only mean baseline and undefined constant-prediction Pearson handling;
- an exact repeated run for each target, including selected alpha, all candidate R² values, selected metrics, and Validation predictions;
- independent recomputation of selected Validation metrics and baseline metrics from the saved prediction CSV;
- 524 prediction rows: 262 Validation IDs for each of two targets;
- prediction ID set exactly equal to the frozen Validation ID set;
- no Test ID in the prediction artifact.

The independent artifact verifier completed with `passed: true`.

## Output Artifacts

`outputs/results/module_c_stage3_validation.json` stores:

- frozen protocol and resolved representation index;
- SHA-256 identities, paths, and sizes for the cache, labels, and split;
- complete data-assembly verification;
- the alpha grid and every candidate Validation R²;
- selected alpha and selected-model Validation metrics per target;
- Train-mean baseline values and Validation metrics;
- scaling, Ridge, leakage, determinism, software, and pre-Test-gate records.

`outputs/results/module_c_stage3_validation_predictions.csv` contains 524 Sample-ID-aligned long-format rows with only Validation predictions: one row per Validation sample and target. It contains the target, representation name/index, selected alpha, true Validation target, selected-model prediction, and Train-mean baseline prediction. It contains no Test rows or Test predictions.

Strict JSON serialization uses `allow_nan=False`; undefined Pearson values are represented as standards-compliant `null` with an explanatory status.

## Determinism and Software Environment

The recorded environment was:

- Python 3.10.21;
- NumPy 2.2.6;
- pandas 2.3.3;
- SciPy 1.15.3;
- scikit-learn 1.7.2;
- PyTorch 2.14.0+cu130;
- Windows 10 build 26200 as reported by Python.

The actual pipeline was run twice after final code changes. Within each invocation, every target was fitted and selected a second time and required to match exactly. Both targets produced identical selected alpha, candidate R² sequence, selected metrics, and Validation prediction arrays.

## Problems Encountered and Solutions

No substantive conflict or STOP condition was found.

Ordinary implementation issues were handled as follows:

- cache/split `sample_id` and label-table `song_id` were explicitly normalized to the same integer DEAM identity and validated as one-to-one;
- the 1,802-row label table was filtered by the existing `is_full_song` definition and then required to match the 1,744 cache IDs exactly;
- Layer 12 was resolved from the metadata name and checked against the complete frozen ordered mapping;
- constant-input Pearson was represented as `null` with a status rather than zero or non-standard JSON NaN;
- atomic JSON/CSV replacement prevents partially written final artifacts;
- a quiet CLI option supports rerunning without emitting the complete result JSON.

None of these solutions changed the dataset, targets, representation, split, probe family, alpha grid, selection metric, or interpretation.

## Protocol Consistency

The implementation is consistent with the frozen Module A/B/C decisions. It did not rerun MERT, regenerate representations, regenerate or modify the split, remove samples, inspect alternative representation levels, add another probe or baseline, standardize targets, change the alpha grid, add cross-validation, or introduce uncertainty analysis.

No frozen research decision changed in Stage C3.

## Test-Gate Confirmation

Test IDs and membership were read only to verify population completeness, exact count, ID-set alignment, and absence of split overlap. The Test role contained 261 IDs as expected.

Stage C3 did **not**:

- generate Test predictions;
- compute Test MAE, R², or Pearson `r`;
- inspect Test prediction quality;
- use Test data for scaling, fitting, baseline construction, alpha selection, or protocol modification; or
- save any Test prediction/performance artifact.

The saved gate record explicitly marks Test predictions, computation, and inspection as false. Stage C4 must remain a separate, deliberate held-out evaluation after researcher review of this report and the artifacts.

## What I Should Now Be Able to Explain

- The C3 pipeline starts from the frozen Module B cache, resolves Layer 12 from metadata, joins targets and split roles by Sample ID, separates Train and Validation, fits Train-only scaling and Ridge candidates, selects alpha using Validation R², compares the selected model with a Train-mean baseline, verifies determinism, and saves Validation-only results.
- Sample-ID assembly is necessary because cache, labels, and split files can have different row orders and even different identity-column names. Row position is not a stable sample identity.
- Layer 12 mapping must be verified because an off-by-one error would silently probe the wrong representation and invalidate the research interpretation.
- `StandardScaler` must fit only Train because using Validation or Test feature statistics would leak held-out distribution information into model fitting.
- Ridge alpha controls L2 regularization strength. Valence and Arousal can select different alphas because they are separate regression targets with potentially different representation-to-target relationships.
- Validation supports hyperparameter selection and implementation assessment. It is intentionally no longer untouched after selection, so its performance is not final held-out generalization evidence.
- The Train-mean baseline asks whether the probe improves on a trivial constant predictor without using held-out target means.
- MAE measures absolute error on the original rating scale; R² compares prediction against a mean-reference model and can be negative; Pearson `r` measures aligned linear variation but not error magnitude or calibration.
- The Test gate should open only after identity, layer mapping, leakage controls, candidate selection, baseline behavior, artifacts, and determinism have all been verified. C3 supplies that precondition but does not itself open the gate.
- C3 shows that the frozen protocol is implementable and that Layer-12 representations have positive Validation decodability under the selected Ridge models. It does not yet provide final held-out Test evidence, establish the best MERT layer, compare MERT with conventional features, or demonstrate human-like emotion understanding.

## Stage Outcome

**Stage C3 passed.** The complete Layer-12 Train/Validation Ridge probing pipeline was implemented and independently verified. Valence and Arousal each selected `alpha = 1000` from the frozen grid. Validation artifacts are complete, ID-aligned, reproducible, and free of Test predictions. No frozen research decision changed.

The repository is ready for researcher review at the held-out Test gate. Stage C4 was not started, and no Test probing performance was computed, inspected, or used.
