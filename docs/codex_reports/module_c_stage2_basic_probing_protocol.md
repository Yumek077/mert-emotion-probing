# Module C — Stage C2: Basic Probing Protocol

## Stage Objective

Freeze the research protocol for Module C's primary basic probing experiment before any probe is implemented or evaluated. This stage defines the representation, targets, probe family, preprocessing, hyperparameter selection, metrics, baseline, test policy, reproducibility expectations, required artifacts, and scope boundaries needed to answer RQ1 without data leakage or result-driven redesign.

## Research Question

Primary RQ1 is:

> Can continuous Valence and Arousal be decoded from frozen MERT representations?

The experiment tests the **linear decodability** of emotion information under a deliberately restricted probe. It does not test or establish human-like musical emotion understanding.

## Decisions Frozen in This Stage

### C2.1 Primary Probe

The primary probe is **Ridge Regression**. Ridge remains a linear probe: for an input representation `x`, its prediction has the form `y_hat = w^T x + b`, while an L2 penalty controls the magnitude of the learned coefficients.

Ridge is selected because it keeps probe capacity low, is appropriate for a frozen 768-dimensional input, and can reduce instability or overfitting when features are numerous and correlated. The purpose is not to construct the strongest possible downstream predictor, but to test whether emotion information is available in a simple, stable, linearly readable form.

### C2.2 Input Standardization

Apply `StandardScaler` to the input representation `X` under the following rules:

- fit the scaler on the Train split only;
- use that same Train-fitted scaler to transform Train, Validation, and Test;
- never refit the scaler on Validation or Test; and
- keep Valence and Arousal on their original rating scale; do not standardize target `y`.

### C2.3 Targets

Static Valence and static Arousal are continuous targets. Train two independent regression probes:

```text
MERT representation -> Valence
MERT representation -> Arousal
```

Do not convert either target into a classification problem. The two targets do not have to share one regularization strength.

### C2.4 Hyperparameter Selection

Use this predeclared logarithmic Ridge grid:

```text
alpha in {
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1,
    10,
    100,
    1000,
    10000
}
```

For each target independently:

1. Fit `StandardScaler` on Train only.
2. Fit one candidate Ridge model per alpha using Train.
3. Evaluate the candidates on Validation.
4. Select alpha using Validation R² as the primary selection metric.
5. Keep Test completely outside alpha selection.

Do not alter the grid in response to Test results. Valence and Arousal may select different alphas because they are separate prediction problems.

If multiple alphas have exactly equal Validation R², the implementation must use a deterministic tie-breaking rule. The repository has no existing rule, so the exact rule is a small implementation detail that must be fixed before Stage C3 coding rather than chosen after observing results.

### C2.5 Evaluation Metrics

Report all three metrics for each target:

- **MAE:** absolute prediction error on the original emotion scale; lower is better.
- **R²:** predictive performance relative to a mean-prediction reference; higher is better, and values may be negative.
- **Pearson `r`:** linear association between predictions and targets; stronger positive values indicate more aligned variation, but high correlation does not imply small absolute error.

Validation R² is the alpha-selection metric. Test results must not be used to change either model selection or the reporting policy according to whichever metric looks most favorable.

### C2.6 Primary Representation Level

Pre-specify **Transformer Layer 12** as the primary representation for RQ1. It is the final Transformer representation of the MERT encoder and can be selected on architectural grounds before any probing result is known.

This choice does not assert that Layer 12 is the best MERT layer. The primary experiment must not compare Test performance across all 13 levels and then select the best result. Systematic comparison of the Pre-Transformer representation and Transformer Layers 1–12 belongs to the later RQ2 layer-wise probing work.

The canonical cache metadata orders its levels as `pre_transformer`, followed by `transformer_layer_1` through `transformer_layer_12`. Stage C3 must resolve and assert the Layer 12 index from this documented metadata and the verified Module B mapping; it must not rely on an unexplained hard-coded index.

### C2.7 Baseline

Use a trivial mean predictor. For each target, calculate the target mean from Train only and use that single value as the constant prediction for Validation and Test:

```text
Train target mean -> constant Validation/Test prediction
```

Do not use Validation or Test target means to construct the baseline. A handcrafted or conventional audio-feature baseline is outside Module C and is reserved for RQ3.

Because a constant prediction has zero variance, Pearson `r` for the mean baseline is mathematically undefined. Stage C3 must represent this transparently, for example as a documented null/NaN value, rather than inventing a correlation value. This does not affect the requirement to report the probe's MAE, R², and Pearson `r`.

### C2.8 Held-out Test Policy

Test is the final held-out evaluation. Before the first formal Test evaluation, complete and record:

- cache, label, and split alignment verification;
- correct Layer 12 selection verification;
- StandardScaler leakage verification;
- Train/Validation pipeline verification;
- alpha selection;
- mean-baseline verification;
- required numerical and structural sanity checks; and
- protocol freeze.

Test must not influence preprocessing, alpha selection, model choice, representation-level choice, metric choice, or reporting policy. After the first formal Test evaluation, poor results are not a reason to revise the protocol and present a later run as an untouched primary result.

If a genuine implementation bug invalidates a Test evaluation, a corrected rerun is allowed only with a traceable record of what was wrong, why the earlier result was invalid, what changed, and why rerunning was necessary.

### C2.9 Reproducibility and Determinism

The primary pipeline does not require multi-seed averaging. It uses frozen cached representations, a fixed split, deterministic preprocessing, and should prefer a deterministic linear-regression procedure. Artificially repeating the same deterministic pipeline with multiple seeds would not add neural-network-style robustness evidence.

Stage C3 must record the actual Ridge solver behavior, relevant library defaults, and software versions. If any chosen solver can introduce nondeterminism, that behavior must be identified and controlled or documented before Test evaluation.

### C2.10 Required Result Artifacts

The eventual Stage C3/C4 probing pipeline must save at least:

- protocol and configuration;
- target name;
- representation level;
- alpha grid;
- selected alpha;
- selection metric;
- Validation metrics;
- Test metrics;
- baseline metrics;
- predictions aligned with Sample ID; and
- relevant provenance and software versions.

The exact JSON/CSV layout is not frozen in Stage C2. Stage C3 should choose the smallest format consistent with the repository's existing conventions: structured JSON for protocol, verification, metrics, and provenance where appropriate, and a tabular CSV for Sample-ID-aligned predictions where appropriate. Generated experiment artifacts belong under the existing `outputs/results/` hierarchy; no new top-level documentation or output hierarchy is required.

### C2.11 Uncertainty Analysis

The primary Module C experiment reports point estimates only:

- MAE;
- R²; and
- Pearson `r`.

Bootstrap confidence intervals and other statistical uncertainty analyses are possible later enhancements, not requirements of the primary pipeline.

### C2.12 Scope Boundary

Module C addresses basic emotion decodability and RQ1 only. It does not include:

- systematic 13-level comparison;
- best-layer selection;
- nonlinear probes;
- conventional audio-feature comparison;
- tempo or energy confound analysis;
- bootstrap confidence intervals;
- statistical significance testing; or
- MERT fine-tuning.

These belong to later research questions or extensions.

## Why These Decisions Were Made

The protocol deliberately separates representation quality from downstream model capacity. A regularized linear model asks whether emotion information is simply readable from a frozen representation while reducing coefficient instability in a feature space with 768 dimensions. Train-only standardization makes the regularization penalty meaningful across feature dimensions without allowing held-out distribution information into fitting.

Pre-specifying Layer 12 avoids result-driven representation selection in RQ1. Independent Valence and Arousal probes preserve the continuous targets and allow their optimal regularization strengths to differ. Validation-only alpha selection separates model selection from final Test evidence. Reporting MAE, R², and Pearson `r` provides complementary views of absolute error, performance relative to a mean reference, and aligned variation. The Train-mean baseline supplies a minimal reference without introducing another learned audio representation.

Together, these decisions keep the primary experiment narrow, interpretable, reproducible, and resistant to leakage or post-hoc optimization.

## Data Leakage Controls

Stage C3 must enforce all of the following before Test is touched:

- fit `StandardScaler` only on Train features;
- apply the unchanged Train scaler to Validation and Test;
- fit Ridge candidates only on Train samples;
- select alpha only from Validation R²;
- compute the baseline constant only from the Train target mean;
- exclude Test targets and metrics from all model, hyperparameter, preprocessing, layer, and protocol decisions;
- align cache rows, labels, and split membership by Sample ID rather than row position;
- verify unique IDs, complete set equality, allowed split values, exact frozen counts, and no cross-split duplication; and
- keep Valence and Arousal selection processes independent.

Any preprocessing or selection statistic derived from Validation or Test beyond the roles specified above would invalidate the primary protocol.

## Connection to Module B

Module C must load the Module B canonical cache directly:

```text
outputs/embeddings/deam_mert_v1_95m_meanpool_all_layers.pt
```

Its expected structure is:

```text
representations: [1744, 13, 768] float32
sample_ids:      [1744] int64
metadata:        cache provenance and ordered representation-level definitions
```

Module C must not rerun MERT. It must reuse the frozen split:

```text
data/metadata/deam_primary_split_seed42.csv
```

with 1,221 Train, 262 Validation, and 261 Test samples. Sample ID is the authoritative key connecting cached representations, static `valence_mean` and `arousal_mean` targets, and split membership. Row order is not an identity mechanism.

Stage C2 changes none of Module B's frozen decisions about the sample population, preprocessing, checkpoint, model state, representation definitions, pooling, cache, identity mapping, or split.

## What I Should Now Be Able to Explain

After Stage C2, I should be able to explain the following in research terms:

- Ridge is used because it remains a low-capacity linear probe while regularization can stabilize learning from 768 potentially correlated features.
- StandardScaler puts input dimensions on comparable scales before Ridge applies one coefficient penalty, but it must learn its statistics from Train only to prevent leakage.
- Valence and Arousal remain separate continuous regression targets because they represent distinct dimensions and may require different mappings and alpha values.
- Alpha controls Ridge regularization strength: larger values penalize coefficient magnitude more strongly, while smaller values approach less-regularized linear regression.
- Alpha is selected on Validation because hyperparameter selection is a design decision; using Test would contaminate the final held-out evidence.
- Valence and Arousal may select different alphas because their target relationships and noise characteristics need not be identical.
- MAE measures average absolute error on the original rating scale, R² compares predictive performance with a mean-reference prediction and can be negative, and Pearson `r` measures aligned linear variation without guaranteeing good calibration or small errors.
- All three metrics are reported because none alone captures absolute accuracy, relative predictive value, and association.
- Layer 12 is pre-specified because it is the final encoder representation and can be chosen without observing probing outcomes.
- Pre-specifying Layer 12 does not mean it is the best layer; answering that question requires a separately controlled 13-level comparison under RQ2.
- The mean baseline asks whether the probe improves on a trivial constant predictor, and its value must come from Train only so held-out targets do not influence it.
- Test must remain untouched until alignment, layer selection, scaling, selection, baseline, sanity checks, and protocol freeze are complete.
- Linear probing can support the claim that emotion information is linearly decodable under this protocol. It cannot prove that all emotion information is linear, that weak performance means no information exists, or that MERT has human-like emotion understanding.

## Stage Outcome

- The primary RQ1 basic probing protocol is frozen as specified above.
- No probe was implemented or fitted in this stage.
- No Validation or Test probing performance was inspected.
- No empirical probing result was produced.
- No Module A or Module B frozen decision was changed.
- The deterministic exact-tie rule for Validation R² remains a small implementation detail to fix before Stage C3 coding.
- Exact result-file serialization and Ridge solver/default details remain engineering choices for Stage C3 and must be documented before formal Test evaluation.
- Stage C2 is complete. Stage C3 will implement and verify this protocol without broadening its scope.
