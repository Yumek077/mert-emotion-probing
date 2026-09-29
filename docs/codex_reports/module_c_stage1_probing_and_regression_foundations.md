# Module C — Stage C1: Probing and Regression Foundations

## Stage Objective

Establish the conceptual foundation for Module C's basic probing experiment: what a probe tests, why the initial probe is linear, how continuous Valence and Arousal targets make this a regression problem, and why held-out evaluation and leakage control are necessary for defensible evidence about emotion decodability.

## Why This Stage Exists

Module B produced a frozen, probing-ready MERT representation dataset, but a representation cache alone does not show that emotion information is decodable. Before defining or implementing an experiment, the project needs a shared interpretation of what a probe can establish, how it should be evaluated, and what claims remain outside its scope. Stage C1 provides that foundation without making protocol choices that belong to Stage C2.

## Concepts Established

### Probe

A probe is a small supervised prediction model attached to an existing frozen representation. It tests whether a target property can be read out from that representation. During probing, MERT remains frozen: the probe learns a mapping from cached MERT features to emotion targets, while no MERT parameter is updated.

### Linear Probe

The planned basic experiment uses a deliberately low-capacity linear model. This does not assume that the true relationship between a representation and emotion is linear. Instead, restricting probe capacity tests whether emotion information is available in a simple, linearly readable form and reduces the risk that a powerful downstream model creates performance that is difficult to attribute to the representation itself.

Strong linear-probe performance would support linear decodability. It would not demonstrate human-like emotion understanding. Weak linear-probe performance would not prove that the representation contains no emotion information, because information may exist in a form that is not linearly accessible.

### Representation

MERT converts raw audio into internal numerical representations. Module C inherits Module B's canonical pooled representation cache rather than rerunning MERT. Each sample has 13 representation levels—the Pre-Transformer representation and Transformer Layers 1–12—and each level is a 768-dimensional feature vector.

### Regression

DEAM static Valence and Arousal are continuous targets, so basic probing is a supervised regression problem rather than a classification problem. Valence and Arousal are initially treated as two separate regression targets. A linear probe can be written as:

```text
y_hat = w^T x + b
```

Here, `x` is one frozen 768-dimensional MERT representation, and the probe learns `w` and `b`. MERT itself is not trained.

### Fitting

Fitting means learning the probe parameters from the training samples by optimizing a defined regression objective. Good fit on the training data alone is not sufficient evidence, because a probe can learn patterns that do not transfer beyond the samples used for parameter estimation.

### Generalization

The central empirical question is whether a representation-to-emotion relationship learned from training samples transfers to unseen samples. Held-out performance, rather than training fit alone, is therefore a core source of evidence for the later research question.

### Train / Validation / Test

Module C must reuse the fixed sample-ID-based split created in Module B:

| Split | Count | Role |
|---|---:|---|
| Train | 1,221 | Learn probe parameters and any data-dependent preprocessing parameters. |
| Validation | 262 | Select models, hyperparameters, or protocol choices where appropriate. |
| Test | 261 | Provide final held-out evaluation after design choices are fixed. |
| **Total** | **1,744** | |

The split must not be regenerated, reshuffled, or reconstructed from row order in Module C.

### Held-out Evaluation

Held-out evaluation measures performance on samples that were not used to fit the probe. The test set has a narrower role than the validation set: it is reserved for final evaluation after the relevant design choices have been fixed, and it must not be used to choose models, hyperparameters, preprocessing, or reporting protocol.

### Data Leakage

Data leakage occurs when information that should be unavailable during training or selection influences the learned model or protocol. In Module C:

- preprocessing parameters learned from data, such as standardization means and standard deviations, must be fitted using training data only;
- validation data may be used for model, hyperparameter, or protocol selection where appropriate;
- test data must not be used for selection;
- joins between representations, targets, and split membership must use the authoritative sample ID rather than row position; and
- information from one sample must not cross split boundaries.

## Connection to the Frozen Module B Dataset

Module C inherits the canonical cache:

```text
outputs/embeddings/deam_mert_v1_95m_meanpool_all_layers.pt
representations: [1744, 13, 768]
sample_ids:      [1744]
```

The dimensions mean 1,744 primary DEAM samples, 13 representation levels, and 768 features per pooled level. Module C must load this cache directly and preserve its explicit sample-ID alignment. It must also reuse:

```text
data/metadata/deam_primary_split_seed42.csv
```

The frozen counts are 1,221 train, 262 validation, and 261 test. Stage C1 changes none of Module B's decisions about the sample population, MERT checkpoint or frozen state, preprocessing, representation levels, pooling, identity mapping, cache, or split.

## Interpretation Boundaries

Stage C1 is conceptual and supplies no empirical evidence that:

- MERT can predict Valence or Arousal;
- any representation level is better than another;
- MERT representations outperform traditional audio features;
- emotion information is necessarily linear;
- prediction performance generalizes to held-out samples; or
- MERT has human-like musical emotion understanding.

Future probing results, if valid, should be interpreted as evidence about decodability under the tested probe and evaluation protocol. They should not be promoted into broader claims about understanding or causality.

## What I Should Now Be Able to Explain

After Stage C1, I should be able to explain:

- what a probe is and why MERT stays frozen;
- why a deliberately simple linear probe tests linear readability rather than assuming emotion is inherently linear;
- why continuous Valence and Arousal require regression and are initially modeled separately;
- what `x`, `w`, and `b` mean in `y_hat = w^T x + b`;
- the difference between fitting training data and generalizing to unseen samples;
- the distinct roles of train, validation, and test data;
- why data-dependent preprocessing must be fitted only on the training split;
- why test data cannot guide model or protocol selection;
- why sample ID, not row order, must align representations, targets, and split membership; and
- what a linear probing result would and would not justify claiming.

## Stage Outcome

- No experiment was run in this stage.
- No new empirical result was produced.
- No Module B frozen decision was changed.
- Stage C1 is complete.
- Stage C2 will define the actual basic probing protocol before implementation.
