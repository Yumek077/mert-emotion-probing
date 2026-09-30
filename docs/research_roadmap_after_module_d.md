# Post-Module-D Research Roadmap

## 1. Status, Purpose, and Authority

**Frozen planning document — 2026-09-30.** Modules A–D, RQ1, and RQ2 are complete. Module D implementation, verification, ChatGPT research interpretation review, and the researcher learning checkpoint are complete, and its protocol/results are accepted for project use. Module D was committed and pushed to `main` as `868579f52eac6814d16765698e8d7c69505242bd` (`Complete Module D layer-wise emotion decodability analysis`); remote `main` was checked against that commit during this roadmap task.

**Next research task: Module E design / RQ3.** This document freezes the remaining research route and its scope boundaries. It does not freeze the still-open acoustic extraction recipe or Module F statistical methods, and it does not authorize experiment execution. No feature extraction, MERT inference, probe fitting, selection, or Test evaluation was run to create this roadmap.

This roadmap is an authoritative planning source for future ChatGPT/Codex context recovery, alongside the [project workflow](project_workflow.md), completed Module logs, and the original research intentions. It records **how evidence and repository/data reconnaissance after Modules A–D refined the remaining research route**. It does not replace earlier records or alter accepted results.

## 2. Preserve the Original Research Intentions

No standalone original research-plan document was found in the inspected repository/workspace. The original tracked research questions and experiment outline are recoverable in the [initial README at commit b8955d5](https://github.com/Yumek077/mert-emotion-probing/blob/b8955d5/README.md). The current [README research questions](../README.md#research-questions) retain the conceptual RQ3/RQ4 directions. Neither the historical plan nor those question statements is overwritten by this roadmap.

The original intentions ask about emotion decodability, layer differences, comparison with traditional low-level audio features, and plausible confounding factors such as tempo and energy. The researcher's broader original confound scope also considered Genre and Instrumentation; the current request explicitly refines their status below. The initial tracked README does not itself enumerate those two metadata variables, so this document does not attribute that detail to a nonexistent original-plan file.

The original plan continues to represent **what the project originally intended to investigate**. This roadmap specifies the operational questions and practical scope that now guide Modules E/F. The broad aspiration to examine information “beyond” conventional features is retained as motivation, not treated as a claim already established or guaranteed by a standalone baseline comparison.

## 3. Completed Research State

### Module A — Define (complete)

The [Module A log](research_logs/module_a_data_and_representation_protocol.md) fixes the DEAM primary population of 1,744 approximately 45-second excerpts, excluding 58 metadata-defined full songs; static `valence_mean` and `arousal_mean` targets on their original scale; the sample unit; MERT-v1-95M; audio handling; representation-level mapping; and temporal mean pooling.

The inherited A/B data contract also includes authoritative Sample-ID alignment and the fixed Train/Validation/Test split. For historical precision, the split was created and verified in Module B, not Module A. These decisions remain frozen for the remaining project.

### Module B — Represent (complete)

The [Module B log](research_logs/module_b_representation_extraction_and_dataset_construction.md) records canonical frozen FP32 MERT extraction in evaluation mode with gradients disabled, using independent single-item inference. The canonical cache is `outputs/embeddings/deam_mert_v1_95m_meanpool_all_layers.pt`, with shape `[1744, 13, 768]` and explicit sample IDs. Index 0 is Pre-Transformer; indices 1–12 are Transformer Layers 1–12. Representations are temporal means, with 768 dimensions per level.

Representation validation, cache construction, and Sample-ID-based target/split alignment are complete. The fixed split in `data/metadata/deam_primary_split_seed42.csv` has 1,221 Train, 262 Validation, and 261 Test samples. Downstream work reuses the cache and split; neither is regenerated.

### Module C — Probe / RQ1 (complete)

**RQ1: Can continuous Valence and Arousal be linearly decoded from the pre-specified frozen MERT Layer-12 representation?**

The [Module C log](research_logs/module_c_basic_probing_and_emotion_decodability.md) answers this positively under the fixed Ridge protocol. Authoritative Layer-12 Test R² values are 0.579780312 for Valence and 0.511421706 for Arousal. Both probes improve over their Train-mean baselines. This is evidence for linear decodability in the defined setting, not human-like understanding.

### Module D — Layer-wise Analysis / RQ2 (complete)

**RQ2: How does linear Valence and Arousal decodability vary across the 13 frozen MERT representation levels?**

The [Module D log](research_logs/module_d_layerwise_emotion_decodability_analysis.md), [final technical report](codex_reports/module_d_stage1_layerwise_emotion_decodability_analysis.md), and [complete numerical table](../outputs/results/module_d_stage1_layerwise_results.csv) establish:

- Both targets are already substantially linearly decodable at Pre-Transformer: Test R² 0.526720395 for Valence and 0.536628663 for Arousal.
- Valence shows early gains followed by comparatively stable, non-monotonic variation.
- Arousal shows a broad mid-depth high region at Layers 3–9, followed by a pronounced late-layer decline. Layer-9-to-12 Test R² decreases from 0.597765309 to 0.511421706.
- Greater representation depth does not monotonically improve emotion decodability.
- Observed numerical maxima are secondary descriptive results, not universally optimal layers. Valence Layer 9 and Arousal Layer 4 are not selected as new comparators.

Layer 12 reuses authoritative RQ1 Test results; it is not a new untouched observation or independent replication. Module D does not establish human-like emotion understanding, causality, total information content, statistically superior layers, or universally best layers.

## 4. Why Refine the Remaining Route?

Modules A–D establish:

**Can decode → Where across depth**

A plausible alternative explanation remains: MERT emotion decodability may partly reflect conventional acoustic information that is itself associated with Valence/Arousal. The remaining project therefore moves from establishing decodability to **testing simpler acoustic explanations and representation value**.

This is an operational refinement of the original RQ3/RQ4 direction, not a rejection of the original plan. A conventional-feature comparison can describe predictive performance under a matched probe. It cannot, by itself, identify unique information, causal mechanisms, or the complete content of either representation.

## 5. Frozen Operational RQ3 and Primary Comparator

> **RQ3 — How does emotion decodability from the pre-specified MERT Layer-12 representation compare with a conventional low-level acoustic feature baseline under the same linear probing framework?**

The comparison is:

```text
Pre-specified MERT Layer-12 representation → Ridge → Valence / Arousal
Conventional acoustic representation      → Ridge → Valence / Arousal
```

**The primary MERT comparator is Transformer Layer 12 (index 12).**

1. Layer 12 was the pre-specified primary representation for RQ1.
2. Module D has already used held-out Test to describe the full depth trajectory.
3. Choosing Valence Layer 9 or Arousal Layer 4 because of their Module D Test maxima would introduce Test-informed layer selection.
4. Keeping Layer 12 avoids this form of cherry-picking and preserves continuity with RQ1.
5. Module D remains a separate, complete body of RQ2 evidence.

Module E must not reselect the MERT layer from the Test trajectory. Existing authoritative Layer-12 results retain their original provenance and should be reused for the comparison rather than presented as new independent evidence. The fixed Test set has already been evaluated in C/D; its prior exposure must remain explicit, and its results cannot guide acoustic feature design or tuning.

The controlled-comparison principles are the same population, original-scale separate targets, fixed split, Train-only scaling, Ridge probe family, Validation-only alpha selection, MAE/R²/Pearson r, and protected final Test use. The inherited framework uses a fixed nine-value alpha grid, maximum Validation R² with larger alpha on an exact numerical tie, a Train-mean reference baseline, and final Train-only fitting. Module E design must preserve comparability with this framework rather than silently change it.

The intended experimental difference is **representation type**. A compact acoustic representation may have fewer dimensions than the 768-dimensional MERT representation; matching the linear probing framework does not imply equal dimensionality or proof of equal effective capacity. A performance difference describes this particular comparison, not unique/additional emotion information absent from all conventional features. No feature-fusion, residualization, or conditional-information experiment is silently added to RQ3.

## 6. Module E — Conventional Acoustic Baseline / RQ3

**Research role:** build a conventional waveform-derived acoustic representation and compare its Valence/Arousal linear decodability with the pre-specified MERT Layer-12 representation.

### Repository and data reconnaissance

No standalone prior acoustic-reconnaissance report was found. This section preserves the relevant findings, grounded in the local files rather than assuming an existing acoustic feature artifact:

- All 1,744 primary audio paths exist under `data/raw/deam/audio/`; the Module A verification record documents audio readability.
- `environment.yml` includes librosa 0.11.0, soundfile 0.14.0, NumPy, SciPy, and scikit-learn. The existing `src/mert_emotion_probing/audio.py` provides decoding, mono conversion, and resampling. These supply the data and software ingredients for conventional extraction.
- The current `data/raw/deam/verification/deam_item_mapping.csv` contains audio/identity/subset/target information, not a validated BPM, Energy, MFCC, spectral-feature, Genre, or Instrumentation representation.
- No conventional acoustic feature cache or completed extraction pipeline is present. Dependency availability is not an end-to-end extraction validation result. The preceding reconnaissance's synthetic smoke check did not complete and was stopped; it is not counted as a passed compatibility check. No such runtime check was repeated for this roadmap.
- Raw metadata inspection confirms Genre availability and the absence of a unified Instrumentation field, as detailed below. Raw data and private notes remain local and ignored.

**Priority candidate families:** Tempo, RMS Energy, MFCC, and basic spectral descriptors.

The future Module E extraction will make one independent conventional-feature pass over the 1,744 primary excerpts and cache fixed-dimensional vectors keyed by Sample ID. This is not a MERT rerun. The existing canonical representations, split, targets, and accepted A–D evidence stay unchanged.

### Decisions deliberately left open for Module E design

| Decision | Current status |
|---|---|
| Exact included acoustic feature set | Not frozen; families above are candidates |
| MFCC count and coefficient conventions | Not frozen |
| Exact spectral descriptor list | Not frozen |
| Per-excerpt aggregation policy | Not frozen |
| Acoustic sampling/frame/window/hop/frequency parameters | Not frozen |
| Final acoustic dimensionality | Not frozen; the reconnaissance example of 51 dimensions is not a protocol |
| Tempo failure / undefined-value policy | Must be frozen before formal extraction |
| Silence, log-floor, other invalid-value handling and diagnostics | To be made explicit in Module E design |
| Exact extraction implementation and artifact schema | To be settled after the feature definition is agreed |

Module E design must distinguish numerical failures from meaningful zeros and uncertain estimates, without silently dropping primary samples or inventing values. If a learned missing-value transform is adopted, its fitting must respect the Train-only principle. No particular imputation policy is selected by this roadmap.

### Energy extraction boundary (frozen constraint)

RMS/Energy must be extracted from a waveform path that preserves the amplitude meaning of the decoded recording. Do not compute the conventional Energy baseline from MERT-input-normalized waveform values. The existing `prepare_mert_inputs` documentation in `src/mert_emotion_probing/mert.py` records zero-mean/unit-variance feature-extractor normalization, which changes absolute amplitude information.

The exact Energy definition and extraction implementation will be frozen in Module E. RMS is an acoustic amplitude-related measure, not a perfect perceptual loudness measure and not the Arousal target itself. Recording gain and production effects remain possible contributors.

### Tempo boundary (frozen constraint)

Estimated Tempo/BPM can serve as a conventional feature and later acoustic correlate, but it is not perfect ground truth. Known risks include weak/no onset, half/double-tempo ambiguity, beat-tracking failure, and possible 0 BPM / empty beats. A finite numerical BPM must not automatically be interpreted as reliable true tempo.

**Module E must freeze its Tempo failure / undefined policy before formal extraction, not after observing Test results.** The algorithm, quality diagnostics, and exact handling remain Module E design decisions. The actual failure frequency in the primary population has not yet been measured.

## 7. Frozen Core RQ4 Direction

> **RQ4 — To what extent are the observed emotion-decoding results associated with simple acoustic correlates, particularly Tempo and Energy?**

The purpose is to examine whether simple acoustic correlates provide plausible partial explanations for observed emotion-decoding performance. Association does not demonstrate causal explanation, and this scope does not promise to eliminate all confounding.

| Candidate | Frozen roadmap status | Reason |
|---|---|---|
| Tempo | Core RQ4 focus | A simple acoustic correlate, subject to estimation/failure diagnostics |
| Energy / RMS-derived information | Core RQ4 focus | A simple amplitude-related correlate with a defined preprocessing boundary |
| Genre | Optional / deferred; outside core RQ4 | Metadata exists, but semantics and label structure are not validated as one common variable |
| Instrumentation | Removed from core RQ4 | No reliable unified ground truth; would require dedicated annotation/modeling design |

### Genre: available metadata, deferred analysis

The local `data/raw/deam/metadata/metadata/metadata_2013.csv` contains 744 primary records and eight relatively simple categorical Genre strings. `metadata_2014.csv` contains the other 1,000 primary records and 125 distinct, often composite Genre strings. Nonblank Genre entries cover all 1,744 primary IDs, but coverage is not evidence of a unified taxonomy or validated semantic consistency.

The 2014 CSV also expands folksonomy tags into additional fields beyond the short header. Naive parsing or blindly splitting every hyphen would be unsafe, including for labels such as `Hip-Hop`. The canonical item mapping does not already provide a validated unified Genre variable.

**Genre is optional/deferred, not part of frozen core RQ4.** Revisit it only if a simple, reliable, scientifically defensible common treatment becomes available and is discussed explicitly. Do not expand Module F merely to include Genre or construct a large metadata ontology.

### Instrumentation: outside the core project

No unified Instrumentation ground truth was found. Existing tags are incomplete and heterogeneous across the population, mixed with genre/emotion/personal labels, and not a validated common annotation system. Sparse tags cannot establish reliable instrument presence or absence.

**Instrumentation is removed from core RQ4 analysis.** Future instrumentation work would require a dedicated annotation or recognition-model design and validation. That is outside this mini-project's current core scope; no instrument-recognition task is added.

## 8. Module F — Acoustic Correlate / Confound & Error Analysis / RQ4

Module F is designed and executed **only after Module E is complete**. The acoustic feature table and extraction diagnostics from E will provide actual input facts, including Tempo reliability and any exceptional values.

Only Module F's research role is frozen here. Its exact statistical analysis is not frozen. Possible limited analyses include Tempo/Energy relationships with targets; relationships with MERT/acoustic predictions or errors; and whether these patterns offer plausible partial acoustic explanations. The exact method and permissible use of existing predictions/partitions must be discussed after E, with prior Test exposure acknowledged and no feedback into accepted model selection.

Do not preempt that discussion by adding causal claims, nonlinear confound models, large metadata taxonomies, instrument recognition, or unnecessary statistical machinery.

### Supporting error analysis

Error analysis retains the spirit of the original plan but is **supporting analysis, not an independent Research Question**. If E/F reveals an interpretable failure pattern, examine large errors, shared MERT/acoustic failures, or possible target/acoustic patterns. Label such findings appropriately as descriptive or exploratory. If there is no clear evidence, do not force a narrative or modify the frozen models to create one.

## 9. Remaining Project Route

```text
Module A — Define                                      [complete]
    ↓
Module B — Represent                                   [complete]
    ↓
Module C — Probe / RQ1                                 [complete]
    ↓
Module D — Layer-wise Analysis / RQ2                    [complete]
    ↓
Module E — Conventional Acoustic Baseline / RQ3         [design next]
    ↓
Module F — Acoustic Correlate / Confound & Error
           Analysis / RQ4                              [design after E]
    ↓
Final Synthesis
```

Final Synthesis will connect:

**Can decode → Where across depth → Compared with simple acoustics → Alternative acoustic explanations**

This synthesis must separate completed observations from remaining hypotheses, comparison results from unique-information claims, and acoustic associations from causal explanations.

## 10. Minimal Sufficient Rigor and Scope

Keep the smallest defensible design that answers each core question. Do not automatically introduce nonlinear probes, cross-validation, bootstrap, significance testing, multi-seed averaging, MERT fine-tuning, new foundation models, dynamic emotion annotations, causal intervention, a large Genre taxonomy, instrumentation recognition, or additional datasets.

Optional robustness/ablation ideas in the original plan remain historical possibilities, not automatic next tasks. Discuss extensions only after the core RQ1–RQ4 evidence chain is complete and a clear research reason exists. No claim should extend to human-like emotion understanding, complete information content, causality, universal optimality, or automatic generalization beyond the observed setting.

## 11. Learning Priorities and Checkpoint

1. **Level 1 — Research Mainline:** understand why each Module exists, what it receives, and which question it answers.
2. **Level 2 — Key Methodological Decisions:** understand why the comparator remains Layer 12, why representation comparisons use a common linear framework, why acoustic failure policies precede evaluation, and why Genre/Instrumentation have narrower scope than originally contemplated.
3. **Level 3 — Engineering Details:** understand the problems solved by identity joins, caches, diagnostics, and gates; memorizing APIs, helper names, CLI flags, or debugging details is not required.

### What I should now be able to explain

- Why positive RQ1/RQ2 decoding evidence still leaves conventional acoustic explanations open.
- Why a standalone acoustic baseline comparison does not prove unique emotion information in MERT.
- Why Module D Test maxima cannot select the RQ3 comparator, and why Layer 12 remains the pre-specified comparison.
- Which choices are frozen here and which require Module E design.
- Why Energy must precede MERT amplitude normalization and Tempo needs a failure policy.
- Why Genre is deferred, Instrumentation is outside core RQ4, and Module F methods wait for E.
- How the four research questions form one evidence chain without causal overclaiming.

## 12. Context Recovery and Document Boundaries

For the next conversation, read the [workflow](project_workflow.md), [original README history](https://github.com/Yumek077/mert-emotion-probing/blob/b8955d5/README.md), the [current README](../README.md), the linked A–D research logs, the [Module D technical report](codex_reports/module_d_stage1_layerwise_emotion_decodability_analysis.md), and this roadmap. The source/data paths above allow the acoustic reconnaissance facts to be inspected locally without treating private handoffs as published authority.

Original intentions, completed empirical records, and this refined planning document coexist. In future, a concrete conflict with any frozen research decision must be reported for researcher decision rather than silently repaired. A later Module E protocol should record the currently open choices explicitly, without retrospectively rewriting this roadmap or A–D evidence.

This is a documentation-only planning freeze. Completed logs, source, cache, split, metrics, predictions, and figures are unchanged. Private `personal_notes/`, `handoffs/`, raw data, and large caches remain ignored/local; unrelated `docs/.obsidian/` state is not part of this roadmap commit.

**Post-Module-D roadmap is frozen. The next research task is Module E design; no Module E experiment has been run.**
