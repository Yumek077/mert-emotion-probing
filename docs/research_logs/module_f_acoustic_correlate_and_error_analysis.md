# Module F — Acoustic Correlate and Error Analysis / RQ4

## Research question

How are the observed Valence and Arousal decoding results associated with Tempo and Energy under the frozen analysis protocol?

**Module F / operational RQ4 is finalized.** Stage 1 Validation and Stage 2 frozen Test analysis both passed researcher and ChatGPT review. This module-level record consolidates the accepted interpretation on 2026-10-01. RQ4 core evidence is complete; no Stage 3 experiment is needed. Finalization creates documentation and the private learning note without changing the frozen design or generating new scientific results. The next research task is Final Synthesis of RQ1–RQ4.

## Why this Module was needed

Module C established linear emotion decodability from the pre-specified MERT Layer-12 representation. Module D described how decodability varies across depth. Module E found stronger held-out Valence/Arousal decoding from Layer 12 than from the selected conventional 51-D acoustic baseline under the shared probing framework.

Those results do not establish independence from simple acoustic correlates. The emotion labels themselves can be associated with acoustic properties, and MERT predictions can track some of that structure. RQ4 therefore examines two pre-specified descriptors at three points: the true targets, the predictions, and the remaining prediction errors. It supplies an alternative acoustic explanation to consider when interpreting RQ1–RQ3, without identifying a causal mechanism or unique emotion information.

## Frozen design

- Tempo is the cached automated global BPM estimate, `tempo_bpm`.
- Energy is the cached temporal mean of frame RMS, `rms_mean`, calculated on decoded-amplitude audio before MERT input normalization. It is a signal-amplitude proxy, not calibrated perceptual loudness or Arousal.
- MERT Layer 12 remains the fixed primary representation. The frozen conventional acoustic 51-D branch is a supporting comparator.
- Pearson r is computed separately for Valence and Arousal between each descriptor and five objects: true target, MERT prediction, acoustic prediction, MERT residual, and acoustic residual.
- The three-layer logic is **target → prediction → residual**. Residual is `y_true - prediction`; a positive residual means the model under-predicts the target.
- Validation is analyzed first, followed by the same frozen analysis on Test after the review gate. The partitions are kept separate.

The analysis reuses the Module E acoustic cache, Module C MERT predictions, Module E acoustic predictions, original-scale static targets, and the fixed Sample-ID split. It performs no fitting, alpha selection, layer selection, acoustic extraction, or MERT inference. No Train association analysis, partial correlation, all-feature screening, layer sweep, significance test, or post-hoc extension was added.

The operational scope follows the [Post-Module-D Research Roadmap](../research_roadmap_after_module_d.md). Full definitions, implementation and exhaustive coefficients remain in the Stage reports.

## Stage 1 — Validation

Stage 1 contains the 262 frozen Validation samples: 524 Sample-ID/target analysis rows and 20 associations. Validation had already served model-selection purposes in earlier modules; it is not an untouched evaluation population.

All 15 runner checks and 17 independent verification checks passed. Identity alignment, exact cached descriptors, labels, reused predictions, residual definition and all saved coefficients were verified; all associations were defined. The saved independent Pearson comparison had maximum absolute difference `2.7755575615628914e-16`.

Tempo associations with both targets, both prediction branches and both residual branches were small in absolute magnitude. Energy was positively associated with both targets and MERT predictions. MERT residuals retained smaller positive Energy association; acoustic predictions tracked Energy more strongly, while acoustic residual correlations were near zero.

Stage 1 closed with the Test gate closed and no Test association analysis performed. Its [report](../codex_reports/module_f_stage1_validation_acoustic_association_analysis.md) and [pre-Test verification record](../../outputs/results/module_f_stage1_pre_test_verification.json) retain that historical state unchanged. Subsequent researcher approval authorized Stage 2; the present module-level record records the completed reviews rather than rewriting the Stage history.

## Stage 2 — Test and Validation/Test comparison

Stage 2 applies the unchanged Stage 1 analysis to 261 frozen Test samples: 522 analysis rows, 20 Test associations and 20 Validation/Test comparison rows. All 20 runner checks and 26 independent verification checks passed. The independent verification checked the Test coefficients, existing Validation coefficients and exact comparison source values. The saved maximum Test Pearson difference was `9.992007221626409e-16`; no association was undefined. All eight Stage 1 files were preserved.

The Test set had already been viewed in Modules C–E. This analysis is a **frozen descriptive consistency check**, not untouched confirmatory evidence, independent replication or statistical replication. It reuses existing predictions and does not provide a new model-performance evaluation.

Tempo associations remained small. The main Energy pattern persisted in both partitions, with differences in magnitude: target and prediction correlations were higher in Test; MERT Valence residual association was similar, while its Arousal residual association was smaller. Acoustic residual correlations changed from small negative Validation values to near-zero positive Test values. Some small Tempo correlations also changed sign. These differences remain descriptive; exact signs are not a statistical consistency criterion.

The following table is formatted directly from the authoritative [Validation associations](../../outputs/results/module_f_stage1_validation_associations.csv) and [Test associations](../../outputs/results/module_f_stage2_test_associations.csv). Values are Pearson r with Energy (`rms_mean`), rounded to six decimal places.

| Analysis object | Valence Validation | Valence Test | Arousal Validation | Arousal Test |
|---|---:|---:|---:|---:|
| True target | 0.265603 | 0.330605 | 0.235629 | 0.273462 |
| MERT Layer-12 prediction | 0.205489 | 0.309959 | 0.209699 | 0.300741 |
| Acoustic 51-D prediction | 0.479110 | 0.544525 | 0.364227 | 0.422842 |
| MERT Layer-12 residual | 0.177979 | 0.183380 | 0.122148 | 0.084605 |
| Acoustic 51-D residual | -0.049686 | 0.000123 | -0.020146 | 0.016597 |

Full values for all 40 Tempo/Energy coefficients and their comparison are preserved in the [comparison CSV](../../outputs/results/module_f_stage2_validation_test_comparison.csv), [Stage 2 report](../codex_reports/module_f_stage2_frozen_test_acoustic_association_analysis.md), and [final verification record](../../outputs/results/module_f_stage2_final_verification.json).

![Frozen Validation and Test acoustic associations](../../outputs/figures/module_f_stage2_validation_test_associations.png)

## RQ4 result

**Under the frozen automated BPM measurement, Tempo provides little descriptive evidence for a simple linear explanation of the observed emotion-decoding pattern.** All analyzed Tempo associations are small in absolute magnitude across Validation and Test. This does not mean Tempo is irrelevant, unrelated to emotion, or absent from every form of dependence.

**Energy provides a plausible partial acoustic explanation.** Energy has positive associations with the true Valence/Arousal targets and MERT Layer-12 predictions in both partitions. MERT predictions therefore track some Energy-associated structure. Smaller positive Energy associations remain in MERT residuals, so Energy does not fully explain MERT prediction behavior. This is a descriptive partial explanation, not a complete or causal account. The analysis does not identify what the remaining information represents.

## Supporting acoustic comparison

The acoustic 51-D predictions show stronger positive Energy association than MERT predictions, and their residual Energy association is near zero. This pattern is interpretable in light of `rms_mean` itself being included in the acoustic representation. It does not isolate the contribution of that feature, identify a mechanism, or imply that the acoustic branch predicts emotion better overall.

[Module E / RQ3](module_e_conventional_acoustic_baseline.md) established lower Test MAE and higher Test R²/Pearson r for MERT Layer 12 than for the selected 51-D baseline on both targets. Module F / RQ4 describes association with two acoustic descriptors. **Predictive performance and acoustic association are different analytical questions.** Stronger acoustic-prediction association with Energy does not contradict the RQ3 performance result or create a new model ranking.

## Evidence and interpretation boundaries

- Association is not causation or evidence of causal confounding. No conditional analysis was performed.
- Pearson r measures simple linear association; it does not quantify the fraction of decoding performance explained by Tempo or Energy.
- Near-zero Tempo r does not exclude nonlinear, conditional or other dependence. Automated BPM retains beat-tracking and half/double-tempo uncertainty. The known one-beat estimate for ID 437 was retained in the acoustic cache; it belongs to Train and is absent from both analyzed partitions.
- RMS is a decoded-amplitude descriptor that can reflect recording gain and production as well as signal properties. It is not calibrated loudness and is not an emotion label.
- No significance testing, uncertainty interval or claim of statistical replication was added.
- Residual association does not establish unique emotion information, non-acoustic information or high-level emotion understanding. Its absence would not establish those claims either.
- The evidence concerns these frozen descriptors, predictions and DEAM partitions. It does not establish cross-dataset generalization.
- Validation previously supported model selection, and Test had prior exposure in Modules C–E. Keeping the analysis frozen preserves its descriptive provenance without making Test untouched.

## What RQ4 answered and did not answer

RQ4 answered whether the two fixed acoustic descriptors show simple linear association with the true targets, predictions and residuals, and whether their descriptive patterns are similar across the two fixed partitions. Tempo supplies little evidence for that simple linear explanation; Energy supplies a plausible partial explanation for MERT decoding.

RQ4 did not establish a causal explanation, remove acoustic confounding, estimate a proportion of performance explained, identify the contents of remaining information, or establish independence from all acoustic factors. The supporting 51-D comparison does not exhaust possible conventional acoustic information.

## Verification at finalization

Finalization used read-only integrity and documentation checks. The 17 Stage 1/2 scripts, results, figures and reports remain byte-for-byte unchanged, and both saved verification records retain `passed=true`. Their historical checks and gates remain preserved. The compact numerical table was checked against the authoritative CSVs; README and the private Chinese learning note retain the accepted interpretation and provenance. Earlier module files and the frozen protocol were not changed.

Only the Stage 1/2 GitHub-facing artifacts, this research log and the README update belong to the Module F checkpoint. The Chinese note remains under ignored `personal_notes/`; raw audio and local caches remain excluded. The unrelated `docs/.obsidian/` directory is untouched and excluded from the commit.

Git whitespace checks pass for the documentation, code, JSON and CSV files. The full staged check reports only generator-produced trailing whitespace inside the two unchanged SVG figures' path data. That formatting is retained to preserve the reviewed artifacts and recorded hashes.

## Inputs to Final Synthesis

The complete core evidence chain is now **Can decode → Where across depth → Compared with simple acoustics → Alternative acoustic explanations**. Final Synthesis should connect the accepted Module C/RQ1, Module D/RQ2, Module E/RQ3 and Module F/RQ4 records while keeping decodability, representation comparison, acoustic association, and causality/unique information conceptually separate.

Module F contributes the frozen two-partition associations, the target/prediction/residual distinction, the Energy partial-explanation result, and the Tempo measurement boundary. Final Synthesis is the next task; it is not started during this finalization.

## What I should now be able to explain

- Why emotion decodability and a stronger RQ3 performance result do not establish independence from acoustic information.
- What target, prediction and residual associations each ask, including why a positive residual means under-prediction.
- Why Tempo supports only a limited statement about simple linear association under this automated measurement.
- Why Energy is a plausible partial explanation, and why remaining residual association cannot identify unique or non-acoustic emotion information.
- Why stronger Energy association in acoustic predictions does not mean stronger overall prediction performance.
- Why Pearson r is neither a causal estimate nor the fraction of performance explained.
- Why prior Validation selection and Test exposure remain explicit, and how RQ4 fits the RQ1–RQ4 evidence chain.
