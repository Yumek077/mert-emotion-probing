# Module D — Layer-wise Emotion Decodability Analysis

## Objective and Research Question

**RQ2: How does linear Valence and Arousal decodability vary across the 13 frozen MERT representation levels?**

Module C established held-out linear decodability at the pre-specified Transformer Layer 12. Module D extends the same probing framework across representation depth. Its aim is to describe early availability, changes, broad high-performing regions and late-depth behavior, rather than select a universally best layer.

Module A defined the data and representation protocol, Module B supplied the canonical frozen cache and fixed split, and Module C supplied the verified Ridge framework. Module D changes the representation level while keeping these inherited decisions fixed. Implementation, verification, ChatGPT research interpretation review, and the researcher learning checkpoint are complete. The Module D research protocol and results are accepted for project use.

## Data and Frozen Protocol

The study uses the 1,744 approximately 45-second DEAM excerpts and excludes the 58 metadata-defined full songs. Static continuous `valence_mean` and `arousal_mean` remain on their original scale and are modeled separately. Sample ID is the authoritative key for cache, target and split alignment.

The canonical MERT-v1-95M cache has shape `[1744, 13, 768]`. It was previously extracted with frozen FP32 MERT, evaluation mode, disabled gradients, independent single-item inference and temporal mean pooling. No extraction was repeated. Index 0 denotes Pre-Transformer; indices 1–12 denote Transformer Layers 1–12. Each probe receives 768 features, never a concatenation of layers.

The unchanged split contains 1,221 Train, 262 Validation and 261 Test items. Each of 26 level × target configurations uses a Train-fitted StandardScaler and Ridge regression with intercept and Cholesky solver. Each independently selects alpha from `1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000`, maximizing Validation R² with larger alpha on an exact numerical tie. Final fitting remains Train-only. MAE, R² and Pearson r are reported together with the constant Train-target-mean baseline.

Point estimates only are reported. No bootstrap, confidence interval, significance test, multi-seed average or cross-validation was introduced. Descriptive depth ranges below summarize the observed curve; they are not additional fitted comparisons or inferential groupings.

## Protected Test Policy

All 26 Train/Validation selections were completed and verified before the unified Test sweep. The frozen gate records configurations, metrics, plot policy, input/code identities and selection artifacts. Test was never used for fitting, alpha selection, layer selection, metric selection or reporting-policy revision. No Train + Validation refit occurred and no post-Test tuning was performed.

The monolithic cache and label table were loaded for integrity and Sample-ID alignment checks. This technical access did not expose Test values to fitting or selection; poisoning all Test features and targets with NaN left every selection and Validation prediction unchanged. The complete primary plot policy was frozen before the Test sweep.

> Layer 12 values were already obtained as the authoritative RQ1 held-out evaluation and are reused in Module D for the complete depth-wise analysis.

Accordingly, the sweep comprises 24 new level-0-through-11 evaluations and two reused Layer-12 evaluations. Layer 12 is not a new untouched Test discovery or an independent replication. Its prior known performance did not guide other levels' alpha selection or the fixed protocol.

## Verification

Verification passed for all 26 configurations: level metadata, exact population and split counts, Sample-ID alignment, row-order invariance, Train-only scaler statistics, target mapping, independent alpha selection, exact versus near-tie behavior, deterministic selection with poisoned Test values, and independent metric recomputation. The saved CSVs contain 6,812 Validation rows and 6,786 Test rows with unique Sample-ID/level/target keys.

Layer-12 Validation candidate scores and selected metrics exactly reproduce C3. Layer-12 Test metrics, baselines and ID-aligned predictions are exactly equal to authoritative C4 artifacts. The original Module C artifact verifiers also pass. No protocol conflict, invalidating error or correction was required.

## Results Across Depth

Valence selected alpha 1000 at all levels. Arousal selected alpha 100 at Pre-Transformer and Layers 1–2, then 1000 at Layers 3–12. These outcomes came from Validation, not Test.

| Level | Valence alpha | Arousal alpha | Valence Test R² | Arousal Test R² |
|---|---|---|---|---|
| Pre-Transformer | 1000 | 100 | 0.526720395 | 0.536628663 |
| Layer 1 | 1000 | 100 | 0.559071673 | 0.557815064 |
| Layer 2 | 1000 | 100 | 0.568835066 | 0.554008616 |
| Layer 3 | 1000 | 1000 | 0.556921405 | 0.605582142 |
| Layer 4 | 1000 | 1000 | 0.554643527 | 0.611146533 |
| Layer 5 | 1000 | 1000 | 0.582014093 | 0.598320849 |
| Layer 6 | 1000 | 1000 | 0.585611861 | 0.601162974 |
| Layer 7 | 1000 | 1000 | 0.583344063 | 0.611004060 |
| Layer 8 | 1000 | 1000 | 0.570337948 | 0.603732201 |
| Layer 9 | 1000 | 1000 | 0.586966344 | 0.597765309 |
| Layer 10 | 1000 | 1000 | 0.573169560 | 0.575585635 |
| Layer 11 | 1000 | 1000 | 0.555938272 | 0.545197324 |
| Layer 12 (reused) | 1000 | 1000 | 0.579780312 | 0.511421706 |

The machine-readable full table also reports selected Validation MAE/R²/r, Test MAE/R²/r and both Validation/Test baseline metrics for every row.

### Valence Trajectory

Valence is already linearly decodable at Pre-Transformer (Test R² 0.526720). It rises to 0.559072 at Layer 1 and 0.568835 at Layer 2. Subsequent changes are nonmonotonic: Layers 3–4 fall slightly, Layers 5–7 are around 0.582–0.586, Layer 8 dips, and Layer 9 returns to 0.586966. Layers 10–11 decline before Layer 12 returns to 0.579780.

The main description is early availability followed by comparatively modest fluctuations and a broad high region around Layers 5–9. There is no continuous depth-by-depth improvement and no sharp unique optimum. The Layer-12 endpoint is within the broad range of the later Valence values.

### Arousal Trajectory

Arousal is also decodable at Pre-Transformer (0.536629). After values near 0.554–0.558 at Layers 1–2, it rises to 0.605582 at Layer 3. Layers 3–9 form a relatively flat high region spanning 0.597765–0.611147. This plateau does not mean the numbers are equal or their differences are statistically indistinguishable; it is a descriptive summary of the curve.

Arousal then declines consecutively: Layer 9 0.597765, Layer 10 0.575586, Layer 11 0.545197 and Layer 12 0.511422. The decline from Layer 9 to Layer 12 is approximately 0.08634 R² and extends across several levels. Its depth pattern is more pronounced than the relatively stable Valence pattern.

### Comparison Between Targets

Both targets show substantial positive held-out decoding before the first Transformer layer, so linear accessibility does not first appear in the final Transformer layers. Pre-Transformer includes learned frontend and positional/normalization processing; the observation does not imply that raw audio directly yields these scores.

Arousal gains more clearly into the middle layers and loses more towards the final layer. Valence changes more modestly and nonmonotonically. The different curves show that greater representation depth does not improve both emotion targets in the same way. Comparing target-normalized R² values alone does not prove one emotion dimension intrinsically easier than the other.

### Secondary Numerical Maxima

The highest observed Valence R² is Layer 9, 0.586966; Layer 6 is 0.585612, only about 0.001354 lower. The highest observed Arousal R² is Layer 4, 0.611147; Layer 7 is 0.611004, only about 0.000142 lower. These are secondary descriptive observations from one fixed Test split. They do not support statistical superiority, universal optimality or a unique best MERT layer. No layer was selected for a new tuned experiment.

## Baseline and Other Metrics

| Target | Train mean | Validation MAE | Validation R² | Test MAE | Test R² | Pearson r |
|---|---|---|---|---|---|---|
| valence | 4.894692875 | 0.924825572 | -0.000046064 | 1.002563190 | -0.003113807 | undefined (JSON null) |
| arousal | 4.816953317 | 1.012681884 | -0.000018369 | 1.131995632 | -0.000320133 | undefined (JSON null) |

Every level/target probe has positive Test R² and lower Test MAE than its Train-mean baseline. Pearson correlations are positive for all configurations. Constant-baseline Pearson correlations are undefined due to zero prediction variance and are stored as JSON null. Slightly negative baseline R² is valid because the prediction is the Train mean, not the Test mean.

Across levels, Valence Test MAE ranges from 0.623049 to 0.658280 and Pearson r from 0.728280 to 0.768855. Arousal Test MAE ranges from 0.652016 to 0.744556 and Pearson r from 0.715509 to 0.781930. The primary predeclared trajectory uses R²; the full metrics are retained rather than choosing a metric after seeing which looks best.

## Visualization

![Layer-wise held-out Test R² trajectory](../../outputs/figures/module_d_stage1_test_r2_trajectory.png)

The complete connected-point plot displays both targets across all 13 levels without smoothing or uncertainty bands. The zero reference places positive decoding in context. Open endpoint markers and the caption identify reused Layer-12 evidence. SVG is also saved for scalable reuse. The figure describes the complete frozen evaluation and was not used to choose a layer for retuning.

## Interpretation and Evidence Boundary

The evidence supports the following conclusion:

**Linear decodability of Valence and Arousal varies across frozen MERT representation depth under the fixed Ridge probing protocol. Both targets are readable early; Arousal shows a broad middle-depth high region followed by a late decline, while Valence remains comparatively stable after early gains.**

This is a descriptive result for one DEAM excerpt population and one fixed split. Larger systematic changes across several levels can be discussed as observed patterns, but no claim of statistical significance or uncertainty bounds is available. Small between-layer gaps should not be assigned strong scientific meaning.

The study does not establish human-like emotion understanding, causal emotion encoding, linear accessibility of all emotion information, a universally optimal layer, generalization to other datasets or full songs, superiority over conventional acoustic features, or freedom from acoustic confounds. It does not identify why the curves change. Differences can reflect accessibility to this particular regularized linear probe rather than a complete measure of information content.

## What I Should Now Be Able to Explain

- Why RQ2 compares a complete depth trajectory rather than searching for a winning layer.
- Why every configuration uses the same model family, feature dimension, target scale and tuning opportunity.
- Why each configuration may select its own alpha, but must do so on Validation only.
- Why all configurations and reporting rules are frozen before Test.
- Why reused Layer 12 is not new untouched evidence.
- How the observed Valence and Arousal patterns differ, and why small numeric maxima are weak evidence for ranking.
- Which conclusions require new questions or experiments rather than stronger wording.

## Artifacts and Module Outcome

The detailed technical history is in `docs/codex_reports/module_d_stage1_layerwise_emotion_decodability_analysis.md`. All selection, gate, prediction, Test, table and verification artifacts use the `outputs/results/module_d_stage1_*` prefix. Figures use `outputs/figures/module_d_stage1_test_r2_trajectory.*`. The private Chinese learning note is `personal_notes/module_d_what_i_should_understand.md`.

Module D implementation, verification, ChatGPT research interpretation review, and the researcher learning checkpoint are complete. The research protocol and results are accepted for project use. **Module D is finalized and RQ2 is complete under the frozen protocol.** Finalization involved documentation and GitHub synchronization only; no research experiment was rerun and no frozen protocol or result changed.
