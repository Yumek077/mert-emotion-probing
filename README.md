# Probing Musical Emotion in Pretrained Music Representations

This project investigates what a simple linear probe can read about continuous musical emotion from a frozen pretrained music model. Using DEAM excerpts and MERT representations, it connects emotion decodability, representation depth, comparison with conventional acoustic features, and possible acoustic explanations.

The broader motivation includes whether pretrained representations offer emotion-related information beyond low-level acoustics and whether cues such as tempo and energy help explain prediction behavior. The operational questions below deliberately narrow that motivation to claims supported by the frozen experiments. They do not promise unique information or a quantified account of confounding. The [original research intentions](https://github.com/Yumek077/mert-emotion-probing/blob/b8955d5/README.md) and [frozen Post-Module-D roadmap](docs/research_roadmap_after_module_d.md) preserve the planning history.

## Research Questions

**Can decode → Where across depth → Compared with simple acoustics → Alternative acoustic explanations**

1. **RQ1:** Can continuous Valence and Arousal be linearly decoded from the pre-specified frozen MERT Layer-12 representation?
2. **RQ2:** How does linear Valence and Arousal decodability vary across the 13 frozen MERT representation levels?
3. **RQ3:** How does emotion decodability from the pre-specified MERT Layer-12 representation compare with a conventional low-level acoustic representation under the same linear probing framework?
4. **RQ4:** How are the observed Valence and Arousal decoding results associated with Tempo and Energy under the frozen analysis protocol?

RQ1 establishes decodability at the pre-specified representation; RQ2 describes its distribution across depth. RQ3 places the result against a selected conventional acoustic baseline. That comparison cannot establish acoustic independence, so RQ4 examines two simple acoustic correlates of targets, predictions, and errors. Modules C–F answer these questions; Modules A/B supply their methodological foundation.

## Methodological Foundation

[Module A — Data and Representation Protocol](docs/research_logs/module_a_data_and_representation_protocol.md) defines the population, targets, audio handling, representation levels, and pooling. [Module B — Extraction and Dataset Construction](docs/research_logs/module_b_representation_extraction_and_dataset_construction.md) validates the canonical representations, Sample-ID alignment, and fixed split. Neither is an additional emotion research question.

- **Population and targets:** 1,744 primary DEAM approximately 45-second excerpts; 58 metadata-defined full songs are excluded. Each audio item is one sample. Static averaged `valence_mean` and `arousal_mean` remain on their original scale and are modeled separately.
- **Representation:** frozen `m-a-p/MERT-v1-95M`, extracted by independent single-item inference over actual decoded excerpts. Temporal mean pooling gives a canonical `[1744, 13, 768]` cache: index 0 is Pre-Transformer and indices 1–12 are Transformer Layers 1–12. Each level supplies 768 features.
- **Split:** fixed 1,221 / 262 / 261 Train / Validation / Test samples, created once with seed 42. Sample ID is the join key.
- **Probe:** separate Ridge regressions with Train-only StandardScaler and coefficient fitting. Each selects alpha on Validation only from the predeclared grid. Final fitting uses Train only, without combining Train and Validation.
- **Representation roles:** Layer 12 is the pre-specified RQ1 representation and remains the RQ3/RQ4 comparator; the RQ2 numerical maxima do not reselect it. The frozen 51-D acoustic representation contains Tempo, RMS, MFCC, and basic spectral descriptors.
- **Evaluation:** MAE, R², and Pearson r, with a constant Train-target-mean reference. Test is excluded from fitting and selection; frozen configurations are evaluated after their review gates, with no post-Test tuning.

C4 supplied the original protected Layer-12 Test evaluation. D/E reuse that evidence and disclose prior use of the same Test partition. F reuses saved predictions for acoustic associations; Validation previously supported selection and Test had already been viewed in C–E. Its Test analysis is a frozen descriptive consistency check.

Detailed preprocessing, selection rules, commands, and verification remain in the accepted logs and Stage reports.

## Main Results

### RQ1 — Held-out Linear Decodability

**Question:** Can Valence and Arousal be linearly decoded from the pre-specified frozen Layer-12 representation?

**Answer:** Both Ridge probes improve over their Train-mean references on the 261 held-out Test excerpts, supporting linear decodability in this setting. This does not establish human-like emotion understanding.

| Target | Predictor | Test MAE | Test R² | Test Pearson r |
|---|---|---:|---:|---:|
| Valence | MERT Layer 12 | 0.635144565 | 0.579780312 | 0.768855324 |
| Valence | Train-mean reference | 1.002563190 | -0.003113807 | undefined |
| Arousal | MERT Layer 12 | 0.744556180 | 0.511421706 | 0.715509311 |
| Arousal | Train-mean reference | 1.131995632 | -0.000320133 | undefined |

The table reuses the existing C result/reference tables, displaying source metrics to nine decimal places. Constant-reference Pearson r is undefined and stored as JSON `null`; its slightly negative R² is valid because the constant comes from Train.

Read more: [accepted Module C log](docs/research_logs/module_c_basic_probing_and_emotion_decodability.md) · [authoritative Test JSON](outputs/results/module_c_stage4_test.json) · [C4 Test report](docs/codex_reports/module_c_stage4_held_out_test_evaluation.md).

### RQ2 — Decodability Across Depth

**Question:** How does linear decodability vary across all 13 frozen representation levels?

**Answer:** Both targets are readable early. Valence remains comparatively stable after early gains; Arousal shows a broad middle-depth high region followed by a late decline. Greater depth is not monotonically better.

![Layer-wise held-out Test R² trajectory](outputs/figures/module_d_stage1_test_r2_trajectory.png)

The complete trajectory is the primary result. Pre-Transformer already includes learned frontend and positional/normalization processing; it is not raw audio. Numerical maxima are descriptive observations from one split, not universally optimal layers. Open Layer-12 markers identify reused C4 values, not new untouched evidence.

Read more: [accepted Module D log](docs/research_logs/module_d_layerwise_emotion_decodability_analysis.md) · [authoritative layer-wise CSV](outputs/results/module_d_stage1_layerwise_results.csv) · [D technical report](docs/codex_reports/module_d_stage1_layerwise_emotion_decodability_analysis.md).

### RQ3 — Comparison with Conventional Acoustics

**Question:** How does the pre-specified MERT Layer-12 representation compare with the selected conventional acoustic representation under the same linear probing framework?

**Answer:** On the same frozen population and split, MERT has lower Test MAE and higher R²/Pearson r for both targets. The acoustic baseline itself retains positive held-out decodability.

| Target | Representation | Selected alpha | Test MAE | Test R² | Test Pearson r |
|---|---|---:|---:|---:|---:|
| Valence | MERT Layer 12 (reused C) | 1000 | 0.635144565 | 0.579780312 | 0.768855324 |
| Valence | Conventional Acoustic 51-D | 100 | 0.770499050 | 0.368149823 | 0.609708274 |
| Arousal | MERT Layer 12 (reused C) | 1000 | 0.744556180 | 0.511421706 | 0.715509311 |
| Arousal | Conventional Acoustic 51-D | 10 | 0.860429090 | 0.374161643 | 0.611792750 |

This reuses the existing E comparison table, with source metrics displayed to nine decimal places. The MERT rows reuse C4; MERT was not evaluated again for this comparison. The result is descriptive and specific to the selected representations, probe, and split. It does not prove unique/non-acoustic emotion information or statistical superiority. Shared probing rules do not imply equal dimensionality or effective capacity: MERT has 768 features and the baseline 51.

Read more: [accepted Module E log](docs/research_logs/module_e_conventional_acoustic_baseline.md) · [authoritative RQ3 comparison CSV](outputs/results/module_e_stage3_rq3_comparison.csv) · [E3 comparison report](docs/codex_reports/module_e_stage3_frozen_test_evaluation_and_rq3_comparison.md).

### RQ4 — Tempo/Energy Acoustic Associations

**Question:** How are the observed decoding results associated with Tempo and Energy under the frozen analysis protocol?

Pearson r describes each cue's association with the true target, MERT/acoustic predictions, and residuals, separately on Validation and Test. Tempo is automated `tempo_bpm`; Energy is `rms_mean` from decoded-amplitude audio before MERT input normalization. Residual is `y_true - prediction`; a positive residual means under-prediction.

**Answer:** Under the frozen automated BPM measurement, Tempo provides little descriptive evidence for a simple linear explanation. Energy is positively associated with the true targets and MERT predictions in both partitions; weaker positive associations remain in MERT residuals. Energy therefore provides a plausible partial acoustic explanation, without a complete or causal account.

![Frozen Validation and Test acoustic associations](outputs/figures/module_f_stage2_validation_test_associations.png)

Pearson r does not quantify the fraction of prediction performance explained, and residual association does not identify the remaining information. Small Tempo r does not exclude nonlinear, conditional, or other Tempo dependence. The supporting acoustic predictions track Energy more strongly, with near-zero Energy association in their residuals; this does not mean better overall emotion prediction. **Predictive performance and acoustic association are different questions.**

Test had already been viewed in C–E. F Test is a **frozen descriptive consistency check**, not untouched confirmation or independent replication.

Read more: [accepted Module F log](docs/research_logs/module_f_acoustic_correlate_and_error_analysis.md) · [authoritative Validation/Test comparison CSV](outputs/results/module_f_stage2_validation_test_comparison.csv) · [F2 association report](docs/codex_reports/module_f_stage2_frozen_test_acoustic_association_analysis.md).

## Overall Interpretation

The frozen evidence supports held-out linear emotion decodability from MERT, different accessibility across depth, and stronger decoding from the pre-specified Layer 12 than from the selected compact acoustic baseline. Tempo/Energy analysis constrains that interpretation: Energy-associated structure is a plausible partial acoustic explanation, while Tempo offers little evidence for a simple linear explanation under its automated measurement. These observations describe what the fixed probes can read and how predictions covary with two cues; they do not establish emotion understanding, acoustic independence, or a causal mechanism.

## Limitations and Claim Boundaries

- Decodability is not human-like emotion understanding or a complete measure of information content.
- The depth trajectory does not establish a unique or universally best layer; observed maxima did not select the later comparator.
- A stronger result than the selected 51-D baseline does not establish unique/non-acoustic information or superiority over all conventional acoustic representations.
- Association is not causality or identified causal confounding. Pearson r is not the fraction of performance explained.
- Residual association does not identify the nature of remaining information.
- Small Tempo Pearson r does not rule out other dependence. Automated BPM retains half/double-tempo and beat-tracking uncertainty.
- RMS is an amplitude proxy affected by recording gain/production, not calibrated loudness or Arousal.
- Results are descriptive point estimates from one frozen DEAM excerpt population/split, without significance testing or uncertainty intervals. They do not establish universal, cross-dataset, or full-song generalization.
- Reused C evidence is not independent new evidence. F Test follows prior Test exposure and is not untouched confirmation or independent replication.

The core study does not fine-tune MERT, train foundation models, generate music, or conduct a large-scale human study. Genre remains optional/deferred; Instrumentation is outside core scope. Historical ablation ideas are not automatic next tasks.

## Repository Navigation and Reproducibility

For the complete synthesis, see the [Final Research Report](docs/final_research_report.md) and [Project Evidence Map](outputs/figures/project_evidence_map.svg).

Accepted research narratives are in [research logs](docs/research_logs/); detailed methods, commands, review gates, and verification history are in [Stage reports](docs/codex_reports/). The RQ sections above link directly to the accepted C–F logs and primary numerical sources. The [project workflow](docs/project_workflow.md) documents the collaboration process, while the [Post-Module-D roadmap](docs/research_roadmap_after_module_d.md) preserves historical planning and operational scope.

- [outputs/results/](outputs/results/): frozen metrics, saved predictions, associations, and verification records.
- [outputs/figures/](outputs/figures/): existing research figures.
- [src/](src/) and [scripts/](scripts/): reusable implementation and documented command-line workflows.
- [configs/](configs/) and [notebooks/](notebooks/): configuration and notebook locations.
- [data/metadata/](data/metadata/): the [fixed Sample-ID split](data/metadata/deam_primary_split_seed42.csv); raw data and generated processed/representation caches remain local.

Documented runs used Python 3.10 with CUDA-enabled PyTorch on an NVIDIA RTX 4060 Laptop GPU. See [environment.yml](environment.yml) and the Stage reports for the verified environment and execution details. Raw DEAM audio, large processed data, model weights, and cached MERT embeddings are excluded from version control; private learning notes and context-recovery handoffs remain local.

## Current Project Status

Modules A–F and the RQ1–RQ4 core evidence are complete and frozen. Final Synthesis Stages 1–3 (repository audit, presentation/navigation cleanup, and final research synthesis) are completed and have passed researcher + ChatGPT review. The next task is Stage 4 — personal learning synthesis. Optional research extensions remain deferred; synthesis does not reopen experiments.
