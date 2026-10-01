# Final Synthesis — Stage 3: Final Research Synthesis Design

Date: 2026-10-01 (Australia/Sydney). **Design only; researcher + ChatGPT review is pending.**

## Checkpoint and Recovered State

Stage 1/2 closure was approved by the researcher. README Current Project Status was updated without further substantive editing. Exactly seven public documentation files were committed as `043434651c12c9f5c4638837385290a32a297935`, with message `Complete Final Synthesis repository audit and presentation cleanup`. The push to `origin/main` succeeded; local HEAD, the local remote-tracking ref, and live remote main were verified to have that same hash. Private/local material and the four pre-existing untracked Obsidian files were excluded.

The [README](../../README.md) is the accepted landing page. [A](../research_logs/module_a_data_and_representation_protocol.md) and [B](../research_logs/module_b_representation_extraction_and_dataset_construction.md) supply the methodological foundation; [C](../research_logs/module_c_basic_probing_and_emotion_decodability.md), [D](../research_logs/module_d_layerwise_emotion_decodability_analysis.md), [E](../research_logs/module_e_conventional_acoustic_baseline.md), and [F](../research_logs/module_f_acoustic_correlate_and_error_analysis.md) supply accepted RQ1–RQ4 evidence and interpretation. The [frozen roadmap](../research_roadmap_after_module_d.md) preserves scope and motivation; the [Stage 1 audit](final_synthesis_stage1_repository_audit.md) and [Stage 2 report](final_synthesis_stage2_repository_cleanup.md) explain the presentation decisions. Their earlier pending-review/next-step statements remain historical; the current closure is recorded in the checkpoint README and this design report.

All A–F scientific evidence, operational RQs, methodological decisions, and interpretation boundaries remain frozen. This document proposes two future synthesis artifacts; it creates neither artifact and contains no draft mini-paper sections or rendered Evidence Map.

## 1. Recommended Role of the Final Research Report

Produce an English, self-contained **concise research report / mini-paper** titled *Probing Musical Emotion in Pretrained Music Representations*. Aim for approximately 3,000–4,000 words, with a working prose budget around 3,500. Completeness of the method, reasoning, and limitations takes precedence over a mechanical word target.

The report should let a research reader assess one study: why linear decodability matters, what was tested, how the questions build on each other, what was observed, and where interpretation stops. Organize around the questions and evidence, using module names primarily in source links. A/B explain the foundation; they do not become two additional research questions or implementation-results chapters.

The broader motivation about information beyond simple acoustics and possible confounding belongs in the Introduction. The Research Questions section uses the four accepted operational formulations from the README. Explain their deliberate narrowing once, and carry that distinction into Discussion and Conclusion. Do not describe the broader unique-information or causal/performance-explanation aspiration as fully answered.

The report should stand alone without requiring all logs, while linking them for deeper inspection. It should synthesize and interpret, rather than repeat each module's objective/stages/verification/outcome template. No additional experiment or literature-review project is needed. Retain concise dataset/model attribution using the sources already documented in the repository; any background citation must support only its actual background claim, not extend the project's empirical evidence.

## 2. Recommended Structure and Section-Level Content

Use eight main sections. The budgets below are planning guides, totaling approximately 3,520 words; tables, captions, and a short references/source note need not be forced into that total.

| Section | Question it should answer | Planned content | Approximate prose words |
|---|---|---|---:|
| Abstract | What was studied and what is the bounded answer? | Brief motivation, frozen DEAM/MERT/Ridge setting, the four connected findings, and the principal interpretation boundary. Write last; avoid implementation history. | 180 |
| Introduction / Motivation | Why probe emotion, and why are acoustic explanations relevant? | Explain restricted linear readability of a pretrained representation, the broader acoustic-information motivation, and why depth/comparison/association are successive questions. Introduce the Evidence Map near the end. | 450 |
| Research Questions | What exactly did the completed experiments ask? | State operational RQ1–RQ4 in their accepted order and explain their progression. Distinguish them from broader conceptual aspirations; identify A/B as foundation. | 170 |
| Method | What population, inputs, comparisons, and analyses produced the evidence? | Data/representations; probing/evaluation; acoustic association and evidence provenance. Include only choices necessary to interpret or reproduce the scientific comparison. | 750 |
| Results | What was observed for each question? | Four subsections in RQ1 → RQ4 order. Use the two planned result tables and existing D/F figures below. Separate observations from explanation and keep partitions/reused evidence explicit. | 850 |
| Discussion | How do these observations form one study? | Connect accessibility, target-specific depth patterns, selected-baseline comparison, and acoustic associations. Explain why stronger prediction and stronger Energy association are different findings. Discuss the accepted partial explanation without identifying a mechanism or remaining information. | 600 |
| Limitations | Which claims and settings remain outside the evidence? | Restricted probe, one excerpt population/split, representation/capacity comparison, point estimates, measurement/context/pooling limits, and prior Test exposure. Describe limits without proposing an expanded experiment programme. | 400 |
| Conclusion | What can a reader take away? | A compact integrated answer to the operational questions, bounded to this setting. No new hypothesis, metric, ranking, or causal claim. | 120 |

### Essential Method Content

Keep Method to approximately three coherent subsections:

- **Data and representations:** one DEAM item per sample; 1,744 primary approximately 45-second excerpts, excluding 58 metadata-defined full songs; static averaged Valence/Arousal on the original scale. Preserve actual decoded duration, mono conversion and 24-kHz resampling. Describe frozen MERT-v1-95M, independent single-item extraction, temporal mean pooling, 13 levels and 768 dimensions per level. Explain that Pre-Transformer is already a learned frontend/encoder-input representation. Summarize the fixed 51-D waveform-derived Tempo/RMS/MFCC/spectral recipe, including the amplitude-preserving RMS path, and link its exact extraction specification.
- **Probing and evaluation:** Sample-ID alignment and the fixed seed-42 1,221/262/261 split; independent target regressions; Train-only StandardScaler/Ridge fitting; the inherited nine-alpha grid, Validation R² selection and exact-tie rule; final Train-only fitting, no Train + Validation refit or post-Test tuning. Define MAE/R²/Pearson r and the Train-mean reference. State why pre-specified Layer 12 remains the comparator after the descriptive depth analysis, and why equal probing rules do not equalize 768 versus 51 dimensions/effective capacity.
- **Acoustic association and provenance:** cached `tempo_bpm` and `rms_mean`; Pearson associations with true target, both predictions and both signed residuals, separately on Validation/Test. Define residual as `y_true - prediction` and positive residual as under-prediction. State that F reuses saved predictions and performs no new model fitting/evaluation. Include the partition-history paragraph described in Section 7.

One sentence can explain independent extraction as a representation-consistency choice; the detailed failed padded-batch investigation stays in B. Software compatibility experiments, GPU benchmarks, cache parts/serialization, hashes, individual assertions, and gate commands do not occupy the report body. Link them through the logs/Stage reports.

## 3. Evidence, Table, and Figure Inclusion Plan

Recommend **two result tables and three figures**, including the future qualitative overview. Number them by order of appearance; retain existing scientific figure bytes and add report-level captions outside the images.

| Planned item | Report location and purpose | Frozen basis / content |
|---|---|---|
| Figure 1 — Project Evidence Map | Introduction: orient the reader to the study's logical progression. | A new presentation of the accepted evidence chain, with no new quantitative result; design in Section 4. |
| Table 1 — Frozen Test decoding comparison | First introduced under RQ1; referenced again under RQ3 to avoid duplicating MERT results. | Combine the existing C/E content into six rows: each target × MERT Layer 12, selected Acoustic 51-D, and shared Train-mean reference. Columns: target, representation/reference, selected alpha where applicable, Test MAE, Test R², Pearson r. |
| Figure 2 — Full depth trajectory | RQ2: show both targets across all 13 levels. | Reuse the [D trajectory PNG](../../outputs/figures/module_d_stage1_test_r2_trajectory.png) unchanged, sourced by the [full D CSV](../../outputs/results/module_d_stage1_layerwise_results.csv). Preserve the open reused-Layer-12 markers and existing provenance caption. |
| Figure 3 — Validation/Test acoustic associations | RQ4: show both cues, all five objects, and both partitions. | Reuse the [F comparison PNG](../../outputs/figures/module_f_stage2_validation_test_associations.png) unchanged. It already covers the full 40 coefficients and preserves the residual definition and prior-Test-exposure boundary. |
| Table 2 — Energy associations across partitions | RQ4: make the main target/prediction/residual pattern readable at the existing six-decimal precision. | Reuse the five-row Energy table in the accepted F log: true target, MERT/acoustic predictions, and MERT/acoustic residuals; columns are target × Validation/Test. |

**Table 1 authority:** use the [C4 Test JSON](../../outputs/results/module_c_stage4_test.json) for MERT and references, the [E3 Test JSON](../../outputs/results/module_e_stage3_test.json) for acoustic metrics/alphas, and the [existing E comparison CSV](../../outputs/results/module_e_stage3_rq3_comparison.csv) for comparison provenance. Retain nine-decimal metric display consistent with the current README/C/E tables. Mark MERT as reused C evidence; the same MERT rows support RQ1 and RQ3. Explain shared references, 261 Test samples, undefined constant-reference correlation (`null`, not zero), and non-applicable reference alpha. Combining accepted rows is presentation work, not recomputation.

**Table 2 / Figure 3 authority:** [F1 Validation CSV](../../outputs/results/module_f_stage1_validation_associations.csv), [F2 Test CSV](../../outputs/results/module_f_stage2_test_associations.csv), and [F comparison CSV](../../outputs/results/module_f_stage2_validation_test_comparison.csv). Keep 262 Validation and 261 Test samples separate. Format saved coefficients; do not calculate new associations or a performance-explained quantity.

Results should use Table 1's MERT/reference rows for RQ1, then the full D curve for RQ2, then Table 1's representation comparison for RQ3, then F's cue/target/prediction/residual distinctions for RQ4. Acoustic rows in the shared table should be identified as the later RQ3 comparator, rather than implied to belong to the original C experiment.

**Link rather than duplicate:** the full 26-configuration D numerical/alpha table; all Validation candidate-grid results; exhaustive source coefficients already visible in the F figure/CSV; per-sample predictions; extraction diagnostics; verification JSON; software/version loading history; commands; Stage review chronology. The F1 standalone Validation figure remains historical provenance, while the two-partition F2 figure is sufficient for the report. Existing SVG counterparts remain available; do not crop, retitle internally, regenerate, or replace any scientific figure.

No new RQ1/RQ3 chart, performance-difference calculation, layer-ranking panel, uncertainty band, supplementary experiment, or separate CSV export is needed.

## 4. Recommended Role and Design of the Project Evidence Map

The map is a **research-facing overview of how evidence accumulates**. It helps a first-time reader distinguish A/B foundation, four successive questions, and the bounded interpretation. Its arrows express analytical progression; the caption must acknowledge shared data and reused evidence rather than imply seven independent tests or a serial prediction pipeline.

Recommend one simple vertical flow with three visually restrained groups: **Foundation (A/B)**, **Research evidence (RQ1–RQ4/C–F)**, and **Bounded interpretation**. Use seven nodes, short titles, and one or two concise annotation lines per node. A consistent type size and explicit labels should carry meaning independently of color. Keep the selected acoustic representation inside the RQ3 description, with its waveform-derived origin clear; no elaborate architecture branch is necessary.

| Node | Intended title / role | Content to communicate |
|---|---|---|
| 1 | DEAM / data protocol — A | Fixed excerpt population, one sample per item, static Valence/Arousal targets. |
| 2 | Frozen representations / dataset construction — B | MERT, temporal mean pooling, verified levels, Sample-ID alignment and fixed split. The split was created in B, not A. |
| 3 | RQ1 — Can decode? | Pre-specified Layer-12 Ridge decoding improves the Train-mean reference for both targets. |
| 4 | RQ2 — Where across depth? | Early readability; comparatively stable Valence; broad middle-depth Arousal high region and late decline. |
| 5 | RQ3 — Compared with selected acoustics | Stronger Layer-12 held-out metrics than the selected waveform-derived 51-D representation under shared probing rules. |
| 6 | RQ4 — Tempo/Energy associations | Little simple-linear Tempo evidence under automated BPM; Energy provides a plausible partial acoustic explanation. |
| 7 | Bounded interpretation | Linear accessibility in this setting; descriptive comparison/association. No proof of human-like understanding, unique/non-acoustic information, or causality. |

Add a short caption/footer covering the single frozen DEAM split, C evidence reuse, and F's prior-exposed Test consistency check. The map's final boundary should remain visually part of the chain, not a disclaimer detached from the result.

Omit metric values, alpha grids, cache shapes/file hashes, code paths, software/hardware details, all 13 individual layers, individual coefficients, significance symbols, and Genre/Instrumentation branches. Avoid ranking badges or causal arrows. Exact methods, metrics, and limitations belong in the report.

Recommend an editable SVG plus a PNG export of the same artwork, following the existing paired figure convention. Embed the PNG once in the final report, after the Introduction's progression explanation. For the already approved README, recommend only a small final-report link and optional map link after the files exist; another large embedded figure is unnecessary. No README change or map creation occurs in this design stage.

## 5. Proposed Public Artifact Paths

These are proposed paths relative to the Git root; none is created by this design task:

| Artifact | Recommended path | Reason |
|---|---|---|
| Final Research Report | `docs/final_research_report.md` | A standalone English/GitHub-readable research document belongs in the existing general `docs/` directory, separate from module logs and Codex process reports. |
| Evidence Map, editable source | `outputs/figures/project_evidence_map.svg` | Uses the established public figure location and descriptive naming; supports editable/vector reuse. |
| Evidence Map, display export | `outputs/figures/project_evidence_map.png` | Matches existing PNG/SVG pairs and the README/report's current image-embedding convention. |

The present design record remains `docs/codex_reports/final_synthesis_stage3_design.md`. Tables live inside the report's Markdown; no new table files or directory hierarchy are proposed. A PDF/DOCX/LaTeX version is not part of the recommended minimal implementation.

## 6. Document Responsibilities and Avoiding Duplication

| Layer | Responsibility |
|---|---|
| README | Approved landing page: concise motivation, methods, operational questions, principal evidence and boundaries. Preserve its accepted substance; later add only agreed links to completed synthesis artifacts. |
| Final Research Report | One self-contained scientific argument, with sufficient method and integrated Results/Discussion/Limitations for a research reader. It adds synthesis depth, not another progress summary. |
| Module research logs | Accepted module-specific decisions, interpretations and handoffs. Link them for detail; preserve their content and historical scope. |
| Stage reports | Detailed implementation, verification, gates, commands and chronology. Keep technical history out of the mini-paper body unless necessary to explain a method choice. |
| Public result artifacts | Full-precision numerical authority, saved predictions and association data. Link them beside report tables/figure captions; formatting does not create a new authority. |
| Local raw data / caches | Reproduction inputs and derived caches, excluded from public delivery. Explain availability/exclusions through existing public records; do not copy or link private handoffs/learning notes as research evidence. |

Use source links where a claim/table/figure appears, plus a short final sources/reproducibility note if helpful. Avoid a new artifact registry or an appendix that reproduces all A–F reports. Stage review vocabulary should not interrupt the scientific narrative; the important Test/provenance facts still belong in Method and captions.

## 7. Scientific Interpretation Safeguards

The report and map must preserve these distinctions in the relevant result/discussion passages, not only in a final limitations list:

| Evidence | Required boundary |
|---|---|
| Held-out linear decoding | Decodability is not human-like emotion understanding, a causal emotion mechanism, or complete information content. |
| Depth trajectory | Describes accessibility to the fixed regularized linear probe; numerical maxima do not establish universally optimal or statistically superior layers. |
| MERT versus 51-D acoustics | Describes the selected representations/probe/split. It does not establish unique/non-acoustic information or exhaustive superiority over conventional acoustics; dimensions/effective capacity differ. |
| Tempo/Energy association | Pearson association does not establish causality or identified causal confounding. |
| Pearson r | Does not measure a fraction of decoding performance explained; neither r nor a newly calculated r² should be presented as that fraction. |
| Residual association | Does not identify the nature of remaining information, whether present or absent. |
| Small Tempo r | Gives limited evidence for simple linear association under automated BPM, not absence of nonlinear/conditional/other dependence. |
| One population/split | Does not establish universal, cross-dataset, or full-song generalization. Point estimates do not provide significance/uncertainty claims. |
| F Test | Is a frozen descriptive consistency check after prior Test exposure, not untouched confirmation or independent replication. |

**Evidence-history paragraph:** distinguish C4's original protected Layer-12 Test evaluation from D's complete trajectory, E's acoustic comparison and F's later association analysis. D contains 24 new level/target evaluations plus two reused C Layer-12 results. E evaluates the frozen acoustic branch while reusing C MERT evidence; the Test partition had already been viewed in C/D. F reuses C/E predictions, with Validation previously involved in selection and Test viewed in C–E. None of that reuse becomes independent new MERT evidence. Preserve Train-only fitting, Validation-only selection, no post-Test tuning, and the pre-specified Layer-12 comparator.

State this succinctly in Method, repeat the necessary reuse label in Table 1/Figure 2 captions, and identify F's descriptive status in its Results/Figure 3 caption. Discussion should explicitly distinguish prediction quality from Energy association strength.

Measurement/representation limits remain part of the reasoning: RMS is decoded amplitude affected by recording gain/production, not calibrated loudness or Arousal; automated BPM has beat-tracking and half/double-tempo ambiguity. Static targets and mean pooling limit temporal interpretation, and A documents the approximately 45-second extraction context relative to the model's shorter pretraining context. These are existing caveats, not prompts for new ablations. Genre stays optional/deferred and Instrumentation outside core scope.

## 8. Minimal Stage 3 Implementation Plan

Only after this design is approved, implement one coherent synthesis pass:

1. Write the report from accepted A–F records, copying/formatting numerical content from the named frozen sources. Use the proposed structure, two tables and unchanged D/F figures. Create the qualitative SVG map and its matching PNG export. Do not derive new metrics or run scientific pipelines.
2. Verify report wording, displayed values, source links/captions, readable map labels, and preservation of all frozen scientific files. Check that the evidence is distinguishable from interpretation and that partition/evidence reuse is accurate. Add only the approved minimal README links after their targets exist, with no further substantive landing-page polish.
3. Present the report/map and a concise implementation record for researcher + ChatGPT review, then stop. No automatic Stage 4, personal synthesis, experiment, or further Git checkpoint is included.

The future work needs three public artifacts, with existing figures referenced in place. It does not need extraction, fitting, inference, metric recomputation, significance/uncertainty/conditional/causal analysis, source refactoring, directory changes, or historical-document rewrites.

**Current design integrity:** after the successful Stage 2 checkpoint, this task creates only this untracked design report. All 102 checkpoint-tracked file contents and the index remain unchanged during design. The report body, map SVG/PNG and personal synthesis remain absent. The Stage 3 design is not staged, committed, or pushed. The four original untracked Obsidian files remain outside the task; no scientific experiment or new result was produced.

## 9. Decisions Requiring Researcher + ChatGPT Review

Recommend approving the following package before implementation:

- English Markdown mini-paper at `docs/final_research_report.md`, approximately 3,000–4,000 words, using the eight-section structure and modest source/reproducibility note.
- Two main result tables: one combined six-row Test performance table serving RQ1/RQ3, and the existing F Energy table; unchanged D/F result figures.
- One seven-node vertical Evidence Map with foundation/evidence/bounded-interpretation groups, exported as SVG/PNG; embed in the report and use links in the already accepted README.
- Presentation-only implementation followed by researcher + ChatGPT review, with existing scientific/provenance artifacts preserved.

These are editorial/artifact-design approvals. No unresolved scientific conflict or new researcher decision about population, split, targets, layers, features, probe, statistics, or accepted interpretation was identified or requested. Different preferences for the proposed paths/layout can be resolved within presentation scope before implementation.

### What I should now be able to explain

- Why the final report is one coherent study rather than six module summaries.
- How its table/figure choices provide every RQ's evidence while avoiding duplicate C results.
- Why the Evidence Map depicts research progression and bounded interpretation, rather than adding quantitative evidence.
- Where original Test evidence ends and later reuse/descriptive analysis begins.

**Stop gate:** await researcher + ChatGPT review of this design. No final report body, Evidence Map, README synthesis-artifact link, personal synthesis, Stage 4 work, or Stage 3 design commit is authorized or performed here.
