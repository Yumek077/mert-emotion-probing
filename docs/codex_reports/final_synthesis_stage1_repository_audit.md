# Final Synthesis — Stage 1: Final Repository Audit

Audit date: 2026-10-01 (Australia/Sydney). Audited local repository: `mert-emotion-probing`, branch `main`, HEAD `0dfb97f5e5b7b9ec0de7088d4fb7562c2cdff670`.

Scope: a first-time RA/supervisor reader's assessment of the frozen A–F research story, evidence presentation, navigation, organization, and interpretation boundaries. This is not a general code review, a new experimental stage, or the final research synthesis report. The public-facing assessment uses tracked repository contents; no remote GitHub synchronization was performed. The requested audit report is the sole new deliverable, an explicit exception to keeping the existing repository unchanged.

## 1. Executive Assessment

**The repository contains a complete, traceable, appropriately bounded answer to the four accepted operational research questions. Its main weakness is the reader-facing entry point, rather than missing scientific evidence.** The completed research logs explain the method and conclusions well. RQ1 and RQ3 already have readable Markdown result tables; RQ2 and RQ4 already have suitable figures and tables. Frozen numerical artifacts and saved predictions support deeper inspection.

A first-time reader can recover the coherent story, but must assemble it from several documents. The README introduces the topic, lists conceptual questions, and reports completion, yet devotes considerable space to module progress. Its results appear in RQ4 → RQ3 → RQ2 order, with no distinct RQ1 result presentation or direct link to the Module C research log. Important method details and the reason for the progression remain in the logs.

**No genuine `POTENTIAL SCIENTIFIC GAP` was identified in the current accepted claims.** The original broad RQ3/RQ4 intentions need clearer separation from their completed operational answers, but the README already labels them conceptual and explicitly limits the reported conclusions. The improvements below concern presentation and organization. They do not reopen any module or imply that additional experiments are required.

## 2. Research Story Audit

The scientific progression is coherent in the formal records:

**Data/protocol foundation → frozen MERT representations → RQ1 decodability → RQ2 depth trajectory → RQ3 conventional acoustic comparison → RQ4 Tempo/Energy associations → bounded interpretation.**

[Module A](../research_logs/module_a_data_and_representation_protocol.md) defines the population, sample unit, static targets, audio handling, representation levels, and pooling. [Module B](../research_logs/module_b_representation_extraction_and_dataset_construction.md) validates independent single-item extraction, constructs the canonical cache, and creates the fixed Sample-ID split. They are methodological foundations, not two additional emotion research questions.

C asks whether the pre-specified Layer-12 representation supports held-out linear decoding. D asks how accessibility to the same probe varies across depth. E asks whether the selected conventional acoustic representation performs differently under the shared probing rules. F asks whether two simple acoustic descriptors are associated with targets, predictions, and remaining signed errors. The performance comparison does not settle the acoustic explanation, which is why E naturally leads to F. The [Module F log](../research_logs/module_f_acoustic_correlate_and_error_analysis.md#why-this-module-was-needed) already explains this connection clearly.

The README's one-sentence introduction states the topic but gives little motivation for probing a frozen representation or for moving from decoding to acoustic explanations. Its Method Overview shows the MERT prediction pipeline, rather than the complete research progression. It does not specify the 1,744-excerpt population and 58-full-song exclusion, static averaged targets, temporal mean pooling, fixed 1,221/262/261 split, or the initial Layer-12 probe. These details are available in A–F; their omission at the entry point is a `PRESENTATION GAP`.

The same applies to the final conclusion and limitations. The README already gives careful RQ3/RQ4 wording and the depth pattern, but does not assemble them into one compact, bounded project answer. A short narrative and method summary based on the existing logs would be sufficient. A new scientific analysis or a separate overview artifact is unnecessary for Stage 2.

## 3. RQ1–RQ4 Evidence Audit

### RQ1 — Held-out linear decodability

**Question:** Can continuous Valence and Arousal be linearly decoded from the pre-specified frozen, mean-pooled MERT Layer-12 representation?

**Accepted answer:** Both Ridge probes improve over their Train-mean references on the 261 Test excerpts. The authoritative Test R² values are `0.5797803121408952` for Valence and `0.5114217056357836` for Arousal. This supports linear decodability within the frozen population, split, representation, and probe; it does not establish human-like understanding or acoustic independence.

**Visible presentation:** The [Module C held-out table](../research_logs/module_c_basic_probing_and_emotion_decodability.md#held-out-test-evaluation) reports MAE, R², and Pearson r, with a separate [Train-mean reference table](../research_logs/module_c_basic_probing_and_emotion_decodability.md#baseline-comparison). The [C4 report](module_c_stage4_held_out_test_evaluation.md) preserves the original protected Test gate and detailed evaluation history. The README only exposes Layer-12 R² indirectly in its RQ3 comparison paragraph; it lacks a direct C-log entry and an explicit RQ1 result summary.

**Authoritative source:** [C4 Test JSON](../../outputs/results/module_c_stage4_test.json), supported by [saved C4 predictions](../../outputs/results/module_c_stage4_test_predictions.csv), which contain 522 rows covering 261 Test IDs and two targets. The log's rounded numbers are presentation, not the full-precision source.

**Classification:** `KEEP AS-IS` for the existing evidence and tables; `PRESENTATION GAP` for README visibility. No standalone RQ1 figure is necessary.

### RQ2 — Decodability across depth

**Question:** How does linear Valence and Arousal decodability vary across the 13 frozen representation levels?

**Accepted answer:** Both targets are readable early. Valence remains comparatively stable after early gains; Arousal has a broad middle-depth high region followed by a late decline. Greater depth is not monotonically better. Numerical maxima are secondary descriptions, not selected or universally optimal layers.

**Visible presentation:** The README embeds the [complete Test R² trajectory](../../outputs/figures/module_d_stage1_test_r2_trajectory.png) and links the full CSV and technical report. The [Module D log](../research_logs/module_d_layerwise_emotion_decodability_analysis.md#results-across-depth) contains a 13-level Markdown table, explanations of both curves, baseline context, and limitations. Visual inspection found readable labels, both complete curves, a zero reference, and open Layer-12 markers with an explicit reused-evidence caption. This is sufficient presentation; no replacement plot is warranted.

**Authoritative source:** [D layer-wise results CSV](../../outputs/results/module_d_stage1_layerwise_results.csv), [D Test JSON](../../outputs/results/module_d_stage1_test.json), and [saved D Test predictions](../../outputs/results/module_d_stage1_test_predictions.csv). The table contains 26 level/target configurations: 24 new evaluations and two rows explicitly marked as reused authoritative C evidence. Its Layer-12 MAE/R²/r values match C4. The [D report](module_d_stage1_layerwise_emotion_decodability_analysis.md) documents the table/figure production and provenance.

**Classification:** `KEEP AS-IS` for results and visualization. The limited issue is navigation: the README lacks a direct D-log link, and the log's artifact section uses plain code paths and wildcard prefixes rather than clickable primary-result links.

### RQ3 — Comparison with the selected acoustic representation

**Question:** How does Layer-12 emotion decodability compare with the frozen conventional acoustic representation under the same linear probing framework?

**Accepted answer:** MERT has lower Test MAE and higher R²/Pearson r for both targets. The 51-D baseline also has positive held-out decodability. Acoustic Test R² is `0.36814982283192577` for Valence and `0.37416164284512432` for Arousal, compared with the reused C4 MERT values above. The common fitting/selection framework does not equalize 768 versus 51 dimensions or effective capacity.

**Visible presentation:** The [Module E log](../research_logs/module_e_conventional_acoustic_baseline.md#stage-3--official-test-evaluation-and-rq3-comparison) and [E3 report](module_e_stage3_frozen_test_evaluation_and_rq3_comparison.md#official-held-out-comparison) already contain readable four-row comparisons and shared Train-mean references. The README gives R² in prose and links the comparison CSV and E3 report, but does not link the E log or show the existing Markdown comparison. A CSV is itself a tabular artifact; this result is not available only as opaque JSON.

**Authoritative source:** [E3 RQ3 comparison CSV](../../outputs/results/module_e_stage3_rq3_comparison.csv), whose metrics match the [E3 acoustic Test JSON](../../outputs/results/module_e_stage3_test.json) and C4 JSON. [Saved acoustic Test predictions](../../outputs/results/module_e_stage3_test_predictions.csv) contain 522 rows and 261 Test IDs. MERT rows retain their reused-C flag. No new MERT evaluation supplies this comparison.

**Classification:** `KEEP AS-IS` for the completed comparison and existing tables; `PRESENTATION GAP` for the README's compact comparison display. A new RQ3 comparison figure is not required.

### RQ4 — Tempo/Energy association

**Question:** How are the observed decoding results associated with cached Tempo and Energy under the frozen target → prediction → residual analysis?

**Accepted answer:** Automated Tempo supplies little descriptive evidence for a simple linear explanation across either partition. Energy is positively associated with the true targets and MERT predictions, with smaller positive associations remaining in MERT residuals. This supports the accepted qualitative, plausible partial acoustic explanation, not causality, a fraction of performance explained, or identification of remaining information. Stronger Energy association in acoustic predictions does not contradict MERT's stronger RQ3 performance.

**Visible presentation:** The [Module F log](../research_logs/module_f_acoustic_correlate_and_error_analysis.md#stage-2--test-and-validationtest-comparison) contains the accepted Energy table, interpretation, linked sources, and embedded [Validation/Test association figure](../../outputs/figures/module_f_stage2_validation_test_associations.png). The [F2 report](module_f_stage2_frozen_test_acoustic_association_analysis.md#test-results) also provides the full Test Tempo/Energy table. The README links the log and figure but does not embed that figure. Visual inspection confirmed all 40 coefficients across four panels, partition counts, consistent Pearson scale, tiny-value notation, residual sign definition, and explicit prior-Test-exposure and causal/variance-explanation boundaries.

**Authoritative source:** [F1 Validation associations](../../outputs/results/module_f_stage1_validation_associations.csv), [F2 Test associations](../../outputs/results/module_f_stage2_test_associations.csv), and the [F2 comparison CSV](../../outputs/results/module_f_stage2_validation_test_comparison.csv). There are 20 coefficients per partition and 20 comparison rows, with 262 Validation and 261 Test samples. Every comparison value matches its corresponding saved association source. The [F2 analysis data](../../outputs/results/module_f_stage2_test_analysis_data.csv) and report retain the reused prediction and residual provenance.

**Classification:** `KEEP AS-IS` for evidence, visualization, and scientific interpretation; `PRESENTATION GAP` for optional promotion of the existing figure into the README's main results sequence. No missing experiment or visualization was found.

These checks inspected saved metrics, source equality, schemas, row counts, and historical verification records. D/E final verification and both F verification JSONs retain `passed=true`. No verifier, probe, metric-recomputation pipeline, or analysis runner was executed for this audit.

## 4. Navigation / Traceability Audit

**Semantic traceability is strong; the click-through reading path is uneven.** Module/stage prefixes, saved predictions, source identities, reused-evidence flags, and technical reports distinguish authoritative numerical evidence from summaries. C4 remains the original Layer-12 authority; D/E copies do not become independent evidence. F preserves the distinction between model performance and acoustic association.

The automatic local-link inspection checked 41 file/directory Markdown links in pre-existing tracked Markdown documents. All resolved to tracked public files or directories. External URLs and heading anchors were not live-tested; plain code paths are not clickable Markdown links. No broken core local Markdown link was found.

The actual reader paths are:

- **RQ1:** browse the research-log directory to locate C, then use plain artifact paths or the C4 report filename. The README does not provide a direct C-log link.
- **RQ2:** README → figure/CSV/D report works, but the interpretation-focused D log requires directory browsing; its primary result/report paths are plain text.
- **RQ3:** README → comparison CSV/E3 report works. E-log Stage-report links work, but the README omits the concise E-log entry and the log identifies result families mainly by prefix.
- **RQ4:** README → F log → linked source CSVs/report/verification and figure is the most complete current route.

Small navigation edits can make each route equally natural without asking ordinary readers to inspect all provenance. The roles should remain: README for the bounded overview, module logs for accepted interpretation, Stage reports for implementation/gates, and saved artifacts for numerical authority.

Historical states deserve contextual navigation, not retrospective correction. The roadmap already calls itself a frozen planning document dated 2026-09-30. F's final log explicitly reconciles its Stage reports' closed/pending-review states. However, [workflow recovery text](../project_workflow.md#12-new-chat--session-recovery-protocol) still names Module E design as the next objective, and the E log's opening and closing sections still say F has not started. Those statements record earlier moments, but can look current when reached independently. A short historical-status notice linking the completed F record would address this confusion while preserving the original content.

Naming and directory conventions are otherwise consistent. PNG/SVG pairs are alternate formats of the same figures; F1/F2 figures and gate/start records serve distinct chronological roles. They are not cleanup candidates. Private/local caches are explicitly distinguished from public artifacts. No public clutter or directory problem justifies a new hierarchy, refactor, or artifact deletion.

## 5. Claim / Interpretation Boundary Audit

The accepted public-facing conclusions are appropriately bounded. C distinguishes decodability from human-like understanding and limits generalization to the observed excerpt setting. D describes a complete trajectory, notes that Pre-Transformer is already a learned representation, and rejects universal best-layer or significance claims. E explicitly limits superiority to the selected baseline, probe, and split; it rejects unique/non-acoustic information and notes the dimensionality/capacity difference. The README preserves those RQ3 restrictions.

F and the README describe Energy as a plausible partial acoustic explanation. The F log and figure explicitly reject causality and interpreting Pearson r as a fraction of performance explained. Residual association does not identify remaining information. Small Tempo r is qualified by the simple-linear analysis and automated measurement; it does not imply Tempo irrelevance. RMS is an amplitude proxy affected by gain/production, not calibrated loudness or the Arousal label. No causal confounding or acoustic independence is established.

Evidence reuse and partition history are also stated correctly. C4 records its original protected evaluation. D/E reuse C4 Layer-12 evidence. F acknowledges that Validation supported selection and Test was already viewed in C–E; its Test analysis is a frozen descriptive consistency check, not untouched confirmation or independent replication. Existing figure captions reinforce these boundaries.

The main wording risk is the top-level conceptual RQ3 about information “beyond” low-level features and RQ4 about how much performance is “explained” by confounding. These are questions, not asserted findings, and the adjacent conceptual/operational notice plus later caveats prevent classifying them as a genuine unsupported conclusion. Nevertheless, the completed operational questions should be readable without visiting a historical planning document. This is a `PRESENTATION GAP`, not a reason to add conditional, causal, or unique-information experiments.

For a future compact README synthesis, preserve the single-population/split restriction, descriptive point-estimate status, selected-baseline limitation, Tempo/RMS measurement boundaries, residual limitation, and prior-Test-exposure distinction. These qualifications already exist; they need promotion, not a stronger or different interpretation.

## 6. Prioritized Findings

Priority denotes importance to final research presentation, not authorization to make changes now.

1. **F1 — `PRESENTATION GAP` — High: explain the mainline and minimal method at the entry point.** README introduction, Method Overview, and Project Roadmap (lines 3–47) emphasize topic/pipeline/completion without explaining why C → D → E → F forms one investigation or why A/B are foundations. Add concise motivation, progression, and frozen dataset/target/representation/probe context from A/B and F. No new methodological decision is needed.

2. **F2 — `PRESENTATION GAP` — High: give all four answers balanced, ordered visibility.** README Status (lines 75–111) presents F, E, then D; C is indirect. Reuse the existing C/E tables, keep the D figure, and promote the existing F evidence as appropriate, in RQ1 → RQ4 order with immediate source links. This is not a missing-results finding; no new figure is necessary.

3. **F3 — `PRESENTATION GAP` — Medium: expose completed operational RQ3/RQ4 alongside the conceptual intentions.** README Research Questions (lines 5–12) leaves the reader to recover the narrower answered questions from the roadmap. Make the accepted comparison and association questions explicit while retaining the original intentions as history/motivation. Existing E/F conclusions remain unchanged.

4. **F4 — `ORGANIZATION ISSUE` — Medium: make accepted logs and primary artifacts directly navigable.** Add README links to C/D/E logs, and a few clickable primary-result/report/prediction links where C/D/E currently use code paths or wildcard families. C's Artifacts and Source Files and D's Artifacts and Module Outcome are concrete examples. The problem is interaction friction, not absent or conflicting authority. Follow F's existing route rather than creating an evidence-map file or a new directory.

5. **F5 — `ORGANIZATION ISSUE` — Medium: distinguish historical next steps from current status at recovery entry points.** Workflow line 311 and E-log lines 9/73 can be read as active tasks despite current README/F completion. Add a short as-of/historical notice and current F-log pointer. Preserve the frozen roadmap and Stage-report gate/review statements; do not reopen E/F or rewrite their chronology.

The following should be retained:

- **K1 — `KEEP AS-IS` — High:** accepted protocols, numerical artifacts, predictions, evidence reuse, and interpretation boundaries. No genuine `POTENTIAL SCIENTIFIC GAP` is flagged.
- **K2 — `KEEP AS-IS` — Medium:** existing D/F figures and C/E/F tables. They are readable and sufficiently sourced through the logs/reports; missing README prominence does not justify replacing them.
- **K3 — `KEEP AS-IS` — Medium:** established directories, descriptive filenames, historical records, PNG/SVG pairs, and private/cache exclusions. There is no research-presentation basis for a structural refactor or deletion sweep.

## 7. Proposed Stage 2 Cleanup Scope

After researcher + ChatGPT review, the minimal sufficient proposal is **one focused README presentation revision and a small set of navigation-only additions to existing documents**:

1. Explain the motivation and evidence progression in a short paragraph; add the minimal frozen method context. Use A/B for population/representation/split and C–F for probing, comparison, and association roles.
2. Present the accepted operational questions and results in RQ order. Reuse the C/E human-readable result content, the unchanged D trajectory figure, and the unchanged F comparison figure or compact existing result content. Link full-precision sources next to displayed evidence and retain C-evidence reuse labels. No new scientific figure/table file is required.
3. Give each RQ a direct accepted-log entry plus a primary artifact/report route. Add only the missing clickable links in C/D/E. Retain technical detail in Stage reports instead of copying it into the README.
4. Add brief historical/current-status navigation notices to workflow recovery and the E log. Preserve historical statements and the already dated frozen roadmap. Consolidate duplicated README progress/status wording around the current Final Synthesis state.
5. Bring forward the existing concise limitations: decodability versus understanding; selected-baseline comparison versus unique information; association versus causality/performance share; imperfect Tempo/RMS measurements; one DEAM excerpt split; and reused/prior-exposed Test evidence.

**REQUIRES RESEARCHER REVIEW:** the presentation choice for the original conceptual RQ3/RQ4 wording—whether it remains adjacent to the accepted operational formulations or is moved into a clearly labeled original-intentions note. This is an editorial choice about preserved history, not redesign of the RQs or their answers. The cleanup scope itself remains subject to the Stage 1 review gate.

Acceptance should be simple: a new reader can find each operational question, accepted answer, readable evidence, and authoritative source; a supervisor can continue to the relevant log/report without browsing unrelated history; all frozen results and scientific boundaries remain intact. Do not add experiments, inferential statistics, new analyses, cache regeneration, refactors, or a directory reorganization. The final research report, Project Evidence Map, and personal synthesis note belong to later stages and are excluded from this proposal's implementation.

### What I should now be able to explain

- Why all four operational questions are complete even though the README can be improved.
- Why A/B support the research chain rather than add two emotion questions.
- Which saved artifact is authoritative for each answer, and where C evidence is reused.
- Why a baseline performance comparison and acoustic associations answer different questions.
- Why the proposed cleanup changes visibility/navigation rather than scientific scope.

## 8. Repository Integrity Check

The integrity baseline was taken before report creation: HEAD as recorded above, 100 tracked files, a clean tracked working tree and index, and four pre-existing untracked `docs/.obsidian/` files. SHA-256 hashes were recorded in session memory for all tracked file contents and the Git index. A command-scoped `safe.directory` setting permitted read-only Git inspection; global Git configuration was not changed.

Post-write integrity checks confirm:

- All 100 pre-existing tracked files remain byte-for-byte unchanged. Tracked working-tree and index diffs are empty; HEAD and the index hash are unchanged.
- The only added deliverable is this untracked audit report. The four pre-existing untracked `docs/.obsidian/` files remain outside the deliverable scope; their contents were not read or edited.
- No scientific experiment, fitting, inference, Test reevaluation, selection, extraction, or result regeneration was run. The audit only read documents/artifacts, inspected existing figures, compared saved source values, checked links, and checked integrity.
- No stage, commit, push, cleanup, or Stage 2 action occurred.
- No private handoff or personal note was copied into public deliverables. `handoffs/`, `personal_notes/`, raw audio/data, processed acoustic cache, and MERT embedding/model caches remain excluded. Tracked files in raw/processed/embedding directories are directory placeholders only; no `docs/.obsidian/` file is tracked.

**“Unchanged” applies to all pre-existing tracked files and Git history/index. The filesystem is not literally unchanged because the explicitly requested audit report was added.** No existing repository file was modified.

Stage 1 stops here. Await researcher + ChatGPT review before any cleanup or later synthesis deliverable.
