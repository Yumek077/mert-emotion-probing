# Module D Stage 1 — Layer-wise Emotion Decodability Analysis

## Status and Objective

Executed locally on 2026-09-30. Implementation and verification passed. The researcher confirmed completion of ChatGPT research interpretation review and the researcher learning checkpoint on 2026-09-30. The Module D research protocol and results are accepted for project use.

**RQ2: How does linear Valence and Arousal decodability vary across the 13 frozen MERT representation levels?**

This coherent Stage extends Module C's verified framework along one representation-level dimension. It reports complete depth trajectories rather than selecting a winning layer. The implementation comprises Train/Validation selection, a persisted protected Test gate, and one unified held-out sweep. No MERT extraction or model loading was run.

## Repository Recovery and Frozen Context

Read `docs/project_workflow.md`, all three Module A/B/C research logs, Module C runners and verifiers, `src/mert_emotion_probing/probing.py`, Module C result artifacts, the canonical cache manifest, environment specification, and existing artifact/private-note conventions. The Module C Stage C4 report was also inspected. No applicable AGENTS.md was found. The README was stale (Module A only); the authoritative research logs and result artifacts established that A/B/C were complete. No frozen research-protocol conflict was found.

The actual project Git root is the nested `mert-emotion-probing/`, inside the workspace's outer Git directory. An initial outer-root Git status encountered ownership protection; a per-command safe-directory setting allowed inspection without changing global Git configuration. Subsequent Git operations used the actual nested project root.

The starting project working tree contained researcher changes to `.gitignore` and `docs/project_workflow.md`, plus untracked `docs/.obsidian/`. These were preserved and excluded from this task's intended changes. No new directory hierarchy was created: scripts, results, figures, reports, research logs, personal notes, and handoffs use existing directories.

## Inherited Protocol

- DEAM primary population: 1,744 approximately 45-second excerpts, excluding the 58 metadata-defined full songs.
- Canonical frozen MERT-v1-95M cache: FP32 `[1744, 13, 768]`, independently extracted single items, evaluation mode, gradients disabled, temporal mean pooling. Input cache SHA-256: `8174fc08a0155af6436fc0a6e1f459d0d6965df94268804612fda339adcf55bc`.
- Index 0 = Pre-Transformer; indices 1–12 = corresponding Transformer layers; no concatenation.
- Explicit Sample-ID joins align cache, original-scale `valence_mean` / `arousal_mean`, and fixed split.
- Train 1,221 / Validation 262 / Test 261. No split creation, reshuffle, or sample selection.
- Independent Ridge probes with intercept, Cholesky solver, Train-only StandardScaler, and unstandardized targets.
- Alpha grid: `0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000`.
- Independent alpha selection for each level × target using maximum Validation R²; actual exact ties favor larger alpha.
- Final scaler, Ridge and baseline still fit on Train only. Validation is never merged into final fitting.
- MAE, R², Pearson r, and constant Train-target-mean baseline. Undefined baseline correlations are JSON null.
- Point estimates only; no bootstrap, interval, significance test, multi-seed averaging or cross-validation.

## Implementation and Reuse

`src/mert_emotion_probing/probing.py` received a small backward-compatible refactor: `_load_cache` and `assemble_probing_dataset` accept a representation-level name, defaulting to the original Layer 12. The full metadata mapping is checked before resolving the requested name. Diagnostic text was generalized from Layer 12 to the selected level. Module C's default behavior remains unchanged.

All fitting, scaler handling, metrics, exact-tie selection, baseline construction, and split-view functions are reused unchanged: `fit_select_validate`, `fit_train_evaluate_test`, `train_validation_view`, `train_test_view`, and the original metric and selection helpers. Feature conversion to float64 for Ridge remains Module C behavior; the canonical representations remain FP32. No frozen representation was recomputed.

`scripts/run_layerwise_probing.py` provides `select` and `test` commands. It reuses Module C's atomic serialization, timestamp and SHA-256 helpers. It saves per-configuration diagnostics, candidate scores, selected Validation predictions, and final Test predictions. It refuses to overwrite selection artifacts or silently repeat a started Test sweep. A start marker is written before any new Test prediction. A genuine invalidating error would require a traceable correction record before recovery; none occurred.

`scripts/verify_layerwise_probing_results.py` provides pre-Test `selection` verification and post-Test `final` verification. The latter recomputes metrics from saved predictions and does not refit Test models. The plot is generated by the runner from the complete final table using Matplotlib.

## Protected Test Gate and Chronology

- Selection saved: `2026-09-30T09:51:05.196072+00:00`.
- Verification/freeze saved: `2026-09-30T09:51:26.769799+00:00`.
- Unified Test sweep started: `2026-09-30T09:51:48.180783+00:00`.
- Test results saved: `2026-09-30T09:51:49.901562+00:00`.

Before the Test sweep, all 26 selected configurations, complete protocol, metrics, visualization policy, input identities, Module C artifact identities, runner, verifier, serialization helper and probing source hashes were fixed. The gate binds selection and Validation-prediction hashes. The Test command checks them again and verifies selection artifacts before opening the gate. Final verification checks the chronological ordering and artifact hashes.

Selection SHA-256: `ea55c7bf7fca617139f43b23d3982a00e6a2c1d81b1a4c185a7d268044e1db12`.

The primary plot policy was frozen before Test: both complete trajectories, Pre-Transformer through Layer 12, Test R², unsmoothed connected point estimates, zero reference, no uncertainty bands, explicit reused Layer-12 markers, PNG and SVG. No Test-derived layer exclusion, metric choice or protocol revision occurred.

**Access boundary:** the existing Module C loader reads the monolithic cache and mapping table and checks finite values, membership and identity for the whole population. Therefore this is not a claim that no Test bytes were ever loaded. Before the gate, Test received only integrity/alignment handling; Test values never entered scaler fitting, Ridge fitting, baseline means, alpha selection, prediction or metric calculation. Prior Layer-12 Test evidence was already known from Module C context and was not used to design or select any other configuration. Phase 1 hashes the C4 files for provenance but does not calculate C4 performance.

## Pre-Test Verification

All passed:

1. Exactly 26 unique level × target configurations and nine candidate scores per configuration (234 candidate fits).
2. All 13 metadata-resolved levels, `[1744, 13, 768]` FP32 cache, 768 features per level, finite representations and targets, no all-zero selected sample vector, 1,744 unique IDs, exact population membership and 58 exclusions.
3. Frozen split counts, disjoint roles, exact target-column mapping, explicit ID joins, and comparison of every assembled level against the raw canonical tensor via an independent ID lookup.
4. Actual assembly repeated with cache, labels and split rows reversed; feature, target and split outputs remained identical.
5. Train-only scaler sample count, mean and variance checks executed by the reused fitting functions; Ridge and baseline use only Train arrays.
6. Every complete selection repeated after replacing all assembled Test features and targets with NaN, while patching Test-view and Test-evaluation entry points to raise. All 26 result dictionaries and saved Validation prediction vectors matched exactly. This also verifies deterministic repetition without invoking Test inference.
7. Independent MAE and R² formulas and NumPy correlation reproduce all selected Validation metrics and baseline metrics at absolute tolerance `1e-12`; true labels and Train baseline means are independently recovered by Sample ID.
8. Alpha choices independently checked by ordering `(Validation R², alpha)`; synthetic exact-tie and one-ULP non-tie cases both pass.
9. Layer-12 candidate scores and selected Validation metrics agree with authoritative C3. An additional compatibility check confirmed **exact equality**, beyond the declared verification tolerance. Default Layer-12 assembly exactly equals explicit Layer-12 assembly.
10. Validation CSV contains 6,812 rows = 26 × 262, unique Sample-ID/level/target keys and no Test predictions.

## Selected Alphas

Valence selected 1000 at every level. Arousal selected 100 at indices 0, 1 and 2, and 1000 at indices 3–12. These are independent Validation outcomes, not a shared-alpha constraint. No actual selection tie occurred.

## Complete Numerical Results

The table preserves the original target scale. Values below are rounded to nine decimal places; JSON/CSV artifacts retain full serialized precision. Level 12 Test results are reused authoritative Module C evidence.

| Level | Target | Alpha | Val MAE | Val R² | Val r | Test MAE | Test R² | Test r |
|---|---|---|---|---|---|---|---|---|
| 0 | valence | 1000 | 0.609707532 | 0.551373247 | 0.745879500 | 0.657102476 | 0.526720395 | 0.728280433 |
| 0 | arousal | 100 | 0.664717924 | 0.512617163 | 0.720798916 | 0.730428470 | 0.536628663 | 0.735020138 |
| 1 | valence | 1000 | 0.604672107 | 0.567509201 | 0.756888424 | 0.638655044 | 0.559071673 | 0.750148493 |
| 1 | arousal | 100 | 0.658231707 | 0.536775430 | 0.738889618 | 0.703240092 | 0.557815064 | 0.749107631 |
| 2 | valence | 1000 | 0.610802203 | 0.560851999 | 0.751542446 | 0.627566275 | 0.568835066 | 0.756175519 |
| 2 | arousal | 100 | 0.625458514 | 0.584061872 | 0.771458868 | 0.684397053 | 0.554008616 | 0.748884726 |
| 3 | valence | 1000 | 0.617634149 | 0.565806013 | 0.755785239 | 0.650660425 | 0.556921405 | 0.748489227 |
| 3 | arousal | 1000 | 0.601421670 | 0.599798028 | 0.776793276 | 0.652016448 | 0.605582142 | 0.778434817 |
| 4 | valence | 1000 | 0.609995547 | 0.571104006 | 0.757859059 | 0.653755350 | 0.554643527 | 0.746891250 |
| 4 | arousal | 1000 | 0.593854517 | 0.610853514 | 0.783657984 | 0.654565701 | 0.611146533 | 0.781930356 |
| 5 | valence | 1000 | 0.601518653 | 0.580756943 | 0.764901115 | 0.629677391 | 0.582014093 | 0.765366790 |
| 5 | arousal | 1000 | 0.600388720 | 0.618445014 | 0.787259930 | 0.669116511 | 0.598320849 | 0.775054500 |
| 6 | valence | 1000 | 0.597326847 | 0.575309470 | 0.760716275 | 0.623049168 | 0.585611861 | 0.767594892 |
| 6 | arousal | 1000 | 0.582486344 | 0.648392666 | 0.806348357 | 0.663754725 | 0.601162974 | 0.775811146 |
| 7 | valence | 1000 | 0.590434451 | 0.577487017 | 0.762569552 | 0.627276085 | 0.583344063 | 0.765139296 |
| 7 | arousal | 1000 | 0.592944717 | 0.638798014 | 0.799842279 | 0.659528023 | 0.611004060 | 0.781681464 |
| 8 | valence | 1000 | 0.598474612 | 0.563357188 | 0.752098075 | 0.640050844 | 0.570337948 | 0.756424372 |
| 8 | arousal | 1000 | 0.597816463 | 0.630903170 | 0.794663023 | 0.670281419 | 0.603732201 | 0.777007923 |
| 9 | valence | 1000 | 0.627094056 | 0.546827204 | 0.740367008 | 0.630544416 | 0.586966344 | 0.768688083 |
| 9 | arousal | 1000 | 0.588660239 | 0.634530201 | 0.796749334 | 0.673112190 | 0.597765309 | 0.773286387 |
| 10 | valence | 1000 | 0.636378768 | 0.535781512 | 0.733365420 | 0.647214691 | 0.573169560 | 0.759859738 |
| 10 | arousal | 1000 | 0.590302422 | 0.624500084 | 0.790640624 | 0.691900577 | 0.575585635 | 0.758825244 |
| 11 | valence | 1000 | 0.635670382 | 0.532041806 | 0.730518346 | 0.658280302 | 0.555938272 | 0.750532203 |
| 11 | arousal | 1000 | 0.602263201 | 0.604044212 | 0.777533816 | 0.721443270 | 0.545197324 | 0.738425414 |
| 12 | valence | 1000 | 0.633006612 | 0.523872177 | 0.725022828 | 0.635144565 | 0.579780312 | 0.768855324 |
| 12 | arousal | 1000 | 0.631327600 | 0.565970174 | 0.753173069 | 0.744556180 | 0.511421706 | 0.715509311 |

### Baseline (identical across levels within each target)

| Target | Train mean | Validation MAE | Validation R² | Test MAE | Test R² | Pearson r |
|---|---|---|---|---|---|---|
| valence | 4.894692875 | 0.924825572 | -0.000046064 | 1.002563190 | -0.003113807 | undefined (JSON null) |
| arousal | 4.816953317 | 1.012681884 | -0.000018369 | 1.131995632 | -0.000320133 | undefined (JSON null) |

## Layer-12 Consistency and Evidence Provenance

> Layer 12 values were already obtained as the authoritative RQ1 held-out evaluation and are reused in Module D for the complete depth-wise analysis.

The Test sweep performs 24 new evaluations at levels 0–11 and reuses two Layer-12 configurations. It loads C4 predictions by Sample ID, checks target values, layer name/index, frozen alpha and Train-mean baseline, and copies the authoritative metrics. The final verifier requires exact equality of Layer-12 metrics, baselines and prediction arrays with C4. It does not run another Layer-12 Test model. C4 inputs match the unchanged canonical cache, labels and split hashes.

The original Module C Validation and Test artifact verifiers were also run unchanged and passed. Their outputs and exact C3/default-assembly compatibility checks are saved in `module_d_stage1_module_c_compatibility.json`. No inconsistency was silently resolved by choosing one result over another.

## Final Verification and Figure Review

All 6,786 Test rows (26 × 261) have the expected unique keys and the same 261 held-out IDs. True labels, frozen alpha, Train-only fitting records, Train means, MAE, R² and correlations were checked independently. The flat table's complete Validation/Test/baseline metrics agree with authoritative JSON values. No post-Test tuning occurred.

The primary PNG and SVG were created in `outputs/figures/`. File existence, image decoding and hashes were verified; the PNG was visually inspected for labels, both complete curves, zero reference, readable legend and Layer-12 provenance note. No plot-policy adjustment was made after Test. The line segments connect observed levels and do not imply continuous interpolation experiments.

## Descriptive Findings and Limits

Valence is already decodable at Pre-Transformer (R² 0.526720), rises to 0.568835 at Layer 2, and then fluctuates across a relatively narrow range. Layers 5–9 form a broadly high region (0.570338–0.586966), with a local dip at Layer 8; Layer 11 is lower and Layer 12 rebounds. There is no monotonic improvement with depth.

Arousal starts at 0.536629, rises to about 0.606 by Layer 3, and stays in a broad high region at Layers 3–9 (0.597765–0.611147). It then declines at every step from Layer 9 to 12: 0.597765 → 0.575586 → 0.545197 → 0.511422. The Layer-9-to-12 decline is about 0.08634 R². The mid-depth plateau and later decline are more pronounced than in Valence. The Pre-Transformer result means readable information exists before the Transformer stack; it does not isolate raw acoustic inputs because this representation already includes learned frontend processing.

Secondary point-estimate maxima: Valence Layer 9 (0.586966) and Arousal Layer 4 (0.611147). Valence Layer 6 is only about 0.001354 lower; Arousal Layer 7 is only about 0.000142 lower. These tiny gaps do not establish statistical superiority, a unique optimal layer or a universally best representation. Different targets' R² values normalize by their own target variation; relative numeric values do not establish an intrinsic ordering of emotion difficulty.

All 26 probes have positive R² and lower MAE than their respective Train-mean baselines. The evidence supports depth-dependent linear decodability under this fixed protocol. It does not show human-like understanding, causal emotion coding, absence of acoustic confounds, linear access to all emotion information, cross-dataset generalization, or superiority to conventional acoustic features. No inferential uncertainty analysis was added.

## Artifact Inventory

- Modified reusable code: `src/mert_emotion_probing/probing.py`.
- New execution/verification scripts: `scripts/run_layerwise_probing.py`, `scripts/verify_layerwise_probing_results.py`.
- Selection: `outputs/results/module_d_stage1_validation.json`.
- Validation predictions: `outputs/results/module_d_stage1_validation_predictions.csv`.
- Verified freeze gate: `outputs/results/module_d_stage1_pre_test_verification.json`.
- Test start marker: `outputs/results/module_d_stage1_test_started.json`.
- Held-out results: `outputs/results/module_d_stage1_test.json`.
- Test predictions: `outputs/results/module_d_stage1_test_predictions.csv`.
- Complete flat table: `outputs/results/module_d_stage1_layerwise_results.csv`.
- Final verification: `outputs/results/module_d_stage1_final_verification.json`.
- Module C compatibility: `outputs/results/module_d_stage1_module_c_compatibility.json`.
- Primary figure: `outputs/figures/module_d_stage1_test_r2_trajectory.png` and `.svg`.
- This report: `docs/codex_reports/module_d_stage1_layerwise_emotion_decodability_analysis.md`.
- Research log: `docs/research_logs/module_d_layerwise_emotion_decodability_analysis.md`.
- Private note: `personal_notes/module_d_what_i_should_understand.md`.
- Private context handoff: `handoffs/module_d_research_review_handoff.md`.
- README status updated to reflect finalized Module D and completed RQ2 under the frozen protocol.

## Commands and Environment

Run from the project root using `C:/Users/Rinshinozaki/miniconda3/envs/mert-emotion/python.exe` (abbreviated below as `python`). The existing environment was reused without installation: Python 3.10.21, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, scikit-learn 1.7.2, torch 2.14.0+cu130, Matplotlib 3.10.9.

```powershell
python -m py_compile src/mert_emotion_probing/probing.py scripts/run_layerwise_probing.py scripts/verify_layerwise_probing_results.py
python scripts/run_layerwise_probing.py select
python scripts/verify_layerwise_probing_results.py selection
python scripts/run_layerwise_probing.py test
python scripts/verify_layerwise_probing_results.py final
python scripts/verify_basic_probing_results.py
python scripts/verify_held_out_test_results.py
git diff --check
git status --short
```

The Module C verifiers were invoked through a small inline Python subprocess wrapper to save their JSON summaries alongside the additional exact-compatibility checks. No extraction command or split-generation command was executed. The select/Test commands intentionally refuse destructive reruns. To inspect existing results, use saved artifacts and the final verifier; do not delete the gate to rerun Test. Identities record absolute local paths as well as hashes, so relocating the workspace requires explicit provenance-aware handling rather than expecting the original gate to work unchanged.

## Issues, Fixes and Git Status

There were no failed model-selection runs, Test-sweep failures, invalidating implementation errors or protocol corrections. Ordinary repository recovery found the nested Git root and existing unrelated edits. The representation-level parameter is the only shared-code refactor; no fitting helper was changed. The stale README is an informational status issue, not a conflicting frozen research decision.

The initial implementation handoff did not commit or push. After confirming completion of research interpretation review and the learning checkpoint, the researcher authorized Module D finalization and GitHub synchronization. Finalization corrects the conclusion wording and completion status only; the accepted numerical artifacts, source and plotting policy remain unchanged, and no research experiment is rerun. The intended commit includes Module D source, scripts, small result/prediction and verification artifacts, PNG/SVG figures, English documentation and README. The existing researcher changes to `.gitignore` and `docs/project_workflow.md` solely establish the private `handoffs/` convention and are included unchanged. Raw data, canonical cache, `personal_notes/`, and `handoffs/` remain ignored/local. Unrelated `docs/.obsidian/` files remain untracked and are not included.


Finalization checks confirmed unchanged frozen input/source/result/figure hashes, all saved verification statuses, all 26 table configurations and report numbers, Layer-12 equality with C4, documentation links, figure existence, and ignored/untracked private artifacts. No research runner was executed. The staged `git diff --check` reports only Matplotlib-generated trailing spaces inside the authoritative SVG path data; these are retained to preserve the verified artifact bytes and hash. SVG XML parsing passes, and `git diff --cached --check -- . ':(exclude)outputs/figures/module_d_stage1_test_r2_trajectory.svg'` passes for all other intended files. GitHub connectivity required a command-only safe-directory exception under the network-enabled user; no global Git configuration was changed. Remote `main` matched the local parent commit before synchronization.

## What I Should Now Be Able to Explain

- RQ1 tested a pre-specified Layer 12; RQ2 holds the probe framework fixed while describing all 13 levels.
- A separate Validation-selected alpha allows each level/target the same tuning opportunity without introducing Test information.
- Train estimates parameters, Validation selects alpha, and Test evaluates only frozen choices; final fitting still excludes Validation.
- The pre-Test gate freezes all configurations and reporting policy, and the poisoning test checks actual computational isolation.
- Layer 12 contributes an existing RQ1 observation, not a new independent held-out discovery.
- Arousal shows a broad mid-depth high region and later decline; Valence is comparatively stable after early gains.
- Numerical maxima and small local differences do not establish a statistically or universally best layer.
- Positive decoding does not establish causal or human-like emotion understanding, confound freedom, or generalization beyond this study.

## Outcome

Module D implementation, verification, ChatGPT research interpretation review, and the researcher learning checkpoint are complete, with all frozen Module A/B/C decisions preserved. The research protocol and results are accepted for project use. **Module D is finalized and RQ2 is complete under the frozen protocol.** No subsequent research Module was started.
