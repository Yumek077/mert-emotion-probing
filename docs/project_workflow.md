# Project Working Protocol

Project: **Probing Musical Emotion in Pretrained Music Representations**

## 1. Purpose of This Workflow

This document is the project's working protocol: the operating procedure used by the researcher, ChatGPT, and Codex to design, implement, verify, interpret, document, and freeze research work. It is not an experiment protocol and does not replace Module-specific research protocols or research logs.

- **Research protocol** means what an experiment does.
- **Project workflow** means how the researcher, ChatGPT, and Codex work together to carry it out responsibly.

The workflow preserves research continuity across new ChatGPT and Codex sessions, keeps the researcher responsible for understanding and substantive decisions, uses Codex primarily for engineering implementation, prevents silent research-design changes, supports reproducibility, and ensures that implementation progress is accompanied by genuine researcher understanding.

## 2. Core Research Philosophy

- Understanding and research design come before implementation.
- The goal is not merely to obtain working code or high scores.
- A successful experiment must be reproducible, interpretable, and consistent with its locked research protocol.
- Negative and unexpected results are valid research outcomes.
- Codex must not make substantive research decisions silently.
- The researcher need not manually write every line of code. Research understanding and explanation should follow the learning tiers below: core concepts independently, supporting decisions with prompting, and technical detail by lookup.
- Prefer small, validated steps to implementing an entire Module at once.
- Do not over-engineer beyond the current research need.

### Minimal Sufficient Rigor

Use minimal sufficient rigor. Research work should be rigorous enough to make the experiment credible, reproducible, interpretable, and resistant to obvious leakage or implementation errors. Do not add experiments, validation layers, statistical procedures, or process stages merely to make the project appear more rigorous.

Introduce new complexity only when it is required by the current research question, needed to resolve an observed problem, or necessary to prevent a meaningful validity or reproducibility issue. Prefer:

**Observed problem → understand the cause → apply the smallest defensible solution → verify → document**

Avoid proactively expanding the experiment beyond the current research objective.

### Researcher Learning Profile and Explanation Rule

**Long-term default established in Final Synthesis Stage 4A:** the researcher is an AI undergraduate encountering AI Music and representation probing for the first time. Do not assume MIR / AI Music research fluency or familiarity with the representation-probing literature. The researcher may recognize terminology and recall the project route without yet having an independent intuition for every concept.

Preserve both scientific rigor and learning accessibility. Necessary ML concepts, including regression, Transformers, representations, data partitions, regularization and evaluation metrics, can be learned. Start from their purpose in this project. Do not require research-paper wording or a mature researcher's abstraction level as proof of understanding. Simplify language and explanation order, not scientific meaning, evidence or limitations.

For every future learning-facing explanation, use this progression:

| Level | Explanation order | Project requirement |
|---|---|---|
| **A — Intuition first** | What is this doing? | Use language the undergraduate can naturally understand and retell. |
| **B — Project meaning** | What does it correspond to here? | Connect to the relevant DEAM audio, Valence / Arousal targets, MERT vectors, Ridge predictions or RQ1–RQ4. A generic textbook definition is insufficient. |
| **C — Technical name** | What is the professional term? | Introduce the necessary English term after the intuition and project example, and explain it immediately. |
| **D — Necessary research nuance** | Which qualification matters for this question? | Add nuance when it affects experimental reasoning, interpretation, an evidence boundary or RA / supervisor discussion. Do not turn every caveat into an unprompted mastery requirement. |
| **E — Optional detail lookup** | Where can the exact detail be recovered? | Keep engineering mechanisms and exact research settings in reference material unless their purpose directly affects a core scientific decision. |

For example, first explain that the linear predictor weights the input numbers, and that Ridge discourages excessively large weights when many features are related. Connect those weights to MERT vectors and predicted emotion scores. Then name Ridge regression, L2 regularization and alpha. Do not begin with a sentence about coefficient magnitude in correlated high-dimensional representations and expect the terminology to supply the intuition.

Apply different language targets to different audiences:

- **Learning-facing material:** first-time AI Music undergraduate; intuition, project examples, then terms and necessary qualifications.
- **Public research writing:** strong undergraduate research writing. Keep academic structure and scientific precision, use direct sentences, and avoid unnecessary abstraction or imitation of senior research-paper prose.
- **Scientific and technical records:** retain decisions, provenance, history and necessary detail. These are reference sources, not memorization material; accessibility does not justify removing evidence.

The three learning tiers in Section 4 govern future checkpoints and verification. They supersede earlier broader mastery assumptions in learning plans or historical checkpoints without changing accepted scientific decisions or results. Report artifact implementation, document review and researcher mastery verification as separate states; none alone establishes the others.

## 3. Roles and Responsibilities

### Researcher

The researcher is responsible for:

- understanding the research question and core concepts;
- making substantive research decisions;
- approving protocol changes;
- interpreting results and their limitations;
- understanding what the evidence can and cannot support; and
- being able to explain the experiment independently.

### ChatGPT

ChatGPT is responsible for:

- explaining the research logic clearly before implementation;
- identifying decisions that must be made;
- distinguishing research decisions from engineering decisions;
- producing Stage-specific Codex prompts only after the relevant concepts and decisions are understood;
- reviewing Codex reports and experimental outputs;
- helping interpret results and limitations;
- producing the Stage learning checkpoint; and
- helping create final Module documentation and personal notes.

Explanations must follow the researcher profile and A–E sequence in Section 2. Introduce formulas or tensor shapes only when they help the current concept; otherwise retain them as supporting or lookup material.

### Codex

Codex is responsible primarily for:

- environment and dependency work;
- data loading and preprocessing implementation;
- model inference and feature extraction;
- caching and experiment automation;
- metric calculation and verification scripts;
- plots, repository cleanup, and implementation documentation.

Codex must not silently change the sample definition, preprocessing, model choice, layer selection, pooling, split, target definition, probe design, evaluation protocol, or any other locked research decision. If implementation exposes a conflict with the locked protocol, Codex must report the conflict and stop that part of the work rather than silently redesigning the experiment.

### Implementation Issues Versus Research-Level Decisions

Codex need not stop for every ordinary engineering detail. It may independently solve, verify, and document implementation issues such as file and path handling, schema parsing, deterministic serialization, ordinary assertions, reusable helpers, minor ID-column normalization when semantics are already frozen, output formatting, implementation-level numerical edge cases, and straightforward library or API usage. These solutions belong in the relevant Stage report.

Codex must stop for researcher discussion before changing anything that affects a frozen research decision, the research question or interpretation, dataset or sample population, target or representation definition, split policy, primary evaluation protocol, an already frozen model or probe family, result-selection logic, or experiment scope. A research-level decision must never be silently relabeled as an implementation detail.

## 4. Standard Module Workflow

Every research Module follows:

**Understand → Decide → Codex Implement → Verify → Interpret → What I Should Now Be Able to Explain → Iterate / Freeze**

### Understand

Before implementation, explain the Stage's research purpose, inputs, outputs and Tier 1 concepts through project-specific intuition. Use Tier 2 material with support when needed for a decision; keep exact formulas, shapes and implementation failure diagnostics as lookup unless directly necessary.

### Decide

Explicitly distinguish:

- decisions locked by earlier Modules or Stages;
- new research decisions required by the current Stage; and
- engineering choices Codex may make without changing research meaning.

Do not reopen a locked decision without a concrete reason.

### Codex Implement

Implement only the current Stage. Do not implement an entire Module unless explicitly instructed.

The default implementation unit should be one coherent objective, not a sequence of micro-stages. Prefer:

**Research objective → discuss necessary research decisions → one coherent Codex implementation Stage → detailed process report → researcher review → interpret and freeze**

Split implementation into multiple Stages only when a genuine research decision lies between them, later execution depends on interpreting an earlier empirical result, a protected evaluation boundary such as held-out Test requires a deliberate gate, or the work is too large to verify safely as one unit. Ordinary engineering checkpoints do not automatically require separate research Stages.

### Verify

Confirm that the implementation corresponds to the intended experiment. Depending on the Stage, verification may include tensor shapes, sample counts and IDs, split integrity, numerical finiteness, model mode, gradient state, data leakage, label alignment, saved artifacts, and deterministic or reproducible behavior.

“Code runs without errors” is not sufficient verification.

### Interpret

Discuss what the result means, whether it satisfies the Stage objective, alternative explanations, warnings, limitations, and whether further verification is necessary.

### Learning Checkpoint

Every completed Stage must include a researcher-facing section titled **“What I should now be able to explain.”** This checkpoint is mandatory before progressing. Apply the learning tiers below rather than expecting every detail in the report to be reproduced. It should summarize in plain language:

- the Stage's core research logic;
- important concepts and decisions;
- useful formulas or tensor shapes only when needed for the concept, with other detail marked as lookup;
- what was empirically verified;
- what the result means; and
- what the Stage does not establish.

Every research Stage must leave a repository-visible checkpoint, including conceptual Stages that produce no code or experiment. The existing Stage-level documentation convention remains `docs/codex_reports/`. Depending on the nature of the Stage, its report may document conceptual understanding, research decisions, implementation, validation, or experiment execution.

### Three-Tier Learning Priority

**Tier 1 — Must genuinely understand and explain.** These are the current core of future closed-book verification, using the researcher's own words:

1. What the project studies.
2. What DEAM Valence and Arousal mean.
3. What a MERT representation roughly is.
4. Why MERT is frozen and a linear probe is trained.
5. The basic audio → MERT → pooling → Ridge → prediction → evaluation flow.
6. What Ridge roughly does.
7. The distinct roles of Train, Validation and Test.
8. What MAE, R² and Pearson r roughly assess.
9. Why RQ1 → RQ2 → RQ3 → RQ4 arise in sequence.
10. The main findings of the project.
11. Why these findings do not directly establish that MERT truly understands musical emotion.

**Tier 2 — Understand when encountered; explain with prompting.** Supporting material includes StandardScaler, alpha, Sample-ID alignment, the Train-mean reference, the rough 51-D acoustic recipe, residuals, Energy / RMS, Tempo / BPM, the pre-specified Layer-12 decision, the conceptual reason for single-item extraction, basic Test exposure, and why MERT outperforming selected acoustics does not identify an advantage beyond all acoustics. Explain these through examples and questions when relevant. They need not all appear in the first project introduction or an unprompted answer.

**Tier 3 — Lookup only.** This includes GroupNorm internals, exact padding propagation, the alpha grid and tie-break rule, all 51 acoustic dimensions, exact split counts unless needed, exact MAE / r decimals, all RQ4 coefficients, cache shape, hashes, serialization, CLI / helper names, verification machinery and software/debug history. Do not use recall of these details as proof of mastery. If a mechanism affected a scientific decision, teach the short reason at Tier 2 and retain the detailed mechanism here.

Learning verification tests conceptual understanding at this level, not the ability to reproduce Codex/ChatGPT academic wording. Accept accurate everyday explanations and use prompts for Tier 2. Essential boundaries must remain correct, but the researcher need not independently recite every methodological caveat. A learning checkpoint is not an additional experiment or a reason to reopen frozen evidence.

### Iterate / Freeze

Proceed only after the Stage is understood and verified. If successful, treat it as frozen and avoid silent later changes. If unsuccessful, preserve and document the failure, understand its cause, and decide explicitly whether the Stage should be revised.

## 5. Stage Scope Rule

Codex prompts should be deliberately narrow. A Stage prompt should state:

- the current Stage objective;
- the locked protocol;
- what Codex may modify;
- what Codex must verify;
- the required report; and
- explicit exclusions.

Codex must stop after the requested Stage and must not automatically continue into the next Stage.

## 6. Research Decision Locking

Once a research decision has been verified and frozen in an earlier Stage or Module, later implementation must treat it as locked. Examples include sample definition, clip strategy, sample rate, model, frozen or fine-tuned status, representation definition, pooling, train/validation/test split, target labels, and evaluation metrics.

A locked decision may change only when:

1. a concrete implementation or scientific problem is identified;
2. the issue is explained;
3. the researcher explicitly approves the change; and
4. the relevant documentation is updated.

Codex must never silently repair a protocol inconsistency by changing the protocol.

## 7. Codex Stage Reports

Every substantial Codex Stage should produce an English, GitHub-facing report under `docs/codex_reports/`.

Use this naming convention:

```text
module_<letter>_stage<number>_<short_description>.md
```

For example, use `module_b_stage1_mert_representation_inspection.md`; do not use an abbreviated Stage-only prefix in a formal filename. Informal discussion may still use B1, B2, C1, and similar abbreviations.

A typical report contains:

1. What was done
2. Files created or modified
3. Data or samples used
4. Implementation details relevant to the research protocol
5. Verification performed
6. Observed results or tensor shapes
7. Protocol consistency
8. Warnings or unexpected findings
9. Exact commands used
10. Conclusion and whether the Stage passed

For larger implementation Stages, the report is the repository's detailed technical memory. It should preserve what was implemented and inspected, verification performed, problems and root causes, solutions, artifacts, important structural or numerical checks, unresolved issues, and whether any frozen decision was affected. The researcher is not expected to memorize all of this. Stage reports are not final Module research narratives.

## 8. Module-Level Research / Implementation Logs

After a Module is complete and all of its Stages have been reviewed, create a polished English Module-level log under `docs/research_logs/`.

This differs from individual Codex Stage reports. It should concisely explain the Module objective, decisions, implementation, verification, key results, interpretation, limitations, and what is passed to the next Module. It serves as the GitHub-facing research record for the completed Module.

## 9. Personal Notes

At the end of each completed Module, create a private researcher learning note under `personal_notes/` using:

```text
module_<letter>_what_i_should_understand.md
```

Examples include `module_a_what_i_should_understand.md` and `module_b_what_i_should_understand.md`.

These notes are Chinese-first, may retain useful English technical terms, and support long-term review, MSc or RA interview preparation, and rebuilding understanding after the project. They are not GitHub-facing and must remain excluded from Git tracking.

Personal notes follow the profile, explanation sequence and tiers above. Separate a short primary student learning layer from deeper reference when needed; do not force every Module to have an equally detailed primary note. Teach the intuition before naming the term, and mark supporting / lookup material explicitly. The default structure below is a review aid, not a requirement to master every subsection independently.

**Current learning state (2026-10-03):** Final Synthesis Stages 1–4 are complete. Stage 4C Researcher Learning Verification is completed / PASS, as confirmed by the researcher. The former paused mastery state and broader closed-book verification standard are superseded. `personal_notes/project_what_i_should_understand.md` remains deeper reference; the Primary Learning Guide and Module notes remain private. Learning verification is distinct from scientific evidence. Preserve the learning profile, A–E explanation rule and three tiers above; do not restart completed verification from an older handoff.

**Current project lifecycle (2026-10-03):** Stage 5 Final Project Consolidation / Closure is completed following final read-only verification. **Research complete → Learning verified → Repository finalized → Maintenance-only.** The [final closure record](codex_reports/final_synthesis_stage5_final_project_closure.md) records the closure and final checkpoint; maintenance-only takes effect after that checkpoint.

**Maintenance-only rule after the final checkpoint:**

- Ordinary documentation, link and reproducibility defects may be repaired.
- Frozen scientific claims, results and protocols must not be changed without a separately reviewed research decision.
- New experiments, new RQs, RA preparation and research extensions require a new explicit task; they are not continuation requirements of the completed mini-project. MSc application-facing materials are outside the remaining scope.

Personal notes are review tools, not duplicate technical histories. Keep them detailed enough to learn from but substantially easier to revisit than the Stage reports. Use this default structure:

### 1. Research Mainline

Tell the complete Module story concisely: why it exists, its input, high-level process, output, and connection to the research question or next Module.

### 2. Key Decisions — Why This, Not the Alternatives?

Include only choices that materially affect methodology or interpretation. For each, explain what was chosen, why, what reasonable alternative existed, and why it was not selected.

### 3. Essential Concepts I Need to Understand

Include only concepts necessary to explain the research. Where useful, introduce important terminology as `English term（简洁中文解释）` rather than building an exhaustive glossary.

### 4. Engineering Details — Understand the Purpose, Not Memorize the Implementation

Summarize what each important engineering mechanism does, why it exists, and what failure it prevents. Do not expect the researcher to memorize commands, helper names, serialization details, assertions, or low-level library syntax.

### 5. What I Should Be Able to Explain

Use a concise set of high-value questions or checkpoints focused on the research mainline, methodological choices, and interpretation boundaries rather than implementation trivia.

### 6. One-Minute Recap

End with a compact summary that can restore the entire Module quickly after a long gap.

## 10. Artifact and Directory Policy

Before creating documentation, artifacts, or directories, inspect the repository's actual organization and preserve established conventions where possible. Do not introduce a new hierarchy merely because it appears theoretically cleaner. Restructure only when the existing organization genuinely cannot support the requirement.

### `docs/codex_reports/`

- Stage-level reports covering conceptual understanding, research decisions, implementation, validation, or experiment execution as appropriate
- English
- GitHub-facing

### `docs/research_logs/`

- Polished Module-level research and implementation logs
- English
- GitHub-facing

### `personal_notes/`

- Researcher learning notes
- Chinese-first
- Private and not uploaded to GitHub

### Experiment outputs and cached representations

Use appropriate project data and output directories. Do not commit large generated embeddings, raw datasets, model caches, or other unsuitable artifacts unless this is explicitly intended. Respect the repository `.gitignore`.

## 11. Git / GitHub Workflow

Use Git as a research history rather than committing every trivial edit. A recommended Stage checkpoint is:

```text
Stage implementation
→ verification
→ ChatGPT review
→ interpretation
→ learning checkpoint
→ Stage freeze
→ commit when appropriate
```

At Module completion:

```text
final verification
→ Module research/implementation log
→ personal note
→ repository cleanup
→ Git status and diff review
→ commit and push
```

Do not commit private personal notes. Do not push large datasets, cached models, or large embeddings unless explicitly intended.

## 12. New Chat / Session Recovery Protocol

When starting a new ChatGPT or Codex conversation for a later Stage or Module, do not rely only on conversational memory. Recover project context from repository artifacts first.

At a Module transition, Codex should generate a repository-grounded Markdown handoff under `handoffs/` for the researcher to paste into the new ChatGPT research conversation. A handoff is a local context-recovery aid, not a formal research artifact: `handoffs/` must remain excluded from Git, and handoff files must not be committed.

Future handoffs must recover the researcher learning profile, A–E explanation sequence, current learning tiers and verification state as well as the scientific state. Read this workflow before using an older handoff's learning expectations. Preserve historical handoffs; their former next-step instructions do not override current user instructions or the Stage 4A alignment rule.

At minimum, inspect:

1. `docs/project_workflow.md`
2. the latest relevant Module research protocol or research log
3. the previous Module's final research log when relevant
4. current Stage reports when continuing an unfinished Module
5. relevant source and configuration files only as needed

> **Historical-status note:** The post-Module-D recovery guidance below records the planning state before Modules E/F. Both are now complete; use the [accepted Module F record](research_logs/module_f_acoustic_correlate_and_error_analysis.md) and [current project status](../README.md#current-project-status) for Final Synthesis recovery. The original guidance is preserved as history.

For post-Module-D work, also read the [original README research intentions](https://github.com/Yumek077/mert-emotion-probing/blob/b8955d5/README.md), completed Module A–D logs in `docs/research_logs/`, and the authoritative [Post-Module-D Research Roadmap](research_roadmap_after_module_d.md). Preserve the original intentions and completed evidence; the roadmap refines the remaining RQ3/RQ4 route. The next objective is **Module E design / RQ3**, not experiment execution; its acoustic feature recipe remains to be decided.

Before new implementation, restore:

- the overall research question;
- current project status and completed Modules or Stages;
- locked research decisions;
- the current Module objective;
- inputs inherited from the previous Module;
- artifact and naming rules; and
- unresolved questions.

A researcher may begin a new ChatGPT conversation with:

> Continue the AI Music RA mini-project. First recover the research context and working protocol from `docs/project_workflow.md` and the latest relevant research logs. Do not implement anything until the current Module status, locked decisions, and next Stage are clear.

The repository is the durable source of project continuity; chat memory is supplementary.

## 13. Failure and Unexpected-Result Policy

Do not hide failed experiments or unexpected findings. If a Stage fails:

- preserve the relevant evidence;
- identify whether the cause is implementation, data, protocol, environment, or hypothesis;
- document the issue; and
- do not silently modify the research question to make the result appear successful.

A scientifically useful negative result is preferable to an undocumented protocol change.

## 14. Reproducibility Principles

Where relevant, record:

- model and checkpoint identity;
- important dependency versions;
- random seeds;
- dataset and sample IDs;
- split definitions;
- preprocessing;
- exact commands;
- output locations; and
- important configuration.

Avoid depending on undocumented local state.

## 15. Module Completion Checklist

A Module is not complete merely because its code ran. Before declaring completion, confirm that:

- the research objective was addressed;
- implementation matches the protocol;
- verification passed;
- important warnings were interpreted;
- the researcher completed the appropriately tiered learning checkpoint, with artifact creation, review and actual learning verification reported separately;
- a GitHub-facing English research/implementation log exists;
- a Chinese personal note exists and remains private;
- repository artifacts are organized correctly;
- the next Module receives clearly defined inputs; and
- Git status and intended commit contents have been reviewed.

## 16. Project-Specific Principle

This project studies whether musical emotion information is **decodable** from pretrained music representations. Probing performance is therefore evidence about decodability; it must not automatically be interpreted as human-like musical emotion understanding. Implementation convenience must not override controlled comparison or research interpretability.
