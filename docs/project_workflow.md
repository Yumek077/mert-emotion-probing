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
- The researcher need not manually write every line of code, but must understand and be able to explain every core experimental decision.
- Prefer small, validated steps to implementing an entire Module at once.
- Do not over-engineer beyond the current research need.

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

Explanations should begin in plain language, then introduce formulas, tensor shapes, and technical terminology when they improve understanding.

### Codex

Codex is responsible primarily for:

- environment and dependency work;
- data loading and preprocessing implementation;
- model inference and feature extraction;
- caching and experiment automation;
- metric calculation and verification scripts;
- plots, repository cleanup, and implementation documentation.

Codex must not silently change the sample definition, preprocessing, model choice, layer selection, pooling, split, target definition, probe design, evaluation protocol, or any other locked research decision. If implementation exposes a conflict with the locked protocol, Codex must report the conflict and stop that part of the work rather than silently redesigning the experiment.

## 4. Standard Module Workflow

Every research Module follows:

**Understand → Decide → Codex Implement → Verify → Interpret → Learning Checkpoint → Iterate / Freeze**

### Understand

Before implementation, explain the Stage's research purpose, inputs, outputs, important concepts, relevant formulas or tensor shapes, and likely failure modes.

### Decide

Explicitly distinguish:

- decisions locked by earlier Modules or Stages;
- new research decisions required by the current Stage; and
- engineering choices Codex may make without changing research meaning.

Do not reopen a locked decision without a concrete reason.

### Codex Implement

Implement only the current Stage. Do not implement an entire Module unless explicitly instructed.

### Verify

Confirm that the implementation corresponds to the intended experiment. Depending on the Stage, verification may include tensor shapes, sample counts and IDs, split integrity, numerical finiteness, model mode, gradient state, data leakage, label alignment, saved artifacts, and deterministic or reproducible behavior.

“Code runs without errors” is not sufficient verification.

### Interpret

Discuss what the result means, whether it satisfies the Stage objective, alternative explanations, warnings, limitations, and whether further verification is necessary.

### Learning Checkpoint

Every completed Stage must include a researcher-facing section titled **“What I should now be able to explain.”** This checkpoint is mandatory before progressing. It should summarize in plain language:

- the Stage's core research logic;
- important concepts and decisions;
- useful formulas or tensor shapes;
- what was empirically verified;
- what the result means; and
- what the Stage does not establish.

This checkpoint normally occurs in the ChatGPT discussion and does not require a separate GitHub file for every Stage.

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

Stage reports are implementation records, not final research narratives.

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

Use this default structure:

### 1. Plain-Language Full Module Review

Provide a continuous, readable explanation rather than a fragmented checklist. Explain what problem the Module solved, why it was needed, what was done, how the experiment progressed, and what was learned.

### 2. What I Should Now Be Able to Answer

Convert important Stage learning checkpoints into focused Q&A form. Remove duplication and emphasize genuine understanding.

### 3. Core Formulas / Tensor Shapes

Include only formulas and tensor shapes useful for understanding and future review. Explain every dimension and symbol.

### 4. If My Supervisor / RA Interviewer Asked Me

Provide concise, natural explanations in both Chinese and English. Focus on research logic rather than memorized formal definitions.

### 5. What This Module Establishes and Does Not Establish

Separate supported conclusions from over-claims.

### 6. Self-Test

Include questions the researcher should attempt without looking at the answers.

## 10. Artifact and Directory Policy

### `docs/codex_reports/`

- Stage-level Codex implementation reports
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

At minimum, inspect:

1. `docs/project_workflow.md`
2. the latest relevant Module research protocol or research log
3. the previous Module's final research log when relevant
4. current Stage reports when continuing an unfinished Module
5. relevant source and configuration files only as needed

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
- the researcher completed the learning checkpoint;
- a GitHub-facing English research/implementation log exists;
- a Chinese personal note exists and remains private;
- repository artifacts are organized correctly;
- the next Module receives clearly defined inputs; and
- Git status and intended commit contents have been reviewed.

## 16. Project-Specific Principle

This project studies whether musical emotion information is **decodable** from pretrained music representations. Probing performance is therefore evidence about decodability; it must not automatically be interpreted as human-like musical emotion understanding. Implementation convenience must not override controlled comparison or research interpretability.
