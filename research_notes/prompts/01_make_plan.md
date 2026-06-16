# Prompt: Make One-Phase Plan

```text
Prepare a detailed plan for exactly one Phase of the MemGen research project.
Do not implement anything yet.

Read:
- research_notes/PROGRESS.md
- research_notes/PLANS.md
- research_notes/DECISIONS.md
- research_notes/TODO.md
- research_notes/BUGS.md
- research_notes/CODE_MAP.md
- research_notes/BASELINE.md
- research_notes/METHOD.md

Inspect only the repository areas necessary to plan this Phase.

The plan must include:
- Phase goal and research question.
- In scope and explicitly out of scope.
- Verified files/symbols likely to change.
- Protected Weaver and Trigger training boundaries.
- Ordered implementation steps.
- Compatibility and regression checks.
- Experiment plan and artifact locations.
- Required updates to PROGRESS, EXPERIMENTS, DECISIONS, TODO, and BUGS.
- Exit criteria and rollback strategy.
- Risks, assumptions, and unresolved questions.

Hard constraints:
- Inference-only method changes.
- Exact original behavior when latent_memory_bank.enabled=false.
- No cross-sample memory sharing in the current approved memory-bank scope.
- Default batch_size to 1 unless a later approved phase explicitly changes it.
- Execute one Phase only, then pause.

End with a concise approval request. Do not edit files or start execution.
```
