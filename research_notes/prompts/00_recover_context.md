# Prompt: Recover Project Context

Use this at the start of a new CLI or VSCode-assisted session.

```text
You are resuming long-term research work on the MemGen repository.

First read only the project memory files needed to recover state:
- research_notes/PROGRESS.md
- research_notes/PLANS.md
- research_notes/DECISIONS.md
- research_notes/TODO.md
- research_notes/BUGS.md
- research_notes/CODE_MAP.md
- research_notes/BASELINE.md
- research_notes/METHOD.md
- research_notes/EXPERIMENTS.md

Then inspect git status and the latest relevant diff/log without modifying files.

Summarize:
1. Current Phase and status.
2. Last completed work and evidence.
3. Active constraints and accepted decisions.
4. Dirty or uncommitted changes.
5. Open bugs, blockers, and uncertainties.
6. The single next action allowed by the current Phase gate.

Hard constraints:
- Do not modify Weaver training.
- Do not modify Trigger training.
- latent_memory_bank.enabled=false must preserve exact original behavior.
- In the current approved memory-bank scope, do not allow cross-sample memory
  sharing and default to batch_size=1 unless a later approved phase says
  otherwise.
- Execute only one Phase at a time.
- Do not begin a new Phase without explicit user confirmation.

Do not edit code or notes during context recovery. Stop after the summary.
```
