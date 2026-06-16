# Prompt: Execute One Approved Phase

```text
Execute only the currently approved Phase.

Before editing:
1. Read research_notes/PROGRESS.md, PLANS.md, DECISIONS.md, TODO.md, BUGS.md,
   CODE_MAP.md, BASELINE.md, METHOD.md, and relevant experiment entries.
2. Inspect git status and preserve unrelated user changes.
3. Restate the approved scope and exit criteria.

During execution:
- Keep changes limited to the approved Phase.
- Do not modify Weaver training or Trigger training.
- Preserve exact original behavior when latent_memory_bank.enabled=false.
- In the current approved memory-bank scope, keep memory session-local with no
  cross-sample sharing unless a later phase explicitly changes this.
- Default batch_size to 1 unless the approved phase explicitly changes it.
- Add focused tests proportional to risk.
- Record commands, revisions, configurations, seeds, outputs, and failures.

Before stopping:
- Run the planned verification.
- Review the final diff for scope violations.
- Update research_notes/PROGRESS.md.
- Update research_notes/EXPERIMENTS.md for every experiment.
- Update research_notes/DECISIONS.md for important choices.
- Reconcile TODO.md and BUGS.md.
- Update CODE_MAP.md, BASELINE.md, METHOD.md, ABLATIONS.md, or PAPER_NOTES.md
  when the Phase changes their verified content.

Report changed files, verification results, unresolved risks, and the Phase outcome.
Then stop and wait for explicit user confirmation. Do not begin another Phase.
```
