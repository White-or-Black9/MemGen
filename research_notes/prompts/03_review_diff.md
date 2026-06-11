# Prompt: Review Current Diff

```text
Review the current MemGen diff as a skeptical senior code reviewer. Do not modify
files unless explicitly asked after the review.

Read the current Phase, constraints, and accepted decisions from research_notes/.
Inspect git status, diff, relevant tests, and only the code needed to understand
the changes.

Prioritize findings by severity:
1. Changes to Weaver or Trigger training behavior.
2. Any behavioral or numerical difference when latent_memory_bank.enabled=false.
3. Cross-sample/session memory leakage.
4. Incorrect batch assumptions or batch_size defaults.
5. State reset, device, dtype, shape, and lifecycle bugs.
6. Retrieval/update correctness and complexity.
7. Missing regression, isolation, reproducibility, or failure tests.
8. Unrecorded experiment or design decisions.

For each finding provide:
- Severity.
- File and line.
- Concrete failure mode.
- Why existing checks do not catch it.
- Minimal corrective action.

Then list open questions and residual test gaps. If no findings exist, say so
clearly. Do not advance to another Phase.
```
