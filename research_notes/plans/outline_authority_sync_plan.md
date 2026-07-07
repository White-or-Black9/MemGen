# Outline Authority Synchronization Plan

**Goal:** Make `paper/outline.md` the authoritative source for the paper title,
problem framing, contributions, research questions, and section structure while
preserving the verified EventQA/P7 evidence boundary.

**Approach:** Update current-state research notes and paper-facing planning and
draft files. Preserve historical experiment notes and measurements as records;
append a superseding decision rather than rewriting accepted historical
decisions.

## Tasks

- [x] Update current project and paper-scope control notes to the long-horizon
  LLM-agent framing.
- [x] Append a decision that supersedes DEC-0078 only for title and paper-goal
  scope while retaining EventQA as the current operational evidence.
- [x] Synchronize method, experiment, writing-roadmap, and completion-plan
  documents with the outline's contributions and RQ1-RQ4.
- [x] Synchronize all current paper-facing draft, skeleton, section, table,
  figure, gap, and TODO documents.
- [x] Preserve LoCoMo as optional diagnostic evidence, not a multi-turn claim or
  a required main-paper benchmark.
- [x] Verify that no current paper-facing file retains the superseded title or
  treats the EventQA-only operational scope as the full paper goal.
- [x] Run `git diff --check` and inspect the final diff and worktree status.

## Non-goals

- Do not alter experimental metrics, artifact provenance, P7 parameters, code,
  prompts, parsers, or scorers.
- Do not run inference or GPU jobs.
- Do not rewrite historical benchmark and experiment records merely to replace
  terminology.
- Do not commit.
