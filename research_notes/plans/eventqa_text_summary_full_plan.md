# EventQA Same-Model Text-Summary Full Pass Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans`, `superpowers:test-driven-development`, and the
> `experiment` skill. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construct one frozen <=128-token rolling summary per EventQA context
and evaluate all 500 questions under the accepted same-model baseline.

**Architecture:** Generalize the validated construction/query runners to
context-aware full scope with explicit run IDs. For each context 0-4 in one
serialized GPU queue, construct the summary in one standalone process, then
launch a second standalone process that loads that exact artifact and answers
q0-99. Aggregate only five complete provenance-linked pairs.

**Tech Stack:** Python 3.10, same frozen MemGen/Qwen model, EventQA
bridge/scorer, PyTorch CUDA metrics, `unittest`, Bash, tmux.

---

## Contract

- Same prompt, 128-token cap and Bank-off generation as the accepted smoke;
  poor summary quality is retained as baseline behavior by user decision.
- Contexts 0-4, 100 questions each; no question/gold enters construction.
- One construction and one query process per context, serialized on one GPU.
- Explicit run IDs and exact construction-artifact paths; no latest-file lookup.
- Required aggregate: 500 unique identities, summary provenance/hash/token
  counts, EM, recall, format, construction/query/end-to-end cost and peak GPU.
- Stop on process/schema/provenance/scope/capacity/scorer/cost failure.

### Task 1: Generalize construction/query runners with TDD

- [x] Add failing tests for context0-4 construction artifacts, full q0-99 query
  artifacts, explicit run IDs and context/provenance mismatch rejection.
- [x] Implement context-aware/full-scope runners without changing prompts.
- [x] Confirm focused and related tests, CLI/compile and diff checks.

### Task 2: Launch serialized construction/query queue

- [x] Inspect GPU occupancy and create an explicit ctx0-4 fail-fast script.
- [x] Launch detached tmux and verify first construction process/GPU/log/output.
- [x] Stop active monitoring after startup report.

### Task 3: Validate and aggregate after completion

- [x] Validate five construction/query pairs and 500 records.
- [x] Generate JSON/Markdown aggregate and compare all paper baselines.
- [x] Update notes and route to P7 no-query-retrieval without auto-launching.

## Non-goals

- No prompt repair, external summarizer, repeated summary pass, P7 change,
  significance claim, dependency installation, or commit.
