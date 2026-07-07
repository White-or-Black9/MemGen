# EventQA Full Method-Separable Cost Pass Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans`, `superpowers:test-driven-development`, and the
> `experiment` skill. Steps use checkbox syntax for tracking.

**Goal:** Produce comparable standalone Disabled and frozen-P7 cost artifacts
for all five EventQA contexts and all 100 questions per context.

**Architecture:** Extend the validated cost runner with an explicit `full`
scope while preserving the existing q0-9 smoke contract. Launch ten independent
processes (`5 contexts x 2 methods`) serially on one A6000, so CUDA peak state,
method state, and failure boundaries remain isolated. Aggregate only after all
ten artifacts pass validation.

**Tech Stack:** Python 3.10, PyTorch CUDA metrics, existing EventQA helpers,
`unittest`, tmux.

## Run Contract

- Tier: main/test cost measurement.
- Dataset: EventQA-65536, contexts 0-4, q0-99 per context.
- Methods: standalone Disabled and frozen P7 (`0.05/0.10/16/top-2/0.05`).
- Hardware: the same physical RTX A6000 for all ten serialized processes.
- Required metrics: construction/query/end-to-end latency, baseline/peak/
  incremental GPU allocation, output tokens, EM, recall, and P7 read-only
  invariants.
- Stop conditions: any process mixes methods, emits incomplete scope, changes
  prompt/scorer/config, records nonzero P7 query writes, changes the bank
  snapshot, or produces non-finite cost fields.

### Task 1: Extend scope contract with TDD

- [x] Add failing tests showing smoke remains context-0 q0-9 and full accepts
  contexts 0-4 q0-99 only.
- [x] Implement the minimal scope generalization.
- [x] Run focused and related tests.

### Task 2: Freeze and launch the full queue

- [x] Create a deterministic ten-job launch script with one method per process.
- [x] Inspect GPU occupancy and select one viable GPU.
- [x] Launch the serialized queue in detached tmux.
- [x] Verify session, PID, first output root, log growth, and GPU occupancy once.

### Task 3: Validate and aggregate after completion

- [x] Validate all ten artifacts and exact 100-row scopes.
- [x] Produce full cost JSON/Markdown aggregation with per-context and global
  summaries.
- [x] Update experiment/progress notes and paper cost table inputs.
- [x] Run tests and `git diff --check`.

## Non-goals

- No BM25, matched-budget, text-summary, no-query-retrieval, method change, or
  cost-efficiency claim before the full queue and aggregation validate.
- No commit.
