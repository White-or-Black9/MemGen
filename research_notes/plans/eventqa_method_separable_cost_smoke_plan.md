# EventQA Method-Separable Cost Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans`, `superpowers:test-driven-development`, and the
> `experiment` skill. Steps use checkbox syntax for tracking.

**Goal:** Measure standalone Disabled and frozen-P7 EventQA costs on context 0,
questions 0-9 without mixing timings or CUDA peaks across methods.

**Architecture:** Add a standalone cost runner that reuses the frozen EventQA
payload, model, scorer, and P7 bank implementation but executes exactly one
method per process. It resets/synchronizes CUDA statistics after model loading,
measures construction and query phases separately, and emits one versioned
cost artifact with invariants and environment metadata.

**Tech Stack:** Python 3.10, PyTorch CUDA metrics, existing MemGen/EventQA
helpers, `unittest`, tmux.

---

## Run Contract

- Research question: can Disabled and P7 cost be measured independently under
  the same EventQA context/question/generation/scorer contract?
- Null hypothesis: the new artifacts remain method-inseparable or violate the
  frozen-bank/schema contract.
- Alternative hypothesis: each process emits valid standalone construction,
  query, end-to-end latency, and peak-memory fields with identical evaluation
  settings and valid P7 read-only invariants.
- Tier: auxiliary/dev smoke; not a paper cost result.
- Dataset: EventQA-65536, context 0, q0-9.
- Methods: standalone Disabled; standalone frozen P7 (`0.05/0.10/16/top-2/0.05`).
- Stop conditions: combined method execution, schema mismatch, nonzero P7 query
  writes, changed bank snapshot, prompt/scorer drift, missing CUDA synchronization
  or peak reset, non-finite metrics, incomplete ten-question output.

### Task 1: TDD the cost contract

**Files:**
- Create: `tests/test_eventqa_method_separable_cost.py`
- Create: `scripts/eval/eventqa_method_separable_cost.py`

- [x] Write failing tests for method-specific CLI validation, cost summary
  fields, finite/nonnegative metrics, and P7 invariant rejection.
- [x] Verify RED because the module does not exist.
- [x] Implement the versioned schema and pure validation/summary helpers.
- [x] Verify GREEN.

### Task 2: Implement standalone execution

- [x] Add tests for Disabled construction cost zero, P7 construction/query
  separation, exact q0-9 scope, and command metadata.
- [x] Verify RED for missing execution-contract behavior.
- [x] Implement one-method execution using existing EventQA helpers.
- [x] Run focused and related unit tests.

### Task 3: Launch bounded smoke

- [x] Inspect GPU occupancy and choose the least-loaded viable GPU.
- [x] Launch Disabled and P7 as separate tmux processes, serialized if needed.
- [x] Verify tmux session, worker PID, GPU, log path, and output-root creation.
- [x] Stop after one-shot startup verification; do not launch the full cost pass.

### Task 4: Validate and record

- [ ] Validate both ten-question cost artifacts and comparability metadata.
- [ ] Record results in `research_notes/EXPERIMENTS.md` and `PROGRESS.md`.
- [ ] Run tests and `git diff --check`.

## Non-goals

- No full five-context cost run, no method change, no effectiveness rerun, no
  baseline expansion, and no paper-facing efficiency claim.
- No commit.
