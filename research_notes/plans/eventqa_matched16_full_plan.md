# EventQA Strict Matched16 Full Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans`, `superpowers:test-driven-development`, and the
> `experiment` skill. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and aggregate a strict five-context EventQA matched16 baseline in
which both selected source tokens and actual rendered prompt delta equal 16.

**Architecture:** Generalize the validated smoke runner to explicit
`smoke|full` scope without changing retrieval, window selection, prompt,
model, parser, or scorer. Execute contexts 0-4 as independent processes
serialized on one GPU, then aggregate only five strictly validated artifacts.

**Tech Stack:** Python 3.10, BM25, Qwen tokenizer, EventQA bridge/scorer,
PyTorch CUDA metrics, `unittest`, Bash, tmux.

---

## Run Contract

- Scope: EventQA-65536 contexts 0-4, q0-99 each, 500 total.
- Every record: two contiguous 8-token source windows and rendered prompt delta
  exactly 16; no visible framing or source IDs in the model prompt.
- Method and scoring are identical to the corrected canonical smoke.
- One context per standalone process, serialized on one GPU; model loading and
  scoring excluded from method cost.
- Stop on incomplete identities, any token/delta mismatch, provenance/hash
  failure, capacity overflow, config drift, non-finite cost, or process error.
- One deterministic full pass supports point estimates only.

### Task 1: Generalize runner with TDD

- [x] Add failing tests for smoke/full scope selection and 100-record full
  artifact validation across contexts 0-4.
- [x] Implement context-aware scope, output naming, records and validation.
- [x] Confirm focused tests GREEN.

### Task 2: Add strict full aggregator with TDD

- [x] Add failing tests for exact five-context coverage, 500 unique identities,
  exact token/delta invariants, effectiveness, format, cost and capacity.
- [x] Implement JSON/Markdown aggregator over five explicit artifact paths.
- [x] Run related EventQA tests, compile/CLI checks and `git diff --check`.

### Task 3: Launch serialized full queue

- [x] Inspect GPU occupancy and choose one viable GPU.
- [x] Create and syntax-check a fail-fast ctx0-4 launch script.
- [x] Launch detached tmux and verify session, PID, GPU, log and first output
  directory once.
- [x] Stop active monitoring after startup report.

### Task 4: Validate and route after completion

- [x] Add a deterministic joint window selector that falls back from the
  unconstrained top pair only when the final rendered prompt delta is not 16;
  record candidate ranks, score loss and search limit.
- [x] Add regression tests for boundary re-tokenization and constrained
  fallback, then run a no-inference 500-question budget preflight.
- [x] Verify ctx0-2 remain rank-1/rank-1 equivalent; resume only ctx3 and ctx4
  after preflight reaches 500/500 exact delta=16.

- [x] Validate five 100-record artifacts and generate the 500-record aggregate.
- [x] Compare against Disabled, full BM25 top-2 and P7 with repeat boundaries.
- [x] Update experiment/progress notes and select, but do not launch, the next
  paper experiment.

## Full-Pass Result (2026-07-06)

- Aggregate:
  `outputs/mab/eventqa_matched16_full_aggregate.json` and `.md`.
- Integrity: `500/500` unique questions; selected source tokens `{16}`;
  rendered prompt deltas `{16}`; all capacity/provenance/cost checks passed.
- Budget fallback: exactly three questions, all in ctx3: q67 ranks `[1,2]`
  with zero score loss, q80 `[2,1]` with `0.06669` loss, and q83 `[1,2]`
  with zero loss. Ctx0-2 and ctx4 use original rank `[1,1]` throughout.
- Result: EM `0.068`, recall `0.180`, format failures `347/500`.
- Cost excluding model load/scoring: total `501.761 s`, amortized
  `1.004 s/question`, max incremental peak `171.0 MiB`.
- References: Disabled `0.008/0.178`, `367.448 s`, `142.9 MiB`; full BM25
  top-2 `0.030/0.226`, `692.845 s`, `3597.3 MiB`; standalone P7
  `0.200/0.240`, `387.999 s`, `171.9 MiB` (EM/recall, total, peak). P7's
  paper-facing five-repeat result remains `0.197+-0.020` EM and
  `0.254+-0.028` recall with `121.4+-8.8` format failures.
- Interpretation: matched16 improves EM over Disabled and full BM25 but does
  not improve recall over Disabled, and remains far below P7 on EM, recall and
  output-format reliability at the same 16 rendered positions. It is slower
  than P7 under this implementation because deterministic BM25/window search
  costs `83.045 s` over 500 queries.
- Next paper experiment: text-summary memory baseline. It was not launched.

## Non-goals

- No text-summary baseline, P7 modification, no-query-retrieval ablation,
  repeated matched16 run, significance claim, dependency installation, or
  commit.
