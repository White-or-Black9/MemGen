# EventQA BM25 Top-2 Full Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans`, `superpowers:test-driven-development`, and the
> `experiment` skill. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute a traceable five-context EventQA BM25 top-2 retrieved-text
baseline with 100 questions per context and method-separable cost artifacts.

**Architecture:** Extend the validated smoke runner with an explicit
`smoke|full` scope contract while preserving the same BM25, prompt, model,
parser, scorer, and provenance logic. Run contexts 0-4 as five independent
processes serialized on one GPU, then aggregate only after all five artifacts
pass strict validation.

**Tech Stack:** Python 3.10, standard-library BM25, existing EventQA bridge and
scorer, PyTorch CUDA metrics, `unittest`, Bash, tmux.

---

## Run Contract

- Tier: main/test baseline pass.
- Scope: EventQA-65536 contexts 0-4, q0-99 per context, 500 questions total.
- Retrieval: unchanged deterministic BM25 (`k1=1.5`, `b=0.75`), top-2 exact
  source chunks, stable tie break by source chunk index.
- Query/generation/scoring: unchanged non-strict official question and
  candidates, frozen MemGen checkpoint, generation length 40, unchanged local
  parser/scorer, no latent bank.
- Isolation: one context per standalone process; five processes serialized on
  one GPU; model loading excluded from recorded method cost.
- Required outputs: per-context manifest, artifact and JSONL; final aggregate
  with 500-question EM, recall, format failures, cost, capacity and provenance
  checks; exact commands and GPU identity.
- Stop conditions: any missing/duplicate question, provenance/hash failure,
  non-finite cost, prompt overflow/truncation, config/scorer drift, process
  failure, or incomplete context coverage.
- Statistical boundary: one deterministic full pass supports a baseline point
  estimate, not variance or significance claims.

### Task 1: Generalize the runner contract with TDD

**Files:**
- Modify: `tests/test_eventqa_bm25_retrieved_text.py`
- Modify: `scripts/eval/eventqa_bm25_retrieved_text.py`

- [x] Add failing tests that require `expected_question_indices("smoke", 0,
  10)` to return q0-9, require `expected_question_indices("full", ctx, 100)`
  to return q0-99 for ctx0-4, and reject all other scope combinations.
- [x] Add failing artifact-validation tests for a valid 100-record full
  context, missing q99, wrong context identity, and prompt overflow.
- [x] Run
  `/home/baishilong/miniconda3/envs/memgen/bin/python -m unittest tests.test_eventqa_bm25_retrieved_text`
  and confirm RED from the missing full-scope API.
- [x] Implement `--measurement-scope {smoke,full}`, scope-derived question
  indices, context-aware artifact validation, and a full output-root default
  without changing retrieval or prompt construction.
- [x] Re-run the focused tests and confirm GREEN.

### Task 2: Add strict offline aggregation

**Files:**
- Create: `tests/test_eventqa_bm25_aggregate.py`
- Create: `scripts/eval/eventqa_bm25_aggregate.py`

- [x] Add failing tests for exact context coverage `{0,1,2,3,4}`, 500 unique
  `(context_index, query_index)` identities, invariant BM25 config, finite
  metrics/costs, and aggregate EM/recall/format/cost calculations.
- [x] Run the new test module and confirm RED because the aggregator does not
  exist.
- [x] Implement a read-only aggregator that loads five explicit artifact paths,
  invokes the runner validator, rejects duplicate or missing contexts, and
  writes JSON plus Markdown summaries.
- [x] Run focused and related EventQA tests and `git diff --check`.

### Task 3: Launch the serialized full pass

**Files:**
- Create: `runtime_logs/eventqa_bm25_top2_full_20260706/run_full.sh`

- [x] Inspect GPU process occupancy and select one viable GPU.
- [x] Create a fail-fast Bash loop for contexts 0-4, each invoking the runner
  with `--measurement-scope full --context-index <ctx> --question-limit 100`
  and a shared full output root.
- [x] Validate the launch script with `bash -n`.
- [x] Launch one detached tmux session and perform one startup check covering
  session, process, GPU, log growth, and first output directory.
- [x] Stop active monitoring after reporting the startup evidence.

### Task 4: Validate, aggregate, and route after completion

**Files:**
- Modify: `research_notes/PROGRESS.md`
- Modify: `research_notes/EXPERIMENTS.md`
- Modify: `research_notes/plans/eventqa_bm25_top2_full_plan.md`

- [x] Confirm all five processes exited successfully and all five artifacts
  contain 100 valid records.
- [x] Generate and validate the 500-question JSON/Markdown aggregate.
- [x] Compare BM25 against the existing full Disabled and frozen-P7 artifacts
  under explicit effectiveness and cost boundaries.
- [x] Record whether BM25 is retained as the paper retrieved-text baseline and
  select the next experiment; do not automatically launch it.

## Full-Pass Result (2026-07-06)

- Aggregate:
  `outputs/mab/eventqa_bm25_top2_full_aggregate.json` and `.md`.
- Integrity: contexts `0-4`, `500/500` unique questions, all artifact, BM25,
  provenance, finite-cost, and capacity checks passed; maximum rendered prompt
  was `8797/32768` tokens.
- BM25 top-2: EM `0.030`, EventQA recall `0.226`, format failures `265/500`.
- Cost excluding model load and scoring: total `692.845 s`, amortized
  `1.386 s/question`, max incremental peak GPU allocation `3597.3 MiB`.
- Same standalone cost-pass references: Disabled EM/recall `0.008/0.178`,
  total `367.448 s`, peak `142.9 MiB`; P7 EM/recall `0.200/0.240`, total
  `387.999 s`, peak `171.9 MiB`.
- Decision: retain BM25 top-2 as the paper retrieved-text baseline. It improves
  over Disabled on EM and recall, but remains far below P7 on EM while costing
  `1.79x` P7 time and about `20.9x` P7 incremental peak allocation. This is a
  one-pass BM25 point estimate; P7's paper-facing effectiveness remains the
  five-repeat estimate (`0.197+-0.020` EM, `0.254+-0.028` recall).
- Next experiment: the 16-token matched-budget explicit-text baseline. It was
  not launched automatically.

## Non-goals

- No matched-budget explicit-text run, summary-memory baseline, P7 modification,
  repeated BM25 run, paper prose claim, dependency installation, or commit.
