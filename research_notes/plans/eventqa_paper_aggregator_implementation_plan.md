# EventQA Paper Aggregator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` and `superpowers:test-driven-development`.
> Steps use checkbox syntax for tracking.

**Goal:** Build a no-inference EventQA aggregator that validates frozen run
artifacts and emits schema-stable JSON and Markdown rows for Bank-off, P6, and
P7.

**Architecture:** A standalone standard-library Python CLI reads explicit
method specifications and completed run roots. It validates required files,
protocol/config identity, question/context scope, scorer fields, and
query-read-only invariants before aggregating repeat-level metrics. Missing or
incompatible fields fail loudly. Reserved cost and explicit-memory provenance
fields keep future baselines schema-compatible without fabricating values.

**Tech Stack:** Python 3.10 standard library, `unittest`, JSON/JSONL, Markdown.

---

### Task 1: Freeze schema and validation behavior

**Files:**
- Create: `tests/test_eventqa_paper_aggregator.py`
- Create: `scripts/eval/eventqa_paper_aggregator.py`

- [x] Write tests for required-file validation, missing-field failure,
  protocol/config mismatch, scope mismatch, and query-write violations.
- [x] Run the focused tests and verify they fail because the module is absent.
- [x] Implement typed validation helpers and the versioned output schema.
- [x] Run the focused tests and verify they pass.

### Task 2: Aggregate effectiveness and provenance

**Files:**
- Modify: `tests/test_eventqa_paper_aggregator.py`
- Modify: `scripts/eval/eventqa_paper_aggregator.py`

- [x] Add tests for repeat mean/population-standard-deviation metrics,
  Bank-off deduplication, per-context output, transition counts, artifact paths,
  config identity, and reserved cost/token/retrieval fields.
- [x] Verify the new tests fail for missing aggregation behavior.
- [x] Implement the minimal aggregation and Markdown rendering behavior.
- [x] Verify the full focused test file passes.

### Task 3: Run on authoritative P6/P7 artifacts

**Files:**
- Create: `configs/eval/eventqa_paper_aggregator.json`
- Create: `outputs/mab/eventqa_paper_aggregate.json`
- Create: `outputs/mab/eventqa_paper_aggregate.md`
- Modify: `research_notes/EXPERIMENTS.md`
- Modify: `research_notes/PROGRESS.md`

- [x] Freeze the five P6 and five P7 completed run roots in one explicit config.
- [x] Run the CLI without inference and generate JSON/Markdown outputs.
- [x] Verify Bank-off, P6, and P7 headline metrics against the authoritative
  five-repeat summary.
- [x] Record the completed aggregation stage and next cost-smoke gate.

### Task 4: Final verification

- [x] Run focused tests.
- [x] Run the CLI a second time and verify deterministic output.
- [x] Run `git diff --check` and inspect changed paths.
- [x] Confirm no inference process or GPU job was started.

## Non-goals

- No model loading, inference, GPU use, cost claims, new baseline results, or
  modification of prompt/parser/scorer behavior.
- No silent fallback from missing fields and no invented metric values.
- No commit.
