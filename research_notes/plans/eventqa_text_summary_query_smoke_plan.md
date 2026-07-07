# EventQA Frozen Text-Summary Query Smoke Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans`, `superpowers:test-driven-development`, and the
> `experiment` skill. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure context0 q0-9 QA behavior using the already frozen 81-token
same-model rolling summary, without regenerating or repairing it.

**Architecture:** Load and strictly validate the canonical construction
artifact, prepend its final summary with fixed framing to each unchanged
EventQA query, and run ten independent Bank-off queries with the frozen model.
Record summary/prompt provenance, effectiveness, format and method-separable
query cost; carry construction cost from the source artifact separately.

**Tech Stack:** Python 3.10, existing EventQA runner/scorer, frozen summary
artifact, PyTorch CUDA metrics, `unittest`, tmux.

---

## Contract

- Scope: context0 q0-9; no summary regeneration or prompt tuning.
- Summary source: canonical construction artifact ending in an 81-token final
  summary; construction quality is not a stop condition by user decision.
- Query: unchanged official question/candidates and scorer, generation length
  40, Bank-off, no latent memory.
- Injection: fixed `Persistent memory summary:` framing plus exact frozen
  summary; record summary tokens and actual rendered prompt delta.
- Cost: carried construction cost, standalone query cost, total amortized cost,
  output tokens and incremental peak GPU allocation.
- Stop only on provenance/schema/hash failure, summary mutation, query scope
  mismatch, prompt overflow, scorer drift, missing/non-finite cost or process
  failure.

### Task 1: TDD prompt/artifact contract

- [x] Add failing tests for exact frozen-summary injection, provenance hash,
  q0-9 coverage, prompt delta, effectiveness/format and finite costs.
- [x] Implement pure helpers and standalone runner.
- [x] Run focused and related tests, CLI/compile and `git diff --check`.

### Task 2: Launch and validate query smoke

- [x] Inspect GPU occupancy; launch one detached context0 q0-9 process.
- [x] Verify tmux/PID/GPU/log/output once.
- [x] Validate ten records and compare same-scope Disabled/BM25/matched16/P7.
- [x] Record go/no-go for five-context same-model summary full pass and stop
  without launching it.

## Query Smoke Result (2026-07-06)

- Artifact:
  `outputs/mab/eventqa_text_summary_query_smoke/20260706T102214Z-eventqa-text-summary-query-ctx0-q0-9/query_artifact.json`.
- Integrity: `10/10` records valid; frozen summary hash unchanged; summary 81
  tokens; actual rendered prompt delta 85 for every question; no capacity or
  scorer failure.
- Effectiveness: EM `0.10`, recall `0.10`, format failures `2/10`.
- Smoke cost: construction `34.973 s`, query `10.870 s`, total `45.843 s`,
  query incremental peak `179.0 MiB`. GPU 4 had concurrent external load, so
  these timing values are diagnostic only, not paper-facing cost evidence.
- Same-scope effectiveness references: Disabled `0.00/0.20`, BM25 top-2
  `0.10/0.30`, matched16 `0.10/0.30`, P7 `0.40/0.40` (EM/recall).
- Decision by user-defined evidence boundary: **GO for a five-context
  same-model text-summary full pass despite poor summary quality**. The poor
  output is attributed to baseline model capability and remains part of the
  measured result. No full pass was launched in this phase.

## Non-goals

- No summary repair, reconstruction, full pass, external summarizer, P7 change,
  significance claim, dependency installation, or commit.
