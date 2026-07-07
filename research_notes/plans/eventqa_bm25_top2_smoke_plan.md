# EventQA BM25 Top-2 Retrieved-Text Smoke Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans`, `superpowers:test-driven-development`, and the
> `experiment` skill. Steps use checkbox syntax for tracking.

**Goal:** Validate a deterministic, traceable BM25 top-2 explicit-text baseline
on EventQA context 0, q0-9 under the current prompt/parser/scorer and cost
contracts.

**Architecture:** Build a standard-library BM25 index over the exact EventQA
chunks produced by the frozen runner. Rank chunks from the unchanged official
question text, inject the two selected source chunks into a single read-only
query prompt, and run MemGen with no latent memory bank. Record retrieval,
prompt, scoring, token, latency, and GPU provenance per question.

**Tech Stack:** Python 3.10 standard library BM25, existing EventQA preparation
and scoring helpers, PyTorch CUDA metrics, `unittest`, tmux.

## Run Contract

- Tier: auxiliary/dev smoke; not a paper result.
- Scope: EventQA-65536 context 0, q0-9.
- Retrieval: deterministic BM25 (`k1=1.5`, `b=0.75`), top-2 exact source
  chunks, ties broken by chunk index.
- Query: unchanged official non-strict EventQA question and candidates.
- Generation/scoring: frozen MemGen, max generation length 40, unchanged local
  official parser/scorer, no latent memory bank.
- Required provenance: chunk IDs/indices, BM25 scores, chunk text hashes,
  query/prompt hashes, injected token count, capacity check, prediction,
  EM/recall/format flags, retrieval/generation/end-to-end cost, output tokens,
  and peak GPU allocation.
- Stop conditions: fewer than two valid chunks, unstable ranking, missing IDs or
  hashes, gold-specific tuning, prompt capacity overflow or undocumented
  truncation, scorer drift, incomplete ten-question scope, or mixed cost fields.

### Task 1: TDD deterministic retrieval and prompt construction

- [x] Add failing tests for tokenization, BM25 ranking, stable tie-breaking,
  exact top-2 provenance, and prompt construction.
- [x] Implement the minimal standard-library BM25 index and prompt builder.
- [x] Run focused tests.

### Task 2: Implement standalone smoke runner

- [x] Add failing tests for schema validation, q0-9 scope, capacity failure,
  and method-separable cost fields.
- [x] Implement the one-method EventQA runner and artifact writer.
- [x] Run focused and related tests plus `git diff --check`.

### Task 3: Launch bounded smoke

- [x] Inspect GPU occupancy and select one viable GPU.
- [x] Launch one detached BM25 q0-9 process.
- [x] Verify tmux, PID, GPU, log growth, and output-root creation once.
- [x] Stop without launching the five-context full pass.

### Task 4: Validate and route after completion

- [x] Validate all ten records, provenance, token counts, capacity, cost fields,
  and scorer outputs.
- [x] Compare smoke effectiveness/cost with existing Disabled/P7 q0-9 artifacts.
- [x] Record go/no-go for the BM25 full pass.

## Smoke Result (2026-07-06)

- Artifact:
  `outputs/mab/eventqa_bm25_top2_smoke/20260706T025910Z-eventqa-bm25-top2-ctx0-q0-9-smoke/smoke_artifact.json`.
- Integrity: `10/10` records passed schema, provenance, finite-cost, hash, and
  capacity checks; maximum rendered prompt was `8627/32768` tokens.
- BM25 top-2: EM `0.10`, EventQA recall `0.30`, format failures `2/10`.
- Method cost excluding model load and scoring: index `0.023 s`, retrieval
  `0.020 s`, generation `13.629 s`, total `13.671 s`; incremental peak GPU
  allocation `3555.6 MiB`.
- Same-scope reference artifacts: Disabled EM/recall `0.00/0.20`, total
  `7.169 s`; P7 EM/recall `0.40/0.40`, total `19.952 s` including `14.222 s`
  construction.
- Decision: **GO for a separate five-context BM25 full pass**, because the
  smoke is executable, capacity-safe, improves over Disabled on this tiny
  subset, and provides a necessary explicit-text comparator. This is routing
  evidence only: the ten-question smoke does not establish superiority or
  statistical reliability. The full pass was not launched in this phase.

## Non-goals

- No full BM25 run, matched-budget truncation, text-summary baseline, P7 change,
  or paper superiority claim in this phase.
- No dependency installation and no commit.
