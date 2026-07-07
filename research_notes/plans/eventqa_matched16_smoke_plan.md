# EventQA 16-Token Matched-Budget Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans`, `superpowers:test-driven-development`, and the
> `experiment` skill. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate a deterministic EventQA explicit-text baseline that injects
exactly 16 source token IDs per question at query time before any full pass.

**Architecture:** Reuse the frozen BM25 top-2 chunk ranking. Tokenize each
selected chunk with the actual Qwen tokenizer, enumerate contiguous 8-token
windows, score each window from the query terms with the same corpus BM25 IDF
weights, and choose the highest-scoring window with earliest-offset tie break.
Inject the two decoded windows at query position with no latent bank and record
both source-token accounting and actual rendered-prompt token delta.

**Tech Stack:** Python 3.10, existing standard-library BM25 implementation,
Qwen tokenizer, EventQA bridge/scorer, PyTorch CUDA metrics, `unittest`, tmux.

---

## Run Contract

- Tier: auxiliary/dev smoke; not a paper result.
- Scope: EventQA-65536 context 0, q0-9.
- Retrieval: unchanged BM25 (`k1=1.5`, `b=0.75`) top-2 source chunks.
- Window selection: exactly 8 Qwen source token IDs from each retrieved chunk;
  maximize weighted query-term overlap, tie break by lowest token start offset.
- Budget: exactly 16 selected source token IDs per question. Constant prompt
  framing and tokenizer boundary effects are separately reported as rendered
  prompt delta and are not silently counted as source evidence tokens.
- Query/model/scorer: unchanged official question/candidates, frozen MemGen,
  generation length 40, unchanged parser/scorer, bank off.
- Provenance: chunk IDs/indices/scores/hashes; window token start/end, 8 token
  IDs, decoded text/hash/score; total source token count; official/matched
  rendered token counts and delta; prediction/metrics/format/cost/GPU fields.
- Stop conditions: any window shorter than 8 tokens, source total other than
  16, non-contiguous offsets, decode/token-ID provenance mismatch, unstable
  selection, extra unrecorded evidence, prompt overflow, scorer drift,
  incomplete q0-9 scope, or non-finite/mixed cost.

### Task 1: TDD window selection and prompt contract

**Files:**
- Create: `tests/test_eventqa_matched16_retrieved_text.py`
- Create: `scripts/eval/eventqa_matched16_retrieved_text.py`

- [x] Add failing tests for exact 8-token windows, relevance ranking,
  earliest-offset tie break, two-window 16-token accounting, and source
  provenance.
- [x] Run the focused test and confirm RED because the module is absent.
- [x] Implement tokenizer-agnostic selection over supplied token IDs plus a
  real-tokenizer adapter and matched prompt builder.
- [x] Re-run focused tests and confirm GREEN.

### Task 2: TDD smoke artifact and standalone runner

**Files:**
- Modify: `tests/test_eventqa_matched16_retrieved_text.py`
- Modify: `scripts/eval/eventqa_matched16_retrieved_text.py`

- [x] Add failing validation tests for exact context0 q0-9 coverage, exactly
  two 8-token windows, 16 source tokens total, offsets/hashes, prompt delta,
  capacity, effectiveness, format, and method-separable cost.
- [x] Implement the standalone bank-off runner by reusing frozen EventQA and
  BM25 helpers without changing parser/scorer/model configuration.
- [x] Run focused and related EventQA tests, CLI/compile checks, and
  `git diff --check`.

### Task 3: Launch and validate bounded smoke

**Files:**
- Create: `runtime_logs/eventqa_matched16_smoke_20260706/run_smoke.sh`
- Modify: `research_notes/PROGRESS.md`
- Modify: `research_notes/EXPERIMENTS.md`
- Modify: `research_notes/plans/eventqa_matched16_smoke_plan.md`

- [x] Inspect GPU occupancy and select one viable GPU.
- [x] Launch one detached context0 q0-9 process and verify tmux, PID, GPU, log,
  and output root once.
- [x] After completion, validate all ten records and compare with same-scope
  Disabled, BM25 top-2, and P7 artifacts.
- [x] Record go/no-go for a five-context matched16 full pass and stop without
  launching it.

## Smoke Result (2026-07-06)

- The first wiring run
  (`20260706T034900Z-eventqa-matched16-ctx0-q0-9-smoke`) selected 16 source
  token IDs but added 94 rendered prompt tokens because visible framing and
  source hashes entered the prompt. It is rejected as a failed diagnostic.
- The corrected canonical run is
  `outputs/mab/eventqa_matched16_smoke/20260706T035141Z-eventqa-matched16-ctx0-q0-9-smoke/smoke_artifact.json`.
- Integrity: `10/10` records valid; every question has two contiguous 8-token
  source windows, 16 source token IDs total, and rendered prompt delta exactly
  16; provenance, hashes, capacity, scorer, format and cost fields passed.
- Corrected result: EM `0.10`, EventQA recall `0.30`, format failures `7/10`,
  method total `9.758 s`, max incremental peak GPU allocation `128.5 MiB`.
- Same-scope references: Disabled `0.00/0.20`, `7.169 s`, `121.9 MiB`;
  full BM25 top-2 `0.10/0.30`, `13.671 s`, `3555.6 MiB`; P7 `0.40/0.40`,
  `19.952 s` including construction, `129.3 MiB` (EM/recall, time, peak).
- Decision: **GO for a separate five-context matched16 full pass**. The smoke
  validates exact position matching and feasibility only; it is not paper
  evidence. No full pass was launched in this phase.

## Non-goals

- No matched16 full pass, summary baseline, P7 change, no-query-retrieval
  ablation, significance claim, dependency installation, or commit.
