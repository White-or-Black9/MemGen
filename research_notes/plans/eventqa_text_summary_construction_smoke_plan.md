# EventQA Same-Model Text-Summary Construction Smoke Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans`, `superpowers:test-driven-development`, and the
> `experiment` skill. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construct and validate one frozen 128-token rolling text summary for
EventQA context 0 using the same frozen MemGen/Qwen model with no latent bank.

**Architecture:** Process the exact EventQA chunks in source order. At each
step, render a fixed summarization prompt containing only the previous summary
and current chunk, call the same model directly with
`latent_memory_bank=None`, decode only newly generated tokens, and cap the
persistent summary at 128 Qwen tokens. Preserve a complete per-step trace.

**Tech Stack:** Python 3.10, existing EventQA preparation, MemGen direct
generation, Qwen tokenizer, PyTorch CUDA metrics, `unittest`, tmux.

---

## Run Contract

- Tier: auxiliary/dev construction smoke; no QA effectiveness result.
- Scope: EventQA-65536 context 0, every prepared source chunk in original order.
- Model: current frozen `Kana-s/MemGen` Qwen2.5-1.5B checkpoint; bank off,
  `latent_memory_bank=None`; no training or external summarizer.
- Summary budget: maximum 128 Qwen tokens after every update.
- Prompt: fixed English instruction; previous summary plus current chunk only;
  no EventQA question, candidates, question IDs, or gold answers.
- Generation: deterministic, maximum 128 new tokens per update.
- Trace: chunk index/hash/token count; previous summary hash/token count;
  rendered input hash/token count; raw output hash/token count; persisted
  summary hash/token count; truncation/empty/format flags; per-step latency;
  aggregate construction latency and peak GPU allocation.
- Stop conditions: missing/reordered chunk, prompt overflow/truncation, empty
  summary, output above budget after persistence, query/gold leakage, non-finite
  cost, incomplete trace, model/config drift, or generation error.

### Task 1: TDD prompt, persistence and artifact contract

- [x] Add failing tests for fixed prompt content, absence of question/gold
  fields, deterministic 128-token persistence truncation, chunk ordering,
  trace hashes and finite cost.
- [x] Implement the pure prompt/persistence/validation helpers.
- [x] Confirm focused tests GREEN.

### Task 2: Implement construction-only runner

- [x] Implement context0 loading, direct deterministic bank-off generation,
  per-step trace and artifact writing without any QA calls.
- [x] Run focused and related EventQA tests, compile/CLI and `git diff --check`.

### Task 3: Launch and validate construction smoke

- [x] Inspect GPU occupancy and launch one detached context0 construction job.
- [x] Verify tmux, PID, GPU, log and output directory once.
- [x] Validate the full chunk trace and inspect final/step summaries for empty,
  looping, language drift, instruction leakage and malformed output.
- [x] Record go/no-go for context0 q0-9 summary-query smoke and stop without
  launching it.

## Construction Smoke Result (2026-07-06)

- Artifact:
  `outputs/mab/eventqa_text_summary_construction_smoke/20260706T091043Z-eventqa-text-summary-construction-ctx0/construction_artifact.json`.
- Structural integrity: all `17/17` chunks processed in order; hash chain,
  token budget, capacity and finite-cost checks passed; final summary has 81
  tokens; construction `34.973 s`; max incremental peak `1814.2 MiB`.
- Qualitative gate: **failed**. The trace contains Chinese and Russian language
  drift, high repetition, instruction/meta-text leakage, and degenerate
  one-token outputs (`conti`, `結束`). The final summary mostly describes the
  last Kamala event and does not preserve the full-context event state.
- Decision: **NO-GO for q0-9 QA under this direct same-model rolling-summary
  protocol.** No query smoke or full pass was launched. The result is a valid
  construction failure diagnostic, not an effectiveness baseline.
- Next action: decide whether to (a) retain this as a limitation and move to
  P7 no-query-retrieval, or (b) define a materially different summary baseline
  such as an external summarizer. Do not silently prompt-tune this failed
  protocol into a new method.

## Non-goals

- No QA query smoke, five-context summary construction, full pass,
  no-query-retrieval ablation, paper claim, dependency installation, or commit.
