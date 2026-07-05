# P7 LoCoMo-QA Bank Slot Next-Step Decision

Date: 2026-07-05

## Route Question

After tracing LoCoMo-QA repaired P7 bank-slot semantics, what is the smallest correct next action?

## Decision

- Verdict: `trace understood`
- Action: `iterate`
- Route: `protocol/task-fit diagnosis, not counter repair`

## Why

The read-only trace resolves the two counter questions:

1. `final_slot_count = 3`
   - not a capacity bug
   - not a partial-ingestion bug
   - expected because the current LoCoMo construction path created exactly `3` construction chunks and wrote once per chunk

2. `retrieved_latent_count = 16`
   - not `16` slots
   - it is `16` latent vectors/tokens
   - expected under `top_k = 2` and `8` latents per retrieved slot

So the mainline should not branch into counter debugging.

## Strongest Evidence

- full normalized conversations were ingested:
  - `conv-26`: `19` sessions, `419` turns
  - `conv-30`: `19` sessions, `369` turns
- both conversations were chunked into exactly `3` construction chunks
- construction counters are consistent with:
  - chunk 1: write
  - chunk 2: retrieve + write_back
  - chunk 3: retrieve + write_back
- repaired P7 rows still show:
  - active retrieval
  - `query_write_count == 0`
  - weak QA performance

## Main Risk

The remaining issue is not instrumentation ambiguity.
It is likely benchmark fit:

- too few chunk-level writes for a long multi-session conversation
- too coarse latent memory granularity
- answer generation still fails even when retrieval is active

## What Not To Do Next

Do not spend the next step on:

- slot-count bug hunting
- `retrieved_latent_count` bug hunting
- frozen P7 method changes based only on these counters

## Recommended Next Step

Run a focused read-only diagnosis or plan a minimal protocol-level adjustment study around:

1. construction granularity
   - whether chunk-level memorization is too coarse for LoCoMo
2. retrieval usefulness
   - whether the same two-slot / 16-latent retrieval pattern persists on denial and wrong-answer rows
3. generation failure despite retrieval
   - whether the answer path is ignoring usable retrieved memory

## Evidence Paths

- `outputs/mab/locomo_qa_p7_bank_slot_trace.md`
- `outputs/mab/locomo_qa_p7_bank_slot_trace.json`
- `outputs/mab/locomo_qa_pilot_repaired_p7_2conv/conv-26/`
- `outputs/mab/locomo_qa_pilot_repaired_p7_2conv/conv-30/`
- `outputs/mab/locomo_qa_pilot_repaired_2conv_comparison.md`

## Bottom Line

The counters are now explained well enough. The next useful work is mechanism/task-fit diagnosis, not counter repair.
