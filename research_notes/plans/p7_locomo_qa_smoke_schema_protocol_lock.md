# LoCoMo-QA Smoke Schema / Protocol Lock for Frozen P7

Date: 2026-07-04

## Purpose

This document locks the LoCoMo-QA smoke output schema and protocol invariants after the successful `disabled` and frozen-`p7` smoke runs. The smoke is treated as runner/protocol validation only. It is not performance evidence and should not be cited as benchmark evidence.

## Inspected Artifacts

- `outputs/mab/locomo_qa_smoke_disabled/`
- `outputs/mab/locomo_qa_smoke_p7/`
- `outputs/mab/locomo_qa_frozen_runner_check.md`
- `outputs/mab/locomo_qa_frozen_runner_check.json`

## Smoke Status

- Disabled smoke completed on `conv-26` with `2` QA rows.
- P7 smoke completed on `conv-26` with `2` QA rows.
- Both modes emitted:
  - `prediction_records.jsonl`
  - `scored_prediction_records.jsonl`
  - `aggregate_metrics.json`
  - `run_diagnostics.json`
  - `run_summary.md`
- P7 enforced `query_write_count == 0` on all QA rows.
- No P7 method change.
- No model-code change.
- No GPT judge or external API.

## Protocol Lock

The frozen LoCoMo-QA protocol is locked as follows:

1. One LoCoMo conversation maps to one session-local latent bank.
2. Construction happens before QA.
3. Construction ingests all selected conversation chunks sequentially.
4. In `p7` mode, a frozen bank snapshot is created after construction and before QA.
5. Multiple questions may be answered against the same frozen snapshot.
6. If snapshot restoration is needed between questions, the same frozen snapshot is restored before each question.
7. Query-time retrieval is allowed in `p7`.
8. Query-time writes are blocked in `p7`.
9. `query_write_count` must be exactly `0` for every QA row in frozen-bank QA.
10. QA scoring is deterministic only:
   - normalized exact match
   - token F1
11. No GPT/LLM judge is part of the LoCoMo-QA path.

## Final Output Schema Lock

The smoke established the concrete emitted schema below. For pilot and formal runs, the row-level JSONL schema is the authoritative source for per-question scoring and cost analysis.

### `prediction_records.jsonl`

One row per QA prediction with the following fields:

- `bank_snapshot_changed_after_query`
- `category`
- `category_name`
- `construction_retrieve_count`
- `construction_write_count`
- `conversation_id`
- `final_slot_count`
- `gold_answer`
- `latency_seconds`
- `method`
- `mode`
- `output_token_count`
- `peak_gpu_memory`
- `prediction`
- `prediction_status`
- `prediction_text`
- `query_retrieval_active_count`
- `query_write_attempt_count`
- `query_write_count`
- `question`
- `question_id`
- `raw_prediction_text`
- `retrieved_latent_count`
- `trigger_call_count`
- `weaver_call_count`

Notes:

- `conversation_id` and `question_id` are stable derived IDs from normalized adapter outputs.
- `prediction`, `prediction_text`, and `raw_prediction_text` are all emitted. Downstream scoring should use the scorer’s normalized prediction field from the scored file.
- Row-level latency is stored as `latency_seconds`.
- The schema is identical across `disabled` and `p7`.

### `scored_prediction_records.jsonl`

This extends each prediction row with deterministic scorer fields:

- all `prediction_records.jsonl` fields
- `exact_match`
- `invalid_output`
- `normalized_gold_answer`
- `normalized_prediction`
- `scorer_version`
- `status`
- `token_f1`

### `aggregate_metrics.json`

Top-level fields:

- `by_category`
- `by_conversation`
- `cost_summary`
- `invalid_output_count`
- `method`
- `mode`
- `overall_macro_by_conversation`
- `overall_micro`
- `record_count`
- `scorer_version`

Nested metrics currently observed:

- `overall_micro.exact_match_mean`
- `overall_micro.token_f1_mean`
- `overall_macro_by_conversation.exact_match_mean`
- `overall_macro_by_conversation.token_f1_mean`
- `by_category.<category_name>.count`
- `by_category.<category_name>.exact_match_mean`
- `by_category.<category_name>.token_f1_mean`
- `by_conversation.<conversation_id>.count`
- `by_conversation.<conversation_id>.exact_match_mean`
- `by_conversation.<conversation_id>.token_f1_mean`
- `cost_summary.mean_latency_seconds`
- `cost_summary.max_peak_gpu_memory`
- `cost_summary.mean_output_token_count`
- `cost_summary.mean_construction_write_count`
- `cost_summary.mean_construction_retrieve_count`
- `cost_summary.mean_query_retrieval_active_count`
- `cost_summary.mean_retrieved_latent_count`
- `cost_summary.mean_query_write_count`
- `cost_summary.mean_final_slot_count`
- `cost_summary.mean_trigger_call_count`
- `cost_summary.mean_weaver_call_count`

### `run_diagnostics.json`

Common fields observed in both modes:

- `chunk_count`
- `chunk_token_lengths`
- `conversation_id`
- `gpu_used`
- `mode`
- `query_write_count_zero`
- `selected_question_ids`

P7-only fields currently observed:

- `snapshot_metadata`
- `construction_diagnostics`

Observed nested P7 fields:

- `snapshot_metadata.conversation_id`
- `snapshot_metadata.mode`
- `snapshot_metadata.final_slot_count`
- `construction_diagnostics.construction_write_count`
- `construction_diagnostics.construction_retrieve_count`
- `construction_diagnostics.final_slot_count`
- `construction_diagnostics.trigger_call_count`
- `construction_diagnostics.weaver_call_count`
- `construction_diagnostics.construction_latency_seconds`
- `construction_diagnostics.construction_peak_gpu_memory`

### `run_summary.md`

Human-readable smoke summary containing at least:

- mode
- conversation ID
- selected question IDs
- exact match mean
- token F1 mean
- invalid output count
- `query_write_count_zero`

## Smoke Findings Relevant to Pilot Readiness

### Stable and acceptable now

- Disabled and P7 output file sets are consistent.
- Row-level schemas are consistent across modes.
- Category fields are present:
  - `category`
  - `category_name`
- Peak GPU memory is present on prediction rows.
- Trigger / Weaver counters are present on prediction rows.
- `retrieved_latent_count` is present on prediction rows.
- `invalid_output_count` was `0` in both smokes.
- P7 row-level `bank_snapshot_changed_after_query` remained `false`.

### Issues to record before pilot

1. Disabled smoke predictions showed prompt-leak / malformed-answer behavior.
   Example pattern:
   - `"question\n<|I'm sorry, but the"`
   This is not a schema blocker, but it means pilot interpretation must treat early disabled outputs as harness validation only until a broader pilot confirms prompt behavior.

2. `run_diagnostics.json` is not a full substitute for row-level cost/counter analysis.
   In the inspected P7 smoke:
   - row-level `trigger_call_count` / `weaver_call_count` were `1`
   - `construction_diagnostics.trigger_call_count` / `construction_diagnostics.weaver_call_count` were `0`
   Also:
   - `construction_diagnostics.construction_latency_seconds` was `0.0`
   - `construction_diagnostics.construction_peak_gpu_memory` was `null`
   Therefore pilot and formal analysis should treat:
   - `prediction_records.jsonl`
   - `scored_prediction_records.jsonl`
   - `aggregate_metrics.json`
   as authoritative, with `run_diagnostics.json` treated as secondary run metadata.

3. Aggregated score fields are nested, not flat.
   Consumers must read:
   - `overall_micro.exact_match_mean`
   - `overall_micro.token_f1_mean`
   rather than assuming flat top-level score keys.

4. Row-level latency is stored as `latency_seconds`, not `latency`.

5. GPU execution still depends on unsandboxed Python visibility for CUDA.
   This is an execution-environment issue, not a runner protocol issue.

## Locked Interpretation

- The smoke is sufficient to lock the LoCoMo-QA frozen-bank protocol.
- The smoke is sufficient to lock the emitted JSONL/JSON schema for pilot consumption.
- The smoke is not sufficient to claim benchmark quality, model quality, or method benefit.
- The next run should be a small pilot with `disabled` and `p7` only.

## Decision

Proceed to LoCoMo-QA pilot planning with the following guardrails:

- keep the normalized adapter outputs as the dataset contract
- keep `disabled` and `p7` only
- keep deterministic EM/F1 only
- preserve row-level schema unchanged if possible
- treat any future schema change as a deliberate versioned change, not an incidental one
