# P7 LoCoMo-QA Prompt / Diagnostics Repair Plan

Date: 2026-07-05

## Goal

Define the smallest safe repairs needed before scaling LoCoMo-QA beyond the 2-conversation pilot.

This plan is intentionally limited to:

- prompt / output-contract stabilization
- answer extraction cleanup
- diagnostics / cost accounting cleanup

This plan does **not** change:

- frozen P7 thresholds or bank behavior
- the frozen LoCoMo session-local / frozen-bank protocol
- model internals
- paper files

## Executive Decision

- Keep frozen P7 unchanged.
- Keep LoCoMo one-conversation-one-bank protocol unchanged.
- Do **not** scale to all 10 conversations yet.
- Apply the smallest LoCoMo-runner-side prompt/extraction repair first.
- Treat diagnostics repair as required before paper-facing cost claims, but second priority behind prompt/output repair.

## Inspected Code Paths

Primary runner:

- `scripts/eval/mab6b_weaver_space_bank_locomo_qa.py`

Reused helper paths:

- `scripts/eval/mab5a_detectiveqa_compressed_n10.py`
- `scripts/eval/mab6b_weaver_space_bank_detectiveqa_n10.py`
- `scripts/eval/mab6b_weaver_space_bank_eventqa_65536_n5.py`
- `scripts/eval/mab2_bank_off.py`
- `scripts/eval/locomo_qa_scorer.py`

## What the prompt/output audit found

### 1. Disabled and P7 already share the same LoCoMo QA prompt string

In `scripts/eval/mab6b_weaver_space_bank_locomo_qa.py`, both modes use:

- `LOCOMO_QUERY_TEMPLATE`
- `build_question_payload(...)`
- `query_prompt`

Current template:

```text
Based on the conversation history you memorized, answer the question concisely.

Question: {question}

Answer:
```

So the high-level QA prompt string is already shared across Disabled and P7, which is good and should be preserved.

### 2. Construction prompt is separate and not the main issue

Current construction prompt:

```text
Please memorize the following conversation chunk ({i}/{n}) for future question answering.
```

This is not the likely source of the benchmark failure. The main problems appear at query-time answer generation and answer extraction.

### 3. The LoCoMo runner reuses a generic DetectiveQA compressed-turn stack

The LoCoMo runner delegates Disabled execution to:

- `mab5a_detectiveqa_compressed_n10._run_model`

and P7 query/construction behavior to:

- `mab6b_weaver_space_bank_eventqa_65536_n5._run_eventqa_model`
- `mab6b_weaver_space_bank_detectiveqa_n10._run_model`

Those helpers were not originally designed around a LoCoMo-specific short-answer contract.

Key consequence:

- the LoCoMo runner does not own the final turn separation logic directly
- it inherits a generic MAB environment and final-answer path

### 4. The final prediction currently comes from `env.final_answer` with no LoCoMo-specific cleanup

Both reused helper stacks ultimately set:

- `prediction = env.final_answer`

via `MABEpisodeEnv` in `scripts/eval/mab2_bank_off.py`.

The LoCoMo runner then stores:

- `prediction`
- `prediction_text`
- `raw_prediction_text`

all as the same raw returned string.

There is currently no LoCoMo-specific postprocessing such as:

- strip everything before `Answer:`
- strip leading prompt fragments
- remove accidental question restatement
- normalize special chat/template markers

### 5. The scorer only sees the runner’s `prediction_text`

`scripts/eval/locomo_qa_scorer.py`:

- scores `prediction_text`
- does basic lowercase/whitespace/punctuation normalization
- does **not** perform LoCoMo-aware answer extraction
- does **not** strip prompt fragments or `Answer:` delimiters

So if the runner records a contaminated prediction, the scorer preserves that contamination.

### 6. `prediction_status` is too weak to catch current failure modes

Current status values are effectively:

- `ok`
- `missing`
- `empty`

In the pilot:

- all `304` Disabled rows had `prediction_status = ok`
- all `304` P7 rows had `prediction_status = ok`

That means current status flags do not distinguish:

- prompt-leak
- no-context denial output
- answer-wrapped-in-sentence
- malformed answer

### 7. Generation length is short but still long enough to permit drift

Current LoCoMo runner:

- `DEFAULT_GENERATION_MAX_LENGTH = 40`

That is short enough to avoid long rambling, but still long enough for:

- question restatement
- disclaimer responses
- extra sentence wrapping

### 8. Diagnostics/cost data mixes authoritative row-level data with weak run-level construction summaries

Current reliable path:

- row-level `prediction_records.jsonl`
- row-level `scored_prediction_records.jsonl`
- `aggregate_metrics.json`

Current weak path:

- `run_diagnostics.json.construction_diagnostics`

Observed mismatch in pilot:

- row-level P7 `trigger_call_count = 1`
- `run_diagnostics.json` construction `trigger_call_count = 0`
- same mismatch for `weaver_call_count`
- construction latency remained `0.0`
- construction peak memory remained `null`

## Minimal repair proposed

## A. Prompt / output contract

### Objective

Keep the same benchmark protocol and same high-level QA prompt for both modes, but make the expected answer shape much stricter and easier to extract.

### Proposed contract

Use one shared QA prompt for Disabled and P7:

```text
Answer the question using the memorized conversation history.
Return only the short final answer.
Do not restate the question.
Do not explain your reasoning.
If the answer is a date, name, number, or short phrase, return only that answer.

Question: {question}
Answer:
```

Key properties:

- same QA prompt for Disabled and P7
- explicit answer-only instruction
- no chain-of-thought request
- no mention that context may be absent
- retains the `Answer:` delimiter
- short-answer target is explicit

### Why this is minimal

- preserves the current LoCoMo runner structure
- does not touch P7 bank logic
- does not touch model internals
- does not require scorer redesign

## B. Runner-side answer extraction

### Objective

Preserve both raw and cleaned outputs, but make scoring use the cleaned answer text instead of the raw turn string.

### Proposed runner-side fields

Keep:

- `raw_prediction_text`

Add or revise:

- `prediction_text`
  - cleaned answer text used for scoring
- `prediction_extraction_status`
  - `answer_delimiter`
  - `fallback_first_line`
  - `raw_passthrough`
  - `empty_after_clean`
- `prediction_contract_flags`
  - `contains_prompt_leak`
  - `contains_no_context_denial`
  - `contains_question_restatement`

### Proposed extraction order

1. If `Answer:` appears, keep only text after the last `Answer:`.
2. Strip leading/trailing whitespace and chat-marker residue.
3. If the remaining text is multi-line, keep the first non-empty line.
4. If no `Answer:` exists, use first non-empty line as fallback.
5. If output still starts with obvious leak patterns such as:
   - `question`
   - `Question:`
   - `<|`
   - `I'm sorry, but ... no conversation history`
   mark it with contract flags.
6. Preserve the original raw string in `raw_prediction_text`.

### Important rule

Do not overwrite the raw model output. Keep:

- raw output for debugging
- cleaned answer for scoring

## C. Scorer contract

### Minimal scorer-side rule

The scorer should continue to score `prediction_text`, but after runner-side cleaning that field should mean:

- final extracted answer text

Do not move complex prompt cleanup into the scorer if it can be avoided. The runner should own contract-specific extraction.

### Why

- scoring remains deterministic
- LoCoMo contract stays explicit
- raw/debug text remains available separately

## D. Diagnostics / cost accounting repair

### Objective

Make run-level diagnostics stop contradicting row-level outputs.

### Authoritative row-level fields

Continue treating these as authoritative:

- `latency_seconds`
- `peak_gpu_memory`
- `output_token_count`
- `query_write_count`
- `construction_write_count`
- `construction_retrieve_count`
- `query_retrieval_active_count`
- `retrieved_latent_count`
- `final_slot_count`
- `trigger_call_count`
- `weaver_call_count`

### Proposed run-level separation

Split run-level diagnostics into:

1. `construction_summary`
   - one record per conversation construction phase
2. `query_summary`
   - aggregates over QA rows

Do not compute construction metrics by reusing row-level query counters unless the semantics are exact.

### Minimal construction-summary fields

- `construction_executed`
- `construction_write_count`
- `construction_retrieve_count`
- `construction_final_slot_count`
- `construction_trigger_call_count`
- `construction_weaver_call_count`
- `construction_latency_seconds`
- `construction_peak_gpu_memory`
- `construction_metrics_available`

### If a construction metric is not actually measured

Record:

- `null`
- plus `construction_metrics_available = false`

Do **not** emit a misleading numeric placeholder like `0.0` when the metric was not really measured.

### Trigger / Weaver counting rule

Define explicitly:

- construction counters come only from construction-time generations
- query counters come only from query-time generations
- row-level query counters should not be backfilled into construction summaries

### Cost reporting rule

Until repaired, paper-facing cost claims should use:

- row-level query cost only

and treat construction cost as:

- unavailable or provisional

## Proposed tiny validation slice after repair

Use exactly:

- `1` conversation
- `5` QA rows
- Disabled + P7

Suggested purpose:

- prompt/output-contract validation only

### Validation checks

1. Prompt-leak frequency drops sharply in both modes.
2. No-context denial frequency drops sharply in P7.
3. `query_write_count == 0` still holds for every P7 row.
4. `prediction_status` and extraction-status fields become informative.
5. At least some rows produce clean short answers.
6. EM may still be low, but it should no longer be dominated by obvious prompt contamination.

### Success criteria for reopening scale decision

- clear reduction in prompt-leak rows
- clear reduction in no-context denial rows
- stable cleaned-answer extraction
- nonzero EM on at least some rows is strongly preferred before scaling

## Minimal files to modify later

Primary expected file:

- `scripts/eval/mab6b_weaver_space_bank_locomo_qa.py`

Possible small scorer adjustment only if needed:

- `scripts/eval/locomo_qa_scorer.py`

Likely no need to modify:

- P7 model internals
- latent-memory method code
- broader EventQA runner code

## What should remain unchanged

- frozen P7 thresholds and bank policy
- session-local bank reset boundary
- construction-before-QA ordering
- frozen snapshot reuse protocol
- query-time retrieval allowed
- query-time writes blocked
- `query_write_count == 0` invariant
- deterministic LoCoMo scoring

## Recommended next step before coding

Perform one read-only implementation audit focused on:

- exact rendered query prompt text for Disabled and P7
- exact raw decoded answer strings before runner packaging
- where `Answer:` can be reliably extracted
- whether no-context denial outputs correlate with missing or malformed rendered query prompts

After that audit, implement only:

- shared stricter QA prompt text
- runner-side answer extraction / status flags
- diagnostics separation so run-level construction summaries stop contradicting row-level outputs
