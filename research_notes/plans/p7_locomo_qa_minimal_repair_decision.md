# P7 LoCoMo-QA Minimal Repair Decision

Date: 2026-07-05

## Route Question

After the final read-only prompt/answer-path audit, what is the smallest safe repair to implement before any further LoCoMo scaling?

## Decision

- Verdict: `repair approved`
- Action: `iterate`
- Scope: `runner-side prompt/output-contract cleanup only`

## Decisive Evidence

### What is already good

- Disabled and P7 already share the same LoCoMo QA prompt.
- Frozen P7 method does not need to change.
- LoCoMo frozen-bank protocol does not need to change.
- The single shared raw answer source is clearly identified:
  - `env.final_answer`

### What is broken

- raw `env.final_answer` is copied directly into:
  - `prediction_text`
  - `raw_prediction_text`
- scorer only does shallow normalization
- prompt-leak, no-context denial, and malformed-answer patterns are not classified
- current outputs therefore confound:
  - answer correctness
  - answer formatting
  - prompt contamination

## Approved minimal repair

Implement only:

1. shared stricter QA prompt wording
2. runner-side cleaned-answer extraction
3. preservation of untouched `raw_prediction_text`
4. contract flags for failure categories
5. diagnostics separation so run-level summaries stop contradicting row-level outputs

Do not implement:

- P7 method changes
- model-internal changes
- LoCoMo protocol changes
- scorer redesign beyond minor compatibility adjustments if strictly needed

## Exact hook point

Attach the repair:

- after `result["prediction"]` is available
- before `build_prediction_record(...)`

This gives one shared hook for:

- Disabled
- P7

## Exact next stage

1. implement the minimal runner-side repair
2. run the tiny validation slice:
   - `1` conversation
   - `5` QA rows
   - Disabled + P7
3. re-check:
   - prompt-leak frequency
   - no-context denial frequency
   - `query_write_count == 0`
   - whether at least some clean exact answers appear

## Evidence Paths

- `outputs/mab/p7_locomo_qa_prompt_answer_path_audit.md`
- `outputs/mab/p7_locomo_qa_prompt_answer_path_audit.json`
- `outputs/mab/p7_locomo_qa_prompt_diagnostics_repair_plan.md`
- `outputs/mab/locomo_qa_pilot_2conv_diagnosis.md`

## Bottom Line

The repair is approved next. The smallest safe change is runner-side extraction and contract-flagging after `env.final_answer`, with raw text preserved and frozen P7 untouched.
