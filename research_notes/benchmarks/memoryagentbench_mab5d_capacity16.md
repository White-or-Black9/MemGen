# MAB-5D: detective_qa Capacity16 Decoupled Retrieval-Update Thresholds n10

## Purpose
Test whether increasing `max_slots` from 8 to 16 preserves the decoupled-threshold mechanism while reducing eviction churn on detective_qa n10.

## Settings
- Split: `Long_Range_Understanding`
- Subtask: `detective_qa`
- Contexts: 10 local rows
- threshold: `0.03`
- retrieve_threshold: `0.03`
- update_threshold: `0.05`
- top_k: `1`
- max_slots: `16`
- configured_max_slots: `16`
- actual_bank_max_slots: `16`
- retrieve_policy: `threshold_topk`
- update_policy: `thread_update`
- query mode: first-query-only
- query phase: read-only
- full-history detective_qa: `over_capacity_invalid`
- no fallback
- retrieved memory: Reasoner-only
- retrieved_latents_enter_weaver: `false`

## Provenance
- Canonical run:
  `outputs/mab/capacity16_detectiveqa_n10/20260623T022140Z-detectiveqa-capacity16-n10`
- Earlier non-canonical attempt:
  `outputs/mab/capacity16_detectiveqa_n10/20260623T015929Z-detectiveqa-decoupled-thresholds-n10`
  This earlier artifact is not canonical MAB-5D because the run identity and provenance were still aligned to the MAB-5C naming path during the first attempt.

## Run Status
- Valid contexts: `10 / 10`
- Bank-off exact match: `0.0`
- Bank-on exact match: `0.0`
- Delta accuracy: `0.0`
- Output changed: `10`
- Query-turn retrieval active contexts: `10`
- Final slot counts: `[16, 16, 16, 16, 16, 16, 16, 16, 16, 16]`
- Mean final slot count: `16.0`
- Total write count: `326`
- Total retrieval count: `316`
- Total retrieved latent count: `2272`
- Construction-time retrieval count: `306`
- Query-turn retrieved latent count: `80`
- Query write count: `0`
- Query write attempt count: `0`
- Cross-context leakage detected: `False`
- Retrieved latents entered Reasoner: `True`
- Retrieved latents entered Weaver: `False`
- Write action counts: `{'insert': 160, 'replace_matched': 33, 'evict_oldest_insert': 133}`
- Update reason counts: `{'empty_bank': 10, 'matched_thread': 33, 'new_thread': 150, 'new_thread_bank_full': 133}`
- Append/insert count: `160`
- Matched replace count: `33`
- Capacity evict count: `133`
- Query-turn retrieved indices: `[[0], [6], [8], [7], [5], [5], [12], [3], [6], [3]]`
- Query-turn retrieved scores: `[[0.05318152788346277], [0.043195476796175004], [0.05503939785319073], [0.036925165648343146], [0.045982281750766935], [0.05271706039103078], [0.05132365791373481], [0.05248482664481479], [0.0476079179742789], [0.05550386534562272]]`
- Query-turn retrieved score range: `[[0.05318152788346277, 0.05318152788346277], [0.043195476796175004, 0.043195476796175004], [0.05503939785319073, 0.05503939785319073], [0.036925165648343146, 0.036925165648343146], [0.045982281750766935, 0.045982281750766935], [0.05271706039103078, 0.05271706039103078], [0.05132365791373481, 0.05132365791373481], [0.05248482664481479, 0.05248482664481479], [0.0476079179742789, 0.0476079179742789], [0.05550386534562272, 0.05550386534562272]]`

## Baseline Comparison
- Against MAB-5C: capacity increased from `8` to `16` and final slot count increased from `8` to `16` in every context.
- Against MAB-5C: `capacity_evict_count` dropped from `210` to `133`.
- Against MAB-5C: `total_retrieved_latent_count` changed from `2288` to `2272`.
- Against MAB-5C: official exact match remained `0.0`.

## Per-context Result Table
| context_index | exact_match_off | exact_match_on | output_changed | query_turn_retrieval_active | query_turn_retrieved_latent_count |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | True | 1 | 8 |
| 1 | 0 | 0 | True | 1 | 8 |
| 2 | 0 | 0 | True | 1 | 8 |
| 3 | 0 | 0 | True | 1 | 8 |
| 4 | 0 | 0 | True | 1 | 8 |
| 5 | 0 | 0 | True | 1 | 8 |
| 6 | 0 | 0 | True | 1 | 8 |
| 7 | 0 | 0 | True | 1 | 8 |
| 8 | 0 | 0 | True | 1 | 8 |
| 9 | 0 | 0 | True | 1 | 8 |

## Interpretation
MAB-5D is a valid capacity ablation. Increasing `max_slots` from 8 to 16 raises the final slot count to 16 in every context and reduces eviction churn, but it does not improve official exact match. This is mechanism-positive and performance-neutral/negative, so the next meaningful direction is routing/usage work such as MAB-6A / Version B rather than further capacity-only scaling.

## Notes
- Do not treat the earlier 20260623T015929Z artifact as canonical MAB-5D.
- Relaxed or substring-close answers are useful diagnostics only; they do not replace official exact match.
- Context 6 is a useful example of semantic closeness without official exact-match credit:
  - gold_answer: `C. Misty Sketches`
  - bank_on_prediction: `答案：C. Misty Sketches\n答案`
  - official exact_match: `0`

