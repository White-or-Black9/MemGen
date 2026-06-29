# MAB-6B-FR EventQA 65536 n5

Exploratory EventQA expansion using the cautious top_k=1 Weaver-space bank
setting. This note preserves single-context evidence only; it is not a final
benchmark claim.

## Active Result Boundary

- Result type: exploratory single-context evidence
- Context coverage: `1/5` EventQA 65536 contexts
- Evaluated context: `context_index=0`
- Protocol: `frozen_context_bank`
- Bank-off baseline: compressed bridge Bank-off only; not an official
  long-context full-history baseline because full history remains over capacity
- Canonical detective note protected:
  `research_notes/benchmarks/memoryagentbench_mab6b_weaver_space_bank.md`

## Run

- command:
  `CUDA_VISIBLE_DEVICES=3 /home/baishilong/miniconda3/envs/memgen/bin/python scripts/eval/mab6b_weaver_space_bank_eventqa_65536_n5.py --requested-contexts 1 --skip-research-note --eventqa-protocol frozen_context_bank`
- run root:
  `outputs/mab/version_b_weaver_space_bank_eventqa_65536_n5/20260629T121408Z-eventqa-65536-version-b-weaver-space-bank-n5`
- script:
  `scripts/eval/mab6b_weaver_space_bank_eventqa_65536_n5.py`
- protocol: `frozen_context_bank`
- GPU: `3`
- peak CUDA memory: `11151741952` bytes, about `10.39 GiB`

## Settings

- retrieve_threshold: `0.005`
- update_threshold: `0.08`
- top_k: `1`
- max_slots: `16`
- generation_max_length: `40`
- metric: `substring_exact_match`
- optional metric: `eventqa_recall`

## Protocol Validation

- `context_memorization_count=1`
- `same_frozen_bank_reused_across_queries=true`
- all 100 queries shared the same frozen bank instance: `true`
- `bank_snapshot_changed_after_query=false`
- `total_query_write_count_delta=0`
- `max query_write_count_delta=0`
- blocked query write attempts total: `100`
- blocked query write attempts distribution: `{1:100}`

Interpretation: the benchmark-conformant EventQA lifecycle is now runtime
validated for this runner. Context was memorized once, the same frozen bank was
reused across all 100 queries, and query turns remained read-only.

## Mechanism Summary

- construction `chunk_count=17`
- construction `final_slot_count=1`
- `true_insert_count=1`
- `true_matched_replace_count=16`
- `true_capacity_evict_count=0`
- `true_replace_old_slot_count=0`
- candidate slot count before top-k distribution: `{1:100}`
- retrieved indices distribution: `{(0,):100}`
- retrieved latent count distribution: `{8:100}`
- raw candidate score min / max / mean:
  `0.04947 / 0.05574 / 0.05229`

Interpretation: construction-time single-slot collapse remains. Under the
current EventQA context, the bank behaves like one compressed latent memory
slot rather than diverse event slots.

## Context 0 Result

- context_id: `eventqa-aea0f2d603e1c8a3`
- question_count: `100`
- Bank-off substring exact match / accuracy: `0/100 = 0.00`
- Bank-on substring exact match / accuracy: `22/100 = 0.22`
- Bank-off `eventqa_recall`: `15/100 = 0.15`
- Bank-on `eventqa_recall`: `22/100 = 0.22`
- improved / regressed / unchanged: `22 / 0 / 78`
- `output_changed_count=100`
- format failure counts: bank-off `83`, bank-on `19`
- Chinese-script output counts: bank-off `36`, bank-on `0`

Interpretation:

- This is the first benchmark-conformant EventQA positive signal for the
  `frozen_context_bank` protocol.
- It is still only one EventQA context, so it does not support a final
  benchmark-improvement claim.
- Bank-on improved official EM over the compressed-bridge Bank-off baseline on
  `context_index=0`.
- The remaining mechanism risk is the persistent single-slot collapse.

## Representative Examples

- Improved Q2:
  - gold: `Debbie expressed her boredom with the talk of war.`
  - bank-off parsed: `paragraph 1`
  - bank-on parsed:
    `Debbie expressed her boredom with the talk of war.`
- Improved Q3:
  - gold:
    `Debbie mentioned her mother, Lucian O'Kerry, during the conversation.`
  - bank-off parsed: `回答`
  - bank-on parsed:
    `Debbie mentioned her mother, Lucian O'Kerry, during the conversation.`
- Improved Q49:
  - gold:
    `Marianne felt joy and a sense of ownership standing on the foundation of his new plantation.`
  - bank-off parsed: `[list of events]`
  - bank-on parsed:
    `Marianne felt joy and a sense of ownership standing on the foundation of his new plantation.`
- Unchanged wrong early example Q0:
  - gold:
    `Debbie wore a new green flowered-muslin dress that matched her slippers.`
  - bank-off parsed:
    `简短的文本中没有提供具体的后续事件。我需要进一步搜索以获取更多信息。`
  - bank-on parsed:
    `Debbie sported a lovely yellow paisley wrap that accompanied her flip-flops.`
- Unchanged wrong late example Q99:
  - gold:
    `Sue firmly insisted that Lucian should eat supper before praying.`
  - bank-off parsed: `[list of events]`
  - bank-on parsed:
    `Lucian interrupted the conversation about the brat's death.`

## Next-Step Boundary

- Do not run the remaining 4 EventQA contexts until explicitly approved.
- Treat the current signal as exploratory evidence that the
  `frozen_context_bank` protocol can outperform the compressed-bridge Bank-off
  baseline on one context.
- Before scaling, keep the single-slot-collapse risk explicit in any summary or
  recommendation.

## Git Status

```
## rlm-memory-bank...origin/rlm-memory-bank [ahead 8]
 M research_notes/DECISIONS.md
 M research_notes/EXPERIMENTS.md
 M research_notes/PROGRESS.md
 M research_notes/benchmarks/memoryagentbench_next_steps.md
 M tests/test_mab6b_weaver_space_bank.py
?? docs/
?? paper/
?? research_notes/benchmarks/memoryagentbench_mab6b_fr_eventqa_65536_n5.md
?? research_notes/benchmarks/memoryagentbench_mab6b_fr_retrieve_threshold_relaxation.md
?? scripts/eval/mab6b_weaver_space_bank_detectiveqa_n10_retrieve_threshold_relaxation.py
?? scripts/eval/mab6b_weaver_space_bank_detectiveqa_n10_trigger_trace.py
?? scripts/eval/mab6b_weaver_space_bank_eventqa_65536_n5.py
```
