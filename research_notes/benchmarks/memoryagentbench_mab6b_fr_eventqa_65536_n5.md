# MAB-6B-FR EventQA 65536 n5

Exploratory EventQA expansion using a top_k=1 Weaver-space bank. This note
preserves the completed all-5-context frozen-context result. It is strong
exploratory evidence, not a final benchmark claim.

## Active Result Boundary

- Result type: strong exploratory 5-context evidence
- Context coverage: `5/5` EventQA 65536 contexts
- Evaluated contexts: `context_index=0..4`
- Protocol: `frozen_context_bank`
- Bank-off baseline: compressed bridge Bank-off only; not an official
  long-context full-history baseline because full history remains over capacity
- Canonical detective note protected:
  `research_notes/benchmarks/memoryagentbench_mab6b_weaver_space_bank.md`

## Runs

- script:
  `scripts/eval/mab6b_weaver_space_bank_eventqa_65536_n5.py`
- runner scheduling support:
  `--context-index` was added so one EventQA context can be forced into one
  isolated process / output root for safe per-context parallel evaluation.
- run roots:
  - `ctx0`:
    `outputs/mab/version_b_weaver_space_bank_eventqa_65536_n5_ctx0/20260629T131415Z-eventqa-65536-version-b-weaver-space-bank-n5`
  - `ctx1`:
    `outputs/mab/version_b_weaver_space_bank_eventqa_65536_n5_ctx1/20260629T131413Z-eventqa-65536-version-b-weaver-space-bank-n5`
  - `ctx2`:
    `outputs/mab/version_b_weaver_space_bank_eventqa_65536_n5_ctx2/20260629T131413Z-eventqa-65536-version-b-weaver-space-bank-n5`
  - `ctx3`:
    `outputs/mab/version_b_weaver_space_bank_eventqa_65536_n5_ctx3/20260629T133550Z-eventqa-65536-version-b-weaver-space-bank-n5`
  - `ctx4`:
    `outputs/mab/version_b_weaver_space_bank_eventqa_65536_n5_ctx4/20260629T133555Z-eventqa-65536-version-b-weaver-space-bank-n5`
- protocol: `frozen_context_bank`
- peak CUDA memory by context:
  - `ctx0`: `11151741952` bytes
  - `ctx1`: `11162152448` bytes
  - `ctx2`: `11169515008` bytes
  - `ctx3`: `11161227264` bytes
  - `ctx4`: `11167909376` bytes

## Settings

- retrieve_threshold: `0.03`
- update_threshold: `0.05`
- top_k: `1`
- max_slots: `8`
- generation_max_length: `40`
- metric: `substring_exact_match`
- optional metric: `eventqa_recall`

### Runtime-Config Correction

- The preserved all-5 run used the actual runtime config
  `retrieve_threshold=0.03`, `update_threshold=0.05`, `top_k=1`, and
  `max_slots=8`.
- The earlier claim that these artifacts used the intended cautious config
  `retrieve_threshold=0.005`, `update_threshold=0.08`, `top_k=1`, and
  `max_slots=16` is corrected and must not be used to attribute this result.
- The old `run_config.json` and `manifest.json` files contain the intended
  constants rather than the runtime values because the runner passed the
  DetectiveQA `_bank_config()` into execution.
- The frozen-context lifecycle evidence and recorded result numbers remain
  valid; only the configuration provenance is corrected.

## Protocol Validation

- all contexts used `frozen_context_bank`
- `context_memorization_count=1` in all 5 contexts
- `same_frozen_bank_reused_across_queries=true` in all 5 contexts
- all 100 queries per context shared the same frozen bank instance
- `bank_snapshot_changed_after_query=false` in all 5 contexts
- `total_query_write_count_delta=0` in all 5 contexts
- `max query_write_count_delta=0` in all 5 contexts
- blocked query write attempts total: `500`
- blocked query write attempts distribution: `{1:500}`
- `cross_context_leakage_detected=false` in all 5 contexts

Interpretation: the benchmark-conformant EventQA lifecycle is now runtime
validated for this runner across all 5 contexts. Each context was memorized
once, the same frozen bank was reused across all 100 queries, and query turns
remained read-only.

## Mechanism Summary

- single-slot collapse occurred in all 5 contexts
- each context had construction `chunk_count=17`
- each context ended with `final_slot_count=1`
- each context had `true_insert_count=1` and `true_matched_replace_count=16`
- `true_capacity_evict_count=0` in all 5 contexts
- `true_replace_old_slot_count=0` in all 5 contexts
- aggregate candidate slot count before top-k distribution: `{1:500}`
- aggregate retrieved indices distribution: `{(0,):500}`
- aggregate retrieved latent count distribution: `{8:500}`
- aggregate raw candidate score min / max / mean:
  `0.04529 / 0.07478 / 0.05641`

Interpretation: construction-time single-slot collapse remains under the actual
`0.03/0.05/1/8` runtime config. The bank behaves like one compressed latent
memory slot rather than diverse event slots.

## Overall Result

- total questions: `500`
- Bank-off substring exact match / accuracy: `4/500 = 0.008`
- Bank-on substring exact match / accuracy: `83/500 = 0.166`
- absolute improvement: `+0.158`
- Bank-off `eventqa_recall`: `0.178`
- Bank-on `eventqa_recall`: `0.208`
- improved / regressed / unchanged: `81 / 2 / 417`
- `output_changed_count=500`
- format failure counts: bank-off `377`, bank-on `173`
- Chinese-script output counts: bank-off `189`, bank-on `30`
- per-context EM:
  - `ctx0`: `0/100 -> 17/100`
  - `ctx1`: `0/100 -> 3/100`
  - `ctx2`: `0/100 -> 19/100`
  - `ctx3`: `3/100 -> 21/100`
  - `ctx4`: `1/100 -> 23/100`
- per-context improved / regressed / unchanged:
  - `ctx0`: `17 / 0 / 83`
  - `ctx1`: `3 / 0 / 97`
  - `ctx2`: `19 / 0 / 81`
  - `ctx3`: `19 / 1 / 80`
  - `ctx4`: `23 / 1 / 76`

Interpretation:

- Bank-on consistently improves over the compressed-bridge Bank-off baseline
  across all 5 local EventQA contexts.
- This is strong exploratory evidence for the benchmark-conformant
  `frozen_context_bank` protocol.
- It is not a final benchmark-improvement claim and it is not an official full
  long-context baseline comparison.
- The remaining mechanism risk is the persistent single-slot collapse in all 5
  contexts.

## Representative Examples

- Improved `ctx0 q1`:
  - gold: `Debbie expressed her boredom with the talk of war.`
  - bank-off parsed: `paragraph 1`
  - bank-on parsed:
    `Debbie expressed her boredom with the talk of war.`
- Improved `ctx1 q13`:
  - gold:
    `The man stated that he would remain at the inn despite the host's refusal.`
  - bank-off parsed: `[list of events]`
  - bank-on parsed:
    `The event that happens next is: "The man stated that he would remain at the inn despite the host's refusal."`
- Improved `ctx2 q3`:
  - gold: `Julianne complimented Pascal's abilities as a seaman.`
  - bank-off parsed: `简短`
  - bank-on parsed:
    `Julianne complimented Pascal's abilities as a seaman.`
- Improved `ctx3 q1`:
  - gold:
    `Miss Rosie approached the house and looked in at the window.`
  - bank-off parsed: `[list of events that have already occurred]`
  - bank-on parsed:
    `Miss Rosie approached the house and looked in at the window.`
- Improved `ctx4 q0`:
  - gold:
    `Charissa discovered a letter revealing Edie Arkadyevitch's infidelity.`
  - bank-off parsed: `简短`
  - bank-on parsed:
    `: Charissa discovered a letter revealing Edie Arkadyevitch's infidelity.`
- Regressed `ctx3 q42`:
  - gold:
    `Mrs. Sylvia kissed Davy goodbye as he was leaving for Yarmouth.`
  - bank-off parsed:
    `[list] 'Mrs. Sylvia handed Davy a sandwich before he departed for Hastings.', 'Mrs. Sylvia kissed Davy goodbye as he was leaving for Yarmouth.', "Mrs. Sylvia slipped a`
  - bank-on parsed: `文摘`
- Regressed `ctx4 q6`:
  - gold:
    `Edie Arkadyevitch read a liberal newspaper and contemplated its views.`
  - bank-off parsed:
    `without the event that Edie Arkadyevitch read a liberal newspaper and contemplated its views. Based on the context provided, the next event that logically follows is Edie Arkadyevitch visited an`
  - bank-on parsed:
    `: Edie Arkadyevitch attended a political rally and debated the speeches delivered.`
- Unchanged wrong early example `ctx0 q0`:
  - gold:
    `Debbie wore a new green flowered-muslin dress that matched her slippers.`
  - bank-off parsed:
    `简短的文本中没有提供具体的后续事件。我需要进一步搜索以获取更多信息。`
  - bank-on parsed:
    `Debbie sported a lovely yellow paisley wrap that accompanied her flip-flops.`
- Unchanged wrong middle example `ctx2 q33`:
  - gold:
    `M. Julianne arrived at La Reserve and was greeted with applause.`
  - bank-off parsed: `[list]`
  - bank-on parsed: `}`
- Unchanged wrong late example `ctx4 q99`:
  - gold:
    `Agafea Mihalovna greeted Ladonna when he returned home.`
  - bank-off parsed: `[list of events]`
  - bank-on parsed:
    `: "Ladonna taught Agafea Mihalovna how to dance at the village's annual celebration."`

## Next-Step Boundary

- The all-5 result and `--context-index` scheduling support are preserved, with
  the historical runtime config corrected above.
- After runtime-config integrity repair, the recommended next experiment is a frozen-context
  slot-collapse / update-threshold diagnostic, not more immediate full EventQA
  runs.
- Treat the current result as strong exploratory evidence that the
  `frozen_context_bank` protocol can outperform the compressed-bridge Bank-off
  baseline across all 5 local contexts.
- Keep the single-slot-collapse risk explicit in every summary or
  recommendation.
