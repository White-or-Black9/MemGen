# MAB-6B-FR EventQA 65536 n5

EventQA frozen-context-bank configuration sweep on all 5 local EventQA-65536
contexts. This note preserves the accepted 15-run A/B/C sweep as strong
exploratory evidence, not a direct official full-context baseline comparison.

## Result Boundary

- Result type: strong exploratory 5-context sweep evidence
- Context coverage: `5/5` EventQA 65536 contexts
- Evaluated contexts: `context_index=0..4`
- Protocol: `frozen_context_bank`
- Official scorer used: MAB EventQA `substring_exact_match` / Accuracy
- Baseline boundary: compressed frozen-context bridge Bank-off only; not an
  official long-context full-history baseline because full history remains over
  capacity
- Canonical detective note protected:
  `research_notes/benchmarks/memoryagentbench_mab6b_weaver_space_bank.md`

## Artifact Roots

- script:
  `scripts/eval/mab6b_weaver_space_bank_eventqa_65536_n5.py`
- benchmark note:
  `research_notes/benchmarks/memoryagentbench_mab6b_fr_eventqa_65536_n5.md`
- sweep roots:
  - Config A:
    `outputs/mab/eventqa_frozen_context_bank_cfgA_ctx{0..4}/...`
  - Config B:
    `outputs/mab/eventqa_frozen_context_bank_cfgB_ctx{0..4}/...`
  - Config C:
    `outputs/mab/eventqa_frozen_context_bank_cfgC_ctx{0..4}/...`

Latest completed roots:

- Config A:
  - `ctx0`:
    `outputs/mab/eventqa_frozen_context_bank_cfgA_ctx0/20260630T013527Z-eventqa-65536-version-b-weaver-space-bank-n5`
  - `ctx1`:
    `outputs/mab/eventqa_frozen_context_bank_cfgA_ctx1/20260630T014820Z-eventqa-65536-version-b-weaver-space-bank-n5`
  - `ctx2`:
    `outputs/mab/eventqa_frozen_context_bank_cfgA_ctx2/20260630T015305Z-eventqa-65536-version-b-weaver-space-bank-n5`
  - `ctx3`:
    `outputs/mab/eventqa_frozen_context_bank_cfgA_ctx3/20260630T015738Z-eventqa-65536-version-b-weaver-space-bank-n5`
  - `ctx4`:
    `outputs/mab/eventqa_frozen_context_bank_cfgA_ctx4/20260630T020219Z-eventqa-65536-version-b-weaver-space-bank-n5`
- Config B:
  - `ctx0`:
    `outputs/mab/eventqa_frozen_context_bank_cfgB_ctx0/20260630T013529Z-eventqa-65536-version-b-weaver-space-bank-n5`
  - `ctx1`:
    `outputs/mab/eventqa_frozen_context_bank_cfgB_ctx1/20260630T014820Z-eventqa-65536-version-b-weaver-space-bank-n5`
  - `ctx2`:
    `outputs/mab/eventqa_frozen_context_bank_cfgB_ctx2/20260630T015305Z-eventqa-65536-version-b-weaver-space-bank-n5`
  - `ctx3`:
    `outputs/mab/eventqa_frozen_context_bank_cfgB_ctx3/20260630T015738Z-eventqa-65536-version-b-weaver-space-bank-n5`
  - `ctx4`:
    `outputs/mab/eventqa_frozen_context_bank_cfgB_ctx4/20260630T020219Z-eventqa-65536-version-b-weaver-space-bank-n5`
- Config C:
  - `ctx0`:
    `outputs/mab/eventqa_frozen_context_bank_cfgC_ctx0/20260630T013526Z-eventqa-65536-version-b-weaver-space-bank-n5`
  - `ctx1`:
    `outputs/mab/eventqa_frozen_context_bank_cfgC_ctx1/20260630T014820Z-eventqa-65536-version-b-weaver-space-bank-n5`
  - `ctx2`:
    `outputs/mab/eventqa_frozen_context_bank_cfgC_ctx2/20260630T015305Z-eventqa-65536-version-b-weaver-space-bank-n5`
  - `ctx3`:
    `outputs/mab/eventqa_frozen_context_bank_cfgC_ctx3/20260630T015738Z-eventqa-65536-version-b-weaver-space-bank-n5`
  - `ctx4`:
    `outputs/mab/eventqa_frozen_context_bank_cfgC_ctx4/20260630T020219Z-eventqa-65536-version-b-weaver-space-bank-n5`

## Runner Preservation

- runtime config integrity validation is now enforced by the runner:
  `manifest.json` and `run_config.json` matched the intended config for all 15
  completed runs
- `--context-index` remains the scheduling control for isolated one-context
  execution
- `--construction-only` is now supported for EventQA construction-only bank
  inspection
- related EventQA regression tests exist in
  `tests/test_mab6b_weaver_space_bank.py`

## Sweep Design

Common settings:

- `eventqa_protocol=frozen_context_bank`
- `generation_max_length=40`
- `top_k=1`
- `context_index=0..4`
- `--skip-research-note`

Configurations:

- Config A, historical actual control:
  `retrieve_threshold=0.03`, `update_threshold=0.05`, `max_slots=8`,
  `top_k=1`
- Config B, keep retrieve threshold and force larger bank:
  `retrieve_threshold=0.03`, `update_threshold=0.09`, `max_slots=16`,
  `top_k=1`
- Config C, low retrieve-threshold multi-slot candidate:
  `retrieve_threshold=0.005`, `update_threshold=0.09`, `max_slots=16`,
  `top_k=1`

## Integrity

- all 15 runs completed
- `manifest.json` and `run_config.json` matched intended config for every run
- `query_write_count_delta total/max = 0 / 0`
- `bank_snapshot_changed_after_query=false`
- `cross_context_leakage_detected=false`
- `context_memorization_count=1` in every completed run
- query turns remained read-only in every completed run
- canonical detective note unchanged:
  - SHA256:
    `9494ad0ec468633c7703a7ae956dcd00f80cf8c41373652712cdd257bce1fc13`
  - mtime:
    `2026-06-28 16:47:29.322403473 +0800`

## Main Conclusion

- Config A is the best EventQA setting in this sweep:
  `retrieve_threshold=0.03`, `update_threshold=0.05`, `max_slots=8`, `top_k=1`
- Config A obtains Bank-off `4/500` and Bank-on `114/500` on EventQA-65536
  across all 5 contexts
- Config B and C force `15-16` slot construction but reduce Bank-on EM
- multi-slot formation hurts EventQA under `top_k=1` in this compressed bridge
- query-time retrieval still returns exactly one slot, so multi-slot
  construction does not imply multi-slot use
- the result uses the MAB EventQA official `substring_exact_match` / Accuracy
  scorer, but it is still a compressed frozen-context bridge result rather than
  a direct official full-context baseline comparison

## Config A

- `retrieve_threshold=0.03`
- `update_threshold=0.05`
- `max_slots=8`
- `top_k=1`
- Bank-off EM / Accuracy: `4/500 = 0.008`
- Bank-on EM / Accuracy: `114/500 = 0.228`
- Bank-off recall: `0.178`
- Bank-on recall: `0.266`
- improved / regressed / unchanged: `113 / 3 / 384`
- per-context Bank-on EM:
  - `ctx0 0.18`
  - `ctx1 0.42`
  - `ctx2 0.21`
  - `ctx3 0.17`
  - `ctx4 0.16`
- final_slot_count distribution: `{1: 500}`
- retrieved_indices distribution: `{(0,): 500}`
- retrieved_latent_count distribution: `{8: 500}`
- candidate_slot_count_before_topk distribution: `{1: 500}`
- raw candidate score min / max / mean:
  `0.04436 / 0.05992 / 0.05310`
- format failures: Bank-off `377`, Bank-on `123`
- Chinese-script outputs: Bank-off `189`, Bank-on `23`

## Config B

- `retrieve_threshold=0.03`
- `update_threshold=0.09`
- `max_slots=16`
- `top_k=1`
- Bank-on EM / Accuracy: `72/500 = 0.144`
- Bank-on recall: `0.202`
- improved / regressed / unchanged: `72 / 4 / 424`
- final_slot_count distribution: `{16: 500}`
- retrieved_indices distribution: `{(0,): 400, (4,): 100}`
- format failures: Bank-on `141`
- Chinese-script outputs: Bank-on `120`

## Config C

- `retrieve_threshold=0.005`
- `update_threshold=0.09`
- `max_slots=16`
- `top_k=1`
- Bank-on EM / Accuracy: `67/500 = 0.134`
- Bank-on recall: `0.208`
- improved / regressed / unchanged: `66 / 3 / 431`
- final_slot_count distribution: `{15: 100, 16: 400}`
- retrieved_indices distribution: `{(0,): 300, (5,): 100, (12,): 100}`
- format failures: Bank-on `165`
- Chinese-script outputs: Bank-on `116`

## Mechanism Interpretation

- Config A remained a one-slot bridge:
  `construction_final_slot_count distribution = {1: 5}`,
  `true_insert_count distribution = {1: 5}`,
  `true_matched_replace_count distribution = {16: 5}`
- Config B forced multi-slot construction:
  `construction_final_slot_count distribution = {16: 5}`,
  `true_insert_count distribution = {16: 5}`,
  `true_matched_replace_count distribution = {1: 5}`
- Config C also forced multi-slot construction:
  `construction_final_slot_count distribution = {15: 1, 16: 4}`,
  `true_insert_count distribution = {15: 1, 16: 4}`,
  `true_matched_replace_count distribution = {1: 4, 2: 1}`
- Despite larger banks in B and C, query-time retrieval still returned only one
  slot in every query because `top_k=1` and the recorded
  `retrieved_latent_count distribution` stayed `{8: 500}` for every config
- Multi-slot construction therefore changed which single slot was chosen, but
  did not create multi-slot query-time use

## Comparison to the Earlier Preserved Positive Result

- Earlier preserved all-5 result:
  Bank-off `4/500 = 0.008`, Bank-on `83/500 = 0.166`
- Reproduced / repaired Config A:
  Bank-off `4/500 = 0.008`, Bank-on `114/500 = 0.228`
- Config B: `72/500 = 0.144`
- Config C: `67/500 = 0.134`

Interpretation:

- The repaired runner and clean rerun did not merely reproduce the earlier
  `83/500` positive signal; Config A improved it to `114/500`
- The multi-slot candidates did not beat either the earlier preserved signal or
  the repaired Config A result

## Recommendation Boundary

- Keep Config A for the next EventQA setting
- Do not promote Config B or C as better EventQA defaults
- If a later study wants multi-slot benefit, it must explicitly target
  multi-slot query-time retrieval rather than only multi-slot construction
- Keep the caveat explicit in every summary:
  this is strong exploratory compressed frozen-context bridge evidence, not a
  direct official full-context baseline comparison
