# MAB-6B-FR EventQA 65536 n5

EventQA frozen-context-bank configuration sweep on all 5 local EventQA-65536
contexts. This note preserves both the accepted 15-run A/B/C `top_k=1` sweep
and the accepted end-to-end Config B `top_k=2` ablation as strong exploratory
evidence, not a direct official full-context baseline comparison.

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
- end-to-end Config B `top_k=2`:
  `outputs/mab/eventqa_configB_allctx_topk2/20260630T084500Z-eventqa-65536-version-b-weaver-space-bank-n5`
- standalone Config B `top_k=2` ctx4 reproducibility diagnostic:
  `outputs/mab/eventqa_configB_ctx4_topk2_rerun/20260630T121127Z-eventqa-65536-version-b-weaver-space-bank-n5`

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

- Config A remains the highest-EM and most output-stable accepted setting
- Do not promote Config B or C `top_k=1` as better EventQA defaults
- Treat the end-to-end Config B `top_k=2` result below as a strong positive
  multi-slot signal, but do not treat it as the same frozen bank queried with
  one additional slot
- Defer `top_k=4` until the Config B `top_k=2` context-4 collapse is understood
- Keep the caveat explicit in every summary:
  this is strong exploratory compressed frozen-context bridge evidence, not a
  direct official full-context baseline comparison

## End-to-End Config B `top_k=2` Ablation

Accepted artifact:

`outputs/mab/eventqa_configB_allctx_topk2/20260630T084500Z-eventqa-65536-version-b-weaver-space-bank-n5`

This is a valid end-to-end Config B ablation. The mechanism intentionally uses
the same `top_k` during construction-time memory update and query-time
retrieval. Changing `top_k` therefore changes both bank construction and query
retrieval; this result must not be interpreted as the same frozen bank queried
with one extra slot.

### Global Comparison

| Setting | Bank-off EM | Bank-on EM | Bank-on recall | Format failures | Chinese outputs | Final slots |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Config A `top_k=1` | 4/500 | 114/500 | 0.266 | 123 | 23 | `{1:500}` |
| Config B `top_k=1` | 4/500 | 72/500 | 0.202 | 141 | 120 | `{16:500}` |
| Config B `top_k=2` | 4/500 | 109/500 | 0.290 | 131 | 98 | `{15:100,16:400}` |

- Config B `top_k=2` versus Config B `top_k=1`:
  EM `+37`, recall `+0.088`, format failures `-10`, Chinese outputs `-22`
- Config B `top_k=2` versus Config A `top_k=1`:
  EM `-5`, recall `+0.024`, format failures `+8`, Chinese outputs `+75`

### Per-context Results

| Context | Off EM | On EM | On recall | Format | Chinese | Retrieved pair | Final slots | vs B k1 | vs A k1 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 0/100 | 19/100 | 0.23 | 21 | 0 | `(1,0):100` | 16 | +15 | +1 |
| 1 | 0/100 | 45/100 | 0.45 | 1 | 1 | `(1,0):99,(0,1):1` | 16 | +30 | +3 |
| 2 | 0/100 | 22/100 | 0.22 | 1 | 0 | `(0,1):100` | 16 | +20 | +1 |
| 3 | 3/100 | 22/100 | 0.25 | 14 | 14 | `(1,0):100` | 15 | +1 | +5 |
| 4 | 1/100 | 1/100 | 0.30 | 94 | 83 | `(1,0):100` | 16 | -29 | -15 |

### Retrieval and Integrity

- retrieved pairs: `{(1,0):399,(0,1):101}`
- top-1 indices: `{1:399,0:101}`
- top-2 indices: `{0:399,1:101}`
- candidate slots before top-k: `{16:400,15:100}`
- retrieved latent count: `{16:500}`
- routing remained fixed per context; only one context-1 question swapped pair
  order
- one host-access launch completed on GPU 2; no failed attempt occurred
- `500/500` valid questions; manifest and `run_config.json` matched
  `retrieve_threshold=0.03`, `update_threshold=0.09`, `max_slots=16`,
  `top_k=2`, `generation_max_length=40`, `requested_contexts=5`, and
  `eventqa_protocol=frozen_context_bank`
- total / maximum query write-count delta: `0 / 0`
- blocked query write attempts: `500`
- changed bank snapshots: `0`; cross-context leakage: `0`; errors: `0`

### Interpretation

- End-to-end `top_k=2` is a strong positive signal for the multi-slot
  mechanism: it raises Config B from `72/500` to `109/500`, approaches Config
  A's `114/500`, and exceeds Config A recall (`0.290` versus `0.266`).
- Output stability remains worse than Config A, especially Chinese-script
  outputs (`98` versus `23`).
- Gains generalize across contexts 0-3. Context 4 catastrophically regresses
  and is the dominant remaining concern.
- Retrieval remains fixed per context rather than question-specific.
- `top_k=4` remains deferred until the context-4 collapse is understood.

## Context-4 Collapse Diagnosis

Exact comparison from artifacts:

| Setting | Bank-on EM | Recall | Format failures | Chinese outputs | Retrieval |
| --- | ---: | ---: | ---: | ---: | --- |
| Config A `top_k=1` | 16/100 | 0.26 | 56 | 22 | `(0,):100` |
| Config B `top_k=1` | 30/100 | 0.30 | 5 | 1 | `(0,):100` |
| Config B `top_k=2` | 1/100 | 0.30 | 94 | 83 | `(1,0):100` |

The failure is primarily generation-format instability rather than loss of all
answer evidence. Of 83 Chinese-script outputs, 82 are also format failures.
Of 94 format failures, 29 still contain the full gold answer in raw output.
Those 29 are exactly the recall-positive / EM-negative cases: malformed short
prefixes such as `iston`, `人`, or `顿也` become the parsed answer while the
correct answer remains on a later line. This explains recall `0.30` with EM
`0.01`.

Retrieval and construction evidence:

- the final-score top pair is `(1,0)` for all 100 queries; top-1 minus top-2
  margin ranges `0.01208-0.01626` with mean `0.01382`
- after removing the recorded recency factor offline, raw cosine still ranks
  ctx4 slots `(1,0)` first for all 100 queries; the pair is not an index-tie
  artifact
- ctx4 slot 1 has `created_step=2`, `access_count=15`, age `0`; slot 0 has
  `created_step=3`, `access_count=14`, age `0`
- construction performed `16` inserts and `1` matched replacement; capacity
  eviction and replace-old counts are `0`
- top_k=2 repeatedly refreshed both slots through construction while later
  slots remained stale; this recency feedback strongly separates the pair from
  later slots, although cosine also ranks the pair highest
- Config B `top_k=1` constructed a different bank path: ctx4 slot 0 alone had
  `created_step=2`, `access_count=15`, age `0`, and its next final-score slot
  was slot 14 while its next raw-cosine slot was slot 3
- ctx1 and ctx2 have the same `16 insert + 1 matched replace` structure and
  similarly fixed local slot-0/slot-1 routing, but produce 45 and 22 EM with
  almost no format or Chinese failures. Slot indices are context-local; the
  ctx4 slot-1/slot-0 pair, not slot 1 globally, is pathological.

Evidence limits:

- artifacts do not store slot key tensors, query vectors, key/query norms, or
  query-to-query similarity, so query-representation collapse cannot be tested
  directly
- artifacts do not map final slots to source chunk text or preserve which
  chunk last replaced each slot, so semantic slot provenance is unavailable
- raw cosine can be reconstructed from final score and recorded age, but the
  stored diagnostics cannot separate bad slot content from bad query content

Hypothesis assessment:

- H1 slot-pair content/provenance problem: plausible and consistent with the
  context-local failure, but unproven because chunk provenance is absent
- H2 construction instability: strongly supported; changing `top_k` changes
  the construction path and creates a different ctx4 bank, while contexts 0-3
  improve
- H3 recency feedback: supported as an amplifying lock-in mechanism, but not
  the sole ranking cause because raw cosine also ranks `(1,0)` first
- H4 query representation collapse: plausible from fixed routing, but query
  vectors are absent and the hypothesis cannot be tested offline
- H5 generation instability: strongly supported by 94 malformed outputs, 83
  Chinese outputs, and 29 raw outputs that contain gold but fail EM
- H6 dataset difficulty: rejected as the primary cause because the same ctx4
  questions reach 30/100 under Config B `top_k=1`

Primary next action: add score-decomposition and slot/chunk-provenance
diagnostics, then rerun Config B `top_k=2` on context 4 only. This is lower risk
than changing retrieval behavior and can determine whether the pathological
pair is selected because of query collapse, semantic slot content, or recency
feedback before another mechanism ablation is chosen.

## Config B `top_k=2` ctx4 Standalone Reproducibility Diagnostic

Accepted artifact:

`outputs/mab/eventqa_configB_ctx4_topk2_rerun/20260630T121127Z-eventqa-65536-version-b-weaver-space-bank-n5`

This standalone rerun is a reproducibility and stability diagnostic only. It
does not replace the accepted all-context Config B `top_k=2` ablation above.

### Runtime Contract

- `retrieve_threshold=0.03`
- `update_threshold=0.09`
- `max_slots=16`
- `top_k=2`
- `generation_max_length=40`
- `context_index=4`
- `requested_contexts=1`
- `eventqa_protocol=frozen_context_bank`
- `100/100` valid questions
- query write-count delta total / max: `0 / 0`
- bank snapshots changed after query: `0`
- cross-context leakage detected: `0`

### Result

- Bank-off EM: `1/100 = 0.01`
- Bank-on EM: `11/100 = 0.11`
- Bank-off recall: `0.19`
- Bank-on recall: `0.28`
- format failures: `52/100`
- Chinese outputs: `44/100`
- final slot count: `16`
- peak CUDA memory max / mean: `11.17 GiB / 9.46 GiB`

### Comparison

- Versus previous Config B `top_k=2` all-context ctx4:
  EM `+10`, recall `-0.02`, format failures `-42`, Chinese outputs `-39`
- Retrieved pair changed from `(1,0):100` to `(3,0):84` / `(0,3):16`
- Versus Config B `top_k=1` ctx4:
  EM `-19`, recall `-0.02`, format failures `+47`, Chinese outputs `+43`
- Versus Config A `top_k=1` ctx4:
  EM `-5`, recall `+0.02`, format failures `-4`, Chinese outputs `+22`

### Retrieval and Construction

- retrieved pair distribution: `{(3,0):84,(0,3):16}`
- top-1 indices: `{3:84,0:16}`
- top-2 indices: `{0:84,3:16}`
- retrieved latent count: `{16:100}`
- candidate slots before top-k: `{16:100}`
- final slot count: `16`
- routing is still nearly fixed, but fixed to a different pair than the
  accepted all-context ctx4 run
- construction statistics:
  `insert=16`, `matched_replace=1`, `capacity_evict=0`, `replace_old=0`
- the standalone ctx4 bank differs from the all-context ctx4 bank
- top-1 minus top-2 margin mean dropped from `0.01382` in the all-context ctx4
  run to `0.00123` in the standalone rerun

### Output Failure Profile

- Chinese outputs: `44`
- format failures: `52`
- Chinese outputs that are also format failures: `36`
- format failures containing the full gold answer: `18`
- recall-positive but EM-negative: `17`
- answer-present-but-parser-lost behavior remains visible, but less severe than
  in the all-context ctx4 collapse

### Interpretation

- The catastrophic ctx4 collapse is not stable as a single deterministic
  outcome.
- Config B `top_k=2` ctx4 remains much worse than Config B `top_k=1` ctx4.
- The dominant slot pair changed from local `(1,0)` to local `(3,0)` / `(0,3)`,
  indicating construction-path or routing instability rather than pure
  same-bank generation noise.
- This strengthens the need for score-decomposition and slot/chunk-provenance
  diagnostics.
- `top_k=4` remains deferred.

Recommended next step: add score-decomposition and slot/chunk-provenance
diagnostics, then rerun ctx4 only.
