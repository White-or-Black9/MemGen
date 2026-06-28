# MAB-6B-FR Diagnostic Summary

- Date consolidated: 2026-06-27
- Scope: artifact-only review of the format-repair, threshold, capacity, and
  top-k diagnostics on the fixed detective_qa n10 slice
- Evaluation boundary: exploratory mechanism evidence; no new inference was
  run and no final benchmark-improvement claim is supported

## 1. Executive Summary

The four MAB-6B-FR diagnostics isolate a useful working region but do not yet
establish a performance gain. Raising `update_threshold` from `0.05` to `0.08`
breaks the one-slot overwrite collapse and, in the recovered threshold
aggregate, preserves the best observed exact match (`2/10`). With
`retrieve_threshold=0.03` and `top_k=1`, `max_slots=16` gives the best observed
storage/quality balance within the n10 diagnostics: `2/10` Bank-on exact match,
full 16-slot occupancy, and fewer capacity evictions than cap8. Increasing
`top_k` does not improve exact match and makes outputs less controlled relative
to top_k=1. The top_k=4 and top_k=8 settings retrieve only three slots (24
latent tokens), because `threshold_topk` filters candidates by
`retrieve_threshold` before truncating to top-k. The next diagnostic should
therefore relax or disable the retrieval threshold while holding
`max_slots=16`, `update_threshold=0.08`, and `top_k=4` fixed.

## 2. Experiment Timeline

1. **Format repair diagnostic** (canonical run started 2026-06-26 01:46 UTC)
   - Artifact root:
     `outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_format_repair/`
   - Canonical run:
     `20260626T014628Z-detectiveqa-version-b-weaver-space-bank-format-repair-n10`
   - The earlier `20260626T014442Z` attempt is invalid: all 10 contexts ended
     with `RecursionError: maximum recursion depth exceeded`.
2. **Threshold-only diagnostic** (2026-06-26 02:33-02:58 UTC)
   - Artifact root:
     `outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_threshold_diagnostic/`
   - Root aggregate was recovered from complete prediction and bank-debug rows;
     the per-setting manifests remain invalid because postprocessing raised
     `KeyError: 'memory_retrieved_latent_count'` for every context.
3. **Capacity diagnostic** (full sweep 2026-06-26 07:14-07:43 UTC)
   - Artifact root:
     `outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_capacity_diagnostic/`
   - The root aggregate is the full n10 result. The nested `smoke_test/` result
     is a one-context preflight and is not included in the main result table.
4. **Top-k diagnostic** (final sweep 2026-06-26 12:11-12:54 UTC)
   - Artifact root:
     `outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_topk_diagnostic/`
   - The final aggregate uses the 12:11 UTC top_k=1 rerun (`1/10`). The earlier
     valid 11:47 UTC top_k=1 result (`2/10`) is retained as run-to-run variance,
     not substituted into the final sweep.

## 3. Settings Table

| Diagnostic | Fixed settings | Swept variable | Artifact root |
|---|---|---|---|
| Format repair | `rt=0.03`, `ut=0.05`, `max_slots=8`, `top_k=1`, Weaver-space bank | final-query format prefix off -> on | `outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_format_repair/` |
| Threshold-only | `rt=0.03`, `max_slots=8`, `top_k=1`, format repair on | `ut={0.05,0.08,0.10,0.12}` | `outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_threshold_diagnostic/` |
| Capacity | `rt=0.03`, `ut=0.08`, `top_k=1`, format repair on | `max_slots={8,16,32}` | `outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_capacity_diagnostic/` |
| Top-k | `rt=0.03`, `ut=0.08`, `max_slots=16`, format repair on | `top_k={1,2,4,8}` | `outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_topk_diagnostic/` |

`rt` denotes `retrieve_threshold`; `ut` denotes `update_threshold`.

## 4. Artifact Integrity and Organization

| Diagnostic | Primary evidence | Trust / caveat |
|---|---|---|
| Format repair | canonical `paired_results.json`, `diagnostics.jsonl`, `run_config.json`, `manifest.json` | 10/10 valid in `014628Z`; the root has no aggregate, per-context export, summary, or run log |
| Threshold-only | root `threshold_diagnostic_aggregate.*`, `threshold_diagnostic_per_context.*`, `threshold_diagnostic_summary.md`; per-setting diagnostics/configs/manifests | recovered diagnostic: generation outputs and bank traces exist, but per-setting `paired_results.json` and manifests report 0/10 valid after a postprocessing KeyError; no run log is present |
| Capacity | root aggregate/per-context/summary plus per-setting artifacts and `run_capacity_full.log` | 10/10 valid for cap8/cap16/cap32; root aggregate's generic write-action totals are zero because it read the wrong field family, so write counts below use per-context `bank_on_*` traces |
| Top-k | root aggregate/per-context/summary, per-setting artifacts, `run_topk_full.log` | all four final settings are 10/10 valid and aggregation completed; `debug_failure_traceback.log` contains a historical OOM, not a failure of the final aggregate |

The capacity and top-k logs also contain tokenizer over-length warnings while
estimating the rejected full-history path. All scored compressed runs returned
status 0; the warning is not evidence that a full-history input was scored.

## 5. Main Result Table

Rows from different diagnostics are not strict paired reruns unless explicitly
stated; several rows come from separate runs and may show run-to-run variance.

Action triples are `insert / replace_matched / capacity_evict`. Format counts
use the runner's primary output classifier. All rows have `bank_off_EM=0.00`
and `output_changed=10/10` unless noted otherwise.

| Diagnostic / setting | Bank-off EM | Bank-on EM | Final slot count | Query retrieved latent count | Key write/action behavior | Output / format behavior |
|---|---:|---:|---|---|---|---|
| Format repair, `ut=0.05` | 0.00 | 0.00 | all 1 | all 8 | `10 / 316 / 0` | Bank-on clean options improved from 3/10 without repair to 6/10; EM changed from 0.10 to 0.00 |
| Threshold `ut=0.05` | 0.00 | 0.20 | all 1 | all 8 | `10 / 316 / 0` | 6 clean, 1 JSON leak, 3 other; recovered result |
| Threshold `ut=0.08` | 0.00 | 0.20 | all 8 | all 8 | `80 / 246 / 0` | 9 clean, 1 other; recovered result and best working threshold |
| Threshold `ut=0.10` | 0.00 | 0.00 | all 8 | all 8 | `80 / 246 / 0` | 4 clean, 6 other; recovered result |
| Threshold `ut=0.12` | 0.00 | 0.00 | all 8 | all 8 | `80 / 246 / 0` | 4 clean, 3 JSON leaks, 3 other; recovered result |
| Capacity `max_slots=8` | 0.00 | 0.10 | all 8 | all 8 | `80 / 60 / 186` | 8 clean, 1 JSON leak, 1 other |
| Capacity `max_slots=16` | 0.00 | 0.20 | all 16 | all 8 | `160 / 53 / 113` | 9 clean, 1 other; best observed balance |
| Capacity `max_slots=32` | 0.00 | 0.00 | 20-32, mean 25.2 | all 8 | `252 / 54 / 20` | 8 clean, 1 JSON leak, 1 other |
| Top-k `top_k=1` | 0.00 | 0.10 | all 16 | all 8 (1 slot) | `261 / 65 / 101` | 9 clean, 1 JSON leak; final aggregate run |
| Top-k `top_k=2` | 0.00 | 0.00 | all 16 | all 16 (2 slots) | `301 / 25 / 141` | 4 clean, 1 JSON leak, 5 other |
| Top-k `top_k=4` | 0.00 | 0.00 | all 16 | all 24 (3 slots) | `299 / 27 / 139` | 6 clean, 4 other; 4 empty Bank-on outputs |
| Top-k `top_k=8` | 0.00 | 0.00 | all 16 | all 24 (3 slots) | `300 / 26 / 140` | 4 clean, 2 JSON leaks, 4 other; 3 empty Bank-on outputs |

Core invariants for the canonical format-repair run, recovered threshold rows,
and clean capacity/top-k rows are:

- `query_write_count=0`
- `query_write_attempt_count=0`
- `cross_context_leakage_detected=false`
- `retrieved_latents_enter_weaver=true`
- `raw_retrieved_latents_enter_reasoner=false`

For the threshold sweep, the last two fields were recovered from the configured
Version B route because the original per-context postprocessor failed before
writing those booleans. The slot, retrieval, output, query-write, and leakage
traces themselves are present in the raw diagnostics.

## 6. Mechanism Analysis

### Memory Formation

The threshold diagnostic shows that `update_threshold` controls whether the
bank collapses into one rolling overwrite slot or forms multiple slots. At
`ut=0.05`, all contexts ended with one slot and writes were dominated by 316
matched replacements. At `ut=0.08`, all contexts reached eight slots, insert
count rose from 10 to 80, and matched replacements fell to 246. Values 0.10 and
0.12 produced the same eight-slot write pattern but lower EM, making 0.08 the
most promising working setting rather than merely the first non-collapsed one.

### Storage Capacity

The capacity diagnostic shows that `max_slots` is wired correctly. Final slot
counts moved from exactly 8, to exactly 16, to a 20-32 range as capacity grew.
Capacity evictions fell from 186 to 113 to 20. More storage was not always
better: cap16 reached `2/10`, while cap32 reached `0/10` despite retaining more
memory. Under top_k=1, cap32 enlarged the candidate pool without increasing
the final-query evidence budget, consistent with increased slot-selection
noise. This is a mechanism interpretation, not a proven causal decomposition.

### Retrieval Utilization

The top-k diagnostic shows that `top_k` is partially wired: top_k=2 retrieves
two slots and 16 latent tokens. Top_k=4 and top_k=8 both retrieve only three
slots and 24 latent tokens. This follows the implemented `threshold_topk`
ordering: scores below `retrieve_threshold` are removed first, then the
remaining list is truncated to top-k. The sweep is therefore a
threshold-plus-top-k interaction diagnostic, not a force-top-k diagnostic.

### Weaver Utilization

Increasing the final-query latent budget from 8 to 16 or 24 did not improve
EM. Relative to top_k=1, every multi-slot setting had more format failures, and
top_k=4/8 produced empty outputs. The present evidence supports the narrower
claim that Weaver does not reliably use additional retrieved latents under
`retrieve_threshold=0.03`; it does not show that Weaver cannot use multi-latent
memory under a cleaner retrieval intervention.

### Output Format

Format repair improved surface-form controllability but did not solve answer
correctness. Relative to the no-repair MAB-6B run, Bank-on clean-option outputs
rose from 3/10 to 6/10, while exact match moved from 1/10 to 0/10. Later cap16
produced 9/10 clean outputs but only 2/10 exact match. The remaining bottleneck
is therefore not only output formatting; memory compression/content, retrieval
selection, and latent utilization by Weaver remain plausible limiting stages.
The bank changes all ten outputs, but that steering does not reliably move
predictions toward the correct answer.

## 7. Failure Modes

- **Single-slot collapse:** low `update_threshold=0.05` treats nearly every new
  memory as a matched thread and repeatedly overwrites one slot.
- **Capacity/retrieval noise:** cap32 retains more slots and evicts less, but a
  top-1 final query must select from a larger candidate pool; EM falls to zero.
- **Threshold-limited top-k:** top_k=4/8 realize only three slots because only
  three candidates survive `retrieve_threshold=0.03`.
- **Language drift, JSON leakage, and empty output:** format repair reduces but
  does not eliminate JSON/template leakage. Multi-slot top-k settings add
  incomplete/empty outputs; top_k=4 has four empty Bank-on outputs and top_k=8
  has three.
- **Option-letter partial correctness versus strict EM:** diagnostic option
  parsing sometimes finds a correct letter in an output that strict EM rejects.
  For example, the capacity analysis reports cap16 option-letter alignment of
  3/8 parsable outputs versus strict EM of 2/10. This is useful error analysis,
  but it must not replace the official metric.
- **Artifact postprocessing failures:** the threshold sweep is recoverable but
  not cleanly canonical because all per-setting manifests were invalidated by
  a postprocessing KeyError. Capacity write totals in the root aggregate also
  require raw-trace recomputation.

## 8. Current Best Configuration

Current preferred diagnostic configuration:

- `retrieve_threshold=0.03`
- `update_threshold=0.08`
- `max_slots=16`
- `top_k=1`

This choice combines multi-slot formation, lower eviction pressure than cap8,
and the best observed capacity-run exact match. It is not a promoted default or
a final performance result. The sample has only 10 contexts, and identical
top_k=1 settings produced 1/10 in the final top-k aggregate and 2/10 in both
the capacity run and an earlier valid top-k attempt. The observed gain is
therefore unstable at this scale.

## 9. Conclusion Strength

### Strong within the current mechanism diagnostic

- `update_threshold=0.08` breaks the observed single-slot collapse: all ten
  recovered traces move from one slot at 0.05 to eight slots at 0.08.
- `max_slots` is wired and changes effective capacity, insert count, and
  capacity-eviction count.
- `retrieve_threshold` filters realized top-k under `threshold_topk`; both code
  order and the 3-slot/24-token top_k=4/8 traces support this conclusion.

The threshold conclusion is strong for slot mechanics but retains the noted
postprocessing-provenance caveat.

### Moderate

- cap16 is better than cap8/cap32 under the current top_k=1 n10 diagnostic.
- Multi-slot retrieval is associated with worse output control relative to
  top_k=1 under the current threshold and prompt; the effect is not monotonic
  between top_k=2, 4, and 8.

### Weak or not yet supported

- A final benchmark improvement from MAB-6B-FR.
- The claim that force-top-k is harmful; force-top-k was not tested.
- The claim that Weaver cannot use memory at all; outputs consistently change,
  and some runs contain one or two exact-match improvements.

## 10. Next Recommended Experiment

Run a retrieval-threshold relaxation / force-top-k diagnostic only after the
threshold postprocessing and aggregate bookkeeping paths are cleanly validated.

Fixed settings:

- `max_slots=16`
- `update_threshold=0.08`
- `top_k=4`

Sweep:

- `retrieve_threshold={0.03,0.02,0.01,0.00}`
- optionally add an explicit no-threshold / force-top-k mode if already
  supported without changing the shared benchmark contract

Same-run control:

- `max_slots=16`
- `update_threshold=0.08`
- `top_k=1`
- `retrieve_threshold=0.03`

Because top_k=1 has shown `1/10` versus `2/10` run-to-run variation, this
same-run control is needed to distinguish effects of threshold relaxation from
normal run variance.

Success check:

- `query_turn_retrieved_latent_count` should reach 32 in every context for
  top_k=4.

Scientific goal:

- distinguish a threshold bottleneck from a Weaver multi-latent utilization
  bottleneck before making further claims about top-k.
