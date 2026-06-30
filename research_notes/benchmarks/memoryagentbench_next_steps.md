# MemoryAgentBench Next Steps

## Current Decision

The accepted EventQA evidence now includes the 15-run frozen-context A/B/C
`top_k=1` sweep and the full end-to-end Config B `top_k=2` ablation on all five
EventQA-65536 contexts.

- Best setting: Config A
  `retrieve_threshold=0.03`, `update_threshold=0.05`, `max_slots=8`,
  `top_k=1`
- Config A overall:
  Bank-off `4/500 = 0.008`; Bank-on `114/500 = 0.228`;
  Bank-off recall `0.178`; Bank-on recall `0.266`;
  improved/regressed/unchanged `113/3/384`
- Config B (`0.03/0.09/16/1`) forced 16-slot construction but dropped Bank-on
  EM to `72/500 = 0.144`
- Config C (`0.005/0.09/16/1`) forced 15-16-slot construction but dropped
  Bank-on EM to `67/500 = 0.134`
- Query-time retrieval still returned exactly one slot in every setting, so
  multi-slot construction did not produce multi-slot use under `top_k=1`
- End-to-end Config B `top_k=2` (`0.03/0.09/16/2`) reached Bank-off `4/500`,
  Bank-on `109/500`, recall `0.290`, 131 format failures, and 98 Chinese
  outputs. It improved Config B `top_k=1` by 37 EM and approached Config A by
  five EM, but remained much less output-stable.
- This is a valid end-to-end ablation. Because `top_k` affects both bank
  construction and query retrieval, it is not the same frozen bank queried
  with one extra slot.
- Contexts 0-3 improved; context 4 collapsed to `1/100` with 94 format
  failures and 83 Chinese outputs. Retrieval remained fixed per context.
- A standalone ctx4 rerun under the same nominal Config B `top_k=2` settings
  did not reproduce that catastrophic outcome exactly:
  Bank-on EM `11/100`, recall `0.28`, format failures `52`, Chinese outputs
  `44`, retrieved pairs `{(3,0):84,(0,3):16}`. This supplements the accepted
  all-context ablation as a reproducibility/stability diagnostic; it does not
  replace it.
- Use the result as strong exploratory compressed frozen-context bridge
  evidence scored by the official EventQA substring-exact-match metric, not as
  a direct official long-context baseline comparison
- Keep Config A as the highest-EM and most output-stable accepted setting
- Defer `top_k=4`; first add score-decomposition and slot/chunk-provenance
  diagnostics, then rerun ctx4 only after those diagnostics are available

## MAB-5B Status

MAB-5B has completed. The run used `threshold=0.05`, kept query turns
read-only, and remained `over_capacity_invalid` for full-history detective_qa.
It produced the strongest simple-baseline behavior so far: slot counts reached
`8` in every context, retrieval stayed active in every context, and official
exact match remained `0.0` in both modes.

## MAB-5C Status

MAB-5C has completed on the same detective_qa n10 slice. The split-threshold
run kept exact match at `0.0` but achieved the intended mechanism shape:
slot counts stayed at `8` in every context, query-time retrieval stayed active
in every context, retrieved latents remained Reasoner-only, and query writes
remained `0`.

## MAB-5D Status

MAB-5D has completed on the same detective_qa n10 slice. The capacity16
ablation kept exact match at `0.0` but raised final slot counts to `16` in
every context, reduced capacity eviction relative to MAB-5C, kept query-time
retrieval active in every context, and kept retrieved latents Reasoner-only.

## Why the Split Still Matters

MAB-5A showed active retrieval and output changes but no exact-match gain. With
`threshold=0.03`, final slot counts remained `[1, 2, 2, 5, 6, 5, 6, 7, 4, 7]`
after 25-50 chunks per context. MAB-5B raised the shared threshold to `0.05`
and increased slot counts to `[8, 8, 8, 8, 8, 8, 8, 8, 8, 8]` while keeping
retrieval active. MAB-5C now confirms that the split-threshold configuration
can preserve that slot growth while keeping query-time retrieval active. The
canonical run remains dense overall, and the query-turn retrieved latent count
was `80` across 10 contexts.
MAB-5D shows that capacity alone can push the bank to 16 slots and lower
eviction churn, but still does not recover exact match.

The current single threshold controls two distinct decisions:

1. whether retrieved memory is visible to Reasoner;
2. whether `write_back()` replaces the matched slot or inserts a new thread.

Separating those gates directly tests the over-merge diagnosis while retaining
the low read threshold that kept retrieval active.

## Phase 1 Contract

Observed settings:

- `retrieve_threshold=0.03`
- `update_threshold=0.05`
- `max_slots=8`
- `top_k=1`
- `retrieve_policy=threshold_topk`
- `update_policy=thread_update`
- same checkpoint, 10 detective_qa contexts, and first-query-only protocol as
  MAB-5A

Compatibility requirements:

- when new thresholds are unset, both fall back to legacy `threshold`;
- old shared-threshold behavior remains reproducible by default;
- query writes remain disabled;
- bank state remains session-local and resets after every context;
- retrieved memory remains Reasoner-only;
- full-history detective_qa remains `over_capacity_invalid` and is not run.

The detailed test, interface, diagnostics, and artifact contract is in
`memoryagentbench_mechanism_plan.md`.

## Required Comparison

Compare MAB-5C against the fixed MAB-5A and MAB-5B runs:

```text
20260621T013454Z-detectiveqa-compressed-n10
```

At minimum report:

- official exact match and output changes;
- retrieval-active count and delta;
- per-context and average slot-count deltas;
- matched-replace delta;
- thread-insert delta;
- query writes and cross-context leakage;
- Reasoner-only versus Weaver routing checks.

An increase in slots or inserts is mechanism evidence, not an accuracy claim.
`output_changed` remains an activation diagnostic, not improvement.

## Later Phases

Proceed only if a later review still wants a follow-up:

1. **Preserve the accepted sweep as the anchor:**
   the EventQA reference includes the completed 15-run A/B/C sweep and the
   accepted end-to-end Config B `top_k=2` all-context ablation.
2. **Current stable setting:** keep Config A (`0.03/0.05/8/1`) as the
   highest-EM and cleanest-output setting.
3. **Current mechanism diagnostic:** add score decomposition, query-vector
   diagnostics, and final-slot chunk provenance, then rerun Config B
   `top_k=2` on context 4 only. The accepted standalone rerun
   `outputs/mab/eventqa_configB_ctx4_topk2_rerun/20260630T121127Z-eventqa-65536-version-b-weaver-space-bank-n5`
   shows the ctx4 failure mode is unstable, so the next rerun should happen
   only after the extra diagnostics are recorded.
4. **Keep configuration provenance explicit:** the accepted best EventQA result
   in this sweep is Config A, and its runtime config is
   `retrieve_threshold=0.03`, `update_threshold=0.05`, `top_k=1`,
   `max_slots=8`.
5. **Optional prompt-length audit:** isolate why the `132726 > 131072` warning
   is emitted during over-capacity full-history estimation, while keeping
   `full_history_status=over_capacity_invalid` and never scoring full-history
   generation.

## Stop Conditions

- Stop if old shared-threshold tests change behavior.
- Stop if raw retrieved memory still enters Reasoner during a future MAB-6A /
  Version B run.
- Stop if query writes occur.
- Stop if bank reset or context isolation fails.
- Stop if compressed prompts contain chunk text or acknowledgement history.
- Stop rather than run or truncate over-capacity full-history detective_qa.

## Immediate Handoff

1. Use `20260625T122323Z-detectiveqa-version-b-weaver-space-bank-n10` as the
   canonical MAB-6B artifact.
2. Keep Version A as the default path until MAB-6B is replicated.
3. Treat the MAB-6B exact-match gain as exploratory benchmark evidence, not as
   a default-path promotion.
4. The current active EventQA artifacts are the 15 accepted sweep runs under
   `outputs/mab/eventqa_frozen_context_bank_cfg{A,B,C}_ctx*/...` plus
   `outputs/mab/eventqa_configB_allctx_topk2/20260630T084500Z-eventqa-65536-version-b-weaver-space-bank-n5`.
5. Preserve `--context-index`, runtime config integrity validation,
   `--construction-only`, and the related EventQA regression tests as part of
   the runner contract.
6. Keep Config B/C `top_k=1` negative evidence scoped to `top_k=1`. Config B
   `top_k=2` is a strong positive end-to-end multi-slot signal, but it does not
   replace Config A and must retain the context-4 collapse caveat.
7. Defer `top_k=4` until context-4 score decomposition and provenance are
   available and the unstable ctx4 rerun behavior is explained.
8. Keep the bridge boundary explicit before any broader EventQA scaling or
   summary.
