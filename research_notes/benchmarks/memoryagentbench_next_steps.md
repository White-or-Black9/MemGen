# MemoryAgentBench Next Steps

## Current Decision

MAB-6B is completed. The canonical exploratory run is
`20260625T122323Z-detectiveqa-version-b-weaver-space-bank-n10`. It preserved
the MAB-5C / MAB-5D safety constraints, preserved MAB-6A reproducibility, and
changed the storage/query path:

- `retrieve_threshold=0.03`
- `update_threshold=0.05`
- `max_slots=8`
- `top_k=1`
- `retrieved_memory_to_weaver=True`
- `memory_bank_storage_space=weaver`

The canonical run improved exact match from `0.0` to `0.1`, changed outputs in
all 10 contexts, kept query-time retrieval active in all 10 contexts, preserved
`query_write_count=0`, preserved context isolation, routed retrieval and bank
storage fully into Weaver space, and prevented raw retrieved memory from
entering Reasoner directly. This is exploratory positive evidence, not yet
enough to replace Version A as the default.

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

1. **Retrieval-threshold relaxation / force-top-k diagnostic:** hold
   `max_slots=16`, `update_threshold=0.08`, and `top_k=4`, then sweep
   `retrieve_threshold={0.03,0.02,0.01,0.00}`. Include an explicit no-threshold
   mode only if the diagnostic contract already supports it.
2. **Optional prompt-length audit:** isolate why the `132726 > 131072` warning
   is emitted during over-capacity full-history estimation, while keeping
   `full_history_status=over_capacity_invalid` and never scoring full-history
   generation.
3. **Optional future capacity / thread analysis:** if the threshold-relaxation
   diagnostic still caps realized retrieval below four slots, investigate
   whether the remaining bottleneck is thresholding or Weaver multi-latent
   utilization.

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
4. If follow-up is approved, start with retrieval-threshold relaxation /
   force-top-k plus the same-run `max_slots=16`, `update_threshold=0.08`,
   `top_k=1`, `retrieve_threshold=0.03` control rather than another MAB-6B
   replication pass.
