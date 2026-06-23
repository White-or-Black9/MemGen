# MemoryAgentBench Next Steps

## Current Decision

MAB-5D is completed. The canonical result is the capacity16 run
`20260623T022140Z-detectiveqa-capacity16-n10`; the earlier
`20260623T015929Z-detectiveqa-decoupled-thresholds-n10` artifact is historical
only and must not be treated as MAB-5D. The capacity16 run confirmed the
intended split between retrieval visibility and write-back matching while
showing the effect of a larger bank:

- `retrieve_threshold=0.03`
- `update_threshold=0.05`
- `max_slots=16`
- `top_k=1`

The canonical run reached full slot capacity in every context, kept query-time
retrieval active in every context, stayed Reasoner-only, and preserved
`query_write_count=0`. It also reduced eviction churn relative to MAB-5C.

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

1. **MAB-6A / Version B:** exploratory retrieved-memory-to-Weaver conditioning.
   This must remain isolated from Version A and disabled by default.
2. **Optional future capacity follow-up:** only if later evidence requires it,
   revisit capacity or retrieval density separately; do not reintroduce
   fallback here.

## Stop Conditions

- Stop if old shared-threshold tests change behavior.
- Stop if retrieved memory enters Weaver during a future MAB-6A run.
- Stop if query writes occur.
- Stop if bank reset or context isolation fails.
- Stop if compressed prompts contain chunk text or acknowledgement history.
- Stop rather than run or truncate over-capacity full-history detective_qa.

## Immediate Handoff

1. Review `memoryagentbench_mechanism_plan.md`.
2. Implement only its Phase 1 test cases and bank configuration changes.
3. Do not edit `memgen/model/modeling_memgen.py` unless a focused failing test
   proves Phase 1 requires it.
4. Run unit and integration validation before any model inference.
5. Move next to MAB-6A only after compatibility and routing checks pass.
