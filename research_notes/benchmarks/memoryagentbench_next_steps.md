# MemoryAgentBench Next Steps

## Current Decision

Keep MAB-5C pending as the next mechanism experiment:

**MAB-5C: Decoupled Retrieval-Update Thresholds**

The completed MAB-5B diagnostic now shows that a simple raised shared threshold
can push slot counts to the maximum while keeping retrieval active in every
context. That makes the simple baseline stronger, but it does not replace the
need for a later decoupled-threshold check if you still want to isolate read
versus write behavior.
The first clean MAB-5C should start with:

- `retrieve_threshold=0.03`
- `update_threshold=0.05`
- `max_slots=8`
- `top_k=1`

## MAB-5B Status

MAB-5B has completed. The run used `threshold=0.05`, kept query turns
read-only, and remained `over_capacity_invalid` for full-history detective_qa.
It produced the strongest simple-baseline behavior so far: slot counts reached
`8` in every context, retrieval stayed active in every context, and official
exact match remained `0.0` in both modes.

## Why MAB-5C Is Still Optional

MAB-5A showed active retrieval and output changes but no exact-match gain. With
`threshold=0.03`, final slot counts remained `[1, 2, 2, 5, 6, 5, 6, 7, 4, 7]`
after 25-50 chunks per context. MAB-5B raised the shared threshold to `0.05`
and increased slot counts to `[8, 8, 8, 8, 8, 8, 8, 8, 8, 8]` while keeping
retrieval active, so MAB-5C is now a refinement question rather than an urgent
rescue step.

The current single threshold controls two distinct decisions:

1. whether retrieved memory is visible to Reasoner;
2. whether `write_back()` replaces the matched slot or inserts a new thread.

Separating those gates directly tests the over-merge diagnosis while retaining
the low read threshold that kept retrieval active.

## Phase 1 Contract

Planned settings:

- `retrieve_threshold=0.03`
- `update_threshold=0.05`
- `max_slots=16`
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

Compare future MAB-5C against the fixed MAB-5A run:

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

Proceed only after MAB-5C is implemented, tested, run, and interpreted:

1. **MAB-5D:** MAB-5C plus optional `top1_if_empty` retrieval fallback.
   Fallback remains off by default and must not alter update-threshold passage.
2. **MAB-6A / Version B:** exploratory retrieved-memory-to-Weaver conditioning.
   This must remain isolated from Version A and disabled by default.

## Stop Conditions

- Stop if old shared-threshold tests change behavior.
- Stop if retrieved memory enters Weaver during MAB-5C.
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
5. Run MAB-5C only after compatibility and routing checks pass.
