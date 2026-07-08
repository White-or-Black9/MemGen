# DetectiveQA Current-P7 Design

## Goal

Add a current-version DetectiveQA evaluation path that uses the frozen paper
P7 mechanism instead of the historical Version B runner lineage, with two
execution stages:

- a bounded 1-context smoke for protocol validation
- a full n10 run for stress evidence

The output should be appendix/table-ready if the result is useful, but it must
not automatically modify `paper/` or change the current EventQA-first paper
line.

## Background and Problem

The repository already contains several DetectiveQA runners and artifacts, but
they belong to an older MAB-5*/MAB-6* lineage. In particular, the existing
`mab6b_weaver_space_bank_detectiveqa_n10.py` runner is explicitly a historical
Version B / retrieved-memory-to-Weaver path. That does not match the current
paper-facing P7 boundary defined in `research_notes/METHOD.md`.

The current paper-facing P7 boundary is:

- session-local latent memory bank
- Weaver-space bank storage
- fixed parameters
  - `retrieve_threshold=0.05`
  - `update_threshold=0.10`
  - `max_slots=16`
  - `top_k=2`
  - `decay_alpha=0.05`
- frozen Trigger / Weaver / Reasoner
- no cross-sample sharing
- query-time writes blocked where frozen/read-only evaluation applies

Therefore, DetectiveQA can only be used as current-version evidence if it is
rerun with a new runner that explicitly matches this P7 boundary.

## Scope

In scope:

- add a new current-P7 DetectiveQA runner
- keep the existing compressed-memory DetectiveQA task contract
- support three methods:
  - `disabled`
  - `p7`
  - `p7_no_query_retrieval`
- support two execution stages:
  - smoke: 1 context
  - full: n10 contexts
- emit durable appendix/table-ready artifacts and aggregate summaries
- update only research notes after validated runs

Out of scope:

- modifying any training code
- modifying Trigger / Weaver training paths
- changing the current paper main table automatically
- reinterpreting historical DetectiveQA artifacts as current-P7 evidence
- redesigning DetectiveQA into a multi-query frozen-bank benchmark

## Recommended Approach

Use the existing compressed-memory DetectiveQA runner
`scripts/eval/mab5a_detectiveqa_compressed_n10.py` as the execution base and
add a new current-P7 runner beside it.

Why this approach:

- it preserves the already validated DetectiveQA compressed contract
- it avoids mixing current-P7 evidence with historical Version B lineage
- it minimizes implementation risk
- it makes smoke and full runs easy to compare with prior DetectiveQA stress
  evidence while still being explicit that the mechanism has changed

Rejected alternatives:

1. Patch the old Version B DetectiveQA runner in place
   - rejected because it would blur lineage and make current-vs-historical
     interpretation fragile

2. Rebuild DetectiveQA on top of the FactConsolidation paired runner structure
   - rejected because DetectiveQA is first-query-only and already has a mature
     compressed contract

## Evaluation Contract

The new current-P7 DetectiveQA path must preserve the mature DetectiveQA task
contract from `mab5a`:

- split: `Long_Range_Understanding`
- sub-dataset: `detective_qa`
- compressed-memory protocol
- full-history remains `over_capacity_invalid`
- one final query per context (`first-query-only`)
- one session per context

This is not the same protocol as FactConsolidation. DetectiveQA is treated as a
stress benchmark, not a multi-query frozen-bank benchmark.

## Methods

The new runner must compare exactly three methods:

1. `disabled`
   - no bank created

2. `p7`
   - current frozen paper-facing P7 bank configuration
   - retrieval active at query time

3. `p7_no_query_retrieval`
   - same P7 construction as `p7`
   - query-time retrieval disabled
   - used to isolate whether retrieval matters under the same bank construction

## Current-P7 Bank Configuration

The runner must use the same frozen current-P7 config used in the recent
FactConsolidation work:

- `retrieve_threshold=0.05`
- `update_threshold=0.10`
- `max_slots=16`
- `top_k=2`
- `decay_alpha=0.05`
- `update_policy=thread_update`
- `storage_space=weaver`
- `query_phase=read_only`
- session-local bank
- `batch_size=1`

The manifest must record this config explicitly.

## Runner Design

### New runner

Create:

- `scripts/eval/detectiveqa_p7_n10.py`

Responsibilities:

- reuse DetectiveQA payload preparation and scoring from `mab5a`
- reuse current-P7 bank configuration logic from the recent paper-facing work
- run three methods under a single explicit contract
- emit per-context rows and per-run artifact summaries
- keep query-phase writes blocked for bank-enabled methods
- record enough mechanism fields to support appendix/table-ready summaries

### Reused components

Primary reuse:

- `scripts/eval/mab5a_detectiveqa_compressed_n10.py`
  - payload preparation
  - compressed task contract
  - scoring path
  - context-level diagnostic schema

Secondary reuse:

- `scripts/eval/factconsolidation_p7.py`
  - current-P7 config shape
  - paired method structure
  - frozen/read-only invariant expectations

Do not import or depend on the historical Version B DetectiveQA runner for the
mechanism definition.

## Output Artifacts

### Smoke output

- `outputs/mab/detectiveqa_p7_smoke/<run_id>/manifest.json`
- `outputs/mab/detectiveqa_p7_smoke/<run_id>/records.jsonl`
- `outputs/mab/detectiveqa_p7_smoke/<run_id>/artifact.json`

### Full output

- `outputs/mab/detectiveqa_p7_full/<run_id>/manifest.json`
- `outputs/mab/detectiveqa_p7_full/<run_id>/records.jsonl`
- `outputs/mab/detectiveqa_p7_full/<run_id>/artifact.json`

### Aggregate output

Create:

- `scripts/eval/detectiveqa_p7_aggregate.py`

Outputs:

- `outputs/mab/detectiveqa_p7_full_aggregate.json`
- `outputs/mab/detectiveqa_p7_full_aggregate.md`

The aggregate output should be appendix/table-ready by format, but not
automatically paper-promoted.

## Record Schema

Each per-context record should retain at least:

- `run_id`
- `context_index`
- `context_id`
- `query_id`
- `method`
- `gold_answer`
- `prediction`
- `exact_match`
- `output_changed`
- `improved`
- `regressed`
- `bank_created`
- `bank_write_count`
- `bank_retrieval_count`
- `bank_retrieved_latent_count`
- `retrieved_indices_by_turn`
- `retrieved_scores_by_turn`
- `query_write_count`
- `query_write_attempt_count`
- `bank_slot_count_final_before_reset`
- `bank_reset_after_context`
- `cross_context_leakage_detected`
- `latency_seconds`
- `peak_cuda_memory`

For bank-enabled methods, also record:

- pre-query bank state hash
- post-query bank state hash
- whether the query-phase snapshot changed

## Protocol Invariants

The new runner must explicitly validate:

For `disabled`:

- no bank created
- no retrieval activity

For `p7` and `p7_no_query_retrieval`:

- bank created successfully
- query write count is zero
- query write attempts are zero or explicitly blocked
- pre/post query bank snapshot unchanged
- post-context reset slot count is zero
- no cross-context leakage

For `p7_no_query_retrieval`:

- same construction behavior as `p7`
- query-time retrieval disabled

## Smoke Stage

Purpose:

- validate that current-P7 DetectiveQA can run under the preserved compressed
  contract
- verify protocol invariants
- verify no-query-retrieval isolation

Scope:

- 1 context
- all three methods

Exit criteria:

- all three methods produce valid records
- no invariant violation
- artifact schema is complete

The smoke stage is not paper evidence.

## Full n10 Stage

Purpose:

- produce current-version DetectiveQA stress evidence on the full compressed
  n10 slice
- allow appendix/table-ready aggregation if the result is useful

Scope:

- 10 contexts
- all three methods

Exit criteria:

- all valid contexts complete
- aggregate JSON/Markdown created
- mechanism fields are consistent

## Metrics and Interpretation

Primary task metric:

- `exact_match`

Supporting mechanism metrics:

- retrieval-active context count
- mean retrieved latent count
- query-write count
- final slot count
- reset success rate
- latency
- peak GPU memory

Interpretation policy:

- DetectiveQA is stress evidence first
- it may become appendix/table-ready if the result is clean and useful
- it must not automatically change the paper main line

## Research Notes Update Policy

After a validated smoke:

- update `research_notes/EXPERIMENTS.md`
- update `research_notes/PROGRESS.md`

After a validated full run:

- update `research_notes/EXPERIMENTS.md`
- update `research_notes/PROGRESS.md`
- update `research_notes/DECISIONS.md` only if a real routing decision changes

Do not update `paper/` automatically.

## Risks

1. DetectiveQA signal may remain sparse
   - mitigated by keeping it explicitly as stress evidence

2. Current-P7 may not improve over disabled
   - mitigated by designing the output as durable internal/appendix-ready
     evidence rather than assuming paper promotion

3. Mixing historical Version B expectations into current-P7 interpretation
   - mitigated by a brand-new runner and explicit manifest lineage

4. Overfitting the runner to FactConsolidation-style assumptions
   - mitigated by preserving the existing DetectiveQA first-query-only
     compressed contract

## File Plan

Create:

- `docs/superpowers/plans/2026-07-08-detectiveqa-current-p7-implementation.md`
- `scripts/eval/detectiveqa_p7_n10.py`
- `tests/test_detectiveqa_p7_n10.py`
- `scripts/eval/detectiveqa_p7_aggregate.py`
- `tests/test_detectiveqa_p7_aggregate.py`

Modify later only after runs:

- `research_notes/EXPERIMENTS.md`
- `research_notes/PROGRESS.md`
- `research_notes/DECISIONS.md` if routing changes

No planned modification:

- `paper/`
- training code
- Trigger / Weaver training paths

## Success Condition

This design succeeds if:

- a new DetectiveQA current-P7 runner exists
- smoke and full n10 are both runnable under the preserved compressed contract
- artifacts clearly distinguish `disabled`, `p7`, and `p7_no_query_retrieval`
- outputs are durable and appendix/table-ready
- no automatic paper change occurs
