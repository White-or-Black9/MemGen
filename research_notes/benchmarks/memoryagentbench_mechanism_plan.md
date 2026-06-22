# MAB Memory Mechanism Plan

## 1. Purpose

This document plans the next MemGen memory-mechanism changes after MAB-5A. It
does not implement them. The work is deliberately split into independently
testable phases so that each mechanism can be evaluated without changing the
meaning of previous results.

The governing rule is backward compatibility: every new mechanism must be
gated by configuration, and the default configuration must reproduce the
current Version A behavior.

## 2. Current Baseline

MAB-5A evaluated `Long_Range_Understanding / detective_qa` using compressed
Bank-off versus compressed Bank-on over all 10 local contexts, with the first
query only.

| Metric | Result |
| --- | ---: |
| Valid contexts | 10 |
| Compressed Bank-off exact match | 0.0 |
| Compressed Bank-on exact match | 0.0 |
| Output changed | 10 |
| Retrieval-active contexts | 10 |
| Query writes | 0 |
| Cross-context leakage | 0 |
| Final slot counts | `[1, 2, 2, 5, 6, 5, 6, 7, 4, 7]` |

Original full-history detective_qa was not run. Every selected context exceeded
the 32,768-token capacity and was recorded as `over_capacity_invalid`.

The current shared-threshold `thread_update` path is:

1. Trigger fires.
2. Before Weaver generation, the model calls
   `retrieve_with_context(candidate_inputs_embeds.detach(), ...)`.
3. Retrieval compares a query pooled from `candidate_inputs_embeds` with each
   existing `slot.key`.
4. Retrieved memory is injected into Reasoner only and does not enter Weaver.
5. Weaver generates `latent_inputs_embeds = weaver_to_reasoner(weaver_hidden_states)`.
6. `write_back(latent_inputs_embeds.detach(), retrieval_result)` writes the new
   Weaver-generated reasoner-space latent.
7. The same `threshold` controls retrieval filtering and matched-thread
   replacement.

## 3. Current Mechanism Diagnosis

At `threshold=0.03`, retrieval remains active because successful decayed scores
are approximately `0.030-0.064`. The same low threshold also classifies many
writes as matched-thread updates, causing `write_back()` to replace an existing
slot instead of appending a new thread.

The resulting slot counts are low relative to the 25-50 chunks processed per
context. For example, context 0 performed 26 writes and ended with one slot,
while context 8 performed 51 writes and ended with four slots. This is evidence
of over-merge or over-compression, although it does not prove that increasing
slot count alone will improve answer quality.

The matching score does not compare the newly generated Weaver latent with old
memory. Comparing pooled `candidate_inputs_embeds` with Weaver-latent keys may
be a weak thread-matching signal because the query and stored key summarize
representations produced at different stages and with different semantic roles.
The content ultimately inserted or replaced is nevertheless the newly generated
Weaver latent, `latent_inputs_embeds`.

## 4. Design Constraints

- Do not change Weaver training.
- Do not change Trigger training.
- Disabled-bank and original inference paths must remain unchanged.
- Existing shared-threshold behavior must remain exactly reproducible.
- Every new mechanism must be disabled by default.
- Banks remain session-local and reset after every context.
- No cross-context state may survive reset.
- Query phases remain read-only unless an experiment explicitly enables query
  writes.
- Retrieved memory remains Reasoner-only unless Version B is explicitly enabled.
- Full-history detective_qa remains `over_capacity_invalid` and must not be run.
- Over-capacity prompts must not be silently truncated.
- MAB-5A remains the fixed reference baseline.
- Each mechanism receives a distinct runner, output root, manifest identity, and
  research note.

## 5. Mechanism Options

| Option | What it changes | Expected benefit | Main risk | Difficulty | Preserves Version A | Priority |
| --- | --- | --- | --- | --- | --- | ---: |
| A: Decoupled thresholds | Separates retrieval filtering from thread-update matching | Keeps useful reads while allowing more new-thread inserts | Additional slots may still be redundant or low quality | Low | Yes, by default | 1 |
| B: `top1_if_empty` fallback | Returns the argmax slot when threshold filtering returns nothing | Avoids empty reads in higher-threshold regimes | May inject low-relevance or harmful memory | Medium | Yes, when disabled | 2 |
| C: Retrieved memory into Weaver | Conditions Weaver on current context plus retrieved memory | Allows Weaver to fuse old memory into a revised latent | Weaver was not trained for this input distribution | High | Separate Version B path | 3 |

## 6. Recommended Phase Order

### Phase 1: MAB-5C Decoupled Retrieval-Update Thresholds

- Add independently configurable read and update thresholds.
- Preserve the legacy `threshold` as the fallback for both.
- Add unit and integration tests before changing runner behavior.
- Add a separate MAB-5C runner for detective_qa compressed n10.
- Compare MAB-5C directly against the fixed MAB-5A artifact.

### Phase 2: MAB-5D Decoupled Thresholds plus Retrieval Fallback

- Add an optional `top1_if_empty` policy.
- Keep fallback disabled by default.
- Evaluate fallback only after Phase 1 behavior is stable.
- Compare fallback and no-fallback under otherwise identical settings.

### Phase 3: MAB-6A Exploratory Version B Weaver-conditioned Memory

- Allow retrieved memory to condition Weaver only when explicitly enabled.
- Keep Reasoner-only Version A as the default.
- Use a separate runner and research note.
- Treat all initial Version B results as exploratory because Weaver was not
  trained for this changed input distribution.

## 7. Phase 1 Implementation Plan

### 7.1 Files and Boundaries

Expected files:

- Modify `memgen/model/latent_memory_bank.py` for configuration, threshold
  selection, write decisions, and diagnostics.
- Modify `tests/test_latent_memory_bank.py` for threshold-separation unit tests.
- Modify `tests/test_latent_memory_bank_integration.py` only for compatibility
  and Reasoner-only routing assertions.
- Create `scripts/eval/mab5c_decoupled_thresholds_detectiveqa_n10.py`.
- Create `tests/test_mab5c_decoupled_thresholds_detectiveqa_n10.py`.

Do not modify `memgen/model/modeling_memgen.py` during Phase 1 unless a focused,
failing test proves that the required behavior cannot be implemented through
the existing `LatentMemoryBank` object and runner configuration. A convenience,
diagnostic preference, or anticipated need is not sufficient justification.
If such a test fails, document the failure and isolate the smallest required
plumbing change before editing that file. The pre-existing unrelated
`trust_remote_code=True` diff must not be mixed into the mechanism change.

### 7.2 Configuration Interface

Extend `LatentMemoryBankConfig`:

```python
retrieve_threshold: Optional[float] = None
update_threshold: Optional[float] = None
```

Add bank helpers:

```python
def _effective_retrieve_threshold(self) -> float:
    return (
        self.config.threshold
        if self.config.retrieve_threshold is None
        else self.config.retrieve_threshold
    )

def _effective_update_threshold(self) -> float:
    return (
        self.config.threshold
        if self.config.update_threshold is None
        else self.config.update_threshold
    )
```

Validate each non-`None` value within `[-1.0, 1.0]`.

### 7.3 Retrieval Behavior

`retrieve_with_context()` must:

- use `_effective_retrieve_threshold()` for `threshold` and
  `threshold_topk` filtering;
- set `LatentMemoryRetrievalResult.threshold_passed` according to the effective
  retrieval threshold;
- document `threshold_passed` as a compatibility name meaning
  `retrieve_threshold_passed`;
- preserve `max_score` and `argmax_index` before filtering;
- preserve current top-k, decay, access-refresh, and empty-bank behavior.

### 7.4 Write Behavior

`write_back()` must:

- use `_effective_update_threshold()` for `replace_matched` versus new-thread
  insertion;
- never use `retrieval_result.threshold_passed` as the update decision;
- continue using `retrieval_result.max_score` and `argmax_index`;
- retain the existing `threshold_passed` debug field as a compatibility alias
  for retrieval-threshold passage;
- add `effective_retrieve_threshold`, `effective_update_threshold`,
  `retrieve_threshold_passed`, and `update_threshold_passed` to each write-back
  debug event.

### 7.5 Backward Compatibility

When both new fields are `None`:

- retrieval filtering uses `threshold`;
- write matching uses `threshold`;
- existing slot selection, insertion, replacement, eviction, and counters remain
  unchanged;
- serialized config gains two `null` fields, but behavioral output remains
  unchanged.

### 7.6 Tests

Add focused tests for:

1. Shared-threshold configuration with both new fields unset reproduces existing
   retrieval and replacement behavior.
2. A score below `retrieve_threshold` returns no slot.
3. A score above `retrieve_threshold` returns the slot.
4. `write_back()` replaces only when `max_score >= update_threshold`.
5. A score between `0.03` and `0.05` is retrieved but causes new-thread
   insertion rather than `replace_matched`.
6. Full-bank insertion below the update threshold retains last-retrieved-age
   eviction behavior.
7. Invalid optional thresholds raise `ValueError`.
8. Debug summaries record configured and effective thresholds.
9. Disabled mode remains a no-op.
10. Existing Reasoner-only integration assertions continue to pass.

### 7.7 MAB-5C Experiment Contract

Use:

- `retrieve_threshold=0.03`
- `update_threshold=0.05`
- `max_slots=16`
- `top_k=1`
- `retrieve_policy=threshold_topk`
- `update_policy=thread_update`
- the same checkpoint and 10 detective_qa contexts as MAB-5A
- first-query-only compressed Bank-off and Bank-on
- read-only query phase
- no full-history generation

The runner should reuse MAB-5A dataset loading, compressed-prompt validation,
read-only query proxy, scoring, bank lifecycle, and artifact helpers without
changing the MAB-5A script. It can copy `version_a_bank_config()` and override
the fields above; older runner defaults must not change.

Output root:

```text
outputs/mab/decoupled_thresholds_detectiveqa_n10/<timestamp>-detectiveqa-decoupled-thresholds-n10/
```

The MAB-5C manifest and aggregate result must include:

```text
compare_against_run_id = "20260621T013454Z-detectiveqa-compressed-n10"
slot_count_deltas
avg_slot_count_delta
matched_replace_delta
thread_insert_delta
retrieval_active_delta
```

`slot_count_deltas` is the ordered 10-context vector
`MAB-5C final slots - MAB-5A final slots`. Aggregate count deltas use
`MAB-5C - MAB-5A`. The runner must fail clearly if the reference artifact is
missing, has a different context order or IDs, or does not contain the required
comparison fields. It must not silently substitute another run.

Additional required diagnostics:

- effective thresholds;
- final slot count by context;
- `matched_replace_count`;
- `thread_insert_count`;
- `capacity_evict_count`;
- retrieved scores and indices by turn;
- retrieval-active contexts;
- write and retrieval counts;
- exact match and output changes;
- existing gold-substring or relaxed diagnostics, clearly labeled non-official;
- query write count;
- Reasoner/Weaver routing assertions;
- reset and cross-context leakage checks.

## 8. Phase 2 Implementation Plan

Add:

```python
RetrieveFallbackPolicy = Literal["none", "top1_if_empty"]
retrieve_fallback_policy: RetrieveFallbackPolicy = "none"
```

Fallback applies only when the bank is non-empty, the retrieval policy includes
threshold filtering, no slot passes the effective retrieval threshold, and the
fallback policy is `top1_if_empty`.

Fallback returns the precomputed argmax slot. It does not change `max_score`,
`argmax_index`, `retrieve_threshold_passed`, or `update_threshold_passed`.
Specifically, selecting a slot through fallback must not make
`update_threshold_passed` true. The update flag is derived only from
`max_score >= effective_update_threshold`, and `write_back()` must continue to
use that comparison independently of fallback selection.

Add result and debug fields:

```text
fallback_used
fallback_score
fallback_slot_index
fallback_count
```

`fallback_count` is a bank-level counter reset with the bank. Fallback-selected
slots count as retrieved and update their access metadata because they enter
inference.

Create a separate MAB-5D runner and output root. Compare MAB-5D against MAB-5C
with all other settings fixed.

## 9. Phase 3 Implementation Plan

Phase 3 is exploratory Version B work, not a production/default mechanism.

Add:

```python
retrieved_memory_to_weaver: bool = False
```

When disabled, Version A remains unchanged. When enabled and retrieval is
non-empty:

1. Retrieve reasoner-space memory as currently implemented.
2. Concatenate retrieved memory with `candidate_inputs_embeds` in reasoner space.
3. Pass the combined sequence through `reasoner_to_weaver`.
4. Run Weaver on that conditioned sequence.
5. Project the generated/fused latent through `weaver_to_reasoner`.
6. Inject the newly generated fused latent into Reasoner.
7. Do not also inject the raw retrieved memory into Reasoner in Version B.
8. Write the fused latent only when writes are enabled; the query read-only
   proxy must continue blocking query writes.

This phase is expected to require a focused change in
`memgen/model/modeling_memgen.py`. Before editing, preserve or separately
resolve its unrelated local diff so mechanism changes are reviewable in
isolation.

Add diagnostics:

```text
retrieved_memory_to_weaver
retrieved_latents_enter_weaver
weaver_conditioned_on_retrieved_memory
weaver_conditioning_token_count
fused_latent_generated
raw_retrieved_latents_enter_reasoner
query_write_count
```

Integration tests must prove that Version A input lengths and routing remain
unchanged when disabled, and that Weaver input length changes only when Version B
is enabled. The first Version B experiment must not also enable fallback.

## 10. Experiment Naming

| Name | Meaning |
| --- | --- |
| MAB-5A | Existing compressed-memory Version A baseline |
| MAB-5C | Decoupled retrieval/update thresholds |
| MAB-5D | MAB-5C plus `top1_if_empty` fallback |
| MAB-6A / Version B | Exploratory retrieved-memory-to-Weaver path |

Each runner must embed its mechanism identity, effective configuration, and
comparison baseline in `manifest.json` and `run_config.json`.

## 11. Success Criteria

### Phase 1

- Existing bank and integration tests pass.
- Shared-threshold configuration reproduces existing decisions.
- Retrieval remains active with `retrieve_threshold=0.03`.
- Final slot counts or thread insert counts increase relative to MAB-5A.
- MAB-5A comparison fields are complete and context-aligned.
- No cross-context leakage occurs.
- `query_write_count` remains zero.
- Retrieved memory remains outside Weaver.
- Exact match may remain zero; verified gate separation and interpretable memory
  structure are the initial mechanism criteria.

### Phase 2

- Fallback remains disabled by default.
- Every fallback is explicitly logged.
- Fallback selection does not alter `update_threshold_passed`.
- Update decisions remain governed only by `update_threshold`.
- Results compare fallback and no-fallback under otherwise identical settings.

### Phase 3

- The experiment is labeled exploratory in artifacts and notes.
- Retrieved memory reaches Weaver only when enabled.
- Raw retrieved memory is not accidentally double-injected into Reasoner.
- Version A input lengths and routing remain unchanged.
- Query-time writes remain blocked.
- Diagnostics prove Version A and Version B path separation.

## 12. Risks and Open Questions

- Scores are low and may be poorly calibrated across contexts.
- Thresholds operate on decayed scores, not raw cosine similarity.
- Comparing pooled `candidate_inputs_embeds` with Weaver-latent keys may be a
  weak thread-matching signal because the representations come from different
  stages and semantic roles.
- More slots may increase redundancy without improving memory quality.
- Direct replacement still discards prior slot content.
- Fallback may inject irrelevant memory.
- Weaver was not trained to consume retrieved latent memory as context.
- Version B concatenation order and token-length growth may affect Weaver
  behavior.
- Official exact match may remain zero because answer formatting is a separate
  issue.
- Relaxed metrics can diagnose formatting but must not replace official exact
  match.

## 13. Next Immediate Action

1. Review this plan and lock the Phase 1 interface and diagnostic names.
2. Implement Phase 1 only using test-first changes.
3. Do not implement fallback or Weaver conditioning during Phase 1.
4. Run bank unit tests and integration routing tests before model inference.
5. Run MAB-5C only after backward compatibility and routing checks pass.
6. Compare MAB-5C against the fixed MAB-5A run before deciding whether to begin
   Phase 2.
