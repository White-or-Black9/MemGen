# Method Specification

## Working Title

Session-Level Retrieval-Augmented Recurrent Latent Memory Bank for MemGen Inference

## Research Question

Can a session-local latent memory bank help MemGen explicitly preserve,
retrieve, and update early useful latent memories for later reuse in multi-turn,
long-trajectory, or context-truncated inference, without retraining Weaver or
Trigger?

## Scope

- Inference-time modification only.
- Optional feature, disabled by default.
- One bank instance per session.
- No cross-sample sharing until explicitly approved.
- Memory-bank experiments default to `batch_size=1`.

## Non-Goals

- Modifying Weaver or Trigger training.
- Training a globally shared memory.
- Historical Phase 4 scope excluded production inference integration.
- Historical Phase 5 scope excluded Version B.

## Conceptual Pipeline

```text
current input/state
  -> latent query
  -> retrieve session-local latent memories
  -> aggregate retrieved context
  -> recurrent latent update/integration
  -> normal MemGen inference continuation
  -> optional write to the same session memory
```

## Phase 4 Skeleton

Implementation: `memgen/model/latent_memory_bank.py`

At Phase 4 closeout, the implementation was standalone and was not imported by
`memgen/model/__init__.py`, `MemGenModel`, `generate()`, interaction managers,
the runner, trainers, or training scripts. This is a historical isolation
statement; Phase 5 later connected the bank to inference.

### Public Types

- `LatentMemoryBankConfig`: validated immutable configuration.
- `LatentMemorySlot`: one detached latent tensor, pooled key, metadata,
  lifecycle counters, score, and original tensor device/dtype.
- `LatentMemoryBank`: one session-owned container with no global registry.

### Public Interface

- `reset()` / `clear()`: remove all slots and reset the local write step.
- `len(bank)`: return the current slot count.
- `build_query(hidden_states)`: mean-pool the most recent `pool_last_n` tokens.
- `build_key(memory)`: mean-pool all tokens in one memory tensor.
- `retrieve(query_or_hidden_states, device=..., dtype=...)`: score, filter, and
  return detached cloned slot copies on an explicit output device/dtype. The
  tensors and metadata are independent of bank-owned state, so caller mutation
  cannot modify stored slots.
- `retrieve_with_context(...)`: return cloned slots plus full-bank scores,
  current argmax information, selected indices, and the bank step.
- `write(memory, metadata=None)`: validate, detach, clone, optionally move to
  CPU, and store or replace one slot.
- `write_back(memory, retrieval_result, metadata=None)`: apply the
  Version A-aligned `thread_update` policy using the current retrieval context.
- `debug_summary()`: JSON-like configuration and slot metadata summary.
- `state_dict()`: detached debug snapshot, not a trainer checkpoint format.

### Tensor Contract

- Hidden states and memory accept `[tokens, hidden]` or
  `[1, tokens, hidden]`.
- Enabled memory-bank operation currently rejects batch dimensions greater than
  one; Phase 4 established this tensor contract.
- A pre-pooled retrieval query may use `[hidden]`.
- Tensors must be floating point and have no empty dimensions.
- Hidden sizes must match between query and stored keys.
- `write()` always calls `detach().clone()`.
- Retrieval returns detached cloned tensors and deep-copied metadata rather than
  references to bank-owned slots.
- Original device and dtype are recorded.
- `storage_device=cpu` explicitly moves stored values and keys to CPU.
- Retrieval explicitly converts values and keys to requested device/dtype.

### Current Version A-simple Retrieval Skeleton

```text
query = mean(hidden_states[-pool_last_n:])
key_i = mean(memory_i)
age_i = current_memory_write_step - slot_i.created_step
score_i = cosine(query, key_i) * exp(-decay_alpha * age_i)
```

`_step` counts successful memory writes. It is not a generation-token counter,
dialogue-turn counter, or retrieval-call counter. Current decay therefore
measures slot age in successful memory-write steps:

```text
delta_t_i = current_memory_write_step - slot_i.created_step
```

This is write-age decay. It is not the intended Version B definition of turns
since the slot was last retrieved. Although retrieval updates
`last_access_step`, current scoring does not use that field.

Policies:

- `topk`: keep the highest `top_k` scores.
- `threshold`: keep every score at or above `threshold`.
- `threshold_topk`: threshold first, then keep at most `top_k`.

Current `threshold_topk` returns an empty result when no slot reaches the
threshold. It does not fall back to the single best slot. Consequently, Phase
8A groups G1 and G4 compare current write-age decay against no decay; they do
not compare last-retrieved decay against no decay.

#### Structured Retrieval Context

Step 2 adds `LatentMemoryRetrievalResult` and
`retrieve_with_context(...)` without changing retrieval or write semantics.
The result records:

- detached cloned `slots`
- full-bank `scores` in original slot-index order
- pre-filter `max_score` and `argmax_index`
- `threshold_passed`
- post-filter `retrieved_indices` and `retrieved_scores`
- the current memory-write `bank_step`

Threshold and top-k filtering only determine the returned slots and retrieved
indices. Even when threshold filtering returns no slots, the full scores,
maximum score, and argmax index remain available for a future write-back
decision. Equal scores choose the lowest original slot index. The legacy
`retrieve(...)` API remains a wrapper that returns only
`retrieve_with_context(...).slots`.

At Step 2 closeout, this context was preparatory plumbing only and `write()`
did not consume it. Step 3 later added `write_back(...)` and `thread_update`
without changing the legacy `write()` policies. Fallback top-1 and
last-retrieved decay remain unimplemented.

### Update Skeleton

- Below capacity, every enabled write appends.
- `append`: reject a new write when full.
- `replace`: replace the slot with the lowest `last_score`. An unscored slot is
  lower priority than a scored slot; if every slot has `last_score=None`, replace
  the oldest slot by `created_step`.
- `replace_oldest`: replace the earliest-created slot.

### Version A-Aligned Thread-Aware Write-Back

Step 3 adds `update_policy=thread_update` and
`write_back(memory, retrieval_result, metadata=None)`. This policy consumes the
structured result from the current query and applies:

```text
if M is empty:
    insert m_t
elif max_score >= threshold:
    replace slot[argmax_index] with m_t
elif len(M) < max_slots:
    insert m_t as a new thread
else:
    evict the oldest slot and insert m_t as a new thread
```

Matched replacement occurs even when unused capacity remains. New-thread
insertion and capacity eviction are distinct from matched replacement, and the
decision never uses slot `last_score`. `retrieval_result.bank_step` must match
the current bank step so that a write-back cannot consume stale slot indices.

In `MemGenModel.generate()`, only the `thread_update` policy uses
`retrieve_with_context(...)` followed by `write_back(...)`. Retrieved slots
remain Reasoner-only supports; Weaver still receives only the original current
context and produces a reasoner-space latent for write-back.

This is a Version A-aligned write-back variant, not Version B. It does not add
fallback top-1, last-retrieved decay, or retrieved-memory input to Weaver.
Legacy `append`, `replace`, and `replace_oldest` behavior remains on
`retrieve(...)` plus `write(...)`.

### Disabled Behavior

When `enabled=false`:

- `write()` returns `False` and stores nothing.
- `retrieve()` returns an empty list.
- At Phase 4 closeout no production code constructed the class. In the current
  integration, disabled sessions still do not construct a bank, so original
  MemGen inference has no additional memory retrieval or write operations.

## Phase 5 Version A Integration

Implementation:

- `interactions/base_interaction.py`
- `interactions/singleturn_interaction.py`
- `interactions/multiturn_interaction.py`
- `memgen/runner.py`
- `memgen/model/modeling_memgen.py`

### Runtime Wiring

- `run.latent_memory_bank` is an optional runner config subtree.
- Existing baseline configs remain valid because the subtree may be absent.
- Each interaction-manager `run_agent_loop()` creates one local bank only when
  `enabled=true`.
- The bank is passed explicitly into `MemGenModel.generate(...)` as
  `latent_memory_bank=...`.
- `MemGenModel` does not keep any persistent bank field.

### Session Lifecycle

- Single-turn static evaluation: one bank per `run_agent_loop()` call.
- Multi-turn dynamic evaluation: one bank shared across all turns in one
  `run_agent_loop()` episode.
- A new session or episode creates a fresh bank; memory cannot leak across
  calls.

## Version A: Conservative Reasoner-Only Memory Injection

Purpose: provide a low-risk mechanism and stability test while preserving the
original Weaver input distribution.

Let `M` be the session-local memory bank and `s_i` the configured score for
slot `m_i`.

```text
if M is empty:
    R_t = empty
else:
    compute s_i for every slot
    if max_i(s_i) >= tau:
        R_t = {m_i | s_i >= tau}, bounded by the configured top_k policy
    else:
        R_t = empty
```

Version A has no fallback top-1. If `R_t` is empty, inference falls back to the
original MemGen augmentation behavior and injects only the newly generated
latent `m_t`.

When Trigger decides to augment:

1. Build the retrieval query from the current Reasoner-side candidate inputs.
2. Retrieve prior reasoner-space latent memories from the session-local bank.
3. Keep Weaver input equal to the original current context `H_t`; retrieved
   memory never enters Weaver or `reasoner_to_weaver()`.
4. Project Weaver outputs back to Reasoner space as `latent_inputs_embeds`.
5. If `R_t` is non-empty, inject
   `[retrieved_reasoner_latents; new_reasoner_latents]` into Reasoner.
6. If `R_t` is empty, inject only the new reasoner-space latent `m_t`.
7. Whenever Trigger fires and Weaver produces `m_t`, write the new
   reasoner-space `latent_inputs_embeds` back to the session-local bank,
   regardless of whether retrieval returned any slot.

Retrieved memory is never passed into Weaver and never participates in
`reasoner_to_weaver()`.

The implementation and experiments from Phase 5 through Phase 8A belong to
this conservative Version A-simple definition. Steps 2 through 4 subsequently
added the Version A-aligned `thread_update` write-back variant while retaining
Reasoner-only injection, write-age decay, and no fallback top-1.

## Version B: Full Retrieval-Augmented Recurrent Latent Update

Purpose: implement the full proposed `retrieve -> revise/generate -> write-back`
method.

Let `M` be the session-local bank and let:

```text
delta_t_i = number of dialogue turns since slot i was last retrieved
s_i = cosine(q_t, key_i) * exp(-alpha * delta_t_i)
```

Retrieval is defined as:

```text
if M is empty:
    R_t = empty
else if max_i(s_i) >= tau:
    R_t = {m_i | s_i >= tau}
else:
    R_t = {m_argmax}
```

Version B therefore includes fallback top-1 whenever the bank is non-empty but
no slot reaches the threshold. Retrieved slots update an explicit
`last_retrieved_turn` or `last_retrieved_step`, and later decay is computed from
that retrieval event rather than slot creation.

When Trigger fires:

1. Build query `q_t` from the current context.
2. Retrieve `R_t` using the method-aligned last-retrieved-turn score.
3. Feed retrieved memory together with current context into Weaver, using
   `[R_t; H_t]` or an equivalent explicit concatenation.
4. Let Weaver revise or integrate retrieved supports with current context to
   generate a new latent `m_t`.
5. Continue Reasoner inference with the newly generated `m_t`; raw retrieved
   memory need not also be injected into Reasoner.
6. Apply matched write-back:
   - if `M` is empty, insert `m_t`
   - if `max_i(s_i) < tau`, insert `m_t` as a new thread/topic
   - otherwise replace the argmax-matched slot with `m_t` to update the current
     thread

Version B is not implemented. No Phase 5 through Phase 8A result is evidence
for Version B.

### Phase 5 Debug Bookkeeping

The bank debug summary now records:

- `memory_write_count`
- `memory_retrieve_count`
- `retrieved_latent_count`
- `new_latent_count`
- `slot_count`
- `append_count`
- `replace_count`
- `rejected_write_count`
- `last_update_action`
- `update_action_trace`

These counters stay separate so retrieved latents and new Weaver-produced
latents are not conflated.

For Phase 7 capacity-trigger supplements, the debug harness also accepts
bounded CLI-only overrides for:

- `max_slots`
- `top_k`
- `threshold`
- `decay_alpha`
- `update_policy`
- `retrieve_policy`

These are debug-only runtime overrides inside
`scripts/eval/phase5_memory_bank_debug.py`. They do not modify
`configs/latent_memory/gsm8k.yaml` or the frozen baseline configuration.

## Phase 7 Stability Criteria

Phase 7 treats enabled Version A as stable only if all of the following hold in
bounded debug runs:

- every run completes without crash, NaN, OOM, CUDA error, shape mismatch,
  device mismatch, or dtype mismatch
- every single-turn session starts from `initial_slots=0`
- no cross-sample leakage appears across repeated single-turn sessions
- stored slot tensors remain reasoner-space latents with hidden size `1536`
- stored slot metadata preserves explicit storage/original device and dtype
- `slot_count` never exceeds `max_slots`
- `weaver_input_token_counts` matches
  `reasoner_to_weaver_input_token_counts`, which is the Phase 7 trace used to
  confirm that retrieved memory does not enter Weaver

Phase 7 records latency and peak CUDA memory as overhead/debug context only. It
does not treat enabled-path reward or accuracy as a performance claim.

### Disabled-Path Contract

When `latent_memory_bank` is absent or `enabled=false`:

- interaction managers do not construct a bank
- `MemGenModel.generate()` receives `latent_memory_bank=None`
- the original latent-injection branch remains intact
- no new retrieval, write, attention-mask, list-wrapping, or tensor-padding
  code runs on the disabled path

This branch passed exact golden replay against `EXP-20260611-007` in
`EXP-20260612-010`.

## Phase 4 Configuration

```yaml
latent_memory_bank:
  enabled: false
  batch_size: 1
  max_slots: 8
  top_k: 1
  threshold: 0.7
  decay_alpha: 0.05
  pool_last_n: 64
  retrieve_policy: threshold_topk
  update_policy: replace_oldest
  storage_device: cpu
  debug: true
```

Template: `configs/latent_memory_bank/default.yaml`.

The template is separate from existing `configs/latent_memory/*.yaml` files and
is not merged into current runtime configuration in Phase 4.

## Invariants

- Disabled mode has no numerical or stateful effect on original MemGen.
- Memory cannot cross session or sample boundaries.
- Batch size defaults to 1.
- Stored memories do not retain computation graphs.
- Device and dtype conversion is explicit.
- Training code and behavior remain unchanged.
- Every method claim must trace to a recorded experiment.

## Current Limitations

- Current Version A-simple uses write-age decay, not last-retrieved-turn decay.
- Current Version A-simple has no fallback top-1.
- The original Version A-simple policies have no matched-slot thread update;
  the optional Version A-aligned `thread_update` variant now provides it.
- Current Version A-simple is not the full proposed Version B.
- No multi-sample batch support.
- No persistence contract for experiment checkpoints.
- No learned query, key, aggregation, or update function.
- No cross-session or cross-sample memory.
- No enabled-path performance claim; Phase 5 only establishes mechanism and
  compatibility.

## Current Validation Status

Validated on 2026-06-12:

- The current suite passes 47/47 unit and integration tests.
- `latent_memory_bank.enabled=false` exactly reproduces the accepted Phase 3
  golden response-token and augmentation-mask hashes on the full 20-sample
  Phase 6 check, and again on samples `0..2` after Step 3.
- Existing GSM8K configuration remains unchanged and valid when the optional
  config subtree is absent.
- All Weaver/Trigger training paths remain unchanged.
- Version A-simple and Version A-aligned `thread_update` are present.
- Current retrieval still uses write-age decay and no fallback top-1.
- Version B has not started.

## Open Questions

- Which latent representation is the most stable retrieval key/value?
- How should Version B combine retrieved memory with current Weaver input?
- How should retrieved slots be aggregated before Reasoner injection?
- Should turn-aware decay use dialogue turns, retrieval calls, or another
  explicit event clock?
- How should matched-slot thread update behave when multiple slots exceed the
  threshold?
- Which target-task context-truncation regime best exposes useful early-memory
  reuse?

## Step 4 Thread-Update Mechanism Validation

`EXP-20260612-024` validates the Version A-aligned `thread_update` policy on one
real GSM8K inference session. The trace observed one `empty_bank` insertion
followed by three `matched_thread` replacements. Weaver input token counts
remained identical to reasoner-to-Weaver input token counts, while stored
latents remained `[8, 1536]` reasoner-space tensors.

The real smoke did not observe low-score new-thread insertion or capacity
eviction. Those branches remain covered by deterministic unit tests. This
mixed evidence is sufficient for mechanism validation but not for any
performance claim. Retrieval still uses write-age decay, has no fallback
top-1, and does not send retrieved memory into Weaver.

## Phase 8A Pilot Ablation Protocol

Validated on 2026-06-12:

- Purpose:
  - run a short single-turn sanity and negative pilot for Version A-simple
  - confirm that the current retrieval/recency/update variants run stably
  - record early negative directionality without treating GSM8K as the primary
    target task
- Non-goals:
  - no paper-level performance claim
  - no latest-k or random retrieval in this pilot
  - no Version B
- Fixed runtime contract:
  - dataset `gsm8k/main/test`
  - sample IDs `0..19`
  - `sample_count=20`
  - `seed=42`
  - `batch_size=1`
  - greedy decoding
  - `max_response_length=1024`
  - same model path
  - same checkpoint path
  - same base config file `configs/latent_memory/gsm8k.yaml`
- Compared groups:
  - `G0`: disabled anchor
  - `G1`: Version A anchor
  - `G4`: cosine retrieval without recency decay
  - `G6`: append-only update
  - `G7`: replace update
- Required reporting for enabled groups:
  - `compute_reward`
  - correct / total
  - prediction count and summary count
  - total latency
  - mean latency per sample
  - peak CUDA memory
  - Trigger/Weaver call counts
  - aggregated memory debug counts across sessions
  - per-session `initial_slots`
  - confirmation that retrieved memory remains Reasoner-only
  - confirmation that stored latents remain reasoner-space tensors
- Pilot interpretation rule:
  - negative results are valid outcomes and must be recorded directly
  - no 20-sample result may be written up as a final method-quality claim
  - GSM8K single-turn evaluation is not aligned with the primary multi-turn,
    long-trajectory, or context-truncation hypothesis
  - Phase 8A should not be expanded directly into the main experiment without
    changing to an aligned target task
