# Method Specification

## Working Title

Session-Level Retrieval-Augmented Recurrent Latent Memory Bank for MemGen Inference

## Research Question

Can an optional session-local latent memory bank improve recurrent inference
without retraining Weaver or Trigger and without changing disabled-path behavior?

## Scope

- Inference-time modification only.
- Optional feature, disabled by default.
- One bank instance per session.
- No cross-sample sharing until explicitly approved.
- Memory-bank experiments default to `batch_size=1`.

## Non-Goals

- Modifying Weaver or Trigger training.
- Training a globally shared memory.
- Connecting the bank to production inference in Phase 4.
- Implementing Version B in Phase 5.

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

The Phase 4 implementation is standalone. It is not imported by
`memgen/model/__init__.py`, `MemGenModel`, `generate()`, interaction managers,
the runner, trainers, or training scripts.

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
- `write(memory, metadata=None)`: validate, detach, clone, optionally move to
  CPU, and store or replace one slot.
- `debug_summary()`: JSON-like configuration and slot metadata summary.
- `state_dict()`: detached debug snapshot, not a trainer checkpoint format.

### Tensor Contract

- Hidden states and memory accept `[tokens, hidden]` or
  `[1, tokens, hidden]`.
- Batch dimensions greater than one are rejected in Phase 4.
- A pre-pooled retrieval query may use `[hidden]`.
- Tensors must be floating point and have no empty dimensions.
- Hidden sizes must match between query and stored keys.
- `write()` always calls `detach().clone()`.
- Retrieval returns detached cloned tensors and deep-copied metadata rather than
  references to bank-owned slots.
- Original device and dtype are recorded.
- `storage_device=cpu` explicitly moves stored values and keys to CPU.
- Retrieval explicitly converts values and keys to requested device/dtype.

### Retrieval Skeleton

```text
query = mean(hidden_states[-pool_last_n:])
key_i = mean(memory_i)
age_i = current_memory_write_step - slot_i.created_step
score_i = cosine(query, key_i) * exp(-decay_alpha * age_i)
```

`_step` counts successful memory writes. It is not a generation-token counter,
and recency decay in this skeleton therefore measures age in memory-write steps.

Policies:

- `topk`: keep the highest `top_k` scores.
- `threshold`: keep every score at or above `threshold`.
- `threshold_topk`: threshold first, then keep at most `top_k`.

### Update Skeleton

- Below capacity, every enabled write appends.
- `append`: reject a new write when full.
- `replace`: replace the slot with the lowest `last_score`. An unscored slot is
  lower priority than a scored slot; if every slot has `last_score=None`, replace
  the oldest slot by `created_step`.
- `replace_oldest`: replace the earliest-created slot.

### Disabled Behavior

When `enabled=false`:

- `write()` returns `False` and stores nothing.
- `retrieve()` returns an empty list.
- No production code constructs the class in Phase 4, so original MemGen
  inference has no additional state, calls, or numerical operations.

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

### Version A Injection Rule

When Trigger decides to augment:

1. Build the retrieval query from the current Reasoner-side candidate inputs.
2. Retrieve prior reasoner-space latent memories from the session-local bank.
3. Send only the original candidate inputs through `reasoner_to_weaver()` and
   Weaver augmentation.
4. Project Weaver outputs back to Reasoner space as `latent_inputs_embeds`.
5. Inject Reasoner tokens in this order:
   `[retrieved_reasoner_latents; new_reasoner_latents]`.
6. Write only the new reasoner-space `latent_inputs_embeds` into the bank.

Retrieved memory is never passed into Weaver and never participates in
`reasoner_to_weaver()`.

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

- No Version B behavior.
- No multi-sample batch support.
- No persistence contract for experiment checkpoints.
- No learned query, key, aggregation, or update function.
- No cross-session or cross-sample memory.
- No enabled-path performance claim; Phase 5 only establishes mechanism and
  compatibility.

## End-of-Day Isolation Status

Validated on 2026-06-12:

- The standalone skeleton plus Phase 5 integration passes 24 unit and
  integration tests.
- `latent_memory_bank.enabled=false` exactly reproduces the accepted Phase 3
  golden response-token and augmentation-mask hashes on samples `0..2`.
- Existing GSM8K configuration remains unchanged and valid when the optional
  config subtree is absent.
- All Weaver/Trigger training paths remain unchanged.
- Version A integration is present and Version B has not started.

## Open Questions

- Which latent representation is the most stable retrieval key/value?
- Should Version A retrieve before prompt augmentation, inference augmentation,
  or both?
- How should retrieved slots be aggregated before Reasoner injection?
- Which retrieval/update rule gives useful gains at acceptable overhead?
- Which slot score should drive `replace` after real inference queries exist?
