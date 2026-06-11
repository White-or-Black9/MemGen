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
- Implementing Version A or Version B in Phase 4.

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

- No inference integration.
- No Version A or Version B behavior.
- No multi-sample batch support.
- No persistence contract for experiment checkpoints.
- No learned query, key, aggregation, or update function.
- No cross-session or cross-sample memory.
- No GPU integration test; conversion behavior is unit-tested on CPU.

## End-of-Day Isolation Status

Validated on 2026-06-11:

- The standalone skeleton passes all 16 unit tests.
- `latent_memory_bank.enabled` remains `false` in its separate default template.
- `MemGenModel.generate()`, runner, interaction managers, and model exports do
  not reference or construct `LatentMemoryBank`.
- Existing GSM8K configuration and all Weaver/Trigger training paths remain
  unchanged.
- Phase 5 integration has not started.

## Open Questions

- Which latent representation is the most stable retrieval key/value?
- Should Version A retrieve before prompt augmentation, inference augmentation,
  or both?
- How should retrieved slots be aggregated before Reasoner injection?
- Which retrieval/update rule gives useful gains at acceptable overhead?
- Which slot score should drive `replace` after real inference queries exist?
