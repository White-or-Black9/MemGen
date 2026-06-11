# Method Specification

## Working Title

Session-Level Retrieval-Augmented Recurrent Latent Memory Bank for MemGen Inference

## Research Question

Can an optional session-local latent memory bank improve long-context or recurrent
inference quality without retraining Weaver or Trigger and without changing the
original disabled-path behavior?

## Scope

- Inference-time modification only.
- Optional feature, disabled by default.
- Session-level memory lifecycle.
- Phase 1 uses no cross-sample sharing.
- Phase 1 defaults to `batch_size=1`.

## Non-Goals

- Modifying Weaver training.
- Modifying Trigger training.
- Training a globally shared memory in Phase 1.
- Introducing hidden behavior changes when the feature is disabled.

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

## Components

### Memory Item

- Representation:
- Metadata:
- Session ownership:
- Creation time/step:
- Device and dtype:

### Retrieval

- Query source:
- Similarity function:
- Top-k:
- Filtering:
- Normalization:

### Aggregation

- Rule:
- Weighting:
- Empty-memory behavior:

### Recurrent Update

- Update equation:
- Write condition:
- Detach/gradient policy:
- Capacity:
- Eviction:

### Lifecycle

- Initialization:
- Read timing:
- Write timing:
- Reset:
- Error cleanup:

## Configuration Draft

```yaml
latent_memory_bank:
  enabled: false
  scope: session
  capacity: null
  top_k: null
  retrieval: null
  update: null
```

This is a conceptual schema only. Final names and defaults require repository audit
and an accepted decision record.

## Invariants

- Disabled mode has no numerical or stateful effect.
- Memory cannot cross session/sample boundaries in Phase 1.
- Batch size defaults to 1 in Phase 1.
- Training code and training behavior remain unchanged.
- Every method claim must trace to a recorded experiment.

## Open Questions

- Which latent representation is the most stable retrieval key/value?
- Where is the safest inference-only integration point?
- How is a session boundary represented in current code?
- Which retrieval/update rule gives useful gains at acceptable overhead?
- What exact equality criterion is feasible for disabled-mode verification?
