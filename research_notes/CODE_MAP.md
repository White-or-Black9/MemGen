# Code Map

This file is intentionally unpopulated until an approved repository audit. Add
only verified paths and symbols.

## Repository Entry Points

| Area | Path | Symbol/Command | Role | Verified |
|---|---|---|---|---|
| Inference CLI/API | TBD | TBD | TBD | No |
| Weaver training | TBD | TBD | Protected boundary | No |
| Trigger training | TBD | TBD | Protected boundary | No |
| Configuration | TBD | TBD | Runtime settings | No |
| Evaluation | TBD | TBD | Metrics and outputs | No |

## Inference Data Flow

```text
input/session
  -> TBD
  -> latent representation
  -> optional memory retrieval/update
  -> TBD
  -> generated output
```

## State and Lifecycle

- Session identifier:
- Sample boundary:
- Batch boundary:
- Latent tensor shape/dtype/device:
- Existing cache/state:
- Reset point:
- Error handling:

## Protected Training Boundaries

### Weaver

- Entry points:
- Configuration:
- Checkpoints:
- Files that inference work must not modify:

### Trigger

- Entry points:
- Configuration:
- Checkpoints:
- Files that inference work must not modify:

## Candidate Inference Integration Points

| Candidate | Advantages | Risks | Evidence | Decision |
|---|---|---|---|---|
| TBD | TBD | TBD | TBD | Pending audit |

## Verification Commands

```bash
# Add verified commands during Phase 0.
```
