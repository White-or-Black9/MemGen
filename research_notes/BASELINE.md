# Baseline Definition

## Purpose

Define the exact original MemGen inference behavior that all memory-bank variants
must compare against and that disabled mode must preserve.

## Baseline Identity

- Baseline ID:
- Date captured:
- Code revision:
- Working tree state:
- Checkpoint/model:
- Environment:
- Hardware:
- Dataset and split:
- Inference entry point:
- Configuration:
- Random seeds:
- Command:
- Output directory:

## Compatibility Contract

With `latent_memory_bank.enabled=false`:

- Control flow follows the original inference path.
- No memory bank state is created, retrieved, or updated.
- Inputs, outputs, tensor shapes, dtypes, and devices remain unchanged.
- Deterministic outputs match exactly where the original run is deterministic.
- Existing configuration files and commands remain valid.
- Training workflows and checkpoints remain untouched.

## Baseline Metrics

| Metric | Value | Measurement Method |
|---|---:|---|
| Primary task metric | TBD | TBD |
| Secondary metric | TBD | TBD |
| Latency per sample | TBD | TBD |
| Peak device memory | TBD | TBD |
| Throughput | TBD | TBD |

## Golden Cases

| Case ID | Input/Session | Expected Output/Hash | Purpose |
|---|---|---|---|
| TBD | TBD | TBD | Disabled-path equivalence |

## Reproduction

```bash
# Add the verified baseline command during Phase 0.
```

## Acceptance Criteria

- [ ] Baseline command runs from a clean documented environment.
- [ ] Outputs and metrics are archived.
- [ ] Golden cases are defined.
- [ ] Disabled-mode equivalence check is automated or precisely repeatable.
- [ ] Training workflows are confirmed unchanged.
