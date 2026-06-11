# Research Plan

## Objective

Maintain MemGen as a long-term research project and investigate an inference-only,
optional session-level Retrieval-Augmented Recurrent Latent Memory Bank.

## Non-Negotiable Constraints

1. Do not modify the Weaver training workflow.
2. Do not modify the Trigger training workflow.
3. When `latent_memory_bank.enabled=false`, behavior must remain exactly unchanged.
4. Phase 1 must not share memory across samples.
5. Phase 1 defaults to `batch_size=1`.
6. Update `PROGRESS.md` after every completed Phase.
7. Update `EXPERIMENTS.md` for every experiment.
8. Update `DECISIONS.md` for every important design choice.
9. Execute exactly one Phase at a time, then stop for user confirmation.

## Phase Template

### Phase N: <Title>

- Status: `proposed | approved | in_progress | completed | blocked`
- Goal:
- Scope:
- Explicitly out of scope:
- Preconditions:
- Files expected to change:
- Implementation steps:
- Verification:
- Required note updates:
- Exit criteria:
- Rollback plan:
- User approval:

## Initial Roadmap

### Phase 0: Repository Audit and Baseline Definition

- Map inference entry points without changing core code.
- Identify configuration, session boundaries, latent representations, and output paths.
- Define exact compatibility tests for the disabled feature.
- Record a reproducible baseline.

### Phase 1: Session-Local Memory Prototype

- Add an opt-in inference-only memory bank.
- Keep memory isolated per sample/session.
- Default to `batch_size=1`.
- Preserve exact original behavior when disabled.

### Phase 2: Retrieval and Recurrence Evaluation

- Compare retrieval rules, update policies, capacity, and eviction strategies.
- Measure quality, latency, memory use, and stability.

### Phase 3: Robustness and Ablations

- Run controlled ablations and failure analysis.
- Validate isolation, determinism, long-session behavior, and compatibility.

### Phase 4: Consolidation and Paper Evidence

- Freeze the method description.
- Consolidate tables, figures, limitations, and reproducibility materials.

## Phase Gate

Before execution:

- [ ] The Phase is explicitly approved by the user.
- [ ] Scope and exit criteria are written.
- [ ] Baseline and verification commands are known.

After execution:

- [ ] Verification passed or failures are documented.
- [ ] `PROGRESS.md` is updated.
- [ ] Experiments are recorded in `EXPERIMENTS.md`.
- [ ] Important choices are recorded in `DECISIONS.md`.
- [ ] Work is paused for user confirmation.
