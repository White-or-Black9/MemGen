# Research TODO

## Workflow Rules

- Keep tasks grouped by Phase.
- Do not move a task into active work without Phase approval.
- Mark blocked tasks with the blocker and required resolution.
- After a Phase, reconcile this file with `PROGRESS.md`.

## Backlog

### Phase 0: Audit and Baseline

- [ ] Identify inference entry points and call graph.
- [ ] Identify Weaver and Trigger training boundaries that must remain untouched.
- [ ] Locate runtime configuration and default handling.
- [ ] Define session/sample lifecycle.
- [ ] Locate latent representation creation and consumption.
- [ ] Locate generation outputs, logging, and evaluation hooks.
- [ ] Record baseline commands and metrics.
- [ ] Define exact disabled-feature compatibility tests.
- [ ] Update `CODE_MAP.md` and `BASELINE.md`.

### Phase 1: Session-Local Prototype

- [ ] Define memory bank interface and lifecycle.
- [ ] Define retrieval, recurrent update, capacity, and eviction defaults.
- [ ] Add opt-in configuration with disabled default.
- [ ] Implement inference-only integration.
- [ ] Enforce per-session isolation and `batch_size=1` default.
- [ ] Test disabled-path equivalence.
- [ ] Test reset and no cross-sample leakage.

### Later Phases

- [ ] Design retrieval and update ablations.
- [ ] Measure quality, latency, and memory overhead.
- [ ] Run robustness and failure analyses.
- [ ] Consolidate paper-ready evidence.

## Active

- None. Awaiting approval for the next Phase.

## Blocked

- None.

## Done

- [x] Initialize long-term research notes and prompt templates.
