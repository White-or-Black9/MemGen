# Research TODO

## Workflow Rules

- Keep tasks grouped by Phase.
- Do not move a task into active work without Phase approval.
- Mark blocked tasks with the blocker and required resolution.
- After a Phase, reconcile this file with `PROGRESS.md`.

## Backlog

### Phase 0: Audit and Baseline

- [x] Identify inference entry points and call graph.
- [x] Identify Weaver and Trigger training boundaries that must remain untouched.
- [x] Locate runtime configuration and default handling.
- [x] Define session/sample lifecycle.
- [x] Locate latent representation creation and consumption.
- [x] Locate generation outputs, logging, and evaluation hooks.
- [x] Select and hash-verify the official baseline checkpoint.
- [x] Record the canonical baseline command and metric contract.
- [x] Define exact disabled-feature compatibility tests.
- [x] Update `CODE_MAP.md` and `BASELINE.md`.
- [ ] Repair `BUG-0001` in a separately approved Phase.
- [ ] Run and accept the full baseline after loader repair.

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

- None. Phase 0 is paused at the baseline gate.

## Blocked

- Trusted baseline execution is blocked by `BUG-0001`.

## Done

- [x] Initialize long-term research notes and prompt templates.
