# Research Plan

## Objective

Maintain MemGen as a long-term research project and investigate an inference-only,
optional session-level Retrieval-Augmented Recurrent Latent Memory Bank.

## Non-Negotiable Constraints

1. Do not modify the Weaver training workflow.
2. Do not modify the Trigger training workflow.
3. When `latent_memory_bank.enabled=false`, behavior must remain exactly unchanged.
4. Until explicitly approved in a later phase, memory must remain session-local
   and must not be shared across samples.
5. Until explicitly approved in a later phase, memory-bank experiments default
   to `batch_size=1`.
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

### Phase 0: Research Memory System and Repository Snapshot

- Establish the durable `research_notes/` project memory system.
- Record the research objective, constraints, workflow, and initial repository state.
- Capture the branch, commit, working-tree status, environment, and available assets.
- Do not modify core code or run substantive experiments.

### Phase 1: Code Map and Inference Pipeline Audit

- Map inference entry points, configuration flow, session/sample boundaries,
  latent representations, generation outputs, and evaluation hooks.
- Identify Weaver and Trigger training boundaries that must remain unchanged.
- Identify candidate inference-only integration points and their risks.
- Update `CODE_MAP.md` using verified paths and symbols.

### Phase 2: Original Project Smoke Test

- Verify the documented environment, dependencies, model loading, dataset loading,
  and one minimal original-project inference path.
- Use the smallest representative sample count and `batch_size=1`.
- Record all warnings, failures, environment deviations, and output artifacts.
- Do not treat a smoke test as an accepted scientific baseline.

### Phase 3: Original MemGen Baseline

- Establish a trusted, reproducible original MemGen comparator.
- Fix or route around baseline blockers only within an explicitly approved scope.
- Record deterministic golden cases, task metrics, latency, and memory usage.
- Golden cases must fix random seed, decoding parameters, sample IDs, model
  checkpoint, and evaluation script.
- Freeze the disabled-feature compatibility oracle for later phases.

### Phase 4: LatentMemoryBank Module Skeleton

- Add the standalone session-level memory-bank data model and configuration schema.
- Keep `latent_memory_bank.enabled=false` as the default.
- Implement lifecycle, validation, reset, capacity, and isolation scaffolding
  without integrating it into original inference behavior.
- No production inference code path should call the memory bank in this phase.
- Add focused unit tests for the module skeleton.

### Phase 5: Version A Integration — Reasoner Injection Only

- Integrate the optional bank into inference only.
- Retrieve stored latent memories and inject them into the Reasoner path.
- Do not feed retrieved memory into Weaver inputs in Version A.
- Keep memory local to one session/sample and default to `batch_size=1`.
- All stored latent memories must be detached from the computation graph, and
  device/dtype conversions must be explicit.
- Leave Weaver and Trigger training workflows unchanged.

### Phase 6: Disabled-Feature Equivalence Test

- Compare the implementation with `latent_memory_bank.enabled=false` against the
  frozen Phase 3 golden cases.
- Require exact generated token IDs, augmentation masks, metrics, output schema,
  and relevant tensor/control-flow invariants.
- Treat any difference as a blocking regression.

### Phase 7: Version A Stability and Debug Experiment

- Run bounded Version A experiments before performance claims.
- Test session reset, no cross-sample leakage, empty memory, capacity limits,
  dtype/device consistency, deterministic replay, and long-session behavior.
- Measure latency and memory overhead.
- Repair only Version A defects within this Phase.

### Phase 8: Core Ablation Experiments

- Evaluate retrieval on/off, write/update on/off, capacity, top-k, eviction,
  aggregation, and recurrent update choices.
- Compare every variant against the frozen original baseline and Version A.
- Keep datasets, checkpoints, seeds, decoding, and sample counts controlled.
- Minimum required ablations:
  `original MemGen`, `latest-k retrieval`, `random retrieval`,
  `cosine retrieval`, `cosine retrieval without recency decay`,
  `cosine retrieval with recency decay`, `append-only update`,
  `replace update`.
- Record negative results and failure cases.

### Phase 9: Version B Integration — Weaver Input Retrieval

- Extend retrieval so selected memory can condition Weaver input generation.
- Preserve Version A as a separately selectable comparator.
- Keep all changes inference-only and session-local.
- Re-run disabled-feature equivalence and targeted stability checks.

### Phase 10: Paper-Level Evidence Consolidation

- Consolidate baseline, Version A, Version B, ablations, efficiency, robustness,
  and failure analysis.
- Trace every claim, table, and figure to experiment IDs and raw artifacts.
- Freeze method definitions, limitations, reproducibility instructions, and
  paper-facing evidence.
- Do not promote unsupported hypotheses to claims.

## Experiment Logging Standard

Every experiment, including failed, aborted, smoke, debug, and ablation runs,
must be appended to `research_notes/EXPERIMENTS.md` with:

- date and experiment ID
- git branch and commit hash
- command
- config file
- model path
- checkpoint path
- dataset path
- sample count
- random seed
- decoding parameters
- output directory
- prediction file
- metric file
- latency
- memory usage if available
- notes and failures

Commands and paths must be recorded exactly enough to reproduce the run. Missing
values must be written explicitly as `not available` with a reason rather than
silently omitted.

## Output Directory Standard

Experiment artifacts must be grouped by method family:

```text
outputs/
├── baseline/
├── latent_bank_vA/
├── latent_bank_vB/
└── ablations/
```

- `outputs/baseline/`: original-project smoke and accepted MemGen baseline artifacts.
- `outputs/latent_bank_vA/`: Version A stability, debug, and main-run artifacts.
- `outputs/latent_bank_vB/`: Version B stability and main-run artifacts.
- `outputs/ablations/`: controlled ablation artifacts.

Each run should use a unique experiment-ID subdirectory and must not overwrite a
previous run.

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
