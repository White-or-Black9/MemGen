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

### Phase 8A: GSM8K Version A-simple Short Single-Turn Pilot

- Status: `completed`
- Purpose:
  - sanity-check the conservative Reasoner-only mechanism
  - record stable execution and negative pilot evidence
- Result:
  - disabled G0 scored `0.60` (`12/20`)
  - enabled G1/G4/G6/G7 scored `0.50` (`10/20`)
  - all enabled variants were stable
- Interpretation:
  - this is not main evidence for the final method
  - GSM8K is short and single-turn, so it does not test the primary
    multi-turn, long-trajectory, or context-truncation hypothesis
  - G1/G4 compare current write-age decay against no decay, not
    last-retrieved-turn decay against no decay

### Phase 8B: Method / Implementation Alignment

- Status: `completed`
- Completed alignment work:
  - documented Version A-simple, Version A-aligned, and Version B separately
  - recorded that current decay is write-age decay, not
    last-retrieved-turn decay
  - recorded that current `threshold_topk` has no fallback top-1
  - implemented structured retrieval context
  - implemented and mechanism-tested Version A-aligned
    `update_policy=thread_update`
- Remaining method variants, not yet implemented:
  - `last_retrieved_decay`
  - `threshold_topk_with_fallback_top1`
- Version B remains out of scope.

### Phase 8C: Target Task Transition

- Status: `proposed`
- Immediate execution order:
  - complete notes cleanup and read-only review
  - prepare and approve a commit for completed Version A-aligned work
  - plan the TriviaQA baseline protocol
  - run a minimal TriviaQA Original MemGen / disabled-memory smoke
- Move primary evaluation away from GSM8K.
- Use TriviaQA as the next candidate because the current repository implements
  it as a dynamic multi-turn search/answer environment with `max_turns=5`,
  growing interaction history, and observation truncation.
- Establish a trusted Original MemGen / disabled-memory baseline on TriviaQA.
- Validate model, checkpoint, dataset, retrieval backend, output schema, reward,
  latency, and session/turn traces before enabled-memory comparisons.

### Phase 8D: TriviaQA Version A-Aligned Smoke

- Status: `proposed`
- Run the current Reasoner-only Version A-aligned `thread_update` path on
  TriviaQA after the disabled baseline smoke is stable.
- Verify that one session-local bank persists across turns and resets across
  episodes.
- Check whether memories written in early turns are retrieved and used in later
  turns.
- Record context growth, observation truncation, retrieval/write events,
  latency, memory, and failures.
- Make no performance claim until disabled and enabled runs are stable and
  reproducible.

### Phase 8E: Method-Aligned Version A Variants

- Status: `proposed`
- Already completed:
  - matched-slot replacement / `thread_update`
- Consider only after the TriviaQA disabled baseline and Version A-aligned
  smoke are stable:
  - last-retrieved-turn decay
  - fallback top-1 for a non-empty bank
- Preserve Version A-simple as a separately selectable comparator.
- Re-run disabled equivalence and targeted multi-turn stability checks for every
  semantic change.

### Phase 8F: TriviaQA Targeted Ablations

- Status: `proposed`
- Compare:
  - disabled Original MemGen
  - Version A-simple
  - Version A with last-retrieved decay
  - Version A with fallback top-1
  - Version A with matched-slot update
  - threshold sweeps
  - top-k sweeps
- Focus analysis on multi-turn, long-trajectory, and context-truncation
  behavior.
- Do not use GSM8K as the primary evidence for these hypotheses.

### Phase 9: Version B Implementation

- Status: `proposed`
- Begin only after Version A variants on TriviaQA provide sufficient evidence.
- Implement the full `retrieve -> Weaver revise/generate -> matched write-back`
  method.
- Feed retrieved memory into Weaver with current context.
- Include fallback top-1, last-retrieved-turn decay, and matched-slot/thread
  update according to the frozen Version B specification.
- Preserve Version A-simple and method-aligned Version A variants as explicit
  comparators.
- Test Weaver-input distribution risk, disabled-path equivalence, multi-turn
  stability, and Version A versus Version B.

### Phase 10: Paper-Level Consolidation

- Status: `proposed`
- Consolidate target-task main results, controlled ablations, efficiency,
  memory behavior, failure analysis, and limitations.
- Trace every claim, table, and figure to experiment IDs and raw artifacts.
- Freeze method definitions, reproducibility instructions, and paper-facing
  evidence.
- Do not promote GSM8K pilot observations or unsupported hypotheses to final
  claims.

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
