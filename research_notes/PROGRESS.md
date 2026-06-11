# Project Progress

## Current State

- Current Phase: Phase 1 - Code Map and Inference Pipeline Audit
- Status: `completed`
- Last updated: 2026-06-11
- Stop condition: Reached; Phase 1 closeout only.
- Next suggested Phase: Phase 2 - Original Project Smoke Test

## Research Goal

Add an optional session-level Retrieval-Augmented Recurrent Latent Memory Bank at
inference time without changing MemGen's training workflows.

## Phase 0 Outcome

Phase 0 remains complete under the current roadmap.

## Phase 1 Outcome

Phase 1 is complete.

- [x] Audited inference entry file and main dispatch.
- [x] Audited config loading and runtime config handoff.
- [x] Audited static and dynamic session/sample/episode boundaries.
- [x] Located Trigger call sites.
- [x] Located Weaver call sites.
- [x] Located latent memory generation and Reasoner injection sites.
- [x] Located generation outputs and evaluation hooks.
- [x] Marked protected Weaver / Trigger training boundaries.
- [x] Assessed candidate LatentMemoryBank integration points and risks.
- [x] Updated research notes only. No core code or training workflow changed.

## Files Audited in Phase 1

- `main.py`
- `common/config.py`
- `memgen/runner.py`
- `interactions/base_interaction.py`
- `interactions/singleturn_interaction.py`
- `interactions/multiturn_interaction.py`
- `memgen/model/modeling_memgen.py`
- `memgen/model/modeling_utils.py`
- `memgen/model/weaver.py`
- `memgen/model/trigger.py`
- `memgen/utils.py`
- `data/__init__.py`
- `data/base_builder.py`
- `data/base_env.py`

## Repository Snapshot at Phase 1 Closeout

- Branch: `rlm-memory-bank`
- Commit: `7a13d0abb8bdfcb851421d164a9a8223af22a55f`
- Working tree had uncommitted changes before Phase 1 note updates: `no`
- Phase 1 modified research notes only: `yes`

## Key Phase 1 Conclusions

- Inference evaluation enters through `main.py -> MemGenRunner.evaluate()`.
- Static and dynamic evaluations use different interaction managers, but both
  funnel generation through `MemGenModel.generate()`.
- Trigger gating, Weaver latent generation, and latent-to-Reasoner injection all
  happen inside `MemGenModel.generate()` on the inference path.
- The safest future memory reset boundary is the interaction-manager session, not
  a global model lifetime.
- A future memory-bank design should use explicit inference-only state passing,
  not persistent global memory on `MemGenModel`.
- Baseline trust is still blocked by `BUG-0001`; this does not block code audit,
  but it does block scientific baseline claims.

## Phase History

| Date | Phase | Outcome | Evidence |
|---|---|---|---|
| 2026-06-11 | Phase 0 - Research Memory System and Repository Snapshot | Completed | `research_notes/` structure confirmed; repository and environment snapshot recorded |
| 2026-06-11 | Phase 1 - Code Map and Inference Pipeline Audit | Completed | `research_notes/CODE_MAP.md` updated with verified inference path, boundaries, tensor notes, and integration risks |

## Session Handoff

- Phase 1 can now be treated as complete.
- Do not enter smoke test, baseline, or implementation work without explicit
  approval for the next Phase.
- Recommended next step: Phase 2 - Original Project Smoke Test.
