# Project Progress

## Current State

- Current Phase: Phase 2 - Original Project Smoke Test
- Status: `completed`
- Last updated: 2026-06-11
- Stop condition: Reached; Phase 2 closeout only.
- Next suggested Phase: Do not enter Phase 3 baseline work until `BUG-0001` and `BUG-0002` are resolved in a separately approved repair Phase.

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

## Phase 2 Outcome

Phase 2 is complete with blocking caveats.

- [x] Checked environment, repository state, and runnable project environment.
- [x] Located an official minimal evaluation route based on GSM8K + Qwen2.5-1.5B-Instruct + Weaver-SFT.
- [x] Verified that the current `base` environment is not suitable for MemGen execution.
- [x] Verified that the `memgen` environment can initialize the project correctly.
- [x] Verified that local base-model and dataset caches are available offline.
- [x] Ran the original evaluation path on 1 GSM8K sample through `Config -> MemGenModel.from_config -> MemGenRunner.evaluate()`.
- [x] Confirmed that model loading, dataset loading, interaction setup, and generation start all work in the recommended environment.
- [x] Confirmed that the official static-eval path currently fails at result recording, leaving `answer.json` empty.
- [x] Confirmed separately that the original generation path can produce a completion when recorder logic is bypassed in a script-only harness.
- [x] Reconfirmed that official LoRA loading is still not trustworthy; no smoke result from this Phase is a valid scientific baseline.
- [x] Updated research notes only. No core code, Weaver training, or Trigger training logic changed.

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
- Commit: `7b8b9a44eb30325a676a6c9576c35b3a10b52c32`
- Working tree had uncommitted changes before Phase 2 note updates: `yes`
- Phase 2 modified research notes only: `yes`

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

## Key Phase 2 Conclusions

- Recommended runtime environment for this repository is
  `/home/baishilong/miniconda3/envs/memgen` with Python `3.10.20`.
- The current `base` environment is unsuitable for MemGen because it uses Python
  `3.13.9`, the project's `conda` wrapper is broken by CRLF, and sandboxed runs
  cannot expose CUDA to PyTorch.
- Offline execution requires bypassing the inherited `HF_ENDPOINT=https://hf-mirror.com`
  and local proxy variables, and using the local cached Qwen snapshot path.
- The original MemGen evaluation stack reached real generation on GPU for a
  1-sample GSM8K smoke run.
- The official static-eval route did not complete because
  `StaticEvalRecorder.record_batch()` crashed with `KeyError: 0`; the created
  `evaluate/answer.json` file remained empty.
- A script-only harness that kept the original model and interaction logic but
  bypassed the broken recorder produced a completion and wrote
  `manual_answer.json`.
- `BUG-0001` remains open: official LoRA/adapters still report missing trained
  keys under the current nested PEFT loading path, so no Phase 2 output may be
  treated as a valid baseline.

## Phase History

| Date | Phase | Outcome | Evidence |
|---|---|---|---|
| 2026-06-11 | Phase 0 - Research Memory System and Repository Snapshot | Completed | `research_notes/` structure confirmed; repository and environment snapshot recorded |
| 2026-06-11 | Phase 1 - Code Map and Inference Pipeline Audit | Completed | `research_notes/CODE_MAP.md` updated with verified inference path, boundaries, tensor notes, and integration risks |
| 2026-06-11 | Phase 2 - Original Project Smoke Test | Completed with caveats | Original evaluation path reached generation but failed in static recorder; script-only harness produced one completion; baseline remains invalid due `BUG-0001` |

## Session Handoff

- Phase 2 can now be treated as complete.
- Do not treat any Phase 2 artifact as a scientific baseline.
- Do not enter Phase 3 baseline work until checkpoint loading and static result
  recording are repaired and rechecked in a separately approved Phase.
- Recommended next step: approve a repair Phase for `BUG-0001` and `BUG-0002`,
  then rerun the original smoke test before Phase 3.
