# Project Progress

## Current State

- Current activity: Research plan revision
- Status: `completed`
- Last updated: 2026-06-11
- Stop condition: Reached; no implementation or experiment Phase started.
- Required user decision: Review and confirm the revised 11-Phase roadmap.

## Research Goal

Add an optional session-level Retrieval-Augmented Recurrent Latent Memory Bank at
inference time without changing MemGen's training workflows.

## Latest Planning Update

- [x] Replaced the previous five-stage roadmap with Phases 0 through 10.
- [x] Separated repository snapshot, code audit, smoke test, and accepted baseline.
- [x] Separated the module skeleton, Version A integration, disabled equivalence,
  stability/debug, ablations, Version B, and paper consolidation.
- [x] Added the mandatory experiment logging standard.
- [x] Added standardized output directories for baseline, Version A, Version B,
  and ablation artifacts.
- [x] Refined the memory-locality constraint so it applies until a later Phase
  explicitly approves broader sharing or larger batch experiments.
- [x] Tightened Phase 3, 4, 5, and 8 plan text for golden-case control, skeleton
  isolation, explicit latent detachment/device-dtype handling, and minimum
  ablation coverage.
- [x] Modified documentation only; no core code or training workflow changed.
- [x] Ran no new experiment and did not attempt to repair `BUG-0001`.

This planning update does not retroactively declare any newly numbered Phase
complete. Existing audit evidence remains available and will be reconciled when
the corresponding Phase is explicitly approved.

## Evidence From Before the Roadmap Revision

The following work was performed under the previous roadmap. It is retained as
evidence but is not automatically counted as completion of any newly numbered
Phase:

- [x] Mapped CLI, configuration, inference, latent, session, and output paths.
- [x] Identified protected Weaver and Trigger training boundaries.
- [x] Selected the official Qwen2.5-1.5B GSM8K Weaver-SFT comparator.
- [x] Downloaded and hash-verified the required official checkpoint assets.
- [x] Defined the metric contract and disabled-feature compatibility contract.
- [x] Verified CLI parsing, config override parsing, and Python compilation.

The prior baseline acceptance attempt remains unresolved:

- [ ] Official LoRA adapter tensors load correctly.
- [ ] Full GSM8K baseline completes.
- [ ] Deterministic golden outputs are archived.
- [ ] Baseline is accepted as `comparison_ready`.

## Blocking Evidence

- `EXP-20260611-001` failed before generation.
- `BUG-0001` shows that `MemGenModel.from_pretrained()` re-wraps an already
  LoRA-wrapped base and skips all trained adapter keys.
- Continuing despite the warning would produce an invalid scientific comparator.

## Protected Boundaries Confirmed

- No core or training code was modified during the prior audit.
- Weaver training files and commands remain unchanged.
- Trigger training files and commands remain unchanged.
- No Phase in the revised roadmap has been started.

## Current Board

- Current mainline: establish a trusted original MemGen comparator.
- Incumbent comparator: `memgen-gsm8k-sft-official-v1`.
- Latest decisive result: official adapter loading is broken.
- Active blocker: `BUG-0001`.
- Stale route to ignore: treating a warning-producing run as a valid baseline.
- Next decision scope: narrow inference checkpoint-loader repair and regression
  verification only.
- Compute budget class: small; one GPU smoke followed by one full GSM8K baseline.

## Phase History

| Date | Phase | Outcome | Evidence |
|---|---|---|---|
| 2026-06-11 | Memory system initialization | Completed | `research_notes/` templates |
| 2026-06-11 | Audit and baseline attempt under previous roadmap | Blocked at baseline acceptance | `CODE_MAP.md`, `BASELINE.md`, `BUG-0001`, `EXP-20260611-001` |
| 2026-06-11 | Research plan revision | Completed | Revised `PLANS.md`; documentation only |

## Session Handoff

- Review the revised Phases 0 through 10 in `PLANS.md`.
- Existing `BUG-0001`, baseline notes, and audit evidence remain valid inputs to
  future planning.
- Do not start, repair, implement, or run any revised Phase until the user
  explicitly confirms the next Phase.
