# Benchmark Notes Index

This directory contains the canonical MemoryAgentBench (MAB) state, operational
instructions, implementation plans, and historical evidence. When notes differ,
use the precedence below rather than treating every dated recommendation as
current.

## Canonical Current Notes

| Note | Role |
| --- | --- |
| `memoryagentbench_results.md` | Canonical MAB result summary and interpretation |
| `memoryagentbench_next_steps.md` | Canonical current action and experiment sequence |
| `memoryagentbench_runbook.md` | Operational commands, environments, artifacts, and safety rules |
| `memoryagentbench_mechanism_plan.md` | Design and implementation contract for MAB-5C, MAB-5D, and exploratory MAB-6A |
| `memoryagentbench_mab5a_detectiveqa_compressed_n10.md` | Detailed evidence for the fixed MAB-5A reference run |

Current state in one sentence: MAB-5A is a mechanism-active but zero-exact-match
compressed-memory baseline; the next action is Phase 1 only, MAB-5C Decoupled
Retrieval-Update Thresholds.

## Historical Experimental Evidence

These notes remain authoritative for the specific dated run or audit they
describe, but their recommendations are historical. Current routing comes from
`memoryagentbench_next_steps.md`.

| Note | Historical role |
| --- | --- |
| `memoryagentbench_no_api_smoke.md` | MAB-1A loader, chunking, prompt, and metric smoke evidence |
| `memoryagentbench_mab2_bank_off_run.md` | One-context full-history Bank-off evidence |
| `memoryagentbench_mab3_bank_on_full_history_run.md` | One-context full-history Bank-on evidence |
| `memoryagentbench_mab3a_threshold_ablation.md` | One-context shared-threshold activation evidence |
| `memoryagentbench_mab4a_compressed_memory.md` | One-context compressed-memory exploratory evidence |
| `memoryagentbench_paired_bank_off_vs_low_threshold_bank_on_n10.md` | Paired attempt limited to one locally matched context |
| `memoryagentbench_local_task_availability.md` | Local parquet availability and task-selection audit |
| `memgen_over_context_behavior.md` | Over-context source audit, synthetic probes, and detective_qa preflight |

The over-context finding remains active: original full-history MemGen on
`detective_qa` exceeds the 32,768-token capacity and must be recorded as
`over_capacity_invalid`, not executed or silently truncated.

## Superseded Planning and Design Context

These notes contain useful provenance and design reasoning, but their action
plans have been superseded by the canonical runbook and mechanism plan.

| Note | Superseded role |
| --- | --- |
| `memoryagentbench_feasibility_assessment.md` | Initial benchmark-fit assessment |
| `memoryagentbench_configuration_plan.md` | Initial environment, task, and adapter plan |
| `memoryagentbench_fork_and_smoke_plan.md` | Pre-smoke fork and environment plan |
| `memoryagentbench_adapter_strategy_review.md` | Full-history adapter taxonomy and early phase ordering |

## Interpretation Rules

- `output_changed` means the bank affected generation; it does not mean the
  answer improved.
- Zero official `exact_match` does not imply an inactive mechanism. MAB-5A had
  retrieval in all 10 contexts and changed all 10 outputs.
- Official exact match and relaxed diagnostics must remain separately labeled.
- Current Version A injects retrieved memory into Reasoner only, not Weaver.
- Current thread matching compares a query built from
  `candidate_inputs_embeds` with `slot.key`; it does not compare a newly
  generated Weaver latent with an old slot.
- The written memory remains Weaver-generated reasoner-space
  `latent_inputs_embeds`.
- Do not run another threshold-only ablation. The next mechanism phase is
  MAB-5C; fallback and Weaver conditioning remain later, separately gated work.

## Maintenance Rule

Add completed experiments to `../EXPERIMENTS.md`, durable choices to
`../DECISIONS.md`, and the current handoff to `../PROGRESS.md`. Keep raw run
details in a dated evidence note and update the canonical results and next-step
notes rather than rewriting historical conclusions in place.
