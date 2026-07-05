# Ablation Plan

## Harmful Memory Attribution Diagnostic (2026-07-04)

This line is **not** a standard method ablation. It is an oracle
counterfactual diagnostic over a frozen P7 context-4 bank. Gold-derived EM and
no-gold labels are used only after generation to classify effects; they are
not available to a deployable inference policy.

- Completed smoke q0-9:
  `outputs/mab/eventqa_harmful_memory_attribution_smoke/20260704T001049Z-p7-context4-q0-9/`
- Completed expansion q0-99:
  `outputs/mab/eventqa_harmful_memory_attribution_context4_full/20260704T001824Z-p7-context4-q0-99/`
- Conditions: full, leave-one-slot-out, leave-one-tuple-out, slot-only, and
  tuple-only for the dominant ordered tuple `[1,0]`.
- Current result: q0-99 supports a harmful ordered-tuple interaction in one
  frozen bank. It does not establish a general policy or final performance
  improvement.
- Detailed note:
  `research_notes/benchmarks/eventqa_harmful_memory_attribution.md`.

Paused future candidates:

- attribution across the other P7 repeats;
- top-1 fallback;
- dominant-tuple suppression;
- injection budget;
- score-margin gate.

No candidate above is implemented or currently scheduled.

## Goal

Separate the effects of retrieval, recurrent updates, memory capacity, and session
lifecycle while controlling inference settings and compute.

## Experimental Principles

- Change one primary factor per ablation unless testing an explicit interaction.
- Compare against the frozen baseline and a clearly named full method.
- Use identical data, seeds, checkpoints, and decoding settings.
- Report quality, latency, peak memory, and failure rate.
- Record every run in `EXPERIMENTS.md`.

## Core Ablation Matrix

| ID | Factor | Candidate Values | Primary Question | Status |
|---|---|---|---|---|
| ABL-01 | Memory enabled | off / on | Does the method improve over baseline? | planned |
| ABL-02 | Retrieval | none / top-k variants | Is retrieval necessary and how selective should it be? | planned |
| ABL-03 | Update | no-write / recurrent variants | Does recurrent writing add value? | planned |
| ABL-04 | Capacity | small / medium / large | What is the quality-cost frontier? | planned |
| ABL-05 | Eviction | FIFO / similarity / utility | How should bounded memory be maintained? | planned |
| ABL-06 | Aggregation | mean / weighted / gated | How should retrieved memories be combined? | planned |
| ABL-07 | Session length | short / medium / long | How does benefit scale with history? | planned |
| ABL-08 | Reset/isolation | normal / forced tests | Is cross-sample leakage absent? | planned |

## Required Controls

- Original MemGen inference.
- Feature present but `enabled=false`.
- Enabled memory with retrieval disabled.
- Enabled retrieval with writes disabled.
- Empty-memory first step.
- Session reset between consecutive samples.

## Result Template

### ABL-XX: <Factor>

- Linked experiment IDs:
- Fixed settings:
- Compared settings:
- Primary metric:
- Secondary metrics:
- Result:
- Interpretation:
- Confounders:
- Decision impact:
