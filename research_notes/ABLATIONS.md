# Ablation Plan

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
