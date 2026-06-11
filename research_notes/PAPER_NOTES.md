# Paper Notes

## Provisional Story

An inference-only, session-level retrieval-augmented recurrent latent memory can
extend MemGen's usable contextual state without retraining Weaver or Trigger,
while preserving exact original behavior when disabled.

This is a hypothesis, not a claim, until supported by recorded experiments.

## Candidate Contributions

1. An optional inference-only latent memory mechanism for MemGen.
2. Session-local retrieval and recurrent update with explicit isolation guarantees.
3. Compatibility design that preserves the original disabled path.
4. Controlled evidence on quality, latency, memory cost, and long-session behavior.

## Evidence Ledger

| Claim ID | Candidate Claim | Required Evidence | Experiment IDs | Status |
|---|---|---|---|---|
| CLM-01 | Disabled mode preserves baseline behavior | Golden-case equivalence and regression tests | TBD | unsupported |
| CLM-02 | Memory improves target quality | Main comparison across datasets/seeds | TBD | unsupported |
| CLM-03 | Gains depend on retrieval and recurrence | Controlled ablations | TBD | unsupported |
| CLM-04 | Memory remains session-isolated | Leakage and reset tests | TBD | unsupported |
| CLM-05 | Overhead is practical | Latency and memory profiling | TBD | unsupported |

## Paper Outline

### Abstract

- Problem:
- Method:
- Main result:
- Cost:
- Scope/limitation:

### Introduction

- MemGen inference limitation:
- Why inference-only adaptation matters:
- Proposed idea:
- Contributions:

### Related Work

- Latent memory:
- Retrieval-augmented generation:
- Recurrent memory/state:
- Inference-time adaptation:

### Method

- Baseline MemGen inference:
- Session-level memory:
- Retrieval:
- Recurrent update:
- Lifecycle and isolation:
- Complexity:

### Experiments

- Research questions:
- Datasets and metrics:
- Baselines:
- Main results:
- Ablations:
- Efficiency:
- Robustness and failure analysis:

### Discussion

- Interpretation:
- Limitations:
- Broader applicability:

## Writing Rules

- Do not promote hypotheses to claims without experiment IDs.
- Report negative and failed results when they affect interpretation.
- Keep training unchanged as a scoped design constraint, not an unverified advantage.
- Trace every table and figure to reproducible artifacts.
