# Paper Writing Plan

Date: 2026-07-05

## 1. Introduction

- Motivate reusable memory for long-context inference.
- Explain that MemGen has Trigger, Weaver, and Reasoner but no explicit
  session-local bank for reusing latent context across a frozen episode.
- Introduce the session-local Weaver-space latent bank.
- State the scoped EventQA claim and no-retraining contribution.
- Avoid general or multi-turn improvement wording.

## 2. Method

- Frozen Trigger, Weaver, and Reasoner.
- Session-local bank ownership and no cross-sample sharing.
- Write, retrieve, matched update, replacement, and reset.
- P7 thresholds, capacity, top-k, and decay.
- Frozen-bank construction followed by read-only query retrieval.

## 3. Benchmark And Protocol

- EventQA-65536 task and six-candidate next-event query.
- Local MemoryAgentBench official non-strict prompt/parser/scorer.
- Five contexts, 500 questions per repeat.
- Bank-off, P6, and P7 comparison.
- LoCoMo only as diagnostic / limitation evidence.

## 4. Main Results

- P7 Bank-on versus Bank-off across five repeats.
- P7 versus P6 EM, recall, and format failures.
- Context-wise breakdown and context-4 exception.
- Leave explicit-text baseline and cost statements open until runs complete.

## 5. Analysis

- Helpful versus harmful transitions.
- Retrieval activity versus correctness.
- No-gold versus parser-sensitive failures.
- Strict and first-line prompt sensitivity.
- Context-4 fixed routing and harmful tuple attribution.
- Cost caveats and, later, explicit-text versus latent evidence.

## 6. Baselines Still Needed

- method-separable cost;
- text-summary memory;
- BM25 top-2 RAG;
- 16-token matched-budget RAG;
- P7 no-query-retrieval.

## 7. Limitations

- EventQA is not general long-context proof.
- Query-time candidates make EventQA a closed-set reasoning task.
- P7 is not uniformly effective across contexts.
- LoCoMo does not show positive multi-turn QA improvement.
- Latent-only memory struggles with exact conversational fact recovery.

## 8. Appendix

- complete repeat/context tables;
- prompts and scorer verification;
- P4 and historical ablations;
- format and failure examples;
- context-4 and harmful-attribution details;
- LoCoMo audits and diagnostics provenance;
- reproducibility and cost details.

## Writing Sequence

1. Draft Introduction, Method, and Benchmark sections from frozen facts.
2. Draft current EventQA Results and Analysis with explicit missing-row markers.
3. Draft Limitations including LoCoMo.
4. Complete cost and baseline experiments.
5. Replace markers with final tables and comparative prose.
6. Freeze Abstract and Conclusion last.
