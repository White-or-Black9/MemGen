# Paper Skeleton Summary

## Final Claim

The paper claims that adding a session-local latent memory bank to MemGen
improves long-context event reasoning on EventQA-65536 under the local
MemoryAgentBench frozen-bank contract, without retraining Trigger, Weaver, or
Reasoner.

## Planned Structure

The skeleton covers the abstract, introduction, related work, frozen P7 method,
EventQA benchmark/protocol, main results, context and failure analysis,
limitations, conclusion, and appendix. LoCoMo-QA is restricted to limitation
and appendix evidence.

## Drafting Readiness

- Ready now: frozen P7 method, core EventQA setup, P7/Bank-off effectiveness,
  P7/P6 comparison, context-wise and format analyses, context-4 limitation, and
  LoCoMo limitation interpretation.
- Provisional now: abstract, introduction, main-results narrative, and
  conclusion.
- Must wait: final comparator narrative, efficiency claims, and final abstract
  emphasis until explicit-memory baselines and method-separable cost exist.
- Separate prerequisite: Related Work prose requires verified citations.

## Table Readiness

- Ready: P7 versus P6 core table, context-wise table, format-failure table, and
  LoCoMo diagnostic appendix table.
- Partially ready: main EventQA effectiveness table.
- Not ready: explicit-memory baseline table and cost/latency/memory table.

## Figure Readiness

- Conceptually ready: method architecture and frozen-bank protocol.
- Data ready: current Bank-off/P6/P7 result figure and context-wise failure
  figure.
- Optional appendix: LoCoMo limitation figure.
- Not ready: cost-efficiency or explicit-baseline superiority figures.

## Next Experiment

Run a method-separable EventQA cost smoke on context 0, questions 0-9, in
separate Bank-off and P7 processes. Measure construction, query, and end-to-end
latency plus peak GPU memory under one controlled environment. This is a
validation step, not a final cost row.
