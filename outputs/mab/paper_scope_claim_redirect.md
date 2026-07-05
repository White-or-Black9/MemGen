# Paper Scope And Claim Redirect

Date: 2026-07-05

## Working Title

**Session-Local Latent Memory Banks for Long-Context Reasoning in MemGen**

## Final Scoped Claim

We add a session-local latent memory bank to MemGen and show that it improves
long-context event reasoning without retraining the Trigger, Weaver, or
Reasoner. The operational evidence is frozen P7 on EventQA-65536 under the
local MemoryAgentBench frozen-bank contract.

## Claims Explicitly Not Made

- General long-context improvement across benchmarks.
- Multi-turn dialogue or LoCoMo-QA improvement.
- General agent-memory or universal memory-bank effectiveness.
- Uniform EventQA context improvement.
- Superiority over summaries, RAG, or matched-budget text before those rows exist.
- Cost efficiency from currently combined paired cost records.

## Evidence Supporting The Scoped Claim

- P7 EM `0.197+-0.020` versus Bank-off `0.008`.
- P7 recall `0.254+-0.028` versus Bank-off `0.178`.
- P7 versus P6: EM `+0.0280`, recall `-0.0044`, format failures `-44.4`.
- Five-repeat P7/P6 evidence, unchanged prompt/parser/scorer, frozen-bank
  integrity, context analysis, and format analysis are reusable.

## Why LoCoMo Is Not Main Positive Evidence

LoCoMo uses the same P7 mechanism but a harder evidence contract. EventQA
shows a visible event prefix and six candidate answers; LoCoMo shows only an
open question and requires exact latent-to-fact recovery. LoCoMo P7 has EM `0`
and all 304 paired rows are exact-match wrong despite active retrieval.

## LoCoMo Paper Placement

Mention LoCoMo in Limitations and appendix diagnostics. Use it to show that
active latent retrieval does not guarantee exact multi-session conversational
fact decoding. Do not describe it as multi-turn improvement.

## Remaining EventQA Evidence

- method-separable Bank-off/P7 latency and peak memory;
- text-summary memory baseline;
- BM25 top-2 RAG baseline;
- 16-token matched-budget baseline;
- P7 no-query-retrieval ablation;
- unified final tables.

## Next Action Before Full Drafting

Run a method-separable EventQA cost smoke on context 0, q0-9, Disabled and
frozen P7 after the logging/schema contract is ready. Existing Method,
Benchmark, core EventQA Results, and Limitation prose may be drafted now;
final comparison and cost claims must wait.
