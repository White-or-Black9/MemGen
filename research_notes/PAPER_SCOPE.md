# Authoritative Paper Scope

Date fixed: 2026-07-05

This note is the authoritative project-state and paper-claim entry point. Older
notes remain experiment history. Where an older broad target conflicts with
this note, treat that wording as a future research goal rather than a supported
claim of the current paper.

## Working Title

**Session-Local Latent Memory Banks for Long-Context Reasoning in MemGen**

## Final Paper Goal / Claim

We add a session-local latent memory bank to MemGen and show that it improves
long-context event reasoning without retraining the Trigger, Weaver, or
Reasoner.

## Operational Evidence Scope

The measured claim is restricted to frozen P7 on EventQA-65536 under the local
MemoryAgentBench `frozen_context_bank` contract and its unchanged local
official prompt/parser/scorer path.

## Benchmark Roles

- Main positive benchmark: EventQA-65536.
- Diagnostic / limitation benchmark: LoCoMo-QA.
- LoCoMo is not positive evidence of multi-turn dialogue improvement.

## Main Method Version

Frozen P7:

- `retrieve_threshold=0.05`
- `update_threshold=0.10`
- `max_slots=16`
- `top_k=2`
- `decay_alpha=0.05`
- session-local Weaver-space latent memory bank
- no Trigger, Weaver, or Reasoner retraining
- no cross-sample memory sharing
- query-time writes blocked under frozen-bank evaluation
- no utility gate, tuple suppression, top-1 fallback, or learned harmfulness policy

## What The Paper Can Claim

- P7 improves official EventQA substring EM over the compressed Bank-off path.
- Across five repeats, P7 has higher EM and fewer format failures than P6,
  while recall is comparable.
- A session-local latent bank can preserve and reuse Weaver-space latent
  memories across a frozen long context without component retraining.
- Query-time bank isolation and write blocking hold in the evaluated protocol.
- The method has a severe context-specific limitation, especially EventQA
  context 4.

## What The Paper Cannot Claim

- General long-context improvement across benchmarks.
- Multi-turn dialogue or LoCoMo-QA improvement.
- General agent-memory improvement.
- Universal memory-bank effectiveness.
- Uniform gains across EventQA contexts.
- Superiority over text summaries, RAG, or matched-budget explicit text before
  those baselines are complete.
- Paper-facing cost superiority from currently combined paired cost fields.

## Main Positive Evidence

- P7 Bank-on EM: `0.197+-0.020` across five repeats.
- Bank-off EM: `0.008`.
- P7 Bank-on recall: `0.254+-0.028`.
- Bank-off recall: `0.178`.
- P7 versus P6: EM `+0.0280`, recall `-0.0044`, format failures `-44.4`.
- Existing prompt ablations, context breakdowns, transition analysis, format
  analysis, and prompt/scorer verification support interpretation and
  reproducibility.

## Main Limitation Evidence

- EventQA context 4: P7 EM mean `0.006`, recall `0.228`, format failures
  `93.8/100`.
- Single-bank oracle attribution identifies a harmful retrieved tuple but does
  not constitute an implemented correction.
- LoCoMo session-level P7 is mechanically protocol-correct but has EM `0`, F1
  `0.02084`, `138/304` no-context denials, and `153/304` refusals.
- The EventQA/LoCoMo gap is consistent with different evidence contracts:
  EventQA exposes an event prefix and six candidates, while LoCoMo requires
  latent-only recovery of exact conversational facts.

## Remaining Evidence Before Full Results Drafting

- Method-separable Bank-off/P7 latency and peak GPU memory.
- Text-summary memory baseline.
- Deterministic BM25 top-2 retrieved-text/RAG baseline.
- 16-token matched-budget explicit-text baseline.
- P7 no-query-retrieval ablation.
- Unified final effectiveness, cost, and appendix tables.

The method, benchmark protocol, existing result, analysis, and limitation
sections can be drafted now. Final comparative Results and Cost sections must
wait for the missing rows.

## Recommended Next Action

When experiments resume, first run a method-separable EventQA cost smoke on
context 0, questions 0-9, for standalone Disabled and frozen P7. Do not rerun
the completed P7/P6 five-repeat effectiveness rows by default.
