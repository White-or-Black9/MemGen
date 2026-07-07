# Authoritative Paper Scope And Evidence Boundary

Date fixed: 2026-07-05

`paper/outline.md` is authoritative for the paper title, problem framing,
contributions, research questions, and section structure. This note is
authoritative for the current method freeze, verified evidence, limitations,
and experiment gaps. Older notes remain experiment history.

## Working Title

**Inference-Time Latent Memory Management for Long-Horizon LLM Agents**

## Final Paper Goal / Claim

We propose a session-local latent memory bank for MemGen-style LLM agents. The
bank stores, retrieves, updates, replaces, and reuses latent memories during
inference so that historical information can remain available across
long-horizon reasoning steps without retraining the Trigger, Weaver, or
Reasoner.

## Operational Evidence Scope

The paper studies inference-time latent memory management for long-horizon LLM
agents. The current positive performance evidence is restricted to frozen P7
on EventQA-65536 under the local MemoryAgentBench `frozen_context_bank`
contract and its unchanged local official prompt/parser/scorer path. This
operational evidence boundary does not redefine the paper goal as
EventQA-specific.

## Current Evidence Roles

- Main positive long-context reasoning evidence: EventQA-65536.
- Optional diagnostic / limitation evidence: LoCoMo-QA.
- LoCoMo is not a required main-paper benchmark and is not positive evidence
  for the current performance claim.

## Research Questions

- RQ1: Does the proposed bank preserve original MemGen behavior when disabled?
- RQ2: Does it produce meaningful write, retrieval, update, replacement, and
  reset behavior during inference?
- RQ3: Does session-level latent reuse improve long-context reasoning?
- RQ4: Which design choices, including thresholds, top-k, capacity, and
  replacement policy, matter most?

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

## What Current Evidence Can Claim

- P7 improves official EventQA substring EM over the compressed Bank-off path.
- Across five repeats, P7 has higher EM and fewer format failures than P6,
  while recall is comparable.
- A session-local latent bank can preserve and reuse Weaver-space latent
  memories across a frozen long context without component retraining.
- Query-time bank isolation and write blocking hold in the evaluated protocol.
- The method has a severe context-specific limitation, especially EventQA
  context 4.

## What Current Evidence Cannot Yet Claim

- Uniform long-context improvement across benchmarks or contexts.
- LoCoMo-QA improvement.
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

- Text-summary memory baseline.
- Deterministic BM25 top-2 retrieved-text/RAG baseline.
- 16-token matched-budget explicit-text baseline.
- P7 no-query-retrieval ablation.
- Unified final effectiveness, cost, and appendix tables.

## Verified Cost Evidence

- A complete same-GPU serialized pass now provides method-separable Disabled
  and P7 construction, query, end-to-end, and peak-allocation fields across all
  five contexts and 500 questions.
- P7 end-to-end time is `1.056x` Disabled after amortizing construction across
  the 500 queries; max incremental peak allocation is about `29 MiB` higher.
- Treat this as protocol-specific cost evidence, not a throughput or universal
  efficiency claim.

The method, benchmark protocol, existing result, analysis, and limitation
sections can be drafted now. Final comparative Results and Cost sections must
wait for the missing rows.

## Recommended Next Evidence Action

When experiments resume, first run a method-separable EventQA cost smoke on
context 0, questions 0-9, for standalone Disabled and frozen P7. Do not rerun
the completed P7/P6 five-repeat effectiveness rows by default.
