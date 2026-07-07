# Paper Skeleton

Status: planning scaffold, not manuscript prose.

## 1. Working Title

**Inference-Time Latent Memory Management for Long-Horizon LLM Agents**

## 2. One-Sentence Claim

We introduce inference-time latent memory management for long-horizon
MemGen-style LLM agents through a session-local bank that stores, retrieves,
updates, replaces, and reuses latent memories without component retraining.

Evidence boundary: the current positive performance result is EventQA-scoped
under the local MemoryAgentBench frozen-bank contract. It does not yet establish
benchmark-general improvement.

## 3. Abstract Skeleton

### Motivation

- Long-context inference requires a mechanism that can retain and reuse useful
  information after the original context is no longer visible.
- MemGen generates latent representations through Trigger, Weaver, and
  Reasoner, but does not provide an explicit session-local bank for reusing
  those representations across a frozen context.

### Method

- Introduce an inference-time, session-local Weaver-space latent memory bank.
- Summarize its write, thresholded retrieval, matched update, bounded
  replacement, reset, and read-only query behavior.
- State that Trigger, Weaver, and Reasoner are not retrained.

### Main Evidence

- Evaluate frozen P7 on EventQA-65536 using the local MemoryAgentBench
  `frozen_context_bank` protocol and unchanged local official prompt/scorer.
- Report only completed evidence: P7 EM `0.197+-0.020` versus Bank-off `0.008`;
  P7 recall `0.254+-0.028` versus Bank-off `0.178`; P7 has higher EM and fewer
  format failures than P6 with comparable recall.
- Do not claim comparison with summary, RAG, matched-budget, or cost baselines
  until those rows exist.

### Limitation

- State that the result is scoped to a closed-set next-event benchmark.
- Flag severe context-specific failure, especially EventQA context 4.
- Keep the abstract focused on nonuniform EventQA behavior; reserve optional
  LoCoMo diagnostics for the limitation section or appendix.

### Abstract Writing Gate

- A provisional abstract can be drafted from existing evidence.
- Freeze final abstract emphasis only after explicit-text baselines and their
  method-separable cost rows are available; Disabled/P7 cost is complete.

## 4. Introduction Skeleton

### Problem

- Long contexts exceed what can be repeatedly presented to a model.
- Compressing context once is insufficient if useful latent evidence must be
  retrieved and reused across many later questions.

### Why MemGen Needs A Session-Local Bank

- Trigger and Weaver can produce latent state, but latent outputs are not by
  themselves a persistent, explicitly addressable memory across a session.
- A bounded bank provides ownership, retrieval, update, replacement, and reset
  semantics without creating cross-sample state.

### What This Paper Adds

- A session-local latent memory bank attached at inference time.
- A frozen-bank protocol that constructs once, reuses the bank across queries,
  and blocks query-time writes.
- A controlled EventQA evaluation with frozen pretrained components.

### What Is Evaluated

- Main positive benchmark: EventQA-65536.
- Main comparisons currently ready: compressed Bank-off, text-summary, BM25
  retrieved text, matched-budget text, P6, P7 no-query-retrieval, and P7.
- LoCoMo is optional diagnostic limitation evidence rather than a required
  benchmark axis.

### Contributions

1. An inference-time latent memory management mechanism for MemGen-style LLM
   agents.
2. A session-local latent memory bank with explicit write, retrieval, update,
   replacement, and reset operations.
3. Long-context reasoning evaluation covering task performance and internal
   memory behavior.

### Introduction Overclaim Warning

Do not convert the long-horizon agent framing into unsupported benchmark-general
performance or “outperforms RAG” claims.

## 5. Related Work Skeleton

### Latent Memory And MemGen

- Position the method relative to models that generate, compress, retrieve, or
  recurrently reuse latent states.
- Distinguish a session-local inference bank from retraining latent-memory
  components.
- Verified citations cover recurrent, compressed, retrieved, and learned
  latent-memory systems; available formal venue metadata is synchronized.

### Long-Context Reasoning

- Cover full-context attention, compression, recurrent state, and
  memory-augmented inference.
- Position EventQA as long-context event reasoning with a visible event prefix
  and candidate set, not open-ended factual recall.

### Retrieval-Augmented Memory

- Contrast latent retrieval with explicit retrieved-text injection.
- The BM25 and matched-budget rows are now complete, but claims must remain
  scoped to this EventQA protocol and point-estimate repeat boundary.

### Long-Horizon Agent Memory

- Discuss session memory, episodic memory, summaries, and external stores as
  mechanisms for preserving historical information across agent steps.
- Separate the general long-horizon memory problem from the current EventQA
  evidence boundary.

### Optional Positioning Against Exact Conversational Recall

- LoCoMo requires open-ended recovery of exact people, dates, preferences, and
  cross-session relations from dialogue history.
- The current P7 diagnostic fails this use case despite active retrieval.
- Present this contrast as a boundary of the method, not a positive benchmark.

### Related Work Writing Gate

The conceptual structure, verified citations, and closest-neighbor boundary are
now drafted in `paper/draft_v0.md`; only final venue-format and prose polish
remain.

## 6. Method Skeleton

### Frozen MemGen Components

- Define Trigger, Weaver, and Reasoner at the level needed to explain the data
  flow.
- State explicitly that all three remain frozen.
- Distinguish the new inference state from training-time adaptation.

### Session-Local Weaver-Space Latent Bank

- One bank per context/session; no cross-sample sharing.
- Stored value: Weaver-generated latent memory mapped into Reasoner space.
- Capacity: 16 slots; each stored slot contains eight latent vectors.
- Query retrieval: thresholded top-2 under P7.

### Write, Retrieve, Update, Replacement, Reset

- Query/key construction and decayed cosine retrieval.
- `retrieve_threshold=0.05` and no top-1 fallback.
- `update_threshold=0.10` and matched update behavior.
- Insertion and bounded capacity replacement.
- Session reset and detached storage.

### Frozen-Bank Protocol

- Sequentially ingest the EventQA context chunks once.
- Freeze the resulting context-local bank.
- Restore/reuse the same frozen bank across all questions.
- Allow query retrieval but block query writes.
- Verify `query_write_count=0` and unchanged bank snapshots.

### No Retraining

- No Trigger retraining.
- No Weaver retraining.
- No Reasoner retraining.
- No utility gate, tuple suppression, learned harmfulness policy, or top-1
  fallback in frozen P7.

### Method Writing Gate

Ready to draft now from `research_notes/METHOD.md` and frozen P7 artifacts.
Keep local paths, helper names, and debug counters in the appendix.

## 7. Experiment Skeleton

### EventQA-65536 Benchmark

- Five long book contexts, 100 questions per context.
- Official sentence-preserving construction chunks through the local
  MemoryAgentBench path.
- Each query visibly includes prior events and six candidate next events.
- Primary metric: official normalized substring EM.
- Supporting metrics: EventQA recall, format failures, transition categories.

### Bank-Off Versus Bank-On

- Bank-off is the compressed no-persistent-bank path, not a full-history model.
- P7 Bank-on is the frozen session-local latent-bank path.
- Use paired question sets and unchanged prompt/parser/scorer.

### P6 Versus P7

- P6 and P7 share `retrieve_threshold=0.05`, `max_slots=16`, `top_k=2`, and
  `decay_alpha=0.05`.
- Isolated final difference: `update_threshold=0.095` versus `0.10`.
- Report five-repeat mean, population standard deviation, and context spread.

### Completed Baselines

The following comparisons are complete and appear in the manuscript:

- completed method-separable Bank-off/P7 cost (`1.056x` end-to-end ratio under
  the full serialized protocol);
- text-summary memory;
- BM25 top-2 retrieved text;
- 16-token matched-budget retrieved text;
- P7 no-query-retrieval.

### LoCoMo As Limitation Only

- Report the two-conversation diagnostic only in Limitations or appendix.
- State that protocol invariants held but EM remained zero for both modes.
- Do not place LoCoMo in the positive main-results table.

## 8. Analysis Skeleton

### Context-Wise Behavior

- Show P7, P6, and Bank-off per-context EM/recall.
- Emphasize that gains are not uniform and context 4 is an exception.

### Format Failures

- Compare format-failure burden across P7, P6, P4, and Bank-off where valid.
- Separate parser-sensitive cases from outputs that never contain the gold
  event.

### Helpful Versus Harmful Transitions

- Define helpful, harmful, unchanged-correct, unchanged-wrong, and format-harm
  transitions from paired Bank-off/Bank-on outcomes.
- Use P7/P6 repeat aggregates.

### Context 4 Failure

- Report P7 context-4 EM `0.006`, recall `0.228`, and format failures
  `93.8/100` across five repeats.
- Discuss fixed routing and the single-bank harmful tuple diagnostic.
- Do not present oracle tuple removal as part of the method.

### Retrieval Activity Versus Correctness

- Distinguish retrieval execution from useful evidence utilization.
- Use EventQA helpful/harmful transitions and LoCoMo active-retrieval failure
  as complementary boundary evidence.
- Avoid inferring causality from retrieval score alone.

## 9. Limitation Skeleton

- Evidence is scoped to EventQA-65536 under a local frozen-bank contract.
- EventQA is a closed-set next-event task with visible candidate answers; it is
  not general long-context factual QA proof.
- P7 is not uniformly effective across contexts.
- Optional LoCoMo evidence does not support an additional positive performance
  claim: Disabled and P7 EM are both zero on the evaluated 304-question paired
  slice.
- Active latent retrieval does not guarantee exact conversational fact
  recovery.
- Method-separable cost, text-summary, RAG, matched-budget, and
  no-query-retrieval evidence remain incomplete.
- Harmful tuple attribution is oracle, single-bank limitation evidence.

## 10. Conclusion Skeleton

### Scoped Conclusion

- Restate the session-local latent-bank contribution.
- State the completed EventQA result and no-retraining scope.
- Emphasize bounded inference-time memory and frozen query behavior.
- Keep the conclusion specific to long-context event reasoning on EventQA.

### Future Work

- Explicit-text and matched-budget comparisons.
- Better utility-aware retrieval without changing the current paper method.
- More reliable handling of harmful slot combinations.
- Stronger latent-to-fact decoding for conversational and multi-session memory.

### Conclusion Writing Gate

Draft a provisional conclusion now, but freeze it only after the final baseline
and cost tables determine which comparative statements remain valid.
