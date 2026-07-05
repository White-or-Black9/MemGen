# Session-Local Latent Memory Banks for Long-Context Reasoning in MemGen

Draft status: version 0 scaffold. This document establishes the paper's
argument and evidence boundaries; it is not a submission-ready manuscript.

## Abstract

Long-context inference requires models to retain useful information beyond the
portion of context that can be presented repeatedly at query time. We introduce
an inference-time, session-local latent memory bank for MemGen that stores,
retrieves, updates, and reuses Weaver-space latent memories while leaving the
Trigger, Weaver, and Reasoner frozen. The bank is constructed once for a
context, bounded to a fixed capacity, and reused under a frozen-bank protocol
that permits retrieval but blocks query-time writes. On EventQA-65536 under the
local MemoryAgentBench contract, the frozen P7 configuration obtains an exact
match score of `0.197+-0.020`, compared with `0.008` for the compressed
Bank-off condition. P7 obtains recall of `0.254+-0.028`, compared with `0.178`
for Bank-off, and improves exact match over P6 while reducing format failures.
These results support a scoped conclusion: session-local latent reuse can
improve closed-set long-context event reasoning in this evaluation setting.
The improvement is not uniform across contexts, and a separate LoCoMo-QA
diagnostic does not show positive exact conversational question-answering
performance. [TODO-D01: Revise the final abstract comparison and efficiency
sentence after the explicit-text baselines and method-separable cost results
are complete.]

## 1. Introduction

Long-context systems face two related costs. First, the complete history may be
too large to present at every inference step. Second, even when a model can
process a long history once, useful intermediate representations are often
transient: later queries cannot selectively reuse them without repeating the
original computation or exposing the original text again. A persistent memory
mechanism should therefore preserve information across a bounded inference
session while keeping the state isolated between independent samples.

MemGen provides a useful starting point because its Trigger, Weaver, and
Reasoner already define a latent inference pathway. However, transient latent
generation alone does not provide an explicitly addressable store with
session ownership, retrieval, update, replacement, and reset semantics. We add
such a store as an inference-time session-local latent memory bank. The bank
retains Weaver-space memories produced while processing a context and makes
selected memories available to later queries. No Trigger, Weaver, or Reasoner
parameters are updated.

We study the method on EventQA-65536 under a local MemoryAgentBench
`frozen_context_bank` contract. Each long event context is processed before
question answering, after which the resulting bank is frozen and reused across
questions. Query-time retrieval remains active, but writes are blocked. This
contract isolates the contribution of reusable latent state from query-time
memory accumulation and prevents information from leaking across samples.

The empirical claim is deliberately narrow. EventQA asks the model to select a
next event from six candidates given a visible event prefix. It therefore tests
long-context event reasoning in a closed-set format rather than unrestricted
factual reconstruction. We use EventQA as the positive benchmark and retain
LoCoMo-QA only as a diagnostic limitation: the same general latent-memory idea
does not currently recover exact conversational facts reliably in open-ended
multi-session QA.

The paper makes four contributions:

1. We define a bounded, session-local Weaver-space memory bank for frozen
   MemGen inference.
2. We specify a frozen-bank evaluation protocol that constructs memory once,
   reuses it across questions, and prevents query-time writes.
3. We provide five-repeat EventQA evidence for frozen P7 and compare it directly
   with Bank-off and P6 under a shared prompt and scorer contract.
4. We analyze context variability, format failures, helpful and harmful outcome
   transitions, and a severe context-specific failure mode.

[TODO-D02: Add verified citations and tighten the novelty boundary against the
closest latent-memory, recurrent-state, and inference-time memory methods.]

## 2. Related Work

### 2.1 Latent Memory and Recurrent Inference

Latent-memory methods preserve or regenerate internal representations instead
of repeatedly injecting the complete source text. The relevant distinction for
this work is between learning a new recurrent memory mechanism and adding a
bounded inference-time store around frozen components. Our method follows the
latter approach: it reuses Weaver-space representations but does not introduce
a new training objective or adapt the pretrained Trigger, Weaver, or Reasoner.

### 2.2 Long-Context Reasoning

Long-context methods extend attention windows, compress earlier context, or
maintain recurrent summaries. EventQA emphasizes event ordering and selection
over long narrative contexts. Our evaluation does not establish that latent
memory improves every long-context task; it tests whether reusable latent state
helps under one frozen-context event-reasoning contract.

### 2.3 Retrieval-Augmented and Textual Memory

Retrieval-augmented systems expose selected source text to the model, whereas a
latent bank supplies internal representations. This difference matters for
both evidence fidelity and token budget. A text summary can preserve a compact
readable state, and sparse retrieval can provide exact surface evidence that a
latent representation may not reconstruct. The present draft therefore does
not claim superiority to either summaries or retrieved text before those
comparators are evaluated.

### 2.4 Agent and Conversational Memory

Agent-memory and conversational-memory benchmarks often require exact recovery
of people, dates, preferences, updates, and cross-session relations. Such tasks
place stronger demands on latent-to-fact decoding than closed-set event
selection. Our LoCoMo diagnostic suggests that active latent retrieval alone is
insufficient for this setting, so conversational memory is treated as a method
boundary rather than a demonstrated capability.

[TODO-D03: Replace this positioning scaffold with a verified bibliography and
evidence-backed comparisons to the closest work in all four categories.]

## 3. Method

### 3.1 Frozen MemGen Components

MemGen separates latent augmentation into three components: the Trigger decides
when augmentation is invoked, the Weaver produces latent memories from the
current input state, and the Reasoner generates the answer using the resulting
latent support. We retain these components and their learned parameters. The
only new persistent state is the session-local memory bank.

### 3.2 Session-Local Weaver-Space Memory Bank

A bank belongs to exactly one context or inference session and is reset before
the next sample. It has no global registry and does not share slots across
contexts. Each successful write stores a detached Weaver-space latent memory
and the metadata required for retrieval and replacement. Frozen P7 uses at most
16 slots, with each stored memory containing eight latent vectors.

Given a query representation, the bank scores stored memories using similarity
with temporal decay. Retrieval first applies a threshold and then returns at
most the configured top-k memories. P7 uses `retrieve_threshold=0.05`,
`top_k=2`, and `decay_alpha=0.05`; it does not force a top-1 result when no slot
passes the threshold. Consequently, retrieval can return no prior memory.

### 3.3 Write, Update, and Replacement

During context construction, the existing MemGen path produces a candidate
Weaver-space memory for each enabled construction step. The bank either inserts
the candidate as a new slot or updates the matched memory according to the
retrieval context. P7 uses `update_threshold=0.10`. When the bank reaches its
16-slot capacity, the existing bounded replacement policy preserves the
capacity invariant. P7 does not include a learned utility gate, tuple
suppression, or an oracle harmfulness rule.

### 3.4 Frozen-Bank Question Answering

For each EventQA context, the bank is reset and the construction chunks are
processed sequentially. The resulting state is snapshotted before any
question is answered. Each question starts from the same frozen snapshot. The
query may retrieve prior latent memories, but any write attempt is blocked, and
the snapshot must remain unchanged after generation. Thus, all questions for a
context share the same constructed evidence state without allowing earlier
answers to affect later ones.

### 3.5 Training and Scope

The bank is an inference-time addition. Trigger, Weaver, and Reasoner are not
retrained, and no cross-sample memory is introduced. The paper evaluates this
fixed P7 mechanism rather than selecting thresholds independently for each
context. [TODO-D04: Produce the final method and frozen-bank protocol figures,
then add their cross-references and captions.]

## 4. Benchmark and Evaluation Protocol

### 4.1 EventQA-65536

The local EventQA-65536 evaluation contains five long contexts with 100
questions per context, for 500 questions in each complete run. Contexts are
converted into ordered, sentence-preserving construction chunks. At query time,
the model receives a visible prefix of prior events and six candidate next
events. It must output the selected event under the unchanged local
MemoryAgentBench prompt, parser, and scorer.

The primary metric is the local official normalized substring exact match
(EM). We also report EventQA recall, which detects whether the gold event is
present in the raw output, and format failures, which identify responses that
do not satisfy the expected answer contract. These metrics distinguish answer
selection from output-control failures.

### 4.2 Compared Methods

Bank-off uses the compressed evaluation path without a persistent latent bank;
it is not a full-history baseline. P6 and P7 use the session-local bank and
share `retrieve_threshold=0.05`, `max_slots=16`, `top_k=2`, and
`decay_alpha=0.05`. Their final isolated configuration difference is the update
threshold: `0.095` for P6 and `0.10` for P7.

The P6 and P7 effectiveness results use five repeats. Means and population
standard deviations are reported across repeats. The Bank-off anchor is reused
under the same question, prompt, parser, and scorer contract.

[TODO-D05: Add the completed text-summary, BM25 top-2 retrieved-text, and
16-token matched-budget protocols and results to the main comparator table.]

### 4.3 Cost and Reproducibility

Current paired timing artifacts combine Bank-off and Bank-on execution, and
their peak-memory value is the maximum over both paths. They cannot be used as
method-specific cost evidence. [TODO-D06: Run and report method-separable
Bank-off/P7 latency and peak-memory measurements under one controlled EventQA
environment, then extend the same accounting to final baselines.]

## 5. Experiments

### 5.1 Main EventQA Result

Frozen P7 obtains EM `0.197+-0.020`, compared with `0.008` for Bank-off. P7
also obtains recall `0.254+-0.028`, compared with `0.178` for Bank-off. Under
this contract, the bank therefore improves both normalized substring EM and
gold-event recall relative to the compressed no-bank path. This result supports
the paper's scoped EventQA claim, but it does not establish improvement on
other long-context tasks.

### 5.2 P7 Versus P6

P6 obtains EM `0.169+-0.018`, recall `0.258+-0.016`, and
`165.8+-19.8` format failures. P7 obtains EM `0.197+-0.020`, recall
`0.254+-0.028`, and `121.4+-8.8` format failures. Relative to P6, P7 changes
EM by `+0.0280`, recall by `-0.0044`, and format failures by `-44.4`. P7 is
therefore selected as the paper-facing configuration because it improves EM
and output validity while retaining comparable recall. This comparison does
not show that the higher update threshold is universally preferable; it
reports the observed trade-off under the fixed EventQA protocol.

### 5.3 Explicit-Memory and Retrieval Controls

Text summaries and retrieved source text provide important controls because
they expose readable evidence rather than latent state. The matched-budget
condition further tests whether the observed benefit can be explained by any
additional query-time evidence channel. [TODO-D07: Insert the explicit-memory
baseline results only after all methods use the frozen prompt/scorer contract
and their injected text budgets are recorded.]

### 5.4 Query-Retrieval Ablation

The current results compare a complete P7 bank with Bank-off, but they do not
isolate query-time retrieval from construction-side effects. [TODO-D08: Run the
P7 no-query-retrieval ablation with identical bank construction and blocked
query writes, then report whether the EventQA gain depends on retrieved latent
support.]

### 5.5 Final Result Packaging

[TODO-D09: Build the unified final EventQA aggregation table and artifact
manifest after all required rows pass scope, scorer, repeat, and provenance
checks.]

## 6. Analysis

### 6.1 Context-Wise Behavior

The aggregate P7 improvement is not uniform. Context 4 remains a severe
failure: across five repeats, P7 obtains EM `0.006`, recall `0.228`, and
`93.8/100` format failures. The gap between recall and EM indicates that some
outputs contain relevant content without satisfying the answer contract, but
the low recall also shows that formatting is not the sole problem.

[TODO-D10: Consolidate the verified per-context Bank-off, P6, and P7 values into
the context-wise table and variance figure without rerunning effectiveness.]

### 6.2 Format Failures

P7 reduces the mean number of format failures by `44.4` relative to P6. The
available prompt and failure analyses show that answer formatting materially
affects measured EM, but they also identify no-gold outputs in which the correct
event is absent. The EventQA gain should therefore not be attributed solely to
parsing or first-line cleanup.

### 6.3 Helpful and Harmful Transitions

Paired Bank-off/Bank-on outcomes can be separated into helpful transitions,
harmful transitions, unchanged-correct cases, unchanged-wrong cases, and
format-related harm. This view is necessary because aggregate improvement can
coexist with memory-induced regressions. [TODO-D11: Extract the final
five-repeat helpful/harmful transition counts from the authoritative summaries
and add the corresponding analysis table.]

### 6.4 Context 4 and Harmful Memory Interaction

An oracle diagnostic on one frozen context-4 bank found that the ordered slot
pair `[1,0]` was selected on all 100 questions and reproduced the full-bank
collapse. Removing that pair yielded 15 EM rescues and 96 format improvements,
with no EM regressions in that diagnostic. This result demonstrates that a
harmful slot interaction can dominate one bank, but it does not establish a
general causal mechanism or provide a deployable correction. The analysis is
restricted to one bank, one context, and oracle interventions.

### 6.5 Retrieval Activity Is Not Sufficient

The mechanism requires both successful retrieval and useful consumption of the
retrieved latent state. EventQA contains cases in which retrieval changes the
outcome helpfully and cases in which it is harmful. The LoCoMo diagnostic makes
the distinction sharper: retrieval can be mechanically active while the
Reasoner still fails to produce the required fact. Retrieval counters should
therefore be interpreted as mechanism evidence, not as a proxy for correctness.

## 7. Limitations

The evidence supports a result on EventQA-65536 under a local
MemoryAgentBench frozen-bank contract. It does not prove general long-context
improvement, and EventQA's visible prefix and six candidate answers make it
less demanding than open-ended factual reconstruction. The current paper also
lacks final text-summary, BM25, matched-budget, no-query-retrieval, and
method-separable cost rows.

Performance is heterogeneous across contexts. Context 4 nearly collapses under
P7, and the harmful-tuple analysis is an oracle single-bank diagnostic rather
than part of the method. The fixed 16-slot capacity and retrieval policy may
also discard or combine evidence in ways that are not captured by aggregate
metrics.

LoCoMo-QA provides a separate boundary test. On a paired two-conversation,
304-question diagnostic, both Disabled and P7 obtain zero EM; token F1 is
`0.01834` and `0.02084`, respectively, and every paired question is
exact-match wrong. Although P7 retrieval is active and query writes remain
zero, it produces 138 no-context denials and 153 refusals. These results do not
support multi-turn dialogue improvement. Instead, they suggest that latent-only
memory is currently inadequate for exact conversational fact recovery under
the evaluated prompt and evidence contract.

[TODO-D12: Package the LoCoMo protocol, prompt templates, reliable diagnostic
fields, and failure examples into an appendix table without including the
known-unreliable cost counters.]

## 8. Conclusion

We introduced an inference-time, session-local latent memory bank that reuses
Weaver-space memories while keeping MemGen's Trigger, Weaver, and Reasoner
frozen. Under the EventQA-65536 frozen-bank protocol, frozen P7 improves exact
match and recall over the compressed Bank-off path and improves the EM/format
trade-off over P6. The result demonstrates the value of persistent latent reuse
for this closed-set long-context event-reasoning setting.

The same evidence does not justify a broader claim about arbitrary
long-context tasks or multi-session conversational memory. Future work should
compare latent memory with explicit summaries and retrieved text, isolate the
role of query retrieval, improve harmful-memory control, and develop stronger
latent-to-fact decoding. [TODO-D13: Freeze the final conclusion after the
baseline, ablation, cost, and unified-table TODOs are resolved, downgrading any
comparative statement that the completed evidence does not support.]
