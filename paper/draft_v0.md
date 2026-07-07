# Inference-Time Latent Memory Management for Long-Horizon LLM Agents

## Abstract

Long-horizon LLM agents need to retain and selectively reuse historical
information across inference steps. We introduce an inference-time,
session-local latent memory bank for MemGen that stores, retrieves, updates,
replaces, and reuses Weaver-space latent memories while leaving the Trigger,
Weaver, and Reasoner frozen. The bank is constructed once for a
context, bounded to a fixed capacity, and reused under a frozen-bank protocol
that permits retrieval but blocks query-time writes. Under the
MemoryAgentBench-compatible EventQA-65536 protocol, frozen P7 obtains exact
match `0.197±0.020` and recall `0.254±0.028`. The compressed Bank-off condition
obtains `0.008` and `0.178`, respectively. Relative to P6, P7 improves
exact match by `+0.0280` and reduces format failures by `44.4` while keeping
recall comparable (`-0.0044`). P7 also exceeds the completed explicit-memory
controls, including same-model text summary, BM25 top-2 retrieved text, and a
16-token matched-budget retrieved-text condition. A no-query-retrieval
ablation collapses performance exactly to the Disabled line, showing that the
observed gain depends on query-time latent retrieval rather than construction
alone. Method-separable cost measurements show that P7 adds modest overhead
relative to Disabled under this protocol, but do not support a blanket
cost-superiority claim. These results show that session-local latent reuse can
improve closed-set long-context event reasoning under this protocol. The gains
are heterogeneous across contexts and do not establish benchmark-general
long-horizon memory improvement.

## 1. Introduction

Long-context systems face two related costs. First, the complete history may be
too large to present at every inference step. Second, useful intermediate
representations remain transient even when a model can process a long history
once. Later queries cannot selectively reuse them without repeating the
computation or exposing the source text again. Prior work addresses
this problem through segment recurrence and compressed state
[@dai2019transformerxl; @rae2019compressive; @bulatov2022rmt], retrieval over
stored activations [@wu2022memorizing; @he2024camelot], and learned latent
memory modules [@wang2024memoryllm; @behrouz2025titans]. These approaches
establish that persistent internal state can extend effective context. However,
they do not specify the session ownership and frozen query-time protocol needed
for controlled reuse within a bounded agent episode.

MemGen provides a useful starting point because its Trigger, Weaver, and
Reasoner already define a latent inference pathway [@zhang2025memgen]. However,
transient latent generation alone does not provide an explicitly addressable
store with session ownership, retrieval, update, replacement, and reset
semantics. We add such a store as an inference-time session-local latent memory
bank. The bank retains Weaver-space memories produced while processing a
context and makes selected memories available to later queries. No Trigger,
Weaver, or Reasoner parameters are updated.

We study the method on EventQA-65536 under a MemoryAgentBench-compatible
`frozen_context_bank` contract [@hu2025memoryagentbench]. Each long event
context is processed before question answering, after which the resulting bank
is frozen and reused across questions. Query-time retrieval remains active, but
writes are blocked. This contract isolates the contribution of reusable latent
state from query-time memory accumulation and prevents information from
leaking across samples.

The empirical scope is deliberately narrow. EventQA asks the model
to select a next event from six candidates given a visible event prefix. It
therefore tests long-context event reasoning in a closed-set format rather than
all long-horizon agent capabilities. EventQA is therefore the primary positive
benchmark, while diagnostic evidence is kept separate from the main claim.

The paper makes three contributions:

1. We introduce an inference-time latent memory management mechanism for
   MemGen-style LLM agents.
2. We design a session-local latent memory bank with explicit write, retrieval,
   update, replacement, and reset operations.
3. We evaluate the mechanism on long-context reasoning, analyzing both task
   performance and internal memory behavior through repeated EventQA results,
   controlled comparisons, and failure analyses.

The novelty claim is intentionally operational rather than architectural. We
do not claim to introduce latent memory, recurrent state, or test-time memory
in general. The contribution is a bounded, session-owned retrieval bank around
an already trained MemGen latent pathway, together with a frozen-bank protocol
that separates context-time construction from query-time retrieval and blocks
query-time writes.

## 2. Related Work

### 2.1 Latent Memory and Recurrent Inference

Segment-recurrent models reuse hidden states across chunks, as in
Transformer-XL [@dai2019transformerxl], while Compressive Transformer retains
older activations in a compressed memory [@rae2019compressive]. Recurrent
Memory Transformer instead learns to pass dedicated memory tokens between
segments [@bulatov2022rmt], and Infini-attention integrates compressive memory
directly into attention [@munkhdalai2024infini]. These methods alter or train
the sequence model's recurrent computation. Our setting is narrower: the
pretrained Trigger, Weaver, and Reasoner remain frozen, and persistence is
implemented as an external session-local store over Weaver outputs.

Other work is closer in representational substrate. Memorizing Transformers
retrieves stored internal key--value pairs by approximate nearest-neighbor
search [@wu2022memorizing]. CAMELoT attaches a training-free associative memory
to a frozen language model [@he2024camelot], whereas MEMORYLLM maintains a
self-updatable latent memory pool [@wang2024memoryllm]. M+ combines such latent
memory with a co-trained retriever for longer retention [@wang2025mplus], while
Titans learns a neural long-term memory at test time [@behrouz2025titans]. Our
work does not claim priority over these latent-memory designs. It studies a
different integration point: bounded reuse of memories generated by an
existing MemGen Weaver, with explicit session reset, replacement, retrieval,
and a no-write query phase.

### 2.2 Long-Context Reasoning

Long-context methods extend attention windows, compress earlier context, or
maintain recurrent summaries [@dai2019transformerxl; @rae2019compressive;
@munkhdalai2024infini]. MemoryAgentBench evaluates memory agents across several
competencies and includes EventQA as an accurate-retrieval task over event
histories [@hu2025memoryagentbench]. Our EventQA variant emphasizes event
ordering and candidate selection over long narrative contexts. It is therefore
evidence for one frozen-context reasoning contract, not for universal
long-context improvement.

### 2.3 Retrieval-Augmented and Textual Memory

Retrieval-augmented generation combines a parametric generator with retrieved
non-parametric evidence [@lewis2020rag]. Agent memory systems can similarly
store readable records and expose selected text at inference time; MemLLM, for
example, trains an LLM to use an explicit structured read--write memory
[@modarressi2024memllm]. A latent bank instead supplies internal
representations. This difference matters for evidence fidelity and token
budget: summaries provide compact readable state, while sparse retrieval can
preserve exact surface evidence that a latent representation may not
reconstruct. Our EventQA controls therefore include same-model text summary,
BM25 top-2 retrieved text, and a 16-token matched retrieved-text condition.
They bound the empirical claim to the tested protocol; they do not establish
general superiority over RAG or textual memory.

### 2.4 Long-Horizon Agent Memory

Long-horizon agents require persistent state ownership, selective access,
bounded storage, and isolation between independent sessions. MemGPT manages
multiple memory tiers as virtual context for document analysis and multi-session
chat [@packer2023memgpt]. LongMemEval evaluates extraction, multi-session and
temporal reasoning, knowledge updates, and abstention over sustained chat
histories [@wu2024longmemeval], while LoCoMo targets question answering,
summarization, and dialogue generation over very long conversations
[@maharana2024locomo]. These systems and benchmarks motivate the broader agent
memory problem, but our positive evidence is narrower: EventQA tests
closed-set event reasoning under a frozen context bank. The LoCoMo result is
therefore retained as a limitation rather than used to claim conversational or
benchmark-general long-horizon memory.

## 3. Method

### 3.1 Frozen MemGen Components

MemGen separates latent augmentation into three components. The Trigger decides
when to invoke augmentation, the Weaver produces latent memories, and the
Reasoner generates an answer using the resulting latent support. We retain
these components and their learned parameters. The
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

![Method architecture: frozen MemGen components are augmented by a bounded,
session-local Weaver-space latent bank. Construction-time latent memories can
be inserted, matched and updated, or replaced under the capacity constraint;
query-time retrieval injects selected latent support into the Reasoner, while
query-time writes are blocked.](figures/fig1_method_architecture.svg)

**Figure 1 | Inference-time session-local latent memory architecture.** Trigger,
Weaver, and Reasoner remain frozen. The bank stores at most 16 Weaver-space
memories, retrieves at most two slots after similarity, temporal-decay, and
threshold filtering, and is reset between independent sessions. The diagram
shows the operational interface rather than a training procedure.

### 3.4 Frozen-Bank Question Answering

For each EventQA context, the bank is reset and the construction chunks are
processed sequentially. The resulting state is snapshotted before any
question is answered. Each question starts from the same frozen snapshot. The
query may retrieve prior latent memories, but any write attempt is blocked, and
the snapshot must remain unchanged after generation. Thus, all questions for a
context share the same constructed evidence state without allowing earlier
answers to affect later ones.

![Frozen-bank protocol: an EventQA context constructs one bounded bank, which
is snapshotted before question answering. Every question independently restores
the same snapshot, retrieves latent support, and generates an answer without
writing to the bank.](figures/fig2_frozen_bank_protocol.svg)

**Figure 2 | Frozen-context-bank evaluation protocol.** Ordered context chunks
construct one context-owned bank before it is snapshotted and frozen. Each
question restores that same snapshot and may retrieve latent support, but the
protocol requires `query_write_count = 0` and an unchanged post-query bank.

### 3.5 Training and Scope

The bank is an inference-time addition. Trigger, Weaver, and Reasoner are not
retrained, and no cross-sample memory is introduced. The paper evaluates this
fixed P7 mechanism rather than selecting thresholds independently for each
context.

## 4. Benchmark and Evaluation Protocol

### 4.1 EventQA-65536

The EventQA-65536 evaluation contains five long contexts with 100
questions per context, for 500 questions in each complete run. Contexts are
converted into ordered, sentence-preserving construction chunks. At query time,
the model receives a visible prefix of prior events and six candidate next
events. It must output the selected event under a fixed MemoryAgentBench
prompt, parser, and scorer.

The primary metric is normalized substring exact match (EM), following the
benchmark evaluator. We also report EventQA recall, which detects whether the gold event is
present in the raw output, and format failures, which identify responses that
do not satisfy the expected answer contract. These metrics distinguish answer
selection from output-control failures.

### 4.2 Compared Methods

Bank-off uses the compressed evaluation path without a persistent latent bank;
it is not a full-history baseline. The explicit-memory controls are a
same-model rolling text summary, deterministic BM25 top-2 retrieved text, and
a 16-token matched-budget retrieved-text condition. We also evaluate a P7
no-query-retrieval ablation that preserves identical bank construction but
disables latent retrieval at question time. P6 and P7 use the session-local
bank and share `retrieve_threshold=0.05`, `max_slots=16`, `top_k=2`, and
`decay_alpha=0.05`. Their final isolated configuration difference is the update
threshold: `0.095` for P6 and `0.10` for P7.

The P6, P7, and reconstructed Bank-off anchors use five repeats. Means and
population standard deviations are reported across repeats. The text-summary,
BM25, matched-budget, and no-query-retrieval controls use one deterministic
full pass each. They share the same question, prompt, parser, and scorer
contract and are reported as point estimates rather than repeated rows.

### 4.3 Cost and Reproducibility

A same-GPU serialized pass measures Disabled and P7 in separate processes
over all five contexts and 500 questions, excluding model loading. Disabled
requires `367.448 s` end-to-end (`0.735 s/question` amortized), while P7
requires `387.999 s` including `78.454 s` of context-bank construction
(`0.776 s/question` amortized). P7 therefore adds `20.551 s`, or `5.6%`, under
this protocol. Its maximum incremental peak allocation is about `29 MiB` above
Disabled. These values describe protocol-specific inference cost, not
throughput superiority.

The explicit-text controls also have full-pass cost measurements, but they
should be interpreted with care. BM25 top-2 requires `692.845 s`
(`1.386 s/question`) and a peak incremental allocation of about `3.51 GiB`.
The 16-token matched-budget condition requires `501.761 s`
(`1.004 s/question`) with about `171 MiB` peak incremental allocation. The
same-model text-summary baseline requires `691.345 s`, but its timing and peak
memory were collected under shared-GPU contention and are therefore retained as
diagnostic rather than paper-facing cost evidence. The completed cost package
supports a measured overhead claim for P7 relative to Disabled, but not a
blanket cost-superiority claim over all explicit-text alternatives.

## 5. Experiments

### 5.1 Main EventQA Result

Frozen P7 obtains EM `0.197±0.020`, compared with `0.008` for Bank-off. P7
also obtains recall `0.254±0.028`, compared with `0.178` for Bank-off. Under
this contract, the bank therefore improves both normalized substring EM and
gold-event recall relative to the compressed no-bank path. This result supports
the paper's scoped EventQA claim, but it does not establish improvement on
other long-context tasks.

| Method | Repeats | EM | Recall | Format failures |
|---|---:|---:|---:|---:|
| Disabled / compressed Bank-off | 5 | 0.008±0.000 | 0.178±0.000 | 377.0±0.0 |
| Same-model text-summary memory | 1 | 0.012 | 0.078 | 267.0 |
| BM25 top-2 retrieved text | 1 | 0.030 | 0.226 | 265.0 |
| 16-token matched-budget retrieved text | 1 | 0.068 | 0.180 | 347.0 |
| P6 non-strict | 5 | 0.169±0.018 | 0.258±0.016 | 165.8±19.8 |
| P7 with query retrieval disabled | 1 | 0.008 | 0.178 | 377.0 |
| Frozen P7 non-strict | 5 | **0.197±0.020** | **0.254±0.028** | **121.4±8.8** |

**Table 1 | EventQA-65536 effectiveness.** Repeated rows report mean ±
population standard deviation. Single-pass controls are point estimates.
Bank-off is the compressed no-bank path rather than a full-history baseline.

### 5.2 P7 Versus P6

P6 obtains EM `0.169±0.018`, recall `0.258±0.016`, and
`165.8±19.8` format failures. P7 obtains EM `0.197±0.020`, recall
`0.254±0.028`, and `121.4±8.8` format failures. Relative to P6, P7 changes
EM by `+0.0280`, recall by `-0.0044`, and format failures by `-44.4`. P7 is
therefore selected as the paper-facing configuration because it improves EM
and output validity while retaining comparable recall. This comparison does
not show that the higher update threshold is universally preferable; it
reports the observed trade-off under the fixed EventQA protocol.

### 5.3 Explicit-Memory and Retrieval Controls

Text summaries and retrieved source text provide important controls because
they expose readable evidence rather than latent state. The matched-budget
condition further tests whether the observed benefit can be explained by any
additional query-time evidence channel. The same-model text-summary baseline
obtains EM `0.012`, recall `0.078`, and 267 format failures. BM25 top-2
retrieved text obtains EM `0.030`, recall `0.226`, and 265 format failures. The
16-token matched-budget condition obtains EM `0.068`, recall `0.180`, and 347
format failures. All three explicit-memory controls remain below P7 on both EM
and recall. The matched-budget result shows that exposing exactly 16 visible
retrieved tokens is not sufficient to reproduce the latent-bank gain.

### 5.4 Query-Retrieval Ablation

Comparing a complete P7 bank with Bank-off alone cannot isolate query-time
retrieval from construction-side effects. The no-query-retrieval ablation
addresses this gap by preserving identical P7 bank construction while
disabling latent retrieval at question time and blocking
query writes. Under this condition, performance collapses exactly to the
Disabled line: EM `0.008`, recall `0.178`, and 377 format failures. The bank
snapshot remains preserved, while retrieval output is empty for every question.
The EventQA gain therefore depends on query-time latent retrieval rather than
construction alone.

### 5.5 Evidence Summary

Across the repeated main rows and one-pass controls, four conclusions are
supported under the EventQA protocol. P7 improves EM and recall over Disabled.
It improves EM and format validity over P6 while retaining comparable recall.
It exceeds the explicit-memory controls on both metrics. Its gain disappears
when query-time retrieval is disabled. The cost measurements support
a narrower conclusion: P7 incurs modest overhead over Disabled, but the
evidence does not establish blanket cost superiority over explicit-text
alternatives.

| Method | End-to-end (s) | Seconds/question | Peak incremental GPU memory |
|---|---:|---:|---:|
| Disabled / compressed Bank-off | 367.448 | 0.735 | 143 MiB |
| BM25 top-2 retrieved text | 692.845 | 1.386 | 3.51 GiB |
| 16-token matched-budget retrieved text | 501.761 | 1.004 | 171 MiB |
| P7 with query retrieval disabled | 445.004 | 0.890 | 143 MiB |
| Frozen P7 non-strict | 387.999 | 0.776 | 172 MiB |

**Table 2 | Method-separable EventQA inference cost.** Measurements use
serialized same-GPU full passes and exclude model loading. The text-summary
row is omitted because its run was collected under shared-GPU contention.

## 6. Analysis

### 6.1 Context-Wise Behavior

The aggregate P7 improvement is not uniform. Context 4 remains a severe
failure: across five repeats, P7 obtains EM `0.006`, recall `0.228`, and
`93.8/100` format failures. The gap between recall and EM indicates that some
outputs contain relevant content without satisfying the answer contract, but
the low recall also shows that formatting is not the sole problem.

The context-wise comparison also shows that the overall gain is driven mainly
by contexts 0, 1, and 2. P7 improves over Bank-off by `+0.248`, `+0.382`, and
`+0.222` EM on these three contexts, respectively, and remains above Bank-off
on context 3 (`+0.096` EM) despite underperforming P6 there. Context 4 is the
clear failure case: P7 is slightly below Bank-off on EM (`0.006` vs `0.010`)
and below P6 on both EM and recall.

| Context | Bank-off EM | Bank-off Recall | P6 EM | P6 Recall | P7 EM | P7 Recall | P7 Format Failures | P7-Bank-off EM | P7-P6 EM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ctx0 | 0.000 | 0.150 | 0.194±0.073 | 0.236±0.018 | 0.248±0.020 | 0.258±0.016 | 3.6±2.9 | +0.248 | +0.054 |
| ctx1 | 0.000 | 0.250 | 0.264±0.086 | 0.388±0.019 | 0.382±0.065 | 0.404±0.035 | 8.2±5.6 | +0.382 | +0.118 |
| ctx2 | 0.000 | 0.150 | 0.180±0.048 | 0.202±0.037 | 0.222±0.017 | 0.234±0.012 | 5.2±1.7 | +0.222 | +0.042 |
| ctx3 | 0.030 | 0.150 | 0.150±0.071 | 0.198±0.020 | 0.126±0.054 | 0.144±0.046 | 10.6±2.4 | +0.096 | -0.024 |
| ctx4 | 0.010 | 0.190 | 0.056±0.068 | 0.266±0.063 | 0.006±0.012 | 0.228±0.098 | 93.8±5.7 | -0.004 | -0.050 |

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
coexist with memory-induced regressions. P7 shows more helpful transitions than
P6 (`98.4±10.1` vs `83.6±9.4`) in the repeated aggregate. Harmful counts remain
small for both methods (`4.0±0.0` vs `3.2±0.7`). The corresponding net gain is
therefore larger for P7 (`94.4±10.1`) than for P6 (`80.4±8.9`). At the same
time, P7 also shows fewer format-harm cases (`28.4±9.3`) than P6
(`44.6±9.0`). Thus, format cleanup alone does not explain the P7 advantage.

| Method | Repeats | Helpful | Harmful | Unchanged | Format-harm | Net gain |
|---|---:|---:|---:|---:|---:|---:|
| P6 non-strict | 5 | 83.6±9.4 | 3.2±0.7 | 413.2±9.9 | 44.6±9.0 | 80.4±8.9 |
| Frozen P7 non-strict | 5 | 98.4±10.1 | 4.0±0.0 | 397.6±10.1 | 28.4±9.3 | 94.4±10.1 |

Here, unchanged is the residual count over 500 paired questions after removing
helpful and harmful transitions, while format-harm is a diagnostic subset
rather than an additional partition bucket.

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

The evidence supports a result on EventQA-65536 under a
MemoryAgentBench-compatible frozen-bank contract. It does not establish general
long-context improvement, and EventQA's visible prefix and six candidate answers make it
less demanding than open-ended factual reconstruction. The evaluation includes
text-summary, BM25, matched-budget, and no-query-retrieval controls. Most
control rows are single deterministic passes rather than five-repeat estimates.
The text-summary cost row is excluded from Table 2 because it was measured
under shared-GPU contention.

Performance is heterogeneous across contexts. Context 4 nearly collapses under
P7, and the harmful-tuple analysis is an oracle single-bank diagnostic rather
than part of the method. The fixed 16-slot capacity and retrieval policy may
also discard or combine evidence in ways that are not captured by aggregate
metrics.

LoCoMo-QA provides a separate boundary test. On a paired two-conversation,
304-question diagnostic, both Disabled and P7 obtain zero EM; token F1 is
`0.01834` and `0.02084`, respectively, and every paired question is
exact-match wrong. Although P7 retrieval is active and query writes remain
zero, it produces 138 no-context denials and 153 refusals. These results are not
positive evidence for the paper's long-context reasoning claim. Instead, they
suggest that latent-only memory is inadequate for exact conversational fact
recovery under the evaluated prompt and evidence contract.
Appendix A records the protocol, reliable mechanism counters, and representative
failure cases; it excludes the known-unreliable construction-side cost and
Trigger/Weaver counters.

## 8. Conclusion

We introduced an inference-time, session-local latent memory bank that reuses
Weaver-space memories while keeping MemGen's Trigger, Weaver, and Reasoner
frozen. Under the EventQA-65536 frozen-bank protocol, frozen P7 improves exact
match and recall over the compressed Bank-off path and improves the EM/format
trade-off over P6. It also outperforms the completed explicit-memory controls
under the EventQA contract, while the no-query-retrieval ablation shows
that the gain depends on active query-time latent retrieval. Together, these
results support the value of persistent latent reuse for this closed-set
long-context event-reasoning setting.

The same evidence does not justify benchmark-general performance claims.
Context-wise analysis shows that the gain is driven mainly by contexts 0-2,
with context 4 remaining a clear failure case. The cost measurements also show
that P7 adds measurable overhead over Disabled and therefore should not
be presented as a cost-superior method. Future work should extend the
comparison to additional benchmarks, improve harmful-memory control, and
develop stronger latent-to-fact decoding.

## Appendix A. LoCoMo-QA Boundary Diagnostic

### A.1 Scope and Prompt Contract

This diagnostic uses two LoCoMo conversations (`conv-26` and `conv-30`) and all
304 associated QA rows. Conversation construction is session-granular: each
conversation contributes 19 ordered session chunks. Disabled and P7 use the
same construction and question instructions. The active prompts are:

```text
System: You are a helpful assistant that can read the context and memorize it
for future retrieval.

Construction: Please memorize the following conversation chunk (i/n) for
future question answering.

{chunk}

Question: Based on the conversation history you memorized, answer the question
concisely.

Question: {question}

Answer:
```

At question time, both methods receive only the system message and current
question instruction as visible text. The earlier conversation and
acknowledgement turns are not included in the visible prompt. P7 additionally
has access to the frozen latent bank; Disabled does not.

### A.2 Reliable Protocol and Outcome Fields

| Group | Field | Disabled | Frozen P7 | Interpretation |
|---|---|---:|---:|---|
| Scope | Conversations / questions | 2 / 304 | 2 / 304 | Paired evaluation |
| Construction | Session chunks per conversation | 19 | 19 | Same session granularity |
| Construction | Bank writes | 0 | 19 | P7 construction is active |
| Construction | Bank retrievals | 0 | 18 | First chunk sees an empty bank |
| Construction | Final slots | 0 | 16 | P7 reaches its capacity bound |
| Query | Retrieved latent count | 0 | 16 | P7 query retrieval is active |
| Query | Nonzero query-write rows | 0 | 0 | Frozen-bank invariant holds |
| Query | Changed-snapshot rows | 0 | 0 | Questions do not mutate the bank |
| Outcome | Exact match | 0.0000 | 0.0000 | All 304 pairs remain EM-wrong |
| Outcome | Token F1 | 0.01834 | 0.02084 | Small overlap change, not task success |
| Output | Invalid outputs | 15 | 0 | P7 improves output validity only |
| Output | No-context denials | 0 | 138 | P7 often fails to use retrieved state |
| Output | Refusals | 13 | 153 | Refusal dominates many P7 answers |
| Output | Short nonanswers | 0 | 39 | Additional latent-utilization failure |
| Output | Concrete wrong answers | 75 | 87 | Retrieval does not prevent hallucination |

The output flags are deterministic heuristic diagnostics over saved prediction
strings. Refusal and no-context-denial categories can overlap and therefore
must not be summed as disjoint outcomes. Construction latency, construction
peak GPU memory, and construction Trigger/Weaver counts are omitted because
their logging path is disconnected from the preserved bank snapshot and is not
paper-facing reliable.

### A.3 Representative Paired Cases

| Case | Question | Gold | Disabled | Frozen P7 | Diagnostic |
|---|---|---|---|---|---|
| No-context denial (`conv-26::q000`) | When did Caroline go to the LGBTQ support group? | 7 May 2023 | Empty extracted answer | States that no conversation history was provided | Retrieval active, but latent evidence is not consumed |
| Short nonanswer (`conv-26::q004`) | What is Caroline's identity? | Transgender woman | “Caroline's identity is Caroline.” | “无” | Neither method recovers the fact; P7 emits a minimal nonanswer |
| Partially useful (`conv-30::q061`) | What did Gina receive from a dance contest? | a trophy | “without” | “Gina received a trophy from a dance contest.” | P7 token F1 is 0.4 but EM remains zero under the scorer |
| Disabled better by F1 (`conv-26::q090`) | How long have Mel and her husband been married? | 5 years | Malformed text with partial overlap | “称”, followed by a no-context refusal | P7 retrieval does not guarantee usable decoding |

These examples support a failure boundary rather than a positive benchmark
claim. The evaluation pipeline successfully constructs, freezes, and retrieves from the P7
bank, but the Reasoner frequently denies the existence of context or produces
an unusable answer. This separates mechanical retrieval from successful
latent-to-fact decoding and motivates explicit-text sanity checks before any
larger LoCoMo evaluation.
