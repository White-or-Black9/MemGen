## Title

# Inference-Time Latent Memory Management for Long-Horizon LLM Agents

## Abstract

Generated latent memories can encode useful intermediate reasoning in LLM agents, but they are typically transient: once produced, there is no persistent mechanism for deciding which representations should remain available and be reused in later reasoning. We introduce an inference-time, session-local latent memory bank that turns these transient representations into bounded, session-owned memory without updating model parameters. The bank stores generated latent memories, retrieves relevant prior memories to support subsequent generation, and manages them through structured updates and capacity-aware replacement. On the EventQA task in MemoryAgentBench, the latent memory manager improves exact match and recall, reduces parser-format failures, and outperforms the evaluated explicit textual-memory controls, while adding only modest overhead relative to the matched no-bank path. These findings show that persistent, session-local reuse of generated latent memories can provide an effective alternative to repeatedly exposing full textual histories for long-context event reasoning.



## Introduction

Large language model (LLM)-based agents are evolving from single-turn text generators into interactive systems that can reason, use tools, maintain memory, and make decisions over extended interactions. A common abstraction decomposes their capabilities into reasoning or planning, memory, and tool use. Reasoning supports problem decomposition and decision making, memory preserves prior observations, task states, and experiences, and tool use connects the agent to external environments, APIs, databases, and other functional modules [@zhao2023llm_agent_survey; @wang2024survey_autonomous_agents; @zhang2024survey_memory_llm_agents].

These components, however, are not independent. Both reasoning and tool use are fundamentally grounded in memory. Humans rarely solve a problem entirely from scratch; instead, they recall similar experiences, methods, tools, and previous failures while reasoning about the current situation. If a person used a tool yesterday but cannot remember either its existence or how to operate it today, the tool can no longer help solve the problem. The same limitation applies to LLM agents. An agent may have access to many tools, but without remembering when a tool should be invoked, which tool is appropriate, and how it should be used, tool availability alone does not guarantee successful task completion. From this perspective, tool use can be viewed as a form of procedural memory: the agent retrieves a relevant procedure and applies it to the current problem. Memory is therefore not merely a passive store of facts, but a foundational mechanism supporting reasoning, tool selection, task continuity, and long-horizon adaptation.

Memory representations in LLM agents can be broadly divided into explicit and latent forms. Explicit memory preserves information as natural-language text, structured records, or retrieved documents. Such representations are interpretable, editable, and easy to inspect, but they consume context capacity whenever they are presented to the model and are often only loosely coupled with its internal reasoning process. This limitation becomes particularly important in long-context settings. A complete interaction history may be too large to include at every inference step. Moreover, even when a model processes the history once, the useful intermediate representations produced during that computation are typically transient. Later queries must therefore recompute them from the original context or rely on another representation of the past.

Latent-memory methods offer one route to persistent internal state. Transformer-XL and Compressive Transformer reuse or compress hidden states across segments [@dai2019transformerxl; @rae2019compressive]. Memorizing Transformers and CAMELoT retrieve stored internal activations or associative memories [@wu2022memorizing; @he2024camelot]. MEMORYLLM and Titans introduce learned latent-memory mechanisms that maintain or update internal memory during inference or test-time adaptation [@wang2024memoryllm; @behrouz2025titans]. Together, these approaches establish that internal representations can support information reuse beyond a single local context.

This work examines a complementary operational problem: how should an agent manage latent memory tokens that are dynamically generated during reasoning? MemGen synthesizes latent memory tokens through a Trigger–Weaver–Reasoner pathway [@zhang2025memgen]. However, transient generation alone does not provide an explicitly managed store through which these tokens can persist and be selectively reused. We therefore study generated latent token memory management: how generated memories should be stored, retrieved, updated, replaced, reset, and assigned to a bounded inference session.

We introduce an inference-time, session-local latent memory bank for MemGen-style agents. The bank stores Weaver-generated memories while the agent processes a context and retrieves selected memories for later queries. Trigger, Weaver, and Reasoner remain frozen, and no cross-session memory is introduced. We evaluate the mechanism on EventQA-65536 under a MemoryAgentBench-compatible protocol [@hu2026evaluating], where a single context-owned bank is constructed before question answering and reused across the associated questions.

The paper makes three contributions:

1. We introduce an inference-time latent-memory management mechanism for MemGen-style LLM agents.
2. We design a session-local latent memory bank with explicit write, retrieval, update, replacement, and reset operations.
3. We evaluate the mechanism on long-context reasoning through repeated EventQA experiments, controlled comparisons, failure analyses, and analyses of internal memory behavior.




## 2. Related Work

### 2.1 Memory Representations and Management in LLM Agents

Memory is commonly treated as a core component of LLM agents alongside
planning and tool use [@zhao2023llm_agent_survey;
@wang2024survey_autonomous_agents]. Existing systems differ in both their
representational substrate and their management policy
[@zhang2024survey_memory_llm_agents]. Explicit memory stores text, structured
records, summaries, or retrieved documents. These forms are inspectable and
can preserve source-level evidence, but they occupy context space when exposed
to the model. Latent memory instead retains internal activations, learned
states, or memory tokens. It is more tightly coupled to model computation, but
its content is harder to inspect and verify.

This distinction does not by itself define a complete memory system. A
long-horizon agent also needs rules for writing, selecting, updating,
replacing, and clearing stored information. Our work focuses on these lifecycle
operations for latent memories generated by an existing model. The contribution
is therefore a management layer over generated latent state, rather than a new
taxonomy of agent memory.

### 2.2 Recurrent and Retrieved Latent State

Several architectures preserve hidden state beyond a local segment.
Transformer-XL reuses cached hidden states through segment-level recurrence
[@dai2019transformerxl], while Compressive Transformer retains older states in
a compressed form [@rae2019compressive]. Recurrent Memory Transformer passes
dedicated memory tokens between segments [@bulatov2022rmt], and
Infini-attention integrates compressive memory into the attention mechanism
[@munkhdalai2024infini]. These methods embed persistence into the model's
recurrent computation and typically require architecture-specific training.

Other approaches retrieve stored internal representations. Memorizing
Transformers performs approximate nearest-neighbour lookup over a
non-differentiable store of internal key-value pairs [@wu2022memorizing].
CAMELoT attaches a training-free consolidated associative memory to a frozen
language model [@he2024camelot]. MEMORYLLM maintains a self-updatable latent
memory pool [@wang2024memoryllm], and M+ combines latent memory with a trained
retriever for longer retention [@wang2025mplus]. Titans instead learns a neural
long-term memory that is updated at test time [@behrouz2025titans].

Our method shares the goal of persistent internal state but differs in its
integration point. It does not add a new recurrent architecture or train a new
memory module. It manages latent sequences already produced by the frozen
MemGen pathway within a bounded session-owned store.

### 2.3 Generated Latent Memory in MemGen

MemGen dynamically generates machine-native memory during reasoning
[@zhang2025memgen]. A memory Trigger decides when augmentation is needed, and a
Weaver constructs a latent token sequence that supports the Reasoner. This
design makes memory generation conditional on the current reasoning state.

We build on this mechanism rather than replace it. Our question begins after a
Weaver output has been generated. We ask who owns that memory, how long it
should remain available, and which memories a later query should retrieve. We
also define how a bounded store updates or replaces its contents. The proposed
bank makes these lifecycle semantics explicit. It resets state between sessions
and blocks writes during frozen-bank question answering. This operational focus
distinguishes our work from methods whose main contribution is the generation
or training of latent memory itself.

### 2.4 Explicit Memory and Long-Horizon Evaluation

Explicit-memory systems provide an important comparison because they retain
readable evidence. Retrieval-augmented generation supplies selected documents
to a parametric model [@lewis2020rag], while MemLLM trains a model to interact
with an explicit read-write memory [@modarressi2024memllm]. MemGPT manages
multiple memory tiers as virtual context for long-running interactions
[@packer2023memgpt]. In our EventQA evaluation, same-model summaries, BM25
retrieval, and a matched-token-budget condition test whether readable memory or
additional query-time text explains the observed effect.

Long-horizon memory benchmarks also impose different evidence contracts.
LongMemEval tests sustained conversational memory, including temporal
reasoning, knowledge updates, and abstention [@wu2024longmemeval]. LoCoMo
evaluates question answering, summarization, and dialogue generation over long
conversations [@maharana2024locomo]. MemoryAgentBench instead evaluates several
memory competencies through incremental interactions and includes EventQA as
an accurate-retrieval task [@hu2026evaluating].


## 3. Method

### 3.1 Overview

MemGen generates latent memories through a Trigger–Weaver–Reasoner pathway [@zhang2025memgen]. During inference, the Trigger monitors the current reasoning state and determines whether latent-memory augmentation should be invoked. When augmentation is enabled, the Weaver transforms the current hidden state into a latent memory, which is subsequently injected into the Reasoner.

In the original inference pathway, a generated latent memory is transient and is not explicitly retained for later reasoning steps. We extend this pathway with session-local latent memory bank. The bank stores Weaver-generated latent memories from earlier context-processing steps, retrieves relevant memories according to the current reasoning state, and feeds the retrieved memories back into the Weaver. The Weaver therefore conditions on both the current hidden state and previously generated latent memories before producing a new latent memory for the Reasoner.

The resulting method changes the inference flow without modifying the parameters of the Trigger, Weaver, or Reasoner. It introduces no parameter updates, additional training objective, or cross-session memory. All persistent memory state is confined to the current inference session and is reset before processing an independent context.

### 3.2 Session-Local Latent Memory Bank

For an input context, let $H_t$ denote the Reasoner’s hidden state at inference step $t$. When the Trigger emits an invocation signal, the memory bank is queried using a representation derived from the recent reasoning state.

The query representation is obtained by mean-pooling the most recent $L$ hidden states:

$$
q_t = \operatorname{MeanPool}\left(H_t[-L:]\right).
$$

The memory bank at step $t$ is denoted by

$$
\mathcal{M}_t =
\left\{
\left(m_i,k_i,r_i^{\mathrm{last}}\right)
\right\}_{i=1}^{N_t},
$$

where $m_i$ is a detached Weaver-space latent memory, $k_i$ is its retrieval key, and $r_i^{\mathrm{last}}$ records the global retrieval count at which memory $m_i$ was most recently retrieved.

Each retrieval key is obtained by mean-pooling the latent-token vectors of the corresponding memory:

$$
k_i = \operatorname{MeanPool}(m_i).
$$

Before storage, each generated latent memory is detached from the computation graph and transferred to memory:

$$
m_i^{\mathrm{store}} =
{Detach}(m_i)
$$

The bank belongs to exactly one context or inference session. It has no global registry, does not share slots across contexts, and is reset before the next independent sample. Its capacity is bounded by a configurable maximum number of slots $C$. In the evaluated configuration, $C=16$, and each slot stores one Weaver-generated memory consisting of eight latent-token vectors.

### 3.3 Similarity-Based Retrieval with Temporal Decay

For each stored memory, the bank computes a retrieval score by combining cosine similarity with temporal decay:

$$
s_i(q_t) =
\operatorname{cosine}(q_t,k_i)
\exp\left(-\alpha \Delta r_i\right),
$$

where

$$
\Delta r_i =
r_t-r_i^{\mathrm{last}}.
$$

Here, $r_t$ denotes the current global retrieval count, and $r_i^{\mathrm{last}}$ denotes the retrieval count at which memory $m_i$ was
most recently selected. Accordingly, $\Delta r_i$ measures the retrieval
inactivity of memory $m_i$, namely, the number of retrieval operations that
have occurred since it was last retrieved. The term
$\exp(-\alpha \Delta r_i)$ represents the corresponding temporal-decay weight:
memories that have remained unretrieved for longer receive smaller weights.
The coefficient $\alpha$ controls the strength of this decay.

The retrieval procedure first filters memories using a retrieval threshold $\tau_r$:

$$
\mathcal{C}_t =
\left\{
m_i\in\mathcal{M}_t
\mid
s_i(q_t)\geq\tau_r
\right\}.
$$

Among the surviving candidates, the bank returns at most the $K$ highest-scoring memories:

$$
R_t =
\operatorname{TopK}\left(\mathcal{C}_t,K\right).
$$

If no memory passes the retrieval threshold, the bank returns an empty retrieval set:

$$
R_t=\varnothing.
$$

The method does not force a top-1 result when every stored memory falls below the threshold. Whenever a memory is successfully retrieved, its last-retrieval count is updated to the current global retrieval count:

$$
r_i^{\mathrm{last}}\leftarrow r_t,
\qquad
m_i\in R_t.
$$

In the evaluated configuration, we set

$$
\tau_r=0.05,
\qquad
K=2,
\qquad
\alpha=0.05.
$$

These values remain fixed across all evaluated EventQA contexts.

### 3.4 Retrieved-Memory-Conditioned Latent Generation

The central difference from the original transient MemGen pathway is that retrieved memories are returned to the Weaver rather than being directly injected into the Reasoner.

At inference step $t$, the Trigger first evaluates the current reasoning state:

$$
g_t=\operatorname{Trigger}(H_t),
$$

where $g_t\in\{\texttt{INVOKE},\texttt{SKIP}\}$.

When the Trigger emits `INVOKE`, the memory bank retrieves $R_t$ using the current reasoning-state query. The Weaver then conditions on both the current hidden state and the retrieved latent memories:

$$
m_t =
\operatorname{Weaver}(H_t,R_t).
$$

If no prior memory passes the retrieval threshold, the Weaver follows the original generation path:

$$
m_t =
\operatorname{Weaver}(H_t,\varnothing).
$$

The Weaver consolidates the current reasoning state and the retrieved latent memories into a newly generated latent memory $m_t$. This new latent memory is then passed through the existing Weaver-to-Reasoner pathway:

$$
y_t =
\operatorname{Reasoner}(H_t,m_t).
$$

When the Trigger emits `SKIP`, no latent-memory augmentation is performed:

$$
y_t =
\operatorname{Reasoner}(H_t).
$$

This design avoids directly injecting all retrieved memories into the Reasoner. Instead, the frozen Weaver transforms the current hidden state together with the retrieved latent support into a new latent memory. This intermediate Weaver transformation integrates the retrieved memories before they are used by the Reasoner, reducing potential interference while preserving the original MemGen latent-generation pathway.

The method therefore reuses generated latent-token memories without decoding them into natural-language text or exposing their original source context again.

### 3.5 Memory Write and Thread Update

A newly generated latent memory $m_t$ is written to the bank whenever the Weaver is invoked. The write policy determines whether the candidate represents a new latent-memory thread or should refresh an existing one.

The key of the new memory is

$$
k_t =
\operatorname{MeanPool}(m_t).
$$

If the bank is empty, the new memory is inserted directly:

$$
\mathcal{M}_{t+1} =
\left\{
\left(m_t,k_t,r_t\right)
\right\}.
$$

Otherwise, the bank identifies the stored memory with the greatest similarity to the new candidate:

$$
j =
\arg\max_i
\operatorname{cosine}(k_t,k_i).
$$

Let

$$
u_t =
\max_i
\operatorname{cosine}(k_t,k_i)
$$

denote the maximum candidate-to-memory similarity. The bank uses a update threshold $\tau_u$.

If the maximum similarity is lower than the update threshold, the candidate is treated as a distinct latent-memory thread and inserted as a new slot:

$$
u_t<\tau_u
\quad\Rightarrow\quad
\mathcal{M}_{t+1} =
\mathcal{M}_t
\cup
\left\{
\left(m_t,k_t,r_t\right)
\right\}.
$$

If the maximum similarity reaches or exceeds the update threshold, the most similar slot is refreshed by replacing its stored latent memory:

$$
u_t\geq\tau_u
\quad\Rightarrow\quad
\mathcal{M}_{t+1} =
\left(
\mathcal{M}_t
\setminus
\left\{
\left(m_j,k_j,r_j^{\mathrm{last}}\right)
\right\}
\right)
\cup
\left\{
\left(m_t,k_t,r_t\right)
\right\}.
$$

Thus, sufficiently dissimilar generated memories create new slots, whereas similar generated memories replace and refresh an existing latent-memory thread. In the evaluated configuration, the update threshold is fixed to

$$
\tau_u=0.10.
$$

The retrieval threshold $\tau_r$ and update threshold $\tau_u$ serve different purposes. The retrieval threshold determines whether an existing memory is sufficiently relevant to the current reasoning state, whereas the update threshold determines whether a newly generated memory should replace an existing thread or create a new one.

### 3.6 Bounded-Capacity Replacement

The memory bank maintains a fixed capacity $C$:

$$
|\mathcal{M}_t| \leq C.
$$

When inserting a new memory thread would exceed this capacity, the bank first
selects an existing slot for eviction. For each slot $i$, we define its
retrieval age as

$$
a_i = r_t - r_i^{\mathrm{last}},
$$

where $r_t$ is the current global retrieval count and
$r_i^{\mathrm{last}}$ is updated only when slot $i$ is selected during
retrieval. A larger $a_i$ therefore indicates that the memory has remained
unretrieved for more retrieval operations.

The bank first identifies the maximum retrieval age:

$$
a_{\max} = \max_i a_i.
$$



The newly generated memory then replaces the selected slot:

$$
(m_e,k_e,r_e^{\mathrm{last}})
\leftarrow
(m_t,k_t,r_t).
$$

This deterministic policy preferentially preserves recently retrieved latent
memories rather than simply retaining the most recently written slots.
### 3.7 End-to-End Inference Procedure

The complete inference procedure is summarized below.

```text
Input:
    hidden states H_1, ..., H_T
    retrieval threshold τ_r
    update threshold τ_u
    temporal-decay coefficient α
    maximum retrieval count K
    memory capacity C
Initialize:
    M ← ∅
    retrieval_count ← 0
for t = 1, ..., T:
    signal ← Trigger(H_t)
    if signal == INVOKE:
        q_t ← MeanPool(H_t[-L:])
        retrieval_count ← retrieval_count + 1
        for each memory m_i in M:
            k_i ← MeanPool(m_i)
            Δr_i ← retrieval_count - m_i.last_retrieved
            s_i ← cosine(q_t, k_i) · exp(-αΔr_i)
        R_t ← TopK({m_i : s_i ≥ τ_r}, K)
        for each retrieved memory m_i in R_t:
            m_i.last_retrieved ← retrieval_count
        m_t ← Weaver(H_t, R_t)
        y_t ← Reasoner(H_t, m_t)
        k_t ← MeanPool(m_t)
        if M is empty:
            insert detach(m_t).cpu() into M
        else:
            j ← argmax_i cosine(k_t, k_i)
            u_t ← cosine(k_t, k_j)
            if u_t ≥ τ_u:
                replace m_j with detach(m_t).cpu()
            else:
                if |M| = C:
                    evict one slot using the bounded replacement policy
                insert detach(m_t).cpu() into M
    else:
        y_t ← Reasoner(H_t)
```




## 4. Experiments

### 4.1 Experimental Setting

We evaluate on EventQA [@hu2026evaluating], which tests event-centric question answering over long contexts.  Each evaluation run contains five contexts, with 100 questions per context, for a total of 500 questions.  We retain the benchmark's original context partitioning and evaluation protocol.

For the latent-memory condition, the memory bank is constructed while processing each context and then frozen before the corresponding questions are answered.  The question-answering phase is read-only: it can retrieve from the frozen bank but cannot create, update, replace, or evict memory slots.  This protocol isolates the contribution of inference-time memory construction from query-time adaptation.

We use the default EventQA-style query prompt and the unchanged local benchmark scoring path.  In particular, we do not rewrite prompts, repair parser outputs, normalize answers against candidates, or apply output post-processing.  We report exact-match accuracy (EM), answer recall, and the number of format failures produced by the official parser.

Unless otherwise stated, the latent manager uses a maximum bank capacity of 16 slots, retrieval threshold $0.05$, update threshold $0.10$, retrieval top-$k=2$, temporal decay $0.05$, and 40 newly generated tokens per answer.  Bank construction and query-time retrieval operate in the Weaver latent space, while the persistent bank is stored on CPU. The latent manager and the rolling-summary, BM25, and matched-budget controls are each evaluated over five complete process-level passes with the fixed base seed and per-context reseeding. The MemGen recent-text baseline is also evaluated over five complete process-level passes with base seeds 42, 142, 242, 342, and 442 and per-context reseeding. 

For the efficiency analysis, every reported method is timed in a separate serialized run on the same GPU, each covering all 500 questions.  We exclude model-loading time and report end-to-end wall-clock time, amortized time per question, and peak GPU-memory allocation.  


### 4.2 Comparison Methods

**MemGen recent-text baseline (capacity-max).** To compare against original
MemGen inference without the persistent Bank, we use
the same MemGen checkpoint with the bank disabled and provide raw textual
history directly in the prompt. Complete EventQA histories exceed the model’s 32,768-token context capacity. We therefore use a capacity-limited proxy for full-history conditioning rather than claiming an official full-history MemGen baseline. For each question, we prepend the longest chronologically ordered suffix of the event history that fits within the input budget, followed by the unchanged EventQA query template. This baseline tests whether original-MemGen recent
textual context can substitute for the persistent latent bank.

**Rolling text summary.** This same-model explicit-memory control processes the
EventQA context sequentially. For each context chunk, the generator receives
the previous summary and the new chunk, greedily produces an updated summary,
and persists at most 128 tokenizer tokens. After construction, the final
summary is prepended to every question for that context; it is not updated
during question answering, and the latent bank is disabled throughout. This is
a recursive-summarization control rather than a reproduction of the raw-text
rolling-window baseline used in Re3 [@yang2022re3]. It tests whether compact
natural-language history can perform well than our method.

**BM25 retrieval.** Following Okapi BM25 [@robertson1994okapi], we construct a
per-context lexical index over the EventQA chunks using lower-cased
alphanumeric terms, $k_1=1.5$, and $b=0.75$. Each question retrieves the two
highest-scoring chunks (ties broken by chunk order), and their full text is
prepended to the unchanged EventQA query prompt. The same generator answers
the resulting prompt with its latent bank disabled. This baseline provides a
standard sparse-retrieval reference with access to explicit source text.

**Matched-16 BM25 retrieval.** To control the visible-text budget, this variant
uses the same top-two BM25 chunks but selects one query-overlap-scored
eight-token window from each. The two windows are injected only when their
rendered prompt adds exactly 16 tokens; otherwise the run is rejected. It
therefore separates lexical retrieval from simply exposing a larger amount of
source text.

All explicit-memory controls use the same base model, questions, default EventQA prompt, official parser and scorer, and 40-token generation budget as the latent manager condition.



### 4.3 Ablation Study

We study whether the gain requires query-time access to the constructed latent bank.  The **no-query-retrieval** ablation preserves the same bank-construction procedure, capacity, update rule, and frozen snapshot as the latent-manager condition, but disables every query-time retrieval operation.  Consequently, no retrieved latent memory is injected into the Weaver state during answer generation.  This design distinguishes the existence of a constructed bank from its use at inference time.

Additional ablations will be added to this subsection as they become available; their outcomes are reported in Section 5.3.


