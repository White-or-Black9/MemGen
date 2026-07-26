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

We isolate three query-time roles of the latent bank while keeping the context-bank construction procedure and configuration, bank capacity, update policy, and frozen-bank evaluation protocol unchanged.

**No-query-retrieval.** This ablation disables every query-time bank retrieval
operation while preserving the constructed bank. The query-time Weaver is still invoked from the current Reasoner state, but it receives no retrieved historical latent support. This variant distinguishes the existence of a constructed bank from its actual use during answer generation.

**No retrieved-memory conditioning.** This variant preserves the same context-bank construction procedure and configuration, frozen-bank evaluation protocol, query-time retrieval computation, Trigger mechanism, and query-time Weaver invocation pathway as the full method.
It still computes and records the retrieved slots, but withholds their latent
vectors from the Weaver. The query-time latent memory is therefore generated
only from the current Reasoner state. This comparison tests whether the gain depends on retrieved historical latent support entering Weaver generation, rather than on bank construction, retrieval execution, or the additional query-time control flow.

**Direct latent injection.** This structural control preserves the frozen-bank evaluation protocol and query-conditioned similarity retrieval, but selects at most the highest-scoring threshold-qualified slot and bypasses query-time Weaver integration. The selected latent memory slot injected directly into the Reasoner.
It tests whether direct reuse of the most relevant historical latent can
substitute for integration with the current Reasoner state. Outcomes are
reported in Section 5.3.
