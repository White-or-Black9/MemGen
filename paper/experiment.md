## 4. Experiments

### 4.1 Experimental Setting

We evaluate on EventQA [@hu2026evaluating], which tests event-centric question answering over long contexts.  Each evaluation run contains five contexts, with 100 questions per context, for a total of 500 questions.  We retain the benchmark's original context partitioning and evaluation protocol.

For the latent-memory condition, the memory bank is constructed while processing each context and then frozen before the corresponding questions are answered.  The question-answering phase is read-only: it can retrieve from the frozen bank but cannot create, update, replace, or evict memory slots.  This protocol isolates the contribution of inference-time memory construction from query-time adaptation.

We use the default EventQA-style query prompt and the unchanged local benchmark scoring path.  In particular, we do not rewrite prompts, repair parser outputs, normalize answers against candidates, or apply output post-processing.  We report exact-match accuracy (EM), answer recall, and the number of format failures produced by the official parser.

Unless otherwise stated, the latent manager uses a maximum bank capacity of 16 slots, retrieval threshold $0.05$, update threshold $0.10$, retrieval top-$k=2$, temporal decay $0.05$, and 40 newly generated tokens per answer.  Bank construction and query-time retrieval operate in the Weaver latent space, while the persistent bank is stored on CPU.  The latent-manager and Bank-off conditions are each evaluated over five independent runs; we report the mean and population standard deviation.

For the efficiency analysis, the latent-manager and Bank-off conditions are timed in separate serialized runs on the same GPU, each covering all 500 questions.  We exclude model-loading time and report end-to-end wall-clock time, amortized time per question, and peak GPU-memory allocation.  We do not use the textual-baseline timing as primary cost evidence because those runs were affected by shared-GPU contention.

### 4.2 Comparison Methods

**Bank-off.**  This control uses the same compressed latent context and generator as the latent manager but disables the persistent memory bank.  It therefore measures the contribution of persistent latent memory rather than that of the underlying compressed context representation.

**Rolling text summary.**  This explicit-memory control maintains a textual summary with a 128-token budget.  It tests whether simply retaining a compact natural-language history can substitute for the latent bank.

**BM25 retrieval.**  We index context chunks with BM25 ($k_1=1.5$, $b=0.75$) and prepend the top two retrieved chunks to the query context.  This baseline provides a lexical retrieval reference with access to explicit source text.

**Matched-16 BM25 retrieval.**  To control for the amount of visible source text, this variant retrieves the same two BM25 chunks but retains only one eight-token window from each, yielding 16 visible source tokens in total.

All explicit-memory controls use the same base model, questions, default EventQA-style prompt, official parser and scorer, and 40-token generation budget as the latent-manager condition.  Their reported results are deterministic full-pass point estimates and are interpreted as diagnostic controls rather than repeated-run estimates.

### 4.3 Ablation Study

We study whether the gain requires query-time access to the constructed latent bank.  The **no-query-retrieval** ablation preserves the same bank-construction procedure, capacity, update rule, and frozen snapshot as the latent-manager condition, but disables every query-time retrieval operation.  Consequently, no retrieved latent memory is injected into the Weaver state during answer generation.  This design distinguishes the existence of a constructed bank from its use at inference time.

Additional ablations will be added to this subsection as they become available; their outcomes are reported in Section 6.3.
