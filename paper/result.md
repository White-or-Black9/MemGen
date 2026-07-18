## 6. Results

### 6.1 Main Results

Table 1 compares the latent memory manager with the matched Bank-off baseline.
Across five repeated runs, the latent memory manager increases EM from $0.008$
to $0.197\pm0.020$ and recall from $0.178$ to $0.254\pm0.028$. It also
reduces format failures from $377.0$ to $121.4\pm8.8$. These results establish
that a frozen, context-local latent bank improves event reasoning relative to
the same compressed query path without persistent memory.

| Method | Repeats | EM | Recall | Format failures |
|---|---:|---:|---:|---:|
| Bank-off | 5 | 0.008$\pm$0.000 | 0.178$\pm$0.000 | 377.0$\pm$0.0 |
| Latent memory manager | 5 | **0.197$\pm$0.020** | **0.254$\pm$0.028** | **121.4$\pm$8.8** |

**Table 1 | Main EventQA-65536 results.** Values are mean $\pm$ population
standard deviation across five independent runs. Bank-off is a compressed
no-bank baseline rather than a full-history baseline.

### 6.2 Explicit-Memory Comparisons

We next compare the latent memory manager with explicit textual memory. All
three controls remain below the latent memory manager on both EM and recall.
BM25 top-2 achieves recall of 0.226 but EM of only 0.030, while the 16-token
matched-budget condition reaches EM of 0.068. Thus, the observed gain is not
reproduced by supplying a small amount of retrieved text at question time.

| Method | Repeats | EM | Recall | Format failures |
|---|---:|---:|---:|---:|
| Rolling text summary | 1 | 0.012 | 0.078 | 267.0 |
| BM25 top-2 retrieved text | 1 | 0.030 | 0.226 | 265.0 |
| Matched-budget retrieved text (16 tokens) | 1 | 0.068 | 0.180 | 347.0 |
| Latent memory manager | 5 | **0.197$\pm$0.020** | **0.254$\pm$0.028** | **121.4$\pm$8.8** |

**Table 2 | Comparison with explicit-memory controls.** The textual controls
are single deterministic complete passes and are reported as point estimates.

### 6.3 Ablation Results

We test whether the bank must be retrieved at question time. The ablation keeps
the same context-construction procedure, bank capacity, update policy, and
frozen-bank protocol as the latent memory manager. It disables retrieval for
every question while preserving the constructed bank.

Without query-time retrieval, EM falls to 0.008, recall to 0.178, and format
failures rise to 377, exactly matching the Bank-off effectiveness values. The
latent memory manager therefore requires query-time latent reuse; construction
of a bank alone does not produce the observed EventQA improvement. Additional
ablation results will be reported in this subsection.

| Method | EM | Recall | Format failures |
|---|---:|---:|---:|
| Bank-off | 0.008 | 0.178 | 377 |
| No-query-retrieval ablation | 0.008 | 0.178 | 377 |
| Latent memory manager | 0.197$\pm$0.020 | 0.254$\pm$0.028 | 121.4$\pm$8.8 |

**Table 3 | Query-time retrieval ablation.** The ablation is a single complete
pass. It constructs a 16-slot bank for every context but returns no retrieved
latent memories during question answering.

### 6.4 Efficiency Analysis

We measure Bank-off and the latent memory manager in separate serialized
processes on the same GPU over all five contexts and 500 questions, excluding
model loading. The latent memory manager requires 387.999 seconds end-to-end,
or 0.776 seconds per question, compared with 367.448 seconds and 0.735 seconds
per question for Bank-off. This corresponds to a 5.6% end-to-end overhead,
including 78.454 seconds of context-bank construction. Its peak incremental GPU
allocation is approximately 29 MiB higher than Bank-off.

For reference, the BM25 and matched-budget controls require 692.845 seconds
(1.386 seconds per question) and 501.761 seconds (1.004 seconds per question),
respectively. Their peak incremental GPU allocations are approximately 3.51 GiB
and 171 MiB. These measurements show that the latent memory manager remains
close to the matched no-bank cost while exceeding the completed explicit-memory
controls on EventQA. They do not establish universal latency or memory
superiority across models, hardware, or retrieval systems.

| Method | End-to-end (s) | Seconds/question | Peak incremental GPU memory |
|---|---:|---:|---:|
| Bank-off | 367.448 | 0.735 | 143 MiB |
| BM25 top-2 retrieved text | 692.845 | 1.386 | 3.51 GiB |
| Matched-budget retrieved text | 501.761 | 1.004 | 171 MiB |
| Latent memory manager | 387.999 | 0.776 | 172 MiB |

**Table 4 | Method-separable EventQA inference cost.** Measurements use
serialized same-GPU complete passes and exclude model loading. The
rolling-summary cost is omitted because it was collected under shared-GPU
contention.
