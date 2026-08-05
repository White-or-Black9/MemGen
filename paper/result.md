## 5. Results

### 5.1 Main Results

Table 1 compares the latent memory manager with the capacity-max MemGen recent-text baseline. This capacity-limited proxy for original MemGen supplies 32,256 raw source tokens, but reaches only an EM of (0.034) and recall of (0.096) across five complete process-level runs with distinct base seeds. Because decoding is greedy, the baseline produces identical predictions in all five runs. Across the same five-seed evaluation, the latent memory manager reaches an EM of (0.188\pm0.047) and recall of (0.231\pm0.042), with (118.6\pm19.1) format failures compared with (211.0\pm0.0) for the recent-text baseline. Thus, although the latent method exhibits greater run-to-run variation, it consistently outperforms a capacity-max suffix of explicit recent text on both EM and recall while substantially reducing format failures. These results show that exposing the Reasoner to a large recent-text suffix does not reproduce the EventQA performance obtained with the frozen latent memory bank.

| Method | Repeats | EM | Recall | Format failures |
|---|---:|---:|---:|---:|
| MemGen recent-text baseline (32,256 tokens) | 5 | 0.034$\pm$0.000 | 0.096$\pm$0.000 | 211.0$\pm$0.0 |
| Latent memory manager  | 5 | **0.188$\pm$0.047** | **0.231$\pm$0.042** | **118.6$\pm$19.1** |

**Table 1 | Main EventQA-65536 results.** Values are mean $\pm$ population
standard deviation across five complete process-level runs with distinct base
seeds.

### 5.2  Comparisons

We next compare the latent memory manager with other explicit textual memory
controls. All four controls remain below the latent memory manager on EM.
BM25 top-2 achieves recall of 0.226 but EM of only 0.030. Dense
E5 top-2 reaches the same EM of 0.030, with slightly higher recall of 0.240
but more format failures (285 versus 265). The identical EM does not imply
identical retrieval or predictions: the two rankers select different source
chunks for many questions. Rather, it shows that replacing lexical ranking
with this fixed dense encoder does not reproduce the latent manager's EM gain
under the same top-$k$ full-text injection budget. The 16-token matched-budget
condition reaches EM of 0.068. Together with the capacity-max MemGen
recent-text comparison in Table 1, these controls show that the observed gain
is not reproduced by either retrieved textual memory at the evaluated budgets
or a recent-text suffix that nearly fills the model context.

| Method | Repeats | EM | Recall | Format failures |
|---|---:|---:|---:|---:|
| Rolling text summary | 5 | 0.012$\pm$0.000 | 0.078$\pm$0.000 | 267.0$\pm$0.0 |
| BM25 top-2 retrieved text | 5 | 0.030$\pm$0.000 | 0.226$\pm$0.000 | 265.0$\pm$0.0 |
| Dense E5 top-2 retrieved text | 5 | 0.030$\pm$0.000 | 0.240$\pm$0.000 | 285.0$\pm$0.0 |
| Matched-budget retrieved text (16 tokens) | 5 | 0.068$\pm$0.000 | 0.180$\pm$0.000 | 347.0$\pm$0.0 |
| Latent memory manager (P7) | 5 | **0.188$\pm$0.047** | **0.231$\pm$0.042** | **118.6$\pm$19.1** |

**Table 2 | Comparison with explicit-memory controls.** All four textual
controls use five complete process-level passes with aligned base seeds;
values are mean $\pm$ standard deviation. These are not independent-seed
estimates.

### 5.3 Ablation Results


Table 3 separates bank access, retrieved-memory consumption, Weaver-mediated integration, and temporal score weighting. The no-query-retrieval ablation obtains an EM of $0.008$, recall of $0.178$, and $377.0$ format failures, showing that constructing a bank alone is insufficient to produce the full-method improvement.

No retrieved-memory conditioning yields exactly the same aggregate results across all five complete passes. Although this variant still performs query-time retrieval, the retrieved latent vectors are withheld from the Weaver. The identical outcome therefore indicates that retrieval computation and associated state tracking do not improve generation unless the retrieved historical latent support enters the model-visible computation.

Top-1 direct latent injection improves EM to $0.047 \pm 0.008$ and reduces format failures to $354.6 \pm 21.3$, but recall remains close to the two negative controls at $0.186 \pm 0.009$. It also remains substantially below the full latent memory manager. These results indicate that directly injecting a single retrieved latent slot provides limited benefit, whereas conditioning query-time Weaver generation on retrieved historical latent support is necessary to obtain the full improvement.

Removing temporal decay ($\alpha=0$) yields EM of $0.163 \pm 0.047$, recall of $0.251 \pm 0.030$, and $189.0 \pm 52.8$ format failures. Relative to P7, the no-decay condition has lower EM and more format failures, despite slightly higher recall. Since setting $\alpha=0$ changes score-based routing during both bank construction and question answering, this result supports the end-to-end temporal weighting used by P7 but does not isolate a query-only decay effect.



| Method | Complete passes | EM | Recall | Format failures |
|---|---:|---:|---:|---:|
| No-query-retrieval | 5 | 0.008$\pm$0.000 | 0.178$\pm$0.000 | 377.0$\pm$0.0 |
| No retrieved-memory conditioning | 5 | 0.008$\pm$0.000 | 0.178$\pm$0.000 | 377.0$\pm$0.0 |
| Top-1 direct latent injection | 5 | 0.047$\pm$0.008 | 0.186$\pm$0.009 | 354.6$\pm$21.3 |
| No temporal decay ($\alpha=0$) | 5 | 0.163$\pm$0.047 | 0.251$\pm$0.030 | 189.0$\pm$52.8 |
| Latent memory manager (P7) | 5 | **0.188$\pm$0.047** | **0.231$\pm$0.042** | **118.6$\pm$19.1** |

**Table 3 | Latent memory-management ablations.** Values are mean $\pm$
population standard deviation over five complete 5-context, 500-question
process-level passes.













### 5.4 Efficiency Analysis

Table 4 reports three separate serialized complete-pass measurements on the same physical GPU, with model-loading time excluded. End-to-end time includes context-bank construction for the latent memory manager and the corresponding summary-construction or indexing stage for each textual control.

The latent memory manager requires $377.06 \pm 10.49$ seconds end-to-end, or $0.754 \pm 0.021$ seconds per question, with a maximum observed peak incremental GPU-memory allocation of $0.160$ GiB. In comparison, the capacity-max MemGen recent-text baseline requires $2688.90 \pm 9.50$ seconds and $13.094$ GiB. The latent memory manager is therefore approximately $7.1\times$ faster and reduces peak incremental GPU-memory allocation by a factor of approximately $82$. The high allocation of the recent-text baseline is consistent with repeating a 32,256-token prefill for every question; model-loading memory is excluded from this comparison.

Relative to the rolling-summary, BM25 top-2, Dense E5 top-2, and Matched-16 BM25 controls, the latent memory manager is approximately $1.6\times$, $1.7\times$, $2.0\times$, and $1.3\times$ faster, respectively. Dense E5 top-2 requires $739.36 \pm 5.32$ seconds end-to-end, including E5 window indexing and query-time embedding, and has the same $3.505$ GiB peak incremental GPU-memory allocation as BM25 top-2. Its larger latency therefore reflects the additional dense-encoding computation rather than a larger GPU prompt footprint. The latent manager's peak incremental allocation is nearly identical to that of Matched-16 BM25 ($0.160$ versus $0.159$ GiB), suggesting that their latency difference is driven primarily by query-time computation rather than peak incremental GPU-memory usage.

These measurements characterize inference costs under the evaluated EventQA protocol. They do not establish universal latency or memory superiority across models, hardware, or retrieval implementations.

| Method | End-to-end (s) | Seconds/question | Peak incremental GPU memory |
|---|---:|---:|---:|
| MemGen recent-text baseline | $2688.90 \pm 9.50$ | $5.378 \pm 0.019$ | 13.094 GiB |
| Rolling text summary | $598.41 \pm 16.34$ | $1.197 \pm 0.033$ | 1.782 GiB |
| BM25 top-2 retrieved text | $645.76 \pm 4.30$ | $1.292 \pm 0.009$ | 3.505 GiB |
| Dense E5 top-2 retrieved text | $739.36 \pm 5.32$ | $1.479 \pm 0.011$ | 3.505 GiB |
| Matched-16 BM25 retrieval | $497.87 \pm 18.68$ | $0.996 \pm 0.037$ | 0.159 GiB |
| Latent memory manager | **$377.06 \pm 10.49$** | **$0.754 \pm 0.021$** | 0.160 GiB |

**Table 4 | EventQA inference-cost comparison.** Timing values are the mean $\pm$ population standard deviation over three independently launched complete process-level runs, each covering five contexts and 500 questions. End-to-end time excludes model loading but includes memory construction or the corresponding textual-memory preparation stage. Peak memory is the maximum observed incremental GPU allocation relative to the loaded model.
