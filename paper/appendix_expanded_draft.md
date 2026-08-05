# Supplementary Material

## A. Reproducibility and Implementation Details

### A.1 Model and inference configuration

All results use the released MemGen checkpoint
`Kana-s/MemGen@269d9b1/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model`.
The Trigger, Weaver, and Reasoner are loaded from this checkpoint and remain
frozen throughout evaluation; the latent memory manager introduces neither
parameter updates nor auxiliary training. We use batch size one, BF16
inference, greedy decoding, and a maximum of 40 newly generated answer tokens.
The Weaver produces eight latent tokens per invocation. The bank stores one
eight-token Weaver-space latent memory per slot, detached from the computation
graph and moved to CPU storage.

The formal configuration is: 4,096-token construction chunks, bank capacity
$C=16$, query retrieval top-$k=2$, retrieval threshold
$\tau_r=0.05$, update threshold $\tau_u=0.10$, temporal-decay coefficient
$\alpha=0.05$, and a query vector formed by mean-pooling the most recent
$L=64$ Reasoner hidden states. Candidate slots are ranked by cosine similarity
multiplied by $\exp(-\alpha\Delta r)$, where $\Delta r$ is the number of
successful retrieval operations since the slot was last selected. A slot is
retrieved only when its score reaches $\tau_r$; the method does not force a
low-scoring top-1 result.

The recorded formal-P7 environment used Python 3.10.20, PyTorch 2.12.0 with
CUDA 12.6, Transformers 4.55.4, and an NVIDIA RTX A6000. These versions
document the evaluated implementation, not hardware portability of the timings.

### A.2 Two-phase EventQA protocol

Each EventQA pass contains five benchmark contexts and 100 questions per
context (500 questions total). We retain the benchmark context partitioning,
query template, parser, and scorer. Exact-match accuracy is the benchmark's
`substring_exact_match` metric; we additionally report answer recall and the
number of parser format failures. No answer repair, candidate normalization,
or output post-processing is applied.

For each context, we process all construction chunks and build a session-local
bank, then freeze the resulting snapshot. Before every question for that
context, the complete snapshot is restored, including retrieval bookkeeping.
The query phase is read-only: no query-time write, update, replacement, or
eviction is permitted. Retrieval may update ephemeral bookkeeping within one
question, but that state is discarded by the next snapshot restore. The bank is
reset between contexts, so no latent state crosses contexts.

Unless a condition explicitly varies a parameter, each method is evaluated in
five complete process-level passes using aligned base seeds 42, 142, 242, 342,
and 442 with per-context reseeding. We report the mean and population standard
deviation. Greedy controls can therefore be identical across seeds; these
repetitions are process-level stability evidence rather than independent
sampling estimates.

### A.3 Full query-time path

For an invoked query, the full method retrieves up to two threshold-qualified
slots (at most 16 historical latent vectors in total), passes the current
Reasoner state and these retrieved Weaver-space latents to the Weaver, and
injects the Weaver's newly generated eight-token latent memory into the
Reasoner. If the retrieval set is empty, the Weaver follows its ordinary
empty-retrieval path. Thus, retrieved memories support query-conditioned latent
generation rather than forming a sequence directly appended to the Reasoner.

## B. Explicit-Text Control Implementations

All text controls use the same MemGen generator, unchanged EventQA prompt,
parser, scorer, and 40-token generation limit as the latent-manager condition;
the persistent latent bank is disabled.

### B.1 Capacity-max recent-text MemGen

The base model has a 32,768-token input capacity, whereas complete EventQA
histories exceed that capacity. We therefore do not label this as an official
full-history MemGen baseline. For each question, we prepend the longest
chronologically ordered suffix of the context that fits the input budget
(32,256 raw-source tokens) before the unchanged EventQA query template. This
is a capacity-limited recent-text proxy for original MemGen without a
persistent bank.

### B.2 Rolling text summary

For each construction chunk, the same generator receives the previous summary
and the new chunk and greedily emits an updated summary. The stored summary is
capped at 128 tokenizer tokens. Once construction ends, the final summary is
prepended to every question of that context and is not updated during question
answering. This is a recursive textual-summary control; it is not claimed to
reproduce the raw rolling-window procedure of Re3 exactly.

### B.3 BM25 and matched-budget retrieval

BM25 indexes the chunks of the current context using lower-cased alphanumeric
terms, $k_1=1.5$, and $b=0.75$. Each question retrieves the two highest-scoring
chunks, with chunk order breaking ties; their full text is prepended to the
standard query prompt. The matched-budget variant uses the same two BM25
parents but selects one query-overlap-scored eight-token window from each.
The instance is accepted only when the two rendered windows add exactly 16
tokens to the prompt. It therefore controls the visible-text budget rather
than the source retrieval procedure.

### B.4 Dense E5 retrieval

The dense control uses the frozen local `intfloat/e5-base-v2` encoder. Each
4,096-token parent chunk is partitioned into non-overlapping 500-E5-token
windows. The official EventQA question, including candidate events, is encoded
as the query. A parent is assigned the maximum cosine similarity of its
windows, and the two highest-scoring distinct parents are injected in the same
full-text template as BM25. This keeps source scope, top-$k$, text-injection
format, generator, and scoring protocol fixed while changing the ranker.

## C. Query-Time Structural Ablations

All ablations below preserve the construction procedure, frozen snapshot
protocol, capacity, update policy, prompt, parser, scorer, and answer budget
unless stated otherwise.

| Variant | Query retrieval | Retrieved latents supplied to Weaver | Query Weaver output supplied to Reasoner |
|---|---:|---:|---:|
| Full latent memory manager | yes, top-$k=2$ | yes | yes |
| No-query-retrieval | no | no | yes |
| No retrieved-memory conditioning | yes, top-$k=2$ | no | yes |
| Top-1 direct latent injection | yes, at most one slot | no | no |

**No-query-retrieval** preserves the constructed bank but disables all
query-time retrieval. The query-time Weaver still runs from the current
Reasoner state and its output is consumed by the Reasoner.

**No retrieved-memory conditioning** performs the normal similarity,
threshold, top-$k$, and retrieval-bookkeeping computation, but supplies the
Weaver's standard empty-retrieval input rather than the retrieved latents. It
therefore distinguishes actual consumption of historical latents from bank
construction, retrieval code execution, and the presence of a Weaver call.

**Top-1 direct latent injection** retrieves at most the highest-scoring
threshold-qualified slot, bypasses query-time Weaver generation, maps that
eight-token Weaver-space slot through the existing Weaver-to-Reasoner
compatibility path, and injects it directly into the Reasoner. It matches the
full method's eight-token Reasoner-injection budget, but is a structural rather
than single-variable control: the full method consolidates up to two slots into
eight new tokens, whereas this variant reuses one historical slot directly.

| Method | EM | Recall | Format failures |
|---|---:|---:|---:|
| No-query-retrieval | 0.008 $\pm$ 0.000 | 0.178 $\pm$ 0.000 | 377.0 $\pm$ 0.0 |
| No retrieved-memory conditioning | 0.008 $\pm$ 0.000 | 0.178 $\pm$ 0.000 | 377.0 $\pm$ 0.0 |
| Top-1 direct latent injection | 0.047 $\pm$ 0.008 | 0.186 $\pm$ 0.009 | 354.6 $\pm$ 21.3 |
| Full latent memory manager | **0.188 $\pm$ 0.047** | **0.231 $\pm$ 0.042** | **118.6 $\pm$ 19.1** |

The two negative query controls produce the same aggregate outcome: executing
retrieval without exposing its latent output to the Weaver is insufficient.
Directly reusing one historical slot improves on those controls but remains
below the full pathway. Because direct injection also changes the number of
retrieved supports, this comparison should not be interpreted as a perfectly
matched single-factor test of Weaver integration.

## D. Sensitivity and Construction-Policy Diagnostics

### D.1 Bank capacity

| Capacity | EM | Recall | Format failures |
|---|---:|---:|---:|
| $C=8$ | 0.156 $\pm$ 0.035 | 0.209 $\pm$ 0.027 | 136.4 $\pm$ 13.0 |
| $C=16$ (default) | **0.188 $\pm$ 0.047** | **0.231 $\pm$ 0.042** | **118.6 $\pm$ 19.1** |
| $C=24$ | 0.190 $\pm$ 0.027 | 0.228 $\pm$ 0.023 | 125.8 $\pm$ 24.7 |

The $C=8$ bank reaches capacity in every context and loses both EM and recall.
The $C=24$ bank builds only 16--17 slots in practice, so it does not establish
a benefit beyond the realized occupancy of the default setting.

### D.2 Query retrieval depth and threshold

| Query top-$k$ | EM | Recall | Format failures |
|---|---:|---:|---:|
| 1 | 0.050 $\pm$ 0.006 | 0.220 $\pm$ 0.013 | 338.0 $\pm$ 31.0 |
| 2 (default) | **0.188 $\pm$ 0.047** | **0.231 $\pm$ 0.042** | **118.6 $\pm$ 19.1** |
| 4 | 0.128 $\pm$ 0.029 | 0.143 $\pm$ 0.033 | 181.0 $\pm$ 67.8 |

| Retrieval threshold | EM | Recall | Format failures |
|---|---:|---:|---:|
| 0.03 | 0.176 $\pm$ 0.023 | **0.242 $\pm$ 0.009** | 155.2 $\pm$ 27.2 |
| 0.05 (default) | **0.188 $\pm$ 0.047** | 0.231 $\pm$ 0.042 | **118.6 $\pm$ 19.1** |
| 0.10 | 0.008 $\pm$ 0.000 | 0.178 $\pm$ 0.000 | 377.0 $\pm$ 0.0 |

At threshold 0.10, no candidate passes the retrieval criterion over the
evaluated questions; the condition consequently behaves as a no-retrieved-
memory path. Lowering the threshold to 0.03 raises recall slightly but lowers
EM and produces more format failures. The default top-$k=2$ offers the best
observed overall balance among the tested retrieval-depth settings.

### D.3 Construction update and append operations

| Construction update threshold | EM | Recall | Format failures |
|---|---:|---:|---:|
| 0.00 | 0.153 $\pm$ 0.037 | 0.215 $\pm$ 0.019 | 192.2 $\pm$ 36.5 |
| 0.05 | 0.147 $\pm$ 0.060 | 0.193 $\pm$ 0.051 | 164.4 $\pm$ 67.3 |
| 0.10 (default) | **0.188 $\pm$ 0.047** | **0.231 $\pm$ 0.042** | **118.6 $\pm$ 19.1** |

| Construction policy | EM | Recall | Format failures |
|---|---:|---:|---:|
| No new-thread append | 0.052 $\pm$ 0.009 | 0.215 $\pm$ 0.010 | 350.2 $\pm$ 14.0 |
| Default configuration | **0.188 $\pm$ 0.047** | **0.231 $\pm$ 0.042** | **118.6 $\pm$ 19.1** |

Lower update thresholds overwrite more candidate memories into existing
threads and underperform the default within the tested range. Disabling
unmatched new-thread append removes construction-time retention of distinct
latent threads and substantially degrades EM and format stability.

### D.4 Frozen-bank diagnostic

| Protocol | EM | Recall | Format failures |
|---|---:|---:|---:|
| Mutable sequential queries | 0.140 $\pm$ 0.038 | 0.196 $\pm$ 0.036 | 167.4 $\pm$ 24.8 |
| Frozen default | **0.188 $\pm$ 0.047** | **0.231 $\pm$ 0.042** | **118.6 $\pm$ 19.1** |

The diagnostic mutable variant preserves one bank across ordered questions and
allows query-time updates. Its lower results are consistent with
question-derived state drift. It is a protocol diagnostic, not the default
evaluation condition.

## E. Efficiency Measurement Protocol

We measure each method in three separately launched, serialized continuous
processes on the same physical GPU. Each process loads its model once and
covers all five contexts and 500 questions. Model-loading time is excluded;
end-to-end time includes latent-bank construction or the corresponding
summary-construction/indexing stage. We report wall-clock time, amortized
seconds per question, and the maximum incremental allocated GPU memory above
the loaded-model baseline. Effectiveness-run wall-clock times collected under
shared-GPU contention are not used as cost evidence.

| Method | End-to-end (s) | Seconds/question | Peak incremental GPU memory |
|---|---:|---:|---:|
| MemGen recent-text baseline | 2688.90 $\pm$ 9.50 | 5.378 $\pm$ 0.019 | 13.094 GiB |
| Rolling text summary | 598.41 $\pm$ 16.34 | 1.197 $\pm$ 0.033 | 1.782 GiB |
| BM25 top-2 retrieved text | 645.76 $\pm$ 4.30 | 1.292 $\pm$ 0.009 | 3.505 GiB |
| Dense E5 top-2 retrieved text | 739.36 $\pm$ 5.32 | 1.479 $\pm$ 0.011 | 3.505 GiB |
| Matched-16 BM25 retrieval | 497.87 $\pm$ 18.68 | 0.996 $\pm$ 0.037 | 0.159 GiB |
| Latent memory manager | **377.06 $\pm$ 10.49** | **0.754 $\pm$ 0.021** | 0.160 GiB |

These measurements characterize the evaluated checkpoint, EventQA protocol,
and hardware. They should not be read as universal latency or memory claims
for different models, accelerators, prompt lengths, or retrieval
implementations.

## F. Scope and Interpretation

The evidence is limited to EventQA under the frozen context-bank protocol.
The capacity-max recent-text condition is a necessary proxy because complete
histories exceed the model input capacity; it is not an official full-history
MemGen result. The ablations establish data-flow and protocol-dependent
evidence within this implementation, but do not demonstrate generalization to
other benchmarks or claims about semantic interpretability of individual latent
slots.

