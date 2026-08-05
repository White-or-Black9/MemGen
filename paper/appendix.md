# Appendix

## A. Additional EventQA Diagnostics

### A.1 Shared Evaluation Protocol

This appendix reports supplementary diagnostics for the frozen latent memory
manager on EventQA-65536. Unless a table explicitly varies a setting, all runs
use the main P7 checkpoint, 4,096-token construction chunks, bank capacity
$C=16$, query retrieval top-$k=2$, retrieval threshold $0.05$, construction
update threshold $0.10$, temporal decay $\alpha=0.05$, a 40-token generation
budget, and the unchanged EventQA prompt, parser, and scorer. Query-time
writes are disabled, and the same constructed bank snapshot is restored before
each question.

Each reported condition consists of five complete process-level passes over
five contexts and 500 questions, using base seeds 42, 142, 242, 342, and 442.
All entries report mean $\pm$ population standard deviation. These diagnostics
are effectiveness evidence and are not used for latency or memory-cost claims.

### A.2 Capacity and Query-Retrieval Sensitivity

#### Bank Capacity

| Maximum bank capacity | Complete passes | EM | Recall | Format failures |
|---:|---:|---:|---:|---:|
| $C=8$ | 5 | 0.156$\pm$0.035 | 0.209$\pm$0.027 | 136.4$\pm$13.0 |
| $C=16$ (main P7 reference) | 5 | 0.188$\pm$0.047 | 0.231$\pm$0.042 | 118.6$\pm$19.1 |
| $C=24$ | 5 | 0.190$\pm$0.027 | 0.228$\pm$0.023 | 125.8$\pm$24.7 |

**Table A1. Capacity sensitivity on EventQA-65536.** Reducing capacity to
$C=8$ lowers EM and recall and increases format failures. The $C=8$ bank
reached its capacity in every context. In contrast, the $C=24$ condition
constructed only 16--17 slots; the expanded maximum was not reached, so this
comparison does not establish a benefit from capacity beyond realized
occupancy.

#### Query Retrieval Depth

| Query retrieval top-$k$ | Complete passes | EM | Recall | Format failures |
|---:|---:|---:|---:|---:|
| $k=1$ | 5 | 0.050$\pm$0.006 | 0.220$\pm$0.013 | 338.0$\pm$31.0 |
| $k=2$ (main P7 reference) | 5 | **0.188$\pm$0.047** | **0.231$\pm$0.042** | **118.6$\pm$19.1** |
| $k=4$ | 5 | 0.128$\pm$0.029 | 0.143$\pm$0.033 | 181.0$\pm$67.8 |

**Table A2. Query retrieval top-$k$ sensitivity.** One retrieved slot sharply
reduces EM and format stability. Increasing the maximum to four slots does not
recover the $k=2$ result and lowers recall, indicating that the two-slot
setting is the most balanced among the tested values.

#### Query Retrieval Threshold

| Retrieval threshold | Complete passes | EM | Recall | Format failures |
|---:|---:|---:|---:|---:|
| 0.03 | 5 | 0.176$\pm$0.023 | **0.242$\pm$0.009** | 155.2$\pm$27.2 |
| 0.05 (main P7 reference) | 5 | **0.188$\pm$0.047** | 0.231$\pm$0.042 | **118.6$\pm$19.1** |
| 0.10 | 5 | 0.008$\pm$0.000 | 0.178$\pm$0.000 | 377.0$\pm$0.0 |

**Table A3. Retrieval-threshold sensitivity.** Lowering the threshold to 0.03
slightly increases recall but reduces EM and format stability. At 0.10, no
candidate slot passed the threshold in any of the 2,500 questions (observed
maximum final scores ranged from 0.078 to 0.095); no retrieved latent entered
the Weaver, reproducing the no-query-retrieval outcome.

### A.3 Construction-Policy Diagnostics

#### Construction Update Threshold

The construction update threshold is distinct from the query retrieval
threshold: it determines whether an incoming construction latent updates an
existing thread or is treated as a new thread. Query retrieval remains fixed at
threshold $0.05$ and top-$k=2$ in this diagnostic.

| Construction update threshold | Complete passes | EM | Recall | Format failures |
|---:|---:|---:|---:|---:|
| 0.00 | 5 | 0.153$\pm$0.037 | 0.215$\pm$0.019 | 192.2$\pm$36.5 |
| 0.05 | 5 | 0.147$\pm$0.060 | 0.193$\pm$0.051 | 164.4$\pm$67.3 |
| 0.10 (main P7 reference) | 5 | **0.188$\pm$0.047** | **0.231$\pm$0.042** | **118.6$\pm$19.1** |

**Table A4. Construction update-threshold sensitivity.** Both lower
thresholds produce more matched-thread replacements during construction and
underperform the $0.10$ reference on all three reported metrics. Within the
tested range, $0.10$ is the most balanced setting.

#### New-Thread Append

This ablation preserves construction retrieval, matched-thread replacement,
capacity, and the frozen query protocol, but suppresses unmatched new-thread
appends after empty-bank initialization. Such latents are discarded rather than
written as new slots.

| Construction policy | Complete passes | EM | Recall | Format failures |
|---|---:|---:|---:|---:|
| No new-thread append | 5 | 0.052$\pm$0.009 | 0.215$\pm$0.010 | 350.2$\pm$14.0 |
| Full P7 reference | 5 | **0.188$\pm$0.047** | **0.231$\pm$0.042** | **118.6$\pm$19.1** |

**Table A5. New-thread append ablation.** Across the five repeats, 390
construction writes that would otherwise create novel threads were discarded.
The resulting decline in EM and increase in format failures show that
retaining distinct historical threads is necessary under this protocol.

### A.4 Query-State Lifecycle Diagnostic

The main protocol freezes the constructed context bank and restores its
snapshot before every question. The mutable diagnostic instead retains one bank
through the ordered questions of each context and enables query-time
`thread_update` writes; all other settings are unchanged.

| Query-bank protocol | Complete passes | EM | Recall | Format failures |
|---|---:|---:|---:|---:|
| Mutable sequential queries | 5 | 0.140$\pm$0.038 | 0.196$\pm$0.036 | 167.4$\pm$24.8 |
| Frozen P7 reference | 5 | 0.188$\pm$0.047 | 0.231$\pm$0.042 | 118.6$\pm$19.1 |

**Table A6. Query-state lifecycle diagnostic.** Every mutable pass introduced
500 question-derived bank writes. Its lower EM and recall and higher
format-failure count are consistent with query-induced state drift. This is a
diagnostic control, not a replacement evaluation protocol or an inference-cost
comparison.
