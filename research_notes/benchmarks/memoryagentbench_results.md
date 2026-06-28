# MemoryAgentBench Results

## Status

This is the canonical summary of MemoryAgentBench results for MemGen. Detailed
run provenance remains in the linked evidence notes and output artifacts.

The current reference run is MAB-5A:

- run ID: `20260621T013454Z-detectiveqa-compressed-n10`
- artifact:
  `outputs/mab/compressed_memory_detectiveqa_n10/20260621T013454Z-detectiveqa-compressed-n10/`
- split/subtask: `Long_Range_Understanding / detective_qa`
- protocol: compressed-memory, 10 contexts, first query only

## Experiment Summary

| Experiment | Scope | Status | Main result |
| --- | --- | --- | --- |
| MAB-1A no-API smoke | `factconsolidation_sh_6k`, one local row | Infrastructure evidence | Local loading, official chunking/templates, and metric path validated |
| MAB-2 | Full-history Bank-off, one context | Valid historical run | Original MemGen harness and official scoring completed |
| MAB-3 | Full-history Bank-on, one context | Valid historical run | Bank lifecycle and Reasoner-only boundary validated; default threshold retrieved nothing |
| MAB-3A | Shared-threshold ablation, one context | Valid historical diagnostic | Low thresholds activated retrieval; not performance evidence |
| MAB-4A | Compressed-memory Bank-on, one context | Exploratory | Removed chunk/ack history from query and exercised latent retrieval |
| Paired low-threshold attempt | Requested n10 on `factconsolidation_sh_6k` | Dataset-limited | Only one matching local context existed; not n10 evidence |
| Local task audit | All local parquet splits | Completed audit | `detective_qa` has 10 rows but full history is over capacity |
| Over-context diagnostic | Synthetic boundary plus real preflight | Completed diagnostic | Original full-history path has no explicit guard; real over-capacity samples must be rejected before generation |
| MAB-5A | Compressed Bank-off vs Bank-on, detective_qa n10 | Completed reference baseline | Both exact-match accuracies were 0.0; mechanism active in every context |
| MAB-5B | Raised shared-threshold diagnostic, detective_qa n10 | Completed diagnostic | Both exact-match accuracies were 0.0; slot counts rose to the max in every context; retrieval remained active in every context |
| MAB-5C | Decoupled retrieval-update thresholds, detective_qa n10 | Completed diagnostic | Both exact-match accuracies were 0.0; the canonical checked-in runner rerun reached full slot counts; query-time retrieval stayed active in every context; retrieved latents were Reasoner-only |
| MAB-5D | Capacity16 decoupled retrieval-update thresholds, detective_qa n10 | Completed diagnostic | Both exact-match accuracies were 0.0; final slot counts rose to 16 in every context; eviction churn dropped versus MAB-5C; retrieved latents remained Reasoner-only |
| MAB-6A | Version B Weaver-conditioned memory, detective_qa n10 | Completed exploratory diagnostic | Both exact-match accuracies remained 0.0; output_changed stayed 10/10; retrieved memory entered Weaver, raw retrieved memory no longer entered Reasoner, and query writes remained 0 |
| MAB-6B | Version B Weaver-space bank, detective_qa n10 | Completed exploratory diagnostic | Bank-off exact match stayed 0.0 and Bank-on exact match improved to 0.1; output_changed stayed 10/10; memory storage/query space moved to Weaver; retrieved memory avoided `reasoner_to_weaver` reprojection; query writes remained 0 |

## Full-History Capacity Boundary

The checkpoint context capacity is 32,768 tokens. Original MemGen's multi-turn
full-history path has no explicit over-context guard and does not silently
truncate the rebuilt conversation.

The synthetic diagnostic observed:

| Estimated input tokens | Result |
| ---: | --- |
| 32,000 | Generation succeeded |
| 32,760 | Generation succeeded |
| 32,800 | Generation succeeded |
| 35,000 | `OutOfMemoryError` |

The first selected detective_qa context preflight estimated 102,477 full-history
query tokens. All 10 MAB-5A contexts exceeded capacity. Therefore:

- original full-history detective_qa is `over_capacity_invalid`;
- no full-history generation was called for MAB-5A;
- over-capacity output is not a scored baseline;
- silent truncation is prohibited;
- any future truncated-history condition must be named and evaluated separately.

Detailed evidence:

- `memgen_over_context_behavior.md`
- `outputs/mab/memgen_over_context_behavior/20260620T133105Z-over-context/over_context_diagnostic.json`

## MAB-5A Result

| Metric | Value |
| --- | ---: |
| Requested / valid contexts | 10 / 10 |
| Compressed Bank-off exact match | 0.0 |
| Compressed Bank-on exact match | 0.0 |
| Accuracy delta | 0.0 |
| Output changed | 10 |
| Improved / regressed by exact match | 0 / 0 |
| Retrieval-active contexts | 10 |
| Query write count | 0 |
| Cross-context leakage detected | 0 |
| Retrieved memory entered Reasoner | Yes |
| Retrieved memory entered Weaver | No |
| Final slot counts | `[1, 2, 2, 5, 6, 5, 6, 7, 4, 7]` |
| Successful retrieved-score range | approximately `0.030-0.064` |

`output_changed=10` establishes that Bank-on affected generation. It is not an
improvement metric. Likewise, official exact match of zero does not mean the
mechanism was inactive: retrieval occurred in every context and all outputs
changed.

Official exact match remains the benchmark result. Gold-substring, normalized,
or other relaxed checks may be reported only as separately labeled diagnostics.

## Mechanism Interpretation

MAB-5A used:

- `threshold=0.03`
- `top_k=1`
- `max_slots=8`
- `retrieve_policy=threshold_topk`
- `update_policy=thread_update`

The low threshold kept retrieval non-empty, but the same threshold also governed
matched-thread replacement in `write_back()`. The final slot counts remained low
relative to 25-50 chunks per context, consistent with over-merge or
over-compression.

Current source behavior is precise:

1. Before Weaver generates a new latent, `retrieve_with_context()` builds a query
   from `candidate_inputs_embeds`.
2. The score compares that current-context query with each existing `slot.key`.
3. It does not compare the new Weaver latent with an old slot.
4. Weaver then generates `latent_inputs_embeds`.
5. `write_back()` writes or replaces memory with that Weaver-generated
   reasoner-space latent.
6. Retrieved memory enters Reasoner only and does not enter Weaver in Version A.

## Current Conclusion

MAB-5A is preliminary negative performance evidence but positive mechanism
activation evidence. The bank changed behavior without improving official exact
match. The next experiment is not another shared-threshold sweep; it is MAB-5C,
which separates retrieval visibility from update matching while preserving old
behavior by default.

## MAB-6B Result

| Metric | Value |
| --- | ---: |
| Requested / valid contexts | 10 / 10 |
| Compressed Bank-off exact match | 0.0 |
| Compressed Bank-on exact match | 0.1 |
| Accuracy delta | +0.1 |
| Output changed | 10 |
| Improved / regressed by exact match | 1 / 0 |
| Retrieval-active contexts | 10 |
| Query write count | 0 |
| Cross-context leakage detected | 0 |
| Memory bank storage space | `weaver` |
| Retrieval query space | `weaver` |
| Retrieved memory projected to Weaver | No |
| Final slot counts | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]` |

MAB-6B is exploratory diagnostic evidence, but unlike MAB-6A it improved
official exact match on the fixed detective_qa n10 slice from `0.0` to `0.1`.
The routing diagnostics stayed aligned with the intended Weaver-space bank
design: `memory_bank_storage_space=weaver`, `stored_latent_space=weaver`,
`retrieval_query_space=weaver`, and
`retrieved_memory_projected_to_weaver=false`. Query writes remained `0` and
cross-context leakage stayed `false`.

The improvement is still narrow evidence. The same run also collapsed final
slot counts to `1` in every context and wrote by matched replacement almost
exclusively (`insert=10`, `replace_matched=316`), so this should be treated as
an exploratory mechanism result rather than a default-path promotion.

See:

- `memoryagentbench_mab5a_detectiveqa_compressed_n10.md` for per-context evidence;
- `memoryagentbench_next_steps.md` for the current action;
- `memoryagentbench_mechanism_plan.md` for implementation and experiment details;
- `memoryagentbench_runbook.md` for operational commands.

## MAB-5B Result

| Metric | Value |
| --- | ---: |
| Requested / valid contexts | 10 / 10 |
| Compressed Bank-off exact match | 0.0 |
| Compressed Bank-on exact match | 0.0 |
| Accuracy delta | 0.0 |
| Output changed | 5 |
| Improved / regressed by exact match | 0 / 0 |
| Retrieval-active contexts | 10 |
| Query write count | 0 |
| Cross-context leakage detected | 0 |
| Retrieved memory entered Reasoner | Yes |
| Retrieved memory entered Weaver | No |
| Final slot counts | `[8, 8, 8, 8, 8, 8, 8, 8, 8, 8]` |
| Mean final slot count | `8.0` |
| Retrieved latent count | `200` |
| Write count | `326` |
| Retrieval count | `316` |
| Successful retrieved-score range | approximately `0.050-0.064` |

MAB-5B is diagnostic evidence for the raised shared-threshold setting, not a
new reference baseline. Compared with MAB-5A, it increased slot counts to the
maximum on every context while keeping retrieval active and Reasoner-only. It
did not improve official exact match.
The resulting tradeoff strengthens the case for MAB-5C: keep retrieval density
closer to MAB-5A while recovering the slot growth seen in MAB-5B. A clean first
MAB-5C should start with `retrieve_threshold=0.03`,
`update_threshold=0.05`, `max_slots=8`, and `top_k=1`.

## MAB-5C Result

Canonical run:

- `outputs/mab/decoupled_thresholds_detectiveqa_n10/20260622T140741Z-detectiveqa-decoupled-thresholds-n10/`

Preliminary non-canonical runtime-patch run:

- `outputs/mab/decoupled_thresholds_detectiveqa_n10/20260622T131149Z-detectiveqa-decoupled-thresholds-n10/`

| Metric | Value |
| --- | ---: |
| Requested / valid contexts | 10 / 10 |
| Compressed Bank-off exact match | 0.0 |
| Compressed Bank-on exact match | 0.0 |
| Accuracy delta | 0.0 |
| Output changed | 10 |
| Retrieval-active contexts | 10 |
| Query-turn retrieval active contexts | 10 |
| Query write count | 0 |
| Cross-context leakage detected | 0 |
| Retrieved memory entered Reasoner | Yes |
| Retrieved memory entered Weaver | No |
| Final slot counts | `[8, 8, 8, 8, 8, 8, 8, 8, 8, 8]` |
| Mean final slot count | `8.0` |
| Retrieved latent count | `2288` |
| Write count | `326` |
| Retrieval count | `316` |
| Construction-time retrieval count | `306` |
| Query-turn retrieved latent count | `80` |
| Append/insert count | `80` |
| Matched replace count | `36` |
| Capacity evict count | `210` |
| Successful retrieved-score range | approximately `0.030-0.064` |

MAB-5C is the first decoupled-threshold diagnostic. It preserves the
shared-threshold defaults when the new fields are unset, and it confirms the
mechanism split: the low retrieval threshold kept query-time retrieval active
in all contexts while the higher update threshold still allowed the bank to
grow to capacity. Exact match did not improve, but the mechanism signal is
clearer than in either MAB-5A or MAB-5B. The checked-in runner rerun is the
canonical artifact; the earlier runtime-patch result is historical only.

## MAB-5D Result

Canonical run:

- `outputs/mab/capacity16_detectiveqa_n10/20260623T022140Z-detectiveqa-capacity16-n10/`

Non-canonical earlier attempt:

- `outputs/mab/capacity16_detectiveqa_n10/20260623T015929Z-detectiveqa-decoupled-thresholds-n10/`

| Metric | Value |
| --- | ---: |
| Requested / valid contexts | 10 / 10 |
| Compressed Bank-off exact match | 0.0 |
| Compressed Bank-on exact match | 0.0 |
| Accuracy delta | 0.0 |
| Output changed | 10 |
| Query-turn retrieval active contexts | 10 |
| Final slot counts | `[16, 16, 16, 16, 16, 16, 16, 16, 16, 16]` |
| Mean final slot count | `16.0` |
| Total write count | `326` |
| Total retrieval count | `316` |
| Total retrieved latent count | `2272` |
| Construction-time retrieval count | `306` |
| Query-turn retrieved latent count | `80` |
| Query write count | `0` |
| Query write attempt count | `0` |
| Cross-context leakage detected | `0` |
| Retrieved memory entered Reasoner | Yes |
| Retrieved memory entered Weaver | No |
| Write action counts | `{'insert': 160, 'replace_matched': 33, 'evict_oldest_insert': 133}` |
| Update reason counts | `{'empty_bank': 10, 'matched_thread': 33, 'new_thread': 150, 'new_thread_bank_full': 133}` |
| Append/insert count | `160` |
| Matched replace count | `33` |
| Capacity evict count | `133` |
| Successful retrieved-score range | approximately `0.030-0.064` |

MAB-5D is the clean capacity ablation for the split-threshold mechanism. It
confirms that moving from `max_slots=8` to `max_slots=16` raises final slot
counts to the new capacity and reduces eviction churn, but it does not improve
official exact match. The context-6 relaxed diagnostic is semantically close
to the gold answer, but that is not counted as official accuracy.

## MAB-6A Result

Canonical run:

- `outputs/mab/version_b_weaver_conditioned_detectiveqa_n10/20260625T023822Z-detectiveqa-version-b-weaver-conditioned-n10/`

Earlier failed/intermediate runs:

- `outputs/mab/version_b_weaver_conditioned_detectiveqa_n10/20260625T021750Z-detectiveqa-version-b-weaver-conditioned-n10/`
- `outputs/mab/version_b_weaver_conditioned_detectiveqa_n10/20260625T021830Z-detectiveqa-version-b-weaver-conditioned-n10/`
- `outputs/mab/version_b_weaver_conditioned_detectiveqa_n10/20260625T022818Z-detectiveqa-version-b-weaver-conditioned-n10/`

| Metric | Value |
| --- | ---: |
| Requested / valid contexts | 10 / 10 |
| Compressed Bank-off exact match | 0.0 |
| Compressed Bank-on exact match | 0.0 |
| Accuracy delta | 0.0 |
| Output changed | 10 |
| Query-turn retrieval active contexts | 10 |
| Final slot counts | `[8, 8, 8, 8, 8, 8, 8, 8, 8, 8]` |
| Mean final slot count | `8.0` |
| Total write count | `326` |
| Total retrieval count | `316` |
| Total retrieved latent count | `2304` |
| Query write count | `0` |
| Query write attempt count | `0` |
| Cross-context leakage detected | `0` |
| Retrieved memory to Weaver | Yes |
| Retrieved latents entered Weaver | Yes |
| Raw retrieved memory entered Reasoner | No |
| Retrieved latents entered Reasoner | No |
| Weaver conditioned on retrieved memory | Yes |
| Weaver conditioning token count | `80` |
| Fused latent generated | Yes |
| Write action counts | `{'insert': 80, 'replace_matched': 35, 'evict_oldest_insert': 211}` |
| Update reason counts | `{'empty_bank': 10, 'matched_thread': 35, 'new_thread': 70, 'new_thread_bank_full': 211}` |

MAB-6A is exploratory diagnostic evidence only. It differs from the MAB-5C
canonical baseline primarily by routing retrieved reasoner-space memory into
Weaver, then injecting only the fused latent back into Reasoner. The mechanism
was active in all 10 contexts: outputs changed in every case, retrieved memory
entered Weaver, raw retrieved memory no longer entered Reasoner directly, and
query writes remained disabled. Official exact match did not improve, so this
is not a performance win and Version A remains the default.

During the run, one context emitted a tokenizer/model warning at
`132726 > 131072` tokens while estimating the over-capacity full-history path.
That context still remained `full_history_status=over_capacity_invalid`, and no
full-history detective_qa generation was scored or silently truncated.
