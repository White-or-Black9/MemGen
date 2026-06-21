# MAB-4A: LatentBank Compressed-memory Bank-on Exploratory

Date: 2026-06-20  
Status: completed, exploratory one-context run  
Canonical artifact: `outputs/mab/memgen_bank_on_compressed_memory/20260620T111903Z-factconsolidation-sh-6k-onectx`

## 1. Objective

Test whether the added session-local LatentBank can support query answering when the final query turn no longer includes the previous chunk dialogue history in the visible prompt.

## 2. Why This Is Exploratory

This is not the main causal comparison. The main paired comparison remains MAB-2 versus MAB-3 under full visible history.

MAB-4A intentionally changes the query-time prompt regime:

- chunk turns still run normally and can write to the bank
- query turn no longer sees prior chunk text or acknowledgement history
- query relies on visible query prompt plus retrieved latent memory only

That makes this a diagnostic of whether LatentBank can independently carry history under a compressed query prompt.

## 3. Difference From Full-history Bank-on

Relative to MAB-3 and MAB-3A:

- same dataset row/context
- same first query
- same official chunks and templates
- same checkpoint
- same Version-A Reasoner-only boundary
- same batch size 1

Only the query-turn visible prompt changed:

- MAB-3 / MAB-3A: full previous chunk dialogue history was included
- MAB-4A: query turn used only `system + current official query prompt`

## 4. Data, Context, and Query

- Dataset root: `/mnt/18T/baishilong/datasets/MemoryAgentBench`
- Split: `Conflict_Resolution`
- Sub-dataset: `factconsolidation_sh_6k`
- Context ID: `conflict-resolution-e46cb14b53eedd71`
- Query ID: `0`
- Gold answers: `["pesäpallo"]`
- Official chunk token lengths: `[4319, 2119]`

Paired artifacts:

- MAB-2: `outputs/mab/memgen_bank_off/20260620T034034Z-factconsolidation-sh-6k-onectx`
- MAB-3: `outputs/mab/memgen_bank_on_full_history/20260620T085407Z-factconsolidation-sh-6k-onectx`
- MAB-3A: `outputs/mab/memgen_bank_on_threshold_ablation/20260620T103852Z-factconsolidation-sh-6k-onectx`

## 5. Thresholds Tested

- `top_k_only`
- `0.00`
- `0.03`
- `0.035`
- `0.70`

`top_k_only` was safely supported by using `retrieve_policy=topk` with `top_k=1`.

## 6. Query Prompt Token Length vs Full-history

- Full-history reference query prompt length from MAB-2/MAB-3: `7677`
- Compressed-memory query prompt length in every MAB-4A threshold case: `192`

This is a reduction of `7485` tokens. The compressed query prompt is about `40x` shorter than the full-history query prompt.

## 7. Evidence Full Chunk History Was Excluded

For every threshold case:

- `history_policy=compressed`
- query-turn `full_history_included=false`
- query-turn `query_prompt_contains_chunk_text=false`

The compressed runner also included an explicit leak guard:

- if the rendered query prompt contained a contiguous 128-character substring from either chunk, the run would be marked invalid

No leakage was detected in the canonical run.

## 8. Bank Write, Retrieval, and Injection Summary

Across all cases:

- bank created once per session
- chunk turns wrote into the bank
- retrieved latents, when present, entered the Reasoner only
- retrieved latents never entered Weaver
- bank reset after session

Per threshold query-turn summary:

| Threshold | Query retrieved latents | Retrieved indices | Retrieved scores | Reasoner-only? | Weaver leak? |
|---|---:|---|---|---|---|
| `top_k_only` | `8` | `[1]` | `[0.03669293190212715]` | yes | no |
| `0.00` | `8` | `[0]` | `[0.03669293190212715]` | yes | no |
| `0.03` | `8` | `[0]` | `[0.03669293190212715]` | yes | no |
| `0.035` | `8` | `[0]` | `[0.03669293190212715]` | yes | no |
| `0.70` | `0` | `[]` | `[]` | vacuous | no |

## 9. Prediction and Score Per Threshold

| Threshold | Prediction | `substring_exact_match` |
|---|---|---:|
| `top_k_only` | `純粹的搜索結果中沒有找到答案。` | `0` |
| `0.00` | `純粹的搜索結果中沒有找到答案。` | `0` |
| `0.03` | `純粹的搜索結果中沒有找到答案。` | `0` |
| `0.035` | `純粹的搜索結果中沒有找到答案。` | `0` |
| `0.70` | `简短回答， goaltenders are associated with ice` | `0` |

No threshold produced a correct answer.

## 10. Comparison to MAB-2, MAB-3, and MAB-3A

Prediction comparison:

- MAB-2 Bank-off: `"} rugby\nBased on the provided Knowledge Pool,"`
- MAB-3 full-history Bank-on @ `0.70`: same as MAB-2
- MAB-3A full-history low-threshold retrieved cases: `"2. What is the capital of the United"`
- MAB-4A compressed-memory low-threshold retrieved cases: `純粹的搜索結果中沒有找到答案。`
- MAB-4A compressed-memory @ `0.70`: `简短回答， goaltenders are associated with ice`

Interpretation:

- removing visible history clearly changed the output regime
- low-threshold compressed retrieval produced a different failure mode than low-threshold full-history retrieval
- compressed prompt plus bank retrieval still did not solve the sample

## 11. Whether Compressed-memory Changed the Answer

Yes.

Compared with MAB-2 and MAB-3, MAB-4A changed the final answer in every threshold case.

Compared with MAB-3A full-history low-threshold retrieval:

- compressed-memory low-threshold cases changed from `"2. What is the capital of the United"` to `純粹的搜索結果中沒有找到答案。`

## 12. Whether Compressed-memory Produced a Correct Answer

No.

All five threshold cases remained `substring_exact_match = 0`.

## 13. Failure Mode Interpretation

This run answers the specific exploratory question:

- removing full visible history does change the answer
- low-threshold LatentBank retrieval still activates under compressed query prompts
- the bank alone did not recover the correct answer on this sample

This suggests:

- the query retrieval path is functional even without visible chunk history
- retrieval alone is not sufficient here
- the compressed-memory failure mode differs from the full-history failure mode, so the bank is affecting generation, but not in a task-correct direction

## 14. Recommendation for Next Step

Do not elevate MAB-4A to a main claim.

Recommended next step:

1. Inspect retrieved slot content and the generated answer trajectory for the low-threshold compressed cases.
2. Compare compressed vs full-history low-threshold cases turn-by-turn to identify whether the missing visible history removes a useful scaffold or whether retrieved latents themselves are misaligned.
3. Keep context expansion deferred until one retrieved-positive compressed case shows a more interpretable or useful behavior.
