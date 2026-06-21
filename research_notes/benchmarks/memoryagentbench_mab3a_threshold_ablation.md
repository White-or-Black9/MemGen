# MAB-3A: LatentBank Full-history Low-threshold Retrieval Ablation

Date: 2026-06-20  
Status: completed, valid one-context threshold sweep  
Artifact: `outputs/mab/memgen_bank_on_threshold_ablation/20260620T103852Z-factconsolidation-sh-6k-onectx`

## 1. Objective

Determine whether lowering the Version-A retrieval threshold into the observed MAB-3 score range activates retrieved latent injection on the same one-context, one-query full-history setup, and whether that changes answer quality.

## 2. Why Low-threshold Ablation Was Needed

MAB-3 showed:

- turn-2 raw max score: `0.04923355419779086`
- query-turn raw max score: `0.03669293190212715`
- configured threshold: `0.7`
- retrieved latents: `0`

That result established bank creation and writes, but not actual query-time Reasoner injection. This ablation moved the threshold into the observed score range and added raw candidate-score logging.

## 3. Paired Reference Artifacts

- MAB-2 Bank-off: `outputs/mab/memgen_bank_off/20260620T034034Z-factconsolidation-sh-6k-onectx`
- MAB-3 Bank-on baseline: `outputs/mab/memgen_bank_on_full_history/20260620T085407Z-factconsolidation-sh-6k-onectx`
- MAB-3A ablation: `outputs/mab/memgen_bank_on_threshold_ablation/20260620T103852Z-factconsolidation-sh-6k-onectx`

## 4. Fixed Controls

- Same dataset row/context as MAB-2 and MAB-3
- Same first query
- Same official chunks: `[4319, 2119]`
- Same official templates and turn order
- Same decoding settings
- `history_policy=full_rebuild`
- `cross_turn_kv_reuse=false`
- `batch_size=1`
- `max_slots=8`
- `top_k=1`
- `decay_alpha=0.05`
- `pool_last_n=64`
- `update_policy=thread_update`
- Same checkpoint as MAB-2 and MAB-3

## 5. Threshold List

- `top_k_only`
- `0.00`
- `0.01`
- `0.02`
- `0.03`
- `0.035`
- `0.04`
- `0.045`
- `0.05`
- `0.07`
- `0.10`
- `0.70`

`top_k_only` was safely supported by setting `retrieve_policy=topk` with `top_k=1`. The numeric threshold remained at `0.70` in config but was ignored by selection.

## 6. Exact Commands

Regression and focused verification:

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python -m unittest \
  tests.test_mab3_bank_on_full_history \
  tests.test_mab3a_threshold_ablation \
  tests.test_latent_memory_bank \
  tests.test_latent_memory_bank_integration
```

Threshold sweep:

```bash
CUDA_VISIBLE_DEVICES=1 \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH=/mnt/18T/baishilong/MemGen \
/home/baishilong/miniconda3/envs/memgen/bin/python \
scripts/eval/mab3a_threshold_ablation.py \
  --dataset-root /mnt/18T/baishilong/datasets/MemoryAgentBench \
  --mab-repo /mnt/18T/baishilong/benchmarks/MemoryAgentBench \
  --mab-python /home/baishilong/miniconda3/envs/MABench/bin/python \
  --checkpoint-path "$MEMGEN_CHECKPOINT" \
  --model-checkpoint-id 'Kana-s/MemGen@269d9b1/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model'
```

`MEMGEN_CHECKPOINT` denotes the already-local public checkpoint path and is intentionally not copied into this note.

## 7. Candidate-score Logging Status

Enabled.

Per retrieval attempt, diagnostics now record:

- `candidate_slot_indices`
- `candidate_raw_scores`
- `candidate_score_pairs`
- `max_score`
- `matched_slot_index`
- `threshold`
- `threshold_passed`
- `retrieved_indices`
- `retrieved_scores`
- `retrieved_latent_count`
- `retrieved_latents_enter_reasoner`
- `retrieved_latents_enter_weaver`

This was implemented as a harness-level tracing change only. No MemoryAgentBench core logic was modified.

## 8. Per-threshold Retrieval Summary

| Threshold | Query retrieval active? | Retrieved latent count by turn | Retrieved indices by turn | Query max score |
|---|---:|---|---|---:|
| `top_k_only` | yes | `[0, 8, 8]` | `[[], [0], [1]]` | `0.0366929319` |
| `0.00` | yes | `[0, 8, 8]` | `[[], [0], [0]]` | `0.0366929319` |
| `0.01` | yes | `[0, 8, 8]` | `[[], [0], [0]]` | `0.0366929319` |
| `0.02` | yes | `[0, 8, 8]` | `[[], [0], [0]]` | `0.0366929319` |
| `0.03` | yes | `[0, 8, 8]` | `[[], [0], [0]]` | `0.0366929319` |
| `0.035` | yes | `[0, 8, 8]` | `[[], [0], [0]]` | `0.0366929319` |
| `0.04` | no | `[0, 8, 0]` | `[[], [0], []]` | `0.0366929319` |
| `0.045` | no | `[0, 8, 0]` | `[[], [0], []]` | `0.0366929319` |
| `0.05` | no | `[0, 0, 0]` | `[[], [], []]` | `0.0366929319` |
| `0.07` | no | `[0, 0, 0]` | `[[], [], []]` | `0.0366929319` |
| `0.10` | no | `[0, 0, 0]` | `[[], [], []]` | `0.0366929319` |
| `0.70` | no | `[0, 0, 0]` | `[[], [], []]` | `0.0366929319` |

Interpretation:

- `top_k_only` and thresholds `0.00` to `0.035` activated query-time retrieval.
- Thresholds `0.04` and `0.045` activated retrieval on chunk 2 but not on the final query.
- Thresholds `0.05` and above reproduced the original no-retrieval regime.

## 9. Per-threshold Answer and Score Summary

| Threshold band | Prediction pattern | `substring_exact_match` |
|---|---|---:|
| `top_k_only`, `0.00` to `0.035` | `"2. What is the capital of the United"` | `0` |
| `0.04`, `0.045` | `"} rugby"` | `0` |
| `0.05`, `0.07`, `0.10`, `0.70` | `"} rugby\nBased on the provided Knowledge Pool,"` | `0` |

No threshold produced a correct answer.

## 10. Prompt Parity and History

- Turns 0 and 1 matched MAB-2 and MAB-3 prompt hashes for every threshold case.
- Query-turn prompt hash matched MAB-2 and MAB-3 only for thresholds with no earlier retrieval influence: `0.05`, `0.07`, `0.10`, `0.70`.
- Query-turn prompt hash diverged for `top_k_only` and thresholds `0.00` to `0.045` because the chunk-2 acknowledgement changed after bank-mediated retrieval, which then changed the visible query history.
- Full visible history remained included in every case.
- No cross-turn KV reuse was introduced.

## 11. Boundary and Invariant Checks

All completed threshold cases satisfied:

- bank created once per context/session
- bank started empty
- same bank shared across the three turns
- writes occurred
- bank reset after session
- retrieved latents entered the Reasoner whenever retrieval was active
- retrieved latents never entered Weaver
- official MAB metrics scored the final query output

No compressed-memory path was introduced.

## 12. What Changed Scientifically

This is a valid **Case B** result.

- The retrieval gate in MAB-3 was too strict at `0.70`.
- Lower thresholds did activate retrieved latent injection.
- Activated retrieval changed the generated answer pattern.
- Activated retrieval did not fix the answer on this sample.

Therefore:

- the retrieval path is not structurally blocked
- the failure is no longer attributable to threshold gating alone
- expansion to more contexts is premature until slot-content relevance and prediction errors are inspected

## 13. Errors Encountered During Development

Two bounded harness issues occurred and were fixed before the canonical run:

- Metric extraction bug: `substring_exact_match` was read from the wrong score payload branch.
- Mid-sweep OOM: repeated case-level model loads needed explicit `gc.collect()` and `torch.cuda.empty_cache()` between threshold cases.

Neither issue required MemGen model-core changes or MemoryAgentBench core changes.

## 14. Test Results

- `67` relevant tests passed before the canonical run.
- Included:
  - `tests.test_mab3_bank_on_full_history`
  - `tests.test_mab3a_threshold_ablation`
  - `tests.test_latent_memory_bank`
  - `tests.test_latent_memory_bank_integration`

## 15. Git Status Before and After

Before:

```text
## rlm-memory-bank...origin/rlm-memory-bank [ahead 8]
 M memgen/model/modeling_memgen.py
?? research_notes/benchmarks/
?? scripts/eval/mab2_bank_off.py
?? scripts/eval/mab2_mab_bridge.py
?? scripts/eval/mab3_bank_on_full_history.py
?? tests/test_mab2_bank_off.py
?? tests/test_mab3_bank_on_full_history.py
```

After:

```text
## rlm-memory-bank...origin/rlm-memory-bank [ahead 8]
 M memgen/model/modeling_memgen.py
?? research_notes/benchmarks/
?? scripts/eval/mab2_bank_off.py
?? scripts/eval/mab2_mab_bridge.py
?? scripts/eval/mab3_bank_on_full_history.py
?? scripts/eval/mab3a_threshold_ablation.py
?? tests/test_mab2_bank_off.py
?? tests/test_mab3_bank_on_full_history.py
?? tests/test_mab3a_threshold_ablation.py
```

The unrelated modification to `memgen/model/modeling_memgen.py` remained untouched.

## 16. Recommendation for Next Step

Do not expand contexts yet.

Recommended next phase:

1. Inspect the retrieved slot content/score alignment for the successful low-threshold cases.
2. Compare final-token behavior between `0.00`, `0.04`, and `0.70` to determine whether retrieval is harmful, irrelevant, or simply misaligned.
3. Keep compressed-memory deferred until the full-history Bank-on path is better understood on retrieved-positive examples.

This milestone establishes positive mechanism activation, but not positive task utility.
