# MAB-3: MemGen + LatentBank V-A Full-history Rebuild Bank-on

Date: 2026-06-20  
Status: completed, valid paired one-context run  
Canonical run ID: `20260620T085407Z-factconsolidation-sh-6k-onectx`

## 1. Objective

Evaluate the same MemoryAgentBench context and first query used by MAB-2 while changing the principal intervention from Bank-off to the existing Version-A session-level LatentMemoryBank. Full visible dialogue history, turn order, model checkpoint, decoding, and official MAB evaluation remain paired with MAB-2.

Research question: with full prompt history already available, does enabling the existing Version-A bank change memory behavior or answer quality on this one Conflict Resolution sample?

## 2. Why This Is Full-history Rebuild Bank-on

The exact experiment name is **MemGen + LatentBank V-A Full-history Rebuild Bank-on**.

- Each generation call rebuilds the full visible message history.
- Both memorization chunks remain in the final query prompt.
- No cross-turn KV cache is reused.
- A single enabled LatentMemoryBank is created at context start and shared across all three turns.
- Trigger/Weaver prompt-time augmentation remains active under the original fixed-invoke checkpoint behavior.
- Reasoner-space Weaver outputs are written to the session bank.
- The bank is explicitly reset after the context.
- Compressed-memory and strict no-history modes are not implemented.

## 3. Pairing With MAB-2

Paired baseline:

`outputs/mab/memgen_bank_off/20260620T034034Z-factconsolidation-sh-6k-onectx`

Pairing checks passed:

- Context ID: exact match
- First query and gold answers: exact match
- Official chunk lengths: exact match, `[4319, 2119]`
- Official template: unchanged, `factconsolidation/long_context_agent`
- Decoding: greedy, 10 maximum new tokens per turn
- Prompt-history policy: `full_rebuild`
- Rendered prompt hashes: exact match on all three turns
- Prompt-history token lengths: exact match, `[4972, 7497, 7677]`
- Final prediction: exact match

The MAB memorization template contains a timestamp. The MAB-2 first-turn hash was used to recover and pin its timestamp (`2026-06-20 11:40:37`) for MAB-3. This removes an otherwise irrelevant prompt-parity confound. The bridge change is backward compatible: without `--timestamp`, MAB-2 retains its original wall-clock behavior.

## 4. Data and Task

- Dataset root: `/mnt/18T/baishilong/datasets/MemoryAgentBench`
- Split: `Conflict_Resolution`
- Sub-dataset: `factconsolidation_sh_6k`
- Total rows: 8
- Matching rows: 1
- Contexts run: 1
- Queries run: first query only
- Chunk size: 4096
- Official chunks: 2
- Official chunk token lengths: `[4319, 2119]`
- Metric: official `substring_exact_match`, plus EM/F1/ROUGE fields
- MemoryAgentBench commit: `455306dcabc3842526eb83cd4e225e5d486c5c5d`

## 5. Exact Commands

Focused red/green tests and syntax checks:

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python -m unittest tests.test_mab3_bank_on_full_history
/home/baishilong/miniconda3/envs/memgen/bin/python -m py_compile scripts/eval/mab3_bank_on_full_history.py
/home/baishilong/miniconda3/envs/MABench/bin/python -m py_compile scripts/eval/mab2_mab_bridge.py
git diff --check -- scripts/eval/mab2_mab_bridge.py scripts/eval/mab3_bank_on_full_history.py tests/test_mab3_bank_on_full_history.py
```

Relevant regression and integration suite:

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python -m unittest \
  tests.test_latent_memory_bank \
  tests.test_latent_memory_bank_integration \
  tests.test_mab2_bank_off \
  tests.test_mab3_bank_on_full_history \
  tests.test_r4_triviaqa_dynamic_harness
```

Canonical run. `MEMGEN_CHECKPOINT` denotes the local copy of the public checkpoint identified in the manifest; its machine-local path is intentionally omitted.

```bash
CUDA_VISIBLE_DEVICES=1 \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH=/mnt/18T/baishilong/MemGen \
/home/baishilong/miniconda3/envs/memgen/bin/python \
scripts/eval/mab3_bank_on_full_history.py \
  --dataset-root /mnt/18T/baishilong/datasets/MemoryAgentBench \
  --mab-repo /mnt/18T/baishilong/benchmarks/MemoryAgentBench \
  --mab-python /home/baishilong/miniconda3/envs/MABench/bin/python \
  --checkpoint-path "$MEMGEN_CHECKPOINT" \
  --model-checkpoint-id 'Kana-s/MemGen@269d9b1/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model'
```

No external API or network model/dataset access was used.

## 6. Output Artifact

Canonical artifact:

`outputs/mab/memgen_bank_on_full_history/20260620T085407Z-factconsolidation-sh-6k-onectx/`

Files:

- `manifest.json`
- `results.json`
- `diagnostics.jsonl`
- `run_config.json`

An earlier successful instrumentation run exists at `20260620T085220Z-factconsolidation-sh-6k-onectx`. It is non-canonical because its `top_retrieval_scores` field incorrectly emitted only threshold-passing scores. The model result was identical. The canonical rerun followed a regression-tested diagnostics fix and preserves raw top scores even when retrieval is filtered out.

## 7. Result Score

- MAB-3 `substring_exact_match`: `0`
- MAB-2 `substring_exact_match`: `0`
- MAB-3 exact match/F1/ROUGE: `0`
- MAB-3 prediction equals MAB-2 prediction: yes
- Query input/output lengths: 7677 / 10 tokens
- Memory-construction generation time: 2.7885 seconds
- Query generation time: 1.2161 seconds

This is a valid measured non-improvement on one sample. It is not evidence that retrieval is ineffective because no slot passed the configured threshold.

## 8. Comparison to MAB-2

| Property | MAB-2 Bank-off | MAB-3 Bank-on |
|---|---:|---:|
| Visible prompt hashes | Reference | Exact match, 3/3 |
| Full query history tokens | 7677 | 7677 |
| Bank created | No | Yes, once |
| Writes after query | 0 | 3 |
| Slots after query | 0 | 3 |
| Selected retrieved slots | 0 | 0 |
| Retrieved latent tokens | 0 | 0 |
| Prediction changed | N/A | No |
| `substring_exact_match` | 0 | 0 |

The bank intervention successfully changed state construction but not Reasoner input or output because thresholding selected no stored memory.

## 9. Prompt, History, and Cache Policy

- `history_policy=full_rebuild`
- `cross_turn_kv_reuse=false`
- `intra_generation_kv_cache=false`
- `batch_size=1`
- One `run_agent_loop()` episode covers both chunks and the query.
- Rendered prompt-history lengths: `[4972, 7497, 7677]`.
- All prompt hashes match MAB-2 exactly.
- No visible prompt was truncated.

The intra-generation cache value remains false because the unchanged checkpoint has `max_inference_aug_num=0`; after prompt augmentation the original continuation path explicitly uses `use_cache=false`.

## 10. Bank Lifecycle and Counters

Bank configuration:

- `max_slots=8`
- `top_k=1`
- `threshold=0.7`
- `decay_alpha=0.05`
- `pool_last_n=64`
- `update_policy=thread_update`
- `retrieve_policy=threshold_topk`
- `storage_device=cpu`

Lifecycle:

- Creation count: 1
- Initial slots: 0
- Same object across all turns: yes
- Final slots before reset: 3
- Slots after explicit reset: 0

Per-turn cumulative counters:

| Turn | Writes | Retrievals over non-empty bank | Slots | Replacements | Retrieved slots | Retrieved latents |
|---|---:|---:|---:|---:|---:|---:|
| Chunk 1 | 1 | 0 | 1 | 0 | 0 | 0 |
| Chunk 2 | 2 | 1 | 2 | 0 | 0 | 0 |
| Query | 3 | 2 | 3 | 0 | 0 | 0 |

Each write stored 8 tokens with hidden size 1536 in CPU `bfloat16`. Runtime checks confirmed the write input was the `weaver_to_reasoner` output and the stored tensor was detached and cloned.

Raw top retrieval scores:

- Chunk 1: none, because the bank was initially empty
- Chunk 2: `0.0492335542`
- Query: `0.0366929319`, `0.0340197662`

All scores were below `threshold=0.7`. The bank therefore inserted new threads and returned no slots to the Reasoner.

## 11. Evidence Retrieved Latents Did Not Enter Weaver

Runtime result:

- Selected retrieved slots: 0 on all turns
- Retrieved latent tokens: 0 on all turns
- `retrieved_latents_enter_weaver=false` on all turns
- `retrieved_latents_enter_reasoner=false` on all turns, correctly reflecting that thresholding selected nothing

This runtime evidence is necessarily vacuous for positive retrieval: there were no retrieved latents to route. The existing integration test `test_retrieved_memory_only_injected_into_reasoner_and_stores_reasoner_space_latent` supplies the non-vacuous positive-retrieval boundary check. It verifies that a retrieved latent expands Reasoner input, does not expand Weaver input, and that the written tensor is the reasoner-space projection. That integration suite passed in this milestone.

No claim is made that this particular MAB-3 run exercised retrieved-latent injection.

## 12. Trigger and Weaver Status

- Trigger module present: yes
- Original fixed-invoke configuration: `trigger.active=false`
- Trigger decisions: 1 per turn
- Trigger positive decisions: 1 per turn
- Weaver prompt calls: 3
- Weaver inference calls: 0
- New bank latent tokens written: 24 total

Trigger/Weaver behavior matches MAB-2. The absence of retrieval was caused by the bank threshold, not by Trigger or Weaver being disabled.

## 13. Context-Length Preflight

- Reasoner capacity: 32768 tokens
- Largest visible prompt: 7677 tokens
- Guard reserves 8 new prompt latents, up to 8 retrieved latents, and 10 output tokens
- Largest guarded total: 7703 tokens
- Result: pass

Full history was present and no truncation occurred.

## 14. Errors and Stop Reason

- Canonical manifest status: `success`
- Stop reason: none
- Official MAB scoring completed.
- The initial instrumentation run revealed that raw below-threshold scores were omitted from `top_retrieval_scores`. Root cause: diagnostics selected `retrieved_scores` rather than the complete `scores` vector. A failing regression test was added, the mapping was corrected, and the identical bounded run was repeated.
- The official chunker calls `nltk.download('punkt', quiet=True)` and emitted a blocked network-attempt warning during preflight; cached NLTK data still produced the validated chunks. The model runs remained offline.
- Existing Transformers model-type and ignored-temperature warnings were non-fatal and unchanged from MAB-2.

## 15. Test Results

Focused MAB-3 tests:

- 7 tests passed after the diagnostics regression fix.

Combined relevant suite:

- 77 tests passed.
- Included LatentMemoryBank unit tests, model/bank integration tests, MAB-2 harness tests, MAB-3 harness tests, and R4 dynamic harness tests.

Syntax compilation and `git diff --check` passed.

## 16. Git Status Before and After

Before MAB-3:

```text
## rlm-memory-bank...origin/rlm-memory-bank [ahead 8]
 M memgen/model/modeling_memgen.py
?? research_notes/benchmarks/
?? scripts/eval/mab2_bank_off.py
?? scripts/eval/mab2_mab_bridge.py
?? tests/test_mab2_bank_off.py
```

The existing 5-addition/5-deletion modification in `memgen/model/modeling_memgen.py` predates MAB-3 and was not touched or reverted.

After MAB-3:

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

MemoryAgentBench core and MemGen model core were not modified by MAB-3. The only shared MAB-2 file change is the backward-compatible optional timestamp argument in the external bridge. No commit or push was performed.

## 17. Recommendation

Do not move to compressed-memory yet. This one-sample comparison is structurally stable but did not exercise retrieval, so it cannot evaluate the intended benefit of Bank-on.

Next, run a small preregistered full-history threshold ablation on a development subset, using observed score scale to include values such as `0.0`, `0.03`, and `0.05` alongside the current `0.7`. The objective is to obtain nonzero query retrieval while preserving the Reasoner-only boundary. After selecting a threshold without using test-answer correctness as the selection criterion, expand the paired Bank-off/Bank-on comparison to more contexts. Compressed-memory should follow only after full-history Bank-on reliably performs actual retrieval.
