# MAB-2: Original MemGen Full-history Rebuild Bank-off

Date: 2026-06-20  
Status: completed, valid one-context run  
Run ID: `20260620T034034Z-factconsolidation-sh-6k-onectx`

## 1. Objective

Run one real MemoryAgentBench context and its first query through original MemGen as one three-turn episode. The experiment validates the thin MAB-to-MemGen harness, full dialogue-history rebuild, official MAB scoring, and complete absence of the added session-level LatentMemoryBank.

This run does not implement or enable Bank-on behavior, compressed-memory behavior, cross-turn KV reuse, or a strict no-history diagnostic.

## 2. Baseline Name and Meaning

The exact baseline name is **Original MemGen Full-history Rebuild Bank-off**.

- It is not a no-memory baseline: both MAB chunks remain available in the rebuilt prompt history at query time.
- Original MemGen Trigger/Weaver behavior remains present.
- The added session-level LatentMemoryBank is off and is never created.
- Every turn is a new `MemGenModel.generate()` call over the complete message history accumulated by one `run_agent_loop()` episode.
- No KV cache is reused across turns.

The TriviaQA Weaver-SFT configuration has `trigger.active=false`. In this codebase that flag selects the original fixed-invoke Trigger behavior rather than suppressing augmentation: the observed augmentation mask contained one positive prompt-time Trigger decision on every turn, and Weaver prompt augmentation ran three times. Trigger/Weaver were therefore not disabled by the harness.

## 3. Data and Task

- Local dataset root: `/mnt/18T/baishilong/datasets/MemoryAgentBench`
- Parquet split file: `data/Conflict_Resolution-00000-of-00001.parquet`
- Split: `Conflict_Resolution`
- Filter: `metadata.source == factconsolidation_sh_6k`
- Selected contexts: 1 of 1 matched row, from 8 total rows
- Selected queries: first query only
- Official chunk size: 4096
- Official chunk count: 2
- Official tiktoken lengths: `[4319, 2119]`
- Official template: `factconsolidation/long_context_agent`
- Official metric: `substring_exact_match`, with generic EM/F1/ROUGE fields
- MemoryAgentBench commit: `455306dcabc3842526eb83cd4e225e5d486c5c5d`

The official chunker reproduces the MAB-1A result, including the first chunk exceeding the requested size because of the benchmark's sentence accumulation behavior. The harness preserves this behavior rather than rechunking.

## 4. Exact Commands

Unit and syntax checks:

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python -m unittest tests.test_mab2_bank_off
/home/baishilong/miniconda3/envs/memgen/bin/python -m py_compile scripts/eval/mab2_bank_off.py
/home/baishilong/miniconda3/envs/MABench/bin/python -m py_compile scripts/eval/mab2_mab_bridge.py
git diff --check -- scripts/eval/mab2_bank_off.py scripts/eval/mab2_mab_bridge.py tests/test_mab2_bank_off.py
```

Real-data bridge preflight:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  /home/baishilong/miniconda3/envs/MABench/bin/python \
  scripts/eval/mab2_mab_bridge.py prepare \
  --mab-repo /mnt/18T/baishilong/benchmarks/MemoryAgentBench \
  --output /tmp/mab2_payload_preflight.json \
  --parquet /mnt/18T/baishilong/datasets/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet \
  --data-config /mnt/18T/baishilong/benchmarks/MemoryAgentBench/configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml \
  --sub-dataset factconsolidation_sh_6k \
  --chunk-size 4096
```

One-context MemGen run. `MEMGEN_CHECKPOINT` is the local checkout of the public checkpoint identified in the manifest; its private/local path is intentionally not copied into this note.

```bash
CUDA_VISIBLE_DEVICES=1 \
HF_HUB_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH=/mnt/18T/baishilong/MemGen \
/home/baishilong/miniconda3/envs/memgen/bin/python \
scripts/eval/mab2_bank_off.py \
  --dataset-root /mnt/18T/baishilong/datasets/MemoryAgentBench \
  --mab-repo /mnt/18T/baishilong/benchmarks/MemoryAgentBench \
  --mab-python /home/baishilong/miniconda3/envs/MABench/bin/python \
  --checkpoint-path "$MEMGEN_CHECKPOINT" \
  --model-checkpoint-id 'Kana-s/MemGen@269d9b1/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model'
```

No external API was called. Model and dataset access were local/offline.

## 5. Output Artifact

Run directory:

`outputs/mab/memgen_bank_off/20260620T034034Z-factconsolidation-sh-6k-onectx/`

Files:

- `manifest.json`
- `results.json`
- `diagnostics.jsonl`
- `run_config.json`

Full context, full prompts, and memorization acknowledgements are not persisted in these artifacts. Prompt hashes and token lengths provide audit evidence without duplicating benchmark text.

## 6. Result

- `substring_exact_match`: `0`
- `exact_match`: `0`
- `f1`: `0`
- ROUGE fields: `0`
- Query input length: 7677 tokens
- Query output length: 10 tokens
- Memorization-turn generation time: 2.2006 seconds total
- Query generation time: 1.0791 seconds

The generated answer did not match the first gold answer. This is a valid model result: the episode, history checks, and official metric path all completed successfully.

## 7. Prompt, History, and Cache Policy

- `history_policy`: `full_rebuild`
- One MAB context maps to exactly one `run_agent_loop()` call.
- Episode turns: chunk 1, chunk 2, then query 1.
- Rendered prompt-history token lengths: `[4972, 7497, 7677]`.
- The final query render contains both earlier chunk texts and both assistant acknowledgement turns.
- `full_history_included=true` for all three diagnostics records.
- `cross_turn_kv_reuse=false`.
- `intra_generation_kv_cache=false` for this checkpoint/configuration path. After prompt-time augmentation, `max_inference_aug_num=0` routes to the original continuation generation with `use_cache=false`; the harness did not alter that behavior.
- No prompt truncation occurred.

## 8. Bank-off Invariant Check

The manifest and all three turn diagnostics agree:

- `bank_enabled=false`
- `bank_created=false`
- `bank_write_count=0`
- `bank_retrieval_count=0`
- `bank_slot_count=0`

The interaction manager created no bank, every model call received `latent_memory_bank=None`, and one session handled the complete context/query episode.

## 9. Trigger and Weaver Status

- Trigger module present: yes
- Configured Trigger policy: fixed-invoke (`trigger.active=false` in original TriviaQA config)
- Trigger decision count: 1 per turn
- Trigger positive count: 1 per turn
- Weaver prompt calls: 3 total
- Weaver inference calls: 0 total, consistent with `max_inference_aug_num=0`

These observations distinguish original prompt-time latent augmentation from the added LatentMemoryBank. Prompt-time Weaver latents were generated and injected within each independent generation call; they were not written to or retrieved from a cross-turn bank.

## 10. Context-Length Preflight

- Reasoner declared capacity: 32768 tokens
- Largest rendered history: 7677 tokens
- Guarded requirement: rendered history + 8 prompt latents + 10 generated tokens must not exceed capacity
- Final guarded total: 7695 tokens
- Result: pass, with no silent truncation

The harness also verifies that every expected previous chunk is present in the query-turn messages before generation.

## 11. Errors and Stop Reason

- Run stop reason: none
- Manifest status: `success`
- Official metric scoring completed.
- Transformers emitted an existing model-type compatibility warning and a warning that deterministic generation ignores `temperature`; neither stopped or altered the configured greedy run.
- The standalone bridge preflight showed an NLTK download-attempt warning under restricted networking even though cached tokenizer data allowed chunking to complete. The actual benchmark run remained offline and reproduced the expected chunks.

## 12. Git Status Before and After

Before implementation:

```text
## rlm-memory-bank...origin/rlm-memory-bank [ahead 8]
 M memgen/model/modeling_memgen.py
?? research_notes/benchmarks/
```

The tracked modification in `memgen/model/modeling_memgen.py` existed before MAB-2 and was not touched, reverted, or reformatted.

After the run and note:

```text
## rlm-memory-bank...origin/rlm-memory-bank [ahead 8]
 M memgen/model/modeling_memgen.py
?? research_notes/benchmarks/
?? scripts/eval/mab2_bank_off.py
?? scripts/eval/mab2_mab_bridge.py
?? tests/test_mab2_bank_off.py
```

New MAB-2 implementation files are the thin harness, MAB bridge, and focused unit test. No MemGen model core or MemoryAgentBench core file was modified. No commit or push was performed.

## 13. MAB-3 Readiness

MAB-3 Full-history Bank-on is ready to begin from an evaluation-flow perspective: local data loading, official chunking/templates/metrics, one-session multi-turn mapping, prompt-history auditing, context-capacity guards, and artifact writing are validated.

The MAB-2 score of zero is not a blocker; it is the baseline observation MAB-3 should compare against. Before MAB-3 executes, its harness changes must add and test Bank-on lifecycle/counter assertions without weakening the MAB-2 Bank-off guards. MAB-3 was not started in this milestone.
