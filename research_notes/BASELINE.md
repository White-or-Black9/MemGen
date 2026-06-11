# Baseline Definition

## Status

- Acceptance target: `comparison_ready`
- Current verdict: `ready_for_phase3`
- Baseline gate: **closed**
- Blocking defect: none for smoke execution; formal Phase 3 evidence is pending
- Failed smoke experiment: `EXP-20260611-001`
- Successful repair smoke experiment: `EXP-20260611-004`

The Repair Phase smoke is not a scientific baseline. No paper-facing or
downstream comparison may use it as an aggregate result.

## Baseline Identity

- Baseline ID: `memgen-gsm8k-sft-official-v1`
- Date selected: 2026-06-11
- Code revision: `5e59fee296092fa056f140b38a07b927651ffdb5`
- Upstream code parent: `7f1444f`
- Branch: `rlm-memory-bank`
- Working tree during audit: clean
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Dataset: `gsm8k`, configuration `main`, split `test`
- MemGen checkpoint:
  `Kana-s/MemGen@31372b691e334d578de0d78fd9aa01d7da025940`
- Variant: `gsm8k/weaver-sft/pn=1_pl=8_in=3_il=8`
- Local ignored path: `.cache/baselines/memgen-gsm8k-sft/model`
- Inference entry point: `main.py -> MemGenRunner.evaluate()`
- Primary metric: mean `compute_reward`, higher is better
- Decoding: greedy, temperature `0.0`
- Seed: `42`
- Phase 1 default batch size: `1`

## Checkpoint Integrity

| File | SHA-256 |
|---|---|
| `projs.bin` | `54a342a64e62dc6e7b1837a169daec1dfa79f8ac6d787c3bfe1206926f3bb754` |
| `weaver.bin` | `5c80b886508e3eae60648728d9a70a03ec58cc70732f838aca5c5030a4828e04` |
| `trigger.bin` | `7d9f14695be9f6c1e3355ab147aa55b20ce5e594bd9111aab10c314dc8a0630c` |
| Weaver adapter | `1ef3f419d79e73a6cc805c74546f270b90857a5ba93130793b9caafad1eb4f95` |
| Trigger adapter | `ab97fcac48622ad332106277ac58ec732930927b039da1facd5fb211bbc65cbc` |

These hashes match the official Hugging Face LFS metadata.

## Environment

- GPU host: 8 x NVIDIA RTX A6000, 49140 MiB each.
- Driver: `560.35.03`.
- Intended environment: `/home/baishilong/miniconda3/envs/memgen`.
- Python: `3.10.20`.
- PyTorch: `2.12.0+cu126` locally installed.
- Transformers: `4.55.4`.
- PEFT: `0.17.1`.
- TRL: `0.21.0`.
- Flash Attention: `2.8.3`.
- Known deviation: repository manifests disagree on PyTorch CUDA build and TRL
  version. This must remain recorded until the baseline is accepted.

## Reproduction Command

The canonical full evaluation command is:

```bash
CUDA_VISIBLE_DEVICES=7 \
/home/baishilong/miniconda3/envs/memgen/bin/python \
  -m accelerate.commands.launch \
  --config_file=configs/zero2.yaml \
  --num_processes=1 \
  main.py \
  --cfg-path configs/latent_memory/gsm8k.yaml \
  --options \
  model.model_name Qwen/Qwen2.5-1.5B-Instruct \
  model.load_model_path .cache/baselines/memgen-gsm8k-sft/model \
  model.max_prompt_aug_num 1 \
  model.max_inference_aug_num 3 \
  model.weaver.model_name Qwen/Qwen2.5-1.5B-Instruct \
  model.weaver.prompt_latents_len 8 \
  model.weaver.inference_latents_len 8 \
  model.trigger.model_name Qwen/Qwen2.5-1.5B-Instruct \
  model.trigger.active False \
  run.mode evaluate \
  run.interaction.batch_size 1 \
  run.interaction.temperature 0.0 \
  run.interaction.max_response_length 1024
```

`BUG-0001` and `BUG-0002` are repaired. Execute the full baseline only in an
explicitly approved Phase 3.

## Repair Smoke Evidence

- Official path:
  `Config -> MemGenModel.from_config -> MemGenRunner.evaluate()`
- Sample count: 1
- Batch size: 1
- Seed: 42
- Weaver adapter: exact 112/112 tensor match
- Trigger adapter: exact 112/112 tensor match
- Missing/unexpected/shape/value mismatches: none
- Static output:
  `outputs/baseline/EXP-20260611-004/evaluate/answer.json`
- Output status: non-empty; one prediction plus one summary record
- Generation path evidence:
  - Trigger decision entry called 85 times
  - Weaver prompt augmentation called once
  - Weaver inference augmentation called three times
- Verification artifact:
  `outputs/baseline/EXP-20260611-004/verification.json`

## Compatibility Contract

With `latent_memory_bank.enabled=false`:

- The call graph must take the original inference path without constructing,
  retrieving, updating, or resetting a memory bank.
- Original configuration files and commands remain valid.
- Input IDs, attention masks, generation configuration, and checkpoint loading
  are unchanged.
- For deterministic `batch_size=1` golden cases, generated token IDs,
  augmentation masks, and task metrics must match byte-for-byte.
- Dtype, device, tensor shapes, KV-cache behavior, and output schema remain unchanged.
- Weaver and Trigger training files, parameters, commands, and checkpoints remain
  untouched.

## Required Golden Evidence

After `BUG-0001` is fixed:

1. Confirm adapter tensors load with no missing/unexpected keys.
2. Run at least three fixed GSM8K test examples with greedy decoding.
3. Store generated token IDs, augmentation masks, output hashes, rewards, latency,
   and peak CUDA memory.
4. Re-run each case and require identical token and mask hashes.
5. Use these artifacts as the disabled-feature equivalence oracle in Phase 1.

## Metrics

| Metric | Current Value | Status |
|---|---:|---|
| Mean `compute_reward` on GSM8K test | Not measured | Blocked |
| Golden token hash | Not established | Blocked |
| Latency per sample | Not established | Blocked |
| Peak CUDA memory | Not established | Blocked |
| Throughput | Not established | Blocked |

## Acceptance Criteria

- [x] Comparator identity and source revision are explicit.
- [x] Dataset, split, evaluation path, metric, and direction are explicit.
- [x] Official checkpoint files are local and hash-verified.
- [x] Canonical metric contract exists.
- [x] Checkpoint loads all trained adapter tensors.
- [ ] Full baseline command completes.
- [ ] Metrics and raw outputs are archived.
- [ ] Deterministic golden cases are established.
- [ ] Baseline is accepted as `comparison_ready`.
