# Baseline Definition

## Status

- Acceptance target: `comparison_ready`
- Current verdict: `verified_match`
- Baseline gate: **open**
- Acceptance status: `comparison_ready`
- Blocking defect: none
- Accepted experiment: `EXP-20260611-006`
- Golden replay: `EXP-20260611-007`
- Failed smoke experiment: `EXP-20260611-001`
- Successful repair smoke experiment: `EXP-20260611-004`

The accepted metric is scoped to the fixed 20-sample GSM8K subset. It is the
Original MemGen comparator for later LatentMemoryBank experiments using the same
sample IDs and protocol.

## Baseline Identity

- Baseline ID: `memgen-gsm8k-sft-official-v1`
- Date selected: 2026-06-11
- Core code revision: `c0f1f2c3d79828c2d4e4f74eb9756bfb50890653`
- Upstream code parent: `7f1444f`
- Branch: `rlm-memory-bank`
- Working tree during accepted run: evaluation-harness changes only
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
- Batch size: `1`
- Fixed comparison sample IDs: `0..19`
- Maximum response length: `1024`
- Accepted output: `outputs/baseline/EXP-20260611-006`

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

## Accepted Reproduction Command

The canonical full evaluation command is:

```bash
env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  CUDA_VISIBLE_DEVICES=0 \
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  /home/baishilong/miniconda3/envs/memgen/bin/python \
  -m scripts.eval.repair_phase2_smoke \
  --cfg-path configs/latent_memory/gsm8k.yaml \
  --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model \
  --output-dir /mnt/18T/baishilong/MemGen/outputs/baseline/EXP-20260611-006 \
  --sample-start 0 --sample-count 20 --max-response-length 1024
```

The harness still constructs `Config`, calls `MemGenModel.from_config()`, and
executes `MemGenRunner.evaluate()`; it adds verification and fixed-subset
selection without bypassing the official evaluator.

## Accepted Results

| Metric | Value |
|---|---:|
| Mean `compute_reward` | 0.60 |
| Correct / total | 12 / 20 |
| Total latency | 115.728 s |
| Mean latency/sample | 5.786 s |
| Peak allocated CUDA memory | 9,415,716,352 bytes |
| Trigger decision calls | 1,722 |
| Weaver prompt calls | 20 |
| Weaver inference calls | 43 |

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

## Artifacts

- Predictions:
  `outputs/baseline/EXP-20260611-006/evaluate/answer.json`
- Verification, adapter checks, token hashes, masks, latency, and memory:
  `outputs/baseline/EXP-20260611-006/verification.json`
- Canonical comparison contract:
  `outputs/baseline/EXP-20260611-006/json/metric_contract.json`
- Golden replay:
  `outputs/baseline/EXP-20260611-007/verification.json`

## Scope

- This baseline is accepted for comparisons on GSM8K test indices 0 through 19.
- Later variants must use the same sample IDs, seed, checkpoint, decoding
  parameters, evaluator, and batch size.
- The value 0.60 must not be reported as full-test GSM8K accuracy.
- A 50-sample or full-test run is optional stronger evidence, not required to
  begin method development.

## Acceptance Criteria

- [x] Comparator identity and source revision are explicit.
- [x] Dataset, split, evaluation path, metric, and direction are explicit.
- [x] Official checkpoint files are local and hash-verified.
- [x] Canonical metric contract exists.
- [x] Checkpoint loads all trained adapter tensors.
- [x] Baseline command completes.
- [x] Metrics and raw outputs are archived.
- [x] Deterministic golden cases are established.
- [x] Baseline is accepted as `comparison_ready`.

## End-of-Day Artifact Revalidation

Validated on 2026-06-11 without rerunning inference:

- `EXP-20260611-006/evaluate/answer.json`: non-empty JSONL, 20 prediction
  records, one summary, `compute_reward=0.60`.
- `EXP-20260611-006/verification.json`: readable; Weaver 112/112 and Trigger
  112/112, with empty missing/unexpected/shape/value mismatch lists.
- `EXP-20260611-007/evaluate/answer.json`: three prediction records and one
  summary.
- `EXP-20260611-007/verification.json`: readable and scoped to sample IDs
  0, 1, and 2.
- Verdict: the Original MemGen fixed-20 baseline remains `comparison_ready`.
