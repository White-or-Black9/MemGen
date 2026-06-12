# Experiment Log

Record every experiment, including failed, aborted, and exploratory runs. Never
overwrite prior records; append a new entry.

## Experiment Index

| ID | Date | Phase | Question | Status | Key Result |
|---|---|---|---|---|---|
| EXP-20260611-001 | 2026-06-11 | Phase 0 | Can the official GSM8K SFT checkpoint produce a trusted smoke baseline? | `failed` | LoRA keys were not loaded; direct smoke then failed on projection dtype |
| EXP-20260611-002 | 2026-06-11 | Phase 2 | Can the original MemGen inference stack run a one-sample GSM8K smoke test in the recommended environment? | `completed_with_caveats` | Original eval path reached generation but crashed in static recorder; script-only harness produced one completion; result is not a valid baseline because LoRA loading remains broken |
| EXP-20260611-003 | 2026-06-11 | Environment Alignment | Is the existing `memgen` environment suitable for the Repair Phase without package changes? | `completed` | Imports, dependency check, CUDA/BF16, config, model snapshot, checkpoint, and dataset cache validated; no install required |
| EXP-20260611-004 | 2026-06-11 | Temporary Repair | Do the minimal adapter-loader and static-recorder fixes unblock the official one-sample smoke path? | `completed` | Both adapters matched 112/112 tensors exactly; official static eval wrote a non-empty answer file |
| EXP-20260611-005 | 2026-06-11 | Repair Review | Do the repaired loader and recorder remain correct across three sequential batch-size-1 samples? | `completed` | Three predictions plus one summary were written; adapter and augmentation checks passed |
| EXP-20260611-006 | 2026-06-11 | Phase 3 | What is Original MemGen performance on the fixed 20-sample GSM8K comparison subset? | `completed` | 20/20 predictions completed; mean `compute_reward=0.60` |
| EXP-20260611-007 | 2026-06-11 | Phase 3 | Are the fixed golden outputs deterministic under exact replay? | `completed` | Samples 0-2 reproduced identical response-token and augmentation-mask hashes |
| EXP-20260611-008 | 2026-06-11 | Phase 4 | Does the standalone memory-bank skeleton satisfy its tensor, retrieval, capacity, and isolation contracts? | `completed` | 16/16 unit tests passed after cleanup; production inference and training references remained absent |
| EXP-20260611-009 | 2026-06-11 | End-of-Day Validation | Are the Repair fixes, Phase 3 baseline artifacts, and Phase 4 skeleton ready for commit and later continuation? | `completed` | Compilation and 16/16 tests passed; baseline/golden artifacts and adapter evidence remained complete; Phase 4 remained isolated |
| EXP-20260612-010 | 2026-06-12 | Phase 5 | Does `latent_memory_bank.enabled=false` preserve the exact Phase 3 golden behavior after Version A integration? | `completed` | Samples 0-2 matched Phase 3 response-token hashes, augmentation-mask hashes, and Trigger/Weaver call counts exactly |
| EXP-20260612-011 | 2026-06-12 | Phase 5 | Does enabled Version A run on one sample without crashing and produce separate memory write/retrieve bookkeeping? | `completed` | One-sample debug completed with 4 writes, 3 retrievals, 24 retrieved latent tokens, 32 new latent tokens, and 4 resident slots |
| EXP-20260612-013 | 2026-06-12 | Phase 6 | Does the full 20-sample disabled path remain exactly equivalent to the frozen Phase 3 baseline? | `completed` | All 20 response-token hashes, all 20 augmentation-mask hashes, summary metric, and Trigger/Weaver call counts matched `EXP-20260611-006` exactly |
| EXP-20260612-014 | 2026-06-12 | Phase 7 | Does enabled Tier 1 smoke run complete before adding per-session debug trace? | `completed_with_caveats` | One-sample enabled run succeeded, then was superseded by `EXP-20260612-015` to capture session-level initial-slot evidence |
| EXP-20260612-015 | 2026-06-12 | Phase 7 | Does enabled Tier 1 smoke run complete with correct Version A debug and session-local evidence? | `completed` | One-sample enabled run completed with `initial_slots=0`, 4 writes, 3 retrievals, 24 retrieved latent tokens, and Reasoner-only injection evidence |
| EXP-20260612-016 | 2026-06-12 | Phase 7 | Do three enabled single-turn sessions remain isolated and stable on GSM8K samples 0..2? | `completed` | All three sessions started with `initial_slots=0`; no cross-sample leakage, no tensor errors, and slot count stayed within bounds |
| EXP-20260612-017 | 2026-06-12 | Phase 7 | Does enabled Version A remain stable on a bounded five-sample run without exceeding slot limits? | `completed` | Five enabled sessions completed without crash or leakage; slot count never exceeded 4 and no replacement-policy activation was needed |
| EXP-20260612-018 | 2026-06-12 | Phase 7 Supplement | Can the real enabled inference path be forced to trigger replacement by lowering `max_slots` to 2? | `completed` | One enabled sample completed with `memory_write_count=4`, `slot_count=2`, and `update_action_trace=[append, append, replace, replace]` |
| EXP-20260612-019 | 2026-06-12 | Phase 8A G1 | Does the Version A-simple anchor run stably on the fixed GSM8K pilot slice? | `completed` | Stable 20-sample run; `compute_reward=0.50` (`10/20`) |
| EXP-20260612-020 | 2026-06-12 | Phase 8A G4 | What changes when current write-age decay is disabled? | `completed` | Stable 20-sample run; `compute_reward=0.50` (`10/20`); this is not a last-retrieved-decay comparison |
| EXP-20260612-021 | 2026-06-12 | Phase 8A G6 | Does append-only update run stably on the pilot slice? | `completed` | Stable 20-sample run; `compute_reward=0.50` (`10/20`); capacity did not saturate |
| EXP-20260612-022 | 2026-06-12 | Phase 8A G7 | Does the legacy replace policy run stably on the pilot slice? | `completed` | Stable 20-sample run; `compute_reward=0.50` (`10/20`); `replace_count=0` |
| EXP-20260612-023-step3-disabled-replay | 2026-06-12 | Step 3 | Does disabled behavior remain exact after `thread_update` integration? | `completed` | Samples 0-2 exactly matched frozen response-token hashes, augmentation-mask hashes, and Trigger/Weaver call counts |
| EXP-20260612-024-thread-update-smoke | 2026-06-12 | Step 4 | Does Version A-aligned `thread_update` operate correctly on the real enabled inference path? | `completed` | Mechanism smoke only: one empty-bank insert and three current-argmax matched replacements; Reasoner-only and reasoner-space boundaries held |

## Recorded Experiments

### EXP-20260611-001: Official GSM8K SFT Smoke Baseline

- Phase: 0
- Status: `failed`
- Research question: Can the official checkpoint be loaded faithfully and run on
  one deterministic GSM8K test sample?
- Hypothesis: Official assets plus the documented environment are sufficient for
  a trusted local comparator.
- Baseline/comparator: `memgen-gsm8k-sft-official-v1`
- Code revision: `5e59fee296092fa056f140b38a07b927651ffdb5`
- Working tree state: clean before note updates
- Environment: Python 3.10.20, PyTorch 2.12.0+cu126, Transformers 4.55.4,
  PEFT 0.17.1, RTX A6000
- Dataset and split: cached `gsm8k/main`, test sample index 0
- Configuration: prompt augmentation 1, inference augmentation 3, latent lengths
  8/8, inactive Trigger, greedy decoding, maximum 128 new tokens
- Random seed: 42
- Batch size: 1
- Checkpoint: `.cache/baselines/memgen-gsm8k-sft/model`
- Raw artifact: none; run terminated before generation output

#### Observations

- Official file SHA-256 values matched Hugging Face LFS metadata.
- PEFT warned that all expected named `weaver` and `trigger` adapter keys were
  missing while loading.
- Checkpoint tensors use keys without adapter-name suffixes, while the nested
  loaded model expected keys such as `lora_A.weaver.weight`.
- Direct `MemGenModel.generate()` then failed because reasoner embeddings were
  BF16 while projection weights remained FP32. The normal runner converts the
  whole model to BF16, so this dtype failure is a smoke harness issue, not the
  primary baseline blocker.
- No metric, completion, token hash, or latency result is valid.

#### Conclusion

- Hypothesis supported: No.
- Interpretation: The current checkpoint-loading path cannot establish a trusted
  comparator because trained LoRA tensors are silently skipped.
- Follow-up: Repair and test checkpoint loading in a separately approved Phase
  before running the baseline again.
- Related decisions: `DEC-0005`, `DEC-0006`
- Related bug: `BUG-0001`

### EXP-20260611-002: Phase 2 Original Project Smoke Test

- Phase: 2
- Status: `completed_with_caveats`
- Research question: Can the current repository run the original MemGen
  inference path on a minimal GSM8K sample in the recommended local environment?
- Hypothesis: With the correct local environment, local model snapshot, local
  dataset cache, and `batch_size=1`, the original inference stack should at
  least reach generation.
- Baseline/comparator: none; this is a smoke test only
- Code revision: `7b8b9a44eb30325a676a6c9576c35b3a10b52c32`
- Working tree state: research-note changes present; no core-code edits
- Environment: `/home/baishilong/miniconda3/envs/memgen`, Python 3.10.20,
  PyTorch 2.12.0+cu126, Transformers 4.55.4, PEFT 0.17.1, FlashAttention 2.8.3,
  single RTX A6000
- Dataset and split: cached `gsm8k/main`, test sample count `1`
- Inputs/session definition: one GSM8K test item, greedy decoding, static
  single-turn interaction
- Configuration: `configs/latent_memory/gsm8k.yaml`, prompt aug `1`, inference
  aug `3`, latents `8/8`, inactive Trigger, `batch_size=1`,
  `max_response_length=128`
- Random seeds: `42`
- Model path: local Qwen snapshot
  `/home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Checkpoint path: `.cache/baselines/memgen-gsm8k-sft/model`
- Dataset path:
  `/home/baishilong/.cache/huggingface/datasets/gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866`
- Command:
  1. Original-eval smoke harness using `Config -> MemGenModel.from_config -> MemGenRunner.evaluate()`, with `runner.test_dataset = runner.test_dataset.select(range(1))`
  2. Script-only manual harness using the same model/config/interaction path but bypassing the broken static recorder
- Output directory:
  - original eval run:
    `.cache/evaluate/gsm8k/home/pn=1_pl=8_in=3_il=8_20260611-103526_phase2_smoke`
  - manual completion run:
    `.cache/evaluate/gsm8k/home/pn=1_pl=8_in=3_il=8_20260611-104054_phase2_manual_smoke_cuda`
- Start/end time: 2026-06-11 Phase 2 execution window

#### Observations

- Quantitative:
  - original eval path processed one test sample and entered generation
  - original eval output file `evaluate/answer.json` was created but remained
    empty (`0` bytes)
- Qualitative:
  - original runner path failed after generation in `StaticEvalRecorder.record_batch`
  - manual harness produced a completion ending with `\\boxed{18}`
- Runtime/latency:
  - original runner reached the single-step progress bar and failed after about
    8 seconds on the only sample
- Peak memory:
  - not measured in this Phase
- Failures or anomalies:
  - inherited proxy and `HF_ENDPOINT` variables caused offline cache misses until
    they were unset
  - sandboxed execution hid CUDA from PyTorch, so GPU-backed smoke verification
    required unsandboxed execution
  - `BUG-0001` remained reproducible
  - `BUG-0002` was newly confirmed on the original static evaluation path

#### Conclusion

- Hypothesis supported: partially
- Interpretation: The recommended local environment can initialize the original
  MemGen project, load the cached base model and dataset, and reach real
  generation on one sample. However, the official static evaluation path is not
  currently end-to-end runnable because result recording crashes.
- Limitations:
  - not a scientific baseline
  - no trustworthy LoRA-loading guarantee
  - no aggregate metric should be used
- Follow-up:
  - repair `BUG-0001` and `BUG-0002` in a separately approved Phase
  - rerun the same one-sample smoke test before Phase 3
- Related decision IDs: `DEC-0005`, `DEC-0006`, `DEC-0009`
- Related bug IDs: `BUG-0001`, `BUG-0002`
- Artifacts:
  - empty original output:
    `.cache/evaluate/gsm8k/home/pn=1_pl=8_in=3_il=8_20260611-103526_phase2_smoke/evaluate/answer.json`
  - original run log:
    `.cache/evaluate/gsm8k/home/pn=1_pl=8_in=3_il=8_20260611-103526_phase2_smoke/logs/log.txt`
  - manual completion artifact:
    `.cache/evaluate/gsm8k/home/pn=1_pl=8_in=3_il=8_20260611-104054_phase2_manual_smoke_cuda/evaluate/manual_answer.json`

### EXP-20260611-003: Existing Environment Alignment Validation

- Phase: Temporary Environment Alignment Phase
- Status: `completed`
- Research question: Can the existing `memgen` environment be used as the
  controlled runtime for repairing `BUG-0001` and `BUG-0002` without changing
  installed packages?
- Hypothesis: The existing Python 3.10 environment is sufficient because it
  already reached real GPU generation in Phase 2.
- Baseline/comparator: checked-in `requirements.txt` and `memgen.yml`
- Code revision: `dd6eda02c3c06823670e217c8b0217199b24235c`
- Git branch: `rlm-memory-bank`
- Working tree state: clean before environment-alignment note updates
- Environment:
  `/home/baishilong/miniconda3/envs/memgen`, Python `3.10.20`
- Config file: `configs/latent_memory/gsm8k.yaml`
- Model path:
  `/home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Checkpoint path: `.cache/baselines/memgen-gsm8k-sft/model`
- Dataset path:
  `/home/baishilong/.cache/huggingface/datasets/gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866`
- Dataset/sample count: no samples evaluated
- Random seed: not applicable
- Decoding parameters: not applicable
- Output directory: none
- Prediction file: none
- Metric file: none
- Commands:
  - `/home/baishilong/miniconda3/bin/conda env list`
  - `python --version`
  - `/home/baishilong/miniconda3/envs/memgen/bin/python --version`
  - `/home/baishilong/miniconda3/envs/memgen/bin/python -c "import torch; ..."`
  - `/home/baishilong/miniconda3/envs/memgen/bin/python -c "import transformers, peft, accelerate, datasets; ..."`
  - `/home/baishilong/miniconda3/envs/memgen/bin/python -m pip check`
  - OmegaConf load of `configs/latent_memory/gsm8k.yaml`
  - filesystem readability checks for model, checkpoint, and dataset caches
- Latency: not measured
- Memory usage: not measured

#### Observations

- Base environment:
  - Python `3.13.9`
  - active prefix `/home/baishilong/miniconda3`
  - unsuitable for MemGen execution
- Existing `memgen` environment:
  - Python `3.10.20`
  - PyTorch `2.12.0+cu126`
  - Transformers `4.55.4`
  - PEFT `0.17.1`
  - Accelerate `1.10.1`
  - Datasets `4.0.0`
  - FlashAttention `2.8.3`
- `pip check`: no broken requirements
- CUDA outside sandbox:
  - available: `True`
  - device: NVIDIA RTX A6000
  - BF16 supported: `True`
- Sandbox-only CUDA result:
  - unavailable because the execution sandbox hides CUDA/NVML
  - this is not an environment-package failure
- Local assets:
  - Qwen snapshot readable, including single-file `model.safetensors`
  - MemGen projection, Weaver, Trigger, and adapter files readable
  - cached GSM8K loads successfully in offline mode with 7,473 train rows and
    1,319 test rows
  - a sandboxed dataset load was blocked only because Datasets attempted to
    create a lock file under the read-only home cache; the same command
    succeeded outside the sandbox without downloading data
- Environment variables:
- `HTTP_PROXY` and `HTTPS_PROXY` target `127.0.0.1:7898`
- `NO_PROXY` only covers localhost

### EXP-20260612-010: Phase 5 Disabled-Path Golden Replay

- Phase: 5
- Status: `completed`
- Research question: After Version A integration, does
  `latent_memory_bank.enabled=false` preserve the accepted Phase 3 golden
  behavior exactly?
- Hypothesis: The disabled path should remain byte-for-byte identical to
  `EXP-20260611-007` on samples `0..2`.
- Baseline/comparator: `EXP-20260611-007`
- Git branch: `rlm-memory-bank`
- Working tree state: uncommitted Phase 5 inference-only integration, tests,
  validation script, and research-note updates
- Environment:
  `/home/baishilong/miniconda3/envs/memgen`, Python `3.10.20`, PyTorch
  `2.12.0+cu126`, single NVIDIA RTX A6000 via `CUDA_VISIBLE_DEVICES=7`
- Config file: `configs/latent_memory/gsm8k.yaml`
- Optional config override:
  `run.latent_memory_bank.enabled=false`
- Model path:
  `/home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Checkpoint path:
  `/mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model`
- Dataset path:
  `/home/baishilong/.cache/huggingface/datasets/gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866`
- Dataset and split: cached `gsm8k/main`, test samples `0..2`
- Random seed: `42`
- Batch size: `1`
- Decoding: greedy, temperature `0.0`, max response length `1024`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/latent_bank_vA/EXP-20260612-010-disabled-replay --sample-start 0 --sample-count 3 --max-response-length 1024`
- Output directory:
  `outputs/latent_bank_vA/EXP-20260612-010-disabled-replay`
- Prediction file:
  `outputs/latent_bank_vA/EXP-20260612-010-disabled-replay/evaluate/answer.json`
- Verification file:
  `outputs/latent_bank_vA/EXP-20260612-010-disabled-replay/verification.json`

#### Observations

- Adapter verification remained exact: Weaver `112/112`, Trigger `112/112`.
- Response-token SHA-256 hashes matched `EXP-20260611-007` exactly for all
  three records.
- Augmentation-mask SHA-256 hashes matched `EXP-20260611-007` exactly for all
  three records.
- Trigger decision calls matched exactly: `193`.
- Weaver prompt calls matched exactly: `3`.
- Weaver inference calls matched exactly: `8`.
- `memory_bank_debug` remained `null`, confirming no bank was created on the
  disabled path.
- `answer.json` contained three prediction records and one summary record.
- Summary metric on this three-sample subset was `compute_reward=1.0`.
- Total latency was `18.026` seconds; peak allocated CUDA memory was
  `9,391,621,120` bytes.

#### Conclusion

- Hypothesis supported: yes
- Interpretation: The disabled Version A path preserved the accepted golden
  behavior exactly on samples `0..2`.
- Scope note: This is an equivalence check only; it does not replace a future
  broader Phase 6 disabled-path campaign.
- Related decisions: `DEC-0002`, `DEC-0017`, `DEC-0018`

### EXP-20260612-011: Phase 5 Enabled Version A Debug

- Phase: 5
- Status: `completed`
- Research question: Can enabled Version A run on a real GSM8K sample, write and
  retrieve session-local reasoner-space memories, and keep the mechanism within
  the intended scope?
- Hypothesis: One-sample enabled debug should complete without crashing and
  should record separate write/retrieve bookkeeping.
- Baseline/comparator: none; debug only
- Git branch: `rlm-memory-bank`
- Working tree state: uncommitted Phase 5 inference-only integration, tests,
  validation script, and research-note updates
- Environment:
  `/home/baishilong/miniconda3/envs/memgen`, Python `3.10.20`, PyTorch
  `2.12.0+cu126`, single NVIDIA RTX A6000 via `CUDA_VISIBLE_DEVICES=7`
- Config file: `configs/latent_memory/gsm8k.yaml`
- Optional config override:
  `run.latent_memory_bank.enabled=true`
- Model path:
  `/home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Checkpoint path:
  `/mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model`
- Dataset path:
  `/home/baishilong/.cache/huggingface/datasets/gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866`
- Dataset and split: cached `gsm8k/main`, test sample `0`
- Random seed: `42`
- Batch size: `1`
- Decoding: greedy, temperature `0.0`, max response length `1024`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/latent_bank_vA/EXP-20260612-011-enabled-debug --sample-start 0 --sample-count 1 --max-response-length 1024 --memory-enabled`
- Output directory:
  `outputs/latent_bank_vA/EXP-20260612-011-enabled-debug`
- Prediction file:
  `outputs/latent_bank_vA/EXP-20260612-011-enabled-debug/evaluate/answer.json`
- Verification file:
  `outputs/latent_bank_vA/EXP-20260612-011-enabled-debug/verification.json`

#### Observations

- Adapter verification remained exact: Weaver `112/112`, Trigger `112/112`.
- The run completed without crash on one sample.
- `memory_bank_debug` recorded:
  - `memory_write_count=4`
  - `memory_retrieve_count=3`
  - `retrieved_latent_count=24`
  - `new_latent_count=32`
  - `slot_count=4`
- Stored slots remained in Reasoner hidden size `1536` and were stored on CPU
  with original source device recorded as `cuda:0`.
- Weaver prompt and inference call counts were `1` and `3`.
- The trace recorded identical token counts for `reasoner_to_weaver` inputs and
  Weaver inputs on every augmentation call, consistent with retrieved memory not
  being passed into Weaver.
- `answer.json` contained one prediction record and one summary record.
- Summary metric on this one-sample debug run was `compute_reward=1.0`.
- Total latency was `9.255` seconds; peak allocated CUDA memory was
  `9,385,351,168` bytes.

#### Conclusion

- Hypothesis supported: yes
- Interpretation: Enabled Version A mechanism works on a real sample and records
  separate write/retrieve statistics without touching training code.
- Scope note: This is a mechanism debug only. It must not be treated as a
  performance or quality claim relative to the baseline.
- Related decisions: `DEC-0017`, `DEC-0018`

### EXP-20260612-013: Phase 6 Full Disabled-Path Equivalence

- Phase: 6
- Status: `completed`
- Research question: After Phase 5 integration, does the disabled path remain
  exactly equivalent to the frozen 20-sample Phase 3 baseline
  `EXP-20260611-006`?
- Hypothesis: With `latent_memory_bank` disabled, the official evaluation path
  should reproduce every frozen baseline artifact and control-flow statistic on
  GSM8K test IDs `0..19`.
- Baseline/comparator: `EXP-20260611-006`
- Git branch: `rlm-memory-bank`
- Working tree state: no Phase 6 core-code changes; only existing Phase 5 code
  and note updates present
- Environment:
  `/home/baishilong/miniconda3/envs/memgen`, Python `3.10.20`, PyTorch
  `2.12.0+cu126`, single NVIDIA RTX A6000 via `CUDA_VISIBLE_DEVICES=7`
- Config file: `configs/latent_memory/gsm8k.yaml`
- Optional config override:
  `run.latent_memory_bank.enabled=false`
- Model path:
  `/home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Checkpoint path:
  `/mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model`
- Dataset path:
  `/home/baishilong/.cache/huggingface/datasets/gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866`
- Dataset and split: cached `gsm8k/main`, test samples `0..19`
- Random seed: `42`
- Batch size: `1`
- Decoding: greedy, temperature `0.0`, max response length `1024`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/baseline/EXP-20260612-013-phase6-disabled-equivalence --sample-start 0 --sample-count 20 --max-response-length 1024 --reference-verification outputs/baseline/EXP-20260611-006/verification.json`
- Output directory:
  `outputs/baseline/EXP-20260612-013-phase6-disabled-equivalence`
- Prediction file:
  `outputs/baseline/EXP-20260612-013-phase6-disabled-equivalence/evaluate/answer.json`
- Verification file:
  `outputs/baseline/EXP-20260612-013-phase6-disabled-equivalence/verification.json`

#### Equivalence criteria

- `answer.json` exists and is non-empty
- prediction count is `20`
- one summary record exists
- summary `compute_reward` matches the baseline exactly
- every response-token SHA-256 hash matches the baseline record-by-record
- every augmentation-mask SHA-256 hash matches the baseline record-by-record
- Trigger decision call count matches exactly
- Weaver prompt augmentation call count matches exactly
- Weaver inference augmentation call count matches exactly
- adapter verification remains exact and has zero missing, unexpected, shape, or
  value mismatches
- `memory_bank_debug` remains `null`, proving no bank was constructed

#### Observations

- `answer.json` was non-empty and contained 20 prediction records plus one
  summary record.
- Summary `compute_reward=0.60`, matching `EXP-20260611-006`.
- All 20 response-token hashes matched `EXP-20260611-006` exactly.
- All 20 augmentation-mask hashes matched `EXP-20260611-006` exactly.
- Trigger decision calls matched exactly: `1722`.
- Weaver prompt calls matched exactly: `20`.
- Weaver inference calls matched exactly: `43`.
- Weaver adapter verification remained `112/112`.
- Trigger adapter verification remained `112/112`.
- Missing, unexpected, shape-mismatch, and value-mismatch lists remained empty.
- `memory_bank_debug` was `null`.
- Total latency was `96.615` seconds; peak allocated CUDA memory was
  `9,415,716,352` bytes.

#### Conclusion

- Hypothesis supported: yes
- Interpretation: The disabled path remains exactly equivalent to the frozen
  20-sample Phase 3 baseline under the accepted comparator protocol.
- Consequence: Phase 5 integration does not introduce a disabled-path
  regression on the accepted baseline.
- Related decisions: `DEC-0002`, `DEC-0017`, `DEC-0018`, `DEC-0019`
  - no `HF_ENDPOINT` was present in the final alignment shell
- Manifest differences:
  - README specifies Python 3.10
  - `memgen.yml` specifies Python 3.11.13
  - `requirements.txt` specifies PyTorch 2.7.1+cu128
  - `memgen.yml` specifies PyTorch 2.7.1+cu118
  - installed PyTorch is 2.12.0+cu126

#### Conclusion

- Hypothesis supported: yes
- Interpretation: The existing `memgen` environment is suitable for controlled
  Repair Phase work. Rebuilding or changing packages now would add risk without
  evidence of benefit.
- Limitations: The environment manifests are internally inconsistent and do not
  exactly reproduce the installed environment.
- Failures:
  - the PATH-level `/home/baishilong/bin/conda` shim has a CRLF shebang and
    cannot execute normally
- Follow-up:
  - use the direct Python path or activate with the real Miniconda activation
    script
  - preserve package versions through the Repair Phase
- Related decision IDs: `DEC-0009`, `DEC-0010`
- Related bug IDs: `BUG-0003`, `BUG-0004`

### EXP-20260611-004: Repaired Official Static Smoke Test

- Phase: Temporary Repair Phase
- Status: `completed`
- Date: 2026-06-11
- Research question: Do the minimal fixes for `BUG-0001` and `BUG-0002`
  restore a trustworthy one-sample original MemGen smoke path?
- Baseline/comparator: official Qwen2.5-1.5B GSM8K Weaver-SFT checkpoint; smoke
  verification only
- Git branch: `rlm-memory-bank`
- Base commit: `ed741d9be111b3f549740dce6db0f90c4ae11632`
- Working tree: uncommitted Repair Phase changes in the adapter loader, static
  evaluator, smoke harness, and research notes
- Environment:
  `/home/baishilong/miniconda3/envs/memgen/bin/python`
- Package versions: Python 3.10.20, PyTorch 2.12.0+cu126, Transformers 4.55.4,
  PEFT 0.17.1, Accelerate 1.10.1, Datasets 4.0.0
- Config file: `configs/latent_memory/gsm8k.yaml`
- Model path:
  `/home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Checkpoint path:
  `/mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model`
- Dataset path:
  `/home/baishilong/.cache/huggingface/datasets/gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866`
- Dataset/split/sample count: `gsm8k/main`, test index 0, one sample
- Random seed: 42
- Batch size: 1
- Decoding: greedy, temperature 0.0, maximum response length 128, Weaver and
  Trigger sampling disabled
- Output directory: `outputs/baseline/EXP-20260611-004`
- Prediction file:
  `outputs/baseline/EXP-20260611-004/evaluate/answer.json`
- Metric file: prediction file summary record plus
  `outputs/baseline/EXP-20260611-004/verification.json`
- Successful command:

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
  --output-dir /mnt/18T/baishilong/MemGen/outputs/baseline/EXP-20260611-004
```

#### Results

- Weaver adapter: 112 runtime keys and 112 checkpoint keys; zero missing,
  unexpected, shape-mismatched, or value-mismatched tensors.
- Trigger adapter: 112 runtime keys and 112 checkpoint keys; zero missing,
  unexpected, shape-mismatched, or value-mismatched tensors.
- Adapter loading warnings: none related to missing or unexpected keys.
- Prediction: `\boxed{18}` for the selected GSM8K sample.
- Metric: `compute_reward=1.0` for this one sample; no aggregate performance
  conclusion is permitted.
- `answer.json`: 1,006 bytes, two JSONL records, non-empty.
- Generation trace: Trigger decision entry 85 calls, Weaver prompt augmentation
  1 call, Weaver inference augmentation 3 calls.
- Latency: 8.438 seconds for `runner.evaluate()`.
- Peak allocated CUDA memory: 9,391,613,952 bytes.
- Initial failed launch: direct script execution raised
  `ModuleNotFoundError: common` before model loading; module execution fixed the
  harness import path without project changes.

#### Conclusion

- Both Phase 2 smoke blockers are repaired.
- This run establishes readiness to execute Phase 3, not a formal baseline.
- Related decisions: `DEC-0011`, `DEC-0012`
- Related bugs: `BUG-0001`, `BUG-0002`

#### Implementation Summary

- Adapter fix:
  - removed the constructor-created placeholder adapter only during checkpoint
    restoration
  - loaded the saved adapter into the existing PEFT model under the original
    component name
  - avoided creating a second nested PEFT wrapper
- Static recorder fix:
  - preserved the recorder's `List[str]` and `List[Dict]` batch contract
  - flattened only the optional rank nesting introduced by distributed gather
  - did not bypass the official recorder or metric hook
- Scope:
  - no changes to Weaver or Trigger training initialization
  - no changes to trainer classes or training scripts
  - no LatentMemoryBank implementation
  - no dependency or environment changes

### EXP-20260611-005: Repair Review Three-Sample Sanity Check

- Phase: Temporary Repair Review and Sanity Check
- Status: `completed`
- Date: 2026-06-11
- Purpose: Review the Repair Phase diff and verify the repaired official static
  evaluation path across more than one sample.
- Scientific status: sanity check only; not a formal baseline
- Git branch: `rlm-memory-bank`
- Base commit: `ed741d9be111b3f549740dce6db0f90c4ae11632`
- Environment:
  `/home/baishilong/miniconda3/envs/memgen/bin/python`
- Package versions: PyTorch 2.12.0+cu126, Transformers 4.55.4, PEFT 0.17.1,
  Accelerate 1.10.1, Datasets 4.0.0
- Config file: `configs/latent_memory/gsm8k.yaml`
- Model path:
  `/home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Checkpoint path:
  `/mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model`
- Dataset path:
  `/home/baishilong/.cache/huggingface/datasets/gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866`
- Dataset/split/sample IDs: `gsm8k/main`, test indices 0, 1, and 2
- Sample count: 3
- Batch size: 1
- Random seed: 42
- Decoding: greedy, temperature 0.0, maximum response length 128, Weaver and
  Trigger sampling disabled
- Output directory: `outputs/baseline/EXP-20260611-005`
- Prediction file:
  `outputs/baseline/EXP-20260611-005/evaluate/answer.json`
- Verification file:
  `outputs/baseline/EXP-20260611-005/verification.json`
- Command:

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
  --output-dir /mnt/18T/baishilong/MemGen/outputs/baseline/EXP-20260611-005 \
  --sample-count 3
```

#### Diff Review

- Core files reviewed:
  - `memgen/model/modeling_memgen.py`
  - `memgen/runner.py`
- Harness reviewed and parameterized:
  - `scripts/eval/repair_phase2_smoke.py`
- Protected training paths checked with `git diff --name-only`:
  - `memgen/trainer/`
  - `scripts/train/`
  - `scripts/weaver_sft.sh`
  - `scripts/weaver_grpo.sh`
  - `scripts/trigger_train.sh`
  - `memgen/model/modeling_utils.py`
- Protected training path diff result: empty
- Review verdict: the repair is narrowly scoped to checkpoint restoration and
  static evaluation result collation.

#### Results

- `answer.json`: 2,549 bytes and four JSONL records.
- Prediction records: 3.
- Summary records: 1.
- All three prediction records contain non-empty completions.
- One-sample rewards: 1.0, 1.0, and 0.0.
- Summary reward: 0.6666666666666666; not accepted as a baseline metric.
- Weaver adapter: 112/112 exact tensor match.
- Trigger adapter: 112/112 exact tensor match.
- Missing keys: 0.
- Unexpected keys: 0.
- Shape mismatches: 0.
- Value mismatches: 0.
- Adapter-related load warnings: 0.
- Trigger decision calls: 193.
- Weaver prompt augmentation calls: 3.
- Weaver inference augmentation calls: 8.
- Three augmentation masks were captured.
- Evaluation latency: 14.633 seconds.
- Peak allocated CUDA memory: 9,391,613,952 bytes.

#### Caveats

- Transformers warned that `temperature` may be ignored under greedy decoding;
  sampling was disabled, so this does not change the intended deterministic
  protocol.
- Accelerate warned that Linux kernel 5.4 is below its recommended 5.5 minimum;
  the run completed without a hang.
- This experiment does not establish aggregate GSM8K performance.

#### Conclusion

- No Repair Phase regression was found.
- The fixes remain suitable for proceeding to an explicitly approved Phase 3.
- At this Repair Review closeout, the baseline gate remained closed until the
  explicitly approved Phase 3 run.
- Related bugs: `BUG-0001`, `BUG-0002`

### EXP-20260611-006: Original MemGen Fixed-Subset Baseline

- Phase: Phase 3 - Original MemGen Baseline
- Status: `completed`
- Baseline ID: `memgen-gsm8k-sft-official-v1`
- Date: 2026-06-11
- Git branch: `rlm-memory-bank`
- Core code revision: `c0f1f2c3d79828c2d4e4f74eb9756bfb50890653`
- Working tree during run: evaluation-harness changes only
- Environment:
  `/home/baishilong/miniconda3/envs/memgen/bin/python`
- Config: `configs/latent_memory/gsm8k.yaml`
- Model path:
  `/home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Checkpoint:
  `/mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model`
- Dataset cache:
  `/home/baishilong/.cache/huggingface/datasets/gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866`
- Split and sample IDs: GSM8K `main/test`, indices 0 through 19
- Sample count: 20
- Seed: 42
- Batch size: 1
- Decoding: greedy, temperature 0.0, maximum response length 1024, Weaver and
  Trigger sampling disabled
- Output directory: `outputs/baseline/EXP-20260611-006`
- Prediction file:
  `outputs/baseline/EXP-20260611-006/evaluate/answer.json`
- Verification file:
  `outputs/baseline/EXP-20260611-006/verification.json`
- Metric contract:
  `outputs/baseline/EXP-20260611-006/json/metric_contract.json`
- Command:

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

#### Results

- Prediction records: 20/20, all non-empty.
- Summary records: 1.
- Mean `compute_reward`: 0.60.
- Correct samples: 12; incorrect samples: 8.
- Incorrect sample IDs: 7, 8, 11, 12, 13, 14, 15, 19.
- All 20 completions contained a boxed answer.
- Response length: minimum 53, maximum 235, mean 134.35 tokens.
- No sample reached the 1024-token limit.
- Weaver adapter: exact 112/112 tensor match.
- Trigger adapter: exact 112/112 tensor match.
- Missing/unexpected/shape/value mismatches: all zero.
- Adapter-related loading warnings: zero.
- Trigger decision calls: 1,722.
- Weaver prompt augmentation calls: 20.
- Weaver inference augmentation calls: 43.
- Total evaluation latency: 115.728 seconds.
- Mean evaluation latency: 5.786 seconds/sample.
- Peak allocated CUDA memory: 9,415,716,352 bytes.
- No NaN, OOM, CUDA error, empty completion, or incomplete sample.

#### Artifact Hashes

- `answer.json`:
  `b8e824b4c82c9fc0e6dcfd35b56bd96f26390756ceefef57ee2c35a36e21baea`
- `verification.json`:
  `da94bf8f27fbc67472c30dce35e001bdc054ee7fe59a357bbf1c84e65a6bd333`
- `metric_contract.json`:
  `facf67c5ff4d0742d6640583c41714a4ec767e70c976b767a0ae9e198e7e0026`

#### Interpretation

This is the accepted Original MemGen comparison point for later
LatentMemoryBank experiments on the same fixed subset. It is not an estimate of
full GSM8K test performance.

### EXP-20260611-007: Golden-Case Deterministic Replay

- Phase: Phase 3 - Original MemGen Baseline
- Status: `completed`
- Purpose: Replay fixed test indices 0, 1, and 2 under the exact baseline
  configuration.
- Core code revision: `c0f1f2c3d79828c2d4e4f74eb9756bfb50890653`
- Sample count: 3
- Seed: 42
- Batch size: 1
- Decoding: greedy, maximum response length 1024
- Output directory: `outputs/baseline/EXP-20260611-007`
- Result: all three response-token hashes and all three augmentation-mask hashes
  exactly matched `EXP-20260611-006`.
- Sample 0 response/mask:
  `b263835e26587cffe0d540125dc63a6acf27e924dfa9d5cb45885ce4081218f0` /
  `7dcc914e338423f3616d3d0139ac0df8a959cc0117c4343fb577c26bfd0b1cb4`
- Sample 1 response/mask:
  `560a6a6ffca3241005289a07d36b7c7820b6e13dea354e2beaf5b81e7f67849a` /
  `d042e76299bf72b3847744d4f3b0633de65ede4c6115c40f974188f495fc575b`
- Sample 2 response/mask:
  `dc2bbbddf83b56513d68a590277761b46e097eeaf71fec3429f1715ddd0f20fe` /
  `d6e708979b29292866684d928520c151868f06f891f456161ef56c9daca185e4`
- Conclusion: deterministic golden evidence established for later disabled-path
  equivalence tests.

### EXP-20260611-008: LatentMemoryBank Skeleton Unit Verification

- Phase: Phase 4 - LatentMemoryBank Module Skeleton
- Status: `completed`
- Date: 2026-06-11
- Code revision before Phase 4 changes:
  `506bd21ffd53531a0cac442093ccce403e8b3891`
- Environment:
  `/home/baishilong/miniconda3/envs/memgen/bin/python`
- Python: 3.10.20
- PyTorch: 2.12.0+cu126
- Dataset/model/checkpoint: not applicable; no inference experiment was run
- Sample count/seed/decoding: not applicable
- Output directory and prediction files: none
- Commands:

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python \
  -m py_compile \
  memgen/model/latent_memory_bank.py \
  tests/test_latent_memory_bank.py

/home/baishilong/miniconda3/envs/memgen/bin/python \
  -m unittest discover -s tests -v
```

- Test result after Phase 4 cleanup: 16 passed, 0 failed, 0 errors.
- Covered behavior:
  - disabled no-op and empty retrieval
  - explicit batch-size-1 configuration enforcement
  - detach, clone, and source tensor metadata
  - capacity, append refusal, replace-oldest, and replace-lowest-score
  - top-k, threshold, and recency decay
  - hidden-state and pre-pooled query input
  - reset and recent-token query pooling
  - explicit output dtype/device
  - retrieved tensor and nested-metadata mutation isolation
  - `replace` oldest-slot fallback when all slots are unscored
  - invalid shape and dtype errors
- Isolation checks:
  - no production references to the new module
  - no changes to model generation, runner, trainers, or training scripts
  - importing `MemGenModel` did not load
    `memgen.model.latent_memory_bank`
- Failures/anomalies: none.
- Conclusion: the Phase 4 skeleton is testable and isolated, with no performance
  or inference-integration claim.
- Related decisions: `DEC-0014`, `DEC-0015`, `DEC-0016`

### EXP-20260611-009: End-of-Day Validation

- Phase: End-of-Day Validation; no new roadmap phase
- Status: `completed`
- Date: 2026-06-11
- Purpose: Verify that the Repair fixes, accepted Phase 3 baseline, and
  standalone Phase 4 skeleton are recoverable and ready to commit.
- Code revision: `506bd21ffd53531a0cac442093ccce403e8b3891`
- Branch: `rlm-memory-bank`
- Working tree: dirty with uncommitted Phase 4 module, config, tests, and
  research-note updates
- Environment:
  `/home/baishilong/miniconda3/envs/memgen/bin/python`
- Model/dataset/checkpoint: no model or dataset was loaded; existing artifacts
  were inspected only
- Sample count/seed/decoding: not applicable; no inference run
- Commands:

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python \
  -m py_compile \
  memgen/model/modeling_memgen.py \
  memgen/runner.py \
  memgen/model/latent_memory_bank.py \
  scripts/eval/repair_phase2_smoke.py \
  tests/test_latent_memory_bank.py

/home/baishilong/miniconda3/envs/memgen/bin/python \
  -m unittest discover -s tests -v
```

- Compilation: passed with exit code 0.
- Unit tests: 16 passed, 0 failed, 0 errors.
- Phase 3 artifact verification:
  - `EXP-20260611-006/evaluate/answer.json` is non-empty JSONL
  - 20 prediction records plus one summary record
  - summary `compute_reward=0.60`
  - `EXP-20260611-007` contains three prediction records, one summary, and a
    readable verification artifact for sample IDs 0, 1, and 2
- Repair verification:
  - Weaver adapter 112/112 and Trigger adapter 112/112
  - missing, unexpected, shape-mismatch, and value-mismatch lists are empty
  - the accepted baseline output confirms `StaticEvalRecorder` writes complete,
    non-empty JSONL
- Isolation verification:
  - no diff in protected training paths
  - no `LatentMemoryBank` reference in `MemGenModel.generate()`, runner,
    interaction managers, `main.py`, or `memgen.model` exports
  - `configs/latent_memory_bank/default.yaml` remains `enabled: false`
  - existing `configs/latent_memory/gsm8k.yaml` has no diff
- Failures/anomalies:
  - direct whole-file `json.loads()` is invalid because `answer.json` is JSONL;
    line-by-line parsing succeeded and confirmed the expected record counts
- Conclusion: current Phase 4 changes are ready to commit. Phase 5 has not
  started and still requires explicit approval.

## Experiment Template

### EXP-YYYYMMDD-NNN: <Short Name>

- Phase:
- Status: `planned | running | completed | failed | aborted`
- Research question:
- Hypothesis:
- Baseline/comparator:
- Code revision:
- Working tree state:
- Environment:
- Dataset and split:
- Inputs/session definition:
- Configuration:
- Random seeds:
- Command:
- Output directory:
- Start/end time:

#### Metrics

| Metric | Baseline | Variant | Delta |
|---|---:|---:|---:|
| TBD | TBD | TBD | TBD |

#### Observations

- Quantitative:
- Qualitative:
- Runtime/latency:
- Peak memory:
- Failures or anomalies:

#### Conclusion

- Hypothesis supported:
- Interpretation:
- Limitations:
- Follow-up:
- Related decision IDs:
- Artifacts:

### EXP-20260612-014: Phase 7 Tier 1 Pre-Trace Smoke

- Phase: 7
- Status: `completed_with_caveats`
- Research question: Does enabled Version A complete a bounded one-sample smoke
  run before adding per-session trace capture?
- Baseline/comparator: none; debug only
- Sample IDs: `0`
- Sample count: `1`
- Seed: `42`
- Batch size: `1`
- Output directory:
  `outputs/latent_bank_vA/EXP-20260612-014-phase7-tier1-smoke`
- Key result: the run succeeded with 4 writes, 3 retrievals, 24 retrieved
  latents, 32 new latents, and 4 resident slots, but it did not yet expose
  session-level `initial_slots`, so it was superseded for durable Phase 7
  evidence by `EXP-20260612-015`.

### EXP-20260612-015: Phase 7 Tier 1 Enabled Smoke

- Phase: 7
- Status: `completed`
- Research question: Can enabled Version A complete a one-sample bounded smoke
  run with session-local debug evidence?
- Baseline/comparator: none; debug only
- Sample IDs: `0`
- Sample count: `1`
- Seed: `42`
- Batch size: `1`
- Decoding: greedy, temperature `0.0`, max response length `1024`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/latent_bank_vA/EXP-20260612-015-phase7-tier1-smoke --sample-start 0 --sample-count 1 --max-response-length 1024 --memory-enabled`
- Output directory:
  `outputs/latent_bank_vA/EXP-20260612-015-phase7-tier1-smoke`
- Prediction file:
  `outputs/latent_bank_vA/EXP-20260612-015-phase7-tier1-smoke/evaluate/answer.json`
- Verification file:
  `outputs/latent_bank_vA/EXP-20260612-015-phase7-tier1-smoke/verification.json`

#### Observations

- `answer.json` contained one prediction and one summary record.
- No crash, NaN, OOM, CUDA error, or shape/device/dtype mismatch occurred.
- Session trace recorded `initial_slots=0`.
- Adapter verification remained exact: Weaver `112/112`, Trigger `112/112`.
- Final bank stats:
  - `memory_write_count=4`
  - `memory_retrieve_count=3`
  - `retrieved_latent_count=24`
  - `new_latent_count=32`
  - `slot_count=4`
- Stored latent tensors were reasoner-space `[8, 1536]` tensors.
- Stored slot metadata remained explicit:
  - `storage_device=cpu`
  - `storage_dtype=torch.bfloat16`
  - `original_device=cuda:0`
  - `original_dtype=torch.bfloat16`
- `weaver_input_token_counts` matched `reasoner_to_weaver_input_token_counts`
  exactly, which is consistent with retrieved memory not entering Weaver.
- Total latency: `8.658 s`
- Peak allocated CUDA memory: `9,385,351,168` bytes
- Auxiliary summary metric: `compute_reward=1.0`

#### Conclusion

- Hypothesis supported: yes
- Interpretation: Enabled Version A completed a bounded one-sample run with the
  expected write/retrieve behavior and session-local initialization evidence.
- Scope note: This is a mechanism/stability check only, not a performance
  result.

### EXP-20260612-016: Phase 7 Tier 2 Small Stability

- Phase: 7
- Status: `completed`
- Research question: Do three enabled single-turn sessions remain isolated and
  stable on GSM8K samples `0..2`?
- Baseline/comparator: none; debug only
- Sample IDs: `0..2`
- Sample count: `3`
- Seed: `42`
- Batch size: `1`
- Decoding: greedy, temperature `0.0`, max response length `1024`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/latent_bank_vA/EXP-20260612-016-phase7-tier2-stability --sample-start 0 --sample-count 3 --max-response-length 1024 --memory-enabled`
- Output directory:
  `outputs/latent_bank_vA/EXP-20260612-016-phase7-tier2-stability`
- Prediction file:
  `outputs/latent_bank_vA/EXP-20260612-016-phase7-tier2-stability/evaluate/answer.json`
- Verification file:
  `outputs/latent_bank_vA/EXP-20260612-016-phase7-tier2-stability/verification.json`

#### Observations

- `answer.json` contained three prediction records and one summary record.
- Each recorded session started with `initial_slots=0`.
- Session bank ids differed across all three samples.
- No cross-sample leakage was observed.
- Per-session bank summaries:
  - sample 0: writes `4`, retrieves `3`, retrieved latents `24`, new latents
    `32`, slot count `4`
  - sample 1: writes `2`, retrieves `1`, retrieved latents `8`, new latents
    `16`, slot count `2`
  - sample 2: writes `4`, retrieves `3`, retrieved latents `24`, new latents
    `32`, slot count `4`
- `slot_count` never exceeded `max_slots=8`.
- No crash, NaN, OOM, CUDA error, shape mismatch, device mismatch, or dtype
  mismatch occurred.
- `weaver_input_token_counts` matched `reasoner_to_weaver_input_token_counts`
  exactly.
- Total latency: `14.066 s`
- Mean latency: `4.689 s/sample`
- Peak allocated CUDA memory: `9,385,351,168` bytes
- Auxiliary summary metric: `compute_reward=0.6666666666666666`

#### Conclusion

- Hypothesis supported: yes
- Interpretation: Enabled Version A remained session-local and stable across
  three independent single-turn samples.
- Scope note: This is a bounded stability check only, not a comparative reward
  result.

### EXP-20260612-017: Phase 7 Tier 3 Bounded Capacity

- Phase: 7
- Status: `completed`
- Research question: Does enabled Version A remain stable on a bounded
  five-sample run without exceeding slot limits or showing leakage?
- Baseline/comparator: none; debug only
- Sample IDs: `0..4`
- Sample count: `5`
- Seed: `42`
- Batch size: `1`
- Decoding: greedy, temperature `0.0`, max response length `1024`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/latent_bank_vA/EXP-20260612-017-phase7-tier3-capacity --sample-start 0 --sample-count 5 --max-response-length 1024 --memory-enabled`
- Output directory:
  `outputs/latent_bank_vA/EXP-20260612-017-phase7-tier3-capacity`
- Prediction file:
  `outputs/latent_bank_vA/EXP-20260612-017-phase7-tier3-capacity/evaluate/answer.json`
- Verification file:
  `outputs/latent_bank_vA/EXP-20260612-017-phase7-tier3-capacity/verification.json`

#### Observations

- `answer.json` contained five prediction records and one summary record.
- All five recorded sessions started with `initial_slots=0`.
- No cross-sample leakage was observed.
- Per-session bank summaries:
  - sample 0: writes `4`, retrieves `3`, retrieved latents `24`, new latents
    `32`, slot count `4`
  - sample 1: writes `2`, retrieves `1`, retrieved latents `8`, new latents
    `16`, slot count `2`
  - sample 2: writes `4`, retrieves `3`, retrieved latents `24`, new latents
    `32`, slot count `4`
  - sample 3: writes `2`, retrieves `1`, retrieved latents `8`, new latents
    `16`, slot count `2`
  - sample 4: writes `4`, retrieves `3`, retrieved latents `24`, new latents
    `32`, slot count `4`
- `slot_count` never exceeded `4`; therefore the configured replacement policy
  was not triggered in this bounded run.
- No crash, NaN, OOM, CUDA error, shape mismatch, device mismatch, or dtype
  mismatch occurred.
- `weaver_input_token_counts` matched `reasoner_to_weaver_input_token_counts`
  exactly.
- Total latency: `21.562 s`
- Mean latency: `4.312 s/sample`
- Peak allocated CUDA memory: `9,395,434,496` bytes
- Auxiliary summary metric: `compute_reward=0.8`

#### Conclusion

- Hypothesis supported: yes
- Interpretation: Enabled Version A remained stable in a bounded five-sample
  run and did not expose leakage or capacity overruns.
- Scope note: No method-quality claim follows from this debug result.

### EXP-20260612-018: Phase 7 Capacity-Trigger Supplement

- Phase: 7 supplement
- Status: `completed`
- Research question: Can the real enabled Version A inference path be forced to
  trigger replacement by lowering `max_slots` to `2`?
- Baseline/comparator: none; debug only
- Sample IDs: `0`
- Sample count: `1`
- Seed: `42`
- Batch size: `1`
- Decoding: greedy, temperature `0.0`, max response length `1024`
- Memory overrides:
  - `max_slots=2`
  - `update_policy=replace_oldest`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/latent_bank_vA/EXP-20260612-018-phase7-capacity-trigger --sample-start 0 --sample-count 1 --max-response-length 1024 --memory-enabled --memory-max-slots 2 --memory-update-policy replace_oldest`
- Output directory:
  `outputs/latent_bank_vA/EXP-20260612-018-phase7-capacity-trigger`
- Prediction file:
  `outputs/latent_bank_vA/EXP-20260612-018-phase7-capacity-trigger/evaluate/answer.json`
- Verification file:
  `outputs/latent_bank_vA/EXP-20260612-018-phase7-capacity-trigger/verification.json`

#### Observations

- `answer.json` contained one prediction and one summary record.
- No crash, NaN, OOM, CUDA error, or shape/device/dtype mismatch occurred.
- Session trace recorded `initial_slots=0`.
- Adapter verification remained exact: Weaver `112/112`, Trigger `112/112`.
- Final bank stats:
  - `memory_write_count=4`
  - `memory_retrieve_count=3`
  - `retrieved_latent_count=24`
  - `new_latent_count=32`
  - `slot_count=2`
  - `append_count=2`
  - `replace_count=2`
  - `rejected_write_count=0`
  - `last_update_action=replace`
  - `update_action_trace=["append", "append", "replace", "replace"]`
- This run therefore satisfied both trigger conditions:
  - `memory_write_count > max_slots`
  - `replace_count > 0`
- Stored latent tensors remained reasoner-space `[8, 1536]` tensors.
- Stored slot metadata remained explicit:
  - `storage_device=cpu`
  - `storage_dtype=torch.bfloat16`
  - `original_device=cuda:0`
  - `original_dtype=torch.bfloat16`
- `weaver_input_token_counts` matched `reasoner_to_weaver_input_token_counts`
  exactly, which is consistent with retrieved memory not entering Weaver.
- Total latency: `8.563 s`
- Peak allocated CUDA memory: `9,385,351,168` bytes
- Auxiliary summary metric: `compute_reward=1.0`

#### Conclusion

- Hypothesis supported: yes
- Interpretation: The real enabled Version A inference path can trigger
  replacement cleanly under bounded debug conditions when `max_slots` is
  lowered to `2`.
- Scope note: This supplement verifies capacity/replacement behavior only. It
  is not a performance experiment and makes no baseline-improvement claim.

## Reproducibility Checklist

- [ ] Exact command recorded.
- [ ] Code revision and dirty state recorded.
- [ ] Configuration snapshot preserved.
- [ ] Seeds recorded.
- [ ] Dataset version/split recorded.
- [ ] Raw outputs retained.
- [ ] Metrics can be regenerated.
- [ ] Failures and exclusions documented.

## Phase 8A - Core Ablation Pilot

### Protocol

- Date: 2026-06-12
- Scope: pilot only; not a performance experiment
- Dataset: `gsm8k/main/test`
- Sample IDs: `0..19`
- Sample count: `20`
- Seed: `42`
- Batch size: `1`
- Decoding: greedy
- Max response length: `1024`
- Shared config path: `configs/latent_memory/gsm8k.yaml`
- Shared model path:
  `/home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Shared checkpoint path:
  `/mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model`
- Output root: `outputs/ablations/`

### Group Table

| Group | Experiment | Config difference | compute_reward | Correct / Total | Total latency (s) | Mean latency (s/sample) | Peak CUDA memory (bytes) |
|---|---|---|---:|---:|---:|---:|---:|
| G0 | `EXP-20260612-013` | disabled anchor | 0.60 | 12 / 20 | 96.615 | 4.831 | 9,415,716,352 |
| G1 | `EXP-20260612-019` | enabled, `decay_alpha=0.05`, `update_policy=replace_oldest` | 0.50 | 10 / 20 | 296.500 | 14.825 | 9,420,448,256 |
| G4 | `EXP-20260612-020` | enabled, `decay_alpha=0.0`, `update_policy=replace_oldest` | 0.50 | 10 / 20 | 239.576 | 11.979 | 9,420,448,256 |
| G6 | `EXP-20260612-021` | enabled, `decay_alpha=0.05`, `update_policy=append` | 0.50 | 10 / 20 | 295.256 | 14.763 | 9,420,448,256 |
| G7 | `EXP-20260612-022` | enabled, `decay_alpha=0.05`, `update_policy=replace` | 0.50 | 10 / 20 | 293.830 | 14.691 | 9,420,448,256 |

### G0: Disabled Anchor Reuse

- Status: `reused`, not rerun
- Comparator artifacts:
  - accepted baseline: `EXP-20260611-006`
  - current-harness disabled equivalence: `EXP-20260612-013`
- Rationale: Phase 6 already verified current disabled-path equivalence against
  the frozen Phase 3 baseline, so this pilot reused the validated disabled
  anchor instead of spending another full 20-sample run.
- Output directory:
  `outputs/baseline/EXP-20260612-013-phase6-disabled-equivalence`
- Key results:
  - `answer.json` non-empty
  - prediction count `20`
  - summary count `1`
  - `compute_reward=0.60`
  - correct / total `12 / 20`
  - Trigger decision calls `1722`
  - Weaver prompt calls `20`
  - Weaver inference calls `43`
  - no memory bank constructed
  - `memory_bank_debug=null`

### EXP-20260612-019: G1 Version A Anchor

- Status: `completed`
- Output directory:
  `outputs/ablations/EXP-20260612-019-phase8a-g1-anchor`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/ablations/EXP-20260612-019-phase8a-g1-anchor --sample-start 0 --sample-count 20 --max-response-length 1024 --memory-enabled --memory-max-slots 8 --memory-top-k 1 --memory-threshold 0.7 --memory-decay-alpha 0.05 --memory-update-policy replace_oldest --memory-retrieve-policy threshold_topk`
- Config:
  - `enabled=true`
  - `retrieve_policy=threshold_topk`
  - `top_k=1`
  - `threshold=0.7`
  - `decay_alpha=0.05`
  - `update_policy=replace_oldest`
  - `max_slots=8`
- Results:
  - `answer.json` non-empty
  - prediction count `20`
  - summary count `1`
  - `compute_reward=0.50`
  - correct / total `10 / 20`
  - total latency `296.500 s`
  - mean latency `14.825 s/sample`
  - peak CUDA memory `9,420,448,256` bytes
  - Trigger decision calls `1439`
  - Weaver prompt calls `20`
  - Weaver inference calls `50`
- Aggregated memory debug:
  - `memory_write_count=70`
  - `memory_retrieve_count=50`
  - `retrieved_latent_count=392`
  - `new_latent_count=560`
  - `max observed slot_count=4`
  - `append_count=70`
  - `replace_count=0`
  - `rejected_write_count=0`
  - every session started with `initial_slots=0`
- Boundary checks:
  - no cross-sample leakage observed
  - `weaver_input_token_counts` matched
    `reasoner_to_weaver_input_token_counts`
  - stored latents stayed in reasoner space with hidden size `1536`
  - no crash, NaN, OOM, CUDA error, or shape/device/dtype mismatch

### EXP-20260612-020: G4 Cosine Retrieval Without Recency Decay

- Status: `completed`
- Output directory:
  `outputs/ablations/EXP-20260612-020-phase8a-g4-cosine-no-decay`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/ablations/EXP-20260612-020-phase8a-g4-cosine-no-decay --sample-start 0 --sample-count 20 --max-response-length 1024 --memory-enabled --memory-max-slots 8 --memory-top-k 1 --memory-threshold 0.7 --memory-decay-alpha 0.0 --memory-update-policy replace_oldest --memory-retrieve-policy threshold_topk`
- Config difference from G1:
  - `decay_alpha=0.0`
- Results:
  - `answer.json` non-empty
  - prediction count `20`
  - summary count `1`
  - `compute_reward=0.50`
  - correct / total `10 / 20`
  - total latency `239.576 s`
  - mean latency `11.979 s/sample`
  - peak CUDA memory `9,420,448,256` bytes
  - Trigger decision calls `1434`
  - Weaver prompt calls `20`
  - Weaver inference calls `50`
- Aggregated memory debug:
  - `memory_write_count=70`
  - `memory_retrieve_count=50`
  - `retrieved_latent_count=392`
  - `new_latent_count=560`
  - `max observed slot_count=4`
  - `append_count=70`
  - `replace_count=0`
  - `rejected_write_count=0`
  - every session started with `initial_slots=0`
- Boundary checks:
  - no cross-sample leakage observed
  - retrieved memory remained Reasoner-only
  - stored latents stayed in reasoner space with hidden size `1536`
  - no crash, NaN, OOM, CUDA error, or shape/device/dtype mismatch

### EXP-20260612-021: G6 Append-Only Update

- Status: `completed`
- Output directory:
  `outputs/ablations/EXP-20260612-021-phase8a-g6-append`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/ablations/EXP-20260612-021-phase8a-g6-append --sample-start 0 --sample-count 20 --max-response-length 1024 --memory-enabled --memory-max-slots 8 --memory-top-k 1 --memory-threshold 0.7 --memory-decay-alpha 0.05 --memory-update-policy append --memory-retrieve-policy threshold_topk`
- Config difference from G1:
  - `update_policy=append`
- Results:
  - `answer.json` non-empty
  - prediction count `20`
  - summary count `1`
  - `compute_reward=0.50`
  - correct / total `10 / 20`
  - total latency `295.256 s`
  - mean latency `14.763 s/sample`
  - peak CUDA memory `9,420,448,256` bytes
  - Trigger decision calls `1439`
  - Weaver prompt calls `20`
  - Weaver inference calls `50`
- Aggregated memory debug:
  - `memory_write_count=70`
  - `memory_retrieve_count=50`
  - `retrieved_latent_count=392`
  - `new_latent_count=560`
  - `max observed slot_count=4`
  - `append_count=70`
  - `replace_count=0`
  - `rejected_write_count=0`
  - every session started with `initial_slots=0`
- Boundary checks:
  - no cross-sample leakage observed
  - retrieved memory remained Reasoner-only
  - stored latents stayed in reasoner space with hidden size `1536`
  - no crash, NaN, OOM, CUDA error, or shape/device/dtype mismatch

### EXP-20260612-022: G7 Replace Update

- Status: `completed`
- Output directory:
  `outputs/ablations/EXP-20260612-022-phase8a-g7-replace`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/ablations/EXP-20260612-022-phase8a-g7-replace --sample-start 0 --sample-count 20 --max-response-length 1024 --memory-enabled --memory-max-slots 8 --memory-top-k 1 --memory-threshold 0.7 --memory-decay-alpha 0.05 --memory-update-policy replace --memory-retrieve-policy threshold_topk`
- Config difference from G1:
  - `update_policy=replace`
- Results:
  - `answer.json` non-empty
  - prediction count `20`
  - summary count `1`
  - `compute_reward=0.50`
  - correct / total `10 / 20`
  - total latency `293.830 s`
  - mean latency `14.691 s/sample`
  - peak CUDA memory `9,420,448,256` bytes
  - Trigger decision calls `1439`
  - Weaver prompt calls `20`
  - Weaver inference calls `50`
- Aggregated memory debug:
  - `memory_write_count=70`
  - `memory_retrieve_count=50`
  - `retrieved_latent_count=392`
  - `new_latent_count=560`
  - `max observed slot_count=4`
  - `append_count=70`
  - `replace_count=0`
  - `rejected_write_count=0`
  - every session started with `initial_slots=0`
- Boundary checks:
  - no cross-sample leakage observed
  - retrieved memory remained Reasoner-only
  - stored latents stayed in reasoner space with hidden size `1536`
  - no crash, NaN, OOM, CUDA error, or shape/device/dtype mismatch

### Pilot Interpretation

- `G0` vs `G1`:
  - on this 20-sample pilot, enabled Version A anchor underperformed the
    disabled anchor (`0.50` vs `0.60`)
  - this is a pilot observation only, not a final claim
- `G1` vs `G4`:
  - removing current write-age decay did not change `compute_reward` on this
    slice
  - G4 reduced latency relative to G1 in this pilot
  - this comparison is not last-retrieved-turn decay versus no decay
- `G1` vs `G6`:
  - append-only update matched G1 on `compute_reward`
  - no capacity pressure appeared because no session exceeded `4` slots
- `G1` vs `G7`:
  - score-based `replace` matched G1 on `compute_reward`
  - `replace_count=0` because `max_slots=8` was not reached in this pilot

### Pilot Conclusion

- Phase 8A pilot status: `pass`
- All currently implemented groups ran stably on the 20-sample slice.
- No new blocker was observed.
- Quantitative observation:
  - disabled G0: `compute_reward=0.60`, `12/20`
  - enabled G1/G4/G6/G7: `compute_reward=0.50`, `10/20`
  - every enabled variant underperformed the disabled anchor on this pilot
- Stability observation:
  - all enabled variants completed without runtime or tensor-contract failure
  - no cross-sample leakage or retrieved-memory-to-Weaver leakage was observed
- Update-policy interpretation:
  - no session saturated `max_slots=8`
  - `replace_count=0` in G1, G4, G6, and G7
  - Phase 8A therefore did not produce an effective update-policy comparison
- Retrieval interpretation:
  - current decay is write-age decay measured in successful memory writes
  - current `threshold_topk` has no fallback top-1
  - G1/G4 compare write-age decay against no decay
- Scope:
  - Phase 8A is a short single-turn sanity and negative pilot
  - it is not aligned with the primary multi-turn, long-trajectory, or
    context-truncation hypothesis
  - it is not evidence that the full unimplemented Version B method fails
- Next-step motivation:
  - do not expand GSM8K directly into the primary main experiment
  - establish a dynamic multi-turn TriviaQA baseline
  - evaluate method-aligned Version A variants only after target-task stability
    is established

### EXP-20260612-023-step3-disabled-replay: Step 3 Disabled Replay

- Step: 3
- Status: `completed`
- Purpose: Compatibility verification after integrating
  `update_policy=thread_update`; this is not a performance experiment.
- Dataset: cached `gsm8k/main/test`, sample IDs `0..2`
- Runtime:
  - `sample_count=3`
  - `seed=42`
  - `batch_size=1`
  - greedy decoding
  - `max_response_length=1024`
- Output:
  `outputs/latent_bank_vA/EXP-20260612-023-step3-disabled-replay/`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/latent_bank_vA/EXP-20260612-023-step3-disabled-replay --sample-start 0 --sample-count 3 --max-response-length 1024`
- Frozen comparator: `EXP-20260611-007`
- Results:
  - three predictions plus one summary
  - all response-token hashes matched exactly
  - all augmentation-mask hashes matched exactly
  - Trigger decision calls matched at `193`
  - Weaver prompt calls matched at `3`
  - Weaver inference calls matched at `8`
  - no memory bank was constructed
  - `memory_bank_debug=null`
- Interpretation:
  - Step 3 did not change disabled-path behavior
  - this replay is compatibility evidence only

### EXP-20260612-024-thread-update-smoke: Thread-Update Mechanism Smoke

- Step: 4
- Status: `completed`
- Purpose: Mechanism validation only; this is not a performance experiment.
- Dataset: cached `gsm8k/main/test`, sample ID `0`
- Runtime:
  - `sample_count=1`
  - `seed=42`
  - `batch_size=1`
  - greedy decoding
  - `max_response_length=1024`
- Memory configuration:
  - `enabled=true`
  - `update_policy=thread_update`
  - `retrieve_policy=threshold_topk`
  - `threshold=0.7`
  - `top_k=1`
  - `max_slots=8`
  - `decay_alpha=0.05`
- Output:
  `outputs/latent_bank_vA/EXP-20260612-024-thread-update-smoke/`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/latent_bank_vA/EXP-20260612-024-thread-update-smoke --sample-start 0 --sample-count 1 --max-response-length 1024 --memory-enabled --memory-max-slots 8 --memory-top-k 1 --memory-threshold 0.7 --memory-decay-alpha 0.05 --memory-update-policy thread_update --memory-retrieve-policy threshold_topk`
- Artifact checks:
  - non-empty `evaluate/answer.json`
  - prediction count `1`
  - summary count `1`
  - no crash, NaN, OOM, CUDA, shape, device, or dtype error
- Memory results:
  - `memory_write_count=4`
  - `memory_retrieve_count=3`
  - `thread_insert_count=1`
  - `matched_replace_count=3`
  - `capacity_evict_count=0`
  - final `slot_count=1`
  - observed reasons:
    `empty_bank`, `matched_thread`, `matched_thread`, `matched_thread`
  - `new_thread` and `new_thread_bank_full` were not observed in this real
    one-sample smoke
- Controlled mechanism evidence:
  - `empty_bank -> insert`: unit test passed
  - low score, available capacity -> `new_thread`: unit test passed
  - high score -> `replace_matched` / `matched_thread`: unit test passed and
    observed in real inference
  - low score, full bank -> `evict_oldest_insert` /
    `new_thread_bank_full`: unit test passed
- Boundary checks:
  - Weaver input token counts exactly matched reasoner-to-Weaver input token
    counts: `[96, 116, 140, 193]`
  - retrieved memory therefore remained Reasoner-only
  - stored latent shape was `[8, 1536]`, confirming reasoner-space storage
  - session started with `initial_slots=0`
- Interpretation:
  - Version A-aligned `thread_update` mechanism is operational
  - this run does not establish accuracy or performance benefit
  - current retrieval still has no fallback top-1
  - current decay remains write-age decay
  - Version B has not started
