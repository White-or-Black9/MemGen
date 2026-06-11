# Experiment Log

Record every experiment, including failed, aborted, and exploratory runs. Never
overwrite prior records; append a new entry.

## Experiment Index

| ID | Date | Phase | Question | Status | Key Result |
|---|---|---|---|---|---|
| EXP-20260611-001 | 2026-06-11 | Phase 0 | Can the official GSM8K SFT checkpoint produce a trusted smoke baseline? | `failed` | LoRA keys were not loaded; direct smoke then failed on projection dtype |
| EXP-20260611-002 | 2026-06-11 | Phase 2 | Can the original MemGen inference stack run a one-sample GSM8K smoke test in the recommended environment? | `completed_with_caveats` | Original eval path reached generation but crashed in static recorder; script-only harness produced one completion; result is not a valid baseline because LoRA loading remains broken |
| EXP-20260611-003 | 2026-06-11 | Environment Alignment | Is the existing `memgen` environment suitable for the Repair Phase without package changes? | `completed` | Imports, dependency check, CUDA/BF16, config, model snapshot, checkpoint, and dataset cache validated; no install required |

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

## Reproducibility Checklist

- [ ] Exact command recorded.
- [ ] Code revision and dirty state recorded.
- [ ] Configuration snapshot preserved.
- [ ] Seeds recorded.
- [ ] Dataset version/split recorded.
- [ ] Raw outputs retained.
- [ ] Metrics can be regenerated.
- [ ] Failures and exclusions documented.
