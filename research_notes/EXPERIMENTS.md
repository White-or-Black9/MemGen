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
- Baseline gate remains closed.
- Related bugs: `BUG-0001`, `BUG-0002`

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
