# Bug and Anomaly Log

Record code defects, experiment anomalies, regressions, and suspected data leaks.
Do not delete resolved entries.

## Bug Index

| ID | Date | Severity | Status | Summary |
|---|---|---|---|---|
| BUG-0001 | 2026-06-11 | high | `open` | Official Weaver/Trigger LoRA adapters are not loaded by `MemGenModel.from_pretrained()` |
| BUG-0002 | 2026-06-11 | high | `open` | Static evaluation crashes in `StaticEvalRecorder.record_batch()` after generation starts |
| BUG-0003 | 2026-06-11 | medium | `open` | Checked-in environment specifications disagree on Python, CUDA, and package versions |
| BUG-0004 | 2026-06-11 | low | `open` | PATH-level `conda` wrapper has a CRLF shebang and cannot execute |

## Recorded Bugs

### BUG-0001: Nested PEFT Loading Skips Official LoRA Weights

- Date found: 2026-06-11
- Severity: high
- Status: `open`
- Phase/experiment: Phase 0 / `EXP-20260611-001`
- Environment: PEFT 0.17.1, Transformers 4.55.4, Python 3.10.20
- Revision: `5e59fee296092fa056f140b38a07b927651ffdb5`
- Symptoms: `PeftModel.from_pretrained()` reports all expected named Weaver and
  Trigger adapter keys as missing.
- Expected behavior: The 112 tensors in each official adapter checkpoint load into
  the corresponding LoRA layers.
- Actual behavior: The existing LoRA-wrapped model's `base_model` is wrapped
  again. Expected keys gain an additional model nesting and adapter-name suffix,
  while checkpoint keys remain unsuffixed.
- Minimal reproduction: Load the official GSM8K SFT checkpoint through
  `MemGenModel.from_config()` with `model.load_model_path` set.
- Evidence: Checkpoint keys begin with
  `base_model.model.model.layers...lora_A.weight`; runtime warnings expect
  `base_model.model.model.model.layers...lora_A.weaver.weight`.
- Suspected root cause: `MemGenModel.from_pretrained()` passes
  `model.weaver.model.base_model` and `model.trigger.model.base_model`, which are
  already PEFT/Lora model wrappers, into a new `PeftModel.from_pretrained()`.
- Compatibility impact: Baseline metrics are untrusted; Phase 1 disabled-path
  equivalence cannot be established.
- Phase 2 recheck:
  - A focused CPU-side PEFT load check on 2026-06-11 still reported missing
    adapter keys under the nested load path.
  - The GPU smoke run therefore remains invalid as a baseline even though
    generation could be reached.
- Required fix verification:
  - No unexplained missing/unexpected adapter keys.
  - Loaded tensor equality against official safetensors.
  - Deterministic generation repeat.
  - No changes to training paths.

### BUG-0002: Static Eval Recorder Expects the Wrong Batch Shape

- Date found: 2026-06-11
- Severity: high
- Status: `open`
- Phase/experiment: Phase 2 / `EXP-20260611-002`
- Environment:
  `/home/baishilong/miniconda3/envs/memgen`, Python 3.10.20, Transformers 4.55.4,
  Accelerate single-process evaluation on RTX A6000
- Revision: `7b8b9a44eb30325a676a6c9576c35b3a10b52c32`
- Symptoms:
  - `MemGenRunner.evaluate()` enters `_static_evaluate()`
  - generation starts and completes for the one-sample batch
  - result recording crashes with `KeyError: 0`
- Expected behavior:
  - `evaluate/answer.json` should contain at least one record plus summary
- Actual behavior:
  - `StaticEvalRecorder.record_batch(comps, batch)` receives a `batch` object
    that is not indexable as `examples[0]`
  - `answer.json` is created but remains empty
- Minimal reproduction:
  - GSM8K static evaluation with `batch_size=1`, one test sample, through the
    original `MemGenRunner.evaluate()` path
- Logs/artifacts:
  - `.cache/evaluate/gsm8k/home/pn=1_pl=8_in=3_il=8_20260611-103526_phase2_smoke/logs/log.txt`
  - `.cache/evaluate/gsm8k/home/pn=1_pl=8_in=3_il=8_20260611-103526_phase2_smoke/evaluate/answer.json`
- Suspected cause:
  - `_static_evaluate()` gathers `test_batch` through `gather_objects` and then
    iterates `for comps, batch in zip(all_completions, all_batches):`
  - `StaticEvalRecorder.record_batch()` expects `examples` to behave like
    `List[Dict]`, but the provided `batch` shape does not match that contract
- Compatibility impact:
  - The original static evaluation path is not end-to-end runnable, even when
    model and dataset loading succeed
  - Smoke outputs cannot be produced by the official recorder path
- Required fix verification:
  - one-sample static eval writes a non-empty `answer.json`
  - no crash in `record_batch`
  - summary metrics append correctly
  - no change to Weaver/Trigger training workflows

### BUG-0003: Environment Specifications Are Inconsistent

- Date found: 2026-06-11
- Severity: medium
- Status: `open`
- Phase/experiment: Temporary Environment Alignment / `EXP-20260611-003`
- Revision: `dd6eda02c3c06823670e217c8b0217199b24235c`
- Symptoms:
  - README setup specifies Python `3.10`
  - `memgen.yml` specifies Python `3.11.13`
  - `requirements.txt` specifies PyTorch `2.7.1+cu128`
  - `memgen.yml` specifies PyTorch `2.7.1+cu118`
  - additional versions such as Accelerate, NumPy, TRL, safetensors, and wandb
    also differ between the two manifests
- Expected behavior: The documented environment sources should describe one
  reproducible runtime or clearly identify supported alternatives.
- Actual behavior: Recreating from README, requirements, or YAML can yield
  materially different Python/CUDA/package stacks.
- Compatibility impact:
  - environment recreation is not deterministic
  - dependency changes could confound Repair Phase verification
- Current mitigation:
  - preserve the validated existing `memgen` environment
  - record exact installed versions in `EXP-20260611-003`
- Required resolution:
  - reconcile environment manifests in a separately approved documentation or
    reproducibility phase

### BUG-0004: PATH-Level Conda Shim Has CRLF Shebang

- Date found: 2026-06-11
- Severity: low
- Status: `open`
- Phase/experiment: Temporary Environment Alignment / `EXP-20260611-003`
- Environment: current shell PATH resolves `conda` to
  `/home/baishilong/bin/conda`
- Symptoms: invoking `conda` fails with
  `/bin/bash^M: bad interpreter: No such file or directory`
- Expected behavior: `conda env list` and `conda activate` should resolve to a
  valid shell entry point.
- Actual behavior: the wrapper's first line contains a CRLF terminator.
- Compatibility impact:
  - interactive activation through the PATH-level wrapper is unreliable
  - direct environment Python and `/home/baishilong/miniconda3/bin/conda` still
    work
- Current mitigation:
  - use `/home/baishilong/miniconda3/bin/conda`
  - or source `/home/baishilong/miniconda3/bin/activate memgen`
  - or invoke `/home/baishilong/miniconda3/envs/memgen/bin/python` directly
- Fix deferred: This phase did not modify shell files outside the repository.

## Bug Template

### BUG-NNNN: <Summary>

- Date found:
- Severity: `critical | high | medium | low`
- Status: `open | investigating | fixed | accepted | cannot_reproduce`
- Phase/experiment:
- Environment:
- Revision:
- Symptoms:
- Expected behavior:
- Actual behavior:
- Minimal reproduction:
- Logs/artifacts:
- Suspected cause:
- Root cause:
- Fix:
- Regression test:
- Compatibility impact:
- Related decision IDs:
- Date resolved:

## Research-Specific Watchlist

- Disabled mode differs from original outputs.
- Memory persists across samples or sessions.
- Retrieval mutates training behavior or training configuration.
- Batch behavior silently mixes memory state.
- Session reset is missing or incomplete.
- Latent shape, device, dtype, or precision mismatch.
- Added latency or memory use is not measured.
- Results cannot be reproduced from recorded commands.
