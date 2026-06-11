# Bug and Anomaly Log

Record code defects, experiment anomalies, regressions, and suspected data leaks.
Do not delete resolved entries.

## Bug Index

| ID | Date | Severity | Status | Summary |
|---|---|---|---|---|
| BUG-0001 | 2026-06-11 | high | `fixed` | Official Weaver/Trigger LoRA adapters were not loaded by `MemGenModel.from_pretrained()` |
| BUG-0002 | 2026-06-11 | high | `fixed` | Static evaluation crashed in `StaticEvalRecorder.record_batch()` after generation started |
| BUG-0003 | 2026-06-11 | medium | `open` | Checked-in environment specifications disagree on Python, CUDA, and package versions |
| BUG-0004 | 2026-06-11 | low | `open` | PATH-level `conda` wrapper has a CRLF shebang and cannot execute |

## Recorded Bugs

### BUG-0001: Nested PEFT Loading Skips Official LoRA Weights

- Date found: 2026-06-11
- Severity: high
- Status: `fixed`
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
- Root cause: `from_pretrained()` passed an already wrapped `LoraModel` into a
  second `PeftModel.from_pretrained()` call. This added an extra model prefix and
  retained the YAML's broader target-module configuration instead of restoring
  the checkpoint's q/v-only adapter configuration.
- Fix: Delete the constructor's placeholder named adapter, then call
  `load_adapter()` on the existing PEFT model using the checkpoint directory and
  original adapter name.
- Regression verification:
  - Weaver: 112/112 tensors, exact key/shape/value match.
  - Trigger: 112/112 tensors, exact key/shape/value match.
  - No adapter missing/unexpected warnings.
  - Official one-sample generation completed.
- Training impact: Initial adapter construction and all Weaver/Trigger trainer
  code are unchanged. Only checkpoint restoration is corrected.
- Related experiment: `EXP-20260611-004`
- Related decision: `DEC-0011`
- Date resolved: 2026-06-11
- Repair review verification:
  - `EXP-20260611-005` loaded Weaver and Trigger adapters with exact 112/112
    tensor matches across a three-sample official static eval.
  - No missing, unexpected, shape-mismatched, or value-mismatched entries
    reappeared.
  - Protected training files had no git diff.
- Phase 3 verification:
  - `EXP-20260611-006` again loaded both adapters with exact 112/112 key, shape,
    and value matches.
  - No missing or unexpected trained keys appeared across 20 completed samples.
- End-of-day verification:
  - `EXP-20260611-006/verification.json` remains readable and records Weaver
    112/112 and Trigger 112/112.
  - Missing, unexpected, shape-mismatch, and value-mismatch lists remain empty.

#### 修复说明（中文）

模型构造阶段已经根据 YAML 创建了命名为 `weaver` 和 `trigger` 的
`PeftModel/LoraModel`。原 checkpoint 恢复代码却把
`model.<component>.model.base_model` 再传给 `PeftModel.from_pretrained()`，
相当于在已有 LoRA 包装层外再次创建 PEFT 包装层。

这会产生两个问题：

1. runtime key 比 checkpoint 多一层 `base_model.model` 前缀以及 adapter
   名后缀；
2. runtime 继续采用 YAML 中覆盖七类 projection 的宽 LoRA 配置，共预期
   392 个 tensor，而官方 checkpoint 实际采用 q/v-only 配置，每个组件
   只有 112 个训练 tensor。

修复时没有修改训练阶段如何创建 adapter，而是只修改 checkpoint 恢复：

```python
component.model.delete_adapter(adapter_name)
component.model.load_adapter(checkpoint_path, adapter_name=adapter_name)
component.model.set_adapter(adapter_name)
```

这样会删除构造阶段的占位 adapter，再由 checkpoint 的
`adapter_config.json` 恢复正确的 q/v-only 结构和权重，避免二次包装。
最终对 Weaver 和 Trigger 分别进行了 key、shape 和 tensor value 的逐项
比较，均为 112/112 完全一致。

### BUG-0002: Static Eval Recorder Expects the Wrong Batch Shape

- Date found: 2026-06-11
- Severity: high
- Status: `fixed`
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
- Root cause: In a non-distributed run, `gather_objects()` returns the original
  flat lists. The runner then zipped those lists and passed one completion
  string and one example dictionary to `record_batch()`, whose contract is
  `List[str]` plus `List[Dict]`. Accessing `examples[0]` therefore raised
  `KeyError: 0`.
- Fix: Normalize rank-nested gathered lists only when necessary, then call
  `record_batch()` once with aligned flat completion and example lists.
- Regression verification:
  - isolated recorder contract test wrote one record plus summary
  - official one-sample static evaluation wrote a 1,006-byte `answer.json`
    containing one prediction plus summary
  - sample reward remained computed through the official metric hook
- Training impact: None; only static evaluation output collation changed.
- Related experiment: `EXP-20260611-004`
- Related decision: `DEC-0012`
- Date resolved: 2026-06-11
- Repair review verification:
  - `EXP-20260611-005` wrote exactly three prediction records and one summary
    record through the official recorder path.
  - No `KeyError: 0` or output collation error reappeared.
  - Trigger and Weaver augmentation tracing remained active for all samples.
- Phase 3 verification:
  - `EXP-20260611-006` wrote 20 prediction records and one summary without
    recorder errors.
  - All samples completed and the output remained valid JSONL.
- End-of-day verification:
  - The existing Phase 3 `answer.json` remains non-empty and parses as 20
    prediction JSONL records plus one summary record.
  - The summary remains readable with `compute_reward=0.60`.

#### 修复说明（中文）

`StaticEvalRecorder.record_batch()` 的接口契约是：

```python
record_batch(completions: List[str], examples: List[Dict])
```

单进程运行时，`gather_objects()` 直接返回原来的扁平列表。原 runner
随后对两个列表执行 `zip()` 并逐项调用 recorder，实际传入的类型变成：

```text
completions = str
examples = dict
```

因此 recorder 执行 `examples[0]` 时会把 `0` 当作字典键，触发
`KeyError: 0`。这与 `Dataset.select(range(1))` 无关，也不是单样本字段
发生了变化。

修复方案保持 recorder 接口和评测语义不变：

- 单进程结果已经是扁平列表，直接批量传入；
- 分布式结果如果存在 rank 级嵌套，只展开这一层；
- completion 与 example 保持相同顺序，一次调用 `record_batch()`。

修复后官方 static eval 写出一条预测记录和一条 summary，指标仍通过原
`compute_reward` hook 计算。

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
