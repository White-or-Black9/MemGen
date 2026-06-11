# Bug and Anomaly Log

Record code defects, experiment anomalies, regressions, and suspected data leaks.
Do not delete resolved entries.

## Bug Index

| ID | Date | Severity | Status | Summary |
|---|---|---|---|---|
| BUG-0001 | 2026-06-11 | high | `open` | Official Weaver/Trigger LoRA adapters are not loaded by `MemGenModel.from_pretrained()` |

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
- Required fix verification:
  - No unexplained missing/unexpected adapter keys.
  - Loaded tensor equality against official safetensors.
  - Deterministic generation repeat.
  - No changes to training paths.

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
