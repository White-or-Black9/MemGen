# Decision Log

Record important architectural, experimental, and scope choices. Use immutable
IDs and append superseding decisions rather than silently rewriting history.

## Decision Index

| ID | Date | Status | Decision |
|---|---|---|---|
| DEC-0001 | 2026-06-11 | accepted | Modify inference only; preserve Weaver and Trigger training workflows |
| DEC-0002 | 2026-06-11 | accepted | Disabled memory bank must preserve exact original behavior |
| DEC-0003 | 2026-06-11 | superseded | Early isolation rule for Phase 1 only |
| DEC-0004 | 2026-06-11 | accepted | Execute one Phase at a time and pause after completion |
| DEC-0005 | 2026-06-11 | accepted | Use official Qwen2.5-1.5B GSM8K Weaver-SFT as the primary comparator |
| DEC-0006 | 2026-06-11 | accepted | Keep the baseline gate closed until all official LoRA tensors load without mismatch |
| DEC-0007 | 2026-06-11 | accepted | Until later approval, memory remains session-local and memory-bank experiments default to batch size 1 |
| DEC-0008 | 2026-06-11 | accepted | Future memory-bank state should be owned by the interaction session and passed explicitly into inference |
| DEC-0009 | 2026-06-11 | accepted | Use the `memgen` environment plus local cached snapshot paths for smoke verification; treat `base` as unsupported for MemGen runs |
| DEC-0010 | 2026-06-11 | accepted | Preserve the existing validated `memgen` package set through the Repair Phase; do not rebuild or install dependencies without new evidence and approval |

## Decision Template

### DEC-NNNN: <Title>

- Date:
- Status: `proposed | accepted | superseded | rejected`
- Context:
- Decision:
- Alternatives considered:
- Rationale:
- Consequences:
- Verification required:
- Related experiments:
- Supersedes:
- Superseded by:

## Standing Decisions

### DEC-0001: Inference-Only Research Scope

- Date: 2026-06-11
- Status: accepted
- Decision: Do not modify Weaver or Trigger training workflows.
- Consequence: All method integration and state management must occur in inference paths.

### DEC-0002: Strict Disabled-Path Compatibility

- Date: 2026-06-11
- Status: accepted
- Decision: `latent_memory_bank.enabled=false` must produce exactly the original behavior.
- Consequence: The disabled path must avoid new state, retrieval, mutation, and numerical effects.

### DEC-0003: Phase 1 Isolation

- Date: 2026-06-11
- Status: superseded
- Decision: Do not share memory across samples in Phase 1; default to `batch_size=1`.
- Consequence: Historical scope note only.

### DEC-0007: Pre-Approval Isolation Rule

- Date: 2026-06-11
- Status: accepted
- Context: The roadmap now allows the session-local and small-batch constraint to
  span multiple early phases rather than only Phase 1.
- Decision: Until explicitly approved in a later phase, memory must remain
  session-local, must not be shared across samples, and memory-bank experiments
  default to `batch_size=1`.
- Alternatives considered: limiting the rule to Phase 1 only.
- Rationale: The broader rule better protects disabled-path compatibility,
  reproducibility, and leakage control while the method is still being stabilized.
- Consequences: Any request to share memory across samples or increase batch size
  for memory-bank experiments requires explicit later-phase approval.
- Supersedes: `DEC-0003`

### DEC-0004: Phase Execution Gate

- Date: 2026-06-11
- Status: accepted
- Decision: Execute one approved Phase, update required notes, then pause.
- Consequence: No automatic progression to the next Phase.

### DEC-0008: Session-Owned Inference Memory State

- Date: 2026-06-11
- Status: accepted
- Context: Phase 1 code audit shows that static and dynamic evaluations both
  call `MemGenModel.generate()` repeatedly inside interaction-manager lifecycles,
  while session reset semantics live outside the model object.
- Decision: If a LatentMemoryBank is added later, its lifecycle owner should be
  the interaction-manager session, and any memory state should be passed
  explicitly into inference rather than persisted as a global field on
  `MemGenModel`.
- Alternatives considered:
  - storing persistent memory directly on `MemGenModel`
  - attaching memory only inside `generate()` local variables
- Rationale: Session ownership matches the verified reset boundary, reduces
  cross-sample leakage risk, and keeps training code paths isolated.
- Consequences:
  - future integration should add explicit inference-only state plumbing
  - global model-level memory is rejected under current constraints
- Verification required:
  - session reset must clear all bank contents
  - disabled path must remain numerically identical
  - no training caller should observe new persistent state

### DEC-0009: Recommended Smoke-Test Runtime

- Date: 2026-06-11
- Status: accepted
- Context: Phase 2 showed that the inherited shell environment points Hugging
  Face traffic at `hf-mirror.com` through a local proxy, the current `base`
  environment uses Python 3.13.9, and sandboxed execution hides CUDA from
  PyTorch.
- Decision: For smoke verification of the original MemGen project, use
  `/home/baishilong/miniconda3/envs/memgen` with Python 3.10.20, clear inherited
  proxy/HF endpoint variables for offline runs, and point model names at the
  local cached Qwen snapshot path when network access is unavailable.
- Alternatives considered:
  - using the current `base` environment
  - relying on repo-name resolution through proxy/mirror settings
- Rationale: The recommended environment is the only one observed to initialize
  FlashAttention-backed models and reach GPU generation consistently during
  Phase 2.
- Consequences:
  - future smoke or repair verification should start from the `memgen`
    environment
  - `base` should not be treated as a supported MemGen runtime
- Verification required:
  - cached model path loads without network access
  - CUDA is visible to PyTorch in the chosen execution context

### DEC-0010: Freeze the Validated Repair Environment

- Date: 2026-06-11
- Status: accepted
- Context: Environment alignment found inconsistent checked-in manifests, but
  the existing Python 3.10 environment passes imports, `pip check`, CUDA/BF16
  checks, local asset checks, and previously reached GPU generation.
- Decision: Use
  `/home/baishilong/miniconda3/envs/memgen/bin/python` unchanged for the Repair
  Phase. Do not recreate the environment, downgrade PyTorch, or install/update
  packages unless a repair test produces evidence that a dependency change is
  required and the user explicitly approves the command.
- Alternatives considered:
  - recreate from `memgen.yml`
  - reinstall from `requirements.txt`
  - downgrade PyTorch to either checked-in version
- Rationale: Changing the environment before repairing known code defects would
  introduce an uncontrolled variable and weaken causal diagnosis.
- Consequences:
  - Repair Phase results must record the existing exact package versions
  - environment changes require a separate explanation and approval
  - direct absolute Python invocation is preferred in automation
- Related experiments: `EXP-20260611-003`

### DEC-0005: Primary Baseline Comparator

- Date: 2026-06-11
- Status: accepted
- Context: A lightweight, official, static-task comparator is needed before the
  memory-bank implementation.
- Decision: Use the official
  `Qwen2.5-1.5B-Instruct/gsm8k/weaver-sft/pn=1_pl=8_in=3_il=8` checkpoint.
- Alternatives considered: random untrained MemGen, GSM8K GRPO, KodCode,
  TriviaQA, and SmolLM3.
- Rationale: The checkpoint is official, small to download, uses a deterministic
  static evaluator, has cached base weights/data, and exercises both prompt and
  inference latent augmentation.
- Consequences: Phase 1 compatibility evidence will initially target GSM8K,
  greedy decoding, and `batch_size=1`.
- Related experiments: `EXP-20260611-001`

### DEC-0006: Refuse Unloaded-Adapter Baselines

- Date: 2026-06-11
- Status: accepted
- Context: The current loader emits missing-key warnings for every trained LoRA
  tensor but may continue execution.
- Decision: A run is not a valid MemGen baseline unless adapter loading reports no
  unexplained missing or unexpected trained keys.
- Alternatives considered: Accepting outputs because generation can continue.
- Rationale: Such outputs would mostly represent random/unadapted components and
  cannot support scientific comparison.
- Consequences: The Phase 0 baseline gate remains closed pending `BUG-0001`.
- Related experiments: `EXP-20260611-001`
