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
| DEC-0011 | 2026-06-11 | accepted | Restore checkpoint adapters on the existing PEFT model after deleting constructor placeholders |
| DEC-0012 | 2026-06-11 | accepted | Preserve the static recorder batch contract by flattening only rank-nested gather results |
| DEC-0013 | 2026-06-11 | accepted | Accept a fixed first-20 GSM8K test subset as the Phase 3 development baseline |
| DEC-0014 | 2026-06-11 | accepted | Keep the Phase 4 memory bank standalone, session-owned, and disabled by default |
| DEC-0015 | 2026-06-11 | accepted | Detach and clone every stored latent, with explicit storage and retrieval conversion |
| DEC-0016 | 2026-06-11 | accepted | Use mean-pooled cosine retrieval with recency decay and bounded replacement skeletons |

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

### DEC-0011: Replace Placeholder Adapters During Checkpoint Restore

- Date: 2026-06-11
- Status: accepted
- Context: MemGen construction creates named adapters from the current YAML
  before checkpoint restoration. Wrapping the resulting `LoraModel` again
  changes key prefixes and prevents official q/v-only tensors from loading.
- Decision: In `MemGenModel.from_pretrained()`, delete each constructor-created
  placeholder adapter and load the saved adapter into the existing PEFT model
  under the same component name.
- Alternatives considered:
  - wrap `model.base_model` again
  - force checkpoint tensors into the YAML's broader adapter layout
  - change training-time adapter construction
- Rationale: The selected approach uses the checkpoint's own adapter config,
  avoids nested PEFT wrappers, and leaves training initialization untouched.
- Consequences: Restored models use the exact saved target modules and weights.
- Verification: 112/112 Weaver and 112/112 Trigger tensors match checkpoint
  keys, shapes, and values with no missing or unexpected entries.
- Related experiment: `EXP-20260611-004`
- Related bug: `BUG-0001`

### DEC-0012: Keep Static Recorder Inputs Batch-Shaped

- Date: 2026-06-11
- Status: accepted
- Context: `StaticEvalRecorder.record_batch()` requires aligned completion and
  example lists, while distributed gathering may add one rank nesting level.
- Decision: Flatten gathered results only when the first gathered batch element
  is itself a list, then call the recorder once with aligned flat lists.
- Alternatives considered:
  - make the recorder accept scalar strings and dictionaries
  - bypass the official recorder
  - change the shared gathering helper
- Rationale: This preserves the recorder and metric contracts, supports both
  single-process and rank-nested inputs, and limits the fix to static eval.
- Consequences: Static answer logging works without changing evaluation
  semantics or any training workflow.
- Related experiment: `EXP-20260611-004`
- Related bug: `BUG-0002`

### DEC-0013: Fixed 20-Sample Development Baseline

- Date: 2026-06-11
- Status: accepted
- Context: Phase 3 requires a credible comparator before method implementation,
  but a first full-test run would add cost without improving early disabled-path
  and ablation iteration.
- Decision: Accept GSM8K `main/test` indices 0 through 19 as the frozen Phase 3
  development comparison set, with seed 42, batch size 1, greedy decoding, and
  maximum response length 1024.
- Alternatives considered:
  - 50 fixed samples
  - the full 1,319-sample test split
  - retaining the 128-token smoke configuration
- Rationale:
  - 20 samples exercise repeated Trigger/Weaver augmentation and official metric
    recording while keeping the first formal run bounded
  - 1024 tokens matches the official eval setting and avoids smoke-test
    truncation
  - fixed contiguous IDs make later comparisons and replay unambiguous
- Consequences:
  - `compute_reward=0.60` is valid only for the fixed 20-sample subset
  - every later comparison must use the same IDs and protocol
  - larger runs may strengthen evidence but do not replace this oracle silently
- Verification:
  - 20/20 predictions plus one summary completed
  - three golden samples replayed with exact response and mask hashes
- Related experiments: `EXP-20260611-006`, `EXP-20260611-007`

### DEC-0014: Standalone Session-Local Phase 4 Skeleton

- Date: 2026-06-11
- Status: accepted
- Context: Phase 4 must create a testable module without changing original
  inference or training behavior.
- Decision:
  - each `LatentMemoryBank` instance owns one session's slots
  - there is no global registry or cross-sample storage
  - `enabled` defaults to `false`
  - the module is not exported from `memgen.model` and is not imported by any
    production inference or training path
  - Phase 4 accepts only batch size 1 tensor inputs
- Alternatives considered:
  - attach a bank field to `MemGenModel`
  - add the config directly to existing GSM8K configuration
  - create a process-global bank
- Rationale: Physical module isolation is the strongest guarantee that Phase 4
  cannot alter the accepted Original MemGen baseline.
- Consequences: Phase 5 must explicitly design lifecycle ownership and inference
  plumbing before the bank can be used.
- Verification: repository search found no production references; importing
  `MemGenModel` does not load the new module.
- End-of-day verification: compilation and 16/16 unit tests passed; production
  inference, existing GSM8K configuration, and protected training paths still
  have no Phase 4 integration diff.
- Related experiment: `EXP-20260611-008`

### DEC-0015: Detached Storage and Explicit Tensor Conversion

- Date: 2026-06-11
- Status: accepted
- Context: Stored latent tensors must not retain inference computation graphs or
  rely on implicit device/dtype movement.
- Decision:
  - `write()` stores `detach().clone()`
  - retrieval returns detached clones
  - original device and dtype are recorded
  - `storage_device` is explicitly `cpu` or `same`
  - retrieval accepts explicit output `device` and `dtype`
- Alternatives considered:
  - store original tensor references
  - automatically follow the current model device without recording conversion
- Rationale: Detached copies prevent graph retention and explicit conversion
  makes future CPU/GPU transfer costs and precision behavior auditable.
- Consequences: CPU storage may add transfer latency in later phases; that cost
  must be measured after inference integration.
- Verification: tests mutate source tensors after write, inspect grad
  properties, and validate output dtype/device.
- Related experiment: `EXP-20260611-008`

### DEC-0016: Minimal Retrieval and Capacity Policies

- Date: 2026-06-11
- Status: accepted
- Context: Phase 4 needs deterministic mechanics without claiming an optimal
  retrieval algorithm.
- Decision:
  - query: mean of the most recent `pool_last_n` hidden tokens
  - key: mean of all tokens in one memory slot
  - score: cosine similarity multiplied by exponential age decay
  - retrieval: `threshold`, `topk`, or `threshold_topk`
  - full-bank update: reject under `append`, replace lowest score under
    `replace`, or replace oldest under `replace_oldest`
- Alternatives considered:
  - learned query/key projections
  - attention aggregation
  - immediate implementation of paper ablations
- Rationale: These policies expose necessary research controls while remaining
  small enough to validate independently.
- Consequences: Phase 8 must compare these choices; Phase 4 makes no performance
  claim.
- Related experiment: `EXP-20260611-008`

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
- Consequences at decision time: The baseline gate remained closed pending
  `BUG-0001`; the later Repair Phase resolved the bug and Phase 3 opened the
  gate.
- Related experiments: `EXP-20260611-001`
