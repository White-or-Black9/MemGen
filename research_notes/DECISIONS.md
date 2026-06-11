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
