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
| DEC-0017 | 2026-06-12 | accepted | Phase 5 keeps the bank interaction-owned and passes it explicitly into `MemGenModel.generate()` |
| DEC-0018 | 2026-06-12 | accepted | Version A stores reasoner-space latents and injects retrieved memory only into the Reasoner path |
| DEC-0019 | 2026-06-12 | accepted | Phase 6 disabled-path equivalence requires exact baseline hashes, metrics, and augmentation call counts on the frozen 20-sample comparator |
| DEC-0020 | 2026-06-12 | accepted | Phase 7 enabled-path stability passes only on bounded session-local debug evidence and without performance claims |
| DEC-0021 | 2026-06-12 | accepted | Phase 7 replacement-path evidence may use debug-only CLI overrides, but must stay on the real enabled inference path |
| DEC-0022 | 2026-06-12 | accepted | Define Version A as conservative Reasoner-only memory injection without fallback top-1 |
| DEC-0023 | 2026-06-12 | accepted | Define Version B as full retrieval-to-Weaver recurrent latent update with fallback top-1 and matched-slot write-back |
| DEC-0024 | 2026-06-12 | accepted | Do not interpret current write-age decay as last-retrieved-turn decay |
| DEC-0025 | 2026-06-12 | accepted | Treat Phase 8A GSM8K as a sanity and negative pilot, not main method evidence |
| DEC-0026 | 2026-06-12 | accepted | Move primary evaluation focus toward TriviaQA dynamic multi-turn inference |
| DEC-0027 | 2026-06-12 | accepted | Add structured retrieval context without changing retrieval or write/update semantics |
| DEC-0028 | 2026-06-12 | accepted | Add thread_update as a method-aligned Version A write-back policy |
| DEC-0029 | 2026-06-12 | accepted | Accept bounded real-path plus controlled-test evidence for thread_update mechanism validation |
| DEC-0030 | 2026-06-12 | accepted | Reuse the verified Phase 6 disabled anchor for Phase 8A |
| DEC-0031 | 2026-06-12 | accepted | Treat Phase 8A as a stability-first pilot rather than a final performance experiment |
| DEC-0032 | 2026-06-12 | accepted | Complete Version A-aligned thread_update and gate Version B behind a TriviaQA target-task baseline |
| DEC-0033 | 2026-06-12 | accepted | Use a controlled three-turn fallback only as mechanism evidence while TriviaQA infrastructure is blocked |
| DEC-0034 | 2026-06-13 | accepted | Freeze strict and deterministic relaxed scoring before controlled group comparison |
| DEC-0035 | 2026-06-16 | accepted | Revise Version A-aligned decay and full-bank eviction to last-retrieved semantics without entering Version B |

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

### DEC-0017: Interaction-Owned Phase 5 Runtime State

- Date: 2026-06-12
- Status: accepted
- Context: Phase 5 must integrate the bank into inference without storing any
  cross-session state on `MemGenModel`, and without forcing existing baseline
  configs to change.
- Decision: The interaction manager owns the session-local bank for the duration
  of one `run_agent_loop()` call and passes it explicitly into
  `MemGenModel.generate(...)`.
- Alternatives considered:
  - storing the bank on `MemGenModel`
  - storing the bank as a long-lived runner field
  - restructuring the entire config system around a required memory-bank schema
- Rationale: This matches the verified reset boundary, preserves session
  isolation, and keeps the configuration change optional and minimal.
- Consequences:
  - single-turn calls create one bank per sample session
  - multi-turn calls create one bank per episode
  - disabled mode can skip bank construction entirely
- Verification required:
  - no cross-sample leakage across repeated `run_agent_loop()` calls
  - no enabled-path access when `batch_size > 1`
  - disabled-path golden replay remains exact
- Related experiments: `EXP-20260612-010`, `EXP-20260612-011`

### DEC-0018: Reasoner-Space Version A Storage and Injection

- Date: 2026-06-12
- Status: accepted
- Context: Phase 5 only permits Reasoner-side retrieval and injection; retrieved
  memory must not enter Weaver, and stored latent dimensionality must already
  match the Reasoner path.
- Decision:
  - retrieve using Reasoner-side candidate inputs
  - store only `latent_inputs_embeds` after `weaver_to_reasoner(...)`
  - inject retrieved memories only into the Reasoner sequence
  - keep the disabled path on the original branch
- Alternatives considered:
  - storing `weaver_hidden_states`
  - sending retrieved memory through `reasoner_to_weaver()`
  - unifying enabled and disabled branches behind one shared tensor pipeline
- Rationale: Storing Reasoner-space latents avoids hidden-size mismatch and
  keeps Version A tightly scoped to Reasoner injection.
- Consequences:
  - retrieved memory and new latent memory require separate mask bookkeeping
  - debug counters must distinguish retrieved and newly generated latents
  - Version B remains a separate future phase
- Verification required:
  - retrieved memory never reaches Weaver inputs
  - written memory matches Reasoner-space latent tensors
  - disabled-path hashes and call counts remain exact
- Related experiments: `EXP-20260612-010`, `EXP-20260612-011`

### DEC-0019: Phase 6 Equivalence Acceptance Standard

- Date: 2026-06-12
- Status: accepted
- Context: Phase 6 needs a clear pass/fail rule for disabled-path equivalence so
  that any regression becomes a blocking bug rather than an informal judgment.
- Decision: Treat Phase 6 as passing only if the disabled-path run on frozen
  GSM8K test IDs `0..19` matches `EXP-20260611-006` exactly on:
  - response-token SHA-256 hashes
  - augmentation-mask SHA-256 hashes
  - prediction count and summary-record presence
  - summary `compute_reward`
  - Trigger decision call count
  - Weaver prompt augmentation call count
  - Weaver inference augmentation call count
  - adapter loading integrity
  - absence of any constructed memory-bank debug state
- Alternatives considered:
  - metric-only comparison
  - hash-only comparison on a smaller golden subset
  - allowing call-count drift if outputs remained identical
- Rationale: The frozen 20-sample baseline is the accepted development oracle,
  and exact matching across outputs plus control-flow statistics is the strongest
  practical disabled-path guarantee before later enabled-path studies.
- Consequences:
  - any mismatch is a blocking regression
  - no enabled-path claim is implied by a Phase 6 pass
  - passing Phase 6 only authorizes consideration of later phases, not their
    automatic execution
- Related experiments: `EXP-20260612-013`

### DEC-0020: Phase 7 Enabled-Path Stability Acceptance Standard

- Date: 2026-06-12
- Status: accepted
- Context: Phase 7 needs a bounded pass/fail rule for enabled Version A that
  checks mechanism stability without turning the phase into a performance study.
- Decision: Treat Phase 7 as passing only if bounded enabled runs complete
  without crash, NaN, OOM, CUDA error, shape mismatch, device mismatch, or
  dtype mismatch; each single-turn session starts from `initial_slots=0`; no
  cross-sample leakage appears; stored slot tensors remain reasoner-space
  latents; `slot_count` never exceeds `max_slots`; and the debug trace remains
  consistent with retrieved memory staying out of Weaver.
- Alternatives considered:
  - judge Phase 7 primarily by reward or accuracy changes
  - skip session-level trace capture and rely only on final bank summaries
  - extend Phase 7 directly into longer or larger enabled runs
- Rationale: Enabled Version A still needs mechanism validation more than
  quality comparison. Session-local isolation and tensor correctness are the
  main claims at this stage.
- Consequences:
  - reward and `compute_reward` may be recorded as auxiliary outputs only
  - bounded tiers are sufficient for a pass when all invariants hold
  - larger enabled studies belong to later approved phases
- Verification required:
  - one-sample Tier 1 smoke
  - three-sample Tier 2 session-isolation check
  - five-sample Tier 3 bounded-capacity check
- Related experiments:
  - `EXP-20260612-015`
  - `EXP-20260612-016`
  - `EXP-20260612-017`

### DEC-0021: Phase 7 Replacement-Path Supplement Standard

- Date: 2026-06-12
- Status: accepted
- Context: Phase 7 passed with one warning because the bounded five-sample run
  did not naturally reach `max_slots=8`, so the real enabled replacement path
  was not observed directly.
- Decision: A Phase 7 supplement may lower memory-bank capacity through
  debug-harness-only CLI overrides, provided the run still uses the real
  enabled inference path, fixed seed, batch size `1`, the frozen GSM8K config
  file, and no training or baseline artifacts are changed.
- Alternatives considered:
  - leave the warning unresolved and defer replacement-path evidence to a later
    phase
  - modify the main config file to force lower capacity
  - add a synthetic non-inference test instead of using the real enabled path
- Rationale: Lowering `max_slots` in the debug harness is the smallest way to
  trigger replacement in the real mechanism without turning the supplement into
  a broader experiment or mutating the frozen baseline config.
- Consequences:
  - the supplement remains a mechanism check, not a performance study
  - debug-only CLI overrides are allowed for capacity-trigger evidence
  - disabled-path equivalence evidence remains untouched because no disabled
    branch or generate semantics changed
- Verification required:
  - `memory_write_count > max_slots`
  - `slot_count <= max_slots`
  - explicit replacement evidence such as `replace_count > 0` or
    `update_action_trace`
  - no runtime or tensor-contract failure
- Related experiments:
  - `EXP-20260612-018`

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

### DEC-0030: Phase 8A Reuses the Verified Disabled Anchor

- Date: 2026-06-12
- Status: accepted
- Context: Phase 8A needs a disabled comparator but Phase 6 already verified
  exact disabled-path equivalence on the same 20 GSM8K test IDs.
- Decision:
  - reuse `EXP-20260612-013` as the current-harness disabled anchor
  - continue to cite `EXP-20260611-006` as the frozen original baseline
  - do not spend another full disabled rerun unless the harness or disabled
    path changes
- Alternatives considered:
  - rerun G0 again only for table symmetry
- Rationale: Phase 6 already established that the current disabled path matches
  the frozen baseline exactly, so reusing that anchor preserves rigor without
  duplicating compute.
- Consequences:
  - Phase 8A can focus its compute budget on enabled variants
  - any future disabled rerun becomes necessary only if disabled-path semantics
    or the reporting harness changes
- Related experiments:
  - `EXP-20260611-006`
  - `EXP-20260612-013`

### DEC-0031: Treat Phase 8A as a Stability-First Pilot

- Date: 2026-06-12
- Status: accepted
- Context: The current repository supports a small set of Version A controls,
  but broader retrieval variants such as latest-k and random are not yet
  implemented.
- Decision:
  - Phase 8A compares only currently implemented groups `G0`, `G1`, `G4`,
    `G6`, and `G7`
  - the first pass stays on the fixed 20-sample slice `0..19`
  - negative or flat results are recorded directly
  - no pilot result is treated as a final performance conclusion
- Alternatives considered:
  - implement more retrieval policies immediately
  - skip the pilot and jump straight to a larger main ablation
- Rationale: A narrow pilot reduces moving parts, verifies that current controls
  behave cleanly, and gives an early signal before widening the protocol.
- Consequences:
  - after the method audit, the next expansion should transition to an aligned
    dynamic multi-turn target rather than directly scaling GSM8K
  - new retrieval policies remain separately gated design variants
- Related experiments:
  - `EXP-20260612-019`
  - `EXP-20260612-020`
  - `EXP-20260612-021`
  - `EXP-20260612-022`

### DEC-0022: Conservative Version A Definition

- Date: 2026-06-12
- Status: accepted
- Context: Phase 5 through Phase 8A implemented a low-risk mechanism whose
  retrieved memories stay outside Weaver.
- Decision:
  - Version A is conservative Reasoner-only memory injection
  - if the bank is empty, retrieval returns empty
  - if no score reaches threshold, retrieval returns empty
  - Version A has no fallback top-1
  - Weaver receives only current context `H_t`
  - Reasoner receives `[R_t; m_t]` when retrieval succeeds, otherwise only
    newly generated `m_t`
  - every Weaver-generated `m_t` is written back when Trigger fires
- Rationale: This definition matches the implemented behavior and preserves the
  original Weaver input distribution.
- Consequences:
  - Phase 5 through Phase 8A results are Version A-simple results
  - they must not be described as evidence for the full proposed method
- Related experiments:
  - `EXP-20260612-011`
  - `EXP-20260612-015`
  - `EXP-20260612-019`

### DEC-0023: Full Version B Definition

- Date: 2026-06-12
- Status: accepted
- Context: The original method proposal requires retrieved memory to influence
  generation of the next recurrent latent and to support thread-aware updates.
- Decision:
  - Version B performs `retrieve -> Weaver revise/generate -> write-back`
  - a non-empty bank falls back to the argmax slot when no score reaches
    threshold
  - retrieved memory enters Weaver together with current context
  - Reasoner continues with the newly generated latent `m_t`
  - write-back inserts a new slot for a new thread and replaces the matched
    argmax slot for an existing thread
  - decay uses turns since last retrieval and updates explicit
    `last_retrieved_turn` or `last_retrieved_step`
- Rationale: This separates the full research method from the conservative
  integration used for early compatibility and stability work.
- Consequences:
  - Version B remains unimplemented
  - Version B requires separate implementation, compatibility, stability, and
    target-task evidence

### DEC-0024: Current Decay Is Write-Age Decay

- Date: 2026-06-12
- Status: accepted
- Context: Read-only implementation audit found that current scoring uses
  `_step - created_step`, while `_step` counts successful writes.
- Decision:
  - describe current decay as write-age exponential decay
  - do not call it last-retrieved-turn decay
  - describe Phase 8A G1 versus G4 as write-age decay versus no decay
- Rationale: Retrieval does update `last_access_step`, but current scoring does
  not use that field and does not count dialogue turns.
- Consequences:
  - method-aligned last-retrieved decay becomes an explicit future variant
  - prior experimental values remain valid, but their interpretation is
    narrowed
- Related experiments:
  - `EXP-20260612-019`
  - `EXP-20260612-020`

### DEC-0025: Phase 8A Is Sanity and Negative Pilot Evidence

- Date: 2026-06-12
- Status: accepted
- Context: Phase 8A used 20 short single-turn GSM8K samples and found stable but
  lower enabled results.
- Decision:
  - treat Phase 8A as a short single-turn sanity and negative pilot
  - record the observed enabled underperformance directly
  - do not interpret it as failure of the full Version B method
  - do not expand it directly into the primary main experiment without a
    target-task change
- Rationale: GSM8K does not test the primary multi-turn, long-trajectory, or
  context-truncation hypothesis.
- Consequences:
  - Phase 8A remains useful stability and negative evidence
  - the next main-evidence plan must use a better-aligned task
- Related experiments:
  - `EXP-20260612-013`
  - `EXP-20260612-019`
  - `EXP-20260612-020`
  - `EXP-20260612-021`
  - `EXP-20260612-022`

### DEC-0026: TriviaQA as the Next Primary Target Candidate

- Date: 2026-06-12
- Status: accepted
- Context: Repository audit found that TriviaQA is the available dynamic task
  with repeated search/answer turns, growing interaction history, and
  observation truncation.
- Decision: Shift the next primary evaluation focus toward TriviaQA, beginning
  with Original MemGen and disabled-memory baseline planning.
- Alternatives considered:
  - continue scaling GSM8K
  - use static GPQA
  - use static KodCode
- Rationale: TriviaQA is better aligned with session-local persistence, reuse
  across turns, long trajectories, and context truncation.
- Consequences:
  - establish a trusted TriviaQA baseline before enabled comparisons
  - verify retrieval backend and dynamic-evaluation reproducibility before
    claiming method evidence
  - Phase 9 remains gated behind Version A evidence on the target task

### DEC-0027: Structured Retrieval Context Before Write-Back Changes

- Date: 2026-06-12
- Status: accepted
- Context: A future matched-thread write-back policy needs the current query's
  complete slot scores, maximum score, argmax slot, and filtered retrieval
  selection. The legacy `retrieve()` API exposes only cloned selected slots.
- Decision:
  - add immutable `LatentMemoryRetrievalResult`
  - add `retrieve_with_context(...)`
  - preserve full scores in original bank slot order
  - compute `max_score` and `argmax_index` before threshold/top-k filtering
  - use the lowest original slot index to break equal-score ties
  - keep `retrieve(...)` as a compatibility wrapper returning only `.slots`
- Rationale: This creates an explicit, testable handoff for a later write-back
  step while isolating the change from current inference and update behavior.
- Consequences:
  - current `write()` and all existing update policies remain unchanged
  - no `thread_update`, fallback top-1, or last-retrieved decay is introduced
  - `MemGenModel.generate()` remains unchanged and does not yet consume the
    structured result

### DEC-0028: Method-Aligned Version A Thread Update

- Date: 2026-06-12
- Status: accepted
- Context: Existing update policies are capacity-driven and cannot express
  low-similarity new-thread insertion versus high-similarity matched-thread
  replacement.
- Decision:
  - add `update_policy=thread_update`
  - add `write_back(memory, retrieval_result, metadata=None)`
  - replace the current argmax slot when `max_score >= threshold`, regardless
    of remaining capacity
  - insert a new slot when similarity is below threshold
  - when a new-thread insertion finds a full bank, evict the oldest slot as a
    separate capacity-management action
  - reject stale retrieval contexts by requiring matching bank steps
- Rationale: The update must use the current query's structured retrieval
  result rather than mutable or stale slot `last_score` state.
- Consequences:
  - existing `append`, `replace`, and `replace_oldest` semantics remain intact
  - retrieved memory remains Reasoner-only and Weaver input remains unchanged
  - this remains Version A and does not implement fallback top-1,
    last-retrieved decay, or Version B retrieval-to-Weaver behavior

### DEC-0029: Thread-Update Mechanism Validation Standard

- Date: 2026-06-12
- Status: accepted
- Context: A one-sample real inference smoke may not naturally exercise every
  score and capacity branch, and enlarging GSM8K is not justified for a
  mechanism-only check.
- Decision:
  - require at least one real enabled inference session to validate runtime,
    write-back traces, Reasoner-only injection, and reasoner-space storage
  - allow deterministic unit tests to supply branch evidence not observed in
    that bounded real session
  - do not treat the resulting evidence as a performance experiment
- Rationale: This validates actual integration while avoiding an unnecessary
  larger run on a task that is not aligned with the primary research
  hypothesis.
- Consequences:
  - `EXP-20260612-024` observes `empty_bank` and `matched_thread` in real
    inference
  - unit tests validate `new_thread` and `new_thread_bank_full`
  - the next main activity should return to target-task planning rather than
    scaling the GSM8K smoke

### DEC-0032: Gate Version B Behind the TriviaQA Baseline

- Date: 2026-06-12
- Status: accepted
- Context: Steps 2 through 4 completed structured retrieval context,
  Version A-aligned `thread_update`, disabled replay, and bounded real-path
  mechanism validation. The project still lacks a baseline on its intended
  dynamic multi-turn target.
- Decision:
  - treat Version A-aligned `thread_update` implementation as completed
  - complete notes review and commit preparation before further experiments
  - plan and establish the TriviaQA Original MemGen / disabled-memory baseline
    next
  - do not enter Version B before the target-task baseline is stable
- Rationale: Target-task evidence is now a larger research gap than additional
  mechanism expansion.
- Consequences:
  - TriviaQA baseline planning is the next research activity
  - last-retrieved decay and fallback top-1 remain later Version A variants
  - retrieved-memory-to-Weaver Version B remains not started
- Related experiments:
  - `EXP-20260612-023-step3-disabled-replay`
  - `EXP-20260612-024-thread-update-smoke`

### DEC-0033: Controlled Multi-Turn Fallback Is Mechanism Evidence Only

- Date: 2026-06-12
- Status: accepted
- Context: TriviaQA cannot currently run because its checkpoint, datasets, and
  retrieval service are unavailable, while cross-turn memory persistence still
  needs a bounded real-model check.
- Decision:
  - add a harness-only deterministic three-turn evaluation
  - strictly remove prior visible history from the final query
  - compare disabled, Version A-simple, and Version A-aligned modes only when
    explicitly run
  - treat all results as mechanism or sanity evidence
  - do not substitute this protocol for a real dynamic target-task baseline
- Rationale: The controlled task isolates lifecycle and leakage behavior at low
  infrastructure cost without changing MemGen core logic.
- Consequences:
  - synthetic exact match cannot support a main performance claim
  - GSM8K-checkpoint distribution mismatch must accompany every result
  - negative outcomes do not reject Version B or the full research hypothesis
  - TriviaQA remains the intended target-task route when infrastructure exists
- Related experiments:
  - `EXP-20260612-025`
  - `EXP-20260612-026`
  - `EXP-20260612-027`

### DEC-0034: Freeze the Controlled Prompt and Dual-Metric Parser Contract

- Date: 2026-06-13
- Status: accepted
- Context: `EXP-20260613-001` generated the correct visible oracle answer but
  omitted `<answer>` tags, causing the strict-only parser to report `0/1`.
- Decision:
  - use one strengthened one-line tagged-output instruction for all groups
  - report both `strict_exact_match` and `relaxed_exact_match`
  - keep strict parsing limited to complete answer tags
  - allow exact-code relaxed extraction only for exactly one standalone
    six-digit candidate
  - treat multiple exact-code candidates as ambiguous
  - evaluate semantic fallback only as normalized complete-response exact match
  - prohibit gold-aware extraction, LLM judges, and fuzzy semantic matching
  - retain legacy `exact_match` only as a deprecated strict-metric alias
- Rationale: The policy separates format compliance from deterministic answer
  correctness without adding subjective scoring or gold-guided extraction.
- Consequences:
  - the same frozen prompt and parser must be used by G0/G1/G2/G3
  - `EXP-20260612-026`, `EXP-20260612-027`, and `EXP-20260613-001` are
    pre-parser-calibration smoke runs, not final comparison results
  - calibrated G0/G2/G3 one-episode reruns are required before considering G1
    or a larger controlled pilot
  - controlled evaluation remains a mechanism study and does not replace
    TriviaQA
  - fallback top-1, last-retrieved decay, and Version B remain unimplemented
- Related experiments:
  - `EXP-20260612-026`
  - `EXP-20260612-027`
  - `EXP-20260613-001`

### DEC-0035: Version A-Aligned Last-Retrieved Decay Revision

- Date: 2026-06-16
- Status: accepted
- Context: Phase R2 changes only the Version A-aligned `thread_update` mechanism.
  Historical Version A-simple and earlier Phase 8A / Phase 8C-alt results remain
  write-age-decay evidence.
- Decision:
  - add an enabled retrieval-turn counter for the bank
  - compute Version A-aligned retrieval score with
    `current_retrieval_step - slot.last_retrieved_step`
  - update `last_retrieved_step` only for final selected / returned slots
  - initialize newly inserted or matched-replacement slots at the current
    retrieval step
  - when `thread_update` inserts a new thread into a full bank, evict the slot
    with largest `last_retrieved_age`
  - break eviction ties by earlier `created_step`, then lower slot index
  - keep `retrieval_result.bank_step` stale-context protection
- Rationale: This aligns Version A-aligned decay and full-bank capacity behavior
  with actual retrieval reuse recency rather than slot creation age.
- Consequences:
  - Version A-aligned no longer uses write-age decay
  - Version A-simple remains a historical / legacy baseline variant
  - no fallback top-1 is introduced
  - retrieved memory remains Reasoner-only and does not enter Weaver
  - Version B remains not started
