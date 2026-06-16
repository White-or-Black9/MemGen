# Code Map

Phase 1 audit completed on revision
`7a13d0abb8bdfcb851421d164a9a8223af22a55f`.

This document records the verified inference path, the protected training
boundaries, and the likely integration points for a later session-level
LatentMemoryBank. This phase used code reading only. No experiment was run.

Historical scope note:

- Sections 1 through 13 preserve the original Phase 1 code-reading audit.
- The implemented memory-bank path was added later and is summarized in
  Section 14 below.

## 1. Inference Entry and Main Dispatch

### CLI entry

- File: `main.py`
- Main function: `main()`
- Verified flow:
  1. parse CLI args with `parse_args()`
  2. load merged config with `Config(args)`
  3. set random seed via `set_seed(config.run.seed)`
  4. derive `working_dir`
  5. initialize logger
  6. build dataset through `get_data_builder(...)`
  7. build model through `MemGenModel.from_config(config.model)`
  8. construct `MemGenRunner`
  9. dispatch by `config.run.mode`

### Evaluation dispatch

- File: `memgen/runner.py`
- Main evaluation entry: `MemGenRunner.evaluate()`
- Verified branch:
  - `env_card == ENV_CARD.STATIC` -> `_static_evaluate()`
  - `env_card == ENV_CARD.DYNAMIC` -> `_dynamic_evaluate()`

## 2. Configuration Loading Flow

### Config object

- File: `common/config.py`
- Main class: `Config`
- Verified loading flow:
  1. `OmegaConf.load(args.cfg_path)` reads the YAML file.
  2. CLI overrides are converted into a dotlist by `_convert_to_dot_list`.
  3. `runner`, `model`, and `dataset` sections are merged with CLI overrides.
  4. `Config.to_dict()` returns a plain container for downstream builder code.

### Runtime config handoff

- `main.py` passes:
  - `config.run` to runner creation
  - `config.dataset` to data builder
  - `config.model` to `MemGenModel.from_config`
- `MemGenRunner._parse_configs()` converts `run` config into
  `InteractionConfig`, including:
  - `batch_size`
  - `max_prompt_length`
  - `max_response_length`
  - `max_turns`
  - `temperature`
  - `weaver_do_sample`
  - `trigger_do_sample`
  - `output_dir = os.path.join(self.working_dir, "evaluate")`

## 3. Session / Sample / Episode Boundaries

### Static evaluation boundary

- File: `memgen/runner.py`
- Function: `_static_evaluate()`
- One DataLoader item is one sample.
- `SingleTurnInteractionManager.run_agent_loop()` handles one batch of prompts.
- For Phase 1 planning, a safe session boundary is one call to
  `run_agent_loop()`.

### Dynamic evaluation boundary

- File: `memgen/runner.py`
- Function: `_dynamic_evaluate()`
- Each task instance is converted into one environment object through
  `_set_batch_envs()`.
- File: `interactions/multiturn_interaction.py`
- Function: `MultiTurnInteractionManager.run_agent_loop()`
- Verified loop:
  - initializes `inter_histories`
  - tracks `active_mask`
  - iterates `for step in range(self.config.max_turns)`
  - rebuilds chat history each turn
  - calls `generate()` each turn
- Therefore:
  - sample boundary = one environment instance
  - episode/session boundary = one full `run_agent_loop()`
  - turn boundary = one iteration of the `for step` loop

### Practical implication for memory-bank design

- The safest reset point is the interaction-manager session boundary.
- Until explicitly approved later, memory must remain session-local and must not
  cross sample boundaries.

## 4. Inference Pipeline Map

```text
main.py:main()
  -> Config
  -> get_data_builder(...).get_dataset_dict()
  -> MemGenModel.from_config()
  -> MemGenRunner.evaluate()
     -> STATIC: _static_evaluate()
        -> DataLoader
        -> tokenizer.apply_chat_template(...)
        -> InteractionDataProto.from_single_prompts(...)
        -> SingleTurnInteractionManager.run_agent_loop()
        -> MemGenModel.generate()
        -> StaticEvalRecorder
     -> DYNAMIC: _dynamic_evaluate()
        -> _set_batch_envs()
        -> InteractionDataProto(init_prompts=..., envs=...)
        -> MultiTurnInteractionManager.run_agent_loop()
        -> MemGenModel.generate() for each turn
        -> env.step(...)
        -> DynamicEvalRecorder
```

## 5. Trigger Call Site

### Trigger decision path

- Main file: `memgen/model/modeling_memgen.py`
- Main function: `MemGenModel.generate()`
- Trigger decision call:
  `augment_decision = self._should_augment(...)`

### Trigger implementation path

- File: `memgen/model/modeling_utils.py`
- Function: `_should_augment(...)`
- Verified behavior:
  - builds candidate subset using `candidate_mask`
  - selects the current hidden position
  - calls `trigger(...)` on selected rows
  - returns a boolean-like augmentation decision tensor

### Trigger tensor notes

- File: `memgen/model/trigger.py`
- Trigger head output shape: `[batch_size, seq_len, 2]`
- This shape is explicit in code.

## 6. Weaver Call Site

### Training path reference only

- File: `memgen/model/modeling_memgen.py`
- Function: `_forward(...)`
- Verified calls:
  - `weaver_inputs_embeds = self.reasoner_to_weaver(current_inputs_embeds)`
  - `self.weaver.augment_prompt(...)` or
    `self.weaver.augment_inference(...)`

### Inference path used by evaluation

- File: `memgen/model/modeling_memgen.py`
- Function: `generate()`
- Verified call sequence:
  1. `candidate_inputs_embeds` selected from reasoner inputs
  2. `weaver_inputs_embeds = self.reasoner_to_weaver(candidate_inputs_embeds)`
  3. `self.weaver.augment_prompt(...)` or
     `self.weaver.augment_inference(...)`
  4. `latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states)`

## 7. Latent Memory Generation and Injection

### Latent generation site

- File: `memgen/model/modeling_memgen.py`
- Function: `generate()`
- Verified source variable names:
  - `candidate_inputs_embeds`
  - `weaver_inputs_embeds`
  - `weaver_hidden_states`
  - `latent_inputs_embeds`

### Latent injection into Reasoner

- File: `memgen/model/modeling_memgen.py`
- Function: `generate()`
- Verified operation:
  - `candidate_inputs_embeds = torch.cat([candidate_inputs_embeds, latent_inputs_embeds], dim=1)`
- The augmented candidate tensors are then merged back into full-batch tensors,
  after which the reasoner generates the next token.

### Training-path reference

- File: `memgen/model/modeling_memgen.py`
- Function: `_forward(...)`
- Verified operation:
  - `latent_inputs_embeds` is concatenated into `current_inputs_embeds`
  - `current_attention_mask` is extended accordingly

## 8. Key Variables and Tensor Shapes

Shapes below are marked explicit when directly stated by code, or
`inferred from code` when deduced from surrounding operations.

### Reasoner-side embeddings

- Variable: `inputs_embeds`
- Location: `MemGenModel.generate()`
- Shape: `[B, L, H_reasoner]` (`inferred from code`)
  - code unpacks `B, _, hidden_size = inputs_embeds.shape`

### Trigger output

- Variable: trigger logits
- Location: `memgen/model/trigger.py`
- Shape: `[B, L, 2]` (explicit in code)

### Weaver augmentation tensors

- File: `memgen/model/weaver.py`
- Internal method: `_augment(...)`
- Input:
  - `inputs_embeds`: `[B, L, H_weaver]` (`inferred from code`)
  - `latents`: `[K, H_weaver]` (`inferred from code`)
- Output:
  - `latents_hidden_states`: `[B, K, H_weaver]` (`inferred from code`)
  - `latents_mask`: `[B, K]` (`inferred from code`)
  - `latents_position_ids`: `[B, K]` (`inferred from code`)

### Reasoner re-injected latents

- Variable: `latent_inputs_embeds`
- Location: `MemGenModel.generate()`
- Shape: `[B_selected, K, H_reasoner]` (`inferred from code`)

### Augmentation bookkeeping

- Variable: `augmentation_pos`
- Location: `MemGenModel.generate()`
- Shape: `[B, max_new_tokens]` (`inferred from code`)

### Forward-pass accumulation

- Variables:
  - `current_inputs_embeds`
  - `current_attention_mask`
- Location: `MemGenModel._forward(...)`
- Initial shapes:
  - `current_inputs_embeds`: `[B, 0, H_reasoner]` (`inferred from code`)
  - `current_attention_mask`: `[B, 0]` (`inferred from code`)

## 9. Generation Outputs and Evaluation Hooks

### Static outputs

- File: `memgen/runner.py`
- Function: `_static_evaluate()`
- Interaction output fields:
  - `responses`
  - `input_ids`
  - `attention_mask`
  - `info_mask`
- Responses are decoded with the tokenizer.
- Recorder:
  - file: `memgen/utils.py`
  - class: `StaticEvalRecorder`
- Output artifact:
  - `evaluate/answer.json`
- Static recorder writes per-example JSON lines and a final summary record.

### Dynamic outputs

- File: `memgen/runner.py`
- Function: `_dynamic_evaluate()`
- Recorder:
  - file: `memgen/utils.py`
  - class: `DynamicEvalRecorder`
- Output artifact:
  - `evaluate/conversations.txt`
- Dynamic recorder writes:
  - conversation transcript
  - per-item reward
  - final average reward

### TensorBoard hook

- File: `memgen/utils.py`
- Helper: `create_tensorboard(save_dir)`
- Output directory:
  - `<run_dir>/runs`

## 10. Weaver / Trigger Training Boundaries

These are protected boundaries under the current research scope and should not
be modified for the inference-only memory-bank phases.

### Runner-level training entry points

- `memgen/runner.py`
  - `MemGenRunner.train()`
  - `MemGenRunner._create_weaver_trainer()`
  - `MemGenRunner._create_trigger_trainer()`

### Trainer implementations

- `memgen/trainer/weaver_grpo_trainer.py`
- `memgen/trainer/trigger_grpo_trainer.py`

### Model training path

- `memgen/model/modeling_memgen.py`
  - `forward()`
  - `_instructional_forward()`
  - `_conversational_forward()`
  - `_forward()`

### Parameter-freeze / parameter-open controls

- `memgen/model/modeling_utils.py`
  - `fix_component()`
  - `open_component()`

### Training scripts and launch wrappers

- `scripts/train/**`
- `scripts/weaver_sft.sh`
- `scripts/weaver_grpo.sh`
- `scripts/trigger_train.sh`

## 11. Candidate LatentMemoryBank Integration Points

### Candidate A: session-owned object in interaction manager

- Possible owners:
  - `SingleTurnInteractionManager.run_agent_loop()`
  - `MultiTurnInteractionManager.run_agent_loop()`
- Advantage:
  - aligns naturally with session reset semantics
  - lower risk of cross-sample leakage
  - keeps training path untouched
- Risk:
  - requires explicit plumbing into `MemGenModel.generate()`
  - interface design must avoid silently mutating disabled path

### Candidate B: optional memory-state argument to `MemGenModel.generate()`

- Advantage:
  - closest to latent creation and latent injection sites
  - simplest place to keep retrieval/update logic numerically consistent
- Risk:
  - easy to accidentally persist state on the model instance
  - must be carefully isolated from training callers and trigger rollouts

### Candidate C: global field on `MemGenModel`

- Advantage:
  - implementation convenience only
- Risk:
  - cross-sample leakage
  - unclear reset semantics
  - higher chance of contaminating training or multi-sample inference
- Assessment:
  - not acceptable under current constraints

## 12. Phase 1 Risk Assessment

### High-risk areas

1. Session reset semantics
   - Dynamic evaluation rebuilds prompts every turn and re-calls `generate()`.
   - A bank attached to the wrong lifecycle can easily leak memory across
     episodes.

2. Disabled-path equivalence
   - `generate()` currently performs augmentation inline.
   - Any new conditional branch must be strictly no-op when disabled.

3. Shape / device / dtype compatibility
   - Latents cross `reasoner_to_weaver` and `weaver_to_reasoner`.
   - A bank storing post-Weaver or post-reasoner latents must make device and
     dtype transitions explicit.

4. Batch semantics
   - Current interaction code supports batching.
   - Memory-bank experiments therefore need conservative default
     `batch_size=1` until later approval.

5. Baseline trust
   - `BUG-0001` remains open.
   - Architectural audit can proceed, but baseline equivalence claims cannot be
     trusted until loader correctness is fixed.

## 13. Recommended Phase 1 Output

The audit supports the following working conclusion for later implementation:

- lifecycle owner: interaction-manager session
- model API: explicit optional session-memory argument into inference-only
  `generate()`
- forbidden approach: persistent global memory on `MemGenModel`
- protected scope: no changes to Weaver or Trigger training code paths

This is an audit conclusion only. No implementation is performed in Phase 1.

## 14. Implemented Memory Bank Path

This section records the current implemented Version A path after Phase 5
integration and the Phase R2 / R2-fix revisions. It does not describe
Version B.

### Core Memory Module

- File: `memgen/model/latent_memory_bank.py`
- Main public types:
  - `LatentMemoryBankConfig`
  - `LatentMemorySlot`
  - `LatentMemoryRetrievalResult`
  - `LatentMemoryBank`
- Main public methods:
  - `retrieve(...)`
  - `retrieve_with_context(...)`
  - `write(...)`
  - `write_back(...)`
  - `debug_summary()`
  - `state_dict()`

### Current Retrieval and Update State

- `_step` counts successful writes.
- `_retrieval_step` counts enabled retrieval turns.
- `last_retrieved_step` stores the last retrieval turn in which a slot was
  actually selected / returned.
- `last_retrieved_age` is derived as
  `current_retrieval_step - slot.last_retrieved_step`.
- Current Version A-aligned scoring uses last-retrieved decay rather than
  write-age decay.
- Only final selected / returned slots update `last_retrieved_step`.
- `threshold_topk` still has no fallback top-1.

### Current Full-Bank Eviction Rule

- `write_back(...)` with `update_policy=thread_update` performs:
  - matched-thread replacement when `max_score >= threshold`
  - new-thread insertion when similarity is below threshold and capacity remains
  - full-bank new-thread insertion by evicting the slot with largest
    `last_retrieved_age`
- Full-bank tie-break is deterministic:
  - earlier `created_step`
  - then lower slot index
- New replacement / inserted slots created by `write_back(...)` bind
  `last_retrieved_step` to `retrieval_result.retrieval_step`.

### Runtime Wiring

- Session owner:
  - `SingleTurnInteractionManager.run_agent_loop()`
  - `MultiTurnInteractionManager.run_agent_loop()`
- One bank is created per interaction-manager session / episode when
  `enabled=true`.
- `MemGenModel` does not hold a persistent global bank field.
- Enabled memory remains restricted to the current safe path:
  `batch_size=1`.

### Model Integration Boundary

- File: `memgen/model/modeling_memgen.py`
- Main inference hook: `MemGenModel.generate()`
- Current behavior:
  - the bank is passed in explicitly as a session-local argument
  - retrieved memories are injected only into the Reasoner-side candidate path
  - retrieved memories do not enter Weaver
  - Weaver still receives only the current context-derived inputs
  - stored memories are reasoner-space `latent_inputs_embeds` produced after
    `weaver_to_reasoner(...)`
- Therefore:
  - no fallback top-1
  - no Version B retrieval-to-Weaver behavior
  - no training-path modification

### Verification Coverage

- `tests/test_latent_memory_bank.py`
  - core retrieval, update, decay, eviction, and debug contracts
- `tests/test_latent_memory_bank_integration.py`
  - session-local lifecycle, disabled path, Reasoner-only injection, and
    no-Weaver-leakage checks
- `tests/test_controlled_multiturn_memory.py`
  - controlled-harness schema and scoring-contract checks used by the
    mechanism-study infrastructure
