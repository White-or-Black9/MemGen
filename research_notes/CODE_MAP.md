# Code Map

Verified during Phase 0 on revision
`5e59fee296092fa056f140b38a07b927651ffdb5`.

## Repository Entry Points

| Area | Path | Symbol/Command | Role | Verified |
|---|---|---|---|---|
| CLI | `main.py` | `main()` | Load config, data, model, runner; dispatch train/evaluate | Yes |
| Configuration | `common/config.py` | `Config` | OmegaConf load and CLI override merge | Yes |
| Model construction | `memgen/model/modeling_memgen.py` | `MemGenModel.from_config()` | Load reasoner, Weaver, Trigger, and checkpoint | Yes |
| Inference orchestration | `memgen/runner.py` | `MemGenRunner.evaluate()` | Select static or dynamic evaluation | Yes |
| Static session | `interactions/singleturn_interaction.py` | `run_agent_loop()` | One `generate()` call per sample batch | Yes |
| Dynamic session | `interactions/multiturn_interaction.py` | `run_agent_loop()` | Rebuild full chat history and call `generate()` each turn | Yes |
| Core generation | `memgen/model/modeling_memgen.py` | `MemGenModel.generate()` | Trigger, Weaver latent injection, reasoner generation | Yes |
| Static metrics | `memgen/utils.py` | `StaticEvalRecorder` | Per-example JSONL plus summary metrics | Yes |
| Dynamic metrics | `memgen/utils.py` | `DynamicEvalRecorder` | Conversation log plus average reward | Yes |

## Inference Data Flow

```text
main.py
  -> Config
  -> get_data_builder(...).get_dataset_dict()
  -> MemGenModel.from_config()
  -> MemGenRunner.evaluate()
     -> STATIC: DataLoader -> SingleTurnInteractionManager.run_agent_loop()
     -> DYNAMIC: env instances -> MultiTurnInteractionManager.run_agent_loop()
        -> MemGenModel.generate()
           -> _should_augment()
           -> reasoner_to_weaver
           -> Weaver.augment_prompt() / augment_inference()
           -> weaver_to_reasoner
           -> latent embeddings appended to reasoner input
           -> generated token IDs
        -> task reward/metric recorder
```

## Latent Tensor Flow

- Reasoner token embeddings: `[B, L, H_reasoner]`.
- `reasoner_to_weaver`: maps to `[B, L, H_weaver]`.
- Weaver appends learned query latents and returns
  `[B_selected, K, H_weaver]`.
- `weaver_to_reasoner`: maps memory to
  `[B_selected, K, H_reasoner]`.
- Latents are appended to `current_inputs_embeds`; they are not token IDs.
- Any new bank should store a clearly selected representation and preserve dtype,
  device, sample ownership, and insertion semantics.

## State and Lifecycle

- Static sample boundary: one item from `_static_evaluate()`'s test DataLoader.
- Static session boundary: one `SingleTurnInteractionManager.run_agent_loop()` call.
- Dynamic sample boundary: one environment instance created in `_set_batch_envs()`.
- Dynamic session boundary: one complete `MultiTurnInteractionManager.run_agent_loop()`.
- Dynamic turn boundary: each iteration rebuilds chat history and calls
  `MemGenModel.generate()` again.
- Existing persistent model state: `MemGenModel.state`, used for training-mode
  instruction/conversation detection; no inference memory bank exists.
- Existing inference-local state: embeddings, masks, KV cache, and augmentation
  counts are local variables inside `generate()` and are discarded on return.
- Required Phase 1 reset point: start/end of one interaction manager session, not
  process lifetime and not global model lifetime.
- Phase 1 batch rule: default and validate `batch_size=1`.

## Protected Training Boundaries

### Weaver

- Runner entry: `MemGenRunner._create_weaver_trainer()` and `train()`.
- Trainers: `trl.SFTTrainer` and
  `memgen/trainer/weaver_grpo_trainer.py`.
- Model training path: `MemGenModel.forward()`, `_instructional_forward()`,
  `_conversational_forward()`, and `_forward()`.
- Parameter controls: `fix_component()` / `open_component()`.

### Trigger

- Runner entry: `MemGenRunner._create_trigger_trainer()` and `train()`.
- Trainer: `memgen/trainer/trigger_grpo_trainer.py`.
- Training signal: `generate(return_augmentation_mask=True)`.
- Parameter controls: `fix_component()` / `open_component()`.

### Files Protected From Phase 1 Changes

- `memgen/trainer/**`
- `scripts/train/**`
- `scripts/weaver_sft.sh`
- `scripts/weaver_grpo.sh`
- `scripts/trigger_train.sh`
- Training branches of `MemGenRunner.train()` and `MemGenModel.forward()`

## Candidate Inference Integration Points

| Candidate | Advantages | Risks | Decision |
|---|---|---|---|
| `MemGenModel.generate()` optional argument/state object | Closest to latent creation and insertion | Easy to leak state through model instance; shared by Trigger training rollout | Conditional candidate |
| `InteractionManager.run_agent_loop()` session object | Owns static/dynamic session lifecycle and reset | Does not directly own latent tensors | Preferred lifecycle owner |
| Global field on `MemGenModel` | Simple | Cross-sample/process leakage and training interference | Rejected |

The likely Phase 1 design is an explicit session-local memory object owned by the
interaction lifecycle and passed into inference-only generation. This remains a
proposal until Phase 1 is approved.

## Evaluation Outputs

- Static: `.cache/evaluate/<dataset>/<model>/<run>/evaluate/answer.json`.
- Static headline metric key: `compute_reward`, mean direction `higher`.
- Dynamic: `.cache/evaluate/<dataset>/<model>/<run>/evaluate/conversations.txt`.
- Dynamic headline metric: final average reward, direction `higher`.
- TensorBoard is written under each run directory.

## Verified Commands

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python main.py --help
/home/baishilong/miniconda3/envs/memgen/bin/python -m compileall -q \
  main.py common data interactions memgen
```

Full baseline execution is currently blocked by `BUG-0001`.
