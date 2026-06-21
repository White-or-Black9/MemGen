# MemGen Over-Context Behavior

## 1. Objective

Inspect how original MemGen handles rebuilt multi-turn/full-history prompts that exceed the model context limit, then validate the behavior with a controlled Bank-off synthetic diagnostic near the observed `32768`-token capacity.

This is not a benchmark run. No MemoryAgentBench performance evaluation was executed.

## 2. Why This Matters for MAB and LatentBank Motivation

Current MAB full-history Bank-off runs rebuild the complete dialogue each turn. The local task audit already showed that candidates such as `Long_Range_Understanding / detective_qa` are far beyond `32768` tokens in full-history mode. If original MemGen silently truncates, continues unsupported, or only fails later via OOM, then those samples cannot be treated as a valid full-history baseline.

This also matters for LatentBank motivation: if full-history Bank-off over-capacity samples are not explicitly blocked, then later Bank-off vs Bank-on comparisons become scientifically ambiguous.

## 3. Checkpoint / Model Context Capacity

- Public checkpoint ID:
  `Kana-s/MemGen@269d9b1/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model`
- Observed reasoner context capacity from the loaded model config:
  `32768`
- Controlled diagnostic artifact:
  [over_context_diagnostic.json](/mnt/18T/baishilong/MemGen/outputs/mab/memgen_over_context_behavior/20260620T133105Z-over-context/over_context_diagnostic.json)

The machine-readable artifact contains the local checkpoint path because that field was explicitly required for the run record. This note keeps the public checkpoint ID as the stable identifier.

## 4. Source Inspection Findings

### 4.1 Where Full Chat History Is Rendered

- Multi-turn full history is rendered in [multiturn_interaction.py](/mnt/18T/baishilong/MemGen/interactions/multiturn_interaction.py#L169) through [multiturn_interaction.py](/mnt/18T/baishilong/MemGen/interactions/multiturn_interaction.py#L176).
- `messages = self._build_chat_history(rollings_active)` contains `init_prompt + inter_history`, so `apply_chat_template()` receives the complete message history for each active sample.

### 4.2 Does `apply_chat_template()` Receive Full Message History?

Yes.

- `_build_chat_history()` concatenates the initial prompt and accumulated interaction history in [multiturn_interaction.py](/mnt/18T/baishilong/MemGen/interactions/multiturn_interaction.py#L45) through [multiturn_interaction.py](/mnt/18T/baishilong/MemGen/interactions/multiturn_interaction.py#L57).
- `run_agent_loop()` then tokenizes that full history with `apply_chat_template()` in [multiturn_interaction.py](/mnt/18T/baishilong/MemGen/interactions/multiturn_interaction.py#L170) through [multiturn_interaction.py](/mnt/18T/baishilong/MemGen/interactions/multiturn_interaction.py#L176).

### 4.3 Does Tokenization Use `truncation=True` Anywhere?

Not in the multi-turn prompt path inspected here.

- The multi-turn `apply_chat_template()` call in [multiturn_interaction.py](/mnt/18T/baishilong/MemGen/interactions/multiturn_interaction.py#L172) has no `truncation=True`.
- The single-turn path also does not use tokenizer truncation before calling `generate()`; it trims only to effective non-padding length in [singleturn_interaction.py](/mnt/18T/baishilong/MemGen/interactions/singleturn_interaction.py#L118) through [singleturn_interaction.py](/mnt/18T/baishilong/MemGen/interactions/singleturn_interaction.py#L137).

### 4.4 Is `max_length` / `max_prompt_length` Applied Before `MemGenModel.generate()`?

For original multi-turn interaction: no explicit prompt-length cap is enforced before `generate()`.

- `InteractionConfig.max_prompt_length` exists in [base_interaction.py](/mnt/18T/baishilong/MemGen/interactions/base_interaction.py#L13) through [base_interaction.py](/mnt/18T/baishilong/MemGen/interactions/base_interaction.py#L31), but original `MultiTurnInteractionManager.run_agent_loop()` does not compare rendered prompt length against it before the model call.
- In single-turn interaction, `max_prompt_length` is applied only when clipping accumulated responses in [singleturn_interaction.py](/mnt/18T/baishilong/MemGen/interactions/singleturn_interaction.py#L97) through [singleturn_interaction.py](/mnt/18T/baishilong/MemGen/interactions/singleturn_interaction.py#L100). That is not the MAB full-history path.

### 4.5 Is There Any Explicit Full-History Truncation Helper?

No explicit multi-turn prompt truncation helper was found.

- The only explicit truncation logic in the interaction layer is observation truncation in [multiturn_interaction.py](/mnt/18T/baishilong/MemGen/interactions/multiturn_interaction.py#L242) through [multiturn_interaction.py](/mnt/18T/baishilong/MemGen/interactions/multiturn_interaction.py#L263).
- Padding helpers in [tensor_utils.py](/mnt/18T/baishilong/MemGen/interactions/tensor_utils.py) and left-pad/clip helpers in [modeling_utils.py](/mnt/18T/baishilong/MemGen/memgen/model/modeling_utils.py#L464) through [modeling_utils.py](/mnt/18T/baishilong/MemGen/memgen/model/modeling_utils.py#L525) are not used as a multi-turn prompt-length guard before generation.

### 4.6 Is There Any Silent Left/Right Truncation?

- Multi-turn prompt history: no explicit silent truncation path was found before generation.
- Observations: yes, explicit truncation with `...` suffix exists in [multiturn_interaction.py](/mnt/18T/baishilong/MemGen/interactions/multiturn_interaction.py#L242) through [multiturn_interaction.py](/mnt/18T/baishilong/MemGen/interactions/multiturn_interaction.py#L263).
- Single-turn response accumulation: yes, response-side clipping to `max_prompt_length` exists in [singleturn_interaction.py](/mnt/18T/baishilong/MemGen/interactions/singleturn_interaction.py#L97) through [singleturn_interaction.py](/mnt/18T/baishilong/MemGen/interactions/singleturn_interaction.py#L100).

## 5. Does Original `MemGenModel.generate()` Have Explicit Over-Context Handling?

No.

Relevant behavior in [modeling_memgen.py](/mnt/18T/baishilong/MemGen/memgen/model/modeling_memgen.py#L406):

- No check against `reasoner.config.max_position_embeddings`.
- No check that `input_len + max_new_tokens <= capacity`.
- No warning path for over-capacity prompt length.
- No invalid-sample marking.

The model directly:

- converts `input_ids` to embeddings in [modeling_memgen.py](/mnt/18T/baishilong/MemGen/memgen/model/modeling_memgen.py#L441) through [modeling_memgen.py](/mnt/18T/baishilong/MemGen/memgen/model/modeling_memgen.py#L450),
- computes position ids in [modeling_memgen.py](/mnt/18T/baishilong/MemGen/memgen/model/modeling_memgen.py#L458) through [modeling_memgen.py](/mnt/18T/baishilong/MemGen/memgen/model/modeling_memgen.py#L463),
- then continues generation.

## 6. Does It Rely on Hugging Face to Error?

Effectively yes, or on lower-level runtime failure.

There is no MemGen-side check. Generation is delegated to:

- direct `reasoner(...)` forward in [modeling_memgen.py](/mnt/18T/baishilong/MemGen/memgen/model/modeling_memgen.py#L650) through [modeling_memgen.py](/mnt/18T/baishilong/MemGen/memgen/model/modeling_memgen.py#L657),
- and in one branch to `reasoner.generate(...)` in [modeling_memgen.py](/mnt/18T/baishilong/MemGen/memgen/model/modeling_memgen.py#L625) through [modeling_memgen.py](/mnt/18T/baishilong/MemGen/memgen/model/modeling_memgen.py#L639).

The synthetic run showed that Hugging Face / reasoner did not stop the `32800`-token case despite `max_position_embeddings = 32768`.

## 7. Does Trigger / Weaver Change Effective Length and Capacity Accounting?

Yes.

- Prompt-stage augmentation always participates in the MemGen generate loop. If trigger chooses augment on the prompt step, new prompt latents are appended in [modeling_memgen.py](/mnt/18T/baishilong/MemGen/memgen/model/modeling_memgen.py#L513) through [modeling_memgen.py](/mnt/18T/baishilong/MemGen/memgen/model/modeling_memgen.py#L539).
- Therefore the effective sequence seen by the reasoner can be longer than the rendered input prompt itself.
- With LatentBank enabled, retrieval can expand the sequence even further before the reasoner call in [modeling_memgen.py](/mnt/18T/baishilong/MemGen/memgen/model/modeling_memgen.py#L553) through [modeling_memgen.py](/mnt/18T/baishilong/MemGen/memgen/model/modeling_memgen.py#L598).

For this diagnostic, LatentMemoryBank remained disabled, but original Trigger/Weaver prompt augmentation behavior still applied.

## 8. Do Current MAB Runners Have Explicit Truncation or Invalid Marking?

Current benchmark harnesses do have explicit preflight guards, even though original MemGen does not.

- The MAB-2 audited Bank-off runner blocks over-capacity rendered history in [mab2_bank_off.py](/mnt/18T/baishilong/MemGen/scripts/eval/mab2_bank_off.py#L269) through [mab2_bank_off.py](/mnt/18T/baishilong/MemGen/scripts/eval/mab2_bank_off.py#L282).
- The paired runner also blocks when `preflight_query_tokens + 8 + 10 > context_capacity` in [mab_paired_bank_off_vs_low_threshold_bank_on.py](/mnt/18T/baishilong/MemGen/scripts/eval/mab_paired_bank_off_vs_low_threshold_bank_on.py#L252) through [mab_paired_bank_off_vs_low_threshold_bank_on.py](/mnt/18T/baishilong/MemGen/scripts/eval/mab_paired_bank_off_vs_low_threshold_bank_on.py#L256).

So the risk is not that prior MAB-2 / MAB-3 artifacts silently used over-capacity prompts. The risk is any future Bank-off full-history run that bypasses these added guards.

## 9. Synthetic Diagnostic Setup

Script:

- [diagnose_memgen_over_context.py](/mnt/18T/baishilong/MemGen/scripts/eval/diagnose_memgen_over_context.py)

Settings:

- Original MemGen Bank-off only.
- Same public checkpoint family used by current MAB runs.
- `batch_size = 1`
- `max_new_tokens = 1`
- No MemoryAgentBench inference.
- Synthetic two-message chat prompt:
  - system
  - one large user message
- Tested actual tokenized prompt lengths:
  - `32000`
  - `32760`
  - `32800`
  - `35000`
- Stop condition:
  first confirmed over-capacity failure.

Execution note:

- The sandbox-visible Python process could not access CUDA even though `nvidia-smi` was visible.
- The actual diagnostic was therefore run once outside the sandbox on a real CUDA device, without changing code or model weights.

## 10. Synthetic Diagnostic Results

| Requested tokens | Actual input tokens | Over capacity? | Generation called | Result | Output tokens | Peak CUDA memory | Notes |
|---:|---:|---:|---:|---|---:|---:|---|
| 32000 | 32000 | no | yes | success | 1 | 23093885952 | below nominal limit |
| 32760 | 32760 | no | yes | success | 1 | 23411605504 | near nominal limit |
| 32800 | 32800 | yes | yes | success | 1 | 23429271040 | over nominal limit, no truncation, no capacity error |
| 35000 | 35000 | yes | yes | `OutOfMemoryError` | — | 15752949760 | failed by CUDA memory, not by explicit context check |

Observed global warning during execution:

- `temperature` was reported as an ignored generation flag by Transformers.
- This warning is unrelated to over-context handling.

No over-context-specific warning was emitted.

## 11. Observed Behavior Summary

### 11.1 Errors?

Yes, but only at `35000`, and the error was CUDA OOM rather than an explicit context-limit exception.

### 11.2 Silent Truncation?

Not observed.

- `32800` requested tokens remained `32800` actual input tokens.
- Generation still succeeded.

### 11.3 Explicit Truncation?

Not observed for multi-turn/full-history prompts before model call.

### 11.4 Continue Generation Despite Over-Capacity?

Yes.

`32800 > 32768` still generated successfully for one token in original MemGen Bank-off mode.

### 11.5 Warning-Only Behavior?

No over-capacity warning was observed.

### 11.6 Path Dependence?

Yes.

- Original MemGen generate path: no explicit guard, can continue beyond nominal context capacity until runtime failure.
- Current MAB benchmark runners: explicitly reject over-capacity full-history prompts before the model call.

## 12. Optional Real MAB Preflight Only

For `Long_Range_Understanding / detective_qa`:

- selected context id: `lru-cd66eabd2f070a38`
- estimated full-history query tokens: `102477`
- context capacity: `32768`
- status: over-capacity
- generation called: `false`

This was a preflight only. No real detectiveQA inference was executed.

## 13. Is Current Behavior Safe Enough to Use as a Baseline?

Original MemGen alone: no.

Reason:

- it has no explicit over-context handling,
- it can continue generating above the nominal context limit,
- and larger cases can fail only later via OOM.

Current guarded MAB harness: yes, if the explicit preflight is preserved and treated as mandatory.

## 14. Recommendation for Full-History Over-Capacity Samples

Recommendation:

- mark them invalid before generation,
- stop explicitly,
- do not use silently continued or runtime-failed outputs as the full-history baseline.

Do not introduce a truncated full-history baseline implicitly. If a truncated baseline is ever studied, it must be defined as a separate controlled condition rather than treated as “original full-history.”

## 15. Recommended Guardrail to Add Later

Later, add one unified explicit preflight guard before every full-history MemGen Bank-off / Bank-on generation call:

- compute rendered full-history token length after `apply_chat_template()`,
- add any known latent/prompt augmentation safety margin,
- compare against the actual loaded reasoner capacity,
- abort with a structured invalid marker before `MemGenModel.generate()`.

That guard should live in the harness layer, not as an implicit side effect of OOM or backend behavior.

## 16. Final Recommendation

The correct policy is:

- treat full-history over-capacity samples as invalid before generation,
- keep the current MAB runner preflight behavior,
- do not trust raw original MemGen behavior as a safe baseline in over-context regimes,
- and only revisit those samples under a separately defined protocol such as compressed-memory or an explicitly validated long-context model.

## 17. Git Status Before and After

```text
## rlm-memory-bank...origin/rlm-memory-bank [ahead 8]
 M memgen/model/modeling_memgen.py
?? research_notes/benchmarks/
?? scripts/eval/diagnose_memgen_over_context.py
?? scripts/eval/mab2_bank_off.py
?? scripts/eval/mab2_mab_bridge.py
?? scripts/eval/mab3_bank_on_full_history.py
?? scripts/eval/mab3a_threshold_ablation.py
?? scripts/eval/mab4a_compressed_memory.py
?? scripts/eval/mab_paired_bank_off_vs_low_threshold_bank_on.py
?? tests/test_mab2_bank_off.py
?? tests/test_mab3_bank_on_full_history.py
?? tests/test_mab3a_threshold_ablation.py
?? tests/test_mab4a_compressed_memory.py
?? tests/test_mab_paired_bank_off_vs_low_threshold_bank_on.py
```
