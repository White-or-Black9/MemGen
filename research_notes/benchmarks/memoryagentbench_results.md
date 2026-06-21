# MemoryAgentBench Results Summary

## 1. Scope

This note summarizes the current MemoryAgentBench work on MemGen and preserves the over-context finding for the original MemGen full-history path.

## 2. Current Baseline Status

- `Original MemGen Full-history Bank-off` is valid only when the rebuilt prompt fits within the model context capacity.
- `Original MemGen Full-history over-capacity` is invalid and should not be used as a baseline.
- `Original MemGen Truncated-history Bank-off` is not the same experiment as full-history and, if studied later, must be labeled separately.
- `Compressed Bank-off` is the valid compressed lower-bound condition.
- `Compressed Bank-on` is the valid LatentBank memory condition.

## 3. Original MemGen Over-Context Behavior

The original MemGen multi-turn/full-history path has no explicit over-context guard.

Source inspection showed:

- full multi-turn history is rendered and tokenized in [multiturn_interaction.py](/mnt/18T/baishilong/MemGen/interactions/multiturn_interaction.py#L169) through [multiturn_interaction.py](/mnt/18T/baishilong/MemGen/interactions/multiturn_interaction.py#L176),
- `apply_chat_template()` is called without `truncation=True`,
- the multi-turn path does not compare rendered prompt length against `max_prompt_length` before `MemGenModel.generate()`,
- `MemGenModel.generate()` has no explicit `max_position_embeddings` or `input_len + max_new_tokens` guard,
- Trigger/Weaver augmentation can expand effective sequence length further.

The current MAB harnesses add explicit preflight guards, for example in [mab2_bank_off.py](/mnt/18T/baishilong/MemGen/scripts/eval/mab2_bank_off.py#L278) and [mab_paired_bank_off_vs_low_threshold_bank_on.py](/mnt/18T/baishilong/MemGen/scripts/eval/mab_paired_bank_off_vs_low_threshold_bank_on.py#L255).

## 4. Synthetic Diagnostic

Controlled Bank-off diagnostic artifact:
[over_context_diagnostic.json](/mnt/18T/baishilong/MemGen/outputs/mab/memgen_over_context_behavior/20260620T133105Z-over-context/over_context_diagnostic.json)

Test setup:

- original MemGen Bank-off
- `batch_size=1`
- `max_new_tokens=1`
- synthetic prompts near the observed `32768` context capacity

Observed result:

| Requested tokens | Actual input tokens | Result |
|---:|---:|---|
| 32000 | 32000 | succeeded |
| 32760 | 32760 | succeeded |
| 32800 | 32800 | succeeded |
| 35000 | 35000 | `OutOfMemoryError` |

Interpretation:

- `32800 > 32768` still generated successfully.
- No silent truncation was observed in the synthetic diagnostic.
- No explicit over-capacity warning was emitted.
- Failure appeared later as CUDA OOM at `35000`.

## 5. DetectiveQA Preflight

The real-data preflight for `Long_Range_Understanding / detective_qa` showed:

- selected context id: `lru-cd66eabd2f070a38`
- estimated full-history query tokens: `102477`
- capacity: `32768`
- status: over-capacity
- generation called: `false`

This means the task is not a valid full-history original-MemGen baseline under the current protocol.

## 6. Baseline Taxonomy

- `Original MemGen Full-history Bank-off`: valid only under capacity
- `Original MemGen Full-history over-capacity`: invalid marker, not run
- `Original MemGen Truncated-history Bank-off`: optional future diagnostic, not the same as full-history
- `Compressed Bank-off`: valid compressed lower bound
- `Compressed Bank-on`: valid LatentBank memory condition

## 7. Recommendation

Do not use raw over-capacity full-history behavior as a valid baseline.

For `detective_qa`, the next valid experiment is compressed-memory `n=10`, not full-history.

Proposed next step:

`MAB-5A: detective_qa Compressed-memory Bank-off vs Bank-on n10`

Reason:

Full-history original MemGen is over-capacity and invalid for this task. Compressed-memory directly tests whether LatentBank can support answer generation when full dialogue history cannot fit.

## 8. MAB-5A Preservation

Completed experiment:

- `MAB-5A: detective_qa Compressed-memory Bank-off vs Bank-on n10`
- artifact root: `outputs/mab/compressed_memory_detectiveqa_n10/20260621T013454Z-detectiveqa-compressed-n10/`
- valid contexts: `10/10`
- compressed Bank-off accuracy: `0.0`
- compressed Bank-on accuracy: `0.0`
- delta: `0.0`
- output changed: `10/10`
- retrieval active in every context
- no cross-context leakage observed
- query write stayed disabled / read-only

Mechanism note:

- retrieved scores were roughly `0.030-0.064`
- final slot counts stayed low (`[1, 2, 2, 5, 6, 5, 6, 7, 4, 7]`), which is consistent with over-merge / over-compression under the current low threshold
- the current `thread_update` path compares `candidate_inputs_embeds` against existing `slot.key` before Weaver produces the new latent
- the stored memory itself is Weaver-generated `latent_inputs_embeds`
- one threshold currently couples retrieval visibility and write/update behavior

Preservation note:

- do not implement decoupled thresholds yet
- next mechanism experiment should separate retrieve/update thresholds after this preservation commit

## 8. Git Status

### Before

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

### After

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
?? research_notes/benchmarks/memoryagentbench_results.md
```
