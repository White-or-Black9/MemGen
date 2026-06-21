# MAB-5A: detective_qa Compressed-memory Bank-off vs Bank-on n10

## Objective
Test whether LatentBank helps on detective_qa when the full dialogue history is over capacity.

## Why Original Full-history Is Invalid
The over-context diagnostic showed the original full-history path exceeds the 32,768-token capacity for detective_qa, so the full-history baseline is marked over_capacity_invalid and was not executed.

## Over-context Reference
See `scripts/eval/diagnose_memgen_over_context.py` and `outputs/mab/memgen_over_context_behavior/20260620T133105Z-over-context/over_context_diagnostic.json`.

## Dataset And Subtask
- Split: `Long_Range_Understanding`
- Subtask: `detective_qa`
- Contexts: 10 local rows

## Protocol
- First query only.
- Process each context as one session.
- Run compressed Bank-off and compressed Bank-on.
- Do not run full-history generation for detective_qa.

## Baseline Taxonomy
- Original MemGen full-history: over_capacity_invalid.
- Compressed Bank-off: no LatentBank, compressed query only.
- Compressed Bank-on: LatentBank enabled, sequential chunk writes, read-only query proxy.

## Settings
- Threshold: `0.03`
- top_k: `1`
- max_slots: `8`
- retrieve_policy: `threshold_topk`
- query_mode: `first-query-only`

## Query Read-only Status
The query turn used a no-op proxy for bank writes, so query_write_count remained 0 while retrieval stayed active.

## Prompt Leakage Checks
The compressed query prompt was checked for chunk-text and acknowledgement-history leakage per context.

## Reasoner-only Injection Checks
Retrieved latents were checked to enter Reasoner and not Weaver.

## Mechanism Finding
The completed run suggests over-merge / over-compression under the current low threshold:

- `threshold=0.03`, `top_k=1`, `max_slots=8`, `update_policy=thread_update`
- retrieved scores were roughly in the `0.030-0.064` range
- final slot counts stayed low: `[1, 2, 2, 5, 6, 5, 6, 7, 4, 7]`
- example evidence:
  - context 0: 25 chunks, 26 writes, final slots = 1
  - context 8: 50 chunks, 51 writes, final slots = 4

Source-level clarification:

- retrieval compares `candidate_inputs_embeds` against existing `slot.key`
- the comparison happens before Weaver produces the new latent
- the written / updated memory is Weaver-generated `latent_inputs_embeds`
- the single threshold currently couples retrieval visibility and write/update behavior

That means the low threshold can keep retrieval non-empty while also causing repeated replace/update behavior instead of appending new slots.

## Per-context Result Table
| context_index | exact_match_off | exact_match_on | output_changed | improved | regressed | retrieval_count | est_full_history_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | True | 0 | 0 | 25 | 112574 |
| 1 | 0 | 0 | True | 0 | 0 | 25 | 112224 |
| 2 | 0 | 0 | True | 0 | 0 | 25 | 112350 |
| 3 | 0 | 0 | True | 0 | 0 | 29 | 130490 |
| 4 | 0 | 0 | True | 0 | 0 | 29 | 127579 |
| 5 | 0 | 0 | True | 0 | 0 | 27 | 120143 |
| 6 | 0 | 0 | True | 0 | 0 | 30 | 132726 |
| 7 | 0 | 0 | True | 0 | 0 | 41 | 179614 |
| 8 | 0 | 0 | True | 0 | 0 | 50 | 220669 |
| 9 | 0 | 0 | True | 0 | 0 | 35 | 156193 |

## Aggregate Result Table
- Compressed Bank-off accuracy: `0.0`
- Compressed Bank-on accuracy: `0.0`
- Delta accuracy: `0.0`
- Output changes: `10`
- Improvements: `0`
- Regressions: `0`
- Retrieval-active contexts: `10`

## Failure Cases
No context-level leakage or Weaver-injection failure was observed in the completed run.

## Interpretation
Mechanism is active but not yet useful; inspect retrieval quality, memory content, and injection effects.

## Recommendation For Next Step
Do not run another threshold-only ablation yet. The next mechanism experiment should be a decoupled retrieve/update threshold design, after this preservation commit.

## Git Status
### Before
```
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
```
## rlm-memory-bank...origin/rlm-memory-bank [ahead 8]
 M memgen/model/modeling_memgen.py
?? research_notes/benchmarks/
?? scripts/eval/diagnose_memgen_over_context.py
?? scripts/eval/mab2_bank_off.py
?? scripts/eval/mab2_mab_bridge.py
?? scripts/eval/mab3_bank_on_full_history.py
?? scripts/eval/mab3a_threshold_ablation.py
?? scripts/eval/mab4a_compressed_memory.py
?? scripts/eval/mab5a_detectiveqa_compressed_n10.py
?? scripts/eval/mab_paired_bank_off_vs_low_threshold_bank_on.py
?? tests/test_mab2_bank_off.py
?? tests/test_mab3_bank_on_full_history.py
?? tests/test_mab3a_threshold_ablation.py
?? tests/test_mab4a_compressed_memory.py
?? tests/test_mab5a_detectiveqa_compressed_n10.py
?? tests/test_mab_paired_bank_off_vs_low_threshold_bank_on.py
```
