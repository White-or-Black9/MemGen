# MAB-5A: detective_qa Compressed-memory Bank-off vs Bank-on n10

## Objective
Test whether LatentBank helps on detective_qa when the full dialogue history is over capacity.

## Why Original Full-history Is Invalid
The over-context diagnostic showed the original full-history path exceeds the 32,768-token capacity for detective_qa, so the full-history baseline is marked `over_capacity_invalid` and was not executed.

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
- Original MemGen full-history: `over_capacity_invalid`.
- Compressed Bank-off: no LatentBank, compressed query only.
- Compressed Bank-on: LatentBank enabled, sequential chunk writes, read-only query proxy.

## Settings
- Threshold: `0.03`
- `top_k`: `1`
- `max_slots`: `8`
- `retrieve_policy`: `threshold_topk`
- `query_mode`: `first-query-only`

## Query Read-only Status
The query turn used a no-op proxy for bank writes, so `query_write_count` remained `0` while retrieval stayed active.

## Prompt Leakage Checks
The compressed query prompt was checked for chunk-text and acknowledgement-history leakage per context.

## Reasoner-only Injection Checks
Retrieved latents were checked to enter Reasoner and not Weaver.

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
The mechanism is active but produced no official exact-match gain. Retrieval was active in every context and all 10 outputs changed, so exact match of zero does not imply an inactive mechanism. `output_changed=10` shows generation was affected; it is not evidence of improvement. Official exact match must remain separate from relaxed or gold-substring diagnostics.

## Recommendation For Next Step
Do not run another threshold-only ablation. The next mechanism experiment is MAB-5C Decoupled Retrieval-Update Thresholds. Do not implement fallback or retrieved-memory-to-Weaver conditioning during MAB-5C.

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
## rlm-memory-bank...origin/rlm-memory-bank [ahead 1]
 M memgen/model/modeling_memgen.py
 M research_notes/PROGRESS.md
 M research_notes/benchmarks/memoryagentbench_next_steps.md
?? research_notes/benchmarks/memoryagentbench_mab5b_raised_shared_threshold.md
?? scripts/eval/mab5b_raised_shared_threshold_detectiveqa_n10.py
?? tests/test_mab5b_raised_shared_threshold_detectiveqa_n10.py
```
