# MAB-6B-FR: detective_qa final-query format repair

## Purpose
Format-only diagnostic for the MAB-6B Weaver-space bank: apply an answer-only prefix to the final query and check whether cleaner surface form improves exact match without changing the underlying storage or retrieval mechanism.

## Settings
- Primary comparison baseline: `outputs/mab/version_b_weaver_conditioned_detectiveqa_n10/20260625T023822Z-detectiveqa-version-b-weaver-conditioned-n10`
- Secondary comparison baseline: `outputs/mab/decoupled_thresholds_detectiveqa_n10/20260622T140741Z-detectiveqa-decoupled-thresholds-n10`
- threshold: `0.03`
- retrieve_threshold: `0.03`
- update_threshold: `0.05`
- top_k: `1`
- max_slots: `8`
- retrieve_policy: `threshold_topk`
- update_policy: `thread_update`
- query mode: first-query-only
- query phase: read-only
- full-history detective_qa: `over_capacity_invalid`
- retrieved_memory_to_weaver: `True`
- memory_bank_storage_space: `weaver`

## Guardrails
- This is a separate diagnostic from the canonical MAB-6B result.
- It must not overwrite `research_notes/benchmarks/memoryagentbench_mab6b_weaver_space_bank.md`.
- Format repair should be interpreted as a surface-form intervention only.

## Run Status
- Output directory: `outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_format_repair/20260626T014628Z-detectiveqa-version-b-weaver-space-bank-format-repair-n10`
- Bank-off exact match: `0.0`
- Bank-on exact match: `0.0`
- Output changed: `10`
- Final slot counts: `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`
- Memory bank storage space: `weaver`
- Stored latent space: `weaver`
- Retrieval query space: `weaver`
- Retrieved memory space: `weaver`
- Stored Weaver latents in bank: `True`
- Retrieved Weaver latents from bank: `True`
- Retrieved memory projected to Weaver: `False`
- Retrieved latents entered Weaver: `True`
- Raw retrieved latents entered Reasoner: `False`
- Fused latent generated: `True`
- Query write count: `0`
- Query write attempt count: `0`
- Cross-context leakage detected: `False`

## Interpretation
- Format repair improved surface control but did not improve exact match.
- The diagnostic supports the conclusion that the bottleneck is not only output formatting.

## Comparison
- Against MAB-6A canonical: `{'baseline_artifact': 'outputs/mab/version_b_weaver_conditioned_detectiveqa_n10/20260625T023822Z-detectiveqa-version-b-weaver-conditioned-n10', 'baseline_summary_available': True, 'bank_on_exact_match_delta': 0.0, 'output_changed_delta': 0, 'retrieved_memory_projection_change': 'reasoner_to_weaver projection removed for retrieved memory'}`
- Against MAB-5C canonical: `{'baseline_artifact': 'outputs/mab/decoupled_thresholds_detectiveqa_n10/20260622T140741Z-detectiveqa-decoupled-thresholds-n10', 'mechanism_change': 'bank stores Weaver-space memory and queries in Weaver space instead of storing reasoner-space memory and re-projecting retrieved memory'}`

## Per-context Result Table
| context_index | exact_match_off | exact_match_on | output_changed | retrieval_query_space | retrieved_memory_projected_to_weaver | raw_retrieved_latents_enter_reasoner |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | True | weaver | False | False |
| 1 | 0 | 0 | True | weaver | False | False |
| 2 | 0 | 0 | True | weaver | False | False |
| 3 | 0 | 0 | True | weaver | False | False |
| 4 | 0 | 0 | True | weaver | False | False |
| 5 | 0 | 0 | True | weaver | False | False |
| 6 | 0 | 0 | True | weaver | False | False |
| 7 | 0 | 0 | True | weaver | False | False |
| 8 | 0 | 0 | True | weaver | False | False |
| 9 | 0 | 0 | True | weaver | False | False |
