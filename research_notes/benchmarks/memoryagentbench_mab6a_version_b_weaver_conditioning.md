# MAB-6A: detective_qa Version B Weaver-conditioned Memory n10

## Purpose
Exploratory diagnostic of Version B routing: retrieved reasoner-space memory conditions Weaver, and only the fused latent is injected into Reasoner.

## Settings
- Comparison baseline: `outputs/mab/decoupled_thresholds_detectiveqa_n10/20260622T140741Z-detectiveqa-decoupled-thresholds-n10`
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

## Guardrails
- MAB-6A is exploratory.
- Weaver was not trained for this input distribution.
- Version A remains the default.
- MAB-6A differs from MAB-5C primarily by routing retrieved memory into Weaver.
- Do not claim performance improvement unless official exact_match improves.
- If exact_match remains 0 but outputs change, call it mechanism-active but not a performance win.
- If outputs degrade, this supports keeping Version A as default.

## Run Status
- Output directory: `outputs/mab/version_b_weaver_conditioned_detectiveqa_n10/20260625T023822Z-detectiveqa-version-b-weaver-conditioned-n10`
- Bank-off exact match: `0.0`
- Bank-on exact match: `0.0`
- Output changed: `10`
- Final slot counts: `[8, 8, 8, 8, 8, 8, 8, 8, 8, 8]`
- Retrieved memory to Weaver: `True`
- Retrieved latents entered Weaver: `True`
- Raw retrieved latents entered Reasoner: `False`
- Weaver conditioned on retrieved memory: `True`
- Weaver conditioning token count: `80`
- Fused latent generated: `True`
- Query write count: `0`
- Query write attempt count: `0`
- Cross-context leakage detected: `False`
- Write action counts: `{'insert': 80, 'replace_matched': 35, 'evict_oldest_insert': 211}`
- Update reason counts: `{'empty_bank': 10, 'matched_thread': 35, 'new_thread': 70, 'new_thread_bank_full': 211}`

## Comparison
- Against MAB-5C canonical: `{'baseline_artifact': 'outputs/mab/decoupled_thresholds_detectiveqa_n10/20260622T140741Z-detectiveqa-decoupled-thresholds-n10', 'exact_match_delta': 0.0, 'mechanism_change': 'retrieved memory routed into Weaver instead of direct Reasoner injection'}`

## Per-context Result Table
| context_index | exact_match_off | exact_match_on | output_changed | raw_retrieved_latents_enter_reasoner | weaver_conditioned_on_retrieved_memory |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | True | False | True |
| 1 | 0 | 0 | True | False | True |
| 2 | 0 | 0 | True | False | True |
| 3 | 0 | 0 | True | False | True |
| 4 | 0 | 0 | True | False | True |
| 5 | 0 | 0 | True | False | True |
| 6 | 0 | 0 | True | False | True |
| 7 | 0 | 0 | True | False | True |
| 8 | 0 | 0 | True | False | True |
| 9 | 0 | 0 | True | False | True |

## Git Status
### Before
```
## rlm-memory-bank...origin/rlm-memory-bank [ahead 4]
 M memgen/model/configuration_memgen.py
 M memgen/model/modeling_memgen.py
 M scripts/eval/mab5a_detectiveqa_compressed_n10.py
?? research_notes/benchmarks/memoryagentbench_mab6a_version_b_weaver_conditioning.md
?? scripts/eval/mab6a_version_b_weaver_conditioned_detectiveqa_n10.py
?? tests/test_mab6a_version_b_weaver_conditioning.py
```
### After
```
## rlm-memory-bank...origin/rlm-memory-bank [ahead 4]
 M memgen/model/configuration_memgen.py
 M memgen/model/modeling_memgen.py
 M scripts/eval/mab5a_detectiveqa_compressed_n10.py
?? research_notes/benchmarks/memoryagentbench_mab6a_version_b_weaver_conditioning.md
?? scripts/eval/mab6a_version_b_weaver_conditioned_detectiveqa_n10.py
?? tests/test_mab6a_version_b_weaver_conditioning.py
```
