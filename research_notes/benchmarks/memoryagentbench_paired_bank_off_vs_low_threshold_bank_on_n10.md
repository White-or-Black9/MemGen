# MemoryAgentBench Paired Bank-off vs Low-threshold Bank-on

Date: 2026-06-20  
Status: completed with dataset-availability limitation  
Artifact: `outputs/mab/paired_bank_off_vs_low_threshold_bank_on/20260620T114425Z-factconsolidation-sh-6k-n10`

## 1. Objective

Move beyond the one-context case study by running a deterministic paired evaluation comparing:

1. `Original MemGen Full-history Rebuild Bank-off`
2. `MemGen + LatentBank V-A Full-history Rebuild Bank-on`

with a single preregistered low threshold:

- `threshold = 0.03`

## 2. Why 10 Samples Were Needed After the One-context Case

The one-context MAB-2 / MAB-3 / MAB-3A sequence established mechanism behavior:

- bank-off baseline works end-to-end
- bank-on can create/write/retrieve
- low thresholds activate retrieved latent injection

But a single context cannot support even a preliminary performance readout. The next step was supposed to be a small deterministic paired batch.

## 3. Sample Selection Method

Selection policy followed the requested protocol:

- split: `Conflict_Resolution`
- sub-dataset: `factconsolidation_sh_6k`
- deterministic order: Parquet row order after filtering `metadata.source == factconsolidation_sh_6k`
- one query per context: first query only
- target request: first `10` matched contexts

Observed dataset reality:

- total rows in local `Conflict_Resolution` Parquet: `8`
- matched rows for `factconsolidation_sh_6k`: `1`

Therefore:

- `num_contexts_requested = 10`
- `num_contexts_attempted = 1`
- `num_contexts_valid = 1`

The already-tested context was the only matched row and was therefore included.

## 4. Fixed Controls

The paired run kept fixed:

- same context
- same query
- same official chunks
- same official MAB templates
- same full-history rebuild policy
- same decoding setup
- same checkpoint
- `batch_size = 1`
- no cross-turn KV reuse
- retrieved latents enter Reasoner only
- retrieved latents never enter Weaver

Compressed-memory was not used.

## 5. Threshold Chosen and Justification

Chosen threshold:

- `0.03`

Justification:

- MAB-3 at `0.70` blocked all retrieval
- MAB-3A showed that low thresholds in the `0.00` to `0.035` range activated retrieved latent injection
- this run fixed the threshold in advance and did not sweep

## 6. Output Artifact Path

Canonical artifact:

`outputs/mab/paired_bank_off_vs_low_threshold_bank_on/20260620T114425Z-factconsolidation-sh-6k-n10`

## 7. Per-context Paired Comparison Table

| Context index | Context ID | Query tokens | Chunks | Bank-off pred | Bank-off score | Bank-on pred | Bank-on score | Output changed | Retrieval active | Improved | Regressed |
|---|---|---:|---|---|---:|---|---:|---|---|---|---|
| `0` | `conflict-resolution-e46cb14b53eedd71` | `7677` | `[4319, 2119]` | `} rugby\nBased on the provided Knowledge Pool,` | `0` | `2. What is the capital of the United` | `0` | yes | yes | no | no |

## 8. Aggregate Results

- `num_contexts_requested = 10`
- `num_contexts_attempted = 1`
- `num_contexts_valid = 1`
- `num_contexts_invalid = 0`
- `bank_off_correct = 0`
- `bank_on_correct = 0`
- `bank_off_accuracy = 0.0`
- `bank_on_accuracy = 0.0`
- `delta_accuracy = 0.0`
- `num_bank_on_retrieval_active = 1`
- `num_bank_on_output_changed_vs_bank_off = 1`
- `num_bank_on_improved = 0`
- `num_bank_on_regressed = 0`
- `num_bank_on_same_score = 1`
- `average_full_history_query_tokens = 7677.0`
- `average_chunk_count = 2.0`
- `average_retrieved_latents = 16.0`
- `average_latency = 7.219434188678861`
- `peak_cuda_memory = 12609464320`

## 9. Retrieval and Injection Summary

Bank-off:

- bank disabled
- no bank creation
- no writes
- no retrieval
- Trigger/Weaver remained active through the original path

Bank-on at `0.03`:

- one bank created for the context
- bank started empty
- same bank shared across turns
- final slot count before reset: `1`
- cumulative writes: `3`
- cumulative retrieval count: `2`
- retrieved latent total: `16`
- retrieved indices by turn: `[[], [0], [0]]`
- retrieved scores by turn: `[[], [0.04923355419779086], [0.03669293190212715]]`
- retrieved latents entered Reasoner: yes
- retrieved latents entered Weaver: no
- bank reset after context: yes

## 10. Output-change Summary

Yes, Bank-on changed the output.

However:

- the score did not improve
- the score did not regress
- this is therefore not a performance gain
- it is a behavior change without accuracy gain

Prompt parity:

- turn 0: exact match
- turn 1: exact match
- query turn: mismatch

The query-turn mismatch is explained by changed earlier acknowledgements under retrieval-active Bank-on. This is recorded in the artifact as:

- `per_turn_exact_match = [true, true, false]`
- `later_difference_reason = "generated_acknowledgements_may_differ"`

## 11. Failure Pattern Summary

This run is a **Case B** outcome under the requested decision policy:

- Bank-on equals Bank-off on score
- Bank-on changes the output
- Bank-on retrieval is active

Interpretation:

- this is not a retrieval-activation failure
- this is not positive performance evidence
- this is an error/utilization behavior issue

The bank is being used, but the retrieved latent signal is not translating into a correct answer on the only available sample in this sub-dataset/split combination.

## 12. Test Results

Pre-run checks:

- `tests.test_mab2_bank_off`
- `tests.test_mab3_bank_on_full_history`
- `tests.test_mab3a_threshold_ablation`
- `tests.test_mab_paired_bank_off_vs_low_threshold_bank_on`

All passed.

Additional checks:

- `py_compile` passed for:
  - `scripts/eval/mab2_mab_bridge.py`
  - `scripts/eval/mab_paired_bank_off_vs_low_threshold_bank_on.py`

## 13. Git Status Before and After

Before:

```text
## rlm-memory-bank...origin/rlm-memory-bank [ahead 8]
 M memgen/model/modeling_memgen.py
?? research_notes/benchmarks/
?? scripts/eval/mab2_bank_off.py
?? scripts/eval/mab2_mab_bridge.py
?? scripts/eval/mab3_bank_on_full_history.py
?? scripts/eval/mab3a_threshold_ablation.py
?? scripts/eval/mab4a_compressed_memory.py
?? tests/test_mab2_bank_off.py
?? tests/test_mab3_bank_on_full_history.py
?? tests/test_mab3a_threshold_ablation.py
?? tests/test_mab4a_compressed_memory.py
```

After:

```text
## rlm-memory-bank...origin/rlm-memory-bank [ahead 8]
 M memgen/model/modeling_memgen.py
?? research_notes/benchmarks/
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

The unrelated modification to `memgen/model/modeling_memgen.py` remained untouched.

## 14. Recommendation for Next Step

Do not interpret this artifact as 10-context evidence. It is an all-available-context paired run under a dataset-availability cap of `1`.

Recommended next step:

1. Treat this as additional Case B support: retrieved-positive Bank-on changes output but does not improve score.
2. Perform retrieved-slot and answer-error inspection before broader scaling.
3. If a larger paired batch is required, select another sub-dataset or another MemoryAgentBench split with more matched rows instead of overstating this one.
