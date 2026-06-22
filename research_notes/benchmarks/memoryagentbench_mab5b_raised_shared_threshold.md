# MAB-5B: detective_qa Raised Shared-threshold Bank-off vs Bank-on n10

## Purpose

Test whether increasing the existing shared threshold from `0.03` to `0.05`
reduces over-merge / `replace_matched` behavior before any decoupled-threshold
mechanism is introduced.

## Settings

- Split: `Long_Range_Understanding`
- Subtask: `detective_qa`
- Contexts: 10 local rows
- Query mode: first-query-only
- Bank-off / Bank-on compressed-memory protocol
- Threshold: `0.05`
- `top_k=1`
- `max_slots=8`
- `retrieve_policy=threshold_topk`
- `update_policy=thread_update`
- Query turn: read-only
- Full-history original MemGen: `over_capacity_invalid`

## Command

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> /home/baishilong/miniconda3/envs/memgen/bin/python \
  scripts/eval/mab5b_raised_shared_threshold_detectiveqa_n10.py
```

## Validation

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python -m py_compile \
  scripts/eval/mab5b_raised_shared_threshold_detectiveqa_n10.py
/home/baishilong/miniconda3/envs/memgen/bin/python -m unittest \
  tests.test_mab5b_raised_shared_threshold_detectiveqa_n10
git diff --check -- scripts/eval tests research_notes
```

## Run Status

MAB-5B completed successfully on `CUDA_VISIBLE_DEVICES=2`.

- Output directory:
  `outputs/mab/raised_shared_threshold_detectiveqa_n10/20260622T073545Z-detectiveqa-raised-shared-threshold-n10/`
- Run ID: `20260622T073545Z-detectiveqa-raised-shared-threshold-n10`
- Requested / valid contexts: `10 / 10`
- Bank-off exact match: `0.0`
- Bank-on exact match: `0.0`
- Accuracy delta: `0.0`
- Output changed: `5`
- Retrieval-active contexts: `10`
- Final slot counts: `[8, 8, 8, 8, 8, 8, 8, 8, 8, 8]`
- Mean final slot count: `8.0`
- Write count: `326`
- Retrieval count: `316`
- Retrieved latent count: `200`
- Query write count: `0`
- Cross-context leakage detected: `false`
- Retrieved latents entered Reasoner: `true`
- Retrieved latents entered Weaver: `false`
- Retrieved score range: approximately `0.050-0.064`
- `matched_replace_count` / `thread_insert_count` / `capacity_evict_count`: not
  exposed in the current run artifact
- Full-history detective_qa remains `over_capacity_invalid` and was not run
- The direct repo-root command now works without a manual `PYTHONPATH`
  override.

## Comparison Against MAB-5A

MAB-5A remains the reference baseline:

- run ID: `20260621T013454Z-detectiveqa-compressed-n10`
- Bank-off exact match: `0.0`
- Bank-on exact match: `0.0`
- `output_changed=10`
- retrieval active in all 10 contexts
- final slot counts: `[1, 2, 2, 5, 6, 5, 6, 7, 4, 7]`
- mean final slot count: `4.5`
- retrieved latent count: `2248`

MAB-5B changed the slot story substantially:

- final slot counts increased from `[1, 2, 2, 5, 6, 5, 6, 7, 4, 7]` to
  `[8, 8, 8, 8, 8, 8, 8, 8, 8, 8]`
- mean final slot count increased from `4.5` to `8.0`
- retrieval remained active in all 10 contexts
- retrieved latent count dropped from `2248` to `200`
- output changes dropped from `10` to `5`

## Interpretation

Raised shared threshold is now a strong simple baseline:

- slot counts increased to the maximum `8` in every context;
- retrieval stayed active in every context;
- exact match did not improve; and
- the run stayed Reasoner-only with no cross-context leakage.

This means MAB-5C is still a reasonable follow-up if the goal is to separate
read/write thresholds more precisely, but the current diagnostic does not make
MAB-5C mandatory. The evidence already supports the simple raised-threshold
baseline as the cleaner next comparator.
