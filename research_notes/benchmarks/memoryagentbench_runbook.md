# MemoryAgentBench / MAB Runbook

## 1. Purpose

This runbook explains how to run the existing MemoryAgentBench evaluations for MemGen and the LatentMemoryBank harnesses in this repository. It is a practical operator guide, not a design note.

## 2. Repository and Environment Paths

- MemGen repo: `/mnt/18T/baishilong/MemGen`
- MemoryAgentBench repo: `/mnt/18T/baishilong/benchmarks/MemoryAgentBench`
- MemoryAgentBench dataset root: `/mnt/18T/baishilong/datasets/MemoryAgentBench`
- MemGen checkpoint: `/home/baishilong/.cache/huggingface/hub/models--Kana-s--MemGen/snapshots/269d9b1741130b94fffa410cdaa3d4bc74081a7f/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model`
- MemGen Python environment: `conda activate memgen` or `/home/baishilong/miniconda3/envs/memgen/bin/python`
- MABench Python environment: `/home/baishilong/miniconda3/envs/MABench/bin/python`
- Current branch: `rlm-memory-bank`
- Reasoner context capacity: `32768`

## 3. Before Running Anything

Run these from the MemGen repo root:

```bash
cd /mnt/18T/baishilong/MemGen
git status --short --branch
nvidia-smi
```

Checklist:

- Check the current branch before you start.
- Check which GPU is actually free.
- Use the `memgen` environment for MemGen runner scripts.
- Do not run MemGen eval scripts under the `MABench` environment; reserve `MABench` Python for the original MemoryAgentBench helper utilities when a script explicitly calls into the MAB bridge.
- Do not accidentally stage `memgen/model/modeling_memgen.py` if it is still an unrelated local edit.

## 4. How Local MAB Data Is Organized

Inspect the local parquet files with:

```bash
find /mnt/18T/baishilong/datasets/MemoryAgentBench -maxdepth 2 -name '*.parquet' | sort
rg --files /mnt/18T/baishilong/datasets/MemoryAgentBench
```

Local availability findings from the repository notes:

- `Accurate_Retrieval` has only 22 rows total and is not a good n=10 target.
- `Conflict_Resolution` factconsolidation sources usually have only 1 matched context each.
- `Long_Range_Understanding / detective_qa` has 10 rows, but full-history query length is far above 32768 tokens.
- `Test_Time_Learning` local sources are limited and generally over capacity.
- `LongMemEval` and `InfBench_sum` require an LLM judge and are not current targets for the MemGen harnesses here.

Useful evidence file:

- [memoryagentbench_local_task_availability.md](file:///mnt/18T/baishilong/MemGen/research_notes/benchmarks/memoryagentbench_local_task_availability.md)

## 5. Existing Experiment Scripts and What Each One Does

| Script | Purpose | Safe to rerun? | Typical runtime | Output root pattern | Key parameters | Expected outputs |
|---|---|---|---|---|---|---|
| `scripts/eval/mab2_bank_off.py` | Original MemGen full-history Bank-off one-context factconsolidation run | Yes; each rerun creates a new timestamped output directory | A few minutes end to end; generation itself is only a few seconds | `outputs/mab/memgen_bank_off/<timestamp>-factconsolidation-sh-6k-onectx/` | `--dataset-root`, `--mab-repo`, `--mab-python`, `--checkpoint-path`, `--model-checkpoint-id`; defaults hardcode the factconsolidation_sh_6k path | `manifest.json`, `results.json`, `diagnostics.jsonl`, `run_config.json` |
| `scripts/eval/mab3_bank_on_full_history.py` | Bank-on full-history one-context factconsolidation run | Yes; each rerun creates a new timestamped output directory | A few minutes end to end | `outputs/mab/memgen_bank_on_full_history/<timestamp>-factconsolidation-sh-6k-onectx/` | Same base paths as MAB-2, plus paired-artifact defaults inside the script | `manifest.json`, `results.json`, `diagnostics.jsonl`, `run_config.json` |
| `scripts/eval/mab3a_threshold_ablation.py` | Threshold ablation on one-context factconsolidation | Yes; each rerun creates a new timestamped output directory | A few minutes end to end because it runs multiple threshold cases | `outputs/mab/memgen_bank_on_threshold_ablation/<timestamp>-factconsolidation-sh-6k-onectx/` | Same base paths as MAB-2, plus paired-artifact inputs | `manifest.json`, `threshold_results.json`, `diagnostics.jsonl`, `run_config.json` |
| `scripts/eval/mab4a_compressed_memory.py` | Compressed-memory one-context exploratory run | Yes; each rerun creates a new timestamped output directory | A few minutes end to end | `outputs/mab/memgen_bank_on_compressed_memory/<timestamp>-factconsolidation-sh-6k-onectx/` | Same base paths as MAB-2, plus paired-artifact inputs | `manifest.json`, `results.json`, `diagnostics.jsonl`, `run_config.json` |
| `scripts/eval/mab_paired_bank_off_vs_low_threshold_bank_on.py` | Paired Bank-off vs low-threshold Bank-on n10 attempt | Safe to rerun only if you understand it will create a new timestamped output directory and use whatever local source rows exist; the earlier target was limited by local data availability | A few minutes for the single available row; longer if data availability changes | `outputs/mab/paired_bank_off_vs_low_threshold_bank_on/<timestamp>-factconsolidation-sh-6k-n10/` | `--requested-contexts`, `--threshold`, base paths | `manifest.json`, `paired_results.json`, `diagnostics.jsonl`, `run_config.json` |
| `scripts/eval/diagnose_memgen_over_context.py` | Over-context diagnostic | Yes; each rerun creates a new timestamped output directory | Short, usually minutes or less | `outputs/mab/memgen_over_context_behavior/<timestamp>-over-context/` | `--test-lengths`, `--max-new-tokens`, checkpoint and model paths | `over_context_diagnostic.json` plus manifest-style output |
| `scripts/eval/mab5a_detectiveqa_compressed_n10.py` | Main detective_qa compressed-memory n10 run | Yes; each rerun creates a new timestamped output directory | About 9 minutes in the latest run on one GPU | `outputs/mab/compressed_memory_detectiveqa_n10/<timestamp>-detectiveqa-compressed-n10/` | Defaults already encode the dataset root, checkpoint, `threshold=0.03`, `top_k=1`, `max_slots=8`, `requested-contexts=10`, `query_mode=first-query-only` | `manifest.json`, `paired_results.json`, `diagnostics.jsonl`, `run_config.json` |

## 6. Main Recommended Run: MAB-5A detective_qa Compressed n10

This is the main run for the current local MAB work. The script has defaults for the repo paths, checkpoint, threshold, and requested context count, so the working command is just the script itself from the MemGen repo root.

Recommended command:

```bash
cd /mnt/18T/baishilong/MemGen
CUDA_VISIBLE_DEVICES=<GPU_ID> /home/baishilong/miniconda3/envs/memgen/bin/python scripts/eval/mab5a_detectiveqa_compressed_n10.py
```

Use a concrete GPU ID only as needed; `3` was the latest example, not a requirement.

What this does:

- loads `Long_Range_Understanding / detective_qa`
- selects all 10 local rows
- runs compressed Bank-off
- runs compressed Bank-on
- uses first query only
- keeps the query turn read-only with respect to bank writes
- uses `threshold=0.03`, `top_k=1`, `max_slots=8`, `retrieve_policy=threshold_topk`
- marks original full-history as `over_capacity_invalid` instead of executing it

Expected output directory:

`outputs/mab/compressed_memory_detectiveqa_n10/<timestamp>-detectiveqa-compressed-n10/`

Expected files:

- `manifest.json`
- `paired_results.json`
- `diagnostics.jsonl`
- `run_config.json`

Known latest result from `20260621T013454Z-detectiveqa-compressed-n10`:

- valid contexts: `10`
- Bank-off exact match: `0.0`
- Bank-on exact match: `0.0`
- output changed: `10`
- retrieval active: `10/10`
- final slot counts: `[1, 2, 2, 5, 6, 5, 6, 7, 4, 7]`
- retrieved score range: roughly `0.030-0.064`
- full-history status: `over_capacity_invalid`

The runbook should treat this as the current evidence baseline, not as a reason to rerun full-history detective_qa.

## 7. How to Run Tests

Run the targeted syntax and unit checks from the MemGen repo root:

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python -m py_compile scripts/eval/mab5a_detectiveqa_compressed_n10.py
/home/baishilong/miniconda3/envs/memgen/bin/python -m unittest tests.test_mab5a_detectiveqa_compressed_n10
/home/baishilong/miniconda3/envs/memgen/bin/python -m unittest \
  tests.test_mab5a_detectiveqa_compressed_n10 \
  tests.test_mab4a_compressed_memory \
  tests.test_mab_paired_bank_off_vs_low_threshold_bank_on \
  tests.test_mab3_bank_on_full_history \
  tests.test_mab2_bank_off
git diff --check
```

If you want the exact broader validation set used when the MAB scripts were added, include:

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python -m unittest \
  tests.test_mab2_bank_off \
  tests.test_mab3_bank_on_full_history \
  tests.test_mab3a_threshold_ablation \
  tests.test_mab4a_compressed_memory \
  tests.test_mab5a_detectiveqa_compressed_n10 \
  tests.test_mab_paired_bank_off_vs_low_threshold_bank_on
```

## 8. How to Read the Output

### `manifest.json`

Important fields:

- `compressed_bank_off_accuracy`
- `compressed_bank_on_accuracy`
- `delta_accuracy`
- `num_output_changed`
- `num_retrieval_active`
- `avg_estimated_full_history_query_tokens`
- `avg_compressed_query_tokens`
- `avg_retrieved_latents`
- `full_history_policy`
- `query_mode`
- `context_capacity`
- `git_status_before`
- `git_status_after`

### `paired_results.json`

Useful per-context fields:

- `context_index`
- `context_id`
- `query_id`
- `chunk_count`
- `chunk_token_lengths`
- `estimated_full_history_query_tokens`
- `context_capacity`
- `full_history_status`
- `compressed_query_tokens_bank_off`
- `compressed_query_tokens_bank_on`
- `query_prompt_contains_chunk_text`
- `query_prompt_contains_ack_history`
- `bank_off_prediction`
- `bank_off_exact_match`
- `bank_on_prediction`
- `bank_on_exact_match`
- `gold_answer`
- `output_changed`
- `improved`
- `regressed`
- `bank_on_write_count`
- `bank_on_retrieval_count`
- `bank_on_retrieved_latent_count`
- `retrieved_indices_by_turn`
- `retrieved_scores_by_turn`
- `retrieved_latents_enter_reasoner`
- `retrieved_latents_enter_weaver`
- `query_write_count`
- `query_write_attempt_count`
- `bank_slot_count_final_before_reset`
- `bank_reset_after_context`
- `cross_context_leakage_detected`

### `diagnostics.jsonl`

This is the turn-level or case-level trace file. Use it when you need to understand why a row succeeded, failed, or changed output. It is the fastest way to debug prompt leakage, retrieval routing, and write-back behavior.

### How to Interpret Common Cases

- Empty retrieval turn `[]`: no memory was retrieved for that turn.
- Retrieved score range around `0.030-0.064`: the low-threshold regime is active but still not producing exact-match gains in MAB-5A.
- Low final slot count with high write count: repeated replacement / over-merge behavior is likely.
- `gold_answer` present but exact match is `0`: the model produced an answer, but the exact string did not match the official scoring rule.
- `retrieved_latents_enter_reasoner=true` and `retrieved_latents_enter_weaver=false`: the intended Reasoner-only boundary is working.
- `query_write_count=0`: query phase stayed read-only with respect to bank writes.
- `cross_context_leakage_detected=false`: session reset worked.
- `full_history_status=over_capacity_invalid`: do not treat full-history as a valid baseline for that context.
- `query_prompt_contains_chunk_text=false`: compressed query prompt did not leak prior chunk text.

## 9. What Not to Run

- Do not run original full-history MemGen on `detective_qa` contexts. The estimated full-history query length is above `32768`.
- Do not silently truncate over-capacity prompts.
- Do not compare over-capacity full-history outputs against compressed-memory outputs as if both were valid baselines.
- Do not mix official `exact_match` with relaxed diagnostic metrics in the same conclusion.
- Do not treat `output_changed` as improvement.
- Do not run another threshold-only sweep as the current next experiment;
  MAB-5C is a mechanism change with separate retrieval and update thresholds.

## 10. How to Run the Over-context Diagnostic

Use the diagnostic runner to confirm the capacity boundary and synthetic probe behavior:

```bash
cd /mnt/18T/baishilong/MemGen
CUDA_VISIBLE_DEVICES=<GPU_ID> /home/baishilong/miniconda3/envs/memgen/bin/python scripts/eval/diagnose_memgen_over_context.py
```

Default behavior from the script:

- synthetic probe lengths: `32000`, `32760`, `32800`, `35000`
- `detective_qa` first-context preflight should report estimated full-history query tokens above `32768`
- no generation should be called for clearly over-capacity real MAB samples

The output directory is:

`outputs/mab/memgen_over_context_behavior/<timestamp>-over-context/`

## 11. Future Mechanism Work

This runbook does not define mechanism changes. The canonical current action is
in `memoryagentbench_next_steps.md`, and the implementation/experiment contract
is in `memoryagentbench_mechanism_plan.md`.

Current routing: implement MAB-5C Phase 1 only. Do not implement fallback or
retrieved-memory-to-Weaver conditioning in that phase.

## 12. Reproducibility Checklist

Before a run:

- branch: `rlm-memory-bank`
- `git status --short --branch`
- checkpoint path
- dataset root
- exact script command
- GPU choice from `nvidia-smi`
- `memgen` environment for MemGen scripts
- `MABench` Python only for the MAB bridge/helper utilities
- `git diff --check` is useful, but unrelated dirty files can make a global check noisy; use targeted checks when you only want to validate the files you changed.

After a run:

- output directory path
- validation tests run
- `git status --short --branch`
- whether `memgen/model/modeling_memgen.py` stayed untouched

## 13. Minimal Commands Cheat Sheet

Status check:

```bash
cd /mnt/18T/baishilong/MemGen
git status --short --branch
```

Run MAB-5A:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> /home/baishilong/miniconda3/envs/memgen/bin/python scripts/eval/mab5a_detectiveqa_compressed_n10.py
```

For example, use `GPU_ID=3` only when GPU 3 is the selected free device.

Run tests:

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python -m py_compile scripts/eval/mab5a_detectiveqa_compressed_n10.py
/home/baishilong/miniconda3/envs/memgen/bin/python -m unittest tests.test_mab5a_detectiveqa_compressed_n10
/home/baishilong/miniconda3/envs/memgen/bin/python -m unittest \
  tests.test_mab5a_detectiveqa_compressed_n10 \
  tests.test_mab4a_compressed_memory \
  tests.test_mab_paired_bank_off_vs_low_threshold_bank_on \
  tests.test_mab3_bank_on_full_history \
  tests.test_mab2_bank_off
```

Inspect output summary:

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python -m json.tool outputs/mab/compressed_memory_detectiveqa_n10/20260621T013454Z-detectiveqa-compressed-n10/manifest.json | sed -n '1,200p'
/home/baishilong/miniconda3/envs/memgen/bin/python -m json.tool outputs/mab/compressed_memory_detectiveqa_n10/20260621T013454Z-detectiveqa-compressed-n10/paired_results.json | sed -n '1,200p'
```

Check git status:

```bash
git status --short --branch
```
