# MemoryAgentBench Local Task Availability Audit

## 1. Objective
Audit the local MemoryAgentBench parquet files and identify the next subtask for a multi-sample paired Bank-off vs low-threshold Bank-on evaluation without running inference.

## 2. Why the Audit Is Needed After the Failed n10 Attempt on `factconsolidation_sh_6k`
The previous paired artifact requested 10 contexts, but the local `Conflict_Resolution / factconsolidation_sh_6k` parquet contained only one matching row. That means the prior result is a one-context case study, not 10-context evidence. This audit checks whether any local subtask actually supports a valid multi-sample paired run under the current 32768-token full-history constraint.

## 3. Local Dataset Path
- `/mnt/18T/baishilong/datasets/MemoryAgentBench`

## 4. Files Inspected
- `Accurate_Retrieval`: `/mnt/18T/baishilong/datasets/MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet`
- `Conflict_Resolution`: `/mnt/18T/baishilong/datasets/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet`
- `Long_Range_Understanding`: `/mnt/18T/baishilong/datasets/MemoryAgentBench/data/Long_Range_Understanding-00000-of-00001.parquet`
- `Test_Time_Learning`: `/mnt/18T/baishilong/datasets/MemoryAgentBench/data/Test_Time_Learning-00000-of-00001.parquet`

## 5. Split-Level Row Counts
- `Accurate_Retrieval`: 22 rows
- `Conflict_Resolution`: 8 rows
- `Long_Range_Understanding`: 110 rows
- `Test_Time_Learning`: 6 rows

## 6. Sub-dataset / Source-Level Row Counts
### Accurate_Retrieval
- `eventqa_131072`: 5
- `eventqa_65536`: 5
- `eventqa_full`: 5
- `longmemeval_s*`: 5
- `ruler_qa1_197K`: 1
- `ruler_qa2_421K`: 1

### Conflict_Resolution
- `factconsolidation_mh_262k`: 1
- `factconsolidation_mh_32k`: 1
- `factconsolidation_mh_64k`: 1
- `factconsolidation_mh_6k`: 1
- `factconsolidation_sh_262k`: 1
- `factconsolidation_sh_32k`: 1
- `factconsolidation_sh_64k`: 1
- `factconsolidation_sh_6k`: 1

### Long_Range_Understanding
- `detective_qa`: 10
- `infbench_sum_eng_shots2`: 100

### Test_Time_Learning
- `icl_banking77_5900shot_balance`: 1
- `icl_clinic150_7050shot_balance`: 1
- `icl_nlu_8296shot_balance`: 1
- `icl_trec_coarse_6600shot_balance`: 1
- `icl_trec_fine_6400shot_balance`: 1
- `recsys_redial_full`: 1

## 7. Metric / Scoreability Assessment
- `eventqa_*` and `ruler_qa*`: `substring_exact_match`.
- `factconsolidation_*`: `substring_exact_match`.
- `detective_qa`: `exact_match`.
- `ICL_*`: `exact_match`.
- `recsys_redial_full`: `Recall@5`, plus extra `entity2id.json` dependency.
- `longmemeval_*` and `infbench_sum_*`: LLM-as-judge, excluded from the next paired run.

## 8. Token / Chunk Feasibility Summary
- `Accurate_Retrieval / eventqa_65536`: 5 rows, median full-history query estimate `66736`, 0 rows under 32768.
- `Accurate_Retrieval / eventqa_131072`: 5 rows, median full-history query estimate `133278`, 0 rows under 32768.
- `Accurate_Retrieval / eventqa_full`: 5 rows, median full-history query estimate `542453`, 0 rows under 32768.
- `Accurate_Retrieval / ruler_qa1_197K`: 1 row, estimated `205110`, 0 rows under 32768.
- `Accurate_Retrieval / ruler_qa2_421K`: 1 row, estimated `436303`, 0 rows under 32768.
- `Conflict_Resolution / factconsolidation_mh_6k`: 1 row, estimated `6753`, 1 row under 32768.
- `Conflict_Resolution / factconsolidation_sh_6k`: 1 row, estimated `6745`, 1 row under 32768.
- `Conflict_Resolution / factconsolidation_*_{32k,64k,262k}`: each 1 row, all over 32768.
- `Long_Range_Understanding / detective_qa`: 10 rows, median full-history query estimate `120191.5`, 0 rows under 32768.
- `Long_Range_Understanding / infbench_sum_eng_shots2`: 100 rows, LLM-judge and over capacity.
- `Test_Time_Learning / ICL_*`: each 1 row, all over 32768.
- `Test_Time_Learning / recsys_redial_full`: 1 row, estimated `1501054`, extra resource required.

## 9. Recommended Next Subtask for 10-Context Paired Evaluation
- No local subtask is a valid 10-context paired-run target under the current setup.
- There is no candidate that simultaneously satisfies: `>=10` contexts, automatic metric, no extra resource dependency, and estimated full-history query length under `32768` tokens.
- Strict n=10 recommendation: `NO-GO` under the current local dataset and current full-history 32k-capacity constraint.

## 10. Backup Subtask
- Best smaller feasible set: `Conflict_Resolution / factconsolidation_mh_6k` with `1` available context and automatic metric `substring_exact_match`.
- This is not a multi-sample evaluation. It is only suitable as an additional one-context mechanism case study.
- Backup smaller feasible set: `Conflict_Resolution / factconsolidation_sh_6k`.
- Best count-sufficient but over-capacity candidate: `Long_Range_Understanding / detective_qa` with `10` rows, but median full-history estimate `120191.5` exceeds `32768`.

## 11. Risks and Caveats
- Full-history token lengths are offline estimates using the official template mapping and an official chunker-equivalent implementation. They are sufficient for screening, but the paired harness must still do runtime preflight before any actual run.
- The local dataset distribution is highly sparse at the per-source level: many subtasks are present as exactly one row.
- The only row-count-rich automatic-metric candidate is `detective_qa`, but it is far beyond the current context capacity in full-history mode.
- `eventqa_65536` is the nearest retrieval-oriented candidate, but even its shortest local rows still exceed 32768 under full-history rebuild.
- The over-context diagnostic in [memgen_over_context_behavior.md](/mnt/18T/baishilong/MemGen/research_notes/benchmarks/memgen_over_context_behavior.md) shows that original MemGen can continue past nominal capacity unless the harness blocks it.

## 12. Exact Proposed Command / Config for the Next Paired Run
- Strict multi-sample `n=10` command: none. The correct decision is to stop rather than run an invalid or silently truncated experiment.
- If you still want the next controlled paired case study under the current setup, use the smaller feasible set below:
  - Official config: `/mnt/18T/baishilong/benchmarks/MemoryAgentBench/configs/data_conf/Conflict_Resolution/Factconsolidation_mh_6k.yaml`
  - Local parquet: `/mnt/18T/baishilong/datasets/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet`
```bash
python scripts/eval/mab_paired_bank_off_vs_low_threshold_bank_on.py \
  --dataset-root /mnt/18T/baishilong/datasets/MemoryAgentBench \
  --mab-repo /mnt/18T/baishilong/benchmarks/MemoryAgentBench \
  --mab-python /home/baishilong/miniconda3/envs/MABench/bin/python \
  --checkpoint-path <memgen-checkpoint> \
  --model-checkpoint-id <checkpoint-id> \
  --requested-contexts 1 \
  --split Conflict_Resolution \
  --sub-dataset factconsolidation_mh_6k \
  --data-config /mnt/18T/baishilong/benchmarks/MemoryAgentBench/configs/data_conf/Conflict_Resolution/Factconsolidation_mh_6k.yaml \
  --parquet /mnt/18T/baishilong/datasets/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet \
  --threshold 0.03
```
- This command is a target interface only. The current paired runner is still hardcoded to `factconsolidation_sh_6k`, so running this exact command would require parameterizing that runner first.

## 13. Git Status Before and After
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
