#!/usr/bin/env bash
# Resume the interrupted no-query-retrieval repeat campaign without rerunning
# its completed seed 42/142/242 passes or seed342/context0 artifact.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <physical-gpu-index> <run-id>" >&2
  exit 2
fi

gpu_index="$1"
run_id="$2"
python_bin="/home/baishilong/miniconda3/envs/memgen/bin/python"
root="outputs/mab/ablation_effect_repeats/${run_id}/no_query_retrieval"
log_root="runtime_logs/ablation_effect_repeats/${run_id}/no_query_retrieval"
runner="eval/exp/no_query_retrieval/eventqa_p7_no_query_retrieval.py"

run_context() {
  local seed="$1"
  local context_index="$2"
  local context_root="$root/seed${seed}/ctx${context_index}"
  CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" "$runner" \
    --measurement-scope full --context-index "$context_index" --question-limit 100 \
    --seed "$seed" --reseed-per-context --skip-research-note \
    --output-root "$context_root" \
    > "$log_root/resume_seed${seed}_ctx${context_index}.log" 2>&1
}

aggregate_seed() {
  local seed="$1"
  local artifacts=()
  local context_index artifact
  for context_index in 0 1 2 3 4; do
    artifact="$(find "$root/seed${seed}/ctx${context_index}" -type f -name smoke_artifact.json -print | sort | tail -1)"
    [[ -n "$artifact" ]] || { echo "missing completed artifact for seed=$seed context=$context_index" >&2; exit 1; }
    artifacts+=("$artifact")
  done
  "$python_bin" eval/exp/ablations/aggregate_repeats.py seed \
    --variant no_query_retrieval --artifacts "${artifacts[@]}" \
    --output-json "$root/seed${seed}/aggregate.json"
}

for context_index in 1 2 3 4; do run_context 342 "$context_index"; done
aggregate_seed 342
for context_index in 0 1 2 3 4; do run_context 442 "$context_index"; done
aggregate_seed 442

"$python_bin" eval/exp/ablations/aggregate_repeats.py repeats \
  --seed-summaries "$root/seed42/aggregate.json" "$root/seed142/aggregate.json" \
  "$root/seed242/aggregate.json" "$root/seed342/aggregate.json" \
  "$root/seed442/aggregate.json" \
  --output-json "$root/repeat_aggregate.json"
printf '%s\n' "NO_QUERY_REPEAT_RESUME_ROOT=$root"
