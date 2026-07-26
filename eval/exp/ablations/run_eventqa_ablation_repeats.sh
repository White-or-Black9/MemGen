#!/usr/bin/env bash
# Five process-level EventQA repeats for one P7 query-time ablation.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <physical-gpu-index> <no_query_retrieval|no_retrieved_memory_conditioning|direct_top1> <run-id>" >&2
  exit 2
fi

gpu_index="$1"
variant="$2"
run_id="$3"
python_bin="/home/baishilong/miniconda3/envs/memgen/bin/python"
seeds=(42 142 242 342 442)
root="outputs/mab/ablation_effect_repeats/${run_id}/${variant}"
log_root="runtime_logs/ablation_effect_repeats/${run_id}/${variant}"
mkdir -p "$root" "$log_root"

case "$variant" in
  no_query_retrieval) runner="eval/exp/no_query_retrieval/eventqa_p7_no_query_retrieval.py" ;;
  no_retrieved_memory_conditioning) runner="eval/exp/no_retrieved_memory_conditioning/eventqa_p7_no_retrieved_memory_conditioning.py" ;;
  direct_top1) runner="eval/exp/direct_top1/eventqa_p7_direct_top1.py" ;;
  *) echo "unsupported variant: $variant" >&2; exit 2 ;;
esac

seed_summaries=()
for seed in "${seeds[@]}"; do
  artifacts=()
  for context_index in 0 1 2 3 4; do
    context_root="$root/seed${seed}/ctx${context_index}"
    CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" "$runner" \
      --measurement-scope full --context-index "$context_index" --question-limit 100 \
      --seed "$seed" --reseed-per-context --skip-research-note \
      --output-root "$context_root" \
      > "$log_root/seed${seed}_ctx${context_index}.log" 2>&1
    artifact="$(find "$context_root" -type f \( -name artifact.json -o -name smoke_artifact.json \) -print -quit)"
    [[ -n "$artifact" ]] || { echo "missing artifact for seed=$seed context=$context_index" >&2; exit 1; }
    artifacts+=("$artifact")
  done
  summary="$root/seed${seed}/aggregate.json"
  "$python_bin" eval/exp/ablations/aggregate_repeats.py seed \
    --variant "$variant" --artifacts "${artifacts[@]}" --output-json "$summary"
  seed_summaries+=("$summary")
done

"$python_bin" eval/exp/ablations/aggregate_repeats.py repeats \
  --seed-summaries "${seed_summaries[@]}" --output-json "$root/repeat_aggregate.json"
printf '%s\n' "ABLATION_REPEAT_ROOT=$root"
