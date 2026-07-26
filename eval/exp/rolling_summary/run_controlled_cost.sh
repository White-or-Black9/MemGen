#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <physical-gpu-index>" >&2
  exit 2
fi

gpu_index="$1"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${run_stamp}-eventqa-text-summary-controlled-cost"
output_root="outputs/mab/eventqa_text_summary_controlled_cost/${run_id}"
log_root="runtime_logs/eventqa_text_summary_controlled_cost/${run_id}"
python_bin="/home/baishilong/miniconda3/envs/memgen/bin/python"

mkdir -p "$output_root" "$log_root"

preflight_paths=()
for context_index in 0 1 2 3 4; do
  preflight_path="$log_root/gpu_ctx${context_index}_preflight.txt"
  nvidia-smi -i "$gpu_index" --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$preflight_path"
  nvidia-smi -i "$gpu_index" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >> "$preflight_path"
  if rg -q '[0-9]+' <(tail -n +2 "$preflight_path"); then
    echo "GPU $gpu_index is occupied before context $context_index; refusing confounded timing run." >&2
    exit 1
  fi
  if ! head -n 1 "$preflight_path" | rg -q '0 %$'; then
    echo "GPU $gpu_index is not idle before context $context_index; refusing confounded timing run." >&2
    exit 1
  fi
  memory_used_mib="$(head -n 1 "$preflight_path" | awk -F, '{gsub(/[^0-9]/, "", $3); print $3}')"
  if [[ "$memory_used_mib" -ge 1024 ]]; then
    echo "GPU $gpu_index has ${memory_used_mib} MiB allocated before context $context_index; refusing confounded timing run." >&2
    exit 1
  fi
  preflight_paths+=("$preflight_path")

  construction_id="${run_id}-construction-ctx${context_index}"
  query_id="${run_id}-query-ctx${context_index}"
  CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" scripts/eval/eventqa_text_summary_construction.py \
    --context-index "$context_index" \
    --output-root "$output_root/construction" \
    --run-id "$construction_id" \
    > "$log_root/construction_ctx${context_index}.log" 2>&1
  construction_artifact="$output_root/construction/$construction_id/construction_artifact.json"
  CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" scripts/eval/eventqa_text_summary_query.py \
    --measurement-scope full \
    --context-index "$context_index" \
    --question-limit 100 \
    --summary-artifact "$construction_artifact" \
    --output-root "$output_root/query" \
    --run-id "$query_id" \
    > "$log_root/query_ctx${context_index}.log" 2>&1
done

evidence_path="$output_root/controlled_cost_evidence.json"
{
  printf '{\n'
  printf '  "schema_version": "eventqa-text-summary-controlled-cost/v1",\n'
  printf '  "gpu_index": %s,\n' "$gpu_index"
  printf '  "context_indices": [0, 1, 2, 3, 4],\n'
  printf '  "serialized_single_gpu": true,\n'
  printf '  "all_preflight_clear": true,\n'
  printf '  "preflight_paths": [\n'
  for index in "${!preflight_paths[@]}"; do
    comma=','
    if [[ "$index" -eq 4 ]]; then comma=''; fi
    printf '    "%s"%s\n' "${preflight_paths[$index]}" "$comma"
  done
  printf '  ]\n}\n'
} > "$evidence_path"

aggregate_args=()
for context_index in 0 1 2 3 4; do
  aggregate_args+=(--construction "$output_root/construction/${run_id}-construction-ctx${context_index}/construction_artifact.json")
  aggregate_args+=(--query "$output_root/query/${run_id}-query-ctx${context_index}/query_artifact.json")
done
"$python_bin" scripts/eval/eventqa_text_summary_aggregate.py "${aggregate_args[@]}" \
  --controlled-cost-evidence "$evidence_path" \
  --output-json "$output_root/eventqa_text_summary_controlled_cost_aggregate.json" \
  --output-md "$output_root/eventqa_text_summary_controlled_cost_aggregate.md"

echo "Controlled cost aggregate: $output_root/eventqa_text_summary_controlled_cost_aggregate.json"
