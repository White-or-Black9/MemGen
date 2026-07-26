#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <physical-gpu-index>" >&2
  exit 2
fi

gpu_index="$1"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${run_stamp}-eventqa-explicit-controls-repeats"
output_root="outputs/mab/eventqa_explicit_controls_repeats/${run_id}"
log_root="runtime_logs/eventqa_explicit_controls_repeats/${run_id}"
python_bin="/home/baishilong/miniconda3/envs/memgen/bin/python"
mkdir -p "$output_root" "$log_root"

preflight() {
  local method="$1" repeat="$2" context_index="$3"
  local path="$log_root/gpu_${method}_rep${repeat}_ctx${context_index}_preflight.txt"
  nvidia-smi -i "$gpu_index" --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$path"
  nvidia-smi -i "$gpu_index" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >> "$path"
  if rg -q '[0-9]+' <(tail -n +2 "$path") || ! head -n 1 "$path" | rg -q '0 %$'; then
    echo "GPU $gpu_index is not exclusively idle before ${method} rep${repeat} ctx${context_index}." >&2
    exit 1
  fi
  local memory_used_mib
  memory_used_mib="$(head -n 1 "$path" | awk -F, '{gsub(/[^0-9]/, "", $3); print $3}')"
  if [[ "$memory_used_mib" -ge 1024 ]]; then
    echo "GPU $gpu_index has ${memory_used_mib} MiB allocated before ${method} rep${repeat} ctx${context_index}." >&2
    exit 1
  fi
}

single_artifact() {
  local root="$1"
  local found
  mapfile -t found < <(find "$root" -name full_artifact.json -type f | sort)
  if [[ "${#found[@]}" -ne 1 ]]; then
    echo "Expected exactly one full_artifact.json under $root, found ${#found[@]}." >&2
    exit 1
  fi
  printf '%s' "${found[0]}"
}

for repeat in 2 3 4 5; do
  text_construction=()
  text_query=()
  bm25_artifacts=()
  matched_artifacts=()
  for context_index in 0 1 2 3 4; do
    preflight text_summary "$repeat" "$context_index"
    text_root="$output_root/text_summary/rep${repeat}"
    construction_id="${run_id}-text-summary-rep${repeat}-construction-ctx${context_index}"
    query_id="${run_id}-text-summary-rep${repeat}-query-ctx${context_index}"
    CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" scripts/eval/eventqa_text_summary_construction.py --context-index "$context_index" --output-root "$text_root/construction" --run-id "$construction_id" > "$log_root/text_summary_rep${repeat}_construction_ctx${context_index}.log" 2>&1
    construction_path="$text_root/construction/$construction_id/construction_artifact.json"
    CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" scripts/eval/eventqa_text_summary_query.py --measurement-scope full --context-index "$context_index" --question-limit 100 --summary-artifact "$construction_path" --output-root "$text_root/query" --run-id "$query_id" > "$log_root/text_summary_rep${repeat}_query_ctx${context_index}.log" 2>&1
    text_construction+=(--construction "$construction_path")
    text_query+=(--query "$text_root/query/$query_id/query_artifact.json")

    preflight bm25 "$repeat" "$context_index"
    bm25_root="$output_root/bm25/rep${repeat}/ctx${context_index}"
    CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" scripts/eval/eventqa_bm25_retrieved_text.py --measurement-scope full --context-index "$context_index" --question-limit 100 --output-root "$bm25_root" > "$log_root/bm25_rep${repeat}_ctx${context_index}.log" 2>&1
    bm25_artifacts+=(--artifact "$(single_artifact "$bm25_root")")

    preflight matched16 "$repeat" "$context_index"
    matched_root="$output_root/matched16/rep${repeat}/ctx${context_index}"
    CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" scripts/eval/eventqa_matched16_retrieved_text.py --measurement-scope full --context-index "$context_index" --question-limit 100 --output-root "$matched_root" > "$log_root/matched16_rep${repeat}_ctx${context_index}.log" 2>&1
    matched_artifacts+=(--artifact "$(single_artifact "$matched_root")")
  done
  "$python_bin" scripts/eval/eventqa_text_summary_aggregate.py "${text_construction[@]}" "${text_query[@]}" --output-json "$output_root/text_summary/rep${repeat}_aggregate.json" --output-md "$output_root/text_summary/rep${repeat}_aggregate.md"
  "$python_bin" scripts/eval/eventqa_bm25_aggregate.py "${bm25_artifacts[@]}" --output-json "$output_root/bm25/rep${repeat}_aggregate.json" --output-md "$output_root/bm25/rep${repeat}_aggregate.md"
  "$python_bin" scripts/eval/eventqa_matched16_aggregate.py "${matched_artifacts[@]}" --output-json "$output_root/matched16/rep${repeat}_aggregate.json" --output-md "$output_root/matched16/rep${repeat}_aggregate.md"
done

echo "Repeat campaign completed: $output_root"
