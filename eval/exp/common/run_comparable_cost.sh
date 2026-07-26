#!/usr/bin/env bash
# Comparable EventQA Table-4 cost campaign.  Does not alter any method path.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <physical-gpu-index> [repeat-count]" >&2
  exit 2
fi
gpu_index="$1"
repeat_count="${2:-3}"
if ! [[ "$repeat_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "repeat-count must be a positive integer" >&2
  exit 2
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${stamp}-eventqa-comparable-cost"
output_root="outputs/mab/eventqa_comparable_cost/${run_id}"
python_bin="/home/baishilong/miniconda3/envs/memgen/bin/python"
mkdir -p "$output_root/logs"

preflight() {
  local repeat="$1" label="$2"
  local path="$output_root/logs/rep${repeat}_${label}_gpu_preflight.txt"
  : > "$path"
  local stable_samples=0
  local attempt gpu_line process_lines memory_used utilization
  for attempt in $(seq 1 12); do
    gpu_line="$(nvidia-smi -i "$gpu_index" --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader)"
    process_lines="$(nvidia-smi -i "$gpu_index" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)"
    printf 'attempt=%s timestamp=%s\n%s\n%s\n' "$attempt" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$gpu_line" "$process_lines" >> "$path"
    memory_used="$(printf '%s\n' "$gpu_line" | awk -F, '{gsub(/[^0-9]/, "", $3); print $3}')"
    utilization="$(printf '%s\n' "$gpu_line" | awk -F, '{gsub(/[^0-9]/, "", $5); print $5}')"
    if [[ "$memory_used" -lt 1024 && "$utilization" -lt 5 && -z "$process_lines" ]]; then
      stable_samples=$((stable_samples + 1))
      if [[ "$stable_samples" -ge 3 ]]; then
        printf '%s' "$path"
        return 0
      fi
    else
      stable_samples=0
    fi
    sleep 5
  done
  echo "GPU ${gpu_index} did not reach three consecutive idle samples before rep${repeat} ${label}." >&2
  return 3
}

run_worker() {
  local repeat="$1" label="$2"
  shift 2
  local run_log="$output_root/logs/rep${repeat}_${label}.log"
  local monitor_log="$output_root/logs/rep${repeat}_${label}_gpu_monitor.txt"
  CUDA_VISIBLE_DEVICES="$gpu_index" "$@" > "$run_log" 2>&1 &
  local worker_pid=$!
  while kill -0 "$worker_pid" 2>/dev/null; do
    printf 'timestamp=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'gpu='; nvidia-smi -i "$gpu_index" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
    nvidia-smi -i "$gpu_index" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/process=/' || true
    sleep 5
  done > "$monitor_log" &
  local monitor_pid=$!
  wait "$worker_pid"
  local worker_status=$?
  wait "$monitor_pid" || true
  if [[ "$worker_status" -ne 0 ]]; then exit "$worker_status"; fi
  if ! awk -F, -v worker_pid="$worker_pid" '/^process=[0-9]+,/ {gsub(/^process=/, "", $1); gsub(/ /, "", $1); if ($1 != worker_pid) exit 1}' "$monitor_log"; then
    echo "External GPU process detected during rep${repeat} ${label}." >&2
    exit 4
  fi
}

for repeat in $(seq 1 "$repeat_count"); do
  repeat_root="$output_root/rep${repeat}"
  mkdir -p "$repeat_root"

  preflight "$repeat" p7 >/dev/null
  run_worker "$repeat" p7 "$python_bin" eval/exp/p7/eventqa_p7_continuous_cost.py --output-root "$repeat_root/p7" --run-id "${run_id}-rep${repeat}-p7" --seed 42

  preflight "$repeat" recent_text >/dev/null
  run_worker "$repeat" recent_text "$python_bin" eval/exp/recent_text/continuous_cost.py --output-root "$repeat_root/recent_text" --run-id "${run_id}-rep${repeat}-recent-text" --seed 42

  rolling_preflight="$(preflight "$repeat" rolling_summary)"
  rolling_evidence="$repeat_root/rolling_summary_preflight_evidence.json"
  printf '{\n  "schema_version": "eventqa-text-summary-controlled-cost/v1",\n  "gpu_index": %s,\n  "context_indices": [0, 1, 2, 3, 4],\n  "serialized_single_gpu": true,\n  "all_preflight_clear": true,\n  "preflight_paths": ["%s"]\n}\n' "$gpu_index" "$rolling_preflight" > "$rolling_evidence"
  run_worker "$repeat" rolling_summary "$python_bin" eval/exp/rolling_summary/continuous_cost.py --output-root "$repeat_root/rolling_summary" --run-id "${run_id}-rep${repeat}-rolling-summary" --seed 42 --controlled-cost-evidence "$rolling_evidence"

  preflight "$repeat" bm25_top2 >/dev/null
  run_worker "$repeat" bm25_top2 "$python_bin" eval/exp/bm25_top2/continuous_cost.py --output-root "$repeat_root/bm25_top2" --run-id "${run_id}-rep${repeat}-bm25-top2" --seed 42

  preflight "$repeat" matched16 >/dev/null
  run_worker "$repeat" matched16 "$python_bin" eval/exp/matched16/continuous_cost.py --output-root "$repeat_root/matched16" --run-id "${run_id}-rep${repeat}-matched16" --seed 42
done

printf '{\n  "schema_version": "eventqa-comparable-cost-campaign/v1",\n  "run_id": "%s",\n  "gpu_index": %s,\n  "repeat_count": %s,\n  "methods": ["p7", "recent_text", "rolling_summary", "bm25_top2", "matched16"],\n  "serialized_single_gpu": true,\n  "continuous_process_per_method": true,\n  "model_loading_excluded": true,\n  "external_process_detection": "passed"\n}\n' "$run_id" "$gpu_index" "$repeat_count" > "$output_root/campaign_evidence.json"
echo "COMPARABLE_COST_OUTPUT_ROOT=$output_root"
