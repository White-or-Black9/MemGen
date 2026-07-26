#!/usr/bin/env bash
# Run the capacity-max MemGen recent-text baseline with auditable GPU isolation.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <physical-gpu-index>" >&2
  exit 2
fi

gpu_index="$1"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${run_stamp}-eventqa-memgen-recent-window-controlled-cost"
output_root="outputs/mab/eventqa_memgen_recent_window_controlled_cost/${run_id}"
log_root="runtime_logs/eventqa_memgen_recent_window_controlled_cost/${run_id}"
python_bin="/home/baishilong/miniconda3/envs/memgen/bin/python"

mkdir -p "$output_root" "$log_root"

preflight() {
  local context_index="$1"
  local path="$log_root/gpu_ctx${context_index}_preflight.txt"
  nvidia-smi -i "$gpu_index" --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$path"
  nvidia-smi -i "$gpu_index" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >> "$path"
  local memory_used
  memory_used="$(head -n 1 "$path" | awk -F, '{gsub(/[^0-9]/, "", $3); print $3}')"
  if [[ "$memory_used" -ge 1024 ]] || ! head -n 1 "$path" | rg -q '0 %$' || [[ -n "$(tail -n +2 "$path")" ]]; then
    echo "GPU ${gpu_index} is not clear before context ${context_index}; refusing cost run." >&2
    exit 3
  fi
}

monitor_worker() {
  local worker_pid="$1"
  local path="$2"
  while kill -0 "$worker_pid" 2>/dev/null; do
    printf 'timestamp=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'gpu='
    nvidia-smi -i "$gpu_index" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
    nvidia-smi -i "$gpu_index" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/process=/' || true
    sleep 5
  done > "$path"
}

for context_index in 0 1 2 3 4; do
  preflight "$context_index"
  context_run_id="${run_id}-ctx${context_index}"
  context_log="$log_root/ctx${context_index}.log"
  monitor_log="$log_root/gpu_ctx${context_index}_monitor.txt"
  CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" scripts/eval/eventqa_memgen_recent_window.py \
    --measurement-scope full \
    --context-index "$context_index" \
    --question-limit 100 \
    --recent-history-token-budget 32768 \
    --seed 42 \
    --output-root "$output_root/ctx${context_index}" \
    --run-id "$context_run_id" > "$context_log" 2>&1 &
  worker_pid=$!
  monitor_worker "$worker_pid" "$monitor_log" &
  monitor_pid=$!
  wait "$worker_pid"
  worker_status=$?
  wait "$monitor_pid" || true
  if [[ "$worker_status" -ne 0 ]]; then
    exit "$worker_status"
  fi
  if awk -F, -v worker_pid="$worker_pid" '
      /^process=[0-9]+,/ {gsub(/^process=/, "", $1); gsub(/ /, "", $1); if ($1 != worker_pid) exit 1}
    ' "$monitor_log"; then
    :
  else
    echo "external GPU process detected during context ${context_index}; invalidating cost run." >&2
    exit 4
  fi
done

printf '{\n  "schema_version": "eventqa-memgen-recent-window-controlled-cost/v1",\n  "run_id": "%s",\n  "gpu_index": %s,\n  "serialized_single_gpu": true,\n  "preflight_required": true,\n  "monitor_interval_seconds": 5,\n  "external_process_detection": "passed",\n  "context_indices": [0, 1, 2, 3, 4]\n}\n' "$run_id" "$gpu_index" > "$output_root/controlled_cost_evidence.json"
echo "Controlled-cost artifacts: $output_root"
