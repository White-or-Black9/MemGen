#!/usr/bin/env bash
# One-process, five-context audited cost run for frozen P7.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <physical-gpu-index>" >&2
  exit 2
fi

gpu_index="$1"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${run_stamp}-eventqa-p7-continuous-controlled-cost"
output_root="outputs/mab/eventqa_p7_continuous_controlled_cost/${run_id}"
log_root="runtime_logs/eventqa_p7_continuous_controlled_cost/${run_id}"
python_bin="/home/baishilong/miniconda3/envs/memgen/bin/python"
mkdir -p "$log_root"

preflight="$log_root/gpu_preflight.txt"
nvidia-smi -i "$gpu_index" --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$preflight"
nvidia-smi -i "$gpu_index" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >> "$preflight"
memory_used="$(head -n 1 "$preflight" | awk -F, '{gsub(/[^0-9]/, "", $3); print $3}')"
if [[ "$memory_used" -ge 1024 ]] || ! head -n 1 "$preflight" | rg -q '0 %$' || [[ -n "$(tail -n +2 "$preflight")" ]]; then
  echo "GPU ${gpu_index} is not clear; refusing continuous cost run." >&2
  exit 3
fi

run_log="$log_root/run.log"
monitor_log="$log_root/gpu_monitor.txt"
CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" scripts/eval/eventqa_p7_continuous_cost.py \
  --output-root "$output_root" --run-id "$run_id" --seed 42 > "$run_log" 2>&1 &
worker_pid=$!
while kill -0 "$worker_pid" 2>/dev/null; do
  printf 'timestamp=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'gpu='; nvidia-smi -i "$gpu_index" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true
  nvidia-smi -i "$gpu_index" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed 's/^/process=/' || true
  sleep 5
done > "$monitor_log" &
monitor_pid=$!
wait "$worker_pid"
worker_status=$?
wait "$monitor_pid" || true
if [[ "$worker_status" -ne 0 ]]; then exit "$worker_status"; fi
if ! awk -F, -v worker_pid="$worker_pid" '/^process=[0-9]+,/ {gsub(/^process=/, "", $1); gsub(/ /, "", $1); if ($1 != worker_pid) exit 1}' "$monitor_log"; then
  echo "external GPU process detected; invalidating continuous cost run." >&2
  exit 4
fi
printf '{\n  "schema_version": "eventqa-p7-continuous-controlled-cost/v1",\n  "run_id": "%s",\n  "gpu_index": %s,\n  "continuous_process": true,\n  "external_process_detection": "passed"\n}\n' "$run_id" "$gpu_index" > "$output_root/controlled_cost_evidence.json"
echo "Continuous controlled-cost artifacts: $output_root"
