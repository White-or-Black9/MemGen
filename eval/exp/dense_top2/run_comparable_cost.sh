#!/usr/bin/env bash
# Three serialized, idle-GPU Dense E5 top-2 Table-4 cost measurements.
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

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${stamp}-eventqa-dense-top2-comparable-cost"
output_root="outputs/mab/eventqa_dense_top2_comparable_cost/${run_id}"
python_bin="/home/baishilong/miniconda3/envs/memgen/bin/python"
mkdir -p "$output_root/logs"
cd "$repo_root"

preflight() {
  local repeat="$1" path="$output_root/logs/rep${repeat}_gpu_preflight.txt"
  : > "$path"
  local stable=0 line processes memory utilization
  for _ in $(seq 1 12); do
    line="$(nvidia-smi -i "$gpu_index" --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader)"
    processes="$(nvidia-smi -i "$gpu_index" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)"
    printf 'timestamp=%s\n%s\n%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$line" "$processes" >> "$path"
    memory="$(printf '%s\n' "$line" | awk -F, '{gsub(/[^0-9]/, "", $2); print $2}')"
    utilization="$(printf '%s\n' "$line" | awk -F, '{gsub(/[^0-9]/, "", $4); print $4}')"
    if [[ "$memory" -lt 1024 && "$utilization" -lt 5 && -z "$processes" ]]; then
      stable=$((stable + 1))
      [[ "$stable" -ge 3 ]] && return 0
    else
      stable=0
    fi
    sleep 5
  done
  echo "GPU ${gpu_index} did not reach three consecutive idle samples." >&2
  return 3
}

for repeat in $(seq 1 "$repeat_count"); do
  preflight "$repeat"
  run_log="$output_root/logs/rep${repeat}_dense_e5_top2.log"
  CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" eval/exp/dense_top2/continuous_cost.py \
    --output-root "$output_root/rep${repeat}" \
    --run-id "${run_id}-rep${repeat}-dense-e5-top2" \
    --seed 42 --embedding-device cpu --embedding-batch-size 16 > "$run_log" 2>&1
done

printf '{\n  "schema_version": "eventqa-dense-top2-comparable-cost-campaign/v1",\n  "run_id": "%s",\n  "gpu_index": %s,\n  "repeat_count": %s,\n  "serialized_single_gpu": true,\n  "continuous_process_per_repeat": true,\n  "model_loading_excluded": true,\n  "external_process_detection": "preflight_passed"\n}\n' \
  "$run_id" "$gpu_index" "$repeat_count" > "$output_root/campaign_evidence.json"
echo "DENSE_COMPARABLE_COST_OUTPUT_ROOT=$output_root"
