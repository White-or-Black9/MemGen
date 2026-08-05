#!/usr/bin/env bash
# Five aligned 500-question passes for paper-facing dense E5 top-2 effectiveness.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <physical-gpu-index> <run-id>" >&2
  exit 2
fi

gpu_index="$1"
run_id="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="outputs/mab/dense_top2_effect_repeats/${run_id}"
log_root="runtime_logs/dense_top2_effect_repeats/${run_id}"
seeds=(42 142 242 342 442)

mkdir -p "$root" "$log_root"
cd "$repo_root"

for seed in "${seeds[@]}"; do
  CUDA_VISIBLE_DEVICES="$gpu_index" bash eval/exp/dense_top2/run_eventqa_full_pass.sh \
    "$root/seed${seed}" "$seed" > "$log_root/seed${seed}.log" 2>&1
done

pass_args=()
for seed in "${seeds[@]}"; do
  aggregate="$root/seed${seed}/aggregate.json"
  test -f "$aggregate"
  pass_args+=(--pass-aggregate "$aggregate")
done
/home/baishilong/miniconda3/envs/memgen/bin/python eval/exp/dense_top2/aggregate_repeats.py \
  "${pass_args[@]}" --output-json "$root/repeat_aggregate.json"
