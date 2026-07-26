#!/usr/bin/env bash
# Five complete EventQA passes for the paper's formal P7 configuration.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <physical-gpu-index> <run-id>" >&2
  exit 2
fi

gpu_index="$1"
run_id="$2"
python_bin="/home/baishilong/miniconda3/envs/memgen/bin/python"
root="outputs/mab/formal_p7_effect_repeats/${run_id}"
log_root="runtime_logs/formal_p7_effect_repeats/${run_id}"
seeds=(42 142 242 342 442)

mkdir -p "$root" "$log_root"

for seed in "${seeds[@]}"; do
  CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
    eval/exp/p7/mab6b_weaver_space_bank_eventqa_65536_n5.py \
    --output-root "$root/seed${seed}" \
    --requested-contexts 5 \
    --seed "$seed" \
    --reseed-per-context \
    --max-slots 16 \
    --top-k 2 \
    --retrieve-threshold 0.05 \
    --update-threshold 0.10 \
    --skip-research-note \
    > "$log_root/seed${seed}.log" 2>&1
done

printf '%s\n' "FORMAL_P7_EFFECT_ROOT=$root"
