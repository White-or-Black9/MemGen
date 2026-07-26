#!/usr/bin/env bash
# Five complete EventQA passes for the pre-specified alpha=0 no-decay ablation.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <physical-gpu-index> <run-id>" >&2
  exit 2
fi

gpu_index="$1"
run_id="$2"
python_bin="/home/baishilong/miniconda3/envs/memgen/bin/python"
root="outputs/mab/no_decay_effect_repeats/${run_id}"
log_root="runtime_logs/no_decay_effect_repeats/${run_id}"
seeds=(42 142 242 342 442)

mkdir -p "$root" "$log_root"

for seed in "${seeds[@]}"; do
  CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
    eval/exp/no_decay/eventqa_p7_no_decay.py \
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

printf '%s\n' "NO_DECAY_EFFECT_ROOT=$root"
