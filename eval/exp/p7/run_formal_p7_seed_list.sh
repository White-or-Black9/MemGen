#!/usr/bin/env bash
# Run one or more independent process-level confirmations of formal P7 seeds.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <physical-gpu-index> <run-root> <seed> [<seed> ...]" >&2
  exit 2
fi

gpu_index="$1"
run_root="$2"
shift 2
python_bin="/home/baishilong/miniconda3/envs/memgen/bin/python"

for seed in "$@"; do
  CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
    eval/exp/p7/mab6b_weaver_space_bank_eventqa_65536_n5.py \
    --output-root "$run_root/seed${seed}" \
    --requested-contexts 5 \
    --seed "$seed" \
    --reseed-per-context \
    --max-slots 16 \
    --top-k 2 \
    --retrieve-threshold 0.05 \
    --update-threshold 0.10 \
    --skip-research-note
done
