#!/usr/bin/env bash
# Run one parameter-locked construction-policy ablation for one or more seeds.
# Usage: bash $0 <physical-gpu-index> <wrapper.py> <run-root> <seed> [<seed> ...]
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <physical-gpu-index> <wrapper.py> <run-root> <seed> [<seed> ...]" >&2
  exit 2
fi

gpu_index="$1"
wrapper="$2"
run_root="$3"
shift 3
python_bin="/home/baishilong/miniconda3/envs/memgen/bin/python"

for seed in "$@"; do
  CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" "$wrapper" \
    --output-root "$run_root/seed${seed}" --requested-contexts 5 \
    --seed "$seed" --reseed-per-context --max-slots 16 --top-k 2 \
    --retrieve-threshold 0.05 --update-threshold 0.10 --decay-alpha 0.05 \
    --skip-research-note --bank-transition-diagnostics
done
