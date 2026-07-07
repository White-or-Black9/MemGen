#!/usr/bin/env bash
set -euo pipefail

cd /mnt/18T/baishilong/MemGen
export PYTHONPATH=/mnt/18T/baishilong/MemGen
export CUDA_VISIBLE_DEVICES=7

PYTHON=/home/baishilong/miniconda3/envs/memgen/bin/python
RUNNER=scripts/eval/eventqa_method_separable_cost.py
OUTPUT_ROOT=outputs/mab/eventqa_method_separable_cost_full
LOG_ROOT=runtime_logs/eventqa_cost_full_20260706

for context_index in 0 1 2 3 4; do
  for method in disabled p7; do
    extra_args=()
    if [[ "$method" == "p7" ]]; then
      extra_args=(
        --retrieve-threshold 0.05
        --update-threshold 0.10
        --max-slots 16
        --top-k 2
        --decay-alpha 0.05
      )
    fi

    "$PYTHON" -u "$RUNNER" \
      --method "$method" \
      --measurement-scope full \
      --output-root "$OUTPUT_ROOT" \
      --context-index "$context_index" \
      --question-limit 100 \
      --eventqa-protocol frozen_context_bank \
      --generation-max-length 40 \
      "${extra_args[@]}" \
      2>&1 | tee "$LOG_ROOT/${method}_ctx${context_index}.log"
  done
done
