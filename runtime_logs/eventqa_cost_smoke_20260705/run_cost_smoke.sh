#!/usr/bin/env bash
set -euo pipefail

cd /mnt/18T/baishilong/MemGen
export PYTHONPATH=/mnt/18T/baishilong/MemGen
export CUDA_VISIBLE_DEVICES=7

PYTHON=/home/baishilong/miniconda3/envs/memgen/bin/python
RUNNER=scripts/eval/eventqa_method_separable_cost.py
OUTPUT_ROOT=outputs/mab/eventqa_method_separable_cost_smoke
LOG_ROOT=runtime_logs/eventqa_cost_smoke_20260705

"$PYTHON" -u "$RUNNER" \
  --method disabled \
  --output-root "$OUTPUT_ROOT" \
  --context-index 0 \
  --question-limit 10 \
  --eventqa-protocol frozen_context_bank \
  --generation-max-length 40 \
  2>&1 | tee "$LOG_ROOT/disabled.log"

"$PYTHON" -u "$RUNNER" \
  --method p7 \
  --output-root "$OUTPUT_ROOT" \
  --context-index 0 \
  --question-limit 10 \
  --eventqa-protocol frozen_context_bank \
  --retrieve-threshold 0.05 \
  --update-threshold 0.10 \
  --max-slots 16 \
  --top-k 2 \
  --decay-alpha 0.05 \
  --generation-max-length 40 \
  2>&1 | tee "$LOG_ROOT/p7.log"
