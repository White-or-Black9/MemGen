#!/usr/bin/env bash
set -euo pipefail

cd /mnt/18T/baishilong/MemGen
export PYTHONPATH=/mnt/18T/baishilong/MemGen
export CUDA_VISIBLE_DEVICES=7

/home/baishilong/miniconda3/envs/memgen/bin/python -u \
  scripts/eval/eventqa_method_separable_cost.py \
  --method p7 \
  --output-root outputs/mab/eventqa_method_separable_cost_smoke \
  --context-index 0 \
  --question-limit 10 \
  --eventqa-protocol frozen_context_bank \
  --retrieve-threshold 0.05 \
  --update-threshold 0.10 \
  --max-slots 16 \
  --top-k 2 \
  --decay-alpha 0.05 \
  --generation-max-length 40 \
  2>&1 | tee runtime_logs/eventqa_cost_smoke_20260705/p7_retry.log
