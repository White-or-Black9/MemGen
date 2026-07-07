#!/usr/bin/env bash
set -euo pipefail

cd /mnt/18T/baishilong/MemGen
export PYTHONPATH=/mnt/18T/baishilong/MemGen
export CUDA_VISIBLE_DEVICES=0

exec /home/baishilong/miniconda3/envs/memgen/bin/python \
  scripts/eval/eventqa_text_summary_construction.py \
  --context-index 0 \
  --summary-token-budget 128 \
  --output-root outputs/mab/eventqa_text_summary_construction_smoke
