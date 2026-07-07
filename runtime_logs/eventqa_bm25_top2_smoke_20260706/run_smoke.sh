#!/usr/bin/env bash
set -euo pipefail

cd /mnt/18T/baishilong/MemGen
export PYTHONPATH=/mnt/18T/baishilong/MemGen
export CUDA_VISIBLE_DEVICES=1

exec /home/baishilong/miniconda3/envs/memgen/bin/python \
  scripts/eval/eventqa_bm25_retrieved_text.py \
  --context-index 0 \
  --question-limit 10 \
  --output-root outputs/mab/eventqa_bm25_top2_smoke
