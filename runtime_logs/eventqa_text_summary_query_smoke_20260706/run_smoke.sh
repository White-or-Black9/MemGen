#!/usr/bin/env bash
set -euo pipefail

cd /mnt/18T/baishilong/MemGen
export PYTHONPATH=/mnt/18T/baishilong/MemGen
export CUDA_VISIBLE_DEVICES=4

exec /home/baishilong/miniconda3/envs/memgen/bin/python \
  scripts/eval/eventqa_text_summary_query.py \
  --context-index 0 \
  --question-limit 10 \
  --summary-artifact outputs/mab/eventqa_text_summary_construction_smoke/20260706T091043Z-eventqa-text-summary-construction-ctx0/construction_artifact.json \
  --output-root outputs/mab/eventqa_text_summary_query_smoke
