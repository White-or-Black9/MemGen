#!/usr/bin/env bash
set -euo pipefail

cd /mnt/18T/baishilong/MemGen
export PYTHONPATH=/mnt/18T/baishilong/MemGen
export CUDA_VISIBLE_DEVICES=7

for context_index in 0 1 2 3 4; do
  echo "MATCHED16_FULL_CONTEXT_START context=${context_index} timestamp=$(date --iso-8601=seconds)"
  /home/baishilong/miniconda3/envs/memgen/bin/python \
    scripts/eval/eventqa_matched16_retrieved_text.py \
    --measurement-scope full \
    --context-index "${context_index}" \
    --question-limit 100 \
    --output-root outputs/mab/eventqa_matched16_full
  echo "MATCHED16_FULL_CONTEXT_DONE context=${context_index} timestamp=$(date --iso-8601=seconds)"
done

echo "MATCHED16_FULL_QUEUE_DONE timestamp=$(date --iso-8601=seconds)"
