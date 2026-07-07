#!/usr/bin/env bash
set -euo pipefail

cd /mnt/18T/baishilong/MemGen
export PYTHONPATH=/mnt/18T/baishilong/MemGen
export CUDA_VISIBLE_DEVICES=3

for context_index in 0 1 2 3 4; do
  construction_id="text-summary-construction-ctx${context_index}-full-20260706"
  query_id="text-summary-query-ctx${context_index}-full-20260706"
  echo "TEXT_SUMMARY_CONSTRUCTION_START context=${context_index} timestamp=$(date --iso-8601=seconds)"
  /home/baishilong/miniconda3/envs/memgen/bin/python \
    scripts/eval/eventqa_text_summary_construction.py \
    --context-index "${context_index}" \
    --summary-token-budget 128 \
    --run-id "${construction_id}" \
    --output-root outputs/mab/eventqa_text_summary_full/construction
  echo "TEXT_SUMMARY_QUERY_START context=${context_index} timestamp=$(date --iso-8601=seconds)"
  /home/baishilong/miniconda3/envs/memgen/bin/python \
    scripts/eval/eventqa_text_summary_query.py \
    --measurement-scope full \
    --context-index "${context_index}" \
    --question-limit 100 \
    --summary-artifact "outputs/mab/eventqa_text_summary_full/construction/${construction_id}/construction_artifact.json" \
    --run-id "${query_id}" \
    --output-root outputs/mab/eventqa_text_summary_full/query
  echo "TEXT_SUMMARY_CONTEXT_DONE context=${context_index} timestamp=$(date --iso-8601=seconds)"
done

echo "TEXT_SUMMARY_FULL_QUEUE_DONE timestamp=$(date --iso-8601=seconds)"
