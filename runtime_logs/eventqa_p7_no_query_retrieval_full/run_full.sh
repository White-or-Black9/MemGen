#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/18T/baishilong/MemGen"
PYTHON="/home/baishilong/miniconda3/envs/memgen/bin/python"
SCRIPT="scripts/eval/eventqa_p7_no_query_retrieval.py"
OUTPUT_ROOT="outputs/mab/eventqa_p7_no_query_retrieval_full"
LOG_ROOT="$ROOT/runtime_logs/eventqa_p7_no_query_retrieval_full"
GPU="${CUDA_VISIBLE_DEVICES:-7}"

mkdir -p "$LOG_ROOT"
cd "$ROOT"

for ctx in 0 1 2 3 4; do
  LOG_FILE="$LOG_ROOT/context_${ctx}.log"
  env PYTHONPATH="$ROOT" CUDA_VISIBLE_DEVICES="$GPU" \
    "$PYTHON" "$SCRIPT" \
    --measurement-scope full \
    --output-root "$OUTPUT_ROOT" \
    --context-index "$ctx" \
    --question-limit 100 \
    >"$LOG_FILE" 2>&1
done
