#!/usr/bin/env bash
set -u

ROOT=/mnt/18T/baishilong/MemGen
PYTHON=/home/baishilong/miniconda3/envs/memgen/bin/python
DATASET=/tmp/longbench-v2/data.json
MANIFEST=configs/eval/longbench_v2_p7_bounded_ids.json
OUTPUT_ROOT=outputs/longbench_v2/prompt_v2_p7_only
LOG_DIR=runtime_logs/longbench_v2_prompt_v2_p7_20260714

cd "$ROOT" || exit 1
mkdir -p "$LOG_DIR"

run() {
  local label=$1
  shift
  env -u LD_LIBRARY_PATH \
    PYTHONPATH="$ROOT" \
    CUDA_VISIBLE_DEVICES=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" scripts/eval/longbench_v2_p7.py \
      --dataset "$DATASET" \
      --manifest "$MANIFEST" \
      --output-root "$OUTPUT_ROOT/$label" \
      --methods p7 \
      --query-prompt-version strict_format_v2 \
      "$@"
}

set +e
run preflight --item-start 0 --item-stop 1 > "$LOG_DIR/preflight.log" 2>&1
preflight_rc=$?
printf '%s\n' "$preflight_rc" > "$LOG_DIR/preflight.rc"
if (( preflight_rc != 0 )); then
  exit "$preflight_rc"
fi

run full60 --item-start 0 --item-stop 60 > "$LOG_DIR/full60.log" 2>&1
full_rc=$?
printf '%s\n' "$full_rc" > "$LOG_DIR/full60.rc"
exit "$full_rc"
