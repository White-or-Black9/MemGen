#!/usr/bin/env bash
# Run each approved GPU worker inside its own detached tmux session.
set -euo pipefail
if [[ $# -ne 2 ]] || [[ ! "$1" =~ ^[0-7]$ ]] || [[ ! "$2" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "usage: bash scripts/eval/run_review_m3_formal.sh <gpu0-7> <campaign-id>" >&2
  exit 2
fi
gpu_index="$1"
campaign_id="$2"
campaign_path="outputs/mab/review_m3_formal/$campaign_id"
log_path="runtime_logs/review_m3_formal/$campaign_id/gpu$gpu_index"
mkdir -p "$log_path"
# Append on explicit recovery; never erase previous failure evidence.
exec >>"$log_path/run.log" 2>&1
date -u
nvidia-smi -i "$gpu_index" --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader
export CUDA_VISIBLE_DEVICES="$gpu_index"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
set +e
/home/baishilong/miniconda3/envs/memgen/bin/python -u scripts/eval/review_m3_formal.py worker --campaign "$campaign_path" --gpu "$gpu_index"
worker_exit_code="$?"
set -e
printf '%s\n' "$worker_exit_code" >> "$log_path/exit_code"
exit "$worker_exit_code"
