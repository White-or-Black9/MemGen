#!/usr/bin/env bash
# One bounded, serialized worker. Run inside a detached tmux session.
set -euo pipefail
if [[ $# -ne 2 ]] || [[ ! "$1" =~ ^[0-7]$ ]] || [[ ! "$2" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "usage: bash scripts/eval/run_review_m2_smoke.sh <gpu0-7> <unique-run-id>" >&2
  exit 2
fi
gpu_index="$1"
run_id="$2"
log_root="runtime_logs/review_m2_smoke/$run_id"
if [[ -e "$log_root" ]] || [[ -e "outputs/mab/review_m2_smoke/$run_id" ]]; then
  echo "refusing to overwrite existing run/log directory" >&2
  exit 2
fi
mkdir -p "$log_root"
exec >"$log_root/run.log" 2>&1
nvidia-smi -i "$gpu_index" --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader
export CUDA_VISIBLE_DEVICES="$gpu_index"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
set +e
/home/baishilong/miniconda3/envs/memgen/bin/python -u scripts/eval/review_m2_smoke.py --run-id "$run_id"
run_exit_code="$?"
set -e
printf '%s\n' "$run_exit_code" > "$log_root/exit_code"
exit "$run_exit_code"
