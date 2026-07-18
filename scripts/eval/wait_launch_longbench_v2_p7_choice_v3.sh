#!/usr/bin/env bash
set -u

ROOT=/mnt/18T/baishilong/MemGen
LOG_DIR=runtime_logs/longbench_v2_choice_v3_waiter_20260714
MIN_FREE_MIB=20000
MAX_UTIL=20
POLL_SECONDS=60

cd "$ROOT" || exit 1
mkdir -p "$LOG_DIR"

while true; do
  while IFS=, read -r index free util; do
    index=${index//[[:space:]]/}
    free=${free//[[:space:]]/}
    util=${util//[[:space:]]/}
    if [[ "$index" =~ ^(1|6)$ ]] && (( free >= MIN_FREE_MIB && util <= MAX_UTIL )); then
      printf '%s selected_gpu=%s free_mib=%s util=%s\n' \
        "$(date --iso-8601=seconds)" "$index" "$free" "$util" \
        >> "$LOG_DIR/waiter.log"
      exec env CUDA_VISIBLE_DEVICES="$index" \
        bash scripts/eval/launch_longbench_v2_p7_choice_v3.sh
    fi
  done < <(
    nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
  )
  printf '%s waiting\n' "$(date --iso-8601=seconds)" >> "$LOG_DIR/waiter.log"
  sleep "$POLL_SECONDS"
done
