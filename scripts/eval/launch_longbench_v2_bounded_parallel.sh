#!/usr/bin/env bash
set -u

ROOT=/mnt/18T/baishilong/MemGen
PYTHON=/home/baishilong/miniconda3/envs/memgen/bin/python
DATASET=/tmp/longbench-v2/data.json
MANIFEST=configs/eval/longbench_v2_p7_bounded_ids.json
OUTPUT_ROOT=outputs/longbench_v2/bounded_60_shards
STATE_DIR=runtime_logs/longbench_v2_bounded_parallel_20260713
POLL_SECONDS=60

cd "$ROOT" || exit 1
mkdir -p "$STATE_DIR"

task_names=(
  longbench_v2_p7_00_20
  longbench_v2_p7_20_40
  longbench_v2_p7_40_60
  longbench_v2_disabled_00_60
)
task_methods=(
  p7,p7_no_query_retrieval
  p7,p7_no_query_retrieval
  p7,p7_no_query_retrieval
  disabled_window_fit
)
task_starts=(0 20 40 0)
task_stops=(20 40 60 60)
task_min_free_mib=(20000 20000 20000 35840)
task_max_util=(20 20 20 20)

log_status() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$STATE_DIR/scheduler.log"
}

worker_is_active() {
  tmux has-session -t "$1" 2>/dev/null
}

reserved_gpu() {
  local candidate=$1
  local task gpu
  for task in "${task_names[@]}"; do
    if worker_is_active "$task" && [[ -f "$STATE_DIR/$task.gpu" ]]; then
      gpu=$(<"$STATE_DIR/$task.gpu")
      [[ "$gpu" == "$candidate" ]] && return 0
    fi
  done
  return 1
}

choose_gpu() {
  local min_free=$1
  local max_util=$2
  local index free util
  while IFS=, read -r index free util; do
    index=${index//[[:space:]]/}
    free=${free//[[:space:]]/}
    util=${util//[[:space:]]/}
    if (( free >= min_free && util <= max_util )) && ! reserved_gpu "$index"; then
      printf '%s\n' "$index"
      return 0
    fi
  done < <(
    nvidia-smi \
      --query-gpu=index,memory.free,utilization.gpu \
      --format=csv,noheader,nounits \
      | sort -t, -k2,2nr
  )
  return 1
}

launch_worker() {
  local task=$1 methods=$2 start=$3 stop=$4 gpu=$5
  local log="$STATE_DIR/$task.log"
  local rc_file="$STATE_DIR/$task.rc"
  printf '%s\n' "$gpu" > "$STATE_DIR/$task.gpu"
  tmux new-session -d -s "$task" \
    "cd '$ROOT' && set +e; env -u LD_LIBRARY_PATH PYTHONPATH='$ROOT' CUDA_VISIBLE_DEVICES='$gpu' PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '$PYTHON' scripts/eval/longbench_v2_p7.py --dataset '$DATASET' --manifest '$MANIFEST' --output-root '$OUTPUT_ROOT' --methods '$methods' --item-start '$start' --item-stop '$stop' > '$log' 2>&1; rc=\$?; printf '%s\\n' \"\$rc\" > '$rc_file'; exit \"\$rc\""
  log_status "launched task=$task gpu=$gpu methods=$methods slice=[$start:$stop]"
}

log_status "scheduler_started poll_seconds=$POLL_SECONDS"
while true; do
  unfinished=0
  for index in "${!task_names[@]}"; do
    task=${task_names[$index]}
    if [[ -f "$STATE_DIR/$task.rc" ]] || worker_is_active "$task"; then
      continue
    fi
    unfinished=$((unfinished + 1))
    if gpu=$(choose_gpu "${task_min_free_mib[$index]}" "${task_max_util[$index]}"); then
      launch_worker \
        "$task" \
        "${task_methods[$index]}" \
        "${task_starts[$index]}" \
        "${task_stops[$index]}" \
        "$gpu"
    fi
  done

  if (( unfinished == 0 )); then
    log_status "scheduler_finished"
    exit 0
  fi
  log_status "waiting unfinished=$unfinished"
  sleep "$POLL_SECONDS"
done
