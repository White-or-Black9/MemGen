#!/usr/bin/env bash
set -euo pipefail

# Queue one explicit EventQA control repeat.  A context starts only after a
# clear-GPU preflight and a per-device lock, so independently queued controls
# cannot select the same card.
if [[ $# -ne 2 ]]; then
  echo "usage: $0 {text_summary|bm25|matched16} REPEAT" >&2
  exit 2
fi

method="$1"
repeat="$2"
case "$method" in text_summary|bm25|matched16) ;; *) exit 2 ;; esac
[[ "$repeat" =~ ^[2-5]$ ]] || exit 2

root="outputs/mab/eventqa_explicit_controls_repeats/20260719T072509Z-eventqa-explicit-controls-repeats"
logs="runtime_logs/eventqa_explicit_controls_repeats/20260719T072509Z-eventqa-explicit-controls-repeats"
py="/home/baishilong/miniconda3/envs/memgen/bin/python"

for context in 0 1 2 3 4; do
  while :; do
    for gpu in 0 1 2 3 4 5 6 7; do
      mem=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc 0-9)
      util=$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -dc 0-9)
      if [[ "$mem" -ge 1024 || "$util" -ne 0 ]] || nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | rg -q '[0-9]'; then
        continue
      fi
      exec 9>"/tmp/memgen_eventqa_gpu${gpu}.lock"
      flock -n 9 || continue
      mem=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -dc 0-9)
      util=$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -dc 0-9)
      if [[ "$mem" -ge 1024 || "$util" -ne 0 ]] || nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | rg -q '[0-9]'; then
        flock -u 9
        continue
      fi
      printf 'method=%s repeat=%s context=%s gpu=%s preflight=%sMiB,%s%%\n' "$method" "$repeat" "$context" "$gpu" "$mem" "$util" | tee "$logs/${method}_rep${repeat}_ctx${context}.queue_preflight.txt"
      case "$method" in
        text_summary)
          base="$root/text_summary/rep${repeat}"
          cid="20260719T072509Z-eventqa-explicit-controls-repeats-text-summary-rep${repeat}-construction-ctx${context}"
          qid="20260719T072509Z-eventqa-explicit-controls-repeats-text-summary-rep${repeat}-query-ctx${context}"
          CUDA_VISIBLE_DEVICES="$gpu" "$py" scripts/eval/eventqa_text_summary_construction.py --context-index "$context" --output-root "$base/construction" --run-id "$cid" > "$logs/text_summary_rep${repeat}_construction_ctx${context}.queue.log" 2>&1
          CUDA_VISIBLE_DEVICES="$gpu" "$py" scripts/eval/eventqa_text_summary_query.py --measurement-scope full --context-index "$context" --question-limit 100 --summary-artifact "$base/construction/$cid/construction_artifact.json" --output-root "$base/query" --run-id "$qid" > "$logs/text_summary_rep${repeat}_query_ctx${context}.queue.log" 2>&1
          ;;
        bm25)
          CUDA_VISIBLE_DEVICES="$gpu" "$py" scripts/eval/eventqa_bm25_retrieved_text.py --measurement-scope full --context-index "$context" --question-limit 100 --output-root "$root/bm25/rep${repeat}/ctx${context}" > "$logs/bm25_rep${repeat}_ctx${context}.queue.log" 2>&1
          ;;
        matched16)
          CUDA_VISIBLE_DEVICES="$gpu" "$py" scripts/eval/eventqa_matched16_retrieved_text.py --measurement-scope full --context-index "$context" --question-limit 100 --output-root "$root/matched16/rep${repeat}/ctx${context}" > "$logs/matched16_rep${repeat}_ctx${context}.queue.log" 2>&1
          ;;
      esac
      flock -u 9
      break 2
    done
    sleep 30
  done
done
