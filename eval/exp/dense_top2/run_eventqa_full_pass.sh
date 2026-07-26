#!/usr/bin/env bash
# One complete 500-question EventQA effectiveness pass for dense top-2 text retrieval.
# Requires CUDA_VISIBLE_DEVICES to name a GPU with enough free memory.
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one GPU before launching.}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin="${PYTHON_BIN:-/home/baishilong/miniconda3/envs/memgen/bin/python}"
output_root="${1:-outputs/mab/eventqa_dense_top2_full_pass}"
cd "$repo_root"

for context_index in 0 1 2 3 4; do
  "$python_bin" eval/exp/dense_top2/eventqa_dense_retrieved_text.py \
    --measurement-scope full \
    --context-index "$context_index" \
    --question-limit 100 \
    --embedding-device cpu \
    --embedding-batch-size 16 \
    --output-root "$output_root"
done

mapfile -t artifacts < <(find "$output_root" -name full_artifact.json -type f | sort)
if [[ "${#artifacts[@]}" -ne 5 ]]; then
  echo "Expected five full artifacts, found ${#artifacts[@]}" >&2
  exit 1
fi

aggregate_args=()
for artifact in "${artifacts[@]}"; do
  aggregate_args+=(--artifact "$artifact")
done
"$python_bin" eval/exp/dense_top2/aggregate.py "${aggregate_args[@]}" \
  --output-json "$output_root/aggregate.json"
