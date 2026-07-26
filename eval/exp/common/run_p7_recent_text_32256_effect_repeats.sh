#!/usr/bin/env bash
# Reproduce P7 and capacity-max recent-text EventQA effectiveness rows.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <physical-gpu-index> <p7|recent_text_32256> <run-id>" >&2
  exit 2
fi

gpu_index="$1"
method="$2"
run_id="$3"
python_bin="/home/baishilong/miniconda3/envs/memgen/bin/python"
seeds=(42 142 242 342 442)
root="outputs/mab/p7_recent_text_32256_effect_repeats/${run_id}/${method}"
log_root="runtime_logs/p7_recent_text_32256_effect_repeats/${run_id}/${method}"
mkdir -p "$root" "$log_root"

case "$method" in
  p7)
    for seed in "${seeds[@]}"; do
      CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
        eval/exp/p7/mab6b_weaver_space_bank_eventqa_65536_n5.py \
        --output-root "$root/seed${seed}" --requested-contexts 5 --seed "$seed" \
        --reseed-per-context --skip-research-note \
        > "$log_root/seed${seed}.log" 2>&1
    done
    ;;
  recent_text_32256)
    for seed in "${seeds[@]}"; do
      artifacts=()
      for context_index in 0 1 2 3 4; do
        context_root="$root/seed${seed}/ctx${context_index}"
        CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
          eval/exp/recent_text/eventqa_memgen_recent_window.py \
          --measurement-scope full --context-index "$context_index" --question-limit 100 \
          --seed "$seed" --reseed-per-context --skip-research-note \
          --recent-history-token-budget 32768 --generation-reserve-tokens 40 \
          --output-root "$context_root" \
          > "$log_root/seed${seed}_ctx${context_index}.log" 2>&1
        artifact="$(find "$context_root" -name full_artifact.json -type f -print -quit)"
        [[ -n "$artifact" ]] || { echo "missing recent-text artifact" >&2; exit 1; }
        artifacts+=(--artifact "$artifact")
      done
      "$python_bin" eval/exp/recent_text/aggregate.py "${artifacts[@]}" \
        --output-json "$root/seed${seed}/aggregate.json" \
        --output-md "$root/seed${seed}/aggregate.md"
    done
    ;;
  *) echo "unsupported method: $method" >&2; exit 2 ;;
esac

printf '%s\n' "P7_RECENT_TEXT_32256_EFFECT_ROOT=$root"
