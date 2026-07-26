#!/usr/bin/env bash
# Resume only the missing independent EventQA passes in the paired 32k-text study.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <physical-gpu-index> <run-id>" >&2
  exit 2
fi

gpu_index="$1"
run_id="$2"
python_bin="/home/baishilong/miniconda3/envs/memgen/bin/python"
root="outputs/mab/p7_recent_text_32256_effect_repeats/${run_id}"
log_root="runtime_logs/p7_recent_text_32256_effect_repeats/${run_id}"

run_p7_seed() {
  local seed="$1"
  if find "$root/p7/seed${seed}" -name eventqa_aggregate.json -type f -print -quit | grep -q .; then
    echo "P7 seed ${seed} already complete; skipping"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
    eval/exp/p7/mab6b_weaver_space_bank_eventqa_65536_n5.py \
    --output-root "$root/p7/seed${seed}" --requested-contexts 5 --seed "$seed" \
    --reseed-per-context --skip-research-note \
    > "$log_root/p7/seed${seed}.resume.log" 2>&1
}

run_recent_seed() {
  local seed="$1" context_index context_root artifact
  local -a artifacts=()
  for context_index in 0 1 2 3 4; do
    context_root="$root/recent_text_32256/seed${seed}/ctx${context_index}"
    artifact="$(find "$context_root" -name full_artifact.json -type f -print -quit 2>/dev/null || true)"
    if [[ -z "$artifact" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
        eval/exp/recent_text/eventqa_memgen_recent_window.py \
        --measurement-scope full --context-index "$context_index" --question-limit 100 \
        --seed "$seed" --reseed-per-context --skip-research-note \
        --recent-history-token-budget 32768 --generation-reserve-tokens 40 \
        --output-root "$context_root" \
        > "$log_root/recent_text_32256/seed${seed}_ctx${context_index}.resume.log" 2>&1
      artifact="$(find "$context_root" -name full_artifact.json -type f -print -quit)"
    fi
    artifacts+=(--artifact "$artifact")
  done
  "$python_bin" eval/exp/recent_text/aggregate.py "${artifacts[@]}" \
    --output-json "$root/recent_text_32256/seed${seed}/aggregate.json" \
    --output-md "$root/recent_text_32256/seed${seed}/aggregate.md"
}

mkdir -p "$log_root/p7" "$log_root/recent_text_32256"
run_p7_seed 342
run_p7_seed 442
for seed in 142 242 342 442; do run_recent_seed "$seed"; done
printf '%s\n' "P7_RECENT_TEXT_32256_EFFECT_ROOT=$root"
