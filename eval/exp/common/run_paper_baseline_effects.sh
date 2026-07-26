#!/usr/bin/env bash
# Reproduce paper-facing EventQA effectiveness rows without changing method logic.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <physical-gpu-index> <p7|recent_text|rolling_summary|bm25_top2|matched16|comma-separated-list>" >&2
  exit 2
fi
gpu_index="$1"
methods_csv="$2"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${stamp}-eventqa-paper-baseline-effects"
root="outputs/mab/paper_baseline_effects/${run_id}"
log_root="runtime_logs/paper_baseline_effects/${run_id}"
python_bin="/home/baishilong/miniconda3/envs/memgen/bin/python"
seeds=(42 142 242 342 442)
mkdir -p "$root" "$log_root"

run_p7() {
  for seed in "${seeds[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" eval/exp/p7/mab6b_weaver_space_bank_eventqa_65536_n5.py \
      --output-root "$root/p7/seed${seed}" --requested-contexts 5 --seed "$seed" \
      --reseed-per-context --skip-research-note \
      > "$log_root/p7_seed${seed}.log" 2>&1
  done
}

run_recent_text() {
  for seed in "${seeds[@]}"; do
    args=()
    for context_index in 0 1 2 3 4; do
      context_root="$root/recent_text/seed${seed}/ctx${context_index}"
      CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" eval/exp/recent_text/eventqa_memgen_recent_window.py \
        --measurement-scope full --context-index "$context_index" --question-limit 100 \
        --seed "$seed" --reseed-per-context --output-root "$context_root" \
        > "$log_root/recent_text_seed${seed}_ctx${context_index}.log" 2>&1
      artifact="$(find "$context_root" -name full_artifact.json -type f -print -quit)"
      [[ -n "$artifact" ]] || { echo "missing recent-text artifact" >&2; exit 1; }
      args+=(--artifact "$artifact")
    done
    "$python_bin" eval/exp/recent_text/aggregate.py "${args[@]}" \
      --output-json "$root/recent_text/seed${seed}/aggregate.json" \
      --output-md "$root/recent_text/seed${seed}/aggregate.md"
  done
}

run_rolling_summary() {
  for seed in "${seeds[@]}"; do
    args=()
    for context_index in 0 1 2 3 4; do
      construction_id="${run_id}-rolling-seed${seed}-construction-ctx${context_index}"
      query_id="${run_id}-rolling-seed${seed}-query-ctx${context_index}"
      construction_root="$root/rolling_summary/seed${seed}/construction"
      query_root="$root/rolling_summary/seed${seed}/query"
      CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" eval/exp/rolling_summary/construction.py \
        --context-index "$context_index" --seed "$seed" --reseed-per-context \
        --output-root "$construction_root" --run-id "$construction_id" \
        > "$log_root/rolling_seed${seed}_construction_ctx${context_index}.log" 2>&1
      construction_artifact="$construction_root/$construction_id/construction_artifact.json"
      CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" eval/exp/rolling_summary/query.py \
        --measurement-scope full --context-index "$context_index" --question-limit 100 \
        --seed "$seed" --reseed-per-context --summary-artifact "$construction_artifact" \
        --output-root "$query_root" --run-id "$query_id" \
        > "$log_root/rolling_seed${seed}_query_ctx${context_index}.log" 2>&1
      args+=(--construction "$construction_artifact" --query "$query_root/$query_id/query_artifact.json")
    done
    "$python_bin" eval/exp/rolling_summary/aggregate.py "${args[@]}" \
      --output-json "$root/rolling_summary/seed${seed}/aggregate.json" \
      --output-md "$root/rolling_summary/seed${seed}/aggregate.md"
  done
}

run_explicit() {
  local method="$1" script aggregate
  if [[ "$method" == "bm25_top2" ]]; then
    script="eval/exp/bm25_top2/eventqa_bm25_retrieved_text.py"
    aggregate="eval/exp/bm25_top2/aggregate.py"
  else
    script="eval/exp/matched16/eventqa_matched16_retrieved_text.py"
    aggregate="eval/exp/matched16/aggregate.py"
  fi
  for seed in "${seeds[@]}"; do
    args=()
    for context_index in 0 1 2 3 4; do
      context_root="$root/$method/seed${seed}/ctx${context_index}"
      CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" "$script" \
        --measurement-scope full --context-index "$context_index" --question-limit 100 \
        --seed "$seed" --reseed-per-context --output-root "$context_root" \
        > "$log_root/${method}_seed${seed}_ctx${context_index}.log" 2>&1
      artifact="$(find "$context_root" -name full_artifact.json -type f -print -quit)"
      [[ -n "$artifact" ]] || { echo "missing $method artifact" >&2; exit 1; }
      args+=(--artifact "$artifact")
    done
    "$python_bin" "$aggregate" "${args[@]}" \
      --output-json "$root/$method/seed${seed}/aggregate.json" \
      --output-md "$root/$method/seed${seed}/aggregate.md"
  done
}

IFS=',' read -r -a methods <<< "$methods_csv"
for method in "${methods[@]}"; do
  case "$method" in
    p7) run_p7 ;;
    recent_text) run_recent_text ;;
    rolling_summary) run_rolling_summary ;;
    bm25_top2|matched16) run_explicit "$method" ;;
    *) echo "unsupported method: $method" >&2; exit 2 ;;
  esac
done

printf '{\n  "schema_version": "eventqa-paper-baseline-effects/v1",\n  "run_id": "%s",\n  "gpu_index": %s,\n  "methods": "%s",\n  "base_seeds": [42, 142, 242, 342, 442],\n  "scope": "5 contexts x 100 questions per seed",\n  "timing_not_paper_facing": true\n}\n' "$run_id" "$gpu_index" "$methods_csv" > "$root/campaign_manifest.json"
echo "PAPER_BASELINE_EFFECTS_ROOT=$root"
