# EventQA Explicit-Memory Control Repeat Campaign

## Objective

Replace the current single-complete-pass interpretation of the rolling-summary,
BM25 top-2, and matched-16 explicit-memory controls with repeat-level
EventQA-65536 estimates comparable in repetition count to the frozen P7 main
result. This campaign does not change P7, the model, prompt, parser, scorer,
dataset, or visible-text budgets.

## Locked Comparison Contract

- Methods: same-model rolling text summary (128-token cap), BM25 top-2, and
  matched-16 BM25 retrieval.
- Existing complete pass for each method is retained as repeat 1. The controlled
  rolling-summary run at
  `outputs/mab/eventqa_text_summary_controlled_cost/20260719T031438Z-eventqa-text-summary-controlled-cost/`
  is its canonical repeat-1 cost artifact.
- Add four complete process-level repetitions per method, giving five complete
  500-question results per method (`12` new full passes; `6,000` new question
  answers).
- Every repetition uses EventQA-65536 contexts `0..4`, questions `0..99`, the
  same model/checkpoint, default prompt, official local parser/scorer,
  generation length `40`, base seed `42`, and per-context reseeding.
- The controlled rolling-summary cost run is accepted only after a clear
  single-GPU preflight: no compute process, zero utilization, and less than 1
  GiB already allocated. Repeats 2--5 may share GPUs because they are used only
  for effectiveness estimates; their timing and memory observations are not
  paper-facing evidence.

## Analysis Contract

- Aggregate each full pass with the existing strict per-method aggregators.
- Add a repeat-level aggregator that rejects missing/duplicate repetitions,
  incompatible coverage, incomplete 500-question passes, and non-finite
  metrics. Validate scorer/prompt/model/budget through the strict per-method
  artifact validators.
- Report mean and standard deviation of EM, EventQA recall, and
  format failures over five separate runs, plus per-run values and ranges.
- Compare P7 against every repeated control descriptively using matched metric
  deltas and bootstrap confidence intervals over run-level means. Do not call
  the repetitions independent-seed estimates: the fixed base seed matches the
  existing P7 protocol, and the campaign measures process-level variability.
- Cost remains protocol-specific. The controlled rolling-summary cost is
  paper-facing; repeated effectiveness runs contribute cost only when their
  preflight contract also passes.

## Execution Order And Stop Rules

1. Implement and unit-test repeat-level aggregation before GPU use.
2. Run one context-0 smoke for each method only if the new repeat launcher or
   artifact schema changes; otherwise reuse the validated runners directly.
3. Run repetitions 2--5 with shared-GPU parallelism permitted for effectiveness
   only; retain single-GPU preflights for any paper-facing cost measurement.
4. Stop the entire campaign on any contract drift, incomplete context,
   prompt/token-budget mismatch, scorer mismatch, or non-finite metric.
   Preserve completed repetitions, but do not merge a partial family into the
   paper table.
5. After five valid repeats per family, rebuild the unified comparison package
   and replace the manuscript's single-pass wording with repeat-level results.

## Scope Boundary

This is an EventQA supplementary comparison campaign only. It does not reopen
MemBench, LongBench, or any other external benchmark, and it does not alter the
frozen P7 main result.
