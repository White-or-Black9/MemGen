# Current Experiment Status For Paper

Date: 2026-07-05

## EventQA Status

- Main positive benchmark and current paper anchor.
- Frozen P7/P6 five-repeat effectiveness rows are complete and reusable.
- P7: EM `0.197+-0.020`, recall `0.254+-0.028`, format failures
  `121.4+-8.8`.
- Bank-off: EM `0.008`, recall `0.178`.
- Context 4 remains a severe limitation: EM `0.006`, recall `0.228`, format
  failures `93.8/100`.

## LoCoMo Status

- Diagnostic / limitation only.
- Protocol-correct construction, retrieval, frozen snapshot, and write block.
- Disabled/P7 EM `0/0`; F1 `0.01834/0.02084`.
- All 304 paired rows exact-match wrong; P7 denial/refusal `138/153`.

## Existing Reliable Evidence

- Frozen P7 method definition and runtime parameters.
- EventQA P7/P6 five-repeat effectiveness.
- Bank-off effectiveness.
- Per-context and transition metrics.
- Prompt/format ablations and failure taxonomy.
- Context-4 analysis and single-bank oracle attribution with caveats.
- LoCoMo row-level protocol invariants and deterministic EM/F1.

## Missing Evidence

- method-separable EventQA costs;
- text-summary baseline;
- BM25/RAG baseline;
- matched-budget baseline;
- P7 no-query-retrieval ablation;
- final unified tables and manifest.

## Unreliable Or Non-Paper-Facing Fields

- Existing EventQA paired latency is Bank-off plus Bank-on; peak memory is the
  maximum over both, not a standalone method cost.
- LoCoMo construction Trigger/Weaver counters, latency, and peak GPU memory
  have known propagation gaps.
- Historical exploratory threshold/capacity rows are not all one-factor P7
  comparisons.
- Harmful tuple attribution is single-bank oracle analysis.

## Frozen Decisions

- Working title and EventQA-scoped claim.
- EventQA main / LoCoMo limitation benchmark roles.
- Frozen P7 configuration and no component retraining.
- No P7/P6 effectiveness rerun by default.
- Unchanged official non-strict EventQA prompt/parser/scorer.

## Open Decisions

- Exact summary generator and summary budget.
- BM25 implementation and capacity-safe text truncation rule.
- Whether to report strict 16-token and storage-matched RAG as separate rows.
- Exact no-query-retrieval runner interface.
- Final split between main effectiveness table and cost table.
