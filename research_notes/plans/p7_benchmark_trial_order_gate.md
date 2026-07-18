# P7 Benchmark Trial Order Gate

Date: 2026-07-10 (historical; paused 2026-07-18)

> This trial order is not active. DEC-0089 pauses all new benchmark routes
> while EventQA supplementary evidence is inventoried and one focused follow-up
> is pre-registered.

## Trial Order

1. `RULER-QA2`
2. `LongBench v2 subset`
3. `MemBench smoke`
4. `LongBench v1 selective`

## Non-Negotiable Invariants

- `EventQA` remains the paper main evidence anchor.
- Failed new benchmarks do not change the current paper mainline.
- `disabled` mode must preserve original MemGen behavior.
- Session-local memory bank reset occurs per context/sample.
- Query-time writes stay blocked in formal comparisons.
- Batch size stays `1`.
- No Weaver or Trigger training-path changes.
- No cross-session or cross-sample memory is introduced.

## Promotion Gate

A benchmark can enter appendix or table-ready status only if:

- the runner has tests for dataset identity, scorer contract, reset boundary, and mode comparability;
- `disabled`, `p7`, and `p7_no_query_retrieval` use the same query set;
- `p7` is not worse than `disabled` on the smoke primary metric;
- memory logs show nonzero construction writes and query retrieval attempts for enabled variants;
- any failure can be explained without a hidden protocol mismatch.

## Stop Conditions

Stop a benchmark path if:

- local data is missing or inconsistent with the assumed scorer;
- answer extraction needs manual judging for the first pass;
- mode comparison cannot be aligned on the same item IDs;
- smoke already shows a clean negative result with no protocol ambiguity.

## Historical Outcome Update (2026-07-10)

- `RULER-QA2` was executed under the adapted frozen-bank route and is now
  closed.
- Closure reason:
  the runner is valid, but the full adapted `p7` run is mechanism-negative:
  construction reaches `16` slots, yet query-time retrieval remains
  `0/100` and total accuracy is only `8/100`.
- Consequence:
  this trial-order gate is historical only. Any future benchmark expansion
  should start from a fresh benchmark-choice audit rather than continuing this
  frozen order automatically.
