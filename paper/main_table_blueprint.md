# Main Table Blueprint

Status: paper-planning blueprint. Missing rows are labeled as experiments
required; no numeric placeholder should be interpreted as a result.

## Table 1. Main EventQA Effectiveness

- Purpose: establish the primary EventQA-65536 comparison under one frozen-bank
  evaluation contract.
- Planned rows:
  - Disabled / Bank-off;
  - text-summary memory baseline;
  - BM25 top-2 retrieved-text baseline;
  - 16-token matched-budget retrieved-text baseline;
  - P6;
  - P7 no-query-retrieval;
  - frozen P7.
- Planned columns: method, memory representation, retrieval form, repeat count,
  EM mean and dispersion, recall mean and dispersion, format failures, and
  protocol notes.
- Existing available numbers:
  - Bank-off: EM `0.008+-0.000`, recall `0.178+-0.000`, format failures
    `377.0+-0.0`.
  - text-summary: EM `0.012`, recall `0.078`, format failures `267/500`.
  - BM25 top-2: EM `0.030`, recall `0.226`, format failures `265/500`.
  - matched16: EM `0.068`, recall `0.180`, format failures `347/500`.
  - P6: EM `0.169+-0.018`, recall `0.258+-0.016`, format failures
    `165.8+-19.8`.
  - P7 no-query-retrieval: EM `0.008`, recall `0.178`, format failures
    `377/500`.
  - P7: EM `0.197+-0.020`, recall `0.254+-0.028`, format failures
    `121.4+-8.8`.
- Missing numbers: none for the unified EventQA comparison package. Single-pass
  rows must stay labeled as one-pass controls rather than repeated main rows.
- Placement: main paper.
- Ready now: yes. The unified package can directly render the final comparison
  rows with repeat-count caveats.

## Table 2. P7 Versus P6

- Purpose: isolate the final frozen-method selection and show the
  effectiveness/format trade-off.
- Planned rows: P6 and P7, plus a delta row.
- Planned columns: update threshold, repeat count, EM, recall, format failures,
  helpful transitions, harmful transitions, and context spread.
- Existing available numbers:
  - P6 EM `0.169+-0.018`, recall `0.258+-0.016`, format failures
    `165.8+-19.8`.
  - P7 EM `0.197+-0.020`, recall `0.254+-0.028`, format failures
    `121.4+-8.8`.
  - P7 minus P6: EM `+0.0280`, recall `-0.0044`, format failures `-44.4`.
- Missing numbers: none for the core comparison; helpful/harmful transition
  columns should be copied only from the verified five-repeat summaries.
- Placement: main paper or compact ablation table.
- Ready now: yes for the core metrics.

## Table 3. Context-Wise Breakdown

- Purpose: expose heterogeneity across the five EventQA contexts rather than
  presenting only aggregate gains.
- Planned rows: contexts 0 through 4.
- Planned columns: Bank-off EM/recall, P6 EM/recall, P7 EM/recall, P7 format
  failures, and P7-minus-Bank-off/P6 deltas.
- Existing available numbers: five-repeat context-wise P6/P7 summaries and the
  Bank-off context breakdown; context 4 P7 has EM `0.006`, recall `0.228`, and
  format failures `93.8/100`.
- Missing numbers: none; the unified context-wise export is complete.
- Placement: main paper if space permits; full version in appendix.
- Ready now: packaged in the manuscript analysis table.

## Table 4. Format-Failure Analysis

- Purpose: show that P7's gain is accompanied by fewer malformed or
  parser-incompatible answers and distinguish format effects from reasoning.
- Planned rows: Bank-off, P4 where comparable, P6, P7, strict-prompt ablation,
  and first-line-prompt ablation.
- Planned columns: total format failures, failures per context, strict/first-line
  prompt setting, and notes on parser compatibility.
- Existing available numbers: P7, P6, Bank-off, P4, and prompt-ablation
  artifacts identified in the EventQA inventory.
- Missing numbers: none for a descriptive appendix table; final provenance and
  comparator-scope checks remain necessary before mixing historical rows.
- Placement: compact main-paper analysis plus full appendix table.
- Ready now: yes after unified formatting.

## Table 5. Cost, Latency, And Memory

- Purpose: quantify the inference cost of persistent latent memory relative to
  no-bank and explicit-text alternatives.
- Planned rows: Bank-off, P6, P7, text summary, BM25 top-2, and 16-token
  matched-budget retrieval.
- Planned columns: construction latency, query latency, end-to-end latency,
  peak GPU memory, Trigger calls, Weaver calls, output tokens, injected text
  tokens, and CPU bank-size estimate when available.
- Existing method-separable numbers for the full five-context pass:
  - Disabled: construction `0.000 s`, query `0.735+-0.207 s/question`,
    end-to-end `367.448 s`, amortized `0.735 s/question`, max incremental peak
    allocation `142.9 MiB`;
  - P7: construction `78.454 s`, query `0.619+-0.175 s/question`, end-to-end
    `387.999 s`, amortized `0.776 s/question`, max incremental peak allocation
    `171.9 MiB`;
  - P7/Disabled end-to-end ratio `1.056`, max incremental peak delta about
    `29.0 MiB`.
- Missing numbers: no additional numbers for the unified package. However,
  same-model text-summary cost remains non-paper-facing because it was measured
  under shared-GPU contention, and cross-method cost claims must respect that
  caveat.
- Placement: main paper if measurements become comparable; detailed accounting
  in appendix.
- Ready now: yes with caveats. Disabled/P7/BM25/matched16/no-query can be
  reported directly; text-summary should be labeled diagnostic-only in cost
  views. Do not present the lower P7 query mean as a throughput claim.

## Table 6. Explicit-Memory And Budget Controls

- Purpose: rule out the explanations that any persistent text, any retrieval,
  or merely injecting a small token budget is sufficient.
- Planned rows: Bank-off, text-summary memory, BM25 top-2 retrieved text,
  16-token matched-budget text, P7 no-query-retrieval, and P7.
- Planned columns: evidence available at query time, injected token count, bank
  capacity/budget, EM, recall, format failures, and cost fields where valid.
- Existing available numbers:
  - Bank-off: EM `0.008`, recall `0.178`, format failures `377/500`.
  - text-summary: EM `0.012`, recall `0.078`, format failures `267/500`.
  - BM25 top-2: EM `0.030`, recall `0.226`, format failures `265/500`.
  - matched16: EM `0.068`, recall `0.180`, format failures `347/500`.
  - P7 no-query-retrieval: EM `0.008`, recall `0.178`, format failures
    `377/500`.
  - P7: EM `0.197+-0.020`, recall `0.254+-0.028`, format failures
    `121.4+-8.8`.
- Missing numbers: none for the effectiveness control table. Cost cells must
  carry the text-summary non-paper-facing caveat where shown.
- Placement: main paper; may be merged into Table 1 if column width permits.
- Ready now: yes.

## Optional Table 7. LoCoMo Diagnostic

- Purpose: document the limitation of latent-only memory for exact open-ended
  multi-session conversational fact recovery.
- Planned rows: Disabled and P7 session-level construction.
- Planned columns: conversations, questions, EM, token F1, invalid outputs,
  prompt leak, no-context denial, refusal, retrieval-active rows,
  retrieved-latent count, and query writes.
- Existing available numbers:
  - 2 conversations and 304 questions per mode;
  - Disabled EM `0`, F1 `0.01834`;
  - P7 EM `0`, F1 `0.02084`;
  - all 304 paired rows are exact-match wrong;
  - P7 no-context denial `138/304`, refusal `153/304`;
  - P7 retrieval is active, retrieves 16 latent vectors, and has zero query
    writes for every row.
- Missing numbers: none for the current diagnostic claim. Existing inconsistent
  LoCoMo cost counters must be excluded.
- Placement: Appendix A with a short limitations reference.
- Ready now: packaged as negative diagnostic evidence, not as a positive
  result.

## Readiness Summary

- Ready from existing evidence: main EventQA effectiveness, P7 versus P6,
  explicit-memory controls, context-wise results, format-failure analysis, and
  the optional LoCoMo diagnostic.
- Conditionally ready: cost table, with text-summary cost restricted to
  diagnostic/appendix use.
- Packaging remaining: convert the unified package into final paper tables and
  figure assets.
