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
  - frozen P7.
- Planned columns: method, memory representation, retrieval form, repeat count,
  EM mean and dispersion, recall mean and dispersion, format failures, and
  protocol notes.
- Existing available numbers:
  - Bank-off: EM `0.008`, recall `0.178`; representative format failures
    `377/500`.
  - P6: EM `0.169+-0.018`, recall `0.258+-0.016`, format failures
    `165.8+-19.8`.
  - P7: EM `0.197+-0.020`, recall `0.254+-0.028`, format failures
    `121.4+-8.8`.
- Missing numbers: all text-summary, BM25, and matched-budget results; a
  consistently aggregated Bank-off dispersion row if the final table requires
  repeated Bank-off statistics.
- Placement: main paper.
- Ready now: no. The P6/P7/Bank-off block is reusable, but the explicit-memory
  comparator rows are missing.

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
- Missing numbers: no new experiment is required, but a single unified export
  must be built from the authoritative repeat artifacts.
- Placement: main paper if space permits; full version in appendix.
- Ready now: evidence-ready, packaging pending.

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
- Existing available numbers: exploratory paired timing artifacts only. They
  combine Bank-off and Bank-on execution and use a shared maximum peak-memory
  value, so they are not valid per-method paper rows.
- Missing numbers: all method-separable latency and peak-memory rows; explicit
  baseline costs.
- Placement: main paper if measurements become comparable; detailed accounting
  in appendix.
- Ready now: no.

## Table 6. Explicit-Memory And Budget Controls

- Purpose: rule out the explanations that any persistent text, any retrieval,
  or merely injecting a small token budget is sufficient.
- Planned rows: Bank-off, text-summary memory, BM25 top-2 retrieved text,
  16-token matched-budget text, P7 no-query-retrieval, and P7.
- Planned columns: evidence available at query time, injected token count, bank
  capacity/budget, EM, recall, format failures, and cost fields where valid.
- Existing available numbers: Bank-off and P7 only.
- Missing numbers: text summary, BM25, matched-budget, and no-query-retrieval.
- Placement: main paper; may be merged into Table 1 if column width permits.
- Ready now: no.

## Table 7. LoCoMo Diagnostic

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
- Placement: appendix and a short limitations reference only.
- Ready now: yes as negative diagnostic evidence, not as a positive result.

## Readiness Summary

- Ready from existing evidence: P7 versus P6, context-wise results,
  format-failure analysis, and LoCoMo diagnostic.
- Partially ready: main EventQA effectiveness table.
- Not ready: full baseline comparison and method-separable cost table.
