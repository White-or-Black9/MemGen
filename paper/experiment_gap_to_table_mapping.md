# Experiment Gap To Table Mapping

This plan maps each unresolved experiment to the exact paper claim and table it
would support. Smoke runs are validation gates, not paper results.

These experiments complete the current EventQA evidence package for the
long-horizon latent-memory-management outline; they do not narrow the paper
title or method motivation to EventQA.

## 1. Method-Separable Cost Smoke

Status: completed through the full five-context pass on 2026-07-06. The smoke
validated instrumentation; the full pass filled the Disabled/P7 Table 5 rows.

- Why needed: existing paired artifacts combine Bank-off and P7 timing and use
  a shared peak-memory maximum, so they cannot support per-method efficiency
  claims.
- Claim supported: the measured overhead of the session-local bank under a
  controlled EventQA protocol.
- Table filled: Table 5, Cost, Latency, And Memory.
- Minimum smoke: EventQA context 0, questions 0-9, separate Bank-off and P7
  processes with construction/query/end-to-end latency and peak GPU memory.
- Full version: all five contexts with identical hardware, environment,
  generation settings, and repeated measurement policy for every final method.
- Stop condition: combined-mode timing, incomparable environments, missing
  peak-memory reset/synchronization, nonzero P7 query writes, or schema mismatch.

## 2. Text-Summary Memory Baseline

- Why needed: tests whether ordinary readable compression explains the gain
  without latent storage.
- Claim supported: P7's contribution relative to a simple persistent textual
  memory rather than only relative to no bank.
- Tables filled: Table 1 and Table 6; cost fields later feed Table 5.
- Minimum smoke: context 0, questions 0-9, with a frozen summary-generation and
  query-injection contract logged in full.
- Full version: all five contexts and 500 questions under the final repeat and
  aggregation policy.
- Stop condition: summary model/prompt is not frozen, text budget is undefined,
  query-time state changes, or the baseline receives hidden extra context.

## 3. BM25 Top-2 Retrieved-Text Baseline

- Why needed: compares latent retrieval with a standard deterministic
  retrieved-text alternative.
- Claim supported: whether P7 remains useful when explicit relevant text can be
  retrieved at query time.
- Tables filled: Table 1, Table 6, and eventually Table 5.
- Minimum smoke: context 0, questions 0-9, top-2 BM25 chunks with retrieved IDs,
  scores, text, and injected token count recorded.
- Full version: all five contexts and 500 questions using the same source
  chunks, prompt, parser, and scorer as the final EventQA protocol.
- Stop condition: untraceable chunk IDs, uncontrolled truncation, missing token
  accounting, data leakage, or query-specific tuning.

## 4. 16-Token Matched-Budget Baseline

- Why needed: separates the value of latent memory from the value of adding any
  small amount of query-time evidence.
- Claim supported: P7's result is not explained solely by an extra evidence
  channel with approximately the same injected token count.
- Tables filled: Table 1 and Table 6; optional cost row in Table 5.
- Minimum smoke: context 0, questions 0-9, exactly documented 16-token retrieved
  text injection with deterministic truncation.
- Full version: all five contexts and 500 questions under the final baseline
  protocol.
- Stop condition: token budget is not actually matched, tokenizer accounting is
  inconsistent, or extra visible text enters outside the measured budget.

## 5. P7 No-Query-Retrieval Ablation

- Why needed: tests whether P7's query-time retrieval, rather than only
  construction side effects or prompt execution, contributes to the result.
- Claim supported: the retrieved latent bank is an active component of the
  EventQA gain.
- Tables filled: Table 6 and a compact ablation table or appendix component
  table.
- Minimum smoke: context 0, questions 0-9, same frozen construction and bank,
  with only query retrieval disabled and query writes still blocked.
- Full version: all five contexts and 500 questions using the same frozen P7
  construction artifacts or exactly matched reconstruction.
- Stop condition: construction differs from P7, other thresholds change,
  query-time writes occur, or the ablation requires model retraining.

## 6. Unified Final Tables And Manifest

- Why needed: prevent mixing incompatible historical rows and make every paper
  number traceable to a frozen artifact.
- Claim supported: reproducibility and internal consistency of all reported
  EventQA comparisons.
- Tables filled: all final paper and appendix tables.
- Minimum version: schema/provenance dry run over existing Bank-off, P6, and P7
  rows, including method name, scope, sample count, repeats, metrics, and source
  paths.
- Full version: aggregate every final method, emit table-ready JSON/Markdown,
  and record artifact checksums and protocol metadata.
- Stop condition: missing methods, inconsistent sample/question scopes,
  incompatible scorer/prompt versions, unresolved duplicate rows, or failed
  checksum/schema validation.

## Recommended Execution Order

1. Method-separable Bank-off/P7 cost smoke.
2. Freeze the shared explicit-memory baseline protocol.
3. Text-summary baseline.
4. BM25 top-2 retrieved-text baseline.
5. 16-token matched-budget baseline.
6. P7 no-query-retrieval ablation.
7. Unified final tables and artifact manifest.

The cost stage is complete for Disabled/P7. Reuse the validated accounting
contract for every later explicit-text baseline.
