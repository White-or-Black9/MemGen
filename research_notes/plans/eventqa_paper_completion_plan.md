# EventQA Evidence Completion Plan

Date: 2026-07-05

## Decision

Use EventQA as the main current positive evidence for **Inference-Time Latent
Memory Management for Long-Horizon LLM Agents**. Reuse the completed P7/P6
five-repeat effectiveness evidence. Do not rerun those rows by default.

This plan completes the EventQA evidence package; it does not redefine the
paper goal or title as EventQA-specific.

## Claim Boundary

Supported:

- frozen P7 improves the local MemoryAgentBench EventQA-65536 result over compressed Bank-off;
- P7 has higher EM and fewer format failures than P6, with comparable recall;
- session-local frozen-bank construction, retrieval, and query write blocking operate correctly.

Not supported:

- benchmark-general long-context improvement;
- superiority to text summaries, RAG, or matched-budget explicit text;
- paper-facing cost superiority.

## Trusted Existing Rows

- P7: five repeats, main row.
- P6: five repeats, closest threshold comparator.
- Bank-off: effectiveness only; cost not separable from paired artifacts.
- P4: two repeats, appendix.
- Strict and first-line prompt variants: negative appendix ablations.
- Context-wise, transition, format, context-4, and harmful-attribution analyses: reusable with stated caveats.

## Blocking Evidence Gaps

1. Method-separable Bank-off/P7 latency and peak GPU memory.
2. Text-summary memory baseline.
3. BM25 top-2 retrieved-text/RAG baseline.
4. Query-position-matched explicit-text baseline at 16 injected text tokens.
5. P7-no-query-retrieval component ablation.
6. Unified effectiveness, cost, and appendix aggregation.

## Execution Frontier

1. Implement a no-inference final aggregator and freeze the output schema.
2. Add separate construction/query/end-to-end cost logging.
3. Run a context-0, q0-9 Disabled/P7 cost smoke.
4. Run the full standalone cost pass after smoke acceptance.
5. Implement and smoke BM25 top-2 RAG on the same q0-9 slice.
6. Run RAG, matched-budget RAG, and text-summary rows.
7. Run P7-no-query-retrieval.
8. Assemble final tables and claim audit.

## First GPU Experiment

Run only `context_index=0`, questions `0..9`, Disabled and frozen P7, with separate per-method cost accounting. Keep the official non-strict prompt/scorer and all P7 parameters unchanged. Stop if costs are combined, query writes are nonzero, bank state changes, or schemas diverge.

## Optional LoCoMo Placement

If retained, place LoCoMo only in Limitations or appendix diagnostic evidence.
It is not required by the outline. State that construction and retrieval were
protocol-correct but exact QA remained at zero EM and frequent
no-context/refusal behavior persisted.
