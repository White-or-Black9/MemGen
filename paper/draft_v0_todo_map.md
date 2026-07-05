# Draft V0 TODO Map

This map covers every `TODO-Dxx` marker in `paper/draft_v0.md`. There are 13
unique TODOs.

| TODO | Draft location | Evidence needed | Planned source | Blocks |
|---|---|---|---|---|
| TODO-D01 | Abstract | Final explicit-text baseline comparison and valid method-separable cost interpretation | Text-summary, BM25 top-2, 16-token matched-budget, and cost experiments | Abstract |
| TODO-D02 | Introduction | Verified closest-neighbor citations and a defensible novelty boundary | Literature-verification pass; no experiment | Introduction positioning; non-blocking for experiment execution |
| TODO-D03 | Related Work | Verified bibliography covering latent memory, long-context reasoning, RAG/text memory, and conversational memory | Literature-verification pass; no experiment | Related Work |
| TODO-D04 | Method | Final method architecture and frozen-bank protocol figures with checked captions | Figure production from frozen method/protocol; no new result | Method presentation; non-blocking for claims |
| TODO-D05 | Evaluation Protocol | Frozen protocols and completed main-table rows for text summary, BM25 top-2, and 16-token matched budget | Three explicit-memory baseline experiments | Main table and benchmark-comparator description |
| TODO-D06 | Cost and Reproducibility | Separate construction, query, end-to-end latency, and peak GPU memory for Bank-off and P7, then final baselines | Method-separable cost smoke followed by full cost runs | Cost table; any abstract/intro efficiency statement |
| TODO-D07 | Experiments: Controls | Valid EM, recall, format, and token-budget results for the three explicit-memory controls | Text-summary, BM25 top-2, and 16-token matched-budget experiments | Main results and main table |
| TODO-D08 | Experiments: Ablation | P7 result with identical construction but query retrieval disabled and writes blocked | P7 no-query-retrieval ablation | Mechanism analysis and ablation table |
| TODO-D09 | Experiments: Packaging | One schema-consistent aggregate with source paths, scopes, repeats, scorer versions, and checksums | Unified final tables/manifest step | Final main table and reproducibility package |
| TODO-D10 | Analysis: Context | Unified verified per-context Bank-off/P6/P7 table and plotting input | Existing EventQA artifacts; aggregation only | Context analysis display; non-blocking for headline result |
| TODO-D11 | Analysis: Transitions | Final five-repeat helpful/harmful/unchanged transition counts | Existing authoritative EventQA summaries; extraction only | Analysis table; non-blocking for headline result |
| TODO-D12 | Limitations | LoCoMo prompt/protocol metadata, reliable counters, and representative failure rows in one appendix package | Existing LoCoMo audits; packaging only | Appendix; non-blocking for EventQA claim |
| TODO-D13 | Conclusion | Final evidence ordering after all comparator, ablation, cost, and manifest work | Resolution of TODO-D05 through TODO-D09 | Conclusion |

## Blocking TODOs

- Final abstract: TODO-D01.
- Final main table and comparator narrative: TODO-D05, TODO-D07, TODO-D09.
- Mechanism attribution: TODO-D08.
- Paper-facing cost claims: TODO-D06.
- Final conclusion: TODO-D13.
- Submission-quality positioning: TODO-D02 and TODO-D03.

## Non-Blocking TODOs For The Current Draft

- Method figures: TODO-D04.
- Context display packaging: TODO-D10.
- Transition-count packaging: TODO-D11.
- LoCoMo appendix packaging: TODO-D12.

## Recommended Resolution Order

1. TODO-D06: method-separable EventQA cost smoke.
2. TODO-D05 and TODO-D07: freeze and run explicit-memory baselines.
3. TODO-D08: P7 no-query-retrieval ablation.
4. TODO-D10 and TODO-D11: package existing analyses.
5. TODO-D09: build unified final tables and manifest.
6. TODO-D02 and TODO-D03: complete verified literature positioning.
7. TODO-D04 and TODO-D12: finish main figures and appendix packaging.
8. TODO-D01 and TODO-D13: revise the abstract and conclusion last.
