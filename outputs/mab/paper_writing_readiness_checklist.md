# Paper Writing Readiness Checklist

Date: 2026-07-05

## Ready For Paper

- [x] Working title and scoped claim.
- [x] Frozen P7 method definition.
- [x] EventQA benchmark and frozen-context protocol.
- [x] EventQA main P7 effectiveness result.
- [x] P7/P6 five-repeat comparison.
- [x] Bank-off effectiveness result.
- [x] Context-wise analysis.
- [x] Helpful/harmful transition analysis.
- [x] Prompt and format-failure analysis.
- [x] Context-4 limitation.
- [x] LoCoMo limitation interpretation.
- [x] Claim boundary and unsupported-claim list.

## Not Ready Yet

- [ ] Final EventQA main table.
- [ ] Method-separable Bank-off/P7 cost row.
- [ ] Text-summary baseline.
- [ ] BM25 top-2 RAG baseline.
- [ ] 16-token matched-budget baseline.
- [ ] P7 no-query-retrieval ablation.
- [ ] Final unified artifact manifest/table.
- [ ] Cost-effectiveness figure.

## Drafting Gate

Can draft now:

- Introduction with scoped claim;
- Method;
- Benchmark and protocol;
- existing EventQA result subsection;
- failure analysis;
- Limitations;
- appendix protocol and diagnostic material.

Must wait:

- final main comparison table prose;
- explicit-text baseline conclusions;
- efficiency/cost claims;
- final Abstract numbers if the new baseline table changes the emphasis;
- final Conclusion wording.

## Next Gate

Pass a method-separable EventQA cost smoke on context 0, q0-9, Disabled and P7,
then proceed to BM25 top-2 RAG on the same slice.
