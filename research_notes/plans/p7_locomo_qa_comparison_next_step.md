# P7 LoCoMo-QA Comparison Next Step

## Decision

- The session-level LoCoMo pilot is protocol-valid but not a usable main-benchmark result.
- Do not scale LoCoMo-QA further under the current latent-memory-only answer contract.

## Evidence

- session-level P7 fixed construction coverage: chunk/write/slot = `19 / 19 / 16` instead of repaired token_chunk `3 / 3 / 3`
- despite that, session-level P7 micro F1 fell to `0.0208` from repaired token_chunk `0.0604`
- no-context denial rose from `31` to `138` rows
- refusal rose from `32` to `153` rows
- retrieval remained active and query-time writes stayed blocked, so the failure is not missing retrieval or protocol breakage

## Interpretation

- The dominant failure mode is latent-memory usability / prompt-context mismatch at QA time.
- The current LoCoMo result should be treated as diagnostic evidence, not as a second main benchmark result for the paper.

## Exact Next Step

- Implement and run a retrieved-text/RAG sanity baseline on the same 2-conversation LoCoMo slice.
- Keep everything else fixed: same normalized adapter outputs, same questions, same scorer, same output-contract flags.
- Use that baseline to answer one decisive question: can the model answer these LoCoMo questions when explicit retrieved text is injected?
- If yes, the bottleneck is latent-memory usability. If no, the bottleneck is the QA prompt/task contract more broadly.

## Deferred Until After RAG Sanity

- turn-level construction diagnostic
- larger LoCoMo scaling
- paper-facing LoCoMo cost claims
- replacing EventQA priority work
