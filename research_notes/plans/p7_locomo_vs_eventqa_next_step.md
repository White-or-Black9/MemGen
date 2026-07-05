# P7 LoCoMo vs EventQA Next-Step Decision

Date: 2026-07-05

## Decision

Keep EventQA as the current positive long-context anchor. Keep LoCoMo-QA as a negative diagnostic, not a positive second main benchmark. Do not scale P7 LoCoMo under the current latent-only query contract.

## Evidence Boundary

- EventQA: five-repeat P7 evidence supports improvement under the official local EventQA prompt/parser/scorer and frozen-context-bank protocol.
- LoCoMo: session-level construction, retrieval, frozen-bank reuse, and write blocking are validated, but answer utilization is not.
- The two benchmarks use the same frozen P7. The result gap is dominated by query evidence form and task alignment, not a method/config difference.
- EventQA displays prior events and all six candidate answers at query time. LoCoMo displays only an open question and requires exact latent-to-fact decoding.

## Highest-Priority Next Diagnostic

Run a retrieved-text/RAG sanity baseline on the same LoCoMo two-conversation, 304-question slice.

Keep fixed:

- normalized adapter records;
- conversation/question selection;
- QA prompt and output cleanup where possible;
- deterministic EM/F1 scorer;
- Disabled and P7 artifacts as comparison references;
- no GPT judge.

The RAG run must expose retrieved dialogue text at query time and log retrieved session/turn IDs plus injected token count. Its purpose is diagnostic, not to replace P7.

Decision rule:

- If RAG materially improves exact match/F1 and reduces no-context denial, the primary bottleneck is latent-memory usability/evidence form.
- If RAG also fails, the broader prompt/task/generation contract is not viable and LoCoMo should be deferred.
- Only if RAG succeeds should a prompt-only latent diagnostic or turn-level construction diagnostic be considered.

## Logging Gate Before The Run

Add forward-looking metadata before any new LoCoMo inference:

- system, construction, and QA prompt templates;
- prompt version and hashes;
- bounded rendered-prompt previews;
- corrected construction Trigger/Weaver trace propagation;
- separate construction and query latency/memory accounting;
- explicit semantics for token-level trigger counters versus actual function calls.

These logging fixes improve auditability. They do not alter the frozen P7 method or current failure conclusion.

## Parallel Paper Work

After the bounded RAG sanity decision, return to EventQA missing rows:

1. comparable Disabled/Bank-off latency and peak-memory row;
2. text-summary baseline;
3. RAG/retrieved-text baseline;
4. matched-budget baseline;
5. unified final aggregation table.

## Paper Claim Boundary

Current safe claim: P7 improves EventQA long-context next-event reasoning under the tested frozen-bank protocol.

Current limitation claim: P7 construction and retrieval operate on LoCoMo, but latent-only retrieval does not yet support reliable exact multi-session conversational fact recovery.

Do not claim general multi-turn improvement until a deterministic conversational benchmark produces positive paired evidence.
