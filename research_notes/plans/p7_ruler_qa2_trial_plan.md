# P7 RULER-QA2 Trial Plan

Date: 2026-07-10

## Outcome Update

- Status: executed and closed on 2026-07-10.
- Outcome:
  the adapted frozen-bank runner is valid and the full `100`-query `p7` run is
  complete, but query-time retrieval stays `0/100` and accuracy is only
  `8/100`.
- Decision:
  stop `RULER-QA2` for the current paper cycle and keep the artifacts as
  internal negative feasibility evidence only.

## Role

`RULER-QA2` is the first additional benchmark attempt after the EventQA mainline.

## Claim Scope

`RULER-QA2` can support only a long-context retrieval and reuse stress claim. It
does not support a conversational or multi-session memory claim by itself.

## Local Evidence

- local MemoryAgentBench availability audit records `ruler_qa2_421K: 1` context;
- the stored metric contract is `substring_exact_match`;
- prior benchmark-replacement audit already marked `RULER-QA2` as the preferred
  second candidate because it is deterministic and local;
- current bridge utility exposes `prepare` and `score` subcommands, so a
  MemGen-side adapter can reuse the existing MAB conversion path.

## Protocol

- one `RULER-QA2` context maps to one MemGen session;
- construction ingests the context sequentially;
- the memory bank is frozen before answering queries;
- query-time writes are blocked;
- the bank resets after the context finishes;
- `disabled`, `p7`, and `p7_no_query_retrieval` must answer the identical query
  list.

## Metrics

- `substring_exact_match`
- total queries
- correct count
- memory write count
- retrieval count
- retrieved latent count
- latency
- peak GPU memory if the runtime already logs it

## Smoke Gate

The smoke passes only if:

- the scorer is deterministic;
- all modes emit the same query IDs;
- enabled modes show nonzero construction writes;
- `p7` does not regress versus `disabled`;
- output JSON records exact config and artifact paths.

## Stop Gate

Stop `RULER-QA2` if:

- local data cannot be prepared through the existing MAB bridge;
- scoring cannot be reproduced without manual judgment;
- `disabled` and enabled modes cannot be aligned on the same query set;
- a 1-context smoke already fails cleanly with no protocol ambiguity.
