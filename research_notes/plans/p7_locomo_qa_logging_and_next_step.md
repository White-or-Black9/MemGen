# P7 LoCoMo-QA Logging And Next-Step Decision

## Decision
Do not run more P7 LoCoMo scaling yet.

## Why
The full audit shows:
- protocol is correct,
- construction is happening,
- query retrieval is happening,
- query-time writes are blocked,
- but construction-side diagnostics are only partially connected.

Most importantly:
- reliable fields already support the main negative conclusion,
- unreliable fields mostly affect logging quality and cost/mechanism claims,
- they do not change the conclusion that LoCoMo currently fails at answer utilization.

## What Must Be Fixed Before Any Future LoCoMo Rerun
1. Persist prompt metadata:
   - `system_prompt_template`
   - `construction_prompt_template`
   - `qa_prompt_template`
   - `prompt_hash`
   - `rendered_prompt_preview`

2. Repair construction-only diagnostics propagation:
   - construction trigger stats
   - construction weaver stats
   - construction latency
   - construction peak GPU memory

3. Clarify counter semantics:
   - current `trigger_call_count` and `weaver_call_count` are query-side token-level trigger-mask counts, not literal function call counts
   - rename or split them

4. Separate cost accounting:
   - P7 construction cost
   - P7 query cost
   - Disabled per-question full-loop cost

## What Does Not Need Immediate Repair
- Deterministic scorer
- row-level cleaned/raw answer contract
- query write blocking
- bank snapshot integrity checks
- session-level construction granularity itself

## Recommended Next Experiment
Run a `retrieved-text / RAG sanity baseline` before more P7 LoCoMo.

Reason:
- reliable diagnostics already show latent retrieval is active,
- if explicit retrieved text still fails, LoCoMo may simply be a poor fit for the current setup,
- if explicit retrieved text helps substantially, the problem is much more likely latent-memory usability plus prompt/context mismatch.

## Benchmark Status
Current status for LoCoMo-QA:
- keep as a diagnostic benchmark,
- do not treat as a paper-facing second main benchmark yet.

## Supported Conclusions From Reliable Fields Only
- Construction is not missing.
- Query retrieval is not missing.
- Query-time bank mutation is not happening.
- Output extraction is not the primary blocker.
- Prompt/context mismatch plus latent-memory usability failure remain the leading diagnosis.
