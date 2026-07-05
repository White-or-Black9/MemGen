# P7 LoCoMo-QA Prompt Inspection: Next-Step Decision

## Decision
Prioritize a `retrieved-text / RAG sanity baseline` before a prompt-only diagnostic.

## Why
- The exact QA prompt is:

```text
Based on the conversation history you memorized, answer the question concisely.

Question: {question}

Answer:
```

- At query time, both `Disabled` and `P7` expose only:
  - the shared system prompt
  - the QA user prompt

- Query-time visible prompt text does **not** include explicit conversation text.
- Query-time visible prompt text does **not** explain that relevant memory may arrive through latent retrieval rather than visible text.
- This makes prompt-induced `no_context_denial` plausible.

However:
- prompt wording alone does not isolate whether latent retrieval is intrinsically unusable for this benchmark,
- while a retrieved-text baseline would directly test whether the model can answer LoCoMo when relevant context is surfaced explicitly.

## Confirmed Prompt Facts
- `Disabled` and `P7` use the same QA prompt.
- `Disabled` and `P7` use the same construction instruction wording.
- `session` and `token_chunk` use the same wording; only construction granularity changes.
- Completed run artifacts do not save rendered prompts or prompt hashes.

## Immediate Next Step
Implement or plan a `retrieved-text / RAG sanity baseline` for LoCoMo-QA with clear diagnostic labeling:
- same normalized dataset contract
- same deterministic scorer
- explicit retrieved text visible at QA time
- no method change to frozen P7
- use as a diagnostic baseline, not as a replacement for P7

## Follow-up After RAG Sanity
If retrieved-text visibly improves answerability:
- run a prompt-only diagnostic that explicitly states memory may have been previously memorized and may not be shown as full visible history,
- keep it diagnostic only,
- do not treat it as a P7 method change.

## Logging Recommendation
Before any further LoCoMo rerun, add prompt metadata logging:
- `construction_prompt_template`
- `qa_prompt_template`
- `prompt_hash`
- `prompt_version_name`
- `rendered_prompt_preview`
