# P7 LoCoMo-QA Next-Step Decision After 2-Conversation Pilot

Date: 2026-07-05

## Route Question

After the completed 2-conversation LoCoMo-QA paired pilot, should we:

- scale directly to all 10 conversations, or
- fix protocol-adjacent prompt / diagnostics issues first?

## Decision

- Verdict: `do not scale yet`
- Action: `iterate`
- Reason: the pilot is protocol-valid but not performance-valid

## Decisive Evidence

### Strongest support for continuing LoCoMo

- frozen-bank P7 protocol completed end to end
- `query_write_count == 0` held for every P7 row
- P7 micro token F1 (`0.0598`) exceeds Disabled (`0.0269`)
- P7 reduces prompt-leak style outputs versus Disabled

### Strongest contradiction against immediate scaling

- EM is `0.0` for both modes on all `304` rows
- all paired exact-match cases are `both_wrong`
- prompt-leak remains high:
  - Disabled: `87 / 304`
  - P7: `31 / 304`
- P7 sometimes answers as if no context exists
- `run_diagnostics.json` construction accounting is still inconsistent with row-level counters

## Why the alternatives lose

### Rejected route: scale directly to all 10 conversations now

Rejected because it would mostly amplify ambiguous evidence:

- current score package is not benchmark-quality yet
- prompt / answer-format failures are still common
- exact-match remains zero everywhere
- cost accounting is not paper-ready

### Rejected route: abandon LoCoMo

Rejected because the protocol itself worked:

- the benchmark path is available locally
- the frozen P7 protocol is implementable
- P7 shows a directional token-F1 signal over Disabled
- this looks like a prompt / output-contract problem, not a fundamental benchmark incompatibility

## What the pilot currently proves

- LoCoMo-QA is usable as a MemGen-side benchmark path
- the frozen-bank P7 protocol works mechanically
- P7 can run over multi-question frozen-bank QA with query writes blocked

## What the pilot does not yet prove

- a credible performance comparison
- a scale-ready second main benchmark
- paper-facing cost or efficiency conclusions

## Required next fix

Highest priority:

- prompt / answer-format repair

Specifically audit:

- QA prompt construction
- query-only payload path
- answer extraction / postprocessing
- why P7 sometimes emits `no conversation history` answers
- why Disabled leaks instruction/prompt fragments so often

Secondary priority:

- diagnostics / construction-cost repair

This should be completed before:

- paper-facing cost tables
- efficiency claims
- formal latency/memory comparisons

## Next Stage

1. Run a read-only prompt/path audit.
2. Implement the smallest prompt/output-contract repair that removes or sharply reduces:
   - prompt-leak
   - no-context denial responses
3. Re-run a tiny paired validation slice:
   - `1` conversation
   - mixed categories
   - Disabled + P7
4. Only if EM becomes nonzero and leak frequency drops materially, reopen the decision to scale beyond 2 conversations.

## Evidence Paths

- `outputs/mab/locomo_qa_pilot_2conv_comparison.md`
- `outputs/mab/locomo_qa_pilot_2conv_comparison.json`
- `outputs/mab/locomo_qa_pilot_2conv_diagnosis.md`
- `outputs/mab/locomo_qa_pilot_2conv_diagnosis.json`

## Bottom Line

The current LoCoMo pilot is a successful protocol checkpoint, not a scale-ready benchmark result. The next correct move is prompt/path repair first, not full 10-conversation expansion.
