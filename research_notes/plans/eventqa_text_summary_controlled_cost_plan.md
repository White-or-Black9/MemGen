# EventQA Rolling-Summary Controlled Cost Supplement

## Goal

Replace the existing shared-GPU-confounded rolling-summary timing with one
paper-facing, method-separable EventQA-65536 cost measurement. This experiment
does not replace the existing effectiveness row and does not alter P7, prompts,
the parser, or the scorer.

## Locked Contract

- Dataset and scope: EventQA-65536, contexts `0..4`, questions `0..99` per
  context (`500` questions total).
- Method: the existing same-model rolling-summary baseline, 128-token cap,
  Bank-off query path, default EventQA prompt, unchanged local scorer/parser,
  and 40 generated answer tokens.
- Execution: one construction process followed by one query process per
  context, serialized on a single physical GPU. Before every context, the GPU
  must have no compute process and zero utilization; otherwise the run stops.
- Required evidence: raw preflight snapshots for all contexts, exact commands,
  construction/query manifests, five provenance-linked artifacts, and one
  controlled-cost attestation JSON.

## Hypothesis And Analysis

The experiment tests whether rolling-summary cost can be measured under the
same controlled single-GPU conditions as the paper-facing Bank-off/P7 cost
rows. It makes no directional latency hypothesis. The primary measurements are
construction, query, end-to-end, amortized seconds per question, and peak
incremental GPU memory. Effectiveness is reported only as a protocol check and
is not promoted to a new repeat estimate.

## Acceptance And Stop Rules

Accept only if all five contexts complete, all 500 identities are unique,
summary provenance and prompt-capacity checks pass, all cost values are finite,
and every preflight is clear. Stop without replacing the old caveat if any
preflight detects another compute process, a process fails, a summary/query
artifact mismatches, or the scorer/prompt/config drifts.

## Post-Run Update

Only after acceptance: aggregate with the controlled-cost attestation, update
the unified EventQA comparison package, and replace the manuscript's
shared-GPU caveat with the new protocol-specific cost description. Otherwise
retain the existing non-paper-facing conclusion.
