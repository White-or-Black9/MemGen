# EventQA Harmful Memory Attribution

## 1. Purpose

P7 shows a severe failure mode on EventQA context 4. This diagnostic asks
whether that failure can be attributed to a small number of memory slots or to
an ordered retrieved tuple. It serves the unchanged project objective,
**Session-Local Latent Memory Bank / Latent Memory Bank Improves Long-Context
Reasoning**, by identifying a concrete memory-policy failure that a later
non-oracle method may avoid.

This is not a downgrade to a diagnostic-only paper and is not a final
performance conclusion. The current experiment is an oracle counterfactual
analysis of one frozen P7 bank.

## 2. Experimental Setup

- Source run:
  `outputs/mab/eventqa_p7_rt005_ut010_cap16_topk2/20260702T084825Z-eventqa-65536-version-b-weaver-space-bank-n5`
- Frozen bank:
  `outputs/mab/eventqa_p7_rt005_ut010_cap16_topk2/20260702T084825Z-eventqa-65536-version-b-weaver-space-bank-n5/frozen_banks/context_4.pt`
- Frozen-bank SHA-256:
  `ec755739057d2b66e0993e1771c8f3406b0a8803791e0de89dd1de69d3df6463`
- Context: `4`
- Smoke questions: `0..9`
- Expanded questions: `0..99`
- Conditions: `full`, `drop-slot:0`, `drop-slot:1`,
  `drop-tuple:1,0`, `slot-only:0`, `slot-only:1`, and
  `tuple-only:1,0`.
- The official EventQA scorer/parser was reused without modification.
- Every counterfactual question starts from a pristine clone of the frozen
  bank; the source `.pt` is read-only.
- Attribution conditions run only after exact full-bank replay validation.
- Diagnostic implementation:
  `scripts/eval/eventqa_harmful_memory_attribution.py`
- Focused tests:
  `tests/test_eventqa_harmful_memory_attribution.py`
- Expanded artifact:
  `outputs/mab/eventqa_harmful_memory_attribution_context4_full/20260704T001824Z-p7-context4-q0-99/`

The forced `slot-only` and `tuple-only` conditions are explicitly marked
`oracle_diagnostic` in `run_config.json`. They are not deployable inference
policies.

## 3. Smoke q0-9 Result

The completed smoke is:
`outputs/mab/eventqa_harmful_memory_attribution_smoke/20260704T001049Z-p7-context4-q0-9/`.
Its full-bank replay matched all 10 questions. The earlier sibling directory
`20260704T001015Z-p7-context4-q0-9` is a preflight-only artifact and is not a
failed result.

| Condition | EM | Recall | No-gold | Format failure | Rescues |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 0/10 | 0.20 | 8/10 | 10/10 | 0 |
| drop-slot:0 | 1/10 | 0.40 | 6/10 | 7/10 | 1 |
| drop-slot:1 | 1/10 | 0.50 | 5/10 | 7/10 | 1 |
| drop-tuple:1,0 | 3/10 | 0.30 | 7/10 | 1/10 | 3 |
| slot-only:0 | 3/10 | 0.30 | 7/10 | 3/10 | 3 |
| slot-only:1 | 5/10 | 0.50 | 5/10 | 0/10 | 5 |
| tuple-only:1,0 | 0/10 | 0.20 | 8/10 | 10/10 | 0 |

This small smoke gave the first counterfactual signal of a tuple-level harmful
interaction. It remained `n=10` exploratory evidence until the q0-99
expansion.

Evidence:
`outputs/mab/eventqa_harmful_memory_attribution_smoke/20260704T001049Z-p7-context4-q0-9/replay_validation.json`
and
`outputs/mab/eventqa_harmful_memory_attribution_smoke/20260704T001049Z-p7-context4-q0-9/attribution_summary.json`.

## 4. Expanded q0-99 Result

The expanded run passed the replay gate for all 100 questions. Per-field
validation matched `official_em`, recall, retrieved original slot IDs, raw
prediction hash, parsed prediction, and format flags for `100/100` questions.

| Condition | EM | Recall | No-gold | Format failure | Rescues | Net rescue |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0/100 | 0.19 | 81/100 | 98/100 | 0 | 0 |
| drop-slot:0 | 1/100 | 0.34 | 66/100 | 89/100 | 1 | +1 |
| drop-slot:1 | 3/100 | 0.14 | 86/100 | 70/100 | 3 | +3 |
| drop-tuple:1,0 | 15/100 | 0.15 | 85/100 | 2/100 | 15 | +15 |
| slot-only:0 | 30/100 | 0.31 | 69/100 | 35/100 | 30 | +30 |
| slot-only:1 | 26/100 | 0.30 | 70/100 | 15/100 | 26 | +26 |
| tuple-only:1,0 | 0/100 | 0.19 | 81/100 | 98/100 | 0 | 0 |

Evidence:
`outputs/mab/eventqa_harmful_memory_attribution_context4_full/20260704T001824Z-p7-context4-q0-99/replay_validation.json`,
`attribution_summary.json`, `attribution_per_context.json`, and
`attribution_per_question.jsonl` under the same directory.

## 5. Main Finding

- Full-bank replay matched `100/100`, so the counterfactual comparison is
  anchored to the original P7 output for this bank.
- The full bank selected ordered tuple `[1,0]` on `100/100` questions.
- `tuple-only:1,0` reproduced the full-bank aggregate collapse exactly:
  EM `0/100`, recall `0.19`, no-gold `81/100`, and format failure `98/100`.
- `slot-only:0` and `slot-only:1` were substantially cleaner and more accurate
  than `tuple-only:1,0`.
- Dropping ordered tuple `[1,0]` reduced format failures from `98/100` to
  `2/100` and increased EM from `0/100` to `15/100`.
- The evidence therefore supports an ordered tuple-level harmful interaction
  in this single frozen P7 context-4 bank.

It does **not** prove general long-context improvement, and it does not show
that slot 0 or slot 1 is individually harmful. It is oracle diagnostic
evidence, not a non-oracle inference-time method. The drop-tuple condition also
left no-gold at `85/100` and recall at `0.15`, so eliminating the format
collapse does not resolve the remaining content-grounding failure.

## 6. Interpretation

The failure is more consistent with ordered-tuple interaction and routing
dominance than with one universally toxic slot. Each slot alone can retain
useful information, while their ordered joint injection reproduces the full
collapse. A future memory policy should therefore avoid harmful combinations
without disabling memory or discarding useful single-slot information.

Possible later directions include top-1 fallback, an injection budget, tuple
dominance suppression, a score-margin gate, and query-construction redesign.
None is implemented or approved by this diagnostic.

## 7. Limitations

- one frozen bank from one P7 source run;
- context 4 only;
- questions `0..99` only;
- oracle counterfactual diagnostic;
- no cross-repeat attribution evidence;
- no non-oracle policy;
- not final paper-level performance evidence;
- no utility gate implemented;
- further attribution expansion is currently paused.

## 8. Current Status

The harmful-attribution line is summarized and its initial q0-99 study is
complete. No further attribution experiment is scheduled now, and no utility
gate has been implemented. The final project goal remains unchanged:
**Latent Memory Bank Improves Long-Context Reasoning**. The next work returns
to project-note consolidation and paper-preparation planning, not paper
writing.
