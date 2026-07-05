# Figure Blueprint

Status: figure planning only. No figure has been rendered or approved.

## Figure 1. Method Architecture

- Purpose: explain where the session-local bank sits in frozen MemGen.
- Should show:
  - frozen Trigger, Weaver, and Reasoner;
  - construction-time Weaver-space memory writes;
  - bounded session-local bank with retrieval, matched update, replacement,
    and reset;
  - query-time retrieval into the Reasoner;
  - blocked query-time writes.
- Source data: `research_notes/METHOD.md` and the frozen P7 definition.
- Readiness: conceptually ready; no experimental data required.
- Risks: the visual must not imply retraining, cross-session sharing, a utility
  gate, tuple suppression, or top-1 fallback.

## Figure 2. Frozen-Bank Protocol

- Purpose: make construction/query separation and bank ownership explicit.
- Should show:
  1. reset one bank for one EventQA context;
  2. ingest ordered context chunks;
  3. freeze the bank snapshot;
  4. restore/reuse that snapshot for each question;
  5. permit retrieval and assert zero query writes.
- Source data: EventQA runner protocol, prompt/scorer verification, and method
  notes.
- Readiness: conceptually ready.
- Risks: distinguish construction-time retrieval/update from query-time
  retrieval; do not imply that visible full context is injected at query time.

## Figure 3. EventQA Main Result

- Purpose: communicate the repeated positive EventQA result and P7/P6
  difference at a glance.
- Should show: Bank-off, P6, and P7 EM with repeat dispersion; optionally pair
  EM with recall or format failures in a second panel.
- Source data: EventQA five-repeat stability summary and P7-versus-P6 final
  summary.
- Readiness: ready for the current three-method result; a full comparator figure
  must wait for summary/RAG/matched-budget rows.
- Risks: Bank-off repeat treatment must be labeled accurately; avoid a dual-axis
  design that obscures recall or format-failure trade-offs.

## Figure 4. Context-Wise Variance And Failure

- Purpose: show that aggregate improvement is heterogeneous and that context 4
  remains a severe limitation.
- Should show: per-context P6/P7/Bank-off EM and recall, with a clearly marked
  context-4 panel or annotation; format failures may be a separate aligned
  panel.
- Source data: five-repeat context-wise artifacts and context-4 diagnosis.
- Readiness: data ready; unified plotting input still needs packaging.
- Risks: do not cherry-pick context 4 without showing all five contexts; use
  consistent axes and report repeat dispersion where available.

## Figure 5. Optional LoCoMo Limitation

- Purpose: contrast mechanically active retrieval with failed exact dialogue
  fact utilization.
- Should show: protocol invariants on one side and outcome/failure-mode counts
  on the other, such as zero EM, denial, and refusal rates.
- Source data: LoCoMo paired comparison, pipeline audit, and prompt inspection.
- Readiness: evidence ready, but inclusion is optional.
- Risks: readers may mistake it for a positive benchmark or a controlled
  EventQA comparison. Keep it in the appendix, label it diagnostic, and exclude
  unreliable cost counters.

## Figure Readiness Summary

- Ready to design now: architecture and frozen-bank protocol.
- Ready to plot from current evidence: three-method EventQA result and
  context-wise analysis.
- Optional appendix-only: LoCoMo limitation.
- Must wait: any figure claiming cost efficiency or superiority to explicit
  text memory.
