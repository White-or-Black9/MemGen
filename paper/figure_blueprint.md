# Figure Blueprint

Status: Figures 1 and 2 rendered and QA-checked on 2026-07-07.

The figure sequence follows the long-horizon LLM-agent outline. EventQA is the
current evidence source, not the title-level definition of the method.

## Figure 1. Method Architecture

- Purpose: explain where the session-local bank sits in frozen MemGen.
- Core conclusion: an inference-time, session-local latent bank makes Weaver
  outputs persistently retrievable without retraining Trigger, Weaver, or
  Reasoner.
- Archetype: schematic-led method figure.
- Backend: Python/Matplotlib only.
- Final size: double-column, approximately 183 mm wide.
- Visual hierarchy:
  1. hero path: input -> frozen Trigger -> frozen Weaver -> frozen Reasoner ->
     answer;
  2. construction path: Weaver-space latent -> bank write/update/replacement;
  3. query path: pooled query -> scored retrieval -> selected latents ->
     Reasoner;
  4. invariants: frozen model components, bounded 16-slot bank, session reset,
     and blocked query-time writes.
- Should show:
  - frozen Trigger, Weaver, and Reasoner;
  - construction-time Weaver-space memory writes;
  - bounded session-local bank with retrieval, matched update, replacement,
    and reset;
  - query-time retrieval into the Reasoner;
  - blocked query-time writes.
- Source data: `research_notes/METHOD.md` and the frozen P7 definition.
- Readiness: rendered in SVG/PDF/TIFF/PNG and integrated into the Method
  section.
- Risks: the visual must not imply retraining, cross-session sharing, a utility
  gate, tuple suppression, or top-1 fallback.
- Style: frozen components in neutral gray, active latent-memory path in muted
  blue, bank-management operations in teal, and blocked writes in red with an
  additional stop symbol so color is not the only encoding.
- Exports: editable SVG, PDF, 600-dpi TIFF, and PNG preview.

## Figure 2. Frozen-Bank Protocol

- Purpose: make construction/query separation and bank ownership explicit.
- Core conclusion: every question retrieves from the same context-built bank
  snapshot, while query-time writes are prohibited and cannot alter later
  questions.
- Archetype: horizontal protocol timeline with a branched repeated-query phase.
- Backend: Python/Matplotlib only.
- Final size: double-column, approximately 183 mm wide.
- Panel map:
  - construction phase: reset -> ordered context chunks -> write/update bounded
    bank;
  - boundary: snapshot and freeze;
  - query phase: independent question branches, each restoring the same
    snapshot, retrieving latent support, and generating an answer;
  - invariant strip: `query_write_count = 0` and
    `bank_after_query = frozen_snapshot`.
- Should show:
  1. reset one bank for one EventQA context;
  2. ingest ordered context chunks;
  3. freeze the bank snapshot;
  4. restore/reuse that snapshot for each question;
  5. permit retrieval and assert zero query writes.
- Source data: EventQA runner protocol, prompt/scorer verification, and method
  notes.
- Readiness: rendered in SVG/PDF/TIFF/PNG and integrated into the Method
  section.
- Risks: distinguish construction-time retrieval/update from query-time
  retrieval; do not imply that visible full context is injected at query time.
- Style: construction in muted blue, snapshot boundary in dark gray, query
  branches in teal, and prohibited writes in red with a stop symbol.
- Exports: editable SVG, PDF, 600-dpi TIFF, and PNG preview.

## Figures 1-2 QA Contract

- All labels must remain readable at final double-column size.
- SVG text must remain editable (`svg.fonttype = none`); PDF fonts use TrueType.
- Use one shared visual vocabulary across both figures.
- Verify that every arrow has one unambiguous direction and that no connector
  crosses a label or component.
- Verify architecture labels against the current Method section and active
  latent-bank implementation before rendering.
- Do not include empirical performance numbers; these are method/protocol
  schematics rather than result figures.
- Deliver the Python source alongside all exports and a short render-QA note.

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

- Completed: architecture and frozen-bank protocol.
- Ready to plot from current evidence: three-method EventQA result and
  context-wise analysis.
- Optional appendix-only: LoCoMo limitation.
- Must wait: any figure claiming cost efficiency or superiority to explicit
  text memory.
