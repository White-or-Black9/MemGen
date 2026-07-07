# Method Figures QA

Date: 2026-07-07

## Deliverables

- `fig1_method_architecture.{svg,pdf,tiff,png}`
- `fig2_frozen_bank_protocol.{svg,pdf,tiff,png}`
- Source: `scripts/figures/make_method_figures.py`
- Test: `tests/test_method_figures.py`

## Export Checks

- Backend: Python/Matplotlib only.
- Width: 7.20 inches, approximately 183 mm before tight bounding-box trimming.
- Figure 1 PNG: 1734 x 1111 pixels at 300 dpi.
- Figure 2 PNG: 1734 x 1064 pixels at 300 dpi.
- Figure 1 TIFF: 3468 x 2222 pixels at 600 dpi.
- Figure 2 TIFF: 3468 x 2129 pixels at 600 dpi.
- SVG text remains editable as `<text>` elements.
- PDF fonts are exported as TrueType (`pdf.fonttype = 42`).

## Visual Inspection

- Shared frozen/active/blocked color vocabulary is consistent across figures.
- Trigger, Weaver, and Reasoner labels are readable at final width.
- Construction-time writes and query-time retrieval use distinct paths.
- Query-time writes use both red color and a stop symbol.
- Frozen snapshot and independent question branches are visually separated.
- Arrows have unambiguous direction and do not cross component labels.
- No empirical performance values or unsupported mechanisms appear.

## Semantic Checks

- Figure 1 includes frozen MemGen components, a Weaver-space bank, 16-slot
  capacity, thresholded top-2 retrieval, session reset, and blocked query-time
  writes.
- Figure 2 includes ordered construction, snapshot/freeze, independent restore
  branches, `query_write_count = 0`, and
  `bank_after_query = frozen_snapshot`.
