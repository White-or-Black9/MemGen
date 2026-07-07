# Method Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate publication-ready Method Architecture and Frozen-Bank
Protocol figures from the approved figure contract.

**Architecture:** One Python/Matplotlib script owns the shared visual language,
draws both schematics from vector primitives, and exports SVG, PDF, TIFF, and
PNG. A focused unittest validates required labels, editable SVG text, dimensions,
and all expected files without depending on experimental results.

**Tech Stack:** Python, Matplotlib, unittest, SVG/XML inspection.

---

### Task 1: Establish export and semantic tests

**Files:**
- Create: `tests/test_method_figures.py`
- Create later: `scripts/figures/make_method_figures.py`

- [ ] **Step 1: Write a failing unittest**

Create tests that import the plotting module, render into a temporary directory,
and require both figure stems in SVG/PDF/TIFF/PNG. Parse SVG as text and assert
that labels for `Trigger`, `Weaver`, `Reasoner`, `Session-local latent bank`,
`query_write_count = 0`, and `bank_after_query = frozen_snapshot` remain text.

- [ ] **Step 2: Run the test and confirm failure**

Run: `python -m unittest tests.test_method_figures`

Expected: import failure because `scripts.figures.make_method_figures` does not
exist.

### Task 2: Implement shared vector drawing and Figure 1

**Files:**
- Create: `scripts/figures/make_method_figures.py`

- [ ] **Step 1: Add publication configuration and reusable primitives**

Set sans-serif fonts, `svg.fonttype = none`, `pdf.fonttype = 42`, a restrained
gray/blue/teal/red palette, rounded boxes, arrows, locks, slot glyphs, phase
bands, and export helpers.

- [ ] **Step 2: Draw Method Architecture**

Draw the frozen Trigger-Weaver-Reasoner hero path, construction-time Weaver
write path, bounded 16-slot session-local bank, similarity/decay/threshold/top-k
retrieval, Reasoner injection, reset boundary, and blocked query-write path.
Exclude training loops, global memory, utility gates, tuple suppression, and
top-1 fallback.

### Task 3: Implement Figure 2 and exports

**Files:**
- Modify: `scripts/figures/make_method_figures.py`

- [ ] **Step 1: Draw Frozen-Bank Protocol**

Draw reset, ordered construction chunks, write/update, snapshot/freeze, and
independent question branches. Each question restores the same snapshot,
retrieves latent support, produces an answer, and shows a blocked write.

- [ ] **Step 2: Export both figures**

Write editable SVG, PDF, 600-dpi TIFF, and 300-dpi PNG preview files under
`paper/figures/`. Keep each figure approximately 183 mm wide.

- [ ] **Step 3: Run semantic/export tests**

Run: `python -m unittest tests.test_method_figures`

Expected: all tests pass.

### Task 4: Render inspection and manuscript synchronization

**Files:**
- Modify: `paper/figure_blueprint.md`
- Modify: `paper/draft_v0.md`
- Modify: `paper/draft_v0_todo_map.md`
- Create: `paper/figures/method_figures_qa.md`

- [ ] **Step 1: Generate final exports**

Run: `python scripts/figures/make_method_figures.py`

Expected: eight figure exports plus the output manifest printed to stdout.

- [ ] **Step 2: Inspect PNG previews**

Open both PNGs at full resolution and verify readable labels, unambiguous arrow
direction, no overlap, common visual vocabulary, and correct blocked-write
encoding. Revise in Python and rerender if any check fails.

- [ ] **Step 3: Record QA and synchronize paper files**

Record dimensions, formats, editable-text check, visual inspection outcome, and
source script in the QA note. Add Figure 1/2 references and captions to the
Method section, remove TODO-D04, and update the TODO map and blueprint status.

- [ ] **Step 4: Run final verification**

Run:

```bash
python -m unittest tests.test_method_figures
python scripts/figures/make_method_figures.py --check
git diff --check
```

Expected: tests pass, check mode reports all expected outputs valid, and git
reports no whitespace errors.
