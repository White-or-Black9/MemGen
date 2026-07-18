# DetectiveQA Disabled Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the DetectiveQA `disabled` query generation contract with the `p7` and `p7_no_query_retrieval` query path so their comparison is not confounded by a shorter disabled response length.

**Architecture:** Keep the fix local to `scripts/eval/detectiveqa_p7_n10.py`. Add a small helper/context-manager that temporarily overrides the disabled branch response length to the runner's `generation_max_length`, use it only inside the disabled query run, and add regression tests that lock the aligned contract.

**Tech Stack:** Python, unittest, apply_patch, pytest/unittest.

---

### Task 1: Lock the alignment contract in tests

**Files:**
- Modify: `tests/test_detectiveqa_p7_n10.py`
- Modify: `scripts/eval/detectiveqa_p7_n10.py`

- [ ] Add a failing test that asserts the runner exposes a disabled-query response-length helper equal to `eventqa.GENERATION_MAX_LENGTH` by default.
- [ ] Add a failing test that exercises the temporary override context manager and verifies `mab3._build_config(...)[\"run\"][\"interaction\"][\"max_response_length\"]` and `mab3._interaction_config(...).max_response_length` are patched inside the context and restored afterward.
- [ ] Run only `tests/test_detectiveqa_p7_n10.py` and confirm the new assertions fail before implementation.

### Task 2: Implement local disabled-path alignment

**Files:**
- Modify: `scripts/eval/detectiveqa_p7_n10.py`
- Test: `tests/test_detectiveqa_p7_n10.py`

- [ ] Add a small helper that returns the disabled query response length from `args.generation_max_length`.
- [ ] Add a context manager that temporarily wraps `mab3._build_config` and `mab3._interaction_config` so disabled runs use that response length.
- [ ] Apply the context manager only around the `disabled` query execution path in the DetectiveQA runner.
- [ ] Re-run `tests/test_detectiveqa_p7_n10.py` and confirm all tests pass.

### Task 3: Minimal evidence check

**Files:**
- Modify: none
- Verify: local ad-hoc command only

- [ ] Re-run the previously identified tiny normalized-disabled diagnostic or an equivalent 1-2 query check to confirm aligned disabled behavior now matches the earlier temporary proof direction.
- [ ] Record the result back into notes only after the code path is verified.
