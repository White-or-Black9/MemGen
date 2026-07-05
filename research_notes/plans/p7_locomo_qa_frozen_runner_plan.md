# P7 LoCoMo-QA Frozen Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimum frozen-P7 LoCoMo-QA runner that can execute Disabled and frozen P7 modes over normalized local LoCoMo conversations, score predictions with deterministic EM/F1, and assert `query_write_count == 0` during QA.

**Architecture:** Reuse the existing `mab6b` EventQA/detective runner conventions for bank wiring, diagnostics, and manifest layout, but switch the dataset contract to the new `locomo_qa_adapter.py` normalized outputs and the scoring contract to `locomo_qa_scorer.py`. Keep one LoCoMo conversation mapped to one session-local bank, freeze the bank after construction, and answer multiple questions from the same frozen snapshot with query-time retrieval enabled and writes blocked.

**Tech Stack:** Python, existing `scripts/eval/mab6b_*` harness patterns, `scripts/eval/locomo_qa_adapter.py`, `scripts/eval/locomo_qa_scorer.py`, `memgen.model.latent_memory_bank`, `torch`, JSON/JSONL artifacts, `unittest`.

---

## Scope

This is a planning document only. It does **not** implement the runner.

Constraints:

- no GPU runs in this planning phase
- no P7 method changes
- no model-code changes
- no `paper/` changes
- no GPT judge
- no external API

## Proposed Files

### Create

- `scripts/eval/mab6b_weaver_space_bank_locomo_qa.py`
  - main runner
  - Disabled and P7 modes
  - construction-time bank build
  - frozen-bank QA loop
  - diagnostics, prediction writing, scoring integration

- `tests/test_mab6b_weaver_space_bank_locomo_qa.py`
  - light/unit tests for runner protocol helpers
  - no-GPU prompt, schema, and query-write assertion tests

### Optional helper only if the runner becomes too large

- `scripts/eval/locomo_qa_runner_common.py`
  - only create if the main runner becomes hard to hold in one file
  - allowed responsibilities:
    - prompt rendering helpers
    - prediction-row builders
    - frozen snapshot serialization helpers

### Reuse directly, no modification expected

- `scripts/eval/locomo_qa_adapter.py`
- `scripts/eval/locomo_qa_scorer.py`
- `scripts/eval/mab6b_weaver_space_bank_eventqa_65536_n5.py`
- `scripts/eval/mab6b_weaver_space_bank_detectiveqa_n10.py`
- `tests/test_mab6b_weaver_space_bank.py`

## Runner Modes

### Required now

- `disabled`
  - no persistent latent bank
  - no construction-time bank writes
  - no bank retrieval during QA
  - still writes the same prediction/scoring artifact schema

- `p7`
  - exact frozen P7 config:
    - `retrieve_threshold = 0.05`
    - `update_threshold = 0.10`
    - `max_slots = 16`
    - `top_k = 2`
    - `decay_alpha = 0.05`
    - Weaver-space bank path / MAB-6B-style mechanism
    - session-local latent bank
    - no Trigger / Weaver retraining
    - no utility gate
    - no tuple suppression
    - no top-1 fallback

### Explicitly deferred in this slice

- `p6`
- `text_summary`
- `rag`

Do not block Disabled/P7 implementation on those later baselines.

## Data Flow

### Input contract

Runner consumes either:

- raw local LoCoMo path:
  - `/mnt/18T/sunyanjia/AMEM/A-mem-main/data/locomo10.json`

or, preferably for smoke/pilot reproducibility:

- normalized adapter artifacts:
  - `normalized_conversations.jsonl`
  - `normalized_qa_records.jsonl`

Recommendation:

- runner should accept normalized files as primary inputs
- optional convenience flags may call the adapter first, but that should not be the core contract

### Record linkage

- `conversation_id = sample_id`
- `question_id = {conversation_id}::q{zero_padded_index}`
- prediction/scoring joins should use `question_id` only

## Construction-Time Protocol

### Conversation-to-model conversion

Use normalized conversation rows from `normalized_conversations.jsonl`.

Per session:

- preserve `session_order`
- preserve turn order
- render each turn into a stable text line

Recommended render format:

```text
[Session 1 | 2023-05-07]
Caroline: Hey Mel! Good to see you! How have you been?
Mel: ...
```

If `timestamp` is missing:

```text
[Session 1]
Caroline: ...
```

### Chunking policy

Required because full conversations are long.

Recommended policy:

- concatenate all rendered session text in stable order
- chunk by token budget, following the same family of chunking behavior used in `mab6b` long-context runners
- keep chunk boundaries monotonic
- never shuffle or regroup across sessions out of order

Initial default:

- `--chunk-size 4096`

Optional later:

- a session-aware chunker that prefers not to split inside a session when the session fits under budget

### Reset boundary

- one LoCoMo conversation = one session-local bank
- hard reset before construction starts for each conversation
- hard reset after all questions for that conversation
- hard reset on exception before advancing to the next conversation

### Construction-time writes

- Disabled mode:
  - no memory bank exists
  - `construction_write_count = 0`

- P7 mode:
  - normal P7 construction-time write/update path is active
  - count every effective bank write/update/replacement in the same style as `mab6b` diagnostics

### Construction-time retrieval

- Disabled mode:
  - `construction_retrieve_count = 0`

- P7 mode:
  - allow construction-time retrieval if the existing MAB-6B bank path naturally does so
  - do not add a new restriction just for LoCoMo
  - log `construction_retrieve_count` explicitly

### Construction diagnostics

Required per conversation:

- chunk count
- chunk token lengths
- construction write count
- construction retrieve count
- final slot count before freeze
- trigger call count
- weaver call count
- peak GPU memory
- total construction latency

### Frozen bank snapshot

After final construction chunk:

- capture a frozen snapshot of the memory bank state
- snapshot must be reusable for multiple QA prompts under the same conversation

Recommended implementation choices:

1. Preferred: clone/copy the bank state in memory
2. Acceptable fallback: serialize to a temp artifact and reload before each question

The runner should isolate snapshot logic in a helper so the restore path is testable.

## QA-Time Protocol

### Question prompting

Each QA prompt should be built from the normalized QA row:

```text
Based on the conversation history you memorized, answer the question concisely.

Question: {question_text}

Answer:
```

Do not inject gold answers or evidence into the prompt.

### Multiple questions under one frozen bank

One conversation can contain many questions.

Required policy:

- construct bank once
- freeze once
- answer many questions from the same frozen state

### Restore strategy before each question

Recommended:

- restore the same frozen bank snapshot before every question

Reason:

- this avoids accidental cross-question bank mutation
- it makes `query_write_count == 0` easier to audit
- it matches the frozen-bank protocol more defensibly

### Query-time retrieval

- Disabled mode:
  - retrieval inactive
  - `query_retrieval_active_count = 0`
  - `retrieved_latent_count = 0`

- P7 mode:
  - retrieval allowed
  - log whether retrieval activated for the question
  - log retrieved latent count and any selected indices/scores that already exist in the `mab6b` trace path

### Query-time writes

Mandatory:

- block query-time writes in P7 mode
- count blocked write attempts if the runner can observe them
- assert final `query_write_count == 0`

Recommended implementation pattern:

- wrap the bank with a query-phase write-blocking proxy similar to the existing frozen-bank EventQA logic
- expose:
  - `query_write_count`
  - `query_write_attempt_count`

### Prediction generation

- generate one answer string per question
- store raw prediction text
- parse with a minimal local rule:
  - keep raw text
  - trimmed visible prediction becomes `prediction_text`
  - scorer handles normalization

### Scoring

- build `prediction_records.jsonl`
- call `locomo_qa_scorer.score_prediction_records(...)`
- write:
  - `scored_prediction_records.jsonl`
  - `aggregate_metrics.json`

No GPT judge path is involved.

## Output Schema

### `prediction_records.jsonl`

Required fields:

- `conversation_id`
- `question_id`
- `category`
- `category_name`
- `question`
- `gold_answer`
- `prediction_text`
- `raw_prediction_text`
- `prediction_status`
- `mode`
- `construction_write_count`
- `construction_retrieve_count`
- `query_retrieval_active_count`
- `retrieved_latent_count`
- `query_write_count`
- `final_slot_count`
- `trigger_call_count`
- `weaver_call_count`
- `latency_seconds`
- `peak_gpu_memory`
- `output_token_count`

### `scored_prediction_records.jsonl`

All prediction-record fields plus:

- `normalized_prediction`
- `normalized_gold_answer`
- `exact_match`
- `token_f1`
- `invalid_output`
- `scorer_version`

### `aggregate_metrics.json`

Required top-level fields:

- `mode`
- `scorer_version`
- `record_count`
- `invalid_output_count`
- `overall_micro`
- `overall_macro_by_conversation`
- `by_category`
- `by_conversation`
- `cost_summary`

Recommended `cost_summary` fields:

- `mean_latency_seconds`
- `max_peak_gpu_memory`
- `mean_output_token_count`
- `mean_construction_write_count`
- `mean_construction_retrieve_count`
- `mean_query_retrieval_active_count`
- `mean_retrieved_latent_count`
- `mean_query_write_count`
- `mean_final_slot_count`
- `mean_trigger_call_count`
- `mean_weaver_call_count`

### `run_diagnostics.json`

Required:

- run manifest
- selected conversation IDs
- selected question IDs
- mode
- frozen P7 config if `mode == p7`
- chunking config
- snapshot policy
- query-write assertion result

### `run_summary.md`

Human-readable summary including:

- mode
- conversation count
- question count
- EM/F1
- invalid output count
- key cost metrics
- whether `query_write_count == 0` passed

## Smoke Test Design

### Scope

- `1` conversation
- `2` QA questions
- both modes:
  - Disabled
  - P7

### Output roots

- `outputs/mab/locomo_qa_smoke_disabled/`
- `outputs/mab/locomo_qa_smoke_p7/`

### Required behavior

- prediction-only generation
- deterministic EM/F1 scoring through `locomo_qa_scorer.py`
- no GPT judge
- explicit `query_write_count == 0` check for P7 mode

### Recommended question selection

- use the first two QA rows from the selected conversation for the first smoke
- keep selection deterministic and recorded in `run_diagnostics.json`

### Candidate smoke commands

Disabled:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> /home/baishilong/miniconda3/envs/memgen/bin/python \
  scripts/eval/mab6b_weaver_space_bank_locomo_qa.py \
  --mode disabled \
  --normalized-conversations outputs/mab/locomo_qa_smoke_subset/normalized_conversations.jsonl \
  --normalized-qa-records outputs/mab/locomo_qa_smoke_subset/normalized_qa_records.jsonl \
  --conversation-id conv-26 \
  --max-questions 2 \
  --output-dir outputs/mab/locomo_qa_smoke_disabled
```

P7:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> /home/baishilong/miniconda3/envs/memgen/bin/python \
  scripts/eval/mab6b_weaver_space_bank_locomo_qa.py \
  --mode p7 \
  --normalized-conversations outputs/mab/locomo_qa_smoke_subset/normalized_conversations.jsonl \
  --normalized-qa-records outputs/mab/locomo_qa_smoke_subset/normalized_qa_records.jsonl \
  --conversation-id conv-26 \
  --max-questions 2 \
  --checkpoint-path <checkpoint_path> \
  --model-checkpoint-id <checkpoint_id> \
  --output-dir outputs/mab/locomo_qa_smoke_p7
```

These are design-time command shapes only. They are not to be run in this planning task.

## Test Plan

### No-GPU unit/light tests

File:

- `tests/test_mab6b_weaver_space_bank_locomo_qa.py`

Required coverage:

- prompt construction from normalized conversation/question rows
- stable question selection and ordering
- prediction-record schema builder
- query-write assertion helper
- scorer integration with synthetic predictions
- Disabled-mode diagnostics default to zero bank counts
- frozen snapshot metadata contract

Planned test cases:

```python
def test_build_question_prompt_uses_question_text_only(self):
    payload = harness.build_question_payload(conversation_row, qa_row)
    self.assertIn(qa_row["question_text"], payload["query_prompt"])
    self.assertNotIn(qa_row["gold_answer"], payload["query_prompt"])
```

```python
def test_prediction_record_contains_required_cost_and_score_fields(self):
    row = harness.build_prediction_record(...)
    self.assertEqual(row["question_id"], "conv-26::q000")
    self.assertIn("query_write_count", row)
    self.assertIn("latency_seconds", row)
```

```python
def test_assert_zero_query_writes_raises_when_nonzero(self):
    with self.assertRaises(RuntimeError):
        harness.assert_zero_query_writes({"query_write_count": 1})
```

```python
def test_disabled_mode_defaults_bank_counts_to_zero(self):
    row = harness.build_disabled_prediction_record(...)
    self.assertEqual(row["construction_write_count"], 0)
    self.assertEqual(row["query_retrieval_active_count"], 0)
```

```python
def test_score_integration_returns_scored_rows_and_aggregate_metrics(self):
    scored_rows, aggregate = harness.score_predictions(qa_rows, prediction_rows)
    self.assertTrue(scored_rows)
    self.assertIn("overall_micro", aggregate)
```

### GPU-required checks deferred to execution phase

- actual P7 smoke run
- actual peak GPU memory collection
- actual trigger/weaver call counts against live model

## Acceptance Criteria For Coding

- no method change to P7
- Disabled mode does not use the memory bank
- P7 mode uses the frozen P7 configuration exactly
- one conversation maps to one session-local bank
- query-time writes are blocked
- `query_write_count == 0` is asserted
- deterministic scorer integration works
- smoke command shape is explicit
- cost metrics are logged in prediction rows and aggregate summary
- no GPT judge/API is introduced

## Implementation Tasks

### Task 1: Freeze runner helper contracts in no-GPU tests

**Files:**
- Create: `tests/test_mab6b_weaver_space_bank_locomo_qa.py`
- Test: `tests/test_mab6b_weaver_space_bank_locomo_qa.py`

- [ ] **Step 1: Write the failing helper-contract tests**

```python
import unittest

from scripts.eval import mab6b_weaver_space_bank_locomo_qa as harness


class MAB6BWeaverSpaceBankLoCoMoQATest(unittest.TestCase):
    def test_build_question_prompt_uses_question_text_only(self):
        conversation_row = {"conversation_id": "conv-26"}
        qa_row = {"question_text": "When did Caroline go?", "gold_answer": "7 May 2023"}
        payload = harness.build_question_payload(conversation_row, qa_row)
        self.assertIn("When did Caroline go?", payload["query_prompt"])
        self.assertNotIn("7 May 2023", payload["query_prompt"])

    def test_assert_zero_query_writes_raises_when_nonzero(self):
        with self.assertRaises(RuntimeError):
            harness.assert_zero_query_writes({"query_write_count": 1})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_mab6b_weaver_space_bank_locomo_qa -v`
Expected: FAIL with missing module or missing helper functions

- [ ] **Step 3: Write minimal implementation**

```python
def build_question_payload(conversation_row, qa_row):
    raise NotImplementedError


def assert_zero_query_writes(row):
    raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_mab6b_weaver_space_bank_locomo_qa -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_mab6b_weaver_space_bank_locomo_qa.py scripts/eval/mab6b_weaver_space_bank_locomo_qa.py
git commit -m "feat: add locomo frozen-runner helper contracts"
```

### Task 2: Implement Disabled-mode record building and scorer integration

**Files:**
- Create: `scripts/eval/mab6b_weaver_space_bank_locomo_qa.py`
- Test: `tests/test_mab6b_weaver_space_bank_locomo_qa.py`

- [ ] **Step 1: Extend tests for Disabled-mode diagnostics and scoring**

```python
    def test_disabled_mode_defaults_bank_counts_to_zero(self):
        row = harness.build_prediction_record(
            mode="disabled",
            conversation_id="conv-26",
            qa_row={"question_id": "conv-26::q000", "category": 2, "category_name": "temporal", "question_text": "Q", "gold_answer": "A"},
            prediction_text="A",
            diagnostics={},
        )
        self.assertEqual(row["construction_write_count"], 0)
        self.assertEqual(row["query_retrieval_active_count"], 0)

    def test_score_integration_returns_scored_rows_and_aggregate_metrics(self):
        qa_rows = [{"question_id": "conv-26::q000", "conversation_id": "conv-26", "category": 2, "category_name": "temporal", "gold_answer": "A"}]
        prediction_rows = [{"question_id": "conv-26::q000", "conversation_id": "conv-26", "method": "disabled", "prediction_text": "A", "prediction_status": "ok"}]
        scored_rows, aggregate = harness.score_predictions(qa_rows, prediction_rows)
        self.assertEqual(scored_rows[0]["exact_match"], 1)
        self.assertIn("overall_micro", aggregate)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_mab6b_weaver_space_bank_locomo_qa -v`
Expected: FAIL with missing `build_prediction_record` or `score_predictions`

- [ ] **Step 3: Write minimal implementation**

```python
def build_prediction_record(*, mode, conversation_id, qa_row, prediction_text, diagnostics):
    return {}


def score_predictions(qa_rows, prediction_rows):
    return [], {}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_mab6b_weaver_space_bank_locomo_qa -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_mab6b_weaver_space_bank_locomo_qa.py scripts/eval/mab6b_weaver_space_bank_locomo_qa.py
git commit -m "feat: add locomo disabled mode schema and scorer integration"
```

### Task 3: Implement normalized-data loading and conversation/question selection

**Files:**
- Create: `scripts/eval/mab6b_weaver_space_bank_locomo_qa.py`
- Test: `tests/test_mab6b_weaver_space_bank_locomo_qa.py`

- [ ] **Step 1: Add tests for normalized JSONL loading and stable selection**

```python
    def test_select_questions_keeps_stable_original_order(self):
        qa_rows = [
            {"question_id": "conv-26::q000", "conversation_id": "conv-26"},
            {"question_id": "conv-26::q001", "conversation_id": "conv-26"},
        ]
        selected = harness.select_questions(qa_rows, conversation_id="conv-26", max_questions=1)
        self.assertEqual([row["question_id"] for row in selected], ["conv-26::q000"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_mab6b_weaver_space_bank_locomo_qa -v`
Expected: FAIL with missing selection helpers

- [ ] **Step 3: Write minimal implementation**

```python
def load_normalized_conversations(path):
    return []


def load_normalized_qa_records(path):
    return []


def select_questions(qa_rows, *, conversation_id, max_questions):
    return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_mab6b_weaver_space_bank_locomo_qa -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_mab6b_weaver_space_bank_locomo_qa.py scripts/eval/mab6b_weaver_space_bank_locomo_qa.py
git commit -m "feat: add locomo normalized-data selection helpers"
```

### Task 4: Add P7 construction/freeze/query protocol hooks

**Files:**
- Create: `scripts/eval/mab6b_weaver_space_bank_locomo_qa.py`
- Test: `tests/test_mab6b_weaver_space_bank_locomo_qa.py`

- [ ] **Step 1: Add tests for snapshot metadata and query-write assertions**

```python
    def test_snapshot_metadata_records_mode_and_conversation(self):
        snapshot = harness.build_snapshot_metadata("conv-26", "p7", final_slot_count=4)
        self.assertEqual(snapshot["conversation_id"], "conv-26")
        self.assertEqual(snapshot["mode"], "p7")
        self.assertEqual(snapshot["final_slot_count"], 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_mab6b_weaver_space_bank_locomo_qa -v`
Expected: FAIL with missing snapshot helper

- [ ] **Step 3: Write minimal implementation**

```python
def build_snapshot_metadata(conversation_id, mode, *, final_slot_count):
    return {}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_mab6b_weaver_space_bank_locomo_qa -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_mab6b_weaver_space_bank_locomo_qa.py scripts/eval/mab6b_weaver_space_bank_locomo_qa.py
git commit -m "feat: add locomo p7 freeze/query protocol hooks"
```

### Task 5: Add CLI, output writing, and smoke-command support

**Files:**
- Create: `scripts/eval/mab6b_weaver_space_bank_locomo_qa.py`
- Test: `tests/test_mab6b_weaver_space_bank_locomo_qa.py`

- [ ] **Step 1: Add tests for parser defaults and output file naming**

```python
    def test_parser_accepts_mode_and_normalized_inputs(self):
        args = harness.build_parser().parse_args([
            "--mode", "disabled",
            "--normalized-conversations", "conv.jsonl",
            "--normalized-qa-records", "qa.jsonl",
            "--conversation-id", "conv-26",
            "--max-questions", "2",
            "--output-dir", "outputs/mab/locomo_qa_smoke_disabled",
        ])
        self.assertEqual(args.mode, "disabled")
        self.assertEqual(args.conversation_id, "conv-26")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_mab6b_weaver_space_bank_locomo_qa -v`
Expected: FAIL with missing `build_parser`

- [ ] **Step 3: Write minimal implementation**

```python
def build_parser():
    parser = argparse.ArgumentParser(description="LoCoMo-QA frozen P7 runner")
    return parser
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_mab6b_weaver_space_bank_locomo_qa -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_mab6b_weaver_space_bank_locomo_qa.py scripts/eval/mab6b_weaver_space_bank_locomo_qa.py
git commit -m "feat: add locomo frozen-runner cli and output contract"
```

## Risks / Blockers

### Main engineering risks

- existing `mab6b` runners are benchmark-specific and may encourage copying too much EventQA/DetectiveQA logic
- bank snapshot/restore may be awkward if the current memory-bank object does not expose a cheap clone path
- real trigger/weaver call counting may depend on hooks that are currently embedded in other harnesses

### Main design choice already fixed

- restore the same frozen bank snapshot before every question

### Non-blockers for this slice

- no GPT judge
- no text-summary/RAG baseline support yet
- no P6 mode yet

## Self-Review

### Spec coverage

- proposed files: covered
- runner modes: covered
- construction-time protocol: covered
- QA-time protocol: covered
- output schema: covered
- smoke test design: covered
- test plan: covered
- acceptance criteria: covered

### Placeholder scan

- no `TODO`
- no `TBD`
- smoke command shapes are explicit

### Type consistency

- runner consumes normalized LoCoMo rows from `locomo_qa_adapter.py`
- runner scoring output uses `locomo_qa_scorer.py`
- `question_id` remains the stable join key across prediction and scoring artifacts

## Next Step Before Coding

Approve one implementation detail before coding:

- use normalized adapter outputs as the runner’s primary dataset contract, not the raw `locomo10.json` file

After that, implement the runner helpers and tests first, then the Disabled path, then the frozen P7 path.
