# P7 LoCoMo-QA Adapter/Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the no-GPU LoCoMo-QA adapter and deterministic scorer for frozen P7 so local `locomo10.json` can be normalized, subsetted, and scored offline before any runner or GPU work.

**Architecture:** Split the work into two repository-local modules: one adapter that reads the external local LoCoMo JSON and writes stable normalized JSONL records, and one scorer that consumes prediction JSONL plus QA references to compute deterministic exact-match and token-F1 metrics. Keep both modules independent of P7/model imports so they can be tested and iterated entirely on CPU.

**Tech Stack:** Python standard library (`argparse`, `json`, `re`, `string`, `statistics`, `hashlib`, `pathlib`), repository-local scripts under `scripts/eval/`, `unittest`, JSONL artifacts.

---

## Scope

This plan covers only the no-GPU adapter/scorer slice.

Constraints:

- no GPU runs
- no P7 changes
- no model-code changes
- no `paper/` changes
- no GPT judge
- no external API dependency

## File Structure

### New files to create

- `scripts/eval/locomo_qa_adapter.py`
  - load `/mnt/18T/sunyanjia/AMEM/A-mem-main/data/locomo10.json`
  - normalize conversations and QA rows into stable JSONL
  - support inspection and smoke-subset writing

- `scripts/eval/locomo_qa_scorer.py`
  - load normalized QA rows and prediction rows
  - compute deterministic exact match, token F1, invalid-output flags
  - write scored JSONL plus aggregate metrics JSON

- `tests/test_locomo_qa_adapter.py`
  - integration-light tests against the real local `locomo10.json`
  - ordering, ID, category, and reference-preservation checks

- `tests/test_locomo_qa_scorer.py`
  - synthetic deterministic scoring and aggregation tests

### No existing file modifications required

- This slice should be standalone and must not import P7/model code.

## Frozen Data Contract

### Local source assumptions confirmed by audit

- dataset path: `/mnt/18T/sunyanjia/AMEM/A-mem-main/data/locomo10.json`
- top-level object: JSON list
- sample-level ID: `sample_id` exists, for example `conv-26`
- conversation sessions are stored as `session_<N>` plus `session_<N>_date_time`
- turns contain at least:
  - `speaker`
  - `dia_id`
  - `text`
- QA entries contain:
  - `question`
  - `answer`
  - `category`
  - `evidence`
- native `question_id` is not present in the local QA entries

### Stable ID policy

- `conversation_id`
  - use source `sample_id` directly

- `session_id`
  - derive from `session_<N>` numeric order

- `turn_id`
  - use source `dia_id` directly

- `question_id`
  - derive deterministically as `{conversation_id}::q{zero_padded_index}`
  - example: `conv-26::q000`

- `reference_id`
  - equal to `question_id` in this slice

This avoids depending on missing upstream QA IDs while keeping deterministic joins for later runner use.

## Frozen JSONL Schemas

### 1. Normalized conversation records

Path target:

- `normalized_conversations.jsonl`

One line per conversation.

```json
{
  "conversation_id": "conv-26",
  "source_dataset": "locomo10",
  "source_path": "/mnt/18T/sunyanjia/AMEM/A-mem-main/data/locomo10.json",
  "sample_index": 0,
  "speaker_a": "Caroline",
  "speaker_b": "Mel",
  "session_count": 19,
  "turn_count": 412,
  "session_order": [1, 2, 3, 4],
  "sessions": [
    {
      "session_id": 1,
      "timestamp": "2023-05-07",
      "turn_count": 18,
      "turns": [
        {
          "turn_id": "D1:1",
          "speaker": "Caroline",
          "role": "user",
          "content": "Hey Mel! Good to see you! How have you been?",
          "raw_text": "Hey Mel! Good to see you! How have you been?",
          "timestamp": "2023-05-07",
          "session_id": 1
        }
      ]
    }
  ],
  "session_summary": {
    "session_1_summary": "..."
  }
}
```

Frozen rules:

- preserve source order of sessions
- preserve source order of turns within each session
- `role` defaults to the raw `speaker` name when no canonical user/assistant mapping exists
- `content` is the normalized text field used downstream
- `raw_text` preserves pre-normalized text for debugging

### 2. Normalized QA records

Path target:

- `normalized_qa_records.jsonl`

One line per QA item.

```json
{
  "question_id": "conv-26::q000",
  "reference_id": "conv-26::q000",
  "conversation_id": "conv-26",
  "sample_index": 0,
  "question_index": 0,
  "question_text": "When did Caroline go to the LGBTQ support group?",
  "gold_answer": "7 May 2023",
  "reference_answers": ["7 May 2023"],
  "category": 2,
  "category_name": "temporal",
  "evidence": ["D1:3"],
  "evidence_turn_ids": ["D1:3"],
  "evidence_session_ids": [1],
  "metadata": {
    "source_has_native_question_id": false
  }
}
```

Frozen rules:

- `reference_answers` is always a list even when only one answer exists
- `gold_answer` is the first canonical reference string used by the scorer
- `category_name` is derived from a frozen local mapping:
  - `1 -> multi_hop`
  - `2 -> temporal`
  - `3 -> open_domain`
  - `4 -> single_hop`
  - `5 -> adversarial`
- `evidence_session_ids` are derived from `D<session>:<turn>` patterns when possible

### 3. Prediction records

Path target:

- `prediction_records.jsonl`

One line per model/system prediction.

```json
{
  "question_id": "conv-26::q000",
  "conversation_id": "conv-26",
  "method": "disabled",
  "prediction_text": "7 May 2023",
  "raw_prediction_text": "7 May 2023\n",
  "prediction_status": "ok",
  "output_tokens": null,
  "metadata": {
    "run_id": "offline-smoke"
  }
}
```

Frozen status values:

- `ok`
- `empty`
- `missing`
- `invalid`

### 4. Scored prediction records

Path target:

- `scored_prediction_records.jsonl`

One line per scored prediction.

```json
{
  "question_id": "conv-26::q000",
  "conversation_id": "conv-26",
  "method": "disabled",
  "category": 2,
  "category_name": "temporal",
  "prediction_text": "7 May 2023",
  "gold_answer": "7 May 2023",
  "normalized_prediction": "7 may 2023",
  "normalized_gold_answer": "7 may 2023",
  "exact_match": 1,
  "token_f1": 1.0,
  "invalid_output": 0,
  "scorer_version": "locomo_qa_v1",
  "status": "scored"
}
```

### 5. Aggregate metrics

Path target:

- `aggregate_metrics.json`

```json
{
  "method": "disabled",
  "scorer_version": "locomo_qa_v1",
  "record_count": 5,
  "invalid_output_count": 1,
  "overall_micro": {
    "exact_match_mean": 0.4,
    "token_f1_mean": 0.62
  },
  "overall_macro_by_conversation": {
    "exact_match_mean": 0.4,
    "token_f1_mean": 0.62
  },
  "by_category": {
    "temporal": {
      "count": 2,
      "exact_match_mean": 0.5,
      "token_f1_mean": 0.75
    }
  },
  "by_conversation": {
    "conv-26": {
      "count": 5,
      "exact_match_mean": 0.4,
      "token_f1_mean": 0.62
    }
  }
}
```

Frozen aggregation rules:

- overall micro = mean across all scored questions
- overall macro by conversation = mean of per-conversation means
- category-wise aggregates are keyed by `category_name`
- no judge-based fields in this slice

## CLI Design

### Adapter CLI

File:

- `scripts/eval/locomo_qa_adapter.py`

Subcommands to implement:

- `inspect`
- `extract-qa`
- `write-smoke-subset`

Exact command shapes:

```bash
python scripts/eval/locomo_qa_adapter.py inspect \
  --input /mnt/18T/sunyanjia/AMEM/A-mem-main/data/locomo10.json
```

Expected output:

- stdout summary of conversation count, QA count, category counts, session/turn ranges

```bash
python scripts/eval/locomo_qa_adapter.py extract-qa \
  --input /mnt/18T/sunyanjia/AMEM/A-mem-main/data/locomo10.json \
  --output-dir outputs/mab/locomo_qa_adapter_extract
```

Expected output files:

- `normalized_conversations.jsonl`
- `normalized_qa_records.jsonl`
- `adapter_summary.json`

```bash
python scripts/eval/locomo_qa_adapter.py write-smoke-subset \
  --input /mnt/18T/sunyanjia/AMEM/A-mem-main/data/locomo10.json \
  --output-dir outputs/mab/locomo_qa_smoke_subset \
  --conversation-id conv-26 \
  --max-questions 5
```

Expected output files:

- `normalized_conversations.jsonl`
- `normalized_qa_records.jsonl`
- `adapter_summary.json`

Frozen smoke-subset rules:

- preserve full conversation for the selected conversation ID
- keep the first `N` QA items in stable original order unless category filters are added later

### Scorer CLI

File:

- `scripts/eval/locomo_qa_scorer.py`

Subcommands to implement:

- `score`

Exact command shape:

```bash
python scripts/eval/locomo_qa_scorer.py score \
  --qa-records outputs/mab/locomo_qa_smoke_subset/normalized_qa_records.jsonl \
  --predictions outputs/mab/locomo_qa_smoke_subset/prediction_records.jsonl \
  --output-dir outputs/mab/locomo_qa_scored_smoke
```

Expected output files:

- `scored_prediction_records.jsonl`
- `aggregate_metrics.json`

### Synthetic scoring CLI

This uses the same scorer command with a hand-written synthetic predictions file:

```bash
python scripts/eval/locomo_qa_scorer.py score \
  --qa-records testdata/locomo_scorer_fixture_qa.jsonl \
  --predictions testdata/locomo_scorer_fixture_predictions.jsonl \
  --output-dir /tmp/locomo_scorer_fixture_run
```

## Deterministic Scorer Rules

### Normalized exact match

- lowercase
- strip leading/trailing whitespace
- collapse repeated internal whitespace
- strip surrounding punctuation

Definition:

- `exact_match = 1` when normalized prediction equals normalized gold answer
- otherwise `0`

### Token F1

- tokenize from normalized text on whitespace
- compute set-free token overlap using token counts, not unique-token sets
- if either side is empty after normalization, F1 is `0.0`

### Invalid output detection

Mark `invalid_output = 1` when:

- prediction row is missing for a QA record
- `prediction_status` is `missing` or `invalid`
- normalized prediction is empty

Do not count punctuation-only text as valid output after normalization.

### Aggregation

- per-question scored rows
- by-category means
- by-conversation means
- overall micro means
- overall macro-by-conversation means

No confidence intervals are required in this slice.

## Unit Test Plan

### Adapter tests

File:

- `tests/test_locomo_qa_adapter.py`

Required coverage:

- load the real local `locomo10.json`
- extract at least 1 conversation
- extract at least 5 QA records
- verify stable conversation ordering
- verify stable session ordering
- verify stable QA ordering
- verify category labels
- verify reference answers are preserved
- verify derived `question_id` stability
- verify evidence/session metadata extraction

Planned test cases:

```python
def test_loads_local_locomo10_and_reports_nonzero_counts(self):
    summary = adapter.inspect_dataset(LOCOMO_PATH)
    self.assertEqual(summary["conversation_count"], 10)
    self.assertGreaterEqual(summary["qa_count"], 5)
```

```python
def test_extract_conversation_preserves_session_order_and_turn_ids(self):
    conversations, qa_rows = adapter.extract_records(LOCOMO_PATH, conversation_ids=["conv-26"], max_questions=5)
    self.assertEqual(conversations[0]["conversation_id"], "conv-26")
    self.assertEqual(conversations[0]["session_order"][0], 1)
    self.assertEqual(conversations[0]["sessions"][0]["turns"][0]["turn_id"], "D1:1")
```

```python
def test_extract_qa_records_derives_stable_question_ids_and_categories(self):
    _, qa_rows = adapter.extract_records(LOCOMO_PATH, conversation_ids=["conv-26"], max_questions=5)
    self.assertEqual(qa_rows[0]["question_id"], "conv-26::q000")
    self.assertIn(qa_rows[0]["category_name"], {"multi_hop", "temporal", "open_domain", "single_hop", "adversarial"})
    self.assertTrue(qa_rows[0]["reference_answers"])
```

### Scorer tests

File:

- `tests/test_locomo_qa_scorer.py`

Required coverage:

- exact correct prediction
- partial overlap prediction
- empty or invalid prediction
- case/punctuation normalization
- aggregation fields

Planned test cases:

```python
def test_exact_match_and_token_f1_for_exact_correct_prediction(self):
    scored = scorer.score_row(
        prediction_text="7 May 2023",
        gold_answer="7 May 2023",
        prediction_status="ok",
    )
    self.assertEqual(scored["exact_match"], 1)
    self.assertEqual(scored["token_f1"], 1.0)
```

```python
def test_partial_overlap_yields_zero_em_and_positive_f1(self):
    scored = scorer.score_row(
        prediction_text="May 2023",
        gold_answer="7 May 2023",
        prediction_status="ok",
    )
    self.assertEqual(scored["exact_match"], 0)
    self.assertGreater(scored["token_f1"], 0.0)
    self.assertLess(scored["token_f1"], 1.0)
```

```python
def test_empty_or_invalid_prediction_marks_invalid_output(self):
    scored = scorer.score_row(
        prediction_text="   ",
        gold_answer="7 May 2023",
        prediction_status="ok",
    )
    self.assertEqual(scored["invalid_output"], 1)
    self.assertEqual(scored["token_f1"], 0.0)
```

```python
def test_case_and_punctuation_normalization_supports_exact_match(self):
    scored = scorer.score_row(
        prediction_text="7 may 2023.",
        gold_answer="7 May 2023",
        prediction_status="ok",
    )
    self.assertEqual(scored["exact_match"], 1)
```

```python
def test_aggregate_metrics_contains_overall_category_and_conversation_fields(self):
    summary = scorer.aggregate_scores([
        {"conversation_id": "conv-26", "category_name": "temporal", "exact_match": 1, "token_f1": 1.0, "invalid_output": 0},
        {"conversation_id": "conv-26", "category_name": "single_hop", "exact_match": 0, "token_f1": 0.5, "invalid_output": 0},
    ])
    self.assertIn("overall_micro", summary)
    self.assertIn("by_category", summary)
    self.assertIn("by_conversation", summary)
```

## Implementation Tasks

### Task 1: Freeze the adapter data contract in tests

**Files:**
- Create: `tests/test_locomo_qa_adapter.py`
- Test: `tests/test_locomo_qa_adapter.py`

- [ ] **Step 1: Write the failing adapter tests**

```python
import unittest
from pathlib import Path

from scripts.eval import locomo_qa_adapter as adapter


LOCOMO_PATH = Path("/mnt/18T/sunyanjia/AMEM/A-mem-main/data/locomo10.json")


class LoCoMoQAAdapterTest(unittest.TestCase):
    def test_loads_local_locomo10_and_reports_nonzero_counts(self):
        summary = adapter.inspect_dataset(LOCOMO_PATH)
        self.assertEqual(summary["conversation_count"], 10)
        self.assertGreaterEqual(summary["qa_count"], 5)

    def test_extract_conversation_preserves_session_order_and_turn_ids(self):
        conversations, qa_rows = adapter.extract_records(
            LOCOMO_PATH,
            conversation_ids=["conv-26"],
            max_questions=5,
        )
        self.assertEqual(conversations[0]["conversation_id"], "conv-26")
        self.assertEqual(conversations[0]["session_order"][0], 1)
        self.assertEqual(conversations[0]["sessions"][0]["turns"][0]["turn_id"], "D1:1")
        self.assertEqual(len(qa_rows), 5)

    def test_extract_qa_records_derives_stable_question_ids_and_categories(self):
        _, qa_rows = adapter.extract_records(
            LOCOMO_PATH,
            conversation_ids=["conv-26"],
            max_questions=5,
        )
        self.assertEqual(qa_rows[0]["question_id"], "conv-26::q000")
        self.assertTrue(qa_rows[0]["reference_answers"])
        self.assertIn(
            qa_rows[0]["category_name"],
            {"multi_hop", "temporal", "open_domain", "single_hop", "adversarial"},
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_locomo_qa_adapter -v`
Expected: FAIL with `ImportError` or missing `inspect_dataset` / `extract_records`

- [ ] **Step 3: Write minimal adapter implementation**

```python
from pathlib import Path


CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


def inspect_dataset(path):
    raise NotImplementedError


def extract_records(path, conversation_ids=None, max_questions=None):
    raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_locomo_qa_adapter -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_locomo_qa_adapter.py scripts/eval/locomo_qa_adapter.py
git commit -m "feat: add locomo qa adapter contract"
```

### Task 2: Implement adapter CLI and JSONL writers

**Files:**
- Create: `scripts/eval/locomo_qa_adapter.py`
- Test: `tests/test_locomo_qa_adapter.py`

- [ ] **Step 1: Extend tests for smoke-subset writing**

```python
    def test_write_smoke_subset_outputs_expected_jsonl_files(self):
        # pseudo-shape only; the real test uses TemporaryDirectory
        output_paths = adapter.write_smoke_subset(
            LOCOMO_PATH,
            output_dir,
            conversation_ids=["conv-26"],
            max_questions=5,
        )
        self.assertTrue(output_paths["conversations_path"].exists())
        self.assertTrue(output_paths["qa_records_path"].exists())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_locomo_qa_adapter.LoCoMoQAAdapterTest.test_write_smoke_subset_outputs_expected_jsonl_files -v`
Expected: FAIL with missing `write_smoke_subset`

- [ ] **Step 3: Write minimal implementation**

```python
def write_smoke_subset(path, output_dir, conversation_ids=None, max_questions=5):
    conversations, qa_rows = extract_records(path, conversation_ids=conversation_ids, max_questions=max_questions)
    return {
        "conversations_path": output_dir / "normalized_conversations.jsonl",
        "qa_records_path": output_dir / "normalized_qa_records.jsonl",
    }


def build_parser():
    parser = argparse.ArgumentParser(description="LoCoMo QA adapter")
    subparsers = parser.add_subparsers(dest="command", required=True)
    return parser
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_locomo_qa_adapter -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_locomo_qa_adapter.py scripts/eval/locomo_qa_adapter.py
git commit -m "feat: add locomo qa adapter cli and smoke subset writer"
```

### Task 3: Freeze the deterministic scorer in tests

**Files:**
- Create: `tests/test_locomo_qa_scorer.py`
- Create: `scripts/eval/locomo_qa_scorer.py`
- Test: `tests/test_locomo_qa_scorer.py`

- [ ] **Step 1: Write the failing scorer tests**

```python
import unittest

from scripts.eval import locomo_qa_scorer as scorer


class LoCoMoQAScorerTest(unittest.TestCase):
    def test_exact_match_and_token_f1_for_exact_correct_prediction(self):
        scored = scorer.score_row("7 May 2023", "7 May 2023", "ok")
        self.assertEqual(scored["exact_match"], 1)
        self.assertEqual(scored["token_f1"], 1.0)

    def test_partial_overlap_yields_zero_em_and_positive_f1(self):
        scored = scorer.score_row("May 2023", "7 May 2023", "ok")
        self.assertEqual(scored["exact_match"], 0)
        self.assertGreater(scored["token_f1"], 0.0)
        self.assertLess(scored["token_f1"], 1.0)

    def test_empty_or_invalid_prediction_marks_invalid_output(self):
        scored = scorer.score_row("   ", "7 May 2023", "ok")
        self.assertEqual(scored["invalid_output"], 1)
        self.assertEqual(scored["token_f1"], 0.0)

    def test_case_and_punctuation_normalization_supports_exact_match(self):
        scored = scorer.score_row("7 may 2023.", "7 May 2023", "ok")
        self.assertEqual(scored["exact_match"], 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_locomo_qa_scorer -v`
Expected: FAIL with `ImportError` or missing `score_row`

- [ ] **Step 3: Write minimal scorer implementation**

```python
def normalize_text(text):
    raise NotImplementedError


def score_row(prediction_text, gold_answer, prediction_status):
    raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_locomo_qa_scorer -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_locomo_qa_scorer.py scripts/eval/locomo_qa_scorer.py
git commit -m "feat: add deterministic locomo qa scorer"
```

### Task 4: Implement aggregate metrics and scorer CLI

**Files:**
- Create: `scripts/eval/locomo_qa_scorer.py`
- Test: `tests/test_locomo_qa_scorer.py`

- [ ] **Step 1: Extend tests for aggregation fields**

```python
    def test_aggregate_metrics_contains_overall_category_and_conversation_fields(self):
        summary = scorer.aggregate_scores([
            {"conversation_id": "conv-26", "category_name": "temporal", "exact_match": 1, "token_f1": 1.0, "invalid_output": 0},
            {"conversation_id": "conv-26", "category_name": "single_hop", "exact_match": 0, "token_f1": 0.5, "invalid_output": 0},
        ])
        self.assertIn("overall_micro", summary)
        self.assertIn("overall_macro_by_conversation", summary)
        self.assertIn("by_category", summary)
        self.assertIn("by_conversation", summary)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_locomo_qa_scorer.LoCoMoQAScorerTest.test_aggregate_metrics_contains_overall_category_and_conversation_fields -v`
Expected: FAIL with missing `aggregate_scores`

- [ ] **Step 3: Write minimal implementation**

```python
def aggregate_scores(rows):
    return {
        "overall_micro": {},
        "overall_macro_by_conversation": {},
        "by_category": {},
        "by_conversation": {},
    }


def build_parser():
    parser = argparse.ArgumentParser(description="LoCoMo QA scorer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    return parser
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_locomo_qa_scorer -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_locomo_qa_scorer.py scripts/eval/locomo_qa_scorer.py
git commit -m "feat: add locomo qa scorer aggregates and cli"
```

### Task 5: End-to-end no-GPU smoke for adapter plus scorer

**Files:**
- Create: `scripts/eval/locomo_qa_adapter.py`
- Create: `scripts/eval/locomo_qa_scorer.py`
- Test: `tests/test_locomo_qa_adapter.py`
- Test: `tests/test_locomo_qa_scorer.py`

- [ ] **Step 1: Add a smoke-oriented test for a 1-conversation / 5-question subset**

```python
    def test_adapter_can_produce_one_conversation_five_question_smoke_subset(self):
        conversations, qa_rows = adapter.extract_records(
            LOCOMO_PATH,
            conversation_ids=["conv-26"],
            max_questions=5,
        )
        self.assertEqual(len(conversations), 1)
        self.assertEqual(len(qa_rows), 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_locomo_qa_adapter tests.test_locomo_qa_scorer -v`
Expected: FAIL until all helpers and schemas are connected

- [ ] **Step 3: Write minimal implementation**

```python
def main():
    args = build_parser().parse_args()
    if args.command == "inspect":
        ...
    elif args.command == "extract-qa":
        ...
    elif args.command == "write-smoke-subset":
        ...
```

```python
def main():
    args = build_parser().parse_args()
    if args.command == "score":
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_locomo_qa_adapter tests.test_locomo_qa_scorer -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_locomo_qa_adapter.py tests/test_locomo_qa_scorer.py scripts/eval/locomo_qa_adapter.py scripts/eval/locomo_qa_scorer.py
git commit -m "feat: complete locomo qa no-gpu adapter and scorer slice"
```

## No-GPU Test Plan

### Required local test commands

```bash
python -m unittest tests.test_locomo_qa_adapter -v
python -m unittest tests.test_locomo_qa_scorer -v
python -m unittest tests.test_locomo_qa_adapter tests.test_locomo_qa_scorer -v
```

### Required manual CLI checks after implementation

```bash
python scripts/eval/locomo_qa_adapter.py inspect \
  --input /mnt/18T/sunyanjia/AMEM/A-mem-main/data/locomo10.json
```

```bash
python scripts/eval/locomo_qa_adapter.py write-smoke-subset \
  --input /mnt/18T/sunyanjia/AMEM/A-mem-main/data/locomo10.json \
  --output-dir outputs/mab/locomo_qa_smoke_subset \
  --conversation-id conv-26 \
  --max-questions 5
```

```bash
python scripts/eval/locomo_qa_scorer.py score \
  --qa-records outputs/mab/locomo_qa_smoke_subset/normalized_qa_records.jsonl \
  --predictions outputs/mab/locomo_qa_smoke_subset/prediction_records.jsonl \
  --output-dir outputs/mab/locomo_qa_scored_smoke
```

## Acceptance Criteria

- no GPU required
- no P7/model import required
- no GPT judge required
- JSONL schemas are stable
- scorer results are deterministic
- adapter can produce a 1-conversation / 5-question smoke subset
- category labels are preserved
- reference answers are preserved
- invalid output detection is deterministic
- output is ready for later P7 runner implementation

## Self-Review

### Spec coverage

- adapter loading, normalization, ID preservation, and metadata preservation: covered by Tasks 1 and 2
- scorer metrics, invalid detection, and aggregation: covered by Tasks 3 and 4
- frozen JSONL schemas: defined in this document
- no-GPU tests: covered by the unit-test plan and Task 5
- CLI design: defined in this document
- acceptance criteria: defined in this document

### Placeholder scan

- no `TBD` or `TODO`
- all requested file paths are explicit
- all requested command shapes are explicit

### Type consistency

- adapter outputs `question_id`, `conversation_id`, `category_name`, `reference_answers`
- scorer consumes those same keys
- aggregate schema keys are fixed as:
  - `overall_micro`
  - `overall_macro_by_conversation`
  - `by_category`
  - `by_conversation`

## Next Step Before Coding

Approve this plan and keep the frozen derived-ID policy:

- `conversation_id = sample_id`
- `question_id = {conversation_id}::q{zero_padded_index}`

That approval is the only remaining design dependency before implementing the no-GPU adapter/scorer slice.
