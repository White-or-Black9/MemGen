# DetectiveQA Current-P7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a current-P7 DetectiveQA runner that preserves the mature compressed-memory/first-query-only contract, supports smoke plus full n10 execution, and emits appendix/table-ready artifacts without touching `paper/`.

**Architecture:** Reuse `mab5a_detectiveqa_compressed_n10.py` for DetectiveQA data preparation, task contract, and scoring, but replace the historical bank-on path with a new current-P7 paired runner using the frozen P7 Weaver-space bank configuration and a query read-only contract. Keep smoke and full execution as the same runner with bounded context-count control, and add a separate aggregate script for appendix/table-ready summaries.

**Tech Stack:** Python 3.10, PyTorch, existing MemGen + MemoryAgentBench bridge scripts, `unittest`, JSON/JSONL artifacts, tmux for detached runs.

---

## File Structure

New files:

- `scripts/eval/detectiveqa_p7_n10.py`: current-P7 DetectiveQA paired runner.
- `tests/test_detectiveqa_p7_n10.py`: runner contract, config, and invariant tests.
- `scripts/eval/detectiveqa_p7_aggregate.py`: smoke/full aggregate and appendix-ready markdown summary.
- `tests/test_detectiveqa_p7_aggregate.py`: aggregate schema, metric, and invariant tests.

Modified only after validated runs:

- `research_notes/EXPERIMENTS.md`
- `research_notes/PROGRESS.md`
- `research_notes/DECISIONS.md` only if the route decision changes

No planned modification:

- `paper/`
- training code
- Trigger/Weaver training paths

---

### Task 1: Add the current-P7 DetectiveQA runner skeleton

**Files:**
- Create: `scripts/eval/detectiveqa_p7_n10.py`
- Test: `tests/test_detectiveqa_p7_n10.py`

- [ ] **Step 1: Write the failing runner-config test**

```python
import unittest

from scripts.eval import detectiveqa_p7_n10 as target


class DetectiveQAP7ConfigTest(unittest.TestCase):
    def test_p7_bank_config_matches_frozen_paper_values(self):
        config = target.p7_bank_config()
        self.assertTrue(config["enabled"])
        self.assertEqual(config["batch_size"], 1)
        self.assertEqual(config["max_slots"], 16)
        self.assertEqual(config["top_k"], 2)
        self.assertEqual(config["retrieve_policy"], "threshold_topk")
        self.assertEqual(config["update_policy"], "thread_update")
        self.assertEqual(config["retrieve_threshold"], 0.05)
        self.assertEqual(config["update_threshold"], 0.10)
        self.assertEqual(config["decay_alpha"], 0.05)
        self.assertEqual(config["storage_space"], "weaver")
        self.assertEqual(config["query_phase"], "read_only")
```

- [ ] **Step 2: Run the focused test and confirm failure**

Run:

```bash
python -m unittest tests.test_detectiveqa_p7_n10.DetectiveQAP7ConfigTest.test_p7_bank_config_matches_frozen_paper_values
```

Expected: FAIL because `detectiveqa_p7_n10.py` does not exist yet.

- [ ] **Step 3: Create the runner skeleton and config helpers**

```python
"""Current-P7 DetectiveQA compressed-memory runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.eval import mab3_bank_on_full_history as mab3
from scripts.eval import mab5a_detectiveqa_compressed_n10 as base


EXPERIMENT_NAME = "DetectiveQA current-P7 compressed-memory n10"
RUN_PREFIX = "detectiveqa-current-p7-n10"
DEFAULT_OUTPUT_ROOT = "outputs/mab/detectiveqa_p7_n10"
SCHEMA_VERSION = "detectiveqa-current-p7-run/v1"
SUPPORTED_METHODS = ("disabled", "p7", "p7_no_query_retrieval")


def p7_bank_config() -> dict[str, Any]:
    config = mab3.version_a_bank_config(
        top_k=2,
        threshold=0.05,
        retrieve_policy="threshold_topk",
    )
    config.update(
        {
            "enabled": True,
            "batch_size": 1,
            "retrieve_threshold": 0.05,
            "update_threshold": 0.10,
            "max_slots": 16,
            "top_k": 2,
            "decay_alpha": 0.05,
            "retrieve_policy": "threshold_topk",
            "update_policy": "thread_update",
            "storage_space": "weaver",
            "query_phase": "read_only",
        }
    )
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--methods", default="disabled,p7,p7_no_query_retrieval")
    parser.add_argument("--max-contexts", type=int, default=1)
    parser.add_argument("--skip-research-note", action="store_true")
    return parser
```

- [ ] **Step 4: Run the focused test and confirm pass**

Run:

```bash
python -m unittest tests.test_detectiveqa_p7_n10.DetectiveQAP7ConfigTest.test_p7_bank_config_matches_frozen_paper_values
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/detectiveqa_p7_n10.py tests/test_detectiveqa_p7_n10.py
git commit -m "feat: add DetectiveQA current-P7 runner skeleton"
```

---

### Task 2: Add method parsing and invariant helpers

**Files:**
- Modify: `scripts/eval/detectiveqa_p7_n10.py`
- Modify: `tests/test_detectiveqa_p7_n10.py`

- [ ] **Step 1: Write failing method/invariant tests**

```python
import unittest

from scripts.eval import detectiveqa_p7_n10 as target


class DetectiveQAP7InvariantTest(unittest.TestCase):
    def test_expected_method_set_preserves_order_and_rejects_duplicates(self):
        self.assertEqual(
            target.expected_method_set("disabled,p7,p7_no_query_retrieval"),
            ["disabled", "p7", "p7_no_query_retrieval"],
        )
        with self.assertRaises(ValueError):
            target.expected_method_set("p7,p7")

    def test_validate_query_phase_invariants_rejects_query_write(self):
        with self.assertRaises(ValueError):
            target.validate_query_phase_invariants(
                {
                    "method": "p7",
                    "query_write_count": 1,
                    "bank_snapshot_changed_after_query": False,
                    "query_read_only_enforced": True,
                }
            )
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```bash
python -m unittest tests.test_detectiveqa_p7_n10.DetectiveQAP7InvariantTest
```

Expected: FAIL because helper functions are missing.

- [ ] **Step 3: Implement method parsing and invariant validation**

```python
def expected_method_set(methods_spec: str) -> list[str]:
    methods = [item.strip() for item in methods_spec.split(",") if item.strip()]
    if not methods:
        raise ValueError("methods cannot be empty")
    seen = set()
    ordered = []
    for method in methods:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"unknown method: {method}")
        if method in seen:
            raise ValueError(f"duplicate method: {method}")
        seen.add(method)
        ordered.append(method)
    return ordered


def validate_query_phase_invariants(run: dict[str, Any]) -> None:
    if run["method"] == "disabled":
        return
    if int(run.get("query_write_count", 0)) != 0:
        raise ValueError("query write isolation failed")
    if bool(run.get("bank_snapshot_changed_after_query")):
        raise ValueError("query changed frozen bank")
    if run.get("query_read_only_enforced") is not True:
        raise ValueError("query read-only contract failed")
```

- [ ] **Step 4: Run the focused tests and confirm pass**

Run:

```bash
python -m unittest tests.test_detectiveqa_p7_n10.DetectiveQAP7InvariantTest
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/detectiveqa_p7_n10.py tests/test_detectiveqa_p7_n10.py
git commit -m "test: add DetectiveQA current-P7 invariant helpers"
```

---

### Task 3: Implement the paired current-P7 execution path

**Files:**
- Modify: `scripts/eval/detectiveqa_p7_n10.py`
- Modify: `tests/test_detectiveqa_p7_n10.py`

- [ ] **Step 1: Write the failing paired-execution contract test**

```python
import unittest

from scripts.eval import detectiveqa_p7_n10 as target


class DetectiveQAP7ContractTest(unittest.TestCase):
    def test_record_from_result_captures_bank_fields(self):
        record = target.record_from_result(
            method="p7",
            payload={"context_id": "ctx", "query_id": 0, "gold_answers": ["x"]},
            result={
                "prediction": "x",
                "query_write_count": 0,
                "query_read_only_enforced": True,
                "bank_reset_after_context": True,
                "cross_context_leakage_detected": False,
                "retrieved_indices_by_turn": [[0, 1]],
                "retrieved_scores_by_turn": [[0.09, 0.08]],
                "bank_write_count": 9,
                "bank_retrieval_count": 1,
                "bank_retrieved_latent_count": 16,
                "bank_slot_count_final_before_reset": 9,
                "pre_query_bank_summary": {"slot_count": 9},
                "post_query_bank_summary": {"slot_count": 9},
            },
            score={"metrics": {"exact_match": True}, "additional": {"parsed_output": "x"}},
        )
        self.assertTrue(record["bank_created"])
        self.assertEqual(record["bank_retrieval_count"], 1)
        self.assertEqual(record["retrieved_indices_by_turn"], [[0, 1]])
```

- [ ] **Step 2: Run the focused test and confirm failure**

Run:

```bash
python -m unittest tests.test_detectiveqa_p7_n10.DetectiveQAP7ContractTest.test_record_from_result_captures_bank_fields
```

Expected: FAIL because `record_from_result` is missing.

- [ ] **Step 3: Implement paired execution by wrapping the existing DetectiveQA base runner**

```python
def record_from_result(*, method: str, payload: dict[str, Any], result: dict[str, Any], score: dict[str, Any]) -> dict[str, Any]:
    record = {
        "method": method,
        "context_id": payload["context_id"],
        "query_id": int(payload["query_id"]),
        "prediction": result["prediction"],
        "metrics": score["metrics"],
        "additional": score["additional"],
        "bank_created": method != "disabled",
        "query_write_count": int(result.get("query_write_count", 0)),
        "query_read_only_enforced": bool(result.get("query_read_only_enforced", method == "disabled")),
        "bank_reset_after_context": bool(result.get("bank_reset_after_context", True)),
        "cross_context_leakage_detected": bool(result.get("cross_context_leakage_detected", False)),
        "retrieved_indices_by_turn": result.get("retrieved_indices_by_turn", []),
        "retrieved_scores_by_turn": result.get("retrieved_scores_by_turn", []),
        "bank_write_count": int(result.get("bank_write_count", 0)),
        "bank_retrieval_count": int(result.get("bank_retrieval_count", 0)),
        "bank_retrieved_latent_count": int(result.get("bank_retrieved_latent_count", 0)),
        "bank_slot_count_final_before_reset": int(result.get("bank_slot_count_final_before_reset", 0)),
    }
    pre = result.get("pre_query_bank_summary")
    post = result.get("post_query_bank_summary")
    if pre is not None and post is not None:
        record["pre_query_bank_sha256"] = json.dumps(pre, sort_keys=True, ensure_ascii=False)
        record["post_query_bank_sha256"] = json.dumps(post, sort_keys=True, ensure_ascii=False)
        record["bank_snapshot_changed_after_query"] = record["pre_query_bank_sha256"] != record["post_query_bank_sha256"]
    else:
        record["bank_snapshot_changed_after_query"] = False
    return record
```

Implementation note:

- call `base._prepare_payload(...)` and `base._score_prediction(...)`
- run `base._run_model(...)` for `disabled`
- add a current-P7 bank-on path that mirrors the base DetectiveQA compressed contract but uses `p7_bank_config()` and supports `disable_query_retrieval=True` for `p7_no_query_retrieval`
- keep `query_mode = first-query-only`

- [ ] **Step 4: Run the focused test and confirm pass**

Run:

```bash
python -m unittest tests.test_detectiveqa_p7_n10.DetectiveQAP7ContractTest.test_record_from_result_captures_bank_fields
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/detectiveqa_p7_n10.py tests/test_detectiveqa_p7_n10.py
git commit -m "feat: add DetectiveQA current-P7 paired execution path"
```

---

### Task 4: Add artifact validation and CLI execution

**Files:**
- Modify: `scripts/eval/detectiveqa_p7_n10.py`
- Modify: `tests/test_detectiveqa_p7_n10.py`

- [ ] **Step 1: Write the failing artifact-validation test**

```python
import unittest

from scripts.eval import detectiveqa_p7_n10 as target


class DetectiveQAP7ArtifactTest(unittest.TestCase):
    def test_validate_artifact_rejects_missing_method_scope(self):
        artifact = {
            "schema_version": target.SCHEMA_VERSION,
            "methods": ["disabled", "p7", "p7_no_query_retrieval"],
            "records": [
                {"context_id": "ctx0", "query_id": 0, "method": "disabled", "post_reset_slot_count": 0},
                {"context_id": "ctx0", "query_id": 0, "method": "p7", "post_reset_slot_count": 0},
            ],
        }
        with self.assertRaises(ValueError):
            target.validate_artifact(artifact)
```

- [ ] **Step 2: Run the focused test and confirm failure**

Run:

```bash
python -m unittest tests.test_detectiveqa_p7_n10.DetectiveQAP7ArtifactTest.test_validate_artifact_rejects_missing_method_scope
```

Expected: FAIL because `validate_artifact` is missing or incomplete.

- [ ] **Step 3: Implement artifact validation and CLI main**

```python
def validate_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected schema version")
    expected = set(artifact["methods"])
    seen = {}
    for record in artifact["records"]:
        validate_query_phase_invariants(record)
        scope = (record["context_id"], int(record["query_id"]))
        seen.setdefault(scope, set()).add(record["method"])
        if int(record.get("post_reset_slot_count", 0)) != 0:
            raise ValueError("bank reset failed")
        if bool(record.get("cross_context_leakage_detected")):
            raise ValueError("cross-context leakage detected")
    for scope, methods in seen.items():
        if methods != expected:
            raise ValueError(f"method scope drift at {scope}")
```

`main()` must:

- parse methods
- prepare the selected DetectiveQA contexts
- run the three methods
- write `manifest.json`, `records.jsonl`, and `artifact.json`
- call `validate_artifact(...)` before final write

- [ ] **Step 4: Run the focused test and confirm pass**

Run:

```bash
python -m unittest tests.test_detectiveqa_p7_n10.DetectiveQAP7ArtifactTest.test_validate_artifact_rejects_missing_method_scope
```

Expected: PASS.

- [ ] **Step 5: Run the full runner test file**

Run:

```bash
python -m unittest tests.test_detectiveqa_p7_n10
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval/detectiveqa_p7_n10.py tests/test_detectiveqa_p7_n10.py
git commit -m "feat: finalize DetectiveQA current-P7 runner"
```

---

### Task 5: Add the aggregate script

**Files:**
- Create: `scripts/eval/detectiveqa_p7_aggregate.py`
- Create: `tests/test_detectiveqa_p7_aggregate.py`

- [ ] **Step 1: Write the failing aggregate-schema test**

```python
import unittest

from scripts.eval import detectiveqa_p7_aggregate as target


class DetectiveQAP7AggregateTest(unittest.TestCase):
    def test_aggregate_rows_produces_method_metrics(self):
        rows = [
            {"method": "disabled", "metrics": {"exact_match": True}, "bank_retrieval_count": 0},
            {"method": "p7", "metrics": {"exact_match": False}, "bank_retrieval_count": 1},
            {"method": "p7_no_query_retrieval", "metrics": {"exact_match": True}, "bank_retrieval_count": 0},
        ]
        agg = target.aggregate_rows(rows)
        self.assertEqual(agg["methods"]["disabled"]["exact_match_hits"], 1)
        self.assertEqual(agg["methods"]["p7"]["retrieval_active_count"], 1)
```

- [ ] **Step 2: Run the focused test and confirm failure**

Run:

```bash
python -m unittest tests.test_detectiveqa_p7_aggregate.DetectiveQAP7AggregateTest.test_aggregate_rows_produces_method_metrics
```

Expected: FAIL because the aggregate script does not exist yet.

- [ ] **Step 3: Implement the aggregate script**

```python
"""Aggregate DetectiveQA current-P7 run artifacts into appendix-ready summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def aggregate_rows(rows):
    methods = {}
    for method in ["disabled", "p7", "p7_no_query_retrieval"]:
        subset = [row for row in rows if row["method"] == method]
        methods[method] = {
            "count": len(subset),
            "exact_match_hits": sum(int(bool(row["metrics"]["exact_match"])) for row in subset),
            "exact_match_acc": (
                sum(int(bool(row["metrics"]["exact_match"])) for row in subset) / len(subset)
                if subset
                else None
            ),
            "retrieval_active_count": sum(int(row.get("bank_retrieval_count", 0) > 0) for row in subset),
            "mean_retrieved_latent_count": (
                sum(int(row.get("bank_retrieved_latent_count", 0)) for row in subset) / len(subset)
                if subset
                else None
            ),
        }
    return {
        "schema_version": "detectiveqa-current-p7-aggregate/v1",
        "methods": methods,
    }
```

- [ ] **Step 4: Run the focused test and confirm pass**

Run:

```bash
python -m unittest tests.test_detectiveqa_p7_aggregate.DetectiveQAP7AggregateTest.test_aggregate_rows_produces_method_metrics
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/detectiveqa_p7_aggregate.py tests/test_detectiveqa_p7_aggregate.py
git commit -m "feat: add DetectiveQA current-P7 aggregate script"
```

---

### Task 6: Execute the 1-context smoke

**Files:**
- Runtime artifacts: `outputs/mab/detectiveqa_p7_smoke/`
- Modify after validation: `research_notes/EXPERIMENTS.md`
- Modify after validation: `research_notes/PROGRESS.md`

- [ ] **Step 1: Launch a detached smoke on one context**

Run pattern:

```bash
tmux new-session -d -s detectiveqa_p7_smoke \
  "cd /mnt/18T/baishilong/MemGen && export PYTHONPATH=/mnt/18T/baishilong/MemGen CUDA_VISIBLE_DEVICES=0 && /home/baishilong/miniconda3/envs/memgen/bin/python scripts/eval/detectiveqa_p7_n10.py --methods disabled,p7,p7_no_query_retrieval --max-contexts 1 --output-root outputs/mab/detectiveqa_p7_smoke --skip-research-note"
```

- [ ] **Step 2: Verify startup once**

Check:

- tmux session exists
- log starts moving
- output directory created
- model process occupies the selected GPU

- [ ] **Step 3: Validate smoke artifact**

Require:

- one valid context
- all three methods present
- `disabled` created no bank
- `p7` retrieval active or explicitly recorded as zero-activity
- `p7_no_query_retrieval` keeps retrieval disabled
- query writes zero
- reset slot count zero

- [ ] **Step 4: Record smoke outcome**

Update:

- `research_notes/EXPERIMENTS.md`
- `research_notes/PROGRESS.md`

No `paper/` change.

- [ ] **Step 5: Commit**

```bash
git add research_notes/EXPERIMENTS.md research_notes/PROGRESS.md
git commit -m "docs: record DetectiveQA current-P7 smoke"
```

---

### Task 7: Execute the full n10 run and aggregate

**Files:**
- Runtime artifacts: `outputs/mab/detectiveqa_p7_full/`
- Runtime outputs:
  - `outputs/mab/detectiveqa_p7_full_aggregate.json`
  - `outputs/mab/detectiveqa_p7_full_aggregate.md`
- Modify after validation:
  - `research_notes/EXPERIMENTS.md`
  - `research_notes/PROGRESS.md`
  - `research_notes/DECISIONS.md` only if route changes

- [ ] **Step 1: Launch the full n10 run**

Run pattern:

```bash
tmux new-session -d -s detectiveqa_p7_full \
  "cd /mnt/18T/baishilong/MemGen && export PYTHONPATH=/mnt/18T/baishilong/MemGen CUDA_VISIBLE_DEVICES=0 && /home/baishilong/miniconda3/envs/memgen/bin/python scripts/eval/detectiveqa_p7_n10.py --methods disabled,p7,p7_no_query_retrieval --max-contexts 10 --output-root outputs/mab/detectiveqa_p7_full --skip-research-note"
```

- [ ] **Step 2: Verify startup once**

Check tmux, log growth, first artifact creation, and GPU occupancy.

- [ ] **Step 3: Aggregate the finished run**

Run:

```bash
python scripts/eval/detectiveqa_p7_aggregate.py \
  --input-root outputs/mab/detectiveqa_p7_full \
  --output-json outputs/mab/detectiveqa_p7_full_aggregate.json \
  --output-md outputs/mab/detectiveqa_p7_full_aggregate.md
```

Expected:

- JSON summary created
- Markdown appendix/table-ready summary created

- [ ] **Step 4: Make the route decision**

Record one of:

- `internal_only`
- `stress_appendix_candidate`
- `not_useful`

Decision criteria:

- protocol clean
- stable artifact schema
- current-P7 lineage explicit
- effect large enough to be worth appendix mention

- [ ] **Step 5: Record the validated full-run result**

Update:

- `research_notes/EXPERIMENTS.md`
- `research_notes/PROGRESS.md`
- `research_notes/DECISIONS.md` only if the route changes materially

Do not modify `paper/` automatically.

- [ ] **Step 6: Commit**

```bash
git add research_notes/EXPERIMENTS.md research_notes/PROGRESS.md research_notes/DECISIONS.md outputs/mab/detectiveqa_p7_full_aggregate.json outputs/mab/detectiveqa_p7_full_aggregate.md
git commit -m "analysis: record DetectiveQA current-P7 full stress result"
```

---

## Self-Review

Spec coverage:

- current-P7-specific new runner: covered by Tasks 1-4
- smoke plus full n10: covered by Tasks 6-7
- appendix/table-ready aggregate: covered by Task 5 and Task 7
- no automatic paper edits: enforced in Tasks 6-7

Placeholder scan:

- no TBD/TODO placeholders remain
- all new file paths and commands are explicit

Type consistency:

- runner file name is consistently `detectiveqa_p7_n10.py`
- aggregate file name is consistently `detectiveqa_p7_aggregate.py`
- method set is consistently `disabled`, `p7`, `p7_no_query_retrieval`
