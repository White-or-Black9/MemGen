import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval.eventqa_paper_aggregator import (
    AggregationError,
    aggregate_paper_results,
    render_markdown,
)


def _row(context_index, query_id, *, off_em=0, on_em=0, off_recall=0.0, on_recall=0.0):
    return {
        "context_index": context_index,
        "context_id": f"ctx-{context_index}",
        "query_id": query_id,
        "qa_pair_id": f"pair-{context_index}-{query_id}",
        "bank_off_substring_exact_match": off_em,
        "bank_on_substring_exact_match": on_em,
        "bank_off_eventqa_recall": off_recall,
        "bank_on_eventqa_recall": on_recall,
        "bank_off_format_flags": {"empty_output": False},
        "bank_on_format_flags": {"empty_output": False},
        "bank_on_query_turn_retrieved_indices": [0, 1],
        "query_write_count": 0,
        "bank_snapshot_changed_after_query": False,
    }


def _write_run(root, run_id, rows, *, update_threshold=0.1, protocol="frozen_context_bank"):
    run = root / run_id
    run.mkdir()
    config = {
        "run_id": run_id,
        "subtask": "eventqa_65536",
        "metric": "substring_exact_match",
        "optional_metric": "eventqa_recall",
        "model_checkpoint_id": "checkpoint-id",
        "eventqa_protocol": protocol,
        "query_phase": "read-only",
        "selected_context_indices": sorted({row["context_index"] for row in rows}),
        "generation_max_length": 40,
        "retrieve_threshold": 0.05,
        "update_threshold": update_threshold,
        "max_slots": 16,
        "top_k": 2,
        "decay_alpha": 0.05,
    }
    for name in ("manifest.json", "run_config.json"):
        (run / name).write_text(json.dumps(config), encoding="utf-8")
    (run / "eventqa_per_question.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    global_metrics = {
        "question_count": len(rows),
        "bank_off_em": sum(row["bank_off_substring_exact_match"] for row in rows) / len(rows),
        "bank_on_em": sum(row["bank_on_substring_exact_match"] for row in rows) / len(rows),
        "bank_off_recall": sum(row["bank_off_eventqa_recall"] for row in rows) / len(rows),
        "bank_on_recall": sum(row["bank_on_eventqa_recall"] for row in rows) / len(rows),
        "bank_off_format_failures": 0,
        "bank_on_format_failures": 0,
        "helpful_memory_count": sum(
            row["bank_on_substring_exact_match"] > row["bank_off_substring_exact_match"]
            for row in rows
        ),
        "harmful_memory_count": sum(
            row["bank_on_substring_exact_match"] < row["bank_off_substring_exact_match"]
            for row in rows
        ),
        "format_harm_count": 0,
    }
    per_context = {}
    for context_index in config["selected_context_indices"]:
        context_rows = [row for row in rows if row["context_index"] == context_index]
        per_context[str(context_index)] = {
            **global_metrics,
            "question_count": len(context_rows),
            "bank_off_em": sum(row["bank_off_substring_exact_match"] for row in context_rows)
            / len(context_rows),
            "bank_on_em": sum(row["bank_on_substring_exact_match"] for row in context_rows)
            / len(context_rows),
            "bank_off_recall": sum(row["bank_off_eventqa_recall"] for row in context_rows)
            / len(context_rows),
            "bank_on_recall": sum(row["bank_on_eventqa_recall"] for row in context_rows)
            / len(context_rows),
        }
    (run / "bank_transition_aggregate.json").write_text(
        json.dumps({"global": global_metrics, "per_context": per_context}),
        encoding="utf-8",
    )
    return run


class EventQAPaperAggregatorTest(unittest.TestCase):
    def test_aggregates_repeat_metrics_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run1 = _write_run(
                root,
                "run-1",
                [_row(0, 0, on_em=1, on_recall=1.0), _row(0, 1)],
            )
            run2 = _write_run(
                root,
                "run-2",
                [_row(0, 0, on_em=1, on_recall=1.0), _row(0, 1, on_em=1)],
            )

            payload = aggregate_paper_results(
                [{"method_id": "p7", "mode": "bank_on", "runs": [str(run1), str(run2)]}]
            )

        row = payload["methods"][0]
        self.assertEqual(payload["schema_version"], "eventqa-paper-aggregate/v1")
        self.assertEqual(row["method_id"], "p7")
        self.assertEqual(row["repeat_count"], 2)
        self.assertEqual(row["question_count_per_repeat"], 2)
        self.assertEqual(row["metrics"]["em"], {"mean": 0.75, "std": 0.25, "values": [0.5, 1.0]})
        self.assertEqual(row["metrics"]["recall"]["mean"], 0.5)
        self.assertEqual(row["per_context"]["0"]["em"]["mean"], 0.75)
        self.assertEqual(row["provenance"]["run_ids"], ["run-1", "run-2"])
        self.assertEqual(row["method_config"]["update_threshold"], 0.1)
        self.assertIsNone(row["cost"]["peak_gpu_memory_bytes"])
        self.assertEqual(row["cost"]["status"], "missing_method_separable_measurement")
        self.assertTrue(row["query_evidence"]["retrieval_ids_recorded"])

    def test_bank_off_selects_off_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run = _write_run(
                Path(tmpdir),
                "run-off",
                [_row(0, 0, off_em=1, off_recall=1.0), _row(0, 1)],
            )
            payload = aggregate_paper_results(
                [{"method_id": "bank_off", "mode": "bank_off", "runs": [str(run)]}]
            )
        self.assertEqual(payload["methods"][0]["metrics"]["em"]["mean"], 0.5)

    def test_missing_required_file_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run = _write_run(Path(tmpdir), "run", [_row(0, 0)])
            (run / "bank_transition_aggregate.json").unlink()
            with self.assertRaisesRegex(AggregationError, "bank_transition_aggregate.json"):
                aggregate_paper_results(
                    [{"method_id": "p7", "mode": "bank_on", "runs": [str(run)]}]
                )

    def test_missing_question_field_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            row = _row(0, 0)
            del row["query_write_count"]
            run = _write_run(Path(tmpdir), "run", [row])
            with self.assertRaisesRegex(AggregationError, "query_write_count"):
                aggregate_paper_results(
                    [{"method_id": "p7", "mode": "bank_on", "runs": [str(run)]}]
                )

    def test_protocol_mismatch_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run = _write_run(Path(tmpdir), "run", [_row(0, 0)], protocol="other")
            with self.assertRaisesRegex(AggregationError, "eventqa_protocol"):
                aggregate_paper_results(
                    [{"method_id": "p7", "mode": "bank_on", "runs": [str(run)]}]
                )

    def test_config_mismatch_between_repeats_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run1 = _write_run(root, "run-1", [_row(0, 0)], update_threshold=0.1)
            run2 = _write_run(root, "run-2", [_row(0, 0)], update_threshold=0.095)
            with self.assertRaisesRegex(AggregationError, "update_threshold"):
                aggregate_paper_results(
                    [{"method_id": "p7", "mode": "bank_on", "runs": [str(run1), str(run2)]}]
                )

    def test_scope_mismatch_between_repeats_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run1 = _write_run(root, "run-1", [_row(0, 0)])
            run2 = _write_run(root, "run-2", [_row(0, 0), _row(0, 1)])
            with self.assertRaisesRegex(AggregationError, "question identity"):
                aggregate_paper_results(
                    [{"method_id": "p7", "mode": "bank_on", "runs": [str(run1), str(run2)]}]
                )

    def test_query_write_violation_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            row = _row(0, 0)
            row["query_write_count"] = 1
            run = _write_run(Path(tmpdir), "run", [row])
            with self.assertRaisesRegex(AggregationError, "query_write_count"):
                aggregate_paper_results(
                    [{"method_id": "p7", "mode": "bank_on", "runs": [str(run)]}]
                )

    def test_markdown_contains_table_ready_metrics(self):
        payload = {
            "schema_version": "eventqa-paper-aggregate/v1",
            "methods": [
                {
                    "method_id": "p7",
                    "repeat_count": 5,
                    "metrics": {
                        "em": {"mean": 0.197, "std": 0.020},
                        "recall": {"mean": 0.254, "std": 0.028},
                        "format_failures": {"mean": 121.4, "std": 8.8},
                    },
                }
            ],
        }
        rendered = render_markdown(payload)
        self.assertIn("| p7 | 5 | 0.197 ± 0.020 |", rendered)


if __name__ == "__main__":
    unittest.main()
