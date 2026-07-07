import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.eval.eventqa_cost_aggregate import (
    CostAggregationError,
    aggregate_cost_artifacts,
    render_markdown,
)
from scripts.eval.eventqa_method_separable_cost import build_cost_summary


def _write_summary(root: Path, method: str, context_index: int, query_latency: float):
    run = root / method / f"run-{method}-{context_index}"
    run.mkdir(parents=True)
    invariants = []
    if method == "p7":
        invariants = [
            {
                "query_index": index,
                "query_write_count": 0,
                "bank_snapshot_changed_after_query": False,
                "retrieved_indices": [0, 1],
                "retrieved_latent_count": 16,
            }
            for index in range(100)
        ]
    summary = build_cost_summary(
        method=method,
        run_id=run.name,
        context_index=context_index,
        question_indices=list(range(100)),
        construction_latency_seconds=10.0 if method == "p7" else 0.0,
        query_latencies_seconds=[query_latency] * 100,
        end_to_end_latency_seconds=(10.0 if method == "p7" else 0.0)
        + query_latency * 100,
        peak_gpu_memory_bytes=120 if method == "p7" else 110,
        baseline_gpu_memory_bytes=100,
        output_tokens=[20] * 100,
        query_invariants=invariants,
        command=["python", "cost.py"],
        measurement_scope="full",
    )
    summary["effectiveness"] = {
        "substring_exact_match": 0.2 if method == "p7" else 0.01,
        "eventqa_recall": 0.3 if method == "p7" else 0.18,
    }
    summary["method_config"] = {
        "retrieve_threshold": 0.05 if method == "p7" else None,
        "update_threshold": 0.10 if method == "p7" else None,
        "max_slots": 16 if method == "p7" else None,
        "top_k": 2 if method == "p7" else None,
        "decay_alpha": 0.05 if method == "p7" else None,
        "generation_max_length": 40,
        "eventqa_protocol": "frozen_context_bank",
    }
    (run / "cost_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "manifest.json").write_text(
        json.dumps({"gpu": {"cuda_visible_devices": "7", "name": "RTX A6000"}}),
        encoding="utf-8",
    )
    return run / "cost_summary.json"


class EventQACostAggregateTest(unittest.TestCase):
    def test_cli_runs_directly_from_repo_root(self):
        result = subprocess.run(
            [sys.executable, "scripts/eval/eventqa_cost_aggregate.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--input-root", result.stdout)

    def test_aggregates_five_contexts_per_method(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = []
            for context_index in range(5):
                paths.append(_write_summary(root, "disabled", context_index, 1.0))
                paths.append(_write_summary(root, "p7", context_index, 0.8))
            payload = aggregate_cost_artifacts(paths)

        self.assertEqual(payload["schema_version"], "eventqa-full-cost-aggregate/v1")
        self.assertEqual(payload["comparability"]["verdict"], "comparable")
        self.assertEqual(payload["methods"]["disabled"]["question_count"], 500)
        self.assertEqual(payload["methods"]["p7"]["construction_latency_seconds_total"], 50.0)
        self.assertEqual(payload["methods"]["p7"]["query_latency_seconds"]["mean"], 0.8)
        self.assertEqual(payload["methods"]["p7"]["end_to_end_latency_seconds_total"], 450.0)
        self.assertAlmostEqual(payload["comparison"]["query_latency_delta_seconds"], -0.2)
        self.assertEqual(payload["comparison"]["end_to_end_latency_delta_seconds"], -50.0)
        self.assertEqual(payload["comparison"]["incremental_peak_delta_bytes"], 10)
        self.assertEqual(sorted(payload["per_context"]), ["0", "1", "2", "3", "4"])

    def test_rejects_missing_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = [
                _write_summary(root, method, context_index, 1.0)
                for method in ("disabled", "p7")
                for context_index in range(4)
            ]
            with self.assertRaisesRegex(CostAggregationError, "contexts"):
                aggregate_cost_artifacts(paths)

    def test_rejects_mixed_gpu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = []
            for context_index in range(5):
                paths.append(_write_summary(root, "disabled", context_index, 1.0))
                paths.append(_write_summary(root, "p7", context_index, 0.8))
            manifest = paths[-1].with_name("manifest.json")
            manifest.write_text(
                json.dumps({"gpu": {"cuda_visible_devices": "6", "name": "RTX A6000"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CostAggregationError, "GPU"):
                aggregate_cost_artifacts(paths)

    def test_markdown_contains_cost_rows_and_caveat(self):
        payload = {
            "methods": {
                "disabled": {
                    "construction_latency_seconds_total": 0.0,
                    "query_latency_seconds": {"mean": 1.0, "std": 0.1},
                    "end_to_end_latency_seconds_total": 500.0,
                    "amortized_end_to_end_seconds_per_question": 1.0,
                    "incremental_peak_gpu_memory_bytes_max": 100,
                },
                "p7": {
                    "construction_latency_seconds_total": 50.0,
                    "query_latency_seconds": {"mean": 0.8, "std": 0.1},
                    "end_to_end_latency_seconds_total": 450.0,
                    "amortized_end_to_end_seconds_per_question": 0.9,
                    "incremental_peak_gpu_memory_bytes_max": 110,
                },
            },
            "comparison": {"end_to_end_latency_ratio": 0.9},
        }
        rendered = render_markdown(payload)
        self.assertIn("| disabled |", rendered)
        self.assertIn("| p7 |", rendered)
        self.assertIn("not a repeated-load benchmark", rendered)


if __name__ == "__main__":
    unittest.main()
