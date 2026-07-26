import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval.eventqa_text_summary_aggregate import (
    TextSummaryAggregateError,
    aggregate_pairs,
)
from tests.test_eventqa_text_summary_query import TextSummaryQueryTest


def make_construction(context_index: int) -> dict:
    traces = []
    previous = ""
    for chunk_index in range(2):
        summary = f"context {context_index} summary {chunk_index}"
        traces.append(
            {
                "chunk_index": chunk_index,
                "chunk_sha256": "a" * 64,
                "chunk_token_count": 100,
                "previous_summary_sha256": hashlib.sha256(previous.encode()).hexdigest(),
                "previous_summary_token_count": 0 if not previous else 4,
                "rendered_input_sha256": "b" * 64,
                "rendered_input_token_count": 200,
                "raw_output_sha256": hashlib.sha256(summary.encode()).hexdigest(),
                "raw_output_token_count": 4,
                "persisted_summary": summary,
                "persisted_summary_sha256": hashlib.sha256(summary.encode()).hexdigest(),
                "persisted_summary_token_count": 4,
                "truncated": False,
                "latency_seconds": 1.0,
            }
        )
        previous = summary
    return {
        "schema_version": "eventqa-text-summary-construction/v1",
        "scope": {"context_index": context_index, "chunk_count": 2},
        "method": {"summary_token_budget": 128, "latent_memory_bank": False},
        "cost": {
            "construction_latency_seconds": 2.0,
            "baseline_gpu_memory_bytes": 100,
            "peak_gpu_memory_bytes": 300,
            "incremental_peak_gpu_memory_bytes": 200,
        },
        "traces": traces,
        "final_summary": previous,
        "final_summary_sha256": hashlib.sha256(previous.encode()).hexdigest(),
        "final_summary_token_count": 4,
    }


def make_query(context_index: int, construction: dict, construction_bytes: bytes) -> dict:
    artifact = TextSummaryQueryTest.artifact()
    template = artifact["records"][0]
    summary_hash = construction["final_summary_sha256"]
    artifact["scope"] = {
        "measurement_scope": "full",
        "context_index": context_index,
        "question_indices": list(range(100)),
    }
    artifact["summary_source"] = {
        "artifact_path": f"construction-{context_index}.json",
        "artifact_sha256": hashlib.sha256(construction_bytes).hexdigest(),
        "final_summary_sha256": summary_hash,
        "final_summary_token_count": 4,
    }
    artifact["cost"] = {
        "construction_latency_seconds": 2.0,
        "query_latency_seconds": 50.0,
        "end_to_end_latency_seconds": 52.0,
        "baseline_gpu_memory_bytes": 100,
        "peak_gpu_memory_bytes": 350,
        "incremental_peak_gpu_memory_bytes": 250,
    }
    artifact["records"] = []
    for query_index in range(100):
        record = copy.deepcopy(template)
        record.update(
            {
                "context_index": context_index,
                "query_index": query_index,
                "summary_sha256": summary_hash,
                "summary_token_count": 4,
                "rendered_prompt_token_delta": 8,
                "injected_rendered_token_count": 108,
                "substring_exact_match": int(query_index % 2 == 0),
                "eventqa_recall": 0.25,
                "format_flags": {"multiline_output": query_index == 0},
            }
        )
        artifact["records"].append(record)
    return artifact


class TextSummaryAggregateTest(unittest.TestCase):
    def write_pairs(self, root: Path):
        construction_paths = []
        query_paths = []
        for context_index in range(5):
            construction = make_construction(context_index)
            construction_path = root / f"construction-{context_index}.json"
            construction_path.write_text(json.dumps(construction) + "\n")
            construction_bytes = construction_path.read_bytes()
            query = make_query(context_index, construction, construction_bytes)
            query_path = root / f"query-{context_index}.json"
            query_path.write_text(json.dumps(query) + "\n")
            construction_paths.append(construction_path)
            query_paths.append(query_path)
        return construction_paths, query_paths

    def test_aggregates_five_provenance_linked_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            construction_paths, query_paths = self.write_pairs(Path(tmpdir))
            result = aggregate_pairs(construction_paths, query_paths)
        self.assertEqual(result["scope"]["question_count"], 500)
        self.assertEqual(result["effectiveness"]["substring_exact_match"], 0.5)
        self.assertEqual(result["effectiveness"]["eventqa_recall"], 0.25)
        self.assertEqual(result["effectiveness"]["format_failure_count"], 5)
        self.assertEqual(result["summary"]["final_summary_token_counts"], [4])
        self.assertEqual(result["summary"]["rendered_prompt_token_deltas"], [8])
        self.assertEqual(result["cost"]["construction_latency_seconds"], 10.0)
        self.assertEqual(result["cost"]["query_latency_seconds"], 250.0)
        self.assertTrue(result["cost"]["confounded_by_shared_gpu"])

    def test_accepts_controlled_single_gpu_cost_evidence(self):
        evidence = {
            "schema_version": "eventqa-text-summary-controlled-cost/v1",
            "gpu_index": 5,
            "context_indices": [0, 1, 2, 3, 4],
            "serialized_single_gpu": True,
            "all_preflight_clear": True,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            construction_paths, query_paths = self.write_pairs(Path(tmpdir))
            result = aggregate_pairs(
                construction_paths,
                query_paths,
                controlled_cost_evidence=evidence,
            )
        self.assertFalse(result["cost"]["confounded_by_shared_gpu"])
        self.assertTrue(result["cost"]["paper_facing"])
        self.assertEqual(result["cost"]["controlled_cost_evidence"], evidence)

    def test_rejects_exact_construction_byte_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            construction_paths, query_paths = self.write_pairs(Path(tmpdir))
            construction_paths[2].write_text(construction_paths[2].read_text() + "\n")
            with self.assertRaisesRegex(TextSummaryAggregateError, "artifact hash"):
                aggregate_pairs(construction_paths, query_paths)

    def test_rejects_missing_or_duplicate_context_pair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            construction_paths, query_paths = self.write_pairs(Path(tmpdir))
            with self.assertRaisesRegex(TextSummaryAggregateError, "contexts 0-4"):
                aggregate_pairs(construction_paths[:-1], query_paths[:-1])
            with self.assertRaisesRegex(TextSummaryAggregateError, "contexts 0-4"):
                aggregate_pairs(
                    construction_paths[:-1] + [construction_paths[0]],
                    query_paths[:-1] + [query_paths[0]],
                )


if __name__ == "__main__":
    unittest.main()
