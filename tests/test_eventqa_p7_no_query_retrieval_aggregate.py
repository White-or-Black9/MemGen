import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval.eventqa_p7_no_query_retrieval_aggregate import (
    NoQueryRetrievalAggregateError,
    aggregate_artifacts,
)
from tests.test_eventqa_p7_no_query_retrieval import (
    EventQAP7NoQueryRetrievalArtifactTest,
)


def make_artifact(context_index: int) -> dict:
    artifact = EventQAP7NoQueryRetrievalArtifactTest.artifact()
    template = artifact["records"][0]
    artifact["scope"] = {
        "measurement_scope": "full",
        "context_index": context_index,
        "question_indices": list(range(100)),
        "question_count": 100,
    }
    artifact["construction"] = {
        "construction_latency_seconds": 10.0 + context_index,
        "final_slot_count": 16,
    }
    artifact["cost"] = {
        "baseline_gpu_memory_bytes": 100,
        "peak_gpu_memory_bytes": 300 + context_index,
        "end_to_end_latency_seconds": 70.0 + context_index,
    }
    artifact["effectiveness"] = {
        "substring_exact_match": 0.0,
        "eventqa_recall": 0.0,
        "format_failure_count": 0,
    }
    artifact["records"] = []
    for query_index in range(100):
        record = copy.deepcopy(template)
        record.update(
            {
                "context_index": context_index,
                "query_index": query_index,
                "qa_pair_id": f"pair-{context_index}-{query_index}",
                "substring_exact_match": int(query_index % 10 == 0),
                "eventqa_recall": 0.25 if query_index % 4 == 0 else 0.0,
                "format_flags": {"multiline_output": query_index % 5 == 0},
                "cost": {
                    "query_latency_seconds": 0.5 + context_index * 0.01,
                    "output_tokens": 2,
                },
                "query_invariants": {
                    "query_write_count": 0,
                    "bank_snapshot_changed_after_query": False,
                    "retrieved_indices": [],
                    "retrieved_latent_count": 0,
                },
            }
        )
        artifact["records"].append(record)
    return artifact


class EventQAP7NoQueryRetrievalAggregateTest(unittest.TestCase):
    def write_artifacts(self, root: Path):
        paths = []
        for context_index in range(5):
            artifact = make_artifact(context_index)
            path = root / f"ctx{context_index}.json"
            path.write_text(json.dumps(artifact) + "\n")
            paths.append(path)
        return paths

    def test_aggregates_five_full_context_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self.write_artifacts(Path(tmpdir))
            result = aggregate_artifacts(paths)
        self.assertEqual(result["scope"]["question_count"], 500)
        self.assertEqual(result["effectiveness"]["substring_exact_match"], 0.1)
        self.assertEqual(result["effectiveness"]["eventqa_recall"], 0.0625)
        self.assertEqual(result["effectiveness"]["format_failure_count"], 100)
        self.assertEqual(result["cost"]["construction_latency_seconds"], 60.0)
        self.assertEqual(result["cost"]["construction_amortized_seconds_per_question"], 0.12)
        self.assertEqual(result["construction"]["final_slot_counts"], [16])
        self.assertTrue(result["invariants"]["all_queries_disable_retrieval"])

    def test_rejects_missing_or_duplicate_contexts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self.write_artifacts(Path(tmpdir))
            with self.assertRaisesRegex(NoQueryRetrievalAggregateError, "contexts 0-4"):
                aggregate_artifacts(paths[:-1])
            with self.assertRaisesRegex(NoQueryRetrievalAggregateError, "contexts 0-4"):
                aggregate_artifacts(paths[:-1] + [paths[0]])

    def test_rejects_nonzero_retrieval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self.write_artifacts(Path(tmpdir))
            bad = json.loads(paths[3].read_text())
            bad["records"][7]["query_invariants"]["retrieved_latent_count"] = 8
            paths[3].write_text(json.dumps(bad) + "\n")
            with self.assertRaisesRegex(NoQueryRetrievalAggregateError, "retrieval"):
                aggregate_artifacts(paths)


if __name__ == "__main__":
    unittest.main()
