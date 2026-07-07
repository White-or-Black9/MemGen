import copy
import unittest

from scripts.eval.eventqa_bm25_aggregate import (
    BM25AggregateError,
    aggregate_artifacts,
)


def make_artifact(context_index: int) -> dict:
    records = []
    for query_index in range(100):
        records.append(
            {
                "context_index": context_index,
                "query_index": query_index,
                "retrieved_chunks": [
                    {
                        "chunk_index": 0,
                        "chunk_id": f"ctx{context_index}-chunk-0000",
                        "bm25_score": 1.0,
                        "text_sha256": "a" * 64,
                    },
                    {
                        "chunk_index": 1,
                        "chunk_id": f"ctx{context_index}-chunk-0001",
                        "bm25_score": 0.5,
                        "text_sha256": "b" * 64,
                    },
                ],
                "query_sha256": "c" * 64,
                "prompt_sha256": "d" * 64,
                "injected_token_count": 100,
                "rendered_prompt_token_count": 120,
                "context_capacity": 32768,
                "capacity_ok": True,
                "prediction": "answer",
                "substring_exact_match": int(query_index % 2 == 0),
                "eventqa_recall": 0.25,
                "format_flags": {"multiline_output": query_index == 0},
                "cost": {
                    "retrieval_latency_seconds": 0.01,
                    "generation_latency_seconds": 1.0,
                    "end_to_end_latency_seconds": 1.01,
                    "output_tokens": 2,
                },
            }
        )
    return {
        "schema_version": "eventqa-bm25-top2/v1",
        "measurement_mode": "standalone_process",
        "run_id": f"run-{context_index}",
        "scope": {
            "measurement_scope": "full",
            "context_index": context_index,
            "question_indices": list(range(100)),
        },
        "bm25": {"k1": 1.5, "b": 0.75, "top_k": 2},
        "cost": {
            "index_construction_latency_seconds": 0.1,
            "baseline_gpu_memory_bytes": 100,
            "peak_gpu_memory_bytes": 200,
            "incremental_peak_gpu_memory_bytes": 100,
        },
        "records": records,
    }


class BM25AggregateTest(unittest.TestCase):
    def test_aggregates_five_contexts_and_500_unique_questions(self):
        result = aggregate_artifacts([make_artifact(index) for index in range(5)])
        self.assertEqual(result["scope"]["context_indices"], list(range(5)))
        self.assertEqual(result["scope"]["question_count"], 500)
        self.assertEqual(result["effectiveness"]["substring_exact_match"], 0.5)
        self.assertEqual(result["effectiveness"]["eventqa_recall"], 0.25)
        self.assertEqual(result["effectiveness"]["format_failure_count"], 5)
        self.assertAlmostEqual(result["cost"]["method_total_seconds"], 505.5)
        self.assertEqual(result["cost"]["incremental_peak_gpu_memory_bytes_max"], 100)

    def test_rejects_missing_context(self):
        with self.assertRaisesRegex(BM25AggregateError, "contexts 0-4"):
            aggregate_artifacts([make_artifact(index) for index in range(4)])

    def test_rejects_duplicate_context(self):
        artifacts = [make_artifact(index) for index in range(5)]
        artifacts[-1] = copy.deepcopy(artifacts[0])
        with self.assertRaisesRegex(BM25AggregateError, "contexts 0-4"):
            aggregate_artifacts(artifacts)

    def test_rejects_bm25_config_drift(self):
        artifacts = [make_artifact(index) for index in range(5)]
        artifacts[3]["bm25"]["top_k"] = 3
        with self.assertRaisesRegex(BM25AggregateError, "BM25"):
            aggregate_artifacts(artifacts)


if __name__ == "__main__":
    unittest.main()
