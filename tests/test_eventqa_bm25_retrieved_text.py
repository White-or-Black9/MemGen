import hashlib
import math
import unittest

from scripts.eval.eventqa_bm25_retrieved_text import (
    BM25Index,
    BM25ContractError,
    build_retrieved_query_prompt,
    expected_question_indices,
    retrieve_top_k,
    tokenize_bm25,
    validate_artifact,
    validate_smoke_artifact,
)


class BM25RetrievalTest(unittest.TestCase):
    def test_tokenizer_lowercases_and_keeps_alphanumeric_tokens(self):
        self.assertEqual(
            tokenize_bm25("Alpha, BETA-2 isn't C++!"),
            ["alpha", "beta", "2", "isn", "t", "c"],
        )

    def test_ranking_prefers_matching_document(self):
        index = BM25Index(["red apple", "blue berry", "apple apple pie"])
        ranked = index.rank("apple", top_k=2)
        self.assertEqual([item[0] for item in ranked], [2, 0])
        self.assertGreater(ranked[0][1], ranked[1][1])

    def test_ranking_breaks_equal_score_ties_by_chunk_index(self):
        index = BM25Index(["same words", "same words", "unrelated"])
        self.assertEqual(
            [item[0] for item in index.rank("same", top_k=2)],
            [0, 1],
        )

    def test_top_two_provenance_contains_exact_hashes(self):
        chunks = ["alpha", "beta beta", "gamma"]
        selected = retrieve_top_k(chunks, "beta", context_id="ctx", top_k=2)
        self.assertEqual([item["chunk_index"] for item in selected], [1, 0])
        self.assertEqual(selected[0]["chunk_id"], "ctx-chunk-0001")
        self.assertEqual(
            selected[0]["text_sha256"], hashlib.sha256(b"beta beta").hexdigest()
        )
        self.assertTrue(all(math.isfinite(item["bm25_score"]) for item in selected))

    def test_prompt_contains_only_selected_chunks_and_official_query(self):
        selected = retrieve_top_k(
            ["chosen alpha", "not selected", "chosen gamma"],
            "alpha gamma",
            context_id="ctx",
            top_k=2,
        )
        prompt = build_retrieved_query_prompt(selected, "OFFICIAL QUESTION")
        self.assertIn("chosen alpha", prompt)
        self.assertIn("chosen gamma", prompt)
        self.assertNotIn("not selected", prompt)
        self.assertTrue(prompt.endswith("OFFICIAL QUESTION"))


class BM25ArtifactContractTest(unittest.TestCase):
    @staticmethod
    def valid_artifact():
        records = []
        for query_index in range(10):
            records.append(
                {
                    "context_index": 0,
                    "query_index": query_index,
                    "retrieved_chunks": [
                        {
                            "chunk_index": 0,
                            "chunk_id": "ctx-chunk-0000",
                            "bm25_score": 1.0,
                            "text_sha256": "a" * 64,
                        },
                        {
                            "chunk_index": 1,
                            "chunk_id": "ctx-chunk-0001",
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
                    "substring_exact_match": 0,
                    "eventqa_recall": 0.0,
                    "format_flags": {},
                    "cost": {
                        "retrieval_latency_seconds": 0.001,
                        "generation_latency_seconds": 0.5,
                        "end_to_end_latency_seconds": 0.501,
                        "output_tokens": 2,
                    },
                }
            )
        return {
            "schema_version": "eventqa-bm25-top2/v1",
            "measurement_mode": "standalone_process",
            "scope": {"context_index": 0, "question_indices": list(range(10))},
            "bm25": {"k1": 1.5, "b": 0.75, "top_k": 2},
            "cost": {
                "index_construction_latency_seconds": 0.01,
                "baseline_gpu_memory_bytes": 100,
                "peak_gpu_memory_bytes": 200,
            },
            "records": records,
        }

    def test_valid_smoke_artifact_is_accepted(self):
        validate_smoke_artifact(self.valid_artifact())

    def test_scope_must_be_context_zero_q0_through_q9(self):
        artifact = self.valid_artifact()
        artifact["records"][-1]["query_index"] = 10
        with self.assertRaisesRegex(BM25ContractError, "q0-9"):
            validate_smoke_artifact(artifact)

    def test_capacity_overflow_is_rejected(self):
        artifact = self.valid_artifact()
        artifact["records"][0]["capacity_ok"] = False
        with self.assertRaisesRegex(BM25ContractError, "capacity"):
            validate_smoke_artifact(artifact)

    def test_nonfinite_method_cost_is_rejected(self):
        artifact = self.valid_artifact()
        artifact["records"][0]["cost"]["generation_latency_seconds"] = math.nan
        with self.assertRaisesRegex(BM25ContractError, "finite"):
            validate_smoke_artifact(artifact)

    def test_scope_contract_distinguishes_smoke_and_full(self):
        self.assertEqual(expected_question_indices("smoke", 0, 10), list(range(10)))
        for context_index in range(5):
            self.assertEqual(
                expected_question_indices("full", context_index, 100),
                list(range(100)),
            )
        with self.assertRaisesRegex(BM25ContractError, "smoke scope"):
            expected_question_indices("smoke", 1, 10)
        with self.assertRaisesRegex(BM25ContractError, "full scope"):
            expected_question_indices("full", 5, 100)
        with self.assertRaisesRegex(BM25ContractError, "full scope"):
            expected_question_indices("full", 0, 99)

    def test_valid_full_context_artifact_is_accepted(self):
        artifact = self.valid_artifact()
        template = artifact["records"][0]
        artifact["scope"] = {
            "measurement_scope": "full",
            "context_index": 3,
            "question_indices": list(range(100)),
        }
        artifact["records"] = []
        for query_index in range(100):
            record = {
                **template,
                "context_index": 3,
                "query_index": query_index,
                "retrieved_chunks": [dict(item) for item in template["retrieved_chunks"]],
                "cost": dict(template["cost"]),
            }
            artifact["records"].append(record)
        validate_artifact(artifact)

    def test_full_artifact_rejects_missing_q99(self):
        artifact = self.valid_artifact()
        template = artifact["records"][0]
        artifact["scope"] = {
            "measurement_scope": "full",
            "context_index": 2,
            "question_indices": list(range(100)),
        }
        artifact["records"] = [
            {**template, "context_index": 2, "query_index": index}
            for index in range(99)
        ]
        with self.assertRaisesRegex(BM25ContractError, "q0-99"):
            validate_artifact(artifact)

    def test_full_artifact_rejects_wrong_record_context(self):
        artifact = self.valid_artifact()
        template = artifact["records"][0]
        artifact["scope"] = {
            "measurement_scope": "full",
            "context_index": 1,
            "question_indices": list(range(100)),
        }
        artifact["records"] = [
            {**template, "context_index": 1, "query_index": index}
            for index in range(100)
        ]
        artifact["records"][50]["context_index"] = 0
        with self.assertRaisesRegex(BM25ContractError, "context"):
            validate_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
