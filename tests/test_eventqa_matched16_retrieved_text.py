import hashlib
import math
import unittest

from scripts.eval.eventqa_matched16_retrieved_text import (
    Matched16ContractError,
    build_matched_prompt,
    expected_question_indices,
    rank_relevant_windows,
    select_budget_constrained_pair,
    select_relevant_window,
    validate_artifact,
    validate_smoke_artifact,
)


class SpaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [int(token) for token in text.split()]

    def decode(self, token_ids, **kwargs):
        del kwargs
        return " ".join(str(token) for token in token_ids)


class MatchedWindowTest(unittest.TestCase):
    def test_selects_exactly_eight_contiguous_token_ids(self):
        token_ids = list(range(12))
        window = select_relevant_window(
            token_ids,
            lambda ids: "target" if ids[0] == 3 else "other",
            "target",
            {"target": 2.0},
            width=8,
        )
        self.assertEqual(window["token_start"], 3)
        self.assertEqual(window["token_end"], 11)
        self.assertEqual(window["token_ids"], list(range(3, 11)))

    def test_tie_breaks_by_earliest_token_offset(self):
        window = select_relevant_window(
            list(range(10)),
            lambda ids: "no overlap",
            "target",
            {"target": 1.0},
            width=8,
        )
        self.assertEqual(window["token_start"], 0)

    def test_rejects_chunks_shorter_than_eight_tokens(self):
        with self.assertRaisesRegex(Matched16ContractError, "8 tokens"):
            select_relevant_window(
                list(range(7)), lambda ids: "text", "query", {}, width=8
            )

    def test_prompt_contains_two_windows_and_official_query(self):
        windows = [
            {"chunk_id": "c0", "text": "first evidence"},
            {"chunk_id": "c1", "text": "second evidence"},
        ]
        prompt = build_matched_prompt(windows, "OFFICIAL QUESTION")
        self.assertIn("first evidence", prompt)
        self.assertIn("second evidence", prompt)
        self.assertTrue(prompt.endswith("OFFICIAL QUESTION"))
        self.assertNotIn("source=", prompt)
        self.assertEqual(prompt, "first evidencesecond evidenceOFFICIAL QUESTION")

    def test_ranked_windows_are_score_then_offset_deterministic(self):
        ranked = rank_relevant_windows(
            list(range(10)),
            lambda ids: "target" if ids[0] in {1, 2} else "other",
            "target",
            {"target": 2.0},
            width=8,
        )
        self.assertEqual([item["token_start"] for item in ranked], [1, 2, 0])
        self.assertEqual([item["candidate_rank"] for item in ranked], [1, 2, 3])

    def test_constrained_pair_falls_back_when_top_pair_exceeds_budget(self):
        candidates = [
            [
                {"text": "A", "window_score": 5.0, "token_start": 0, "candidate_rank": 1},
                {"text": "B", "window_score": 4.0, "token_start": 1, "candidate_rank": 2},
            ],
            [
                {"text": "C", "window_score": 5.0, "token_start": 0, "candidate_rank": 1},
                {"text": "D", "window_score": 4.0, "token_start": 1, "candidate_rank": 2},
            ],
        ]

        def rendered_count(prompt):
            return 116 if prompt.startswith("AD") else 117

        selected = select_budget_constrained_pair(
            candidates,
            "QUESTION",
            rendered_count,
            official_rendered_token_count=100,
            candidate_limits=(2,),
        )
        self.assertEqual([item["text"] for item in selected["windows"]], ["A", "D"])
        self.assertTrue(selected["fallback_used"])
        self.assertEqual(selected["candidate_ranks"], [1, 2])
        self.assertEqual(selected["score_loss"], 1.0)
        self.assertEqual(selected["rendered_prompt_token_delta"], 16)

    def test_constrained_pair_preserves_valid_top_pair_before_tie_search(self):
        candidates = [
            [
                {"text": "A", "window_score": 1.0, "token_start": 10, "candidate_rank": 1},
                {"text": "B", "window_score": 1.0, "token_start": 0, "candidate_rank": 2},
            ],
            [
                {"text": "C", "window_score": 1.0, "token_start": 10, "candidate_rank": 1},
                {"text": "D", "window_score": 1.0, "token_start": 0, "candidate_rank": 2},
            ],
        ]
        selected = select_budget_constrained_pair(
            candidates,
            "QUESTION",
            lambda prompt: 116,
            official_rendered_token_count=100,
            candidate_limits=(2,),
        )
        self.assertEqual(selected["candidate_ranks"], [1, 1])
        self.assertFalse(selected["fallback_used"])


class Matched16ArtifactTest(unittest.TestCase):
    @staticmethod
    def valid_artifact():
        records = []
        for query_index in range(10):
            windows = []
            for chunk_index in range(2):
                text = " ".join(str(value) for value in range(chunk_index * 8, chunk_index * 8 + 8))
                windows.append(
                    {
                        "chunk_index": chunk_index,
                        "chunk_id": f"ctx-chunk-{chunk_index:04d}",
                        "chunk_bm25_score": 1.0 - chunk_index * 0.1,
                        "chunk_text_sha256": "a" * 64,
                        "token_start": 0,
                        "token_end": 8,
                        "token_ids": list(range(chunk_index * 8, chunk_index * 8 + 8)),
                        "text": text,
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "window_score": 1.0,
                    }
                )
            records.append(
                {
                    "context_index": 0,
                    "query_index": query_index,
                    "windows": windows,
                    "source_token_count": 16,
                    "official_query_sha256": "b" * 64,
                    "matched_prompt_sha256": "c" * 64,
                    "official_rendered_token_count": 100,
                    "matched_rendered_token_count": 116,
                    "rendered_prompt_token_delta": 16,
                    "context_capacity": 32768,
                    "capacity_ok": True,
                    "prediction": "answer",
                    "substring_exact_match": 0,
                    "eventqa_recall": 0.0,
                    "format_flags": {},
                    "cost": {
                        "retrieval_and_window_latency_seconds": 0.01,
                        "generation_latency_seconds": 0.5,
                        "end_to_end_latency_seconds": 0.51,
                        "output_tokens": 2,
                    },
                }
            )
        return {
            "schema_version": "eventqa-matched16/v1",
            "measurement_mode": "standalone_process",
            "scope": {"context_index": 0, "question_indices": list(range(10))},
            "method": {
                "retrieval": "bm25_top2",
                "window_tokens_per_chunk": 8,
                "source_token_budget": 16,
            },
            "cost": {
                "index_construction_latency_seconds": 0.01,
                "baseline_gpu_memory_bytes": 100,
                "peak_gpu_memory_bytes": 200,
            },
            "records": records,
        }

    def test_valid_artifact_is_accepted(self):
        validate_smoke_artifact(self.valid_artifact())

    def test_rejects_source_budget_other_than_sixteen(self):
        artifact = self.valid_artifact()
        artifact["records"][0]["source_token_count"] = 15
        with self.assertRaisesRegex(Matched16ContractError, "16"):
            validate_smoke_artifact(artifact)

    def test_rejects_window_other_than_eight_token_ids(self):
        artifact = self.valid_artifact()
        artifact["records"][0]["windows"][0]["token_ids"].pop()
        with self.assertRaisesRegex(Matched16ContractError, "8"):
            validate_smoke_artifact(artifact)

    def test_rejects_inconsistent_prompt_delta(self):
        artifact = self.valid_artifact()
        artifact["records"][0]["rendered_prompt_token_delta"] = 15
        with self.assertRaisesRegex(Matched16ContractError, "delta"):
            validate_smoke_artifact(artifact)

    def test_rejects_rendered_prompt_delta_other_than_sixteen(self):
        artifact = self.valid_artifact()
        artifact["records"][0]["matched_rendered_token_count"] = 117
        artifact["records"][0]["rendered_prompt_token_delta"] = 17
        with self.assertRaisesRegex(Matched16ContractError, "16"):
            validate_smoke_artifact(artifact)

    def test_rejects_nonfinite_cost(self):
        artifact = self.valid_artifact()
        artifact["records"][0]["cost"]["generation_latency_seconds"] = math.nan
        with self.assertRaisesRegex(Matched16ContractError, "finite"):
            validate_smoke_artifact(artifact)

    def test_scope_contract_distinguishes_smoke_and_full(self):
        self.assertEqual(expected_question_indices("smoke", 0, 10), list(range(10)))
        for context_index in range(5):
            self.assertEqual(
                expected_question_indices("full", context_index, 100), list(range(100))
            )
        with self.assertRaisesRegex(Matched16ContractError, "smoke scope"):
            expected_question_indices("smoke", 1, 10)
        with self.assertRaisesRegex(Matched16ContractError, "full scope"):
            expected_question_indices("full", 5, 100)

    def test_valid_full_context_artifact_is_accepted(self):
        artifact = self.valid_artifact()
        template = artifact["records"][0]
        artifact["scope"] = {
            "measurement_scope": "full",
            "context_index": 4,
            "question_indices": list(range(100)),
        }
        artifact["records"] = [
            {**template, "context_index": 4, "query_index": index}
            for index in range(100)
        ]
        validate_artifact(artifact)

    def test_full_context_rejects_missing_q99(self):
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
        with self.assertRaisesRegex(Matched16ContractError, "q0-99"):
            validate_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
