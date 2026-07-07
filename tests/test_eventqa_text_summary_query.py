import math
import unittest

from scripts.eval.eventqa_text_summary_query import (
    SummaryQueryContractError,
    build_summary_query_prompt,
    expected_question_indices,
    validate_query_artifact,
)


class TextSummaryQueryTest(unittest.TestCase):
    def test_prompt_injects_exact_summary_before_official_query(self):
        prompt = build_summary_query_prompt("frozen summary", "OFFICIAL QUERY")
        self.assertEqual(
            prompt,
            "Persistent memory summary:\nfrozen summary\n\nOFFICIAL QUERY",
        )

    @staticmethod
    def artifact():
        records = []
        for index in range(10):
            records.append(
                {
                    "context_index": 0,
                    "query_index": index,
                    "summary_sha256": "a" * 64,
                    "summary_token_count": 81,
                    "official_query_sha256": "b" * 64,
                    "injected_prompt_sha256": "c" * 64,
                    "official_rendered_token_count": 100,
                    "injected_rendered_token_count": 190,
                    "rendered_prompt_token_delta": 90,
                    "context_capacity": 32768,
                    "capacity_ok": True,
                    "prediction": "answer",
                    "substring_exact_match": 0,
                    "eventqa_recall": 0.0,
                    "format_flags": {},
                    "cost": {
                        "query_latency_seconds": 0.5,
                        "output_tokens": 2,
                    },
                }
            )
        return {
            "schema_version": "eventqa-text-summary-query/v1",
            "scope": {"context_index": 0, "question_indices": list(range(10))},
            "summary_source": {
                "artifact_path": "construction.json",
                "artifact_sha256": "d" * 64,
                "final_summary_sha256": "a" * 64,
                "final_summary_token_count": 81,
            },
            "cost": {
                "construction_latency_seconds": 35.0,
                "query_latency_seconds": 5.0,
                "end_to_end_latency_seconds": 40.0,
                "baseline_gpu_memory_bytes": 100,
                "peak_gpu_memory_bytes": 200,
            },
            "records": records,
        }

    def test_valid_q0_q9_artifact_is_accepted(self):
        validate_query_artifact(self.artifact())

    def test_rejects_summary_mutation(self):
        artifact = self.artifact()
        artifact["records"][3]["summary_sha256"] = "e" * 64
        with self.assertRaisesRegex(SummaryQueryContractError, "summary"):
            validate_query_artifact(artifact)

    def test_rejects_nonfinite_cost(self):
        artifact = self.artifact()
        artifact["records"][0]["cost"]["query_latency_seconds"] = math.nan
        with self.assertRaisesRegex(SummaryQueryContractError, "finite"):
            validate_query_artifact(artifact)

    def test_full_scope_accepts_contexts_zero_through_four(self):
        self.assertEqual(expected_question_indices("smoke", 0, 10), list(range(10)))
        self.assertEqual(expected_question_indices("full", 3, 100), list(range(100)))
        artifact = self.artifact()
        template = artifact["records"][0]
        artifact["scope"] = {"measurement_scope": "full", "context_index": 3, "question_indices": list(range(100))}
        artifact["records"] = [{**template, "context_index": 3, "query_index": i} for i in range(100)]
        validate_query_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
