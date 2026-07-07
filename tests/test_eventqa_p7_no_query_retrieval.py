import math
import unittest

from scripts.eval.eventqa_p7_no_query_retrieval import (
    NoQueryRetrievalContractError,
    expected_question_indices,
    validate_artifact,
)


class EventQAP7NoQueryRetrievalArtifactTest(unittest.TestCase):
    @staticmethod
    def artifact():
        records = []
        for query_index in range(10):
            records.append(
                {
                    "context_index": 0,
                    "query_index": query_index,
                    "qa_pair_id": f"pair-{query_index}",
                    "prediction": "answer",
                    "substring_exact_match": 0,
                    "eventqa_recall": 0.0,
                    "format_flags": {},
                    "cost": {
                        "query_latency_seconds": 0.5,
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
        return {
            "schema_version": "eventqa-p7-no-query-retrieval/v1",
            "measurement_mode": "standalone_process",
            "scope": {
                "measurement_scope": "smoke",
                "context_index": 0,
                "question_indices": list(range(10)),
            },
            "method_config": {
                "retrieve_threshold": 0.05,
                "update_threshold": 0.10,
                "max_slots": 16,
                "top_k": 2,
                "decay_alpha": 0.05,
                "generation_max_length": 40,
                "eventqa_protocol": "frozen_context_bank",
                "query_retrieval_disabled": True,
            },
            "construction": {
                "construction_latency_seconds": 3.0,
                "final_slot_count": 5,
            },
            "cost": {
                "baseline_gpu_memory_bytes": 100,
                "peak_gpu_memory_bytes": 200,
                "end_to_end_latency_seconds": 8.0,
            },
            "records": records,
        }

    def test_valid_smoke_artifact_is_accepted(self):
        validate_artifact(self.artifact())

    def test_scope_contract_distinguishes_smoke_and_full(self):
        self.assertEqual(expected_question_indices("smoke", 0, 10), list(range(10)))
        self.assertEqual(expected_question_indices("full", 4, 100), list(range(100)))
        with self.assertRaisesRegex(NoQueryRetrievalContractError, "smoke scope"):
            expected_question_indices("smoke", 1, 10)
        with self.assertRaisesRegex(NoQueryRetrievalContractError, "full scope"):
            expected_question_indices("full", 0, 99)

    def test_retrieval_must_remain_disabled_for_every_query(self):
        artifact = self.artifact()
        artifact["records"][4]["query_invariants"]["retrieved_latent_count"] = 8
        with self.assertRaisesRegex(NoQueryRetrievalContractError, "retrieval disabled"):
            validate_artifact(artifact)

    def test_query_writes_or_snapshot_changes_are_rejected(self):
        artifact = self.artifact()
        artifact["records"][1]["query_invariants"]["query_write_count"] = 1
        with self.assertRaisesRegex(NoQueryRetrievalContractError, "query writes"):
            validate_artifact(artifact)
        artifact = self.artifact()
        artifact["records"][1]["query_invariants"][
            "bank_snapshot_changed_after_query"
        ] = True
        with self.assertRaisesRegex(NoQueryRetrievalContractError, "snapshot"):
            validate_artifact(artifact)

    def test_nonfinite_cost_is_rejected(self):
        artifact = self.artifact()
        artifact["records"][0]["cost"]["query_latency_seconds"] = math.nan
        with self.assertRaisesRegex(NoQueryRetrievalContractError, "finite"):
            validate_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
