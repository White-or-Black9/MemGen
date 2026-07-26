import unittest

from scripts.eval.eventqa_p7_no_retrieved_memory_conditioning import (
    ConditioningAblationContractError,
    expected_question_indices,
    validate_artifact,
)


class EventQAP7NoRetrievedMemoryConditioningTest(unittest.TestCase):
    @staticmethod
    def artifact():
        records = [
            {
                "query_invariants": {
                    "query_write_count": 0,
                    "query_update_count": 0,
                    "conditioned_latent_count": 0,
                }
            }
            for _ in range(10)
        ]
        return {
            "schema_version": "eventqa-p7-no-retrieved-memory-conditioning/v1",
            "method_config": {
                "query_retrieval_disabled": False,
                "query_retrieved_memory_conditioning": False,
            },
            "records": records,
        }

    def test_valid_artifact_is_accepted(self):
        validate_artifact(self.artifact())

    def test_contract_distinguishes_retrieval_from_conditioning(self):
        self.assertEqual(expected_question_indices("smoke", 0, 10), list(range(10)))
        self.assertEqual(expected_question_indices("full", 4, 100), list(range(100)))
        artifact = self.artifact()
        artifact["method_config"]["query_retrieval_disabled"] = True
        with self.assertRaisesRegex(ConditioningAblationContractError, "no-query-retrieval"):
            validate_artifact(artifact)

    def test_contract_rejects_conditioned_latents_or_query_mutation(self):
        artifact = self.artifact()
        artifact["records"][0]["query_invariants"]["conditioned_latent_count"] = 8
        with self.assertRaisesRegex(ConditioningAblationContractError, "conditioned"):
            validate_artifact(artifact)
        artifact = self.artifact()
        artifact["records"][0]["query_invariants"]["query_write_count"] = 1
        with self.assertRaisesRegex(ConditioningAblationContractError, "mutation"):
            validate_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
