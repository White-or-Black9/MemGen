import unittest

from scripts.eval.eventqa_p7_direct_top1 import (
    DirectTop1ContractError,
    validate_artifact,
)


class EventQAP7DirectTop1ArtifactTest(unittest.TestCase):
    def artifact(self):
        return {
            "schema_version": "eventqa-p7-direct-top1/v1",
            "method_config": {
                "query_latent_usage": "direct_top1",
                "query_retrieval_disabled": False,
            },
            "records": [
                {
                    "query_invariants": {
                        "retrieved_slot_count": 1,
                        "query_weaver_invoke_count": 0,
                    }
                }
            ],
        }

    def test_valid_direct_top1_artifact_is_accepted(self):
        validate_artifact(self.artifact())

    def test_contract_rejects_disabled_retrieval_or_multiple_slots_or_weaver(self):
        artifact = self.artifact()
        artifact["method_config"]["query_retrieval_disabled"] = True
        with self.assertRaisesRegex(DirectTop1ContractError, "disable"):
            validate_artifact(artifact)
        artifact = self.artifact()
        artifact["records"][0]["query_invariants"]["retrieved_slot_count"] = 2
        with self.assertRaisesRegex(DirectTop1ContractError, "invalid"):
            validate_artifact(artifact)
        artifact = self.artifact()
        artifact["records"][0]["query_invariants"]["query_weaver_invoke_count"] = 1
        with self.assertRaisesRegex(DirectTop1ContractError, "invalid"):
            validate_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
