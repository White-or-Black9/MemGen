import copy
import unittest

from scripts.eval.eventqa_matched16_aggregate import (
    Matched16AggregateError,
    aggregate_artifacts,
)
from tests.test_eventqa_matched16_retrieved_text import Matched16ArtifactTest


def make_artifact(context_index: int) -> dict:
    artifact = Matched16ArtifactTest.valid_artifact()
    template = artifact["records"][0]
    artifact["run_id"] = f"run-{context_index}"
    artifact["scope"] = {
        "measurement_scope": "full",
        "context_index": context_index,
        "question_indices": list(range(100)),
    }
    artifact["records"] = []
    for query_index in range(100):
        record = copy.deepcopy(template)
        record["context_index"] = context_index
        record["query_index"] = query_index
        record["substring_exact_match"] = int(query_index % 2 == 0)
        record["eventqa_recall"] = 0.25
        record["format_flags"] = {"multiline_output": query_index == 0}
        artifact["records"].append(record)
    artifact["cost"]["incremental_peak_gpu_memory_bytes"] = 100
    return artifact


class Matched16AggregateTest(unittest.TestCase):
    def test_aggregates_five_contexts_and_500_questions(self):
        result = aggregate_artifacts([make_artifact(index) for index in range(5)])
        self.assertEqual(result["scope"]["question_count"], 500)
        self.assertEqual(result["effectiveness"]["substring_exact_match"], 0.5)
        self.assertEqual(result["effectiveness"]["eventqa_recall"], 0.25)
        self.assertEqual(result["effectiveness"]["format_failure_count"], 5)
        self.assertEqual(result["budget"]["source_token_counts"], [16])
        self.assertEqual(result["budget"]["rendered_prompt_token_deltas"], [16])
        self.assertAlmostEqual(result["cost"]["method_total_seconds"], 255.05)

    def test_rejects_missing_or_duplicate_context(self):
        with self.assertRaisesRegex(Matched16AggregateError, "contexts 0-4"):
            aggregate_artifacts([make_artifact(index) for index in range(4)])
        artifacts = [make_artifact(index) for index in range(5)]
        artifacts[-1] = make_artifact(0)
        with self.assertRaisesRegex(Matched16AggregateError, "contexts 0-4"):
            aggregate_artifacts(artifacts)

    def test_rejects_any_budget_drift(self):
        artifacts = [make_artifact(index) for index in range(5)]
        artifacts[2]["records"][50]["rendered_prompt_token_delta"] = 17
        artifacts[2]["records"][50]["matched_rendered_token_count"] = 117
        with self.assertRaisesRegex(Matched16AggregateError, "16"):
            aggregate_artifacts(artifacts)


if __name__ == "__main__":
    unittest.main()
