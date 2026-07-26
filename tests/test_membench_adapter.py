import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval.membench_adapter import (
    inspect_dataset,
    normalize_dataset,
    render_message,
    score_choice,
)


def sample_dataset():
    return {
        "roles": [
            {
                "tid": 7,
                "message_list": [
                    [
                        {
                            "sid": 0,
                            "user_message": "My sister enjoys camping.",
                            "assistant_message": "I will remember that.",
                            "time": "2024-10-01",
                            "place": "Boston",
                        },
                        {
                            "sid": 1,
                            "user_message": "Her hobby changed to hiking.",
                            "assistant_message": "Updated.",
                            "time": "2024-10-02",
                            "place": "Boston",
                        },
                    ]
                ],
                "QA": {
                    "qid": 3,
                    "question": "What hobby does my sister enjoy now?",
                    "answer": "Hiking",
                    "target_step_id": [[1, 0]],
                    "choices": {
                        "A": "Camping",
                        "B": "Hiking",
                        "C": "Traveling",
                        "D": "Fishing",
                    },
                    "ground_truth": "B",
                    "time": "2024-10-03",
                },
            }
        ]
    }


class MemBenchAdapterTest(unittest.TestCase):
    def test_normalize_preserves_turn_order_and_choice_contract(self):
        records = normalize_dataset(sample_dataset())

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["context_id"], "membench-roles-7")
        self.assertEqual([turn["source_step_id"] for turn in record["construction_turns"]], [0, 1])
        self.assertIn("User: My sister enjoys camping.", record["construction_turns"][0]["content"])
        self.assertIn("Time: 2024-10-01", record["construction_turns"][0]["content"])
        self.assertEqual(record["query"]["query_id"], 3)
        self.assertEqual(record["query"]["gold_choice"], "B")
        self.assertEqual(record["query"]["choices"]["B"], "Hiking")

    def test_invalid_choice_contract_fails_loudly(self):
        payload = sample_dataset()
        del payload["roles"][0]["QA"]["choices"]["D"]

        with self.assertRaisesRegex(ValueError, "A/B/C/D"):
            normalize_dataset(payload)

    def test_strict_scorer_does_not_normalize_predictions(self):
        self.assertEqual(score_choice("B", "B")["official_choice_exact_match"], 1.0)
        wrong_format = score_choice(" B\n", "B")
        self.assertEqual(wrong_format["official_choice_exact_match"], 0.0)
        self.assertFalse(wrong_format["valid_choice_output"])

    def test_render_message_supports_flat_user_agent_schema(self):
        rendered = render_message(
            {"user": "hello", "agent": "hi", "time": "now", "place": "lab"}
        )
        self.assertEqual(rendered, "User: hello\nAssistant: hi\nTime: now\nPlace: lab")

    def test_inspect_records_metric_and_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "knowledge_update.json"
            path.write_text(json.dumps(sample_dataset()), encoding="utf-8")
            report = inspect_dataset(path, subset="knowledge_update")

        self.assertEqual(report["trajectory_count"], 1)
        self.assertEqual(report["question_count"], 1)
        self.assertEqual(report["primary_metric"], "official_choice_exact_match")
        self.assertEqual(report["category_counts"], {"roles": 1})
        self.assertEqual(len(report["dataset_sha256"]), 64)

    def test_low_turn_official_subset_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "highlevel.json"
            path.write_text(json.dumps(sample_dataset()), encoding="utf-8")
            report = inspect_dataset(path, subset="highlevel")

        self.assertEqual(report["subset"], "highlevel")


if __name__ == "__main__":
    unittest.main()
