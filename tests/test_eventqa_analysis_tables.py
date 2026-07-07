import unittest

from scripts.eval.eventqa_analysis_tables import (
    EventQAAnalysisTablesError,
    build_tables,
)


class EventQAAnalysisTablesTest(unittest.TestCase):
    def test_build_tables(self):
        paper = {
            "methods": [
                {
                    "method_id": "bank_off",
                    "display_name": "Disabled / compressed Bank-off",
                    "repeat_count": 5,
                    "metrics": {
                        "helpful_memory_count": {"mean": 0.0, "std": 0.0, "values": [0, 0, 0, 0, 0]},
                        "harmful_memory_count": {"mean": 0.0, "std": 0.0, "values": [0, 0, 0, 0, 0]},
                        "format_harm_count": {"mean": 0.0, "std": 0.0, "values": [0, 0, 0, 0, 0]},
                    },
                    "per_context": {
                        str(i): {
                            "em": {"mean": 0.01 * i, "std": 0.0},
                            "recall": {"mean": 0.10 + 0.01 * i, "std": 0.0},
                            "format_failures": {"mean": 70 + i, "std": 0.0},
                        }
                        for i in range(5)
                    },
                },
                {
                    "method_id": "p6",
                    "display_name": "P6",
                    "repeat_count": 5,
                    "metrics": {
                        "helpful_memory_count": {"mean": 80.0, "std": 1.0, "values": [79, 80, 80, 81, 80]},
                        "harmful_memory_count": {"mean": 4.0, "std": 0.0, "values": [4, 4, 4, 4, 4]},
                        "format_harm_count": {"mean": 20.0, "std": 1.0, "values": [19, 20, 20, 21, 20]},
                    },
                    "per_context": {
                        str(i): {
                            "em": {"mean": 0.10 + 0.01 * i, "std": 0.01},
                            "recall": {"mean": 0.20 + 0.01 * i, "std": 0.02},
                            "format_failures": {"mean": 20 + i, "std": 1.0},
                        }
                        for i in range(5)
                    },
                },
                {
                    "method_id": "p7",
                    "display_name": "P7",
                    "repeat_count": 5,
                    "metrics": {
                        "helpful_memory_count": {"mean": 95.0, "std": 2.0, "values": [92, 94, 95, 97, 97]},
                        "harmful_memory_count": {"mean": 3.0, "std": 1.0, "values": [2, 2, 3, 4, 4]},
                        "format_harm_count": {"mean": 10.0, "std": 1.0, "values": [9, 10, 10, 11, 10]},
                    },
                    "per_context": {
                        str(i): {
                            "em": {"mean": 0.15 + 0.02 * i, "std": 0.02},
                            "recall": {"mean": 0.25 + 0.01 * i, "std": 0.03},
                            "format_failures": {"mean": 10 + i, "std": 2.0},
                        }
                        for i in range(5)
                    },
                },
            ]
        }

        tables = build_tables(paper_aggregate=paper)
        self.assertEqual(len(tables["contextwise_table"]), 5)
        self.assertEqual(tables["contextwise_table"][0]["context_id"], "ctx0")
        self.assertEqual(tables["transition_table"][0]["method_id"], "p6")
        self.assertEqual(tables["transition_table"][1]["method_id"], "p7")
        self.assertEqual(tables["transition_table"][0]["unchanged_mean"], 416.0)
        self.assertEqual(tables["transition_table"][1]["net_mean"], 92.0)

    def test_requires_methods(self):
        with self.assertRaisesRegex(EventQAAnalysisTablesError, "missing method"):
            build_tables(paper_aggregate={"methods": []})


if __name__ == "__main__":
    unittest.main()
