import unittest

from scripts.eval.eventqa_final_table_package import (
    FinalTablePackageError,
    build_package,
)


class EventQAFinalTablePackageTest(unittest.TestCase):
    def test_builds_main_table_cost_table_and_claim_audit(self):
        paper = {
            "methods": [
                {
                    "method_id": "bank_off",
                    "display_name": "Disabled / compressed Bank-off",
                    "repeat_count": 5,
                    "metrics": {
                        "em": {"mean": 0.008, "std": 0.0},
                        "recall": {"mean": 0.178, "std": 0.0},
                        "format_failures": {"mean": 377.0, "std": 0.0},
                    },
                },
                {
                    "method_id": "p6",
                    "display_name": "P6 non-strict",
                    "repeat_count": 5,
                    "metrics": {
                        "em": {"mean": 0.1688, "std": 0.018},
                        "recall": {"mean": 0.258, "std": 0.016},
                        "format_failures": {"mean": 165.8, "std": 19.8},
                    },
                },
                {
                    "method_id": "p7",
                    "display_name": "Frozen P7 non-strict",
                    "repeat_count": 5,
                    "metrics": {
                        "em": {"mean": 0.1968, "std": 0.020},
                        "recall": {"mean": 0.2536, "std": 0.028},
                        "format_failures": {"mean": 121.4, "std": 8.8},
                    },
                },
            ]
        }
        cost = {
            "methods": {
                "disabled": {
                    "construction_latency_seconds_total": 0.0,
                    "end_to_end_latency_seconds_total": 367.448,
                    "amortized_end_to_end_seconds_per_question": 0.735,
                    "incremental_peak_gpu_memory_bytes_max": 142900000,
                },
                "p7": {
                    "construction_latency_seconds_total": 78.454,
                    "end_to_end_latency_seconds_total": 387.999,
                    "amortized_end_to_end_seconds_per_question": 0.776,
                    "incremental_peak_gpu_memory_bytes_max": 171900000,
                },
            }
        }
        bm25 = {
            "effectiveness": {
                "substring_exact_match": 0.03,
                "eventqa_recall": 0.226,
                "format_failure_count": 265,
            },
            "cost": {
                "index_construction_latency_seconds": 5.0,
                "retrieval_latency_seconds": 10.0,
                "method_total_seconds": 692.845,
                "amortized_seconds_per_question": 1.386,
                "incremental_peak_gpu_memory_bytes_max": 3597000000,
            },
        }
        matched16 = {
            "effectiveness": {
                "substring_exact_match": 0.068,
                "eventqa_recall": 0.18,
                "format_failure_count": 347,
            },
            "cost": {
                "index_construction_latency_seconds": 7.0,
                "retrieval_and_window_latency_seconds": 9.0,
                "method_total_seconds": 501.761,
                "amortized_seconds_per_question": 1.004,
                "incremental_peak_gpu_memory_bytes_max": 171000000,
            },
        }
        text_summary = {
            "effectiveness": {
                "substring_exact_match": 0.012,
                "eventqa_recall": 0.078,
                "format_failure_count": 267,
            },
            "cost": {
                "construction_latency_seconds": 223.371,
                "end_to_end_latency_seconds": 691.345,
                "end_to_end_amortized_seconds_per_question": 1.383,
                "query_incremental_peak_gpu_memory_bytes_max": 210000000,
                "construction_incremental_peak_gpu_memory_bytes_max": 1920000000,
                "paper_facing": False,
                "confounded_by_shared_gpu": True,
                "caveat": "shared-GPU diagnostic cost only",
            },
        }
        repeated_controls = {
            "schema_version": "eventqa-explicit-controls-repeat-aggregate/v1",
            "methods": [
                {
                    "method_id": method_id,
                    "repeat_count": 5,
                    "metrics": {
                        "em": {"mean": em, "std": 0.0},
                        "recall": {"mean": recall, "std": 0.0},
                        "format_failures": {"mean": failures, "std": 0.0},
                    },
                }
                for method_id, em, recall, failures in (
                    ("text_summary", 0.012, 0.078, 267.0),
                    ("bm25_top2", 0.03, 0.226, 265.0),
                    ("matched16", 0.068, 0.18, 347.0),
                )
            ],
        }
        no_query = {
            "effectiveness": {
                "substring_exact_match": 0.008,
                "eventqa_recall": 0.178,
                "format_failure_count": 377,
            },
            "cost": {
                "construction_latency_seconds": 71.804,
                "end_to_end_latency_seconds": 445.004,
                "end_to_end_amortized_seconds_per_question": 0.890,
                "incremental_peak_gpu_memory_bytes_max": 149800000,
            },
            "invariants": {"all_queries_disable_retrieval": True},
        }

        result = build_package(
            paper_aggregate=paper,
            cost_aggregate=cost,
            bm25_aggregate=bm25,
            matched16_aggregate=matched16,
            text_summary_aggregate=text_summary,
            explicit_controls_repeat_aggregate=repeated_controls,
            no_query_aggregate=no_query,
        )

        self.assertEqual(len(result["main_table"]), 7)
        self.assertEqual(result["main_table"][0]["method_id"], "bank_off")
        self.assertEqual(result["main_table"][-1]["method_id"], "p7")
        self.assertEqual(result["main_table"][1]["repeat_count"], 5)
        self.assertEqual(result["explicit_controls"][-1]["method_id"], "p7")
        self.assertEqual(result["cost_table"][0]["method_id"], "bank_off")
        self.assertFalse(result["cost_table"][1]["paper_facing_cost"])
        self.assertTrue(result["claim_audit"]["claims"]["p7_vs_disabled"]["supported"])
        self.assertTrue(
            result["claim_audit"]["claims"]["query_time_retrieval_is_necessary"]["supported"]
        )
        self.assertFalse(result["claim_audit"]["claims"]["p7_cost_superiority"]["supported"])

    def test_requires_bank_off_p6_p7_methods(self):
        with self.assertRaisesRegex(FinalTablePackageError, "missing method"):
            build_package(
                paper_aggregate={"methods": []},
                cost_aggregate={"methods": {}},
                bm25_aggregate={"effectiveness": {}, "cost": {}},
                matched16_aggregate={"effectiveness": {}, "cost": {}},
                text_summary_aggregate={"effectiveness": {}, "cost": {}},
                explicit_controls_repeat_aggregate={"schema_version": "eventqa-explicit-controls-repeat-aggregate/v1", "methods": []},
                no_query_aggregate={"effectiveness": {}, "cost": {}, "invariants": {}},
            )


if __name__ == "__main__":
    unittest.main()
