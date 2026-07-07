import math
import unittest

from scripts.eval.eventqa_method_separable_cost import (
    CostContractError,
    build_cost_summary,
    build_parser,
    build_p7_query_invariant,
    expected_question_indices,
    validate_cost_summary,
)


class EventQAMethodSeparableCostTest(unittest.TestCase):
    def test_scope_contract_distinguishes_smoke_and_full(self):
        self.assertEqual(expected_question_indices("smoke", 0, 10), list(range(10)))
        for context_index in range(5):
            self.assertEqual(
                expected_question_indices("full", context_index, 100),
                list(range(100)),
            )
        with self.assertRaisesRegex(CostContractError, "smoke scope"):
            expected_question_indices("smoke", 1, 10)
        with self.assertRaisesRegex(CostContractError, "full scope"):
            expected_question_indices("full", 0, 99)

    def test_p7_invariant_reads_retrieval_from_query_turn(self):
        result = {
            "query_write_count": 0,
            "bank_snapshot_changed_after_query": False,
            "generations": [
                {"retrieved_indices": [7]},
                {"retrieved_indices": [1, 0], "retrieved_latent_count": 16},
            ],
        }

        invariant = build_p7_query_invariant(result, query_index=3)

        self.assertEqual(
            invariant,
            {
                "query_index": 3,
                "query_write_count": 0,
                "bank_snapshot_changed_after_query": False,
                "retrieved_indices": [1, 0],
                "retrieved_latent_count": 16,
            },
        )

    def test_parser_requires_one_standalone_method(self):
        parser = build_parser()
        disabled = parser.parse_args(["--method", "disabled"])
        p7 = parser.parse_args(["--method", "p7"])
        self.assertEqual(disabled.method, "disabled")
        self.assertEqual(p7.method, "p7")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--method", "paired"])

    def test_disabled_summary_has_zero_construction_cost(self):
        summary = build_cost_summary(
            method="disabled",
            run_id="run-disabled",
            context_index=0,
            question_indices=list(range(10)),
            construction_latency_seconds=0.0,
            query_latencies_seconds=[1.0] * 10,
            end_to_end_latency_seconds=10.0,
            peak_gpu_memory_bytes=100,
            baseline_gpu_memory_bytes=80,
            output_tokens=[2] * 10,
            query_invariants=[],
            command=["python", "cost.py", "--method", "disabled"],
        )
        self.assertEqual(summary["schema_version"], "eventqa-method-cost/v1")
        self.assertEqual(summary["method"], "disabled")
        self.assertEqual(summary["scope"]["question_indices"], list(range(10)))
        self.assertEqual(summary["cost"]["construction_latency_seconds"], 0.0)
        self.assertEqual(summary["cost"]["query_latency_seconds"]["mean"], 1.0)
        self.assertEqual(summary["cost"]["peak_gpu_memory_bytes"], 100)
        self.assertEqual(summary["cost"]["incremental_peak_gpu_memory_bytes"], 20)
        validate_cost_summary(summary)

    def test_p7_summary_requires_read_only_query_invariants(self):
        invariants = [
            {
                "query_index": index,
                "query_write_count": 0,
                "bank_snapshot_changed_after_query": False,
                "retrieved_indices": [0, 1],
                "retrieved_latent_count": 16,
            }
            for index in range(10)
        ]
        summary = build_cost_summary(
            method="p7",
            run_id="run-p7",
            context_index=0,
            question_indices=list(range(10)),
            construction_latency_seconds=3.0,
            query_latencies_seconds=[1.2] * 10,
            end_to_end_latency_seconds=15.0,
            peak_gpu_memory_bytes=120,
            baseline_gpu_memory_bytes=80,
            output_tokens=[2] * 10,
            query_invariants=invariants,
            command=["python", "cost.py", "--method", "p7"],
        )
        self.assertEqual(summary["cost"]["construction_latency_seconds"], 3.0)
        self.assertTrue(summary["invariants"]["all_query_writes_zero"])
        self.assertTrue(summary["invariants"]["all_bank_snapshots_unchanged"])
        validate_cost_summary(summary)

    def test_p7_query_write_violation_is_rejected(self):
        invariants = [
            {
                "query_index": index,
                "query_write_count": 1 if index == 4 else 0,
                "bank_snapshot_changed_after_query": False,
                "retrieved_indices": [0, 1],
                "retrieved_latent_count": 16,
            }
            for index in range(10)
        ]
        with self.assertRaisesRegex(CostContractError, "query writes"):
            build_cost_summary(
                method="p7",
                run_id="run-p7",
                context_index=0,
                question_indices=list(range(10)),
                construction_latency_seconds=3.0,
                query_latencies_seconds=[1.2] * 10,
                end_to_end_latency_seconds=15.0,
                peak_gpu_memory_bytes=120,
                baseline_gpu_memory_bytes=80,
                output_tokens=[2] * 10,
                query_invariants=invariants,
                command=["python", "cost.py", "--method", "p7"],
            )

    def test_nonfinite_or_wrong_scope_is_rejected(self):
        with self.assertRaisesRegex(CostContractError, "q0-9"):
            build_cost_summary(
                method="disabled",
                run_id="run",
                context_index=0,
                question_indices=[0],
                construction_latency_seconds=0.0,
                query_latencies_seconds=[1.0],
                end_to_end_latency_seconds=1.0,
                peak_gpu_memory_bytes=100,
                baseline_gpu_memory_bytes=80,
                output_tokens=[2],
                query_invariants=[],
                command=["python"],
            )
        with self.assertRaisesRegex(CostContractError, "finite"):
            build_cost_summary(
                method="disabled",
                run_id="run",
                context_index=0,
                question_indices=list(range(10)),
                construction_latency_seconds=0.0,
                query_latencies_seconds=[math.nan] * 10,
                end_to_end_latency_seconds=1.0,
                peak_gpu_memory_bytes=100,
                baseline_gpu_memory_bytes=80,
                output_tokens=[2] * 10,
                query_invariants=[],
                command=["python"],
            )


if __name__ == "__main__":
    unittest.main()
