import unittest

from scripts.eval.factconsolidation_p7 import (
    FactConsolidationRunContractError,
    p7_bank_config,
    validate_batch_size,
    validate_context_start,
    validate_disabled_no_bank,
    validate_no_query_retrieval_construction,
    validate_query_phase_invariants,
    validate_run_invariants,
)


class FactConsolidationP7RunnerTest(unittest.TestCase):
    def test_disabled_passes_no_bank_to_generate(self):
        run = {
            "method": "disabled",
            "bank_created": False,
            "query_write_count": 0,
            "bank_reset_after_context": True,
            "cross_context_leakage_detected": False,
        }

        validate_disabled_no_bank(run)
        validate_run_invariants(run)

    def test_p7_uses_weaver_storage_and_weaver_retrieval_query(self):
        matrix = {
            "p7": {
                "retrieve_threshold": 0.05,
                "update_threshold": 0.10,
                "max_slots": 16,
                "top_k": 2,
                "decay_alpha": 0.05,
                "storage_space": "weaver",
                "query_phase": "read_only",
            }
        }

        config = p7_bank_config(matrix)

        self.assertTrue(config["enabled"])
        self.assertEqual(config["batch_size"], 1)
        self.assertEqual(config["storage_space"], "weaver")
        self.assertEqual(config["retrieve_policy"], "threshold_topk")
        self.assertEqual(config["update_policy"], "thread_update")
        self.assertEqual(config["query_phase"], "read_only")

    def test_each_context_starts_with_zero_slots(self):
        validate_context_start({"initial_slot_count": 0})
        with self.assertRaisesRegex(FactConsolidationRunContractError, "zero slots"):
            validate_context_start({"initial_slot_count": 1})

    def test_query_phase_blocks_write_and_preserves_snapshot(self):
        run = {
            "method": "p7",
            "bank_created": True,
            "query_write_count": 0,
            "bank_snapshot_changed_after_query": False,
            "query_read_only_enforced": True,
            "bank_reset_after_context": True,
            "cross_context_leakage_detected": False,
        }

        validate_query_phase_invariants(run)
        validate_run_invariants(run)

        broken = dict(run, query_write_count=1)
        with self.assertRaisesRegex(FactConsolidationRunContractError, "query write"):
            validate_query_phase_invariants(broken)

    def test_no_query_retrieval_keeps_identical_construction(self):
        shared = {
            "context_id": "ctx-0",
            "construction_bank_write_count": 8,
            "construction_final_slot_count": 8,
            "construction_turn_count": 2,
        }
        validate_no_query_retrieval_construction(
            dict(shared, method="p7"),
            dict(shared, method="p7_no_query_retrieval"),
        )

        with self.assertRaisesRegex(FactConsolidationRunContractError, "construction"):
            validate_no_query_retrieval_construction(
                dict(shared, method="p7"),
                dict(shared, method="p7_no_query_retrieval", construction_final_slot_count=7),
            )

    def test_batch_size_above_one_is_rejected(self):
        validate_batch_size(1)
        with self.assertRaisesRegex(FactConsolidationRunContractError, "batch_size=1"):
            validate_batch_size(2)

    def test_exception_path_resets_bank(self):
        run = {
            "method": "p7",
            "bank_created": True,
            "query_write_count": 0,
            "bank_snapshot_changed_after_query": False,
            "query_read_only_enforced": True,
            "bank_reset_after_context": False,
            "cross_context_leakage_detected": True,
        }

        with self.assertRaisesRegex(FactConsolidationRunContractError, "reset"):
            validate_run_invariants(run)


if __name__ == "__main__":
    unittest.main()
