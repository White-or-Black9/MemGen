import unittest

from scripts.eval import longbench_v2_contract as contract


def record(method, *, capacity_class="over_capacity"):
    return {
        "item_id": "item-1",
        "method": method,
        "query_prompt_version": "v1",
        "capacity_class": capacity_class,
        "prompt_hash": "prompt",
        "question_hash": "question",
        "choices_hash": "choices",
        "construction_hash": "construction",
        "query_write_count": 0,
        "bank_snapshot_changed_after_query": False,
        "post_reset_slot_count": 0,
        "retrieved_latent_count": 2 if method == "p7" else 0,
    }


class LongBenchV2RunContractTest(unittest.TestCase):
    def test_over_capacity_pair_passes_without_disabled(self):
        summary = contract.validate_aligned_records([
            record("p7"), record("p7_no_query_retrieval"),
        ])
        self.assertTrue(summary["contract_valid"])
        self.assertEqual(summary["retrieval_positive_items"], 1)

    def test_window_fit_item_requires_disabled_comparator(self):
        rows = [
            record("p7", capacity_class="window_fit"),
            record("p7_no_query_retrieval", capacity_class="window_fit"),
        ]
        with self.assertRaisesRegex(contract.LongBenchV2RunContractError, "lacks Disabled"):
            contract.validate_aligned_records(rows)
        rows.append(record("disabled_window_fit", capacity_class="window_fit"))
        self.assertTrue(contract.validate_aligned_records(rows)["contract_valid"])

    def test_over_capacity_item_rejects_invalid_disabled_comparator(self):
        rows = [record("p7"), record("p7_no_query_retrieval"), record("disabled_window_fit")]
        with self.assertRaisesRegex(contract.LongBenchV2RunContractError, "invalid Disabled"):
            contract.validate_aligned_records(rows)

    def test_rejects_construction_mismatch_query_write_and_snapshot_change(self):
        for field, value, message in (
            ("construction_hash", "different", "construction mismatch"),
            ("query_write_count", 1, "query write"),
            ("bank_snapshot_changed_after_query", True, "snapshot changed"),
        ):
            p7 = record("p7")
            p7[field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(contract.LongBenchV2RunContractError, message):
                    contract.validate_aligned_records([p7, record("p7_no_query_retrieval")])

    def test_no_query_method_must_not_retrieve(self):
        no_query = record("p7_no_query_retrieval")
        no_query["retrieved_latent_count"] = 1
        with self.assertRaisesRegex(contract.LongBenchV2RunContractError, "retrieval was not disabled"):
            contract.validate_aligned_records([record("p7"), no_query])

    def test_constrained_protocol_requires_every_method_to_activate_it(self):
        p7 = record("p7")
        no_query = record("p7_no_query_retrieval")
        for row in (p7, no_query):
            row["query_prompt_version"] = "constrained_choice_v3"
            row["constrained_choice_active"] = True
        self.assertTrue(contract.validate_aligned_records([p7, no_query])["contract_valid"])
        no_query["constrained_choice_active"] = False
        with self.assertRaisesRegex(contract.LongBenchV2RunContractError, "constrained choice"):
            contract.validate_aligned_records([p7, no_query])


if __name__ == "__main__":
    unittest.main()
