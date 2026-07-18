import unittest
from types import SimpleNamespace
from unittest import mock

from scripts.eval import longbench_v2_p7 as runner
from memgen.model.modeling_memgen import build_constrained_choice_spec


def item(capacity_class="over_capacity"):
    return {
        "item_id": "item-1",
        "domain": "Multi-Document QA",
        "sub_domain": "Academic",
        "difficulty": "easy",
        "length": "short",
        "question": "Which option?",
        "choices": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
        "gold_choice": "A",
        "context": "First. Second.",
        "capacity_class": capacity_class,
    }


def chunks():
    return [
        {"text": "First. ", "token_count": 2},
        {"text": "Second.", "token_count": 2},
    ]


class FakeBank:
    def __init__(self):
        self.slots = 2

    def reset(self):
        self.slots = 0

    def __len__(self):
        return self.slots


class FakeChoiceTokenizer:
    sequences = {
        "A": [10, 20, 30],
        "B": [10, 21, 30],
        "C": [10, 22, 30],
        "D": [10, 23, 30],
    }

    def encode(self, text, add_special_tokens=False):
        return self.sequences[text[-2]]

    def decode(self, ids, skip_special_tokens=False):
        choice = next(key for key, value in self.sequences.items() if value == ids)
        return f"The correct answer is ({choice})"


class LongBenchV2P7RunnerTest(unittest.TestCase):
    def test_payload_keeps_context_out_of_query_and_in_construction(self):
        payload = runner.build_payload(item(), chunks())
        self.assertEqual(payload["chunks"], ["First. ", "Second."])
        self.assertNotIn(item()["context"], payload["query_prompt"])
        self.assertIn("chunk 1 of 2", payload["memorization_prompts"][0])
        query = runner.query_only_payload(payload)
        self.assertEqual(query["chunks"], [])
        self.assertEqual(query["memorization_prompts"], [payload["query_prompt"]])

    def test_strict_format_v2_prompt_is_explicit_and_tracked(self):
        payload = runner.build_payload(
            item(), chunks(), query_prompt_version="strict_format_v2",
        )
        self.assertEqual(payload["query_prompt_version"], "strict_format_v2")
        self.assertIn("Return exactly one line and nothing else", payload["query_prompt"])
        self.assertIn("Do not explain", payload["query_prompt"])

    def test_constrained_choice_v3_reuses_the_v1_query_text(self):
        v1 = runner.build_payload(item(), chunks(), query_prompt_version="v1")
        v3 = runner.build_payload(
            item(), chunks(), query_prompt_version="constrained_choice_v3",
        )
        self.assertEqual(v1["query_prompt"], v3["query_prompt"])
        self.assertEqual(v3["query_prompt_version"], "constrained_choice_v3")

    def test_choice_grammar_only_leaves_one_token_for_each_option(self):
        spec = build_constrained_choice_spec(FakeChoiceTokenizer())
        self.assertEqual(spec["prefix_token_ids"], [10])
        self.assertEqual(spec["choice_token_ids"], [20, 21, 22, 23])
        self.assertEqual(spec["suffix_token_ids"], [30])

    def test_disabled_rejects_over_capacity_item(self):
        with self.assertRaisesRegex(runner.LongBenchV2RunnerError, "invalid"):
            runner.run_disabled(SimpleNamespace(), object(), 32768, item(), runner.build_payload(item(), chunks()))

    def test_run_p7_method_resets_bank_and_records_query_invariants(self):
        bank = FakeBank()
        construction_result = {
            "_retained_bank": bank,
            "pre_query_bank_summary": {"write_count": 2, "slot_count": 2},
        }
        query_result = {
            "_retained_bank": bank,
            "prediction": "The correct answer is (A)",
            "query_write_count_delta": 0,
            "bank_snapshot_changed_after_query": False,
            "query_read_only_enforced": True,
            "retrieved_indices_by_turn": [[], [0, 1]],
            "pre_query_bank_summary": {"write_count": 2, "slot_count": 2},
            "post_query_bank_summary": {"write_count": 2, "slot_count": 2},
        }
        with mock.patch.object(runner.eventqa, "_run_eventqa_model", side_effect=[construction_result, query_result]):
            record, construction = runner.run_p7_method(
                SimpleNamespace(), object(), 32768, item(), runner.build_payload(item(), chunks()),
                "p7", {"enabled": True},
            )
        self.assertEqual(record["strict_correct"], 1)
        self.assertEqual(record["retrieved_latent_count"], 16)
        self.assertEqual(record["post_reset_slot_count"], 0)
        self.assertEqual(construction["construction_turn_count"], 2)

    def test_run_p7_pair_reuses_one_frozen_construction(self):
        bank = FakeBank()
        construction = {
            "_retained_bank": bank,
            "pre_query_bank_summary": {"write_count": 2, "slot_count": 2},
        }
        p7_query = {
            "_retained_bank": bank,
            "prediction": "The correct answer is (A)",
            "query_write_count_delta": 0,
            "bank_snapshot_changed_after_query": False,
            "query_read_only_enforced": True,
            "retrieved_indices_by_turn": [[], [0]],
            "pre_query_bank_summary": {"write_count": 2, "slot_count": 2},
            "post_query_bank_summary": {"write_count": 2, "slot_count": 2},
        }
        no_query = {
            **p7_query,
            "_retained_bank": bank,
            "retrieved_indices_by_turn": [[], []],
        }
        runtime_args = SimpleNamespace(constrained_choice=True)
        p7_query["generations"] = [{"constrained_choice": True}]
        no_query["generations"] = [{"constrained_choice": True}]
        with mock.patch.object(
            runner.eventqa, "_run_eventqa_model", side_effect=[construction, p7_query, no_query]
        ) as mocked:
            records, constructions = runner.run_p7_pair(
                runtime_args, object(), 32768, item(), runner.build_payload(item(), chunks()),
                {"enabled": True},
            )
        self.assertEqual(mocked.call_count, 3)
        self.assertFalse(mocked.call_args_list[0].args[0].constrained_choice)
        self.assertEqual([row["method"] for row in records], ["p7", "p7_no_query_retrieval"])
        self.assertEqual(records[0]["construction_hash"], records[1]["construction_hash"])
        self.assertEqual(records[0]["retrieved_latent_count"], 8)
        self.assertEqual(records[1]["retrieved_latent_count"], 0)
        self.assertTrue(all(row["constrained_choice_active"] for row in records))
        self.assertEqual(len(constructions), 2)
        self.assertEqual(len(bank), 0)

    def test_aggregate_accepts_window_fit_triple_and_over_capacity_pair(self):
        records = []
        for capacity_class in ("window_fit", "over_capacity"):
            current = item(capacity_class)
            current["item_id"] = capacity_class
            payload = runner.build_payload(current, chunks())
            for method in ("p7", "p7_no_query_retrieval"):
                record = runner.common_record(current, method, payload, "The correct answer is (A)")
                record.update({
                    "construction_hash": "same",
                    "query_write_count": 0,
                    "bank_snapshot_changed_after_query": False,
                    "post_reset_slot_count": 0,
                    "retrieved_latent_count": 16 if method == "p7" else 0,
                })
                records.append(record)
            if capacity_class == "window_fit":
                disabled = runner.common_record(current, "disabled_window_fit", payload, "The correct answer is (A)")
                disabled.update({
                    "construction_hash": None,
                    "query_write_count": 0,
                    "bank_snapshot_changed_after_query": False,
                    "post_reset_slot_count": 0,
                    "retrieved_latent_count": 0,
                })
                records.append(disabled)
        summary = runner.aggregate(records, [])
        self.assertTrue(summary["contract"]["contract_valid"])
        self.assertEqual(summary["contract"]["item_count"], 2)

    def test_parse_methods_allows_p7_only_but_requires_p7_for_no_query(self):
        self.assertEqual(runner.parse_methods("p7"), ("p7",))
        self.assertEqual(
            runner.parse_methods("p7,p7_no_query_retrieval"),
            ("p7", "p7_no_query_retrieval"),
        )
        with self.assertRaisesRegex(runner.LongBenchV2RunnerError, "requires p7"):
            runner.parse_methods("p7_no_query_retrieval")

    def test_select_item_slice_uses_manifest_indices(self):
        rows = [{"item_id": str(index)} for index in range(5)]
        self.assertEqual(
            [row["item_id"] for row in runner.select_item_slice(rows, 1, 4)],
            ["1", "2", "3"],
        )
        with self.assertRaisesRegex(runner.LongBenchV2RunnerError, "invalid item slice"):
            runner.select_item_slice(rows, 4, 4)

    def test_partial_method_aggregate_is_explicitly_pending_merge(self):
        current = item()
        payload = runner.build_payload(current, chunks())
        records = []
        for method in runner.P7_METHODS:
            record = runner.common_record(current, method, payload, "The correct answer is (A)")
            records.append(record)
        summary = runner.aggregate(records, [], validate_contract=False)
        self.assertIsNone(summary["contract"]["contract_valid"])
        self.assertEqual(summary["contract"]["status"], "pending_method_shard_merge")


if __name__ == "__main__":
    unittest.main()
