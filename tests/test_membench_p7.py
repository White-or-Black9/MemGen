import unittest

from scripts.eval import membench_p7 as p7


def record():
    return {
        "category": "roles",
        "context_id": "membench-roles-7",
        "trajectory_id": 7,
        "construction_turns": [
            {"turn_index": 0, "source_step_id": 0, "content": "User: Alex likes hiking."},
            {"turn_index": 1, "source_step_id": 1, "content": "Assistant: Noted."},
        ],
        "query": {
            "query_id": 3,
            "question": "What does Alex like?",
            "choices": {"A": "Hiking", "B": "Fishing", "C": "Music", "D": "Art"},
            "gold_choice": "A",
        },
    }


def paired_rows():
    base = {
        "context_id": "membench-roles-7", "query_write_count": 0,
        "bank_snapshot_changed_after_query": False, "post_reset_slot_count": 0,
    }
    return [
        {**base, "method": "no_memory", "construction_hash": None, "construction_bank_write_count": 0, "construction_final_slot_count": 0, "retrieved_latent_count": 0},
        {**base, "method": "text_full_history", "construction_hash": None, "construction_bank_write_count": 0, "construction_final_slot_count": 0, "retrieved_latent_count": 0},
        {**base, "method": "p7", "construction_hash": "same", "construction_bank_write_count": 2, "construction_final_slot_count": 1, "retrieved_latent_count": 8},
        {**base, "method": "p7_no_query_retrieval", "construction_hash": "same", "construction_bank_write_count": 2, "construction_final_slot_count": 1, "retrieved_latent_count": 0},
    ]


class MemBenchP7Test(unittest.TestCase):
    def test_payload_uses_official_query_template_and_keeps_turns_out_of_query(self):
        payload = p7.build_payload(record())
        self.assertEqual(payload["chunks"], ["User: Alex likes hiking.", "Assistant: Noted."])
        self.assertEqual(len(payload["memorization_prompts"]), 2)
        self.assertTrue(payload["memorization_prompts"][0].startswith(p7.OFFICIAL_INITIAL_INSTRUCTION))
        self.assertFalse(payload["memorization_prompts"][1].startswith(p7.OFFICIAL_INITIAL_INSTRUCTION))
        self.assertNotIn("Alex likes hiking", payload["query_prompt"])
        self.assertIn("Past memory: \n", payload["query_prompt"])
        self.assertIn("A. Hiking", payload["query_prompt"])
        self.assertIn("Example: D", payload["query_prompt"])
        self.assertEqual(p7.query_only_payload(payload)["chunks"], [])

    def test_action_adapter_matches_only_the_fixed_memgen_grammar(self):
        row = p7.result_record(record(), "no_memory", "The correct answer is (A)")
        self.assertEqual(row["official_choice_exact_match"], 1.0)
        self.assertEqual(row["action_response"], "A")
        self.assertEqual(row["action_adapter"], "memgen_exact_choice_grammar")
        row = p7.result_record(record(), "no_memory", " A\n")
        self.assertEqual(row["official_choice_exact_match"], 0.0)
        self.assertFalse(row["valid_choice_output"])

    def test_text_full_history_populates_only_the_official_memory_field(self):
        item = record()
        prompt = p7.render_query_prompt(
            item, memory_text=p7.full_history_memory_text(item)
        )
        self.assertIn("Past memory: User: Alex likes hiking.", prompt)
        self.assertIn("Question: (current time is ) What does Alex like?", prompt)

    def test_pair_contract_requires_shared_construction_and_zero_no_query_retrieval(self):
        p7.validate_paired_records(paired_rows())
        rows = paired_rows()
        next(row for row in rows if row["method"] == "p7_no_query_retrieval")["retrieved_latent_count"] = 8
        with self.assertRaisesRegex(p7.MemBenchRunContractError, "no-query"):
            p7.validate_paired_records(rows)

    def test_aggregate_uses_official_exact_match(self):
        rows = []
        for index, method in enumerate(p7.METHODS):
            rows.append({
                "method": method,
                "official_choice_exact_match": float(index == 0),
                "valid_choice_output": method != "p7_no_query_retrieval",
                "retrieved_latent_count": 8 if method == "p7" else 0,
            })
        metrics = p7.aggregate(rows)
        self.assertEqual(metrics["no_memory"]["official_choice_exact_match"], 1.0)
        self.assertEqual(metrics["p7_no_query_retrieval"]["invalid_choice_output_count"], 1)

    def test_default_generation_budget_supports_constrained_choice(self):
        args = p7.build_parser().parse_args(["--dataset", "simple.json", "--subset", "simple"])
        self.assertEqual(args.generation_max_length, 12)


if __name__ == "__main__":
    unittest.main()
