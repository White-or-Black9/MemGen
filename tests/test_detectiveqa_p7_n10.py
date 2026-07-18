import unittest
from types import SimpleNamespace

from scripts.eval import detectiveqa_p7_n10 as target
from scripts.eval import mab3_bank_on_full_history as mab3


class DetectiveQAP7ConfigTest(unittest.TestCase):
    def test_p7_bank_config_matches_frozen_paper_values(self):
        config = target.p7_bank_config()
        self.assertTrue(config["enabled"])
        self.assertEqual(config["batch_size"], 1)
        self.assertEqual(config["max_slots"], 16)
        self.assertEqual(config["top_k"], 2)
        self.assertEqual(config["retrieve_policy"], "threshold_topk")
        self.assertEqual(config["update_policy"], "thread_update")
        self.assertEqual(config["retrieve_threshold"], 0.05)
        self.assertEqual(config["update_threshold"], 0.10)
        self.assertEqual(config["decay_alpha"], 0.05)
        self.assertEqual(config["storage_space"], "weaver")
        self.assertEqual(config["query_phase"], "read_only")


class DetectiveQAP7InvariantTest(unittest.TestCase):
    def test_expected_method_set_preserves_order_and_rejects_duplicates(self):
        self.assertEqual(
            target.expected_method_set("disabled,p7,p7_no_query_retrieval"),
            ["disabled", "p7", "p7_no_query_retrieval"],
        )
        with self.assertRaises(ValueError):
            target.expected_method_set("p7,p7")

    def test_validate_query_phase_invariants_rejects_query_write(self):
        with self.assertRaises(ValueError):
            target.validate_query_phase_invariants(
                {
                    "method": "p7",
                    "query_write_count": 1,
                    "bank_snapshot_changed_after_query": False,
                    "query_read_only_enforced": True,
                }
            )


class DetectiveQAP7ContractTest(unittest.TestCase):
    def test_record_from_result_captures_bank_fields(self):
        record = target.record_from_result(
            method="p7",
            payload={"context_id": "ctx", "query_id": 0, "gold_answers": ["x"]},
            result={
                "prediction": "x",
                "query_write_count": 0,
                "query_read_only_enforced": True,
                "bank_reset_after_context": True,
                "cross_context_leakage_detected": False,
                "retrieved_indices_by_turn": [[0, 1]],
                "retrieved_scores_by_turn": [[0.09, 0.08]],
                "bank_write_count": 9,
                "bank_retrieval_count": 1,
                "bank_retrieved_latent_count": 16,
                "bank_slot_count_final_before_reset": 9,
                "pre_query_bank_summary": {"slot_count": 9},
                "post_query_bank_summary": {"slot_count": 9},
            },
            score={"metrics": {"exact_match": True}, "additional": {"parsed_output": "x"}},
        )
        self.assertTrue(record["bank_created"])
        self.assertEqual(record["bank_retrieval_count"], 1)
        self.assertEqual(record["retrieved_indices_by_turn"], [[0, 1]])
        self.assertFalse(record["bank_snapshot_changed_after_query"])

    def test_expand_query_payloads_emits_all_queries_for_one_context(self):
        context_payload = {
            "context_id": "ctx",
            "context_index": 3,
            "question_count": 2,
            "dataset_config": {"sub_dataset": "detective_qa"},
            "queries": [
                {"query_id": 0, "question": "Q1", "query_prompt": "P1", "gold_answers": ["A1"]},
                {"query_id": 1, "question": "Q2", "query_prompt": "P2", "gold_answers": ["A2"]},
            ],
            "chunks": ["c1"],
            "chunk_token_lengths": [5],
            "memorization_prompts": ["m1"],
        }

        payloads = target.expand_query_payloads(context_payload)

        self.assertEqual(
            payloads,
            [
                {
                    "context_id": "ctx",
                    "context_index": 3,
                    "query_id": 0,
                    "question_count_in_context": 2,
                    "dataset_config": {"sub_dataset": "detective_qa"},
                    "question": "Q1",
                    "query_prompt": "P1",
                    "gold_answers": ["A1"],
                    "chunks": ["c1"],
                    "chunk_token_lengths": [5],
                    "memorization_prompts": ["m1"],
                },
                {
                    "context_id": "ctx",
                    "context_index": 3,
                    "query_id": 1,
                    "question_count_in_context": 2,
                    "dataset_config": {"sub_dataset": "detective_qa"},
                    "question": "Q2",
                    "query_prompt": "P2",
                    "gold_answers": ["A2"],
                    "chunks": ["c1"],
                    "chunk_token_lengths": [5],
                    "memorization_prompts": ["m1"],
                },
            ],
        )

    def test_disabled_query_response_length_defaults_to_generation_max_length(self):
        args = SimpleNamespace(generation_max_length=37)
        self.assertEqual(target.disabled_query_response_length(args), 37)

    def test_disabled_query_response_length_override_is_temporary(self):
        args = SimpleNamespace(
            cfg_path="configs/latent_memory/triviaqa.yaml",
            model_path="dummy-model",
            checkpoint_path="dummy-ckpt",
            seed=42,
            generation_max_length=40,
        )

        original_config = mab3._build_config(args, 32768, {"enabled": False})
        original_length = original_config["run"]["interaction"]["max_response_length"]
        original_interaction = mab3._interaction_config(original_config, 32768)
        self.assertEqual(original_length, 10)
        self.assertEqual(original_interaction.max_response_length, 10)

        with target.override_disabled_query_response_length(args):
            patched_config = mab3._build_config(args, 32768, {"enabled": False})
            patched_interaction = mab3._interaction_config(patched_config, 32768)
            self.assertEqual(
                patched_config["run"]["interaction"]["max_response_length"],
                40,
            )
            self.assertEqual(patched_interaction.max_response_length, 40)

        restored_config = mab3._build_config(args, 32768, {"enabled": False})
        restored_interaction = mab3._interaction_config(restored_config, 32768)
        self.assertEqual(restored_config["run"]["interaction"]["max_response_length"], 10)
        self.assertEqual(restored_interaction.max_response_length, 10)


if __name__ == "__main__":
    unittest.main()
