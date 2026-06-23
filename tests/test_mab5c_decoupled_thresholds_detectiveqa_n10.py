import unittest

from scripts.eval import mab5c_decoupled_thresholds_detectiveqa_n10 as harness


class MAB5CDecoupledThresholdsDetectiveQAN10Test(unittest.TestCase):
    def test_build_manifest_uses_decoupled_threshold_contract(self):
        class Args:
            dataset_root = "/data"
            mab_repo = "/repo"
            checkpoint_path = "/tmp/checkpoint"
            model_checkpoint_id = "ckpt"

        manifest = harness._build_manifest(
            "run",
            Args(),
            "now",
            git_status_before="before",
            git_status_after="after",
        )

        self.assertEqual(manifest["experiment_name"], harness.EXPERIMENT_NAME)
        self.assertEqual(manifest["query_mode"], "first-query-only")
        self.assertEqual(manifest["full_history_policy"], "over_capacity_invalid")
        self.assertEqual(manifest["threshold"], 0.03)
        self.assertEqual(manifest["retrieve_threshold"], 0.03)
        self.assertEqual(manifest["update_threshold"], 0.05)

    def test_parser_defaults_keep_distinct_output_root(self):
        args = harness.build_parser().parse_args([])

        self.assertEqual(args.output_root, "outputs/mab/decoupled_thresholds_detectiveqa_n10")

    def test_bank_config_uses_split_thresholds(self):
        config = harness._bank_config()

        self.assertEqual(config["threshold"], 0.03)
        self.assertEqual(config["retrieve_threshold"], 0.03)
        self.assertEqual(config["update_threshold"], 0.05)
        self.assertEqual(config["max_slots"], 8)
        self.assertEqual(config["top_k"], 1)
        self.assertEqual(config["retrieve_policy"], "threshold_topk")
        self.assertEqual(config["update_policy"], "thread_update")

    def test_row_and_aggregate_include_split_threshold_diagnostics(self):
        payload = {
            "context_id": "ctx-0",
            "chunks": ["c1"],
            "chunk_token_lengths": [8],
            "gold_answers": ["gold"],
        }
        final_generation = {
            "retrieved_latent_count": 1,
            "retrieved_latents_enter_reasoner": True,
            "retrieved_latents_enter_weaver": False,
            "retrieved_indices": [0],
            "retrieved_scores": [0.04],
            "bank_debug": {
                "effective_retrieve_threshold": 0.03,
                "effective_update_threshold": 0.05,
                "last_write_back": {
                    "retrieve_threshold_passed": True,
                    "update_threshold_passed": False,
                },
                "write_action_counts": {"insert": 2},
                "update_reason_counts": {"empty_bank": 1, "new_thread": 1},
                "thread_insert_count": 2,
                "matched_replace_count": 0,
                "capacity_evict_count": 0,
            },
        }
        bank_on_result = {
            "prediction": "pred",
            "prompt_trace": [{"query_prompt_contains_chunk_text": False, "query_prompt_contains_ack_history": False}],
            "generations": [final_generation],
            "bank_write_count": 2,
            "bank_retrieval_count": 2,
            "bank_retrieved_latent_count": 1,
            "query_write_count": 0,
            "query_write_attempt_count": 0,
            "bank_slot_count_final_before_reset": 8,
            "bank_reset_after_context": True,
            "cross_context_leakage_detected": False,
            "retrieved_indices_by_turn": [[0]],
            "retrieved_scores_by_turn": [[0.04]],
        }
        bank_off_result = {
            "prediction": "off",
            "context_capacity": 1024,
            "bank_write_count": 0,
            "bank_retrieval_count": 0,
            "latency_seconds": 1.0,
            "peak_cuda_memory": None,
        }
        bank_on_result["latency_seconds"] = 2.0
        bank_off_score = {"metrics": {"exact_match": 0.0}}
        bank_on_score = {"metrics": {"exact_match": 0.0}}

        row = harness._build_row(
            run_id="run",
            context_index=0,
            payload=payload,
            bank_off_result=bank_off_result,
            bank_on_result=bank_on_result,
            bank_off_score=bank_off_score,
            bank_on_score=bank_on_score,
            estimated_full_history_query_tokens=11,
            compressed_query_tokens_bank_off=9,
            compressed_query_tokens_bank_on=7,
        )
        summary = harness._aggregate([row])

        self.assertEqual(row["query_turn_retrieval_active"], 1)
        self.assertEqual(row["query_turn_retrieved_latent_count"], 1)
        self.assertEqual(row["bank_on_effective_retrieve_threshold"], 0.03)
        self.assertEqual(row["bank_on_effective_update_threshold"], 0.05)
        self.assertTrue(summary["compare_against_mab5a"])
        self.assertTrue(summary["compare_against_mab5b"])
        self.assertEqual(summary["final_slot_counts"], [8])
        self.assertEqual(summary["mean_final_slot_count"], 8.0)
        self.assertEqual(summary["write_action_counts"], {"insert": 2})
        self.assertEqual(summary["update_reason_counts"], {"empty_bank": 1, "new_thread": 1})


if __name__ == "__main__":
    unittest.main()
