import unittest

from scripts.eval import mab3_bank_on_full_history as harness
from scripts.eval import mab2_mab_bridge


class MAB3BankOnFullHistoryTest(unittest.TestCase):
    def test_bridge_accepts_pinned_timestamp_for_prompt_parity(self):
        self.assertEqual(
            mab2_mab_bridge.resolve_timestamp("2026-06-20 11:40:37"),
            "2026-06-20 11:40:37",
        )

    def test_version_a_bank_config_is_enabled_and_batch_one(self):
        config = harness.version_a_bank_config()

        self.assertTrue(config["enabled"])
        self.assertEqual(config["batch_size"], 1)
        self.assertEqual(config["update_policy"], "thread_update")
        self.assertEqual(config["retrieve_policy"], "threshold_topk")

    def test_bank_lifecycle_requires_one_shared_empty_then_reset_bank(self):
        harness.assert_bank_lifecycle(
            create_count=1,
            initial_slot_count=0,
            generation_bank_ids=[7, 7, 7],
            created_bank_id=7,
            post_reset_slot_count=0,
        )

        with self.assertRaisesRegex(RuntimeError, "same bank"):
            harness.assert_bank_lifecycle(
                create_count=1,
                initial_slot_count=0,
                generation_bank_ids=[7, 8, 7],
                created_bank_id=7,
                post_reset_slot_count=0,
            )

    def test_memory_boundary_requires_no_retrieved_latents_in_weaver(self):
        harness.assert_memory_boundary(
            retrieved_latent_count=8,
            retrieved_latents_enter_reasoner=True,
            retrieved_latents_enter_weaver=False,
            stored_latent_reasoner_space=True,
            stored_latent_detached_cloned=True,
        )

        with self.assertRaisesRegex(RuntimeError, "entered Weaver"):
            harness.assert_memory_boundary(
                retrieved_latent_count=8,
                retrieved_latents_enter_reasoner=True,
                retrieved_latents_enter_weaver=True,
                stored_latent_reasoner_space=True,
                stored_latent_detached_cloned=True,
            )

    def test_prompt_parity_requires_first_visible_prompt_hash(self):
        parity = harness.prompt_parity_summary(
            bank_on_hashes=["same", "bank-on-two", "bank-on-three"],
            bank_off_hashes=["same", "bank-off-two", "bank-off-three"],
        )

        self.assertTrue(parity["initial_prompt_exact_match"])
        self.assertFalse(parity["all_turns_exact_match"])
        self.assertEqual(parity["later_difference_reason"], "generated_acknowledgements_may_differ")

        with self.assertRaisesRegex(RuntimeError, "initial visible prompt"):
            harness.prompt_parity_summary(
                bank_on_hashes=["different"],
                bank_off_hashes=["same"],
            )

    def test_top_retrieval_scores_include_raw_scores_below_threshold(self):
        self.assertEqual(
            harness.top_retrieval_scores([0.02, 0.05, -0.1], limit=2),
            [0.05, 0.02],
        )

    def test_manifest_names_the_paired_intervention(self):
        manifest = harness.build_manifest_skeleton(
            run_id="run",
            dataset_path="/data",
            model_checkpoint="checkpoint",
            memgen_branch="rlm-memory-bank",
            memgen_git_status="dirty",
            started_at="now",
        )

        self.assertEqual(
            manifest["baseline_name"],
            "MemGen + LatentBank V-A Full-history Rebuild Bank-on",
        )
        self.assertTrue(manifest["bank_enabled"])
        self.assertEqual(manifest["history_policy"], "full_rebuild")
        self.assertFalse(manifest["cross_turn_kv_reuse"])


if __name__ == "__main__":
    unittest.main()
