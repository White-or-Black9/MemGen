import unittest

from scripts.eval import mab2_bank_off as harness


class MAB2BankOffTest(unittest.TestCase):
    def test_episode_env_keeps_acknowledgements_separate_from_final_answer(self):
        env = harness.MABEpisodeEnv(
            subsequent_prompts=["chunk-two", "query-one"],
            expected_turns=3,
        )

        self.assertEqual(env.step("ack-one"), ("chunk-two", 0.0, False))
        self.assertEqual(env.step("ack-two"), ("query-one", 0.0, False))
        self.assertEqual(env.step("final-answer"), ("", 0.0, True))

        self.assertEqual(env.acknowledgements, ["ack-one", "ack-two"])
        self.assertEqual(env.final_answer, "final-answer")

    def test_episode_env_rejects_extra_turn(self):
        env = harness.MABEpisodeEnv(subsequent_prompts=[], expected_turns=1)
        env.step("answer")

        with self.assertRaisesRegex(RuntimeError, "after episode completion"):
            env.step("extra")

    def test_history_audit_requires_every_prior_chunk_on_query_turn(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "chunk-one"},
            {"role": "assistant", "content": "ack-one"},
            {"role": "user", "content": "chunk-two"},
            {"role": "assistant", "content": "ack-two"},
            {"role": "user", "content": "query-one"},
        ]

        harness.assert_full_history(messages, ["chunk-one", "chunk-two"], "query-one")

        with self.assertRaisesRegex(RuntimeError, "missing prior chunk 2"):
            without_chunk_two = [message for message in messages if message["content"] != "chunk-two"]
            harness.assert_full_history(without_chunk_two, ["chunk-one", "chunk-two"], "query-one")

    def test_bank_off_invariants_reject_any_created_bank_or_activity(self):
        harness.assert_bank_off_invariants(
            bank_enabled=False,
            bank_created=False,
            bank_write_count=0,
            bank_retrieval_count=0,
            bank_slot_count=0,
        )

        with self.assertRaisesRegex(RuntimeError, "Bank-off invariant"):
            harness.assert_bank_off_invariants(
                bank_enabled=False,
                bank_created=True,
                bank_write_count=0,
                bank_retrieval_count=0,
                bank_slot_count=0,
            )

    def test_manifest_uses_unambiguous_baseline_and_cache_policy(self):
        manifest = harness.build_manifest_skeleton(
            run_id="test-run",
            dataset_path="/data/mab",
            model_checkpoint="Kana-s/MemGen/test",
            memgen_branch="rlm-memory-bank",
            memgen_git_status="## rlm-memory-bank",
            started_at="2026-06-20T00:00:00+00:00",
        )

        self.assertEqual(
            manifest["baseline_name"],
            "Original MemGen Full-history Rebuild Bank-off",
        )
        self.assertEqual(manifest["history_policy"], "full_rebuild")
        self.assertFalse(manifest["cross_turn_kv_reuse"])
        self.assertFalse(manifest["bank_enabled"])
        self.assertFalse(manifest["bank_created"])


if __name__ == "__main__":
    unittest.main()
