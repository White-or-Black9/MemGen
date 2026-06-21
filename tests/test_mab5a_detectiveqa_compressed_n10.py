import unittest

from scripts.eval import mab5a_detectiveqa_compressed_n10 as harness


class MAB5ADetectiveQACompressedN10Test(unittest.TestCase):
    def test_select_match_indices_prefers_deterministic_parquet_order(self):
        self.assertEqual(harness.select_match_indices(10, 10), list(range(10)))
        self.assertEqual(harness.select_match_indices(3, 10), [0, 1, 2])

    def test_build_manifest_uses_requested_experiment_contract(self):
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

    def test_compressed_query_render_keeps_only_system_and_question(self):
        messages = harness.render_compressed_query_messages(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "chunk1"}],
            [{"role": "assistant", "content": "ack"}, {"role": "user", "content": "question"}],
        )

        self.assertEqual(messages, [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
        ])

    def test_prompt_leak_detector_flags_large_overlap(self):
        prompt = "prefix " + ("A" * 160) + " suffix"
        chunk = "zzz" + ("A" * 160) + "yyy"
        self.assertTrue(harness.prompt_contains_chunk_leak(prompt, [chunk]))


if __name__ == "__main__":
    unittest.main()
