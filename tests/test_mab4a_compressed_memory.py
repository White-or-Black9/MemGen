import unittest

from scripts.eval import mab4a_compressed_memory as harness


class MAB4ACompressedMemoryTest(unittest.TestCase):
    def test_threshold_cases_match_focused_grid(self):
        cases = harness.build_threshold_cases()

        self.assertEqual(
            [(case["label"], case["threshold"], case["top_k_only"]) for case in cases],
            [
                ("top_k_only", None, True),
                ("0.00", 0.0, False),
                ("0.03", 0.03, False),
                ("0.035", 0.035, False),
                ("0.70", 0.7, False),
            ],
        )

    def test_compressed_query_messages_keep_system_and_current_query_only(self):
        init_prompt = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "chunk1 prompt"},
        ]
        inter_history = [
            {"role": "assistant", "content": "ack1"},
            {"role": "user", "content": "chunk2 prompt"},
            {"role": "assistant", "content": "ack2"},
            {"role": "user", "content": "query prompt"},
        ]

        messages = harness.build_compressed_query_messages(init_prompt, inter_history)

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "query prompt"},
            ],
        )

    def test_chunk_leak_detector_flags_large_substring(self):
        prompt = "prefix " + ("A" * 160) + " suffix"
        chunk = "zzz" + ("A" * 160) + "yyy"

        self.assertTrue(harness.prompt_contains_chunk_leak(prompt, [chunk]))

    def test_chunk_leak_detector_ignores_short_overlap(self):
        prompt = "question about rugby"
        chunk = "rugby"

        self.assertFalse(harness.prompt_contains_chunk_leak(prompt, [chunk]))

    def test_summary_captures_query_prompt_compression(self):
        case = {"label": "0.03", "threshold": 0.03, "top_k_only": False}
        diagnostics = [
            {
                "turn_type": "memorize_chunk",
                "retrieved_latent_count": 0,
                "retrieved_indices": [],
                "retrieved_scores": [],
                "max_score": None,
                "query_prompt_contains_chunk_text": None,
                "prompt_history_token_len": 4972,
                "full_history_included": True,
            },
            {
                "turn_type": "memorize_chunk",
                "retrieved_latent_count": 8,
                "retrieved_indices": [0],
                "retrieved_scores": [0.0492],
                "max_score": 0.0492,
                "query_prompt_contains_chunk_text": None,
                "prompt_history_token_len": 7497,
                "full_history_included": True,
            },
            {
                "turn_type": "query",
                "retrieved_latent_count": 8,
                "retrieved_indices": [0],
                "retrieved_scores": [0.0366],
                "max_score": 0.0366,
                "query_prompt_contains_chunk_text": False,
                "prompt_history_token_len": 123,
                "full_history_included": False,
            },
        ]

        summary = harness.summarize_threshold_result(
            case=case,
            diagnostics=diagnostics,
            prediction="pred",
            gold_answers=["gold"],
            score_value=0,
        )

        self.assertEqual(summary["query_prompt_token_len"], 123)
        self.assertFalse(summary["query_prompt_contains_chunk_text"])
        self.assertEqual(summary["retrieved_latent_count_total"], 16)

    def test_build_manifest_marks_compressed_history_policy(self):
        class Args:
            dataset_root = "/data"
            model_checkpoint_id = "ckpt"
            paired_mab2_artifact = "m2"
            paired_mab3_artifact = "m3"
            paired_mab3a_artifact = "m3a"

        manifest = harness._build_manifest("run", Args(), "now")

        self.assertEqual(manifest["history_policy"], "compressed")
        self.assertTrue(manifest["compressed_memory"])


if __name__ == "__main__":
    unittest.main()
