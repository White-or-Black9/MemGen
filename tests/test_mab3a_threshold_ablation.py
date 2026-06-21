import unittest
from unittest import mock

from scripts.eval import mab3a_threshold_ablation as harness


class MAB3AThresholdAblationTest(unittest.TestCase):
    def test_threshold_cases_match_preregistered_grid(self):
        cases = harness.build_threshold_cases()

        self.assertEqual(cases[0]["label"], "top_k_only")
        self.assertTrue(cases[0]["top_k_only"])
        self.assertEqual(
            [case["threshold"] for case in cases[1:]],
            [0.00, 0.01, 0.02, 0.03, 0.035, 0.04, 0.045, 0.05, 0.07, 0.10, 0.70],
        )

    def test_top_k_only_case_uses_topk_policy(self):
        case = harness.build_threshold_cases()[0]

        config = harness.bank_config_for_case(case)

        self.assertEqual(config["top_k"], 1)
        self.assertEqual(config["retrieve_policy"], "topk")
        self.assertEqual(config["threshold"], 0.7)

    def test_threshold_case_uses_threshold_topk_policy(self):
        case = harness.build_threshold_cases()[4]

        config = harness.bank_config_for_case(case)

        self.assertEqual(config["retrieve_policy"], "threshold_topk")
        self.assertEqual(config["threshold"], 0.03)
        self.assertFalse(case["top_k_only"])

    def test_candidate_score_pairs_preserve_slot_order(self):
        self.assertEqual(
            harness.candidate_score_pairs([0.2, -0.1, 0.0]),
            [
                {"slot_index": 0, "score": 0.2},
                {"slot_index": 1, "score": -0.1},
                {"slot_index": 2, "score": 0.0},
            ],
        )

    def test_summary_records_threshold_filtered_and_raw_scores(self):
        case = {"label": "0.04", "threshold": 0.04, "top_k_only": False}
        diagnostics = [
            {
                "turn_index": 0,
                "turn_type": "memorize_chunk",
                "candidate_raw_scores": [],
                "max_score": None,
                "threshold_passed": False,
                "retrieved_indices": [],
                "retrieved_scores": [],
                "retrieved_latent_count": 0,
            },
            {
                "turn_index": 1,
                "turn_type": "memorize_chunk",
                "candidate_raw_scores": [0.0492],
                "max_score": 0.0492,
                "threshold_passed": True,
                "retrieved_indices": [0],
                "retrieved_scores": [0.0492],
                "retrieved_latent_count": 8,
            },
            {
                "turn_index": 2,
                "turn_type": "query",
                "candidate_raw_scores": [0.0366, 0.0340],
                "max_score": 0.0366,
                "threshold_passed": False,
                "retrieved_indices": [],
                "retrieved_scores": [],
                "retrieved_latent_count": 0,
            },
        ]

        summary = harness.summarize_threshold_result(
            case=case,
            diagnostics=diagnostics,
            prediction="pred",
            gold_answers=["gold"],
            score_value=1,
        )

        self.assertEqual(summary["threshold"], 0.04)
        self.assertEqual(summary["max_score_turn2"], 0.0492)
        self.assertEqual(summary["max_score_turn3"], 0.0366)
        self.assertEqual(summary["slots_passing_threshold_turn2"], 1)
        self.assertEqual(summary["slots_passing_threshold_turn3"], 0)
        self.assertEqual(summary["retrieved_latent_count_total"], 8)
        self.assertEqual(summary["retrieved_latent_count_by_turn"], [0, 8, 0])
        self.assertEqual(summary["retrieved_indices_by_turn"], [[], [0], []])
        self.assertEqual(summary["retrieved_scores_by_turn"], [[], [0.0492], []])
        self.assertEqual(summary["substring_exact_match"], 1)

    def test_extract_substring_exact_match_prefers_metrics_payload(self):
        score_payload = {
            "additional": {"parsed_output": "x"},
            "metrics": {"substring_exact_match": False},
        }

        self.assertEqual(
            harness.extract_substring_exact_match(score_payload),
            0,
        )

    def test_release_cuda_cache_collects_and_empties_when_cuda_available(self):
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = True
        fake_gc = mock.Mock()

        harness.release_cuda_cache(fake_torch, fake_gc)

        fake_gc.collect.assert_called_once_with()
        fake_torch.cuda.empty_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
