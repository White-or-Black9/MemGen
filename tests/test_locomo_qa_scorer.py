import unittest

from scripts.eval import locomo_qa_scorer as scorer


class LoCoMoQAScorerTest(unittest.TestCase):
    def test_exact_match_and_token_f1_for_exact_correct_prediction(self):
        scored = scorer.score_row("7 May 2023", "7 May 2023", "ok")
        self.assertEqual(scored["exact_match"], 1)
        self.assertEqual(scored["token_f1"], 1.0)
        self.assertEqual(scored["invalid_output"], 0)

    def test_partial_overlap_yields_zero_em_and_positive_f1(self):
        scored = scorer.score_row("May 2023", "7 May 2023", "ok")
        self.assertEqual(scored["exact_match"], 0)
        self.assertGreater(scored["token_f1"], 0.0)
        self.assertLess(scored["token_f1"], 1.0)

    def test_empty_or_invalid_prediction_marks_invalid_output(self):
        scored = scorer.score_row("   ", "7 May 2023", "ok")
        self.assertEqual(scored["invalid_output"], 1)
        self.assertEqual(scored["token_f1"], 0.0)
        self.assertEqual(scored["exact_match"], 0)

    def test_case_and_punctuation_normalization_supports_exact_match(self):
        scored = scorer.score_row("7 may 2023.", "7 May 2023", "ok")
        self.assertEqual(scored["exact_match"], 1)

    def test_invalid_status_marks_invalid_output(self):
        scored = scorer.score_row("7 May 2023", "7 May 2023", "invalid")
        self.assertEqual(scored["invalid_output"], 1)
        self.assertEqual(scored["exact_match"], 0)

    def test_aggregate_metrics_contains_overall_category_and_conversation_fields(self):
        summary = scorer.aggregate_scores([
            {"conversation_id": "conv-26", "category_name": "temporal", "exact_match": 1, "token_f1": 1.0, "invalid_output": 0},
            {"conversation_id": "conv-26", "category_name": "single_hop", "exact_match": 0, "token_f1": 0.5, "invalid_output": 0},
            {"conversation_id": "conv-27", "category_name": "temporal", "exact_match": 1, "token_f1": 0.75, "invalid_output": 1},
        ], method="disabled")
        self.assertEqual(summary["method"], "disabled")
        self.assertIn("overall_micro", summary)
        self.assertIn("overall_macro_by_conversation", summary)
        self.assertIn("by_category", summary)
        self.assertIn("by_conversation", summary)
        self.assertEqual(summary["invalid_output_count"], 1)
        self.assertEqual(summary["record_count"], 3)


if __name__ == "__main__":
    unittest.main()
