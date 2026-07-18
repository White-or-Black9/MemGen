import unittest

from scripts.eval import longbench_v2_scorer as scorer


def item():
    return {
        "item_id": "item-1",
        "domain": "Multi-Document QA",
        "sub_domain": "Academic",
        "difficulty": "easy",
        "length": "short",
        "capacity_class": "window_fit",
        "gold_choice": "B",
        "choices": {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
    }


class LongBenchV2ScorerTest(unittest.TestCase):
    def test_strict_contract_accepts_only_official_response_format(self):
        self.assertEqual(scorer.extract_strict_choice("The correct answer is (B)"), "B")
        self.assertEqual(scorer.extract_strict_choice("The correct answer is (b)."), "B")
        self.assertIsNone(scorer.extract_strict_choice("B"))
        self.assertIsNone(scorer.extract_strict_choice("The correct answer is (B) because..."))

    def test_relaxed_choice_is_diagnostic_and_rejects_ambiguity(self):
        choices = item()["choices"]
        self.assertEqual(scorer.extract_relaxed_choice("B", choices), "B")
        self.assertEqual(scorer.extract_relaxed_choice("Beta", choices), "B")
        self.assertIsNone(scorer.extract_relaxed_choice("A or B", choices))

    def test_score_keeps_strict_primary_separate_from_relaxed_diagnostic(self):
        scored = scorer.score_prediction(item(), "B")
        self.assertEqual(scored["strict_correct"], 0)
        self.assertEqual(scored["relaxed_correct"], 1)
        self.assertEqual(scored["invalid_output"], 1)

    def test_aggregate_reports_required_slices(self):
        rows = [
            scorer.score_prediction(item(), "The correct answer is (B)"),
            scorer.score_prediction(item() | {"item_id": "item-2", "gold_choice": "A"}, "invalid"),
        ]
        aggregate = scorer.aggregate_scores(rows, method="p7")
        self.assertEqual(aggregate["method"], "p7")
        self.assertEqual(aggregate["overall"]["count"], 2)
        self.assertEqual(aggregate["overall"]["invalid_output_count"], 1)
        self.assertIn("by_domain", aggregate)
        self.assertIn("by_capacity_class", aggregate)


if __name__ == "__main__":
    unittest.main()
