import unittest

from scripts.eval import longbench_v2_merge as merge


class LongBenchV2MergeTest(unittest.TestCase):
    def test_paired_summary_reports_wins_losses_and_exact_test(self):
        rows = [
            {"item_id": "a", "method": "p7", "strict_correct": 1},
            {"item_id": "a", "method": "p7_no_query_retrieval", "strict_correct": 0},
            {"item_id": "b", "method": "p7", "strict_correct": 0},
            {"item_id": "b", "method": "p7_no_query_retrieval", "strict_correct": 1},
            {"item_id": "c", "method": "p7", "strict_correct": 1},
            {"item_id": "c", "method": "p7_no_query_retrieval", "strict_correct": 1},
        ]
        result = merge.paired_summary(rows, "p7", "p7_no_query_retrieval")
        self.assertEqual(result["pair_count"], 3)
        self.assertEqual((result["left_wins"], result["left_losses"], result["ties"]), (1, 1, 1))
        self.assertEqual(result["exact_two_sided_sign_test_p_value"], 1.0)

    def test_exact_sign_test_returns_none_without_discordant_pairs(self):
        self.assertIsNone(merge.exact_sign_test_p_value(0, 0))


if __name__ == "__main__":
    unittest.main()
