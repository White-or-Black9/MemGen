import unittest

from scripts.eval import mab_paired_bank_off_vs_low_threshold_bank_on as harness


class MABPairedBankOffVsLowThresholdBankOnTest(unittest.TestCase):
    def test_requested_ten_contexts_reduce_to_available_matches(self):
        selection = harness.select_match_indices(total_matches=1, requested=10)

        self.assertEqual(selection, [0])

    def test_summary_counts_output_change_improvement_and_regression(self):
        rows = [
            {
                "bank_off_substring_exact_match": 0,
                "bank_on_substring_exact_match": 1,
                "output_changed": True,
                "bank_on_retrieved_latent_count": 8,
                "full_history_query_tokens": 100,
                "chunk_count": 2,
                "latency_total": 1.0,
                "peak_cuda_memory": 10,
                "error_or_stop_reason": None,
            },
            {
                "bank_off_substring_exact_match": 1,
                "bank_on_substring_exact_match": 0,
                "output_changed": True,
                "bank_on_retrieved_latent_count": 8,
                "full_history_query_tokens": 200,
                "chunk_count": 3,
                "latency_total": 2.0,
                "peak_cuda_memory": 20,
                "error_or_stop_reason": None,
            },
            {
                "bank_off_substring_exact_match": 0,
                "bank_on_substring_exact_match": 0,
                "output_changed": False,
                "bank_on_retrieved_latent_count": 0,
                "full_history_query_tokens": 300,
                "chunk_count": 1,
                "latency_total": 3.0,
                "peak_cuda_memory": 30,
                "error_or_stop_reason": "invalid",
            },
        ]

        summary = harness.aggregate_results(rows, requested=10, attempted=3)

        self.assertEqual(summary["num_contexts_requested"], 10)
        self.assertEqual(summary["num_contexts_attempted"], 3)
        self.assertEqual(summary["num_contexts_valid"], 2)
        self.assertEqual(summary["num_contexts_invalid"], 1)
        self.assertEqual(summary["bank_off_correct"], 1)
        self.assertEqual(summary["bank_on_correct"], 1)
        self.assertEqual(summary["num_bank_on_retrieval_active"], 2)
        self.assertEqual(summary["num_bank_on_output_changed_vs_bank_off"], 2)
        self.assertEqual(summary["num_bank_on_improved"], 1)
        self.assertEqual(summary["num_bank_on_regressed"], 1)
        self.assertEqual(summary["num_bank_on_same_score"], 0)
        self.assertAlmostEqual(summary["average_full_history_query_tokens"], 150.0)
        self.assertAlmostEqual(summary["average_chunk_count"], 2.5)
        self.assertAlmostEqual(summary["average_retrieved_latents"], 8.0)
        self.assertEqual(summary["peak_cuda_memory"], 20)


if __name__ == "__main__":
    unittest.main()
