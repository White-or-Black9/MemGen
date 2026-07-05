import unittest

from scripts.eval import eventqa_score_diagnostic as diag


class EventQAScoreDiagnosticTest(unittest.TestCase):
    def test_quantile_summary_and_missing_top2(self):
        query_records = [
            {
                "config": "A",
                "repeat": "rep1",
                "context_index": 0,
                "question_index": 0,
                "selected_top1_score": 0.031,
                "selected_top2_score": None,
                "score_margin_top1_top2": None,
                "candidate_count": 1,
                "candidate_count_passing_threshold": 1,
                "retrieved_slot_count": 1,
                "retrieved_latent_count": 8,
                "selected_slot_tuple": [0],
                "bank_on_exact_match": 0,
                "bank_on_recall": 0,
                "bank_on_format_failure": 0,
                "bank_on_chinese_output": 1,
                "helpful_memory": 0,
                "harmful_memory": 0,
                "format_harm": 0,
            },
            {
                "config": "A",
                "repeat": "rep1",
                "context_index": 0,
                "question_index": 1,
                "selected_top1_score": 0.071,
                "selected_top2_score": None,
                "score_margin_top1_top2": None,
                "candidate_count": 1,
                "candidate_count_passing_threshold": 1,
                "retrieved_slot_count": 1,
                "retrieved_latent_count": 8,
                "selected_slot_tuple": [0],
                "bank_on_exact_match": 1,
                "bank_on_recall": 1,
                "bank_on_format_failure": 0,
                "bank_on_chinese_output": 0,
                "helpful_memory": 1,
                "harmful_memory": 0,
                "format_harm": 0,
            },
        ]

        summary = diag.summarize_score_distribution(query_records)
        run_summary = summary["runs"]["A/rep1"]
        self.assertEqual(run_summary["query_count"], 2)
        self.assertEqual(run_summary["selected_top2_score"]["missing_count"], 2)
        self.assertEqual(run_summary["score_margin_top1_top2"]["missing_count"], 2)
        self.assertAlmostEqual(run_summary["selected_top1_score"]["min"], 0.031)
        self.assertAlmostEqual(run_summary["selected_top1_score"]["max"], 0.071)
        self.assertEqual(run_summary["retrieved_slot_count_distribution"]["1"], 2)

    def test_score_bucket_summary_keeps_missing_as_missing(self):
        query_records = [
            {
                "config": "B2",
                "repeat": "rep1",
                "selected_top1_score": 0.029,
                "selected_top2_score": None,
                "score_margin_top1_top2": None,
                "bank_on_exact_match": 0,
                "bank_on_recall": 0,
                "bank_on_format_failure": 1,
                "bank_on_chinese_output": 1,
                "helpful_memory": 0,
                "harmful_memory": 1,
                "format_harm": 1,
            },
            {
                "config": "B2",
                "repeat": "rep1",
                "selected_top1_score": 0.091,
                "selected_top2_score": 0.086,
                "score_margin_top1_top2": 0.005,
                "bank_on_exact_match": 1,
                "bank_on_recall": 1,
                "bank_on_format_failure": 0,
                "bank_on_chinese_output": 0,
                "helpful_memory": 1,
                "harmful_memory": 0,
                "format_harm": 0,
            },
            {
                "config": "B2",
                "repeat": "rep1",
                "selected_top1_score": None,
                "selected_top2_score": None,
                "score_margin_top1_top2": None,
                "bank_on_exact_match": 0,
                "bank_on_recall": 0,
                "bank_on_format_failure": 0,
                "bank_on_chinese_output": 0,
                "helpful_memory": 0,
                "harmful_memory": 0,
                "format_harm": 0,
            },
        ]

        summary = diag.summarize_score_buckets(query_records)
        run_summary = summary["runs"]["B2/rep1"]
        self.assertEqual(run_summary["missing_selected_top1_score_count"], 1)
        self.assertEqual(run_summary["buckets"]["<0.03"]["query_count"], 1)
        self.assertEqual(run_summary["buckets"][">=0.09"]["query_count"], 1)
        self.assertEqual(run_summary["buckets"][">=0.09"]["helpful_memory_count"], 1)
        self.assertEqual(run_summary["buckets"]["<0.03"]["harmful_memory_count"], 1)

    def test_routing_summary_and_entropy(self):
        query_records = [
            {
                "config": "D",
                "repeat": "rep2",
                "context_index": 0,
                "selected_slot_tuple": [1],
                "selected_slot_indices": [1],
                "bank_on_exact_match": 1,
                "bank_on_recall": 1,
                "bank_on_format_failure": 0,
                "bank_on_chinese_output": 0,
                "helpful_memory": 1,
                "harmful_memory": 0,
                "format_harm": 0,
            },
            {
                "config": "D",
                "repeat": "rep2",
                "context_index": 0,
                "selected_slot_tuple": [1],
                "selected_slot_indices": [1],
                "bank_on_exact_match": 0,
                "bank_on_recall": 1,
                "bank_on_format_failure": 0,
                "bank_on_chinese_output": 0,
                "helpful_memory": 0,
                "harmful_memory": 0,
                "format_harm": 0,
            },
            {
                "config": "D",
                "repeat": "rep2",
                "context_index": 1,
                "selected_slot_tuple": [0],
                "selected_slot_indices": [0],
                "bank_on_exact_match": 0,
                "bank_on_recall": 0,
                "bank_on_format_failure": 1,
                "bank_on_chinese_output": 1,
                "helpful_memory": 0,
                "harmful_memory": 1,
                "format_harm": 1,
            },
            {
                "config": "D",
                "repeat": "rep2",
                "context_index": 1,
                "selected_slot_tuple": [1],
                "selected_slot_indices": [1],
                "bank_on_exact_match": 1,
                "bank_on_recall": 1,
                "bank_on_format_failure": 0,
                "bank_on_chinese_output": 0,
                "helpful_memory": 1,
                "harmful_memory": 0,
                "format_harm": 0,
            },
        ]

        summary = diag.summarize_selected_slot_utility(query_records)
        routing = summary["routing_by_context"]["D/rep2/context_0"]
        self.assertEqual(routing["unique_selected_slot_tuple_count"], 1)
        self.assertTrue(routing["fixed_routing"])
        self.assertAlmostEqual(routing["selected_slot_entropy_bits"], 0.0)

        mixed = summary["routing_by_context"]["D/rep2/context_1"]
        self.assertEqual(mixed["unique_selected_slot_tuple_count"], 2)
        self.assertFalse(mixed["fixed_routing"])
        self.assertGreater(mixed["selected_slot_entropy_bits"], 0.0)

    def test_construction_sensitivity_tracks_borderline_and_divergence(self):
        paired_turns = {
            "A": {
                "rep1": [
                    {
                        "context_index": 0,
                        "construction_turn_index": 1,
                        "best_matched_score": 0.088,
                        "write_action": "replace_matched",
                    },
                    {
                        "context_index": 0,
                        "construction_turn_index": 2,
                        "best_matched_score": 0.101,
                        "write_action": "insert",
                    },
                ],
                "rep2": [
                    {
                        "context_index": 0,
                        "construction_turn_index": 1,
                        "best_matched_score": 0.091,
                        "write_action": "insert",
                    },
                    {
                        "context_index": 0,
                        "construction_turn_index": 2,
                        "best_matched_score": 0.100,
                        "write_action": "insert",
                    },
                ],
            }
        }

        summary = diag.summarize_construction_threshold_sensitivity(paired_turns)
        config_summary = summary["configs"]["A"]
        self.assertEqual(config_summary["total_construction_decisions"], 4)
        self.assertEqual(config_summary["borderline_counts"]["0.085-0.090"], 1)
        self.assertEqual(config_summary["borderline_counts"]["0.090-0.095"], 1)
        self.assertEqual(config_summary["paired_divergent_write_actions_in_0.080-0.100"], 1)
        self.assertEqual(len(config_summary["first_divergent_examples"]), 1)

    def test_parameter_recommendations_reflect_evidence(self):
        recommendation = diag.recommend_parameter_ranges(
            overall_summary={
                "top1_score_range": {"min": 0.031, "max": 0.089},
                "top1_score_dense_band_fraction": 0.42,
                "retrieve_threshold_filter_rate": 0.01,
                "high_score_harm_rate": 0.25,
                "top2_available_fraction": 0.6,
                "dominant_config": "B2",
                "retrieval_slot_pressure_fraction": 0.35,
            },
            correlation_summary={
                "overall": {
                    "selected_top1_score_vs_bank_on_exact_match": {"pearson": 0.08},
                    "selected_top1_score_vs_bank_on_recall": {"pearson": 0.11},
                    "selected_top1_score_vs_bank_on_format_failure": {"pearson": 0.19},
                }
            },
            construction_summary={
                "configs": {
                    "B2": {
                        "borderline_fraction_0.080-0.100": 0.38,
                    }
                }
            },
        )

        self.assertEqual(recommendation["A"]["immediate_next_experiment"]["priority"], "high")
        self.assertIn(0.02, recommendation["B"]["secondary_parameter_sweep"]["retrieve_threshold"]["worth_testing"])
        self.assertIn(0.09, recommendation["C"]["parameters_to_avoid_for_now"]["update_threshold"]["avoid"])
        self.assertIn("deterministic EventQA mode", recommendation["D"]["mechanism_changes_to_consider"]["options"])


if __name__ == "__main__":
    unittest.main()
