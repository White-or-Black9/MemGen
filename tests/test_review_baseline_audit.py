"""Guard the offline audit against sample and lifecycle corruption."""
import copy
import unittest

from scripts.eval.review_baseline_audit import metric_rows, stats


def rows(latent=False):
    prefix = "bank_on_" if latent else ""
    return [{"context_index": c, "query_id" if latent else "query_index": q,
             prefix + "substring_exact_match": int(q == 0),
             prefix + "eventqa_recall": int(q < 2),
             prefix + "format_flags": {"multiline_output": q == 3},
             "query_write_count": 0, "bank_snapshot_changed_after_query": False}
            for c in range(5) for q in range(100)]


class AuditTests(unittest.TestCase):
    def test_recompute_control_and_latent(self):
        for latent in (False, True):
            self.assertEqual(metric_rows(rows(latent), latent), [.01, .02, 5])

    def test_duplicate_and_missing_questions_rejected(self):
        sample = rows()
        sample[-1] = copy.deepcopy(sample[0])
        with self.assertRaises(ValueError):
            metric_rows(sample)
        with self.assertRaises(ValueError):
            metric_rows(rows()[:-1])

    def test_frozen_mutation_rejected(self):
        for key, value in [("query_write_count", 1), ("bank_snapshot_changed_after_query", True), ("cross_context_leakage_detected", True)]:
            sample = rows(True)
            sample[0][key] = value
            with self.assertRaises(ValueError):
                metric_rows(sample, True)

    def test_population_std_and_invalid_metric(self):
        self.assertEqual(stats([0., 2.])["population_std"], 1.)
        with self.assertRaises(ValueError):
            stats([float("nan")])
        sample = rows()
        sample[0]["eventqa_recall"] = .5
        with self.assertRaises(ValueError):
            metric_rows(sample)


if __name__ == "__main__":
    unittest.main()
