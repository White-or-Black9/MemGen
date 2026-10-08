"""Pure stdlib numerical and pairing gates for M4-B1."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('m4_temporal', Path(__file__).with_name('audit_temporal.py'))
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


class TemporalTests(unittest.TestCase):
    def test_threshold_inclusive_and_no_fallback(self):
        self.assertEqual(t.reconstruct([.05, .049], [0, 0], .05, .05, 2)['full'], [0])
        result = t.reconstruct([.01, .02], [0, 0], .05, .05, 2)
        self.assertEqual(result['full'], [])
        self.assertEqual(result['recency'], [])

    def test_ties_use_index_and_canonical_is_not_score_order(self):
        result = t.reconstruct([.2, .3, .3], [0, 0, 0], .05, .05, 2)
        self.assertEqual(result['native'], [1, 2])
        result = t.reconstruct([.2, .1, .3], [0, 0, 0], .05, .05, 2)
        self.assertEqual(result['native'], [2, 0])
        self.assertEqual(result['full'], [0, 2])

    def test_negative_cosine_decay_does_not_imply_older_always_lower(self):
        result = t.reconstruct([-.1, -.1], [0, 10], .05, -.2, 1)
        self.assertGreater(result['scores'][1], result['scores'][0])
        self.assertEqual(result['native'], [1])

    def test_access_age_and_write_age_are_distinct(self):
        result = t.reconstruct([.1, .1, .1], [1, 5, 1], .05, .05, 2)
        last_write = [2, 16, 17]
        self.assertEqual(result['recency'], [0, 2])
        self.assertEqual(sorted(t.rank(last_write)[:2]), [1, 2])

    def test_recency_tie_can_disagree_with_cosine(self):
        result = t.reconstruct([.1, .1, .2], [1, 1, 1], .05, .05, 2)
        self.assertEqual(result['recency'], [0, 1])
        self.assertEqual(result['full'], [0, 2])

    def test_finite_gate(self):
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            t.rank([float('nan')])

    def test_saved_arithmetic_tolerance(self):
        t.close_list([.1], [.1 + 1e-14], 'score')
        with self.assertRaisesRegex(ValueError, 'arithmetic mismatch'):
            t.close_list([.1], [.101], 'score')

    def test_no_threshold_tolerance_that_changes_selection(self):
        result = t.reconstruct([.05 - 1e-14], [0], .05, .05, 2)
        self.assertEqual(result['native'], [])

    def test_missing_duplicate_and_drifted_policy_gates(self):
        rows = [dict(policy=p, cosine=[.1], ages=[1], last_write=[1], decay=[.95],
                     final_scores=[.095], native_indices=[0], n=1) for p in t.c.POLICIES]
        t.paired(rows)
        with self.assertRaisesRegex(ValueError, 'missing or duplicate'):
            t.paired(rows[:-1])
        duplicate = copy.deepcopy(rows)
        duplicate[-1]['policy'] = duplicate[0]['policy']
        with self.assertRaisesRegex(ValueError, 'missing or duplicate'):
            t.paired(duplicate)
        drift = copy.deepcopy(rows)
        drift[-1]['ages'] = [2]
        with self.assertRaisesRegex(ValueError, 'base drift'):
            t.paired(drift)

    def test_metadata_must_not_change_across_repeated_calls(self):
        cosine, ages = [.1, .08], [1, 10]
        first = t.reconstruct(cosine, ages, .05, .05, 2)
        self.assertEqual(first, t.reconstruct(cosine, ages, .05, .05, 2))
        self.assertEqual(ages, [1, 10])


if __name__ == '__main__':
    unittest.main()
