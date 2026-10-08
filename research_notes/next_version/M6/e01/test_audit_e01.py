import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('e01', Path(__file__).with_name('audit_e01.py'))
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


class AuditTests(unittest.TestCase):
    def test_perfect(self):
        c = e.correlations([1, 2, 3], [3, 4, 5])
        self.assertEqual((c['spearman'], c['kendall_tau_b']), (1, 1))

    def test_reverse(self):
        c = e.correlations([1, 2, 3], [3, 2, 1])
        self.assertEqual((c['spearman'], c['kendall_tau_b']), (-1, -1))

    def test_ties(self):
        c = e.correlations([1, 1, 2], [1, 2, 3])
        self.assertAlmostEqual(c['spearman'], math.sqrt(3)/2)
        self.assertAlmostEqual(c['kendall_tau_b'], 2/math.sqrt(6))
        self.assertEqual(c['ties_x_only'], 1)

    def test_constant(self):
        c = e.correlations([1, 1, 1], [2, 2, 2])
        self.assertIsNone(c['spearman'])
        self.assertIsNone(c['kendall_tau_b'])
        self.assertEqual(c['ties_both'], 3)

    def test_quantile(self):
        self.assertEqual(e.stats([0, 10])['quantiles']['0.25'], 2.5)

    def test_empty(self):
        self.assertIsNone(e.stats([])['mean'])

    def test_nonfinite(self):
        with self.assertRaises(ValueError):
            e.stats([float('nan')])

    def test_donors(self):
        d = e.donors('bank', list(range(100)))
        self.assertEqual(set(d), set(d.values()))
        self.assertTrue(all(k != v for k, v in d.items()))
        self.assertEqual(d, e.donors('bank', list(reversed(range(100)))))
        self.assertNotEqual(d, e.donors('other-bank', list(range(100))))

    def test_duplicate_donor_ids(self):
        with self.assertRaises(ValueError):
            e.donors('b', [1, 1])

    def test_threshold_inclusive(self):
        scores, native, full = e.reconstruct([.05, .1, .01], [0, 0, 0])
        self.assertEqual(native, [1, 0])
        self.assertEqual(full, [0, 1])

    def test_empty_retrieval(self):
        self.assertEqual(e.reconstruct([0, 0, 0], [0, 0, 0])[1:], ([], []))

    def test_decay_changes_ranking(self):
        self.assertEqual(e.reconstruct([.9, .8, .7], [100, 0, 1])[1], [1, 2])

    def test_literal_brackets_in_strings(self):
        q = 'Question clue. Below is a list of possible subsequent events: ["x]",\n "y["]\nAnswer now.'
        p = 'prefix ' + q + '<|im_end|>tail'
        spans = e.token_spans(p, q)
        candidate = next(p[a:b] for a,b,k in spans if k == 'candidate_list')
        self.assertEqual(candidate, '["x]",\n "y["]')

    def test_ambiguous_question(self):
        with self.assertRaises(ValueError):
            e.token_spans('same same', 'same')

    def test_tokenizer_offsets(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        t = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
        t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        q = 'A clue. Below is a list of possible subsequent events: ["a", "b"] Answer.'
        p = 'prefix ' + q + ' <|im_end|> tail'
        n = len(t.encode(p, add_special_tokens=False).ids)
        w = e.window(t, p, q, n, pool=64)
        self.assertEqual(sum(w['counts'].values()), n)
        self.assertGreater(w['counts']['question_clue'], 0)
        self.assertGreater(w['counts']['chat_tail'], 0)
        with self.assertRaises(ValueError):
            e.window(t, p, q, n+1)

    def test_existing_output_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                e.run(Path(d), Path(d))


if __name__ == '__main__':
    unittest.main()
