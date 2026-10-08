import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('e02',Path(__file__).with_name('prepare_e02.py'))
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)


class PreparationTests(unittest.TestCase):
    def rows(self):
        return [dict(context_index=c,query_id=q) for c in range(5) for q in range(10)]

    def question(self):
        return 'A clue. Below is a list of possible subsequent events: ["x]", "y"] Answer.'

    def test_sample_balance(self):
        selected=e.select_pilot(self.rows())
        self.assertEqual(len(selected),10)
        self.assertEqual([sum(r['context_index']==c for r in selected) for c in range(5)],[2]*5)

    def test_sample_order_independent(self):
        self.assertEqual(e.select_pilot(self.rows()),e.select_pilot(list(reversed(self.rows()))))

    def test_sample_duplicate_fails(self):
        rows=self.rows();rows[-1]=rows[-2]
        with self.assertRaises(ValueError):e.select_pilot(rows)

    def test_sample_balance_fails(self):
        rows=self.rows();rows[-1]['context_index']=0
        with self.assertRaises(ValueError):e.select_pilot(rows)

    def test_candidates_literals(self):
        self.assertEqual(e.candidates(self.question()),['x]','y'])

    def test_public_whitelist(self):
        q=self.question();row=dict(context_index=0,context_id='c',question=q,question_text_sha256=e.text_sha(q),
                                  gold_answers=['y'],query_id=1,selected_indices=[0,1])
        public=e.public_question(row,'sample','sha')
        e.check_blind(public)
        self.assertNotIn('gold_answers',public)
        self.assertNotIn('query_id',public)

    def test_nested_leak_fails(self):
        with self.assertRaises(ValueError):e.check_blind({'nested':[{'scores':[1]}]})

    def test_blank_labels(self):
        q=dict(sample_id='s',context_sha256='a',question_sha256='b')
        form=e.blank_form(q);e.check_blank(form)
        self.assertIsNone(json.loads(json.dumps(form))['status'])

    def test_prefilled_label_rejected(self):
        f=e.blank_form(dict(sample_id='s',context_sha256='a',question_sha256='b'));f['status']='supported'
        with self.assertRaises(ValueError):e.check_blank(f)

    def test_unicode_codepoints(self):
        self.assertTrue(e.check_span('a😀çz',dict(start=1,end=2,text='😀')))

    def test_utf16_offsets_rejected(self):
        with self.assertRaises(ValueError):e.check_span('a😀çz',dict(start=1,end=3,text='😀'))

    def test_out_of_range_span_rejected(self):
        with self.assertRaises(ValueError):e.check_span('abc',dict(start=-1,end=2,text='ab'))

    def test_existing_output_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):e.run(Path(d))


if __name__=='__main__':unittest.main()
