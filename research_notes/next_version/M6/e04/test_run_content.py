import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('content',Path(__file__).with_name('run_content.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class ContentTests(unittest.TestCase):
    def mapping(self):
        return dict(eligible=True,required_source_chunks=[4],span_mapping=[dict(fully_covered=True)],
            slots=[dict(slot_index=i,direct_intersection=[4] if i==2 else [],
                        indirect_possible_intersection=[]) for i in range(4)])

    def test_fixed_filler_position(self):
        self.assertEqual(p.choose_pair(self.mapping(),[0,3]),[0,2])

    def test_ineligible(self):
        m=self.mapping();m['eligible']=False
        with self.assertRaises(ValueError):p.choose_pair(m,[0,3])

    def test_filler_direct_link(self):
        m=self.mapping();m['slots'][0]['direct_intersection']=[4]
        with self.assertRaises(ValueError):p.choose_pair(m,[0,3])

    def test_filler_indirect_link(self):
        m=self.mapping();m['slots'][0]['indirect_possible_intersection']=[4]
        with self.assertRaises(ValueError):p.choose_pair(m,[0,3])

    def test_no_unique_sufficient_slot(self):
        m=self.mapping();m['required_source_chunks']=[4,5]
        with self.assertRaises(ValueError):p.choose_pair(m,[0,3])

    def test_uncovered_span(self):
        m=self.mapping();m['span_mapping'][0]['fully_covered']=False
        with self.assertRaises(ValueError):p.choose_pair(m,[0,3])

    def test_wrong_original_filler(self):
        with self.assertRaises(ValueError):p.choose_pair(self.mapping(),[1,3])
        with self.assertRaises(ValueError):p.choose_pair(self.mapping(),[3,0])

    def test_tie_by_index_not_performance(self):
        m=self.mapping();m['slots'][3]['direct_intersection']=[4]
        self.assertEqual(p.choose_pair(m,[0,3]),[0,2])

    def test_existing_directory(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):p.prepare(Path(d))

    def test_paired_full_denominator(self):
        source={};other={}
        for ctx in range(5):
            for q in range(2 if ctx==0 else 1):
                r=dict(context_index=ctx,strict_candidate_correct=False,candidate_valid=False,
                    parsed_nonempty=False,raw_gold_any=False,official_raw_recall=0,prediction='bad')
                source[(ctx,q)]=dict(r);other[(ctx,q)]=dict(r)
        source[(0,0)].update(strict_candidate_correct=True,candidate_valid=True,prediction='good')
        other[(1,0)].update(strict_candidate_correct=True,candidate_valid=True,prediction='good')
        pair=p.paired(source,other)
        self.assertEqual(pair['denominator'],6)
        self.assertEqual(pair['strict']['source_1_other_0'],1)
        self.assertEqual(pair['strict']['source_0_other_1'],1)
        self.assertEqual(pair['strict']['source_0_other_0'],4)
        self.assertEqual(sum(pair['other_to_source_CWI'].values()),6)
        self.assertEqual(pair['text_changed'],2)
        self.assertAlmostEqual(pair['context_equal_mean'],-.1)

    def test_mismatched_pairs(self):
        with self.assertRaises(ValueError):p.paired({'a':{}},{})


if __name__=='__main__':unittest.main()
