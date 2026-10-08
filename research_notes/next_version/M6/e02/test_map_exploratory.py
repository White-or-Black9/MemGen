import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('mapping', Path(__file__).with_name('map_exploratory.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MappingTests(unittest.TestCase):
    def fixture(self):
        text = 'a😀 b'
        digest = hashlib.sha256(text.encode()).hexdigest()
        q = dict(sample_id='s',context_index=0,context_sha256=digest,question_sha256='q',candidates=['x'])
        a = dict(sample_id='s',context_sha256=digest,question_sha256='q',status='supported',
                 chosen_candidate_indices=[0],source_mapping_eligible=True,
                 evidence_sets=[dict(set_id='s1',sufficiency='sufficient',
                     spans=[dict(start=1,end=2,text='😀')])])
        return text,q,a

    def test_unicode_valid(self):
        text,q,a=self.fixture()
        self.assertEqual(m.validate([a],{'s':q},{0:text}),1)

    def test_utf16_rejected(self):
        text,q,a=self.fixture();a['evidence_sets'][0]['spans'][0]['end']=3
        with self.assertRaises(ValueError):m.validate([a],{'s':q},{0:text})

    def test_source_identity(self):
        text,q,a=self.fixture()
        with self.assertRaises(ValueError):m.validate([a],{'s':q},{0:text+'!'})

    def test_question_identity(self):
        text,q,a=self.fixture();a['question_sha256']='other'
        with self.assertRaises(ValueError):m.validate([a],{'s':q},{0:text})

    def test_candidate_range(self):
        text,q,a=self.fixture();a['chosen_candidate_indices']=[1]
        with self.assertRaises(ValueError):m.validate([a],{'s':q},{0:text})

    def test_duplicates(self):
        text,q,a=self.fixture()
        with self.assertRaises(ValueError):m.validate([a,a],{'s':q},{0:text})

    def test_sentence_gap_not_covered(self):
        hit=m.span_chunks(dict(start=0,end=5),[dict(chunk_index=0,sentence_spans=[[0,1],[4,5]])],'a X b')
        self.assertFalse(hit['fully_covered'])
        self.assertEqual(hit['covered_nonwhitespace_chars'],2)

    def test_whitespace_gap_allowed(self):
        hit=m.span_chunks(dict(start=0,end=5),[dict(chunk_index=0,sentence_spans=[[0,1],[4,5]])],'a   b')
        self.assertTrue(hit['fully_covered'])

    def test_cross_chunk(self):
        hit=m.span_chunks(dict(start=0,end=3),[dict(chunk_index=0,sentence_spans=[[0,1]]),
                         dict(chunk_index=1,sentence_spans=[[2,3]])],'a b')
        self.assertEqual(hit['chunk_indices'],[0,1]);self.assertTrue(hit['fully_covered'])

    def test_indirect_not_direct(self):
        s=dict(slot_index=0,slot_version='v',direct_source_chunk=2,indirect_possible_source_chunks=[0])
        self.assertEqual(m.link({0},s)['source_link'],'indirect_possible_only')
        self.assertIn('NOT_semantic_negative',m.link({1},s)['source_link'])

    def test_multispan_union_and_two_slot_budget(self):
        text,q,a=self.fixture()
        a['evidence_sets'][0]['spans']=[dict(start=0,end=1,text='a'),dict(start=3,end=4,text='b')]
        chunks=[dict(context_index=0,chunk_index=i,sentence_spans=[[i*3,i*3+1]]) for i in range(2)]
        slots=[dict(context_index=0,seed=42,slot_index=i,slot_version=f'v{i}',
                    direct_source_chunk=i,indirect_possible_source_chunks=[],potential_source_chunks=[i]) for i in range(2)]
        result=m.map_sources([a],{'s':dict(context_index=0,query_id=1)},chunks,slots,{0:'a  b'})[0]
        self.assertEqual(result['required_source_chunks'],[0,1])
        self.assertEqual(result['two_slot_direct_all_source_pairs'],[[0,1]])
        a['source_mapping_eligible']=False
        self.assertEqual(m.map_sources([a],{'s':dict(context_index=0,query_id=1)},chunks,slots,
                                      {0:'a  b'})[0]['two_slot_direct_all_source_pairs'],[])

    def test_conflict_not_sufficient_after_adjudication(self):
        text,q,a=self.fixture();a['sample_id']='E02P-c1-01';a['status']='conflict'
        a.update(rater_id='AI',elapsed_minutes=1,completed_at='unknown')
        original=copy.deepcopy(a)
        r=m.adjudicate([a],{a['sample_id']:q},{0:text})[0]
        self.assertFalse(r['source_mapping_eligible'])
        self.assertEqual(r['evidence_sets'][0]['sufficiency'],'partial')
        self.assertEqual(a,original)
        self.assertIn('not_independent_human',r['adjudicator_type'])

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):m.run(Path(d))


if __name__=='__main__':unittest.main()
