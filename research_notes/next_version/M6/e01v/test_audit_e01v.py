import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('e01v',Path(__file__).with_name('audit_e01v.py'))
e=importlib.util.module_from_spec(spec);sys.modules[spec.name]=e;spec.loader.exec_module(e)
torch=e.torch


class FakeEmbedding:
    def get(self,ids):
        return torch.tensor([[[float(i),1.] for i in ids]],dtype=torch.bfloat16)


class AuditTests(unittest.TestCase):
    def test_linear_then_mean(self):
        w=torch.eye(2,dtype=torch.bfloat16);b=torch.ones(2,dtype=torch.bfloat16)
        q=e.represent(list(range(65)),FakeEmbedding(),w,b)
        self.assertTrue(torch.equal(q,torch.tensor([33.5,2.],dtype=torch.bfloat16)))

    def test_variants(self):
        w=torch.eye(2,dtype=torch.bfloat16);b=torch.ones(2,dtype=torch.bfloat16)
        for variant in ('bf16','fp32','pool_first'):
            q=e.represent([1,2],FakeEmbedding(),w,b,variant)
            self.assertTrue(torch.equal(q.float(),torch.tensor([2.5,2.])))

    def test_unknown_variant(self):
        with self.assertRaises(ValueError):
            e.represent([1],FakeEmbedding(),torch.eye(2,dtype=torch.bfloat16),torch.zeros(2,dtype=torch.bfloat16),'bad')

    def test_hash_dtype_matters(self):
        q=torch.ones(2,dtype=torch.bfloat16)
        self.assertNotEqual(e.tensor_hash(q),e.tensor_hash(q.float()))
        self.assertEqual(e.tensor_hash(q),e.tensor_hash(q.clone()))

    def test_q_difference(self):
        q=torch.tensor([1.,2.])
        self.assertTrue(e.q_difference(q,q)['exact_equal'])
        self.assertAlmostEqual(e.q_difference(q,q)['cosine'],1.)

    def test_candidate_only(self):
        ids=list(range(80));other=list(range(100,180))
        categories=['candidate_list']*17+['answer_instruction']*47
        changed=e.candidate_variant(ids,other,categories,categories)
        self.assertEqual(changed[:16],ids[:16])
        self.assertEqual(changed[16:33],other[16:33])
        self.assertEqual(changed[33:],ids[33:])

    def test_candidate_budget(self):
        with self.assertRaises(ValueError):
            e.candidate_variant([0]*64,[1]*64,['candidate_list']*64,['candidate_list']*64)

    def test_clue_only(self):
        rest='Below is a list of possible subsequent events: ["a"] Answer.'
        q='Old clue. '+rest; donor='Other clue. '+rest
        changed=e.clue_variant('prefix '+q,q,'prefix '+donor,donor)
        self.assertEqual(changed,'prefix '+donor)

    def test_geometry_identical(self):
        g,m=e.geometry(torch.ones(3,2))
        self.assertEqual(g['total_population_variance'],0)
        self.assertIsNone(g['centered_covariance_participation_effective_rank'])
        self.assertAlmostEqual(g['qq_cosine']['mean'],1.)

    def test_geometry_rank_one(self):
        g,_=e.geometry(torch.tensor([[1.,1.],[2.,1.],[3.,1.]]))
        self.assertAlmostEqual(g['centered_covariance_participation_effective_rank'],1.)
        self.assertAlmostEqual(g['total_population_variance'],2/3)

    def test_score_selection(self):
        q=torch.tensor([1.,0.]);k=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]])
        s=e.scores(q,k,[0,0,0])
        self.assertEqual(s['selected'],[0])
        self.assertEqual(e.comparison(s,s)['threshold_crossings'],0)

    def test_output_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                e.run(Path(d))

    def test_numeric_output_json_serializable(self):
        q=torch.tensor([1.,0.]);k=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]])
        s=e.scores(q,k,[1,2,3])
        payload={**e.comparison(s,s),'crossings':sum(a != (b>=.05) for a,b in zip(s['threshold'],s['final']))}
        self.assertIn('crossings',json.loads(json.dumps(payload,allow_nan=False)))


if __name__=='__main__':
    unittest.main()
