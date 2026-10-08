import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('pilot',Path(__file__).with_name('run_pilot.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class PilotTests(unittest.TestCase):
    def roster(self):
        return [dict(sample_id=f'c{c}-q{q}',context_index=c,query_id=q) for c in range(5) for q in range(2)]

    def result(self,on):
        return dict(generations=[dict(query_weaver_invoke_count=1,weaver_output_generated=True,
            weaver_output_consumed=True,direct_injection_applied=False,
            weaver_conditioning_token_count=16 if on else 0,weaver_conditioned_on_retrieved_memory=on)],
            context_memorization_performed=False,construction_turn_diagnostics=[],effective_generation_max_length=40)

    def test_budget(self):
        tasks=p.tasks(self.roster());self.assertEqual(len(tasks),40)
        self.assertEqual([(t['arm'],t['repeat']) for t in tasks[:4]],[('on',0),('off',0),('on',1),('off',1)])

    def test_roster_missing(self):
        with self.assertRaises(ValueError):p.tasks(self.roster()[:-1])

    def test_roster_duplicate_question(self):
        r=self.roster();r[1]['query_id']=0
        with self.assertRaises(ValueError):p.tasks(r)

    def test_roster_context_imbalance(self):
        r=self.roster();r[0]['context_index']=1
        with self.assertRaises(ValueError):p.tasks(r)

    def test_on_valid(self):
        p.validate_generation(self.result(True),[dict(n=2,latent_count=16)],'on')

    def test_off_still_retrieves(self):
        p.validate_generation(self.result(False),[dict(n=2,latent_count=16)],'off')
        with self.assertRaises(ValueError):p.validate_generation(self.result(False),[dict(n=0,latent_count=0)],'off')

    def test_off_conditioning_rejected(self):
        with self.assertRaises(ValueError):p.validate_generation(self.result(True),[dict(n=2,latent_count=16)],'off')

    def test_construction_rejected(self):
        r=self.result(True);r['context_memorization_performed']=True
        with self.assertRaises(ValueError):p.validate_generation(r,[dict(n=2,latent_count=16)],'on')

    def test_repeat_prediction_and_selection(self):
        a=dict(prediction='a',selection=[1,2],rendered_prompt_sha256='p')
        self.assertTrue(p.same_replay(a,copy.deepcopy(a)))
        self.assertFalse(p.same_replay(a,dict(a,prediction='b')))
        self.assertFalse(p.same_replay(a,dict(a,selection=[2,1])))

    def test_full_denominator_transitions(self):
        on={};off={}
        for c in range(5):
            for q in range(2):
                r=dict(strict_candidate_correct=False,candidate_valid=False,parsed_nonempty=False,
                       raw_gold_any=False,official_raw_recall=0)
                on[(c,q)]=dict(r);off[(c,q)]=dict(r)
        on[(0,0)].update(strict_candidate_correct=True,candidate_valid=True,parsed_nonempty=True)
        off[(1,0)].update(strict_candidate_correct=True,candidate_valid=True,parsed_nonempty=True)
        t=p.pair_tables(on,off)
        self.assertEqual(t['denominator'],10)
        self.assertEqual(t['strict_correctness']['on_1_off_0'],1)
        self.assertEqual(t['strict_correctness']['on_0_off_1'],1)
        self.assertEqual(t['strict_correctness']['on_0_off_0'],8)
        self.assertEqual(sum(t['off_to_on_CWI'].values()),10)
        self.assertEqual(t['context_equal_mean'],0)

    def test_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):p.prepare(Path(d))


if __name__=='__main__':unittest.main()
