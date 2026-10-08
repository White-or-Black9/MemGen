import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec=importlib.util.spec_from_file_location('m4_output',Path(__file__).with_name('audit_output.py'))
b=importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def row(kind,policy='full'):
    return dict(policy=policy,seed=42,context_index=0,query_id=0,context_id='ctx',question='question',
                gold_answers=['gold'],query_sha256='hash',candidate_valid=kind!='invalid',
                strict_candidate_correct=kind=='correct_valid',prediction=kind,parsed=kind,
                selected_candidate=kind if kind!='invalid' else None)


class OutputTests(unittest.TestCase):
    def test_invalid_to_correct_and_wrong_to_correct_conserve_net(self):
        pairs=[(row('correct_valid'),row('invalid','random')),
               (row('correct_valid'),row('wrong_valid','random')),
               (row('invalid'),row('correct_valid','random'))]
        x=b.transitions(pairs)
        self.assertEqual((x['wins'],x['losses'],x['net_correct']),(2,1,1))
        self.assertEqual(x['invalid_correct_net'],0)
        self.assertEqual(x['valid_wrong_correct_net'],1)

    def test_common_valid_is_output_selected_not_full_denominator(self):
        x=b.transitions([(row('correct_valid'),row('invalid','random')),
                         (row('wrong_valid'),row('correct_valid','random'))])
        self.assertEqual(x['common_valid']['denominator'],1)
        self.assertEqual(x['count'],2)
        self.assertIn('selection bias',x['common_valid']['caveat'])

    def test_pair_misalignment_rejected(self):
        f,o=row('invalid'),row('invalid','random')
        o['query_id']=1
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            b.transitions([(f,o)])

    def test_duplicate_identity_rejected(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            b.index_rows([row('invalid'),row('invalid')])

    def test_identical_accuracy_does_not_imply_identical_text(self):
        f,o=row('wrong_valid'),row('wrong_valid','recency_only')
        o['prediction']='another wrong candidate'
        x=b.transitions([(f,o)])
        self.assertEqual(x['equality_counts']['strict_candidate_correct'],1)
        self.assertEqual(x['equality_counts']['prediction'],0)

    def test_empty_common_subset_has_no_invented_accuracy(self):
        x=b.transitions([(row('invalid'),row('invalid','random'))])
        self.assertIsNone(x['common_valid']['full']['value'])

    def test_correct_requires_valid(self):
        f=row('correct_valid'); f['candidate_valid']=False
        with self.assertRaisesRegex(ValueError,'correct but invalid'):
            b.state(f)

    def test_invalid_subtype_none_empty_and_multi(self):
        official=SimpleNamespace(normalize_answer=lambda s:s.lower().strip())
        info=dict(normalized=['gold event','wrong event'])
        base=dict(candidate_valid=False,parser_none=True,parser_empty=False,candidate_match_count=0,parsed=None)
        self.assertEqual(b.invalid_type(base,info,official),'parser_none')
        base.update(parser_none=False,parser_empty=True,parsed='')
        self.assertEqual(b.invalid_type(base,info,official),'parser_empty')
        base.update(parser_empty=False,parsed='gold event and wrong event')
        self.assertEqual(b.invalid_type(base,info,official),'multiple_candidate_text')
        base.update(parsed='not an event')
        self.assertEqual(b.invalid_type(base,info,official),'non_candidate')

    def test_gold_mentioned_invalid_is_not_correct(self):
        official=SimpleNamespace(parse_output=lambda s:s,normalize_answer=lambda s:s.lower().strip(),
                  _process_eventqa_dataset=lambda output,gold:[dict(substring_exact_match=True,eventqa_recall=1)])
        info=dict(normalized=['gold event','wrong event'],gold_normalized=['gold event'],errors=[])
        r=b.m.evaluate('gold event or wrong event',['gold event'],info,{},official)
        self.assertTrue(r['raw_gold_any'])
        self.assertFalse(r['candidate_valid'])
        self.assertEqual(b.state(r),'invalid')


if __name__=='__main__':
    unittest.main()
