import copy
import unittest
from pathlib import Path
import importlib.util

spec=importlib.util.spec_from_file_location('failure',Path(__file__).with_name('analyze_failures.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


class OfficialMock:
    def normalize_answer(self,text):return ' '.join(text.lower().split())


class FailureTests(unittest.TestCase):
    def test_prefix_not_answer_loss(self):
        x=a.diagnostic('行\nCorrect answer.',['Correct answer.','Wrong answer.'],OfficialMock())
        self.assertEqual(x['unique_candidate_mention'],0)
        self.assertEqual(x['last_nonempty_line'],'Correct answer.')
        self.assertEqual(x['leading_nonempty_lines'],['行'])

    def test_multiple_mentions_not_unique(self):
        x=a.diagnostic('Yes. No.',['Yes.','No.'],OfficialMock())
        self.assertIsNone(x['unique_candidate_mention'])

    def test_empty(self):
        x=a.diagnostic('\n',['a'],OfficialMock())
        self.assertEqual(x['last_nonempty_line'],'');self.assertIsNone(x['unique_candidate_mention'])

    def test_blank_lines_ignored_only_for_diagnostic(self):
        x=a.diagnostic('日志\n\nAnswer.\n',['Answer.'],OfficialMock())
        self.assertEqual(x['leading_nonempty_lines'],['日志'])
        self.assertTrue(x['diagnostic_only'])

    def snapshot(self):
        return dict(combined_frozen_bank_hash='bank',slots=[dict(slot_index=0,memory_tensor_hash='m',
            key_tensor_hash='k',memory_shape=[8,1536],key_shape=[1536],memory_dtype='bf16',key_dtype='bf16')])

    def test_snapshot_equal(self):
        self.assertTrue(a.compare_bank_snapshots(self.snapshot(),self.snapshot()))

    def test_snapshot_tensor_mismatch(self):
        x=self.snapshot();x['slots'][0]['memory_tensor_hash']='changed'
        with self.assertRaises(ValueError):a.compare_bank_snapshots(self.snapshot(),x)

    def test_snapshot_shape_mismatch(self):
        x=self.snapshot();x['slots'][0]['memory_shape']=[7,1536]
        with self.assertRaises(ValueError):a.compare_bank_snapshots(self.snapshot(),x)


if __name__=='__main__':unittest.main()
