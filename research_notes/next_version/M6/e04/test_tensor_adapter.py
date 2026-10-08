"""CPU-only tensor checks: run with CUDA_VISIBLE_DEVICES='' in memgen Python."""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import unittest
import torch

spec=importlib.util.spec_from_file_location('content',Path(__file__).with_name('run_content.py'))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
from scripts.eval.review_retrieval_policies import fingerprint


@dataclass
class Slot:
    memory:object
    key:object
    metadata:dict
    created_step:int=0
    last_retrieved_step:int=0


@dataclass
class Retrieved:
    slots:list
    scores:tuple
    retrieved_indices:tuple
    retrieved_scores:tuple


class Bank:
    def __init__(self):
        self._slots=[Slot(torch.full((8,1536),float(i),dtype=torch.bfloat16),
                          torch.full((1536,),float(i),dtype=torch.bfloat16),{'index':i}) for i in range(3)]
    def retrieve_with_context(self,q,**kwargs):
        return Retrieved([self._slots[0],self._slots[2]],(.3,.2,.4),(0,2),(.3,.4))


class AdapterTests(unittest.TestCase):
    def test_actual_slot_values_and_no_mutation(self):
        bank=Bank();before=fingerprint(bank.__dict__);trace=[]
        with p.fixed_return(bank,[0,1],trace):
            r=bank.retrieve_with_context(torch.zeros(1536))
            self.assertEqual(r.retrieved_indices,(0,1));self.assertEqual(r.retrieved_scores,(.3,.2))
            self.assertTrue(torch.equal(r.slots[1].memory,bank._slots[1].memory))
            self.assertNotEqual(r.slots[1].memory.data_ptr(),bank._slots[1].memory.data_ptr())
            self.assertEqual(trace[0]['native_full_indices'],[0,2])
            self.assertEqual(trace[0]['actual_indices'],[0,1])
        self.assertEqual(fingerprint(bank.__dict__),before)
        self.assertNotIn('retrieve_with_context',bank.__dict__)

    def test_identity_adapter_bitwise(self):
        bank=Bank();native=bank.retrieve_with_context(None)
        with p.fixed_return(bank,[0,2],[]):result=bank.retrieve_with_context(None)
        for a,b in zip(native.slots,result.slots):self.assertTrue(torch.equal(a.memory,b.memory))

    def test_restoration_after_error(self):
        bank=Bank()
        with self.assertRaises(RuntimeError):
            with p.fixed_return(bank,[0,1],[]):raise RuntimeError('synthetic')
        self.assertNotIn('retrieve_with_context',bank.__dict__)

    def test_shape_budget_rejected(self):
        bank=Bank();bank._slots[1].memory=bank._slots[1].memory[:7]
        with self.assertRaises(ValueError):
            with p.fixed_return(bank,[0,1],[]):bank.retrieve_with_context(None)

    def test_noncanonical_duplicate_and_missing(self):
        for indices in [[1,0],[0,0],[0,3]]:
            with self.assertRaises(ValueError):
                with p.fixed_return(Bank(),indices,[]):pass


if __name__=='__main__':unittest.main()
