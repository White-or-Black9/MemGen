import random
import tempfile
import unittest
from pathlib import Path

import torch

from memgen.model.latent_memory_bank import LatentMemoryBank, LatentMemoryBankConfig
from scripts.eval.review_retrieval_policies import (
    POLICIES, capture_bank, capture_rng, derive_last_write, fingerprint,
    load_snapshot, policy_adapter, policy_seed, restore_bank, restore_rng,
    save_snapshot, select_indices,
)


class PolicyTests(unittest.TestCase):
    def choose(self, policy, full=(2, 0), **overrides):
        args = dict(full_indices=full, cosine=[.8, .5, .9], ages=[1, 1, 4],
                    last_write=[2, 9, 3], rng_seed=42)
        args.update(overrides)
        return select_indices(policy, **args)

    def bank(self, threshold=-1, top_k=2):
        bank = LatentMemoryBank(LatentMemoryBankConfig(
            enabled=True, top_k=top_k, threshold=threshold, decay_alpha=.05,
            max_slots=4, update_policy="append"))
        for values in [[1., 0.], [0., 1.], [1., 1.]]:
            bank.write(torch.tensor([values] * 8))
        return bank

    def test_stable_rankings_and_order(self):
        self.assertEqual(self.choose("full"), [0, 2])
        self.assertEqual(self.choose("native_order"), [2, 0])
        self.assertEqual(self.choose("last_written"), [1, 2])
        self.assertEqual(self.choose("cosine_only"), [0, 2])
        self.assertEqual(self.choose("recency_only"), [0, 1])
        self.assertEqual(self.choose("cosine_only", cosine=[1, 1, 1]), [0, 1])

    def test_counts_zero_one_two_and_empty(self):
        for policy in POLICIES:
            for full in [(), (0,), (2, 0)]:
                self.assertEqual(len(self.choose(policy, full)), len(full))
            self.assertEqual(select_indices(policy, full_indices=[], cosine=[], ages=[],
                                            last_write=[], rng_seed=1), [])

    def test_invalid_metadata_capacity_policy(self):
        for args in [dict(full=(3,)), dict(full=(0, 0)), dict(ages=[]),
                     dict(last_write=[1, None, 2]), dict(cosine=[float("nan"), 1, 2])]:
            with self.assertRaises(ValueError):
                self.choose("full", **args)
        with self.assertRaises(ValueError):
            self.choose("unknown")

    def test_random_independent_reproducible_without_replacement(self):
        before = random.getstate()
        chosen = self.choose("random")
        self.assertEqual(before, random.getstate())
        self.assertEqual(chosen, self.choose("random"))
        self.assertEqual(len(chosen), len(set(chosen)))
        self.assertNotEqual(policy_seed(42, 0, 0), policy_seed(42, 0, 1))

    def test_last_write_refresh_and_eviction(self):
        events = [dict(target_slot_index=i, write_action=action, write_count_after_write=step)
                  for i, action, step in [(0, "insert", 1), (1, "insert", 2),
                                          (0, "replace_matched", 3), (1, "evict_oldest_insert", 4)]]
        self.assertEqual(derive_last_write(events, 2), [3, 4])
        with self.assertRaises(ValueError):
            derive_last_write(events[:1], 2)

    def test_snapshot_all_state_and_tensor_isolation(self):
        bank = self.bank()
        state = capture_bank(bank)
        clone = restore_bank(state)
        self.assertEqual(fingerprint(state), fingerprint(capture_bank(clone)))
        clone._slots[0].memory.add_(3)
        self.assertNotEqual(fingerprint(state), fingerprint(capture_bank(clone)))
        self.assertTrue(torch.equal(bank._slots[0].memory, state["_slots"][0].memory))

    def test_snapshot_roundtrip_and_hash_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.pt"
            state = capture_bank(self.bank())
            save_snapshot(path, state, capture_rng(), [1, 2, 3], {"test": True})
            payload = load_snapshot(path)
            self.assertEqual(payload["bank_sha256"], fingerprint(state))
            with self.assertRaises(FileExistsError):
                save_snapshot(path, state, capture_rng(), [1, 2, 3], {})
            payload["bank"]["_slots"][0].key.add_(1)
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                load_snapshot(path)

    def test_rng_restore(self):
        state = capture_rng()
        expected = (random.random(), torch.rand(2))
        restore_rng(state)
        self.assertEqual(expected[0], random.random())
        self.assertTrue(torch.equal(expected[1], torch.rand(2)))

    def test_adapter_matches_native_full_and_restores_hook(self):
        state = capture_bank(self.bank())
        query = torch.tensor([.4, .9])
        native = restore_bank(state).retrieve_with_context(query)
        for policy in POLICIES + ("native_order",):
            bank = restore_bank(state)
            trace = []
            with policy_adapter(bank, policy, last_write=[1, 2, 3], seed=42,
                                context=0, query=0, trace=trace):
                result = bank.retrieve_with_context(query)
                self.assertEqual(len(result.slots), len(native.slots))
                self.assertEqual(sum(s.memory.shape[0] for s in result.slots), 16)
                result.slots[0].memory.add_(100)
                self.assertLess(float(bank._slots[result.retrieved_indices[0]].memory.max()), 2)
                if policy in ("full", "native_order"):
                    self.assertEqual(set(result.retrieved_indices), set(native.retrieved_indices))
                    self.assertEqual(result.scores, native.scores)
            self.assertNotIn("retrieve_with_context", bank.__dict__)

    def test_frozen_proxy_policy_order_independence(self):
        from scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5 import _QueryReadOnlyBank
        state = capture_bank(self.bank())
        bank = restore_bank(state)
        proxy = _QueryReadOnlyBank(bank, {}, freeze_retrieval_state=True, preserve_on_reset=True)
        proxy.begin_query()
        expected = {}
        for policy in list(POLICIES) + list(reversed(POLICIES)):
            trace = []
            with policy_adapter(bank, policy, last_write=[1, 2, 3], seed=42,
                                context=0, query=0, trace=trace):
                result = proxy.retrieve_with_context(torch.tensor([1., 0.]))
            selected = result.retrieved_indices
            self.assertEqual(expected.setdefault(policy, selected), selected)
            self.assertEqual(fingerprint(state), fingerprint(capture_bank(bank)))

    def test_empty_and_threshold_rejection_adapter(self):
        for bank in [LatentMemoryBank(LatentMemoryBankConfig(enabled=True)), self.bank(threshold=1)]:
            trace = []
            with policy_adapter(bank, "random", last_write=list(range(len(bank))), seed=42,
                                context=0, query=0, trace=trace):
                result = bank.retrieve_with_context(torch.tensor([-1., -1.]))
                self.assertEqual(result.slots, [])

    def test_unequal_lengths_fail_budget(self):
        bank = self.bank()
        bank._slots[1].memory = bank._slots[1].memory[:1]
        with policy_adapter(bank, "random", last_write=[1, 2, 3], seed=42,
                            context=0, query=0, trace=[]):
            with self.assertRaises(ValueError):
                bank.retrieve_with_context(torch.tensor([1., 1.]))

    def test_smoke_scope_and_config_guards(self):
        from scripts.eval.review_m2_smoke import parser, run
        for cli in [["--seed", "142"], ["--question-limit", "100"], ["--context-index", "1"],
                    ["--top-k", "1"], ["--strict-official-eventqa-prompt"],
                    ["--eventqa-protocol", "independent_episode"]]:
            with self.assertRaises(ValueError):
                run(parser().parse_args(cli))

    def test_query_runtime_contracts(self):
        from scripts.eval.review_m2_smoke import verify_query
        bank = self.bank()
        before = fingerprint(capture_bank(bank))
        valid = dict(query_write_count_delta=0, query_read_only_enforced=True,
                     bank_snapshot_changed_after_query=False, query_retrieval_invocation_count=1)
        verify_query(valid, [{}], before, bank)
        for overrides in [dict(query_write_count_delta=1), dict(query_read_only_enforced=False),
                          dict(bank_snapshot_changed_after_query=True),
                          dict(query_retrieval_invocation_count=2)]:
            with self.assertRaises(RuntimeError):
                verify_query(dict(valid, **overrides), [{}], before, bank)
        with self.assertRaises(RuntimeError):
            verify_query(valid, [], before, bank)
        bank._memory_write_count += 1
        with self.assertRaises(RuntimeError):
            verify_query(valid, [{}], before, bank)


if __name__ == "__main__":
    unittest.main()
