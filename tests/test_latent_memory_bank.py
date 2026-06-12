import unittest

import torch

from memgen.model.latent_memory_bank import (
    LatentMemoryBank,
    LatentMemoryBankConfig,
)


class LatentMemoryBankTest(unittest.TestCase):
    def test_config_rejects_batch_size_greater_than_one(self):
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            LatentMemoryBankConfig(batch_size=2)

    def test_disabled_bank_is_noop(self):
        bank = LatentMemoryBank()
        memory = torch.randn(4, 3, requires_grad=True)

        self.assertFalse(bank.write(memory))
        self.assertEqual(len(bank), 0)
        self.assertEqual(bank.retrieve(memory), [])

    def test_empty_retrieval(self):
        bank = LatentMemoryBank(LatentMemoryBankConfig(enabled=True))

        self.assertEqual(bank.retrieve(torch.randn(2, 3)), [])

    def test_write_detaches_clones_and_tracks_source(self):
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(enabled=True, storage_device="cpu")
        )
        memory = torch.randn(4, 3, requires_grad=True)
        original = memory.detach().clone()

        self.assertTrue(bank.write(memory, {"sample_id": 7}))
        memory.data.add_(10)
        state = bank.state_dict()
        stored = state["slots"][0]

        self.assertEqual(len(bank), 1)
        self.assertFalse(stored["memory"].requires_grad)
        self.assertIsNone(stored["memory"].grad_fn)
        self.assertTrue(torch.equal(stored["memory"], original))
        self.assertEqual(stored["original_dtype"], str(memory.dtype))
        self.assertEqual(stored["original_device"], str(memory.device))
        self.assertEqual(stored["metadata"]["sample_id"], 7)

    def test_max_slots_and_replace_oldest(self):
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                max_slots=2,
                update_policy="replace_oldest",
            )
        )
        bank.write(torch.full((2, 3), 1.0), {"id": 1})
        bank.write(torch.full((2, 3), 2.0), {"id": 2})
        bank.write(torch.full((2, 3), 3.0), {"id": 3})

        ids = [slot["metadata"]["id"] for slot in bank.state_dict()["slots"]]
        self.assertEqual(len(bank), 2)
        self.assertEqual(ids, [3, 2])

    def test_append_policy_refuses_write_when_full(self):
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                max_slots=1,
                update_policy="append",
            )
        )
        self.assertTrue(bank.write(torch.ones(2, 3), {"id": 1}))
        step_before = bank.debug_summary()["step"]
        self.assertFalse(bank.write(torch.zeros(2, 3), {"id": 2}))
        self.assertEqual(bank.state_dict()["slots"][0]["metadata"]["id"], 1)
        self.assertEqual(bank.debug_summary()["step"], step_before)

    def test_topk_retrieval_and_output_dtype(self):
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                top_k=1,
                retrieve_policy="topk",
                decay_alpha=0.0,
                storage_device="cpu",
            )
        )
        bank.write(torch.tensor([[1.0, 0.0], [1.0, 0.0]]), {"id": "near"})
        bank.write(torch.tensor([[0.0, 1.0], [0.0, 1.0]]), {"id": "far"})

        result = bank.retrieve(
            torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
            device="cpu",
            dtype=torch.float64,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].metadata["id"], "near")
        self.assertEqual(result[0].memory.dtype, torch.float64)
        self.assertEqual(result[0].memory.device.type, "cpu")

    def test_retrieve_accepts_prepooled_query(self):
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                top_k=1,
                retrieve_policy="topk",
                decay_alpha=0.0,
            )
        )
        bank.write(torch.tensor([[1.0, 0.0]]), {"id": "match"})

        result = bank.retrieve(torch.tensor([1.0, 0.0]))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].metadata["id"], "match")

    def test_retrieved_slot_mutation_does_not_change_bank_state(self):
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                retrieve_policy="topk",
                decay_alpha=0.0,
            )
        )
        memory = torch.tensor([[1.0, 0.0]])
        bank.write(memory, {"nested": {"id": "original"}})

        result = bank.retrieve(memory)
        result[0].memory.add_(10)
        result[0].key.zero_()
        result[0].metadata["nested"]["id"] = "mutated"
        stored = bank.state_dict()["slots"][0]

        self.assertTrue(torch.equal(stored["memory"], memory))
        self.assertTrue(torch.equal(stored["key"], torch.tensor([1.0, 0.0])))
        self.assertEqual(stored["metadata"]["nested"]["id"], "original")

    def test_threshold_filter(self):
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                threshold=0.9,
                retrieve_policy="threshold",
                decay_alpha=0.0,
            )
        )
        bank.write(torch.tensor([[0.0, 1.0]]))

        result = bank.retrieve(torch.tensor([[1.0, 0.0]]))

        self.assertEqual(result, [])

    def test_recency_decay_affects_score(self):
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                top_k=2,
                retrieve_policy="topk",
                decay_alpha=1.0,
            )
        )
        memory = torch.tensor([[1.0, 0.0]])
        bank.write(memory, {"id": "old"})
        bank.write(memory, {"id": "new"})

        result = bank.retrieve(memory)

        self.assertEqual([slot.metadata["id"] for slot in result], ["new", "old"])
        self.assertGreater(result[0].last_score, result[1].last_score)

    def test_replace_policy_replaces_lowest_scored_slot(self):
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                max_slots=2,
                top_k=2,
                retrieve_policy="topk",
                update_policy="replace",
                decay_alpha=0.0,
            )
        )
        bank.write(torch.tensor([[1.0, 0.0]]), {"id": "high"})
        bank.write(torch.tensor([[0.0, 1.0]]), {"id": "low"})
        bank.retrieve(torch.tensor([1.0, 0.0]))

        bank.write(torch.tensor([[-1.0, 0.0]]), {"id": "replacement"})

        ids = [slot["metadata"]["id"] for slot in bank.state_dict()["slots"]]
        self.assertEqual(ids, ["high", "replacement"])

    def test_replace_policy_falls_back_to_oldest_when_all_unscored(self):
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                max_slots=2,
                update_policy="replace",
            )
        )
        bank.write(torch.tensor([[1.0, 0.0]]), {"id": "oldest"})
        bank.write(torch.tensor([[0.0, 1.0]]), {"id": "newer"})

        bank.write(torch.tensor([[-1.0, 0.0]]), {"id": "replacement"})

        ids = [slot["metadata"]["id"] for slot in bank.state_dict()["slots"]]
        self.assertEqual(ids, ["replacement", "newer"])

    def test_reset_clears_memory_and_step(self):
        bank = LatentMemoryBank(LatentMemoryBankConfig(enabled=True))
        bank.write(torch.ones(2, 3))
        bank.retrieve(torch.ones(2, 3))

        bank.reset()

        self.assertEqual(len(bank), 0)
        self.assertEqual(bank.debug_summary()["step"], 0)
        self.assertEqual(bank.debug_summary()["memory_write_count"], 0)
        self.assertEqual(bank.debug_summary()["memory_retrieve_count"], 0)

    def test_build_query_uses_recent_tokens(self):
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(enabled=True, pool_last_n=2)
        )
        states = torch.tensor(
            [[100.0, 100.0], [1.0, 3.0], [3.0, 5.0]]
        )

        query = bank.build_query(states)

        self.assertTrue(torch.equal(query, torch.tensor([2.0, 4.0])))

    def test_bad_shapes_and_dtypes_raise_clear_errors(self):
        bank = LatentMemoryBank(LatentMemoryBankConfig(enabled=True))

        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            bank.write(torch.randn(2, 4, 3))
        with self.assertRaisesRegex(ValueError, "tokens, hidden"):
            bank.write(torch.randn(3))
        with self.assertRaisesRegex(ValueError, "empty dimension"):
            bank.write(torch.empty(0, 3))
        with self.assertRaisesRegex(TypeError, "floating-point"):
            bank.write(torch.ones(2, 3, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "hidden size mismatch"):
            bank.write(torch.randn(2, 3))
            bank.retrieve(torch.randn(2, 4))


if __name__ == "__main__":
    unittest.main()
