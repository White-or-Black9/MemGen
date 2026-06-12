import unittest

import torch

from memgen.model.latent_memory_bank import (
    LatentMemoryBank,
    LatentMemoryBankConfig,
    LatentMemoryRetrievalResult,
)


class LatentMemoryBankTest(unittest.TestCase):
    def _thread_bank(self, max_slots=8, threshold=0.7):
        return LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                max_slots=max_slots,
                threshold=threshold,
                retrieve_policy="threshold_topk",
                update_policy="thread_update",
                decay_alpha=0.0,
            )
        )

    def _thread_write(self, bank, memory, metadata=None):
        retrieval_result = bank.retrieve_with_context(memory)
        self.assertTrue(
            bank.write_back(memory, retrieval_result, metadata=metadata)
        )
        return retrieval_result

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

    def test_retrieve_with_context_empty_bank(self):
        bank = LatentMemoryBank(LatentMemoryBankConfig(enabled=True))

        result = bank.retrieve_with_context(torch.randn(2, 3))

        self.assertEqual(result.slots, [])
        self.assertEqual(result.scores, ())
        self.assertIsNone(result.max_score)
        self.assertIsNone(result.argmax_index)
        self.assertFalse(result.threshold_passed)
        self.assertEqual(result.retrieved_indices, ())
        self.assertEqual(result.retrieved_scores, ())
        self.assertEqual(result.bank_step, 0)

    def test_disabled_retrieve_with_context_empty(self):
        bank = LatentMemoryBank()

        result = bank.retrieve_with_context(torch.randn(2, 3))

        self.assertEqual(result.slots, [])
        self.assertEqual(result.scores, ())
        self.assertIsNone(result.max_score)
        self.assertIsNone(result.argmax_index)
        self.assertFalse(result.threshold_passed)
        self.assertEqual(result.retrieved_indices, ())
        self.assertEqual(result.retrieved_scores, ())
        self.assertEqual(result.bank_step, 0)

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

    def test_retrieve_with_context_scores_are_full_bank_order(self):
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                top_k=1,
                retrieve_policy="topk",
                decay_alpha=0.0,
            )
        )
        bank.write(torch.tensor([[1.0, 0.0]]), {"id": "positive"})
        bank.write(torch.tensor([[0.0, 1.0]]), {"id": "orthogonal"})
        bank.write(torch.tensor([[-1.0, 0.0]]), {"id": "negative"})

        result = bank.retrieve_with_context(torch.tensor([1.0, 0.0]))

        self.assertEqual(len(result.scores), len(bank))
        self.assertAlmostEqual(result.scores[0], 1.0)
        self.assertAlmostEqual(result.scores[1], 0.0)
        self.assertAlmostEqual(result.scores[2], -1.0)

    def test_retrieve_with_context_max_score_and_argmax(self):
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                top_k=2,
                retrieve_policy="topk",
                decay_alpha=0.0,
            )
        )
        bank.write(torch.tensor([[1.0, 0.0]]), {"id": "first"})
        bank.write(torch.tensor([[1.0, 0.0]]), {"id": "second"})

        result = bank.retrieve_with_context(torch.tensor([1.0, 0.0]))

        self.assertAlmostEqual(result.max_score, 1.0)
        self.assertEqual(result.argmax_index, 0)
        self.assertEqual(result.retrieved_indices, (0, 1))

    def test_retrieve_with_context_threshold_empty_preserves_argmax(self):
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                threshold=0.9,
                retrieve_policy="threshold_topk",
                decay_alpha=0.0,
            )
        )
        bank.write(torch.tensor([[0.0, 1.0]]), {"id": "only"})

        result = bank.retrieve_with_context(torch.tensor([1.0, 0.0]))

        self.assertEqual(result.slots, [])
        self.assertAlmostEqual(result.max_score, 0.0)
        self.assertEqual(result.argmax_index, 0)
        self.assertFalse(result.threshold_passed)
        self.assertEqual(result.retrieved_indices, ())
        self.assertEqual(result.retrieved_scores, ())

    def test_retrieve_with_context_threshold_topk_indices(self):
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                top_k=2,
                threshold=0.5,
                retrieve_policy="threshold_topk",
                decay_alpha=0.0,
            )
        )
        bank.write(torch.tensor([[0.0, 1.0]]), {"id": "far"})
        bank.write(torch.tensor([[1.0, 0.0]]), {"id": "near"})
        bank.write(torch.tensor([[1.0, 1.0]]), {"id": "second"})

        result = bank.retrieve_with_context(torch.tensor([1.0, 0.0]))

        self.assertEqual(result.retrieved_indices, (1, 2))
        self.assertEqual(
            [slot.metadata["id"] for slot in result.slots],
            ["near", "second"],
        )
        self.assertEqual(len(result.retrieved_scores), 2)
        self.assertAlmostEqual(result.retrieved_scores[0], result.scores[1])
        self.assertAlmostEqual(result.retrieved_scores[1], result.scores[2])
        self.assertTrue(result.threshold_passed)

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

    def test_retrieve_with_context_returns_detached_clones(self):
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                retrieve_policy="topk",
                decay_alpha=0.0,
            )
        )
        memory = torch.tensor([[1.0, 0.0]], requires_grad=True)
        bank.write(memory, {"nested": {"id": "original"}})

        result = bank.retrieve_with_context(memory)
        retrieved = result.slots[0]
        self.assertFalse(retrieved.memory.requires_grad)
        self.assertIsNone(retrieved.memory.grad_fn)
        retrieved.memory.add_(10)
        retrieved.key.zero_()
        retrieved.metadata["nested"]["id"] = "mutated"
        stored = bank.state_dict()["slots"][0]

        self.assertTrue(torch.equal(stored["memory"], memory.detach()))
        self.assertTrue(torch.equal(stored["key"], torch.tensor([1.0, 0.0])))
        self.assertEqual(stored["metadata"]["nested"]["id"], "original")

    def test_retrieve_legacy_behavior_unchanged(self):
        config = LatentMemoryBankConfig(
            enabled=True,
            top_k=1,
            threshold=0.5,
            retrieve_policy="threshold_topk",
            decay_alpha=0.0,
        )
        context_bank = LatentMemoryBank(config)
        legacy_bank = LatentMemoryBank(config)
        for bank in (context_bank, legacy_bank):
            bank.write(torch.tensor([[0.0, 1.0]]), {"id": "far"})
            bank.write(torch.tensor([[1.0, 0.0]]), {"id": "near"})

        context_slots = context_bank.retrieve_with_context(
            torch.tensor([1.0, 0.0])
        ).slots
        legacy_slots = legacy_bank.retrieve(torch.tensor([1.0, 0.0]))

        self.assertIsInstance(legacy_slots, list)
        self.assertEqual(
            [slot.metadata["id"] for slot in legacy_slots],
            [slot.metadata["id"] for slot in context_slots],
        )
        self.assertEqual(
            [slot.last_score for slot in legacy_slots],
            [slot.last_score for slot in context_slots],
        )

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

    def test_thread_update_empty_bank_inserts(self):
        bank = self._thread_bank()
        retrieval_result = bank.retrieve_with_context(
            torch.tensor([1.0, 0.0])
        )

        self.assertTrue(
            bank.write_back(
                torch.tensor([[1.0, 0.0]]),
                retrieval_result,
                {"id": "first"},
            )
        )

        summary = bank.debug_summary()
        self.assertEqual(len(bank), 1)
        self.assertEqual(summary["thread_insert_count"], 1)
        self.assertEqual(summary["last_write_back"]["write_action"], "insert")
        self.assertEqual(
            summary["last_write_back"]["update_reason"], "empty_bank"
        )

    def test_thread_update_low_score_not_full_inserts_new_thread(self):
        bank = self._thread_bank(max_slots=3, threshold=0.7)
        self._thread_write(
            bank,
            torch.tensor([[1.0, 0.0]]),
            {"id": "first"},
        )
        retrieval_result = bank.retrieve_with_context(
            torch.tensor([0.0, 1.0])
        )

        bank.write_back(
            torch.tensor([[0.0, 1.0]]),
            retrieval_result,
            {"id": "second"},
        )

        ids = [slot["metadata"]["id"] for slot in bank.state_dict()["slots"]]
        event = bank.debug_summary()["last_write_back"]
        self.assertEqual(ids, ["first", "second"])
        self.assertEqual(event["write_action"], "insert")
        self.assertEqual(event["update_reason"], "new_thread")
        self.assertTrue(event["inserted_new_thread"])

    def test_thread_update_low_score_full_evicts_oldest_then_inserts(self):
        bank = self._thread_bank(max_slots=2, threshold=0.7)
        self._thread_write(
            bank,
            torch.tensor([[1.0, 0.0]]),
            {"id": "oldest"},
        )
        self._thread_write(
            bank,
            torch.tensor([[0.0, 1.0]]),
            {"id": "newer"},
        )
        retrieval_result = bank.retrieve_with_context(
            torch.tensor([[-1.0, 0.0]])
        )

        bank.write_back(
            torch.tensor([[-1.0, 0.0]]),
            retrieval_result,
            {"id": "new-thread"},
        )

        ids = [slot["metadata"]["id"] for slot in bank.state_dict()["slots"]]
        summary = bank.debug_summary()
        event = summary["last_write_back"]
        self.assertEqual(ids, ["new-thread", "newer"])
        self.assertEqual(event["write_action"], "evict_oldest_insert")
        self.assertEqual(event["evicted_slot_index"], 0)
        self.assertIsNone(event["replaced_slot_index"])
        self.assertEqual(summary["capacity_evict_count"], 1)

    def test_thread_update_high_score_replaces_argmax_even_when_not_full(self):
        bank = self._thread_bank(max_slots=3)
        self._thread_write(
            bank,
            torch.tensor([[1.0, 0.0]]),
            {"id": "matched"},
        )
        retrieval_result = bank.retrieve_with_context(
            torch.tensor([1.0, 0.0])
        )

        bank.write_back(
            torch.tensor([[2.0, 0.0]]),
            retrieval_result,
            {"id": "updated"},
        )

        ids = [slot["metadata"]["id"] for slot in bank.state_dict()["slots"]]
        event = bank.debug_summary()["last_write_back"]
        self.assertEqual(ids, ["updated"])
        self.assertEqual(event["write_action"], "replace_matched")
        self.assertEqual(event["replaced_slot_index"], 0)
        self.assertIsNone(event["evicted_slot_index"])

    def test_thread_update_high_score_replaces_argmax_when_full(self):
        bank = self._thread_bank(max_slots=2)
        self._thread_write(
            bank,
            torch.tensor([[1.0, 0.0]]),
            {"id": "x"},
        )
        self._thread_write(
            bank,
            torch.tensor([[0.0, 1.0]]),
            {"id": "y"},
        )
        retrieval_result = bank.retrieve_with_context(
            torch.tensor([0.0, 1.0])
        )

        bank.write_back(
            torch.tensor([[0.0, 2.0]]),
            retrieval_result,
            {"id": "updated-y"},
        )

        ids = [slot["metadata"]["id"] for slot in bank.state_dict()["slots"]]
        self.assertEqual(ids, ["x", "updated-y"])
        self.assertEqual(
            bank.debug_summary()["last_write_back"]["replaced_slot_index"],
            1,
        )

    def test_thread_update_high_score_replacement_does_not_increase_slot_count(self):
        bank = self._thread_bank(max_slots=4)
        self._thread_write(bank, torch.tensor([[1.0, 0.0]]))
        retrieval_result = bank.retrieve_with_context(
            torch.tensor([1.0, 0.0])
        )
        slot_count = len(bank)

        bank.write_back(torch.tensor([[2.0, 0.0]]), retrieval_result)

        self.assertEqual(len(bank), slot_count)
        self.assertEqual(bank.debug_summary()["matched_replace_count"], 1)

    def test_thread_update_uses_current_argmax_not_stale_last_score(self):
        bank = self._thread_bank(max_slots=3)
        self._thread_write(
            bank,
            torch.tensor([[1.0, 0.0]]),
            {"id": "x"},
        )
        self._thread_write(
            bank,
            torch.tensor([[0.0, 1.0]]),
            {"id": "y"},
        )
        bank.retrieve_with_context(torch.tensor([1.0, 0.0]))
        current_result = bank.retrieve_with_context(torch.tensor([0.0, 1.0]))

        bank.write_back(
            torch.tensor([[0.0, 2.0]]),
            current_result,
            {"id": "updated-y"},
        )

        ids = [slot["metadata"]["id"] for slot in bank.state_dict()["slots"]]
        self.assertEqual(current_result.argmax_index, 1)
        self.assertEqual(ids, ["x", "updated-y"])

    def test_thread_update_rejects_stale_bank_step(self):
        bank = self._thread_bank(max_slots=3)
        stale_result = bank.retrieve_with_context(torch.tensor([1.0, 0.0]))
        self._thread_write(bank, torch.tensor([[1.0, 0.0]]))

        with self.assertRaisesRegex(ValueError, "stale retrieval_result"):
            bank.write_back(torch.tensor([[0.0, 1.0]]), stale_result)

    def test_thread_update_rejects_invalid_argmax_index(self):
        bank = self._thread_bank()
        self._thread_write(bank, torch.tensor([[1.0, 0.0]]))
        invalid_result = LatentMemoryRetrievalResult(
            slots=[],
            scores=(1.0,),
            max_score=1.0,
            argmax_index=3,
            threshold_passed=True,
            retrieved_indices=(),
            retrieved_scores=(),
            bank_step=bank.debug_summary()["step"],
        )

        with self.assertRaisesRegex(ValueError, "argmax_index"):
            bank.write_back(torch.tensor([[2.0, 0.0]]), invalid_result)

    def test_thread_update_debug_fields_complete(self):
        bank = self._thread_bank()
        self._thread_write(bank, torch.tensor([[1.0, 0.0]]))
        retrieval_result = bank.retrieve_with_context(
            torch.tensor([1.0, 0.0])
        )
        bank.write_back(torch.tensor([[2.0, 0.0]]), retrieval_result)

        summary = bank.debug_summary()
        event = summary["last_write_back"]
        self.assertEqual(summary["thread_insert_count"], 1)
        self.assertEqual(summary["matched_replace_count"], 1)
        self.assertEqual(summary["capacity_evict_count"], 0)
        self.assertEqual(len(summary["write_back_trace"]), 2)
        self.assertEqual(
            set(event),
            {
                "matched_slot_index",
                "max_score",
                "threshold_passed",
                "retrieved_indices",
                "retrieved_scores",
                "write_action",
                "replaced_slot_index",
                "replaced_slot_score",
                "evicted_slot_index",
                "update_reason",
                "inserted_new_thread",
                "retrieval_bank_step",
            },
        )

    def test_thread_update_disabled_behavior_unchanged(self):
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=False,
                update_policy="thread_update",
            )
        )
        retrieval_result = bank.retrieve_with_context(torch.randn(1, 2))

        self.assertFalse(
            bank.write_back(torch.randn(1, 2), retrieval_result)
        )
        self.assertEqual(len(bank), 0)

    def test_write_back_requires_thread_update_policy(self):
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(enabled=True, update_policy="replace_oldest")
        )
        retrieval_result = bank.retrieve_with_context(torch.randn(1, 2))

        with self.assertRaisesRegex(ValueError, "thread_update"):
            bank.write_back(torch.randn(1, 2), retrieval_result)

    def test_reset_clears_memory_and_step(self):
        bank = LatentMemoryBank(LatentMemoryBankConfig(enabled=True))
        bank.write(torch.ones(2, 3))
        bank.retrieve(torch.ones(2, 3))

        bank.reset()

        self.assertEqual(len(bank), 0)
        self.assertEqual(bank.debug_summary()["step"], 0)
        self.assertEqual(bank.debug_summary()["memory_write_count"], 0)
        self.assertEqual(bank.debug_summary()["memory_retrieve_count"], 0)

    def test_debug_summary_records_append_replace_and_reject_actions(self):
        replace_bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                max_slots=2,
                update_policy="replace_oldest",
            )
        )
        replace_bank.write(torch.ones(1, 2))
        replace_bank.write(torch.ones(1, 2) * 2)
        replace_bank.write(torch.ones(1, 2) * 3)
        replace_summary = replace_bank.debug_summary()
        self.assertEqual(replace_summary["append_count"], 2)
        self.assertEqual(replace_summary["replace_count"], 1)
        self.assertEqual(replace_summary["rejected_write_count"], 0)
        self.assertEqual(
            replace_summary["update_action_trace"],
            ["append", "append", "replace"],
        )
        self.assertEqual(replace_summary["last_update_action"], "replace")

        append_bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                max_slots=1,
                update_policy="append",
            )
        )
        append_bank.write(torch.ones(1, 2))
        self.assertFalse(append_bank.write(torch.ones(1, 2) * 2))
        append_summary = append_bank.debug_summary()
        self.assertEqual(append_summary["append_count"], 1)
        self.assertEqual(append_summary["replace_count"], 0)
        self.assertEqual(append_summary["rejected_write_count"], 1)
        self.assertEqual(
            append_summary["update_action_trace"],
            ["append", "reject_append_full"],
        )
        self.assertEqual(
            append_summary["last_update_action"], "reject_append_full"
        )

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
