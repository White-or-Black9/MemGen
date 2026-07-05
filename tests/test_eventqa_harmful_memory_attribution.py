import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.eval import mab6b_weaver_space_bank_eventqa_65536_n5 as eventqa


SOURCE_RUN = Path(
    "outputs/mab/eventqa_p7_rt005_ut010_cap16_topk2/"
    "20260702T084825Z-eventqa-65536-version-b-weaver-space-bank-n5"
)
FROZEN_BANK = SOURCE_RUN / "frozen_banks/context_4.pt"


class EventQAHarmfulMemoryAttributionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.eval import eventqa_harmful_memory_attribution as attribution

        cls.attribution = attribution

    def test_frozen_loader_restores_config_slot_count_and_tensor_hashes(self):
        source = self.attribution.load_frozen_bank(
            FROZEN_BANK, expected_context_index=4
        )

        self.assertEqual(len(source.bank), 16)
        self.assertEqual(source.bank.config.max_slots, 16)
        self.assertEqual(source.bank.config.top_k, 2)
        self.assertEqual(source.bank.config.retrieve_threshold, 0.05)
        self.assertEqual(source.bank.config.update_threshold, 0.10)
        self.assertEqual(source.bank_snapshot_hash, source.saved_snapshot_hash)
        self.assertEqual(
            source.slot_tensor_hashes[0]["memory_tensor_hash"],
            eventqa._tensor_hash(source.bank._slots[0].memory),
        )

    def test_full_clone_does_not_modify_source_bank(self):
        source = self.attribution.load_frozen_bank(FROZEN_BANK)
        before = self.attribution.bank_state_hash(source.bank)
        clone, spec = self.attribution.clone_bank_for_condition(source, "full")
        clone._slots[0].memory.zero_()

        self.assertEqual(spec.condition_type, "full")
        self.assertEqual(before, self.attribution.bank_state_hash(source.bank))
        self.assertNotEqual(
            self.attribution.bank_state_hash(clone),
            self.attribution.bank_state_hash(source.bank),
        )

    def test_drop_slot_keeps_stable_original_slot_ids(self):
        source = self.attribution.load_frozen_bank(FROZEN_BANK)
        clone, spec = self.attribution.clone_bank_for_condition(
            source, "drop-slot:0"
        )

        ids = [slot.metadata["original_slot_index"] for slot in clone._slots]
        self.assertEqual(ids, list(range(1, 16)))
        self.assertEqual(spec.excluded_original_slot_ids, (0,))

    def test_tuple_order_is_preserved(self):
        source = self.attribution.load_frozen_bank(FROZEN_BANK)
        clone, spec = self.attribution.clone_bank_for_condition(
            source, "tuple-only:1,0"
        )

        ids = [slot.metadata["original_slot_index"] for slot in clone._slots]
        self.assertEqual(ids, [1, 0])
        self.assertEqual(spec.included_original_slot_ids, (1, 0))
        self.assertEqual(spec.forced_original_slot_ids, (1, 0))

    def test_every_clone_starts_from_pristine_bank(self):
        source = self.attribution.load_frozen_bank(FROZEN_BANK)
        first, _ = self.attribution.clone_bank_for_condition(source, "full")
        first._slots.pop()
        second, _ = self.attribution.clone_bank_for_condition(source, "full")

        self.assertEqual(len(second), 16)
        self.assertEqual(
            self.attribution.bank_state_hash(second),
            self.attribution.bank_state_hash(source.bank),
        )

    def test_query_read_only_validator_detects_mutation(self):
        source = self.attribution.load_frozen_bank(FROZEN_BANK)
        before = self.attribution.bank_state_hash(source.bank)

        self.attribution.assert_query_read_only(source.bank, before)
        source.bank._slots[0].access_count += 1
        with self.assertRaisesRegex(RuntimeError, "query modified attribution bank"):
            self.attribution.assert_query_read_only(source.bank, before)

    def test_invalid_frozen_bank_path_fails(self):
        with self.assertRaises(FileNotFoundError):
            self.attribution.load_frozen_bank(Path("missing-context-bank.pt"))

    def test_output_directory_never_overwrites_existing_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            existing = Path(tmpdir) / "already-exists"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                self.attribution.create_output_directory(existing)

    def test_forced_modes_are_explicitly_oracle_diagnostic(self):
        for condition in ("slot-only:0", "tuple-only:1,0"):
            spec = self.attribution.parse_condition(condition)
            self.assertTrue(spec.oracle_diagnostic)
            self.assertEqual(
                spec.forced_original_slot_ids,
                spec.included_original_slot_ids,
            )
        self.assertFalse(self.attribution.parse_condition("drop-slot:0").oracle_diagnostic)

    def test_scorer_and_parser_delegate_to_eventqa_runner(self):
        self.assertIs(self.attribution.score_prediction, eventqa._score_prediction)
        self.assertIs(self.attribution.format_flags, eventqa._format_flags)

    def test_write_outputs_includes_per_context_summary(self):
        rows = [
            {"context_index": 4, "condition": "full", "question_index": 0},
            {"context_index": 4, "condition": "drop-slot:0", "question_index": 0},
        ]
        summaries = {
            "full": {"condition": "full", "question_count": 1},
            "drop-slot:0": {"condition": "drop-slot:0", "question_count": 1},
        }
        replay = {"passed": True}
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            self.attribution._write_outputs(output_dir, rows, summaries, replay)
            per_context = json.loads(
                (output_dir / "attribution_per_context.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(per_context["context_index"], 4)
        self.assertEqual(per_context["question_count"], 1)
        self.assertEqual(per_context["conditions"], summaries)


if __name__ == "__main__":
    unittest.main()
