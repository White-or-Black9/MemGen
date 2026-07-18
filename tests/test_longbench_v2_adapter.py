import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval import longbench_v2_adapter as adapter


def source_row(item_id="item-1"):
    return {
        "_id": item_id,
        "domain": "Multi-Document QA",
        "sub_domain": "Academic",
        "difficulty": "easy",
        "length": "short",
        "question": "Which option is correct?",
        "choice_A": "Alpha",
        "choice_B": "Beta",
        "choice_C": "Gamma",
        "choice_D": "Delta",
        "answer": "A",
        "context": "First sentence. Second sentence.\nA table row without punctuation\n",
    }


def manifest_item(item_id="item-1"):
    return {
        "_id": item_id,
        "domain": "Multi-Document QA",
        "sub_domain": "Academic",
        "difficulty": "easy",
        "length": "short",
        "answer": "A",
        "context_word_count": 9,
        "context_token_count": 12,
        "rendered_chat_token_count": 100,
        "capacity_class": "window_fit",
    }


def manifest(item=None):
    item = item or manifest_item()
    return {
        "schema_version": 1,
        "name": "fixture",
        "dataset": adapter.DATASET_ID,
        "revision": adapter.DATASET_REVISION,
        "rendered_chat_token_cap": 262144,
        "count": 1,
        "items": [item],
    }


class LongBenchV2AdapterTest(unittest.TestCase):
    def test_dataset_contract_rejects_duplicate_ids_and_bad_answers(self):
        with self.assertRaisesRegex(adapter.LongBenchV2ContractError, "duplicate"):
            adapter.validate_dataset_rows([source_row(), source_row()])
        bad = source_row()
        bad["answer"] = "E"
        with self.assertRaisesRegex(adapter.LongBenchV2ContractError, "invalid answer"):
            adapter.validate_dataset_rows([bad])

    def test_manifest_locks_dataset_revision_count_and_token_cap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            path.write_text(json.dumps(manifest()), encoding="utf-8")
            loaded = adapter.load_manifest(path)
            self.assertEqual(loaded["revision"], adapter.DATASET_REVISION)

            invalid = manifest()
            invalid["revision"] = "wrong"
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(adapter.LongBenchV2ContractError, "revision"):
                adapter.load_manifest(path)

    def test_select_manifest_rows_rejects_metadata_drift(self):
        item = manifest_item()
        item["answer"] = "B"
        with self.assertRaisesRegex(adapter.LongBenchV2ContractError, "metadata mismatch"):
            adapter.select_manifest_rows([source_row()], manifest(item))

    def test_normalized_item_and_prompt_preserve_choices_and_context(self):
        item = adapter.select_manifest_rows([source_row()], manifest())[0]
        self.assertEqual(item["gold_choice"], "A")
        self.assertEqual(item["choices"]["D"], "Delta")
        prompt = adapter.render_prompt(item)
        self.assertIn(item["context"], prompt)
        self.assertIn("(A) Alpha", prompt)
        self.assertIn('The correct answer is', prompt)
        memory_prompt = adapter.render_memory_query_prompt(item)
        self.assertNotIn(item["context"], memory_prompt)
        self.assertIn("Based on the context you memorized", memory_prompt)

    def test_memorization_prompt_records_chunk_order(self):
        prompt = adapter.render_memorization_prompt("chunk text", chunk_index=1, chunk_count=3)
        self.assertIn("chunk 2 of 3", prompt)
        self.assertIn("chunk text", prompt)

    def test_chunk_text_is_exactly_reconstructable_and_budget_bounded(self):
        text = "First sentence. Second sentence is longer.\nrow one row two row three\n"
        chunks = adapter.chunk_text(text, token_count=lambda value: len(value), token_budget=18)
        self.assertEqual("".join(chunk["text"] for chunk in chunks), text)
        self.assertTrue(all(chunk["token_count"] <= 18 for chunk in chunks))
        self.assertEqual([chunk["chunk_index"] for chunk in chunks], list(range(len(chunks))))
        self.assertEqual(chunks[0]["start_char"], 0)
        self.assertEqual(chunks[-1]["end_char"], len(text))

    def test_repository_smoke_manifest_passes_static_contract(self):
        loaded = adapter.load_manifest("configs/eval/longbench_v2_p7_smoke_ids.json")
        self.assertEqual(loaded["count"], 18)
        self.assertEqual(len({item["_id"] for item in loaded["items"]}), 18)


if __name__ == "__main__":
    unittest.main()
