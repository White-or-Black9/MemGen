import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval import locomo_qa_adapter as adapter


LOCOMO_PATH = Path("/mnt/18T/sunyanjia/AMEM/A-mem-main/data/locomo10.json")


class LoCoMoQAAdapterTest(unittest.TestCase):
    def test_loads_local_locomo10_and_reports_nonzero_counts(self):
        summary = adapter.inspect_dataset(LOCOMO_PATH)
        self.assertEqual(summary["conversation_count"], 10)
        self.assertGreaterEqual(summary["qa_count"], 5)
        self.assertIn("category_counts", summary)

    def test_extract_conversation_preserves_session_order_and_turn_ids(self):
        conversations, qa_rows = adapter.extract_records(
            LOCOMO_PATH,
            conversation_ids=["conv-26"],
            max_questions=5,
        )
        self.assertEqual(len(conversations), 1)
        self.assertEqual(conversations[0]["conversation_id"], "conv-26")
        self.assertEqual(conversations[0]["session_order"][0], 1)
        self.assertEqual(conversations[0]["sessions"][0]["turns"][0]["turn_id"], "D1:1")
        self.assertEqual(len(qa_rows), 5)

    def test_extract_qa_records_derives_stable_question_ids_and_categories(self):
        _, qa_rows = adapter.extract_records(
            LOCOMO_PATH,
            conversation_ids=["conv-26"],
            max_questions=5,
        )
        self.assertEqual(qa_rows[0]["question_id"], "conv-26::q000")
        self.assertEqual(qa_rows[1]["question_id"], "conv-26::q001")
        self.assertTrue(qa_rows[0]["reference_answers"])
        self.assertIn(
            qa_rows[0]["category_name"],
            {"multi_hop", "temporal", "open_domain", "single_hop", "adversarial"},
        )

    def test_extract_records_preserves_reference_answers_and_evidence_metadata(self):
        _, qa_rows = adapter.extract_records(
            LOCOMO_PATH,
            conversation_ids=["conv-26"],
            max_questions=5,
        )
        first = qa_rows[0]
        self.assertEqual(first["gold_answer"], "7 May 2023")
        self.assertEqual(first["reference_answers"], ["7 May 2023"])
        self.assertEqual(first["evidence"], ["D1:3"])
        self.assertEqual(first["evidence_turn_ids"], ["D1:3"])
        self.assertEqual(first["evidence_session_ids"], [1])

    def test_write_smoke_subset_outputs_expected_jsonl_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            output_paths = adapter.write_smoke_subset(
                LOCOMO_PATH,
                output_dir,
                conversation_ids=["conv-26"],
                max_questions=5,
            )
            self.assertTrue(output_paths["conversations_path"].exists())
            self.assertTrue(output_paths["qa_records_path"].exists())
            self.assertTrue(output_paths["summary_path"].exists())

            conversations = [
                json.loads(line)
                for line in output_paths["conversations_path"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            qa_rows = [
                json.loads(line)
                for line in output_paths["qa_records_path"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(conversations), 1)
            self.assertEqual(len(qa_rows), 5)
            self.assertEqual(conversations[0]["conversation_id"], "conv-26")
            self.assertEqual(qa_rows[0]["question_id"], "conv-26::q000")


if __name__ == "__main__":
    unittest.main()
