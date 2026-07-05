import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval import locomo_qa_adapter as adapter
from scripts.eval import mab6b_weaver_space_bank_locomo_qa as harness


LOCOMO_PATH = Path("/mnt/18T/sunyanjia/AMEM/A-mem-main/data/locomo10.json")


class MAB6BWeaverSpaceBankLoCoMoQATest(unittest.TestCase):
    def _normalized_fixture(self):
        tempdir = tempfile.TemporaryDirectory()
        output_dir = Path(tempdir.name)
        adapter.write_smoke_subset(
            LOCOMO_PATH,
            output_dir,
            conversation_ids=["conv-26"],
            max_questions=5,
        )
        return tempdir, output_dir

    def test_load_normalized_records_and_select_one_conversation(self):
        tempdir, output_dir = self._normalized_fixture()
        self.addCleanup(tempdir.cleanup)
        conversations = harness.load_normalized_conversations(
            output_dir / "normalized_conversations.jsonl"
        )
        qa_rows = harness.load_normalized_qa_records(
            output_dir / "normalized_qa_records.jsonl"
        )
        conversation = harness.select_conversation(conversations, "conv-26")
        selected = harness.select_questions(qa_rows, conversation_id="conv-26", max_questions=2)

        self.assertEqual(len(conversations), 1)
        self.assertEqual(len(qa_rows), 5)
        self.assertEqual(conversation["conversation_id"], "conv-26")
        self.assertEqual([row["question_id"] for row in selected], ["conv-26::q000", "conv-26::q001"])

    def test_build_construction_chunk_prompts_preserves_order(self):
        conversation_row = {
            "conversation_id": "conv-26",
            "sessions": [
                {
                    "session_id": 1,
                    "timestamp": "2023-05-07",
                    "turns": [
                        {"speaker": "Caroline", "content": "Hello"},
                        {"speaker": "Mel", "content": "Hi"},
                    ],
                },
                {
                    "session_id": 2,
                    "timestamp": "2023-05-08",
                    "turns": [
                        {"speaker": "Caroline", "content": "Later"},
                    ],
                },
            ],
        }
        chunk_payload = harness.build_conversation_payload(
            conversation_row,
            chunk_size=8,
            token_counter=lambda text: len(text.split()),
        )

        self.assertGreaterEqual(len(chunk_payload["chunks"]), 1)
        self.assertEqual(len(chunk_payload["chunks"]), len(chunk_payload["memorization_prompts"]))
        self.assertIn("Session 1", chunk_payload["chunks"][0])
        self.assertIn("Caroline: Hello", chunk_payload["chunks"][0])
        self.assertIn("memorize", chunk_payload["memorization_prompts"][0].lower())

    def test_build_conversation_payload_session_granularity_uses_one_chunk_per_session(self):
        conversation_row = {
            "conversation_id": "conv-26",
            "sessions": [
                {
                    "session_id": 1,
                    "timestamp": "2023-05-07",
                    "turns": [
                        {"speaker": "Caroline", "content": "Hello"},
                        {"speaker": "Mel", "content": "Hi"},
                    ],
                },
                {
                    "session_id": 2,
                    "timestamp": "2023-05-08",
                    "turns": [
                        {"speaker": "Caroline", "content": "Later"},
                    ],
                },
            ],
        }
        payload = harness.build_conversation_payload(
            conversation_row,
            construction_granularity="session",
            token_counter=lambda text: len(text.split()),
        )

        self.assertEqual(payload["construction_granularity"], "session")
        self.assertEqual(payload["construction_chunk_unit"], "session")
        self.assertEqual(payload["construction_chunk_count"], 2)
        self.assertEqual(payload["construction_sessions_covered"], [1, 2])
        self.assertEqual(len(payload["chunks"]), 2)
        self.assertIn("[Session 1 | 2023-05-07]", payload["chunks"][0])
        self.assertIn("Caroline: Hello", payload["chunks"][0])
        self.assertIn("[Session 2 | 2023-05-08]", payload["chunks"][1])
        self.assertIn("Caroline: Later", payload["chunks"][1])

    def test_token_chunk_mode_remains_available(self):
        conversation_row = {
            "conversation_id": "conv-26",
            "sessions": [
                {"session_id": 1, "turns": [{"speaker": "A", "content": "one two"}]},
                {"session_id": 2, "turns": [{"speaker": "B", "content": "three four"}]},
            ],
        }
        payload = harness.build_conversation_payload(
            conversation_row,
            construction_granularity="token_chunk",
            chunk_size=5,
            token_counter=lambda text: len(text.split()),
        )

        self.assertEqual(payload["construction_granularity"], "token_chunk")
        self.assertEqual(payload["construction_chunk_unit"], "token_chunk")
        self.assertGreaterEqual(payload["construction_chunk_count"], 1)

    def test_session_mode_does_not_affect_question_selection(self):
        tempdir, output_dir = self._normalized_fixture()
        self.addCleanup(tempdir.cleanup)
        conversations = harness.load_normalized_conversations(
            output_dir / "normalized_conversations.jsonl"
        )
        qa_rows = harness.load_normalized_qa_records(
            output_dir / "normalized_qa_records.jsonl"
        )
        conversation = harness.select_conversation(conversations, "conv-26")
        selected = harness.select_questions(qa_rows, conversation_id="conv-26", max_questions=2)
        payload = harness.build_conversation_payload(
            conversation,
            construction_granularity="session",
        )

        self.assertEqual([row["question_id"] for row in selected], ["conv-26::q000", "conv-26::q001"])
        self.assertEqual(payload["construction_granularity"], "session")
        self.assertEqual(payload["construction_chunk_count"], conversation["session_count"])

    def test_max_questions_affects_qa_rows_only_not_session_construction(self):
        tempdir, output_dir = self._normalized_fixture()
        self.addCleanup(tempdir.cleanup)
        conversations = harness.load_normalized_conversations(
            output_dir / "normalized_conversations.jsonl"
        )
        qa_rows = harness.load_normalized_qa_records(
            output_dir / "normalized_qa_records.jsonl"
        )
        conversation = harness.select_conversation(conversations, "conv-26")
        selected = harness.select_questions(qa_rows, conversation_id="conv-26", max_questions=1)
        payload = harness.build_conversation_payload(
            conversation,
            construction_granularity="session",
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(payload["construction_chunk_count"], conversation["session_count"])

    def test_build_question_prompt_uses_question_text_only(self):
        conversation_row = {"conversation_id": "conv-26"}
        qa_row = {
            "question_id": "conv-26::q000",
            "category": 2,
            "category_name": "temporal",
            "question_text": "When did Caroline go?",
            "gold_answer": "7 May 2023",
        }
        payload = harness.build_question_payload(conversation_row, qa_row)

        self.assertEqual(payload["question_id"], "conv-26::q000")
        self.assertIn("When did Caroline go?", payload["query_prompt"])
        self.assertNotIn("7 May 2023", payload["query_prompt"])

    def test_prediction_record_contains_required_cost_and_score_fields(self):
        qa_row = {
            "question_id": "conv-26::q000",
            "category": 2,
            "category_name": "temporal",
            "question_text": "Q",
            "gold_answer": "A",
        }
        row = harness.build_prediction_record(
            mode="disabled",
            conversation_id="conv-26",
            qa_row=qa_row,
            prediction_text="A",
            raw_prediction_text="A",
            diagnostics={},
        )

        self.assertEqual(row["question_id"], "conv-26::q000")
        self.assertEqual(row["mode"], "disabled")
        self.assertIn("query_write_count", row)
        self.assertIn("latency_seconds", row)
        self.assertEqual(row["construction_write_count"], 0)
        self.assertIn("prompt_leak", row)
        self.assertIn("question_restatement", row)
        self.assertIn("no_context_denial", row)
        self.assertIn("refusal", row)
        self.assertIn("meta_reasoning_or_search", row)
        self.assertIn("answer_extraction_failed", row)
        self.assertIn("construction_granularity", row)
        self.assertIn("construction_chunk_count", row)
        self.assertIn("construction_chunk_unit", row)

    def test_extract_prediction_contract_after_last_answer_marker(self):
        contract = harness.extract_prediction_contract(
            "Preface\nAnswer: wrong\nAnswer: Paris\nQuestion: ignored",
            question_text="What is the destination?",
        )

        self.assertEqual(contract["raw_prediction_text"], "Preface\nAnswer: wrong\nAnswer: Paris\nQuestion: ignored")
        self.assertEqual(contract["prediction_text"], "Paris")
        self.assertFalse(contract["answer_extraction_failed"])

    def test_extract_prediction_contract_first_line_fallback_without_answer_marker(self):
        contract = harness.extract_prediction_contract(
            "  \nThe answer is tomorrow.\nSecond line should be ignored.",
            question_text="When is it?",
        )

        self.assertEqual(contract["prediction_text"], "The answer is tomorrow.")
        self.assertFalse(contract["answer_extraction_failed"])

    def test_extract_prediction_contract_strips_special_tokens(self):
        contract = harness.extract_prediction_contract(
            "<|assistant|>\nAnswer: <s>Paris</s>",
            question_text="Where did they go?",
        )

        self.assertEqual(contract["prediction_text"], "Paris")
        self.assertTrue(contract["prompt_leak"])

    def test_extract_prediction_contract_flags_prompt_leak(self):
        contract = harness.extract_prediction_contract(
            "Question: When did Caroline go to the support group?",
            question_text="When did Caroline go to the support group?",
        )

        self.assertTrue(contract["prompt_leak"])
        self.assertTrue(contract["question_restatement"])
        self.assertTrue(contract["answer_extraction_failed"])

    def test_extract_prediction_contract_flags_no_context_denial(self):
        contract = harness.extract_prediction_contract(
            "question\nI'm sorry, but I cannot provide an answer because there is no conversation history or context provided.",
            question_text="What did Caroline do?",
        )

        self.assertTrue(contract["prompt_leak"])
        self.assertTrue(contract["no_context_denial"])
        self.assertTrue(contract["refusal"])
        self.assertEqual(contract["prediction_text"], "I'm sorry, but I cannot provide an answer because there is no conversation history or context provided.")

    def test_extract_prediction_contract_flags_refusal(self):
        contract = harness.extract_prediction_contract(
            "I'm sorry, but I can't help with that request.",
            question_text="What happened?",
        )

        self.assertTrue(contract["refusal"])
        self.assertFalse(contract["answer_extraction_failed"])

    def test_extract_prediction_contract_flags_meta_reasoning(self):
        contract = harness.extract_prediction_contract(
            "<think> I need to search for the answer to this question.",
            question_text="What was Jon's hobby?",
        )

        self.assertTrue(contract["meta_reasoning_or_search"])
        self.assertTrue(contract["prompt_leak"])

    def test_extract_prediction_contract_flags_extraction_failure(self):
        contract = harness.extract_prediction_contract(
            "<|assistant|>\nQuestion:",
            question_text="What happened?",
        )

        self.assertEqual(contract["prediction_text"], "")
        self.assertTrue(contract["answer_extraction_failed"])

    def test_assert_zero_query_writes_raises_when_nonzero(self):
        with self.assertRaises(RuntimeError):
            harness.assert_zero_query_writes({"query_write_count": 1})

    def test_default_diagnostics_include_construction_granularity_fields(self):
        diagnostics = harness.default_diagnostics_for_mode("p7")

        self.assertIn("construction_granularity", diagnostics)
        self.assertIn("construction_chunk_count", diagnostics)
        self.assertIn("construction_chunk_unit", diagnostics)
        self.assertIn("construction_sessions_covered", diagnostics)

    def test_score_integration_returns_scored_rows_and_aggregate_metrics(self):
        qa_rows = [{
            "question_id": "conv-26::q000",
            "conversation_id": "conv-26",
            "category": 2,
            "category_name": "temporal",
            "question_text": "Q",
            "gold_answer": "A",
        }]
        prediction_rows = [{
            "question_id": "conv-26::q000",
            "conversation_id": "conv-26",
            "method": "disabled",
            "prediction_text": "A",
            "prediction_status": "ok",
        }]
        scored_rows, aggregate = harness.score_predictions(qa_rows, prediction_rows)

        self.assertEqual(scored_rows[0]["exact_match"], 1)
        self.assertIn("overall_micro", aggregate)


if __name__ == "__main__":
    unittest.main()
