import argparse
import unittest

from scripts.eval import r4_triviaqa_dynamic_harness as harness


class FakeRetriever:
    def __init__(self, *, should_fail=False):
        self.should_fail = should_fail

    def batch_search(self, queries):
        if self.should_fail:
            raise RuntimeError("retrieval endpoint unavailable")
        return [f"Doc 1(Title: Paris) result for {queries[0]}"]


class R4TriviaQADynamicHarnessTest(unittest.TestCase):
    def test_strict_answer_parser_extracts_last_answer_tag(self):
        parsed = harness.parse_strict_answer(
            "<answer>London</answer>\n<answer>Paris</answer>"
        )
        self.assertEqual(parsed.answer, "Paris")
        self.assertTrue(parsed.parser_success)
        self.assertEqual(parsed.parser_mode, "strict_answer_tag")

    def test_strict_answer_parser_returns_none_without_tag(self):
        parsed = harness.parse_strict_answer("The answer is Paris.")
        self.assertIsNone(parsed.answer)
        self.assertFalse(parsed.parser_success)
        self.assertEqual(parsed.parser_mode, "none")

    def test_retrieval_accounting_records_success(self):
        accounting = harness.RetrievalAccounting(
            endpoint="http://127.0.0.1:8001/retrieve",
            topk=3,
        )
        wrapper = harness.AccountingRetriever(FakeRetriever(), accounting)

        result = wrapper.batch_search(["capital of france"])

        self.assertEqual(result, ["Doc 1(Title: Paris) result for capital of france"])
        self.assertEqual(accounting.call_count, 1)
        self.assertEqual(accounting.success_count, 1)
        self.assertEqual(accounting.failure_count, 0)
        self.assertFalse(accounting.saw_cannot_find_pages)

    def test_retrieval_accounting_records_exception(self):
        accounting = harness.RetrievalAccounting(
            endpoint="http://127.0.0.1:8001/retrieve",
            topk=3,
        )
        wrapper = harness.AccountingRetriever(
            FakeRetriever(should_fail=True),
            accounting,
        )

        with self.assertRaisesRegex(RuntimeError, "endpoint unavailable"):
            wrapper.batch_search(["capital of france"])

        self.assertEqual(accounting.call_count, 1)
        self.assertEqual(accounting.success_count, 0)
        self.assertEqual(accounting.failure_count, 1)
        self.assertEqual(accounting.failures[0]["type"], "RuntimeError")
        self.assertIn("endpoint unavailable", accounting.failures[0]["message"])

    def test_cannot_find_pages_sets_retrieval_flag(self):
        accounting = harness.RetrievalAccounting(
            endpoint="http://127.0.0.1:8001/retrieve",
            topk=3,
        )

        accounting.observe("Cannot find corresponding pages.")

        self.assertTrue(accounting.saw_cannot_find_pages)

    def test_structured_schema_contains_required_fields(self):
        parsed = harness.parse_strict_answer("<answer>Paris</answer>")
        retrieval = harness.RetrievalAccounting(
            endpoint="http://127.0.0.1:8001/retrieve",
            topk=3,
        )
        record = harness.build_sample_record(
            sample={"prompt": "What is the capital of France?", "answer": ["Paris"]},
            sample_index=0,
            sample_id="triviaqa-validation-0",
            conversation=[{"role": "assistant", "content": "<answer>Paris</answer>"}],
            final_response="<answer>Paris</answer>",
            parsed=parsed,
            reward=None,
            retrieval=retrieval,
            run=harness.RunMetadata(
                batch_size=1,
                memory_mode="disabled",
                memory_enabled=False,
                checkpoint_path="/tmp/checkpoint",
                config_overrides=["run.mode", "evaluate"],
                temperature=0.0,
                max_response_length=1024,
                seed=42,
            ),
            memory_bank_debug=None,
            valid_run=True,
            invalid_reason=None,
        )

        for field in (
            "phase",
            "sample_index",
            "sample_id",
            "question",
            "gold_answers",
            "conversation",
            "final_response",
            "parsed_answer",
            "parser_success",
            "retrieval",
            "run",
            "memory_bank_debug",
            "valid_run",
            "invalid_reason",
        ):
            self.assertIn(field, record)
        self.assertTrue(record["valid_run"])
        self.assertIsNone(record["invalid_reason"])

    def test_summary_counts_invalid_and_retrieval_blocked_runs(self):
        summary = harness.build_summary(
            [
                {"valid_run": False, "invalid_reason": "retrieval_endpoint_unavailable"},
                {"valid_run": True, "invalid_reason": None},
            ]
        )

        self.assertEqual(summary["summary"]["sample_count"], 2)
        self.assertEqual(summary["summary"]["valid_run_count"], 1)
        self.assertEqual(summary["summary"]["invalid_run_count"], 1)
        self.assertEqual(summary["summary"]["retrieval_blocked_count"], 1)

    def test_validate_args_rejects_batch_size_not_one(self):
        args = argparse.Namespace(
            batch_size=2,
            sample_count=1,
            sample_index=0,
            memory_mode="disabled",
            max_response_length=1024,
            retrieval_topk=3,
        )

        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            harness.validate_args(args)

    def test_memory_mode_choices_are_constrained(self):
        parser = harness.build_arg_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--cfg-path",
                    "configs/latent_memory/triviaqa.yaml",
                    "--checkpoint-path",
                    "/tmp/checkpoint",
                    "--output-dir",
                    "/tmp/output",
                    "--memory-mode",
                    "version_b",
                ]
            )


if __name__ == "__main__":
    unittest.main()
