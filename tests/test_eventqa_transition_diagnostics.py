import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.eval.eventqa_transition_diagnostics import (
    aggregate_transition_rows,
    build_transition_diagnostic,
    compare_eventqa_records,
    contains_chinese,
    load_eventqa_records,
)
from scripts.eval.eventqa_compare_runs import build_parser, main as compare_main


def _row(
    *,
    context_index=0,
    query_id=0,
    qa_pair_id="pair-0",
    off_raw="wrong",
    on_raw="wrong",
    off_parsed=None,
    on_parsed=None,
    off_em=0,
    on_em=0,
    off_recall=0.0,
    on_recall=0.0,
    off_flags=None,
    on_flags=None,
):
    return {
        "context_index": context_index,
        "context_id": f"ctx-{context_index}",
        "query_id": query_id,
        "question_id": f"question-{query_id}",
        "qa_pair_id": qa_pair_id,
        "question": f"Question {query_id}?",
        "gold_answers": ["gold answer"],
        "bank_off_rendered_query_prompt": f"off prompt {query_id}",
        "bank_on_rendered_query_prompt": f"on prompt {query_id}",
        "bank_off_prediction": off_raw,
        "bank_on_prediction": on_raw,
        "bank_off_parsed_prediction": off_raw if off_parsed is None else off_parsed,
        "bank_on_parsed_prediction": on_raw if on_parsed is None else on_parsed,
        "bank_off_substring_exact_match": off_em,
        "bank_on_substring_exact_match": on_em,
        "bank_off_eventqa_recall": off_recall,
        "bank_on_eventqa_recall": on_recall,
        "bank_off_format_flags": off_flags or {
            "empty_output": False,
            "contains_json_brace": False,
            "contains_answer_prefix": False,
            "multiline_output": False,
            "verbose_output": False,
        },
        "bank_on_format_flags": on_flags or {
            "empty_output": False,
            "contains_json_brace": False,
            "contains_answer_prefix": False,
            "multiline_output": False,
            "verbose_output": False,
        },
    }


class EventQATransitionDiagnosticsTest(unittest.TestCase):
    def test_helpful_memory(self):
        result = build_transition_diagnostic(
            _row(on_raw="gold answer", on_em=1, on_recall=1.0)
        )
        self.assertTrue(result["helpful_memory"])
        self.assertFalse(result["harmful_memory"])

    def test_harmful_memory(self):
        result = build_transition_diagnostic(
            _row(off_raw="gold answer", off_em=1, off_recall=1.0)
        )
        self.assertTrue(result["harmful_memory"])

    def test_unchanged_correct_and_wrong(self):
        correct = build_transition_diagnostic(
            _row(
                off_raw="gold answer",
                on_raw="gold answer",
                off_em=1,
                on_em=1,
                off_recall=1.0,
                on_recall=1.0,
            )
        )
        wrong = build_transition_diagnostic(_row())
        self.assertTrue(correct["unchanged_correct"])
        self.assertTrue(wrong["unchanged_wrong"])

    def test_recall_gain_and_loss(self):
        gain = build_transition_diagnostic(_row(on_raw="contains gold answer"))
        loss = build_transition_diagnostic(_row(off_raw="contains gold answer"))
        self.assertTrue(gain["recall_gain"])
        self.assertTrue(loss["recall_loss"])

    def test_format_harm_when_raw_contains_gold_but_em_is_false(self):
        result = build_transition_diagnostic(
            _row(
                on_raw="Answer: gold answer\nextra text",
                on_parsed="extra text",
                on_em=0,
                on_recall=1.0,
                on_flags={
                    "empty_output": False,
                    "contains_json_brace": False,
                    "contains_answer_prefix": True,
                    "multiline_output": True,
                    "verbose_output": False,
                },
            )
        )
        self.assertTrue(result["format_harm"])
        self.assertTrue(result["bank_on_recall_positive_em_negative"])

    def test_chinese_output_detection(self):
        self.assertTrue(contains_chinese("答案是 gold answer"))
        self.assertFalse(contains_chinese("the answer is gold"))

    def test_aggregate_transition_counts(self):
        diagnostics = [
            build_transition_diagnostic(_row(query_id=0, on_raw="gold answer", on_em=1)),
            build_transition_diagnostic(_row(query_id=1, off_raw="gold answer", off_em=1)),
            build_transition_diagnostic(
                _row(query_id=2, off_raw="gold answer", on_raw="gold answer", off_em=1, on_em=1)
            ),
            build_transition_diagnostic(_row(query_id=3)),
        ]
        aggregate = aggregate_transition_rows(diagnostics)
        self.assertEqual(aggregate["global"]["helpful_memory_count"], 1)
        self.assertEqual(aggregate["global"]["harmful_memory_count"], 1)
        self.assertEqual(aggregate["global"]["unchanged_correct_count"], 1)
        self.assertEqual(aggregate["global"]["unchanged_wrong_count"], 1)
        self.assertEqual(aggregate["global"]["recall_gain_count"], 1)
        self.assertEqual(aggregate["global"]["recall_loss_count"], 1)
        self.assertIn("0", aggregate["per_context"])

    def test_cross_run_equality_counts(self):
        left = [_row(query_id=0), _row(query_id=1, qa_pair_id="pair-1")]
        right = [dict(left[0]), dict(left[1])]
        right[1]["bank_on_prediction"] = "changed"
        right[1]["bank_on_parsed_prediction"] = "changed"
        right[1]["bank_on_substring_exact_match"] = 1

        comparison = compare_eventqa_records(left, right)

        self.assertTrue(comparison["data_identity"]["all_identity_fields_equal"])
        self.assertEqual(comparison["bank_off_stability"]["raw_prediction_equal_count"], 2)
        self.assertEqual(comparison["bank_off_stability"]["em_changed_count"], 0)
        self.assertEqual(comparison["bank_on_stability"]["raw_prediction_equal_count"], 1)
        self.assertEqual(comparison["bank_on_stability"]["em_changed_count"], 1)
        self.assertEqual(
            comparison["per_context"]["0"]["bank_on_stability"][
                "raw_prediction_equal_count"
            ],
            1,
        )
        self.assertEqual(comparison["conclusion"], "instability_localized_to_bank_on")

    def test_multi_root_loading_merges_contexts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            roots = []
            for context_index in (0, 1):
                parent = Path(tmpdir) / f"ctx{context_index}"
                run = parent / "20260101T000000Z-run"
                run.mkdir(parents=True)
                row = _row(context_index=context_index)
                (run / "eventqa_per_question.jsonl").write_text(
                    json.dumps(row) + "\n", encoding="utf-8"
                )
                roots.append(parent)

            records, resolved = load_eventqa_records(roots)

        self.assertEqual(len(records), 2)
        self.assertEqual([row["context_index"] for row in records], [0, 1])
        self.assertEqual(len(resolved), 2)

    def test_missing_optional_field_is_reported(self):
        left = [_row()]
        right = [_row()]
        del right[0]["bank_off_parsed_prediction"]

        comparison = compare_eventqa_records(left, right)

        self.assertIn(
            "bank_off_parsed_prediction",
            comparison["missing_fields"]["right"],
        )

    def test_compare_cli_accepts_multiple_roots_and_writes_json(self):
        parser = build_parser()
        args = parser.parse_args(
            ["--left", "left0", "left1", "--right", "right", "--output", "out.json"]
        )
        self.assertEqual(args.left, ["left0", "left1"])

        with tempfile.TemporaryDirectory() as tmpdir:
            left = Path(tmpdir) / "left"
            right = Path(tmpdir) / "right"
            left.mkdir()
            right.mkdir()
            row = _row()
            for root in (left, right):
                (root / "eventqa_per_question.jsonl").write_text(
                    json.dumps(row) + "\n", encoding="utf-8"
                )
            output = Path(tmpdir) / "comparison.json"

            status = compare_main(
                [
                    "--left",
                    str(left),
                    "--right",
                    str(right),
                    "--output",
                    str(output),
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(payload["comparison"]["record_count_left"], 1)
        self.assertEqual(payload["comparison"]["conclusion"], "no_bank_on_raw_instability_detected")

    def test_compare_cli_runs_directly_from_repo_root(self):
        result = subprocess.run(
            [sys.executable, "scripts/eval/eventqa_compare_runs.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--left", result.stdout)


if __name__ == "__main__":
    unittest.main()
