import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.eval.ruler_qa2_p7 import (
    RulerQA2RunContractError,
    build_dry_run_artifact,
    build_disabled_query_scored_artifact,
    build_failure_artifact,
    build_mab2_disabled_args,
    build_query_jobs,
    build_ruler_context_payload,
    build_ruler_question_payload,
    build_single_query_payload,
    disabled_expected_turns,
    execute_query_jobs,
    expected_method_set,
    load_prepared_payload,
    make_mab2_disabled_query_runner,
    make_mode_stub_records,
    normalize_text,
    run_adapted_p7_queries,
    run_disabled_query_predictions,
    slice_prepared_payload,
    score_predictions_for_method,
    substring_exact_match,
    validate_prepared_payload,
    validate_same_queries,
)


class RulerQA2RunnerTest(unittest.TestCase):
    def test_normalize_text_lowercases_and_collapses_whitespace(self):
        self.assertEqual(normalize_text("  Alpha   Beta  "), "alpha beta")

    def test_substring_exact_match_accepts_gold_inside_prediction(self):
        self.assertTrue(substring_exact_match("The answer is alpha-17.", ["alpha-17"]))

    def test_substring_exact_match_rejects_missing_gold(self):
        self.assertFalse(substring_exact_match("The answer is alpha-18.", ["alpha-17"]))

    def test_validate_same_queries_rejects_mismatched_ids(self):
        disabled = [{"query_id": "q1"}, {"query_id": "q2"}]
        enabled = [{"query_id": "q1"}, {"query_id": "q3"}]
        with self.assertRaisesRegex(RulerQA2RunContractError, "query identity mismatch"):
            validate_same_queries(disabled, enabled)

    def test_validate_same_queries_accepts_identical_ids(self):
        reference = [{"query_id": "q1"}, {"query_id": "q2"}]
        candidate = [{"query_id": "q1"}, {"query_id": "q2"}]
        validate_same_queries(reference, candidate)

    def test_validate_prepared_payload_checks_benchmark_shape(self):
        payload = {
            "dataset_config": {"sub_dataset": "ruler_qa2_421K"},
            "context_id": "ctx-0",
            "chunks": ["chunk-1"],
            "memorization_prompts": ["mem-1"],
            "queries": [
                {
                    "query_id": 0,
                    "question": "q1",
                    "query_prompt": "QUERY::q1",
                    "gold_answers": ["a1"],
                },
                {
                    "query_id": 1,
                    "question": "q2",
                    "query_prompt": "QUERY::q2",
                    "gold_answers": ["a2"],
                },
            ],
            "question_count": 2,
        }
        summary = validate_prepared_payload(payload, expected_sub_dataset="ruler_qa2_421K")
        self.assertEqual(summary["context_id"], "ctx-0")
        self.assertEqual(summary["question_count"], 2)
        self.assertEqual(summary["chunk_count"], 1)

    def test_validate_prepared_payload_rejects_question_count_mismatch(self):
        payload = {
            "dataset_config": {"sub_dataset": "ruler_qa2_421K"},
            "context_id": "ctx-0",
            "chunks": ["chunk-1"],
            "memorization_prompts": ["mem-1"],
            "queries": [
                {
                    "query_id": 0,
                    "question": "q1",
                    "query_prompt": "QUERY::q1",
                    "gold_answers": ["a1"],
                }
            ],
            "question_count": 2,
        }
        with self.assertRaisesRegex(RulerQA2RunContractError, "question_count mismatch"):
            validate_prepared_payload(payload, expected_sub_dataset="ruler_qa2_421K")

    def test_build_dry_run_artifact_captures_mode_order(self):
        prepared_summary = {
            "context_id": "ctx-0",
            "question_count": 100,
            "chunk_count": 12,
            "sub_dataset": "ruler_qa2_421K",
        }
        artifact = build_dry_run_artifact(
            prepared_summary,
            methods=["disabled", "p7", "p7_no_query_retrieval"],
        )
        self.assertEqual(artifact["schema_version"], "ruler-qa2-dryrun/v1")
        self.assertEqual(artifact["benchmark"], "ruler_qa2_421K")
        self.assertEqual(
            artifact["methods"], ["disabled", "p7", "p7_no_query_retrieval"]
        )
        self.assertEqual(artifact["question_count"], 100)

    def test_expected_method_set_rejects_unknown_or_duplicate_methods(self):
        self.assertEqual(
            expected_method_set("disabled,p7,p7_no_query_retrieval"),
            ["disabled", "p7", "p7_no_query_retrieval"],
        )
        with self.assertRaisesRegex(RulerQA2RunContractError, "duplicate"):
            expected_method_set("disabled,p7,disabled")
        with self.assertRaisesRegex(RulerQA2RunContractError, "unknown"):
            expected_method_set("disabled,p8")

    def test_load_prepared_payload_reads_and_validates_json(self):
        payload = {
            "dataset_config": {"sub_dataset": "ruler_qa2_421K"},
            "context_id": "ctx-0",
            "chunks": ["chunk-1"],
            "memorization_prompts": ["mem-1"],
            "queries": [
                {
                    "query_id": 0,
                    "question": "q1",
                    "query_prompt": "QUERY::q1",
                    "gold_answers": ["a1"],
                }
            ],
            "question_count": 1,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prepared.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded, summary = load_prepared_payload(
                path, expected_sub_dataset="ruler_qa2_421K"
            )
        self.assertEqual(loaded["context_id"], "ctx-0")
        self.assertEqual(summary["question_count"], 1)

    def test_make_mode_stub_records_keeps_identical_query_ids(self):
        payload = {
            "context_id": "ctx-0",
            "queries": [
                {"query_id": 0, "question": "q1", "gold_answers": ["a1"]},
                {"query_id": 1, "question": "q2", "gold_answers": ["a2"]},
            ],
        }
        methods = ["disabled", "p7", "p7_no_query_retrieval"]
        records = make_mode_stub_records(payload, methods=methods)
        self.assertEqual(sorted(records), methods)
        validate_same_queries(records["disabled"], records["p7"])
        validate_same_queries(records["disabled"], records["p7_no_query_retrieval"])
        self.assertEqual(records["p7"][0]["status"], "not_run")
        self.assertEqual(records["p7"][1]["gold_answers"], ["a2"])

    def test_slice_prepared_payload_truncates_queries_and_updates_count(self):
        payload = {
            "dataset_config": {"sub_dataset": "ruler_qa2_421K"},
            "context_id": "ctx-0",
            "chunks": ["chunk-1"],
            "memorization_prompts": ["mem-1"],
            "queries": [
                {"query_id": 0, "question": "q1", "query_prompt": "Q1", "gold_answers": ["a1"]},
                {"query_id": 1, "question": "q2", "query_prompt": "Q2", "gold_answers": ["a2"]},
                {"query_id": 2, "question": "q3", "query_prompt": "Q3", "gold_answers": ["a3"]},
            ],
            "question_count": 3,
        }
        sliced = slice_prepared_payload(payload, max_queries=2)
        self.assertEqual(sliced["question_count"], 2)
        self.assertEqual([item["query_id"] for item in sliced["queries"]], [0, 1])

    def test_score_predictions_for_method_marks_correctness(self):
        payload = {
            "context_id": "ctx-0",
            "queries": [
                {
                    "query_id": 0,
                    "question": "q1",
                    "gold_answers": ["alpha-17"],
                },
                {
                    "query_id": 1,
                    "question": "q2",
                    "gold_answers": ["yes"],
                },
            ],
        }
        predictions = [
            {"query_id": 0, "prediction": "The answer is alpha-17."},
            {"query_id": 1, "prediction": "no"},
        ]
        records = score_predictions_for_method(
            payload, method="disabled", predictions=predictions
        )
        self.assertTrue(records[0]["correct"])
        self.assertFalse(records[1]["correct"])
        self.assertEqual(records[0]["status"], "scored")

    def test_score_predictions_for_method_rejects_query_mismatch(self):
        payload = {
            "context_id": "ctx-0",
            "queries": [
                {"query_id": 0, "question": "q1", "gold_answers": ["a1"]},
                {"query_id": 1, "question": "q2", "gold_answers": ["a2"]},
            ],
        }
        predictions = [{"query_id": 0, "prediction": "a1"}]
        with self.assertRaisesRegex(RulerQA2RunContractError, "query identity mismatch"):
            score_predictions_for_method(payload, method="disabled", predictions=predictions)

    def test_build_query_jobs_preserves_query_order_and_prompt(self):
        payload = {
            "context_id": "ctx-0",
            "queries": [
                {"query_id": 0, "question": "q1", "query_prompt": "QUERY::q1", "gold_answers": ["a1"]},
                {"query_id": 1, "question": "q2", "query_prompt": "QUERY::q2", "gold_answers": ["a2"]},
            ],
        }
        jobs = build_query_jobs(payload, method="disabled")
        self.assertEqual([job["query_id"] for job in jobs], [0, 1])
        self.assertEqual(jobs[0]["method"], "disabled")
        self.assertEqual(jobs[1]["query_prompt"], "QUERY::q2")

    def test_execute_query_jobs_uses_callable_executor(self):
        payload = {
            "context_id": "ctx-0",
            "queries": [
                {"query_id": 0, "question": "q1", "query_prompt": "QUERY::q1", "gold_answers": ["a1"]},
                {"query_id": 1, "question": "q2", "query_prompt": "QUERY::q2", "gold_answers": ["a2"]},
            ],
        }
        jobs = build_query_jobs(payload, method="disabled")

        def executor(job):
            return {"query_id": job["query_id"], "prediction": f"PRED::{job['query_id']}"}

        predictions = execute_query_jobs(jobs, executor=executor)
        self.assertEqual(
            predictions,
            [
                {"query_id": 0, "prediction": "PRED::0"},
                {"query_id": 1, "prediction": "PRED::1"},
            ],
        )

    def test_build_single_query_payload_preserves_context_and_one_query(self):
        payload = {
            "context_id": "ctx-0",
            "dataset_config": {"sub_dataset": "ruler_qa2_421K"},
            "chunks": ["chunk-1", "chunk-2"],
            "chunk_token_lengths": [10, 11],
            "memorization_prompts": ["mem-1", "mem-2"],
            "queries": [
                {
                    "query_id": 0,
                    "question": "q1",
                    "query_prompt": "QUERY::q1",
                    "gold_answers": ["a1"],
                },
                {
                    "query_id": 1,
                    "question": "q2",
                    "query_prompt": "QUERY::q2",
                    "gold_answers": ["a2"],
                },
            ],
        }
        single = build_single_query_payload(payload, query_id=1)
        self.assertEqual(single["context_id"], "ctx-0")
        self.assertEqual(single["query_id"], 1)
        self.assertEqual(single["query_prompt"], "QUERY::q2")
        self.assertEqual(single["gold_answers"], ["a2"])
        self.assertEqual(single["chunks"], ["chunk-1", "chunk-2"])

    def test_run_disabled_query_predictions_uses_query_runner(self):
        payload = {
            "context_id": "ctx-0",
            "dataset_config": {"sub_dataset": "ruler_qa2_421K"},
            "chunks": ["chunk-1"],
            "chunk_token_lengths": [10],
            "memorization_prompts": ["mem-1"],
            "queries": [
                {
                    "query_id": 0,
                    "question": "q1",
                    "query_prompt": "QUERY::q1",
                    "gold_answers": ["a1"],
                },
                {
                    "query_id": 1,
                    "question": "q2",
                    "query_prompt": "QUERY::q2",
                    "gold_answers": ["a2"],
                },
            ],
        }

        def query_runner(single_payload):
            return {"prediction": f"OUT::{single_payload['query_id']}"}

        predictions = run_disabled_query_predictions(payload, query_runner=query_runner)
        self.assertEqual(
            predictions,
            [
                {"query_id": 0, "prediction": "OUT::0"},
                {"query_id": 1, "prediction": "OUT::1"},
            ],
        )

    def test_make_mab2_disabled_query_runner_delegates_to_model_runner(self):
        calls = []

        def fake_model_runner(args, single_payload):
            calls.append((args, single_payload["query_id"], single_payload["query_prompt"]))
            return {"prediction": f"MODEL::{single_payload['query_id']}"}

        runner = make_mab2_disabled_query_runner(
            SimpleNamespace(seed=42),
            model_runner=fake_model_runner,
        )
        result = runner(
            {
                "context_id": "ctx-0",
                "query_id": 3,
                "query_prompt": "QUERY::q3",
            }
        )
        self.assertEqual(result["prediction"], "MODEL::3")
        self.assertEqual(calls, [(SimpleNamespace(seed=42), 3, "QUERY::q3")])

    def test_build_mab2_disabled_args_sets_required_fields(self):
        args = build_mab2_disabled_args(
            seed=42,
            model_path="model-path",
            checkpoint_path="checkpoint-path",
            cfg_path="cfg-path",
        )
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.model_path, "model-path")
        self.assertEqual(args.checkpoint_path, "checkpoint-path")
        self.assertEqual(args.cfg_path, "cfg-path")

    def test_disabled_expected_turns_is_chunk_count_plus_query(self):
        payload = {"chunks": ["c1", "c2", "c3"]}
        self.assertEqual(disabled_expected_turns(payload), 4)

    def test_build_disabled_query_scored_artifact_runs_fake_model_runner(self):
        payload = {
            "context_id": "ctx-0",
            "dataset_config": {"sub_dataset": "ruler_qa2_421K"},
            "chunks": ["chunk-1"],
            "chunk_token_lengths": [10],
            "memorization_prompts": ["mem-1"],
            "queries": [
                {
                    "query_id": 0,
                    "question": "q1",
                    "query_prompt": "QUERY::q1",
                    "gold_answers": ["yes"],
                },
                {
                    "query_id": 1,
                    "question": "q2",
                    "query_prompt": "QUERY::q2",
                    "gold_answers": ["no"],
                },
            ],
            "question_count": 2,
        }
        summary = {
            "context_id": "ctx-0",
            "sub_dataset": "ruler_qa2_421K",
            "question_count": 2,
            "chunk_count": 1,
        }

        artifact = build_disabled_query_scored_artifact(
            payload,
            prepared_summary=summary,
            runner_args=SimpleNamespace(seed=42),
            model_runner=lambda _args, single_payload: {
                "prediction": "yes" if single_payload["query_id"] == 0 else "maybe"
            },
        )
        records = artifact["mode_records"]["disabled"]
        self.assertTrue(records[0]["correct"])
        self.assertFalse(records[1]["correct"])
        self.assertEqual(artifact["methods"], ["disabled"])

    def test_build_failure_artifact_records_error_string(self):
        summary = {
            "context_id": "ctx-0",
            "sub_dataset": "ruler_qa2_421K",
            "question_count": 1,
            "chunk_count": 10,
        }
        artifact = build_failure_artifact(
            summary,
            methods=["disabled"],
            error=RuntimeError("Rendered history exceeds capacity"),
        )
        self.assertEqual(artifact["status"], "failed")
        self.assertIn("Rendered history exceeds capacity", artifact["error"])

    def test_build_ruler_context_and_question_payloads_match_eventqa_like_contract(self):
        prepared = {
            "dataset_config": {"sub_dataset": "ruler_qa2_421K"},
            "context_id": "ctx-0",
            "chunks": ["chunk-1", "chunk-2"],
            "chunk_token_lengths": [10, 11],
            "memorization_prompts": ["mem-1", "mem-2"],
            "queries": [
                {
                    "query_id": 0,
                    "question": "q1",
                    "query_prompt": "QUERY::q1",
                    "gold_answers": ["a1"],
                },
                {
                    "query_id": 1,
                    "question": "q2",
                    "query_prompt": "QUERY::q2",
                    "gold_answers": ["a2"],
                },
            ],
            "question_count": 2,
        }
        context_payload = build_ruler_context_payload(prepared)
        question_payload = build_ruler_question_payload(context_payload, 1)
        self.assertEqual(context_payload["question_count"], 2)
        self.assertEqual(context_payload["questions"], ["q1", "q2"])
        self.assertEqual(question_payload["query_id"], 1)
        self.assertEqual(question_payload["query_prompt"], "QUERY::q2")
        self.assertEqual(question_payload["gold_answers"], ["a2"])

    def test_run_adapted_p7_queries_reuses_one_frozen_bank_across_queries(self):
        prepared = {
            "dataset_config": {"sub_dataset": "ruler_qa2_421K"},
            "context_id": "ctx-0",
            "chunks": ["chunk-1"],
            "chunk_token_lengths": [10],
            "memorization_prompts": ["mem-1"],
            "queries": [
                {
                    "query_id": 0,
                    "question": "q1",
                    "query_prompt": "QUERY::q1",
                    "gold_answers": ["yes"],
                },
                {
                    "query_id": 1,
                    "question": "q2",
                    "query_prompt": "QUERY::q2",
                    "gold_answers": ["no"],
                },
            ],
            "question_count": 2,
        }

        calls = []
        bank_object = object()

        def fake_eventqa_runner(
            _args,
            _model,
            _capacity,
            payload,
            bank_mode,
            _bank_config,
            **kwargs,
        ):
            calls.append(
                {
                    "query_id": payload.get("query_id"),
                    "bank_mode": bank_mode,
                    "construction_only": kwargs.get("construction_only", False),
                    "external_bank_is_same": kwargs.get("external_bank") is bank_object,
                    "disable_query_retrieval": kwargs.get("disable_query_retrieval", False),
                }
            )
            if kwargs.get("construction_only"):
                return {
                    "pre_query_bank_summary": {"slot_count": 1},
                    "construction_turn_diagnostics": [],
                    "query_write_count_delta": 0,
                    "query_read_only_enforced": True,
                    "_retained_bank": bank_object,
                }
            return {
                "prediction": "yes" if payload["query_id"] == 0 else "maybe",
                "query_write_count_delta": 0,
                "query_read_only_enforced": True,
                "bank_snapshot_changed_after_query": False,
                "retrieved_indices": [0],
                "retrieved_latent_count": 1,
                "pre_query_bank_summary": {"slot_count": 1},
                "post_query_bank_summary": {"slot_count": 1},
                "_retained_bank": bank_object,
            }

        artifact = run_adapted_p7_queries(
            prepared,
            args=SimpleNamespace(generation_max_length=40),
            method="p7",
            runtime_bank_config={"retrieve_threshold": 0.05, "update_threshold": 0.10, "max_slots": 16, "top_k": 2, "decay_alpha": 0.05, "retrieve_policy": "threshold_topk", "update_policy": "thread_update"},
            recorded_bank_config={"latent_memory_bank_config": {"retrieve_threshold": 0.05, "update_threshold": 0.10, "max_slots": 16, "top_k": 2, "decay_alpha": 0.05, "retrieve_policy": "threshold_topk", "update_policy": "thread_update"}},
            model_loader=lambda _args: ("model", 32768),
            eventqa_runner=fake_eventqa_runner,
        )

        self.assertEqual(artifact["methods"], ["p7"])
        self.assertEqual(len(artifact["mode_records"]["p7"]), 2)
        self.assertTrue(artifact["mode_records"]["p7"][0]["correct"])
        self.assertFalse(artifact["mode_records"]["p7"][1]["correct"])
        self.assertTrue(calls[1]["external_bank_is_same"])
        self.assertTrue(calls[2]["external_bank_is_same"])


if __name__ == "__main__":
    unittest.main()
