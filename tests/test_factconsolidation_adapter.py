import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.eval.factconsolidation_adapter import (
    SUPPORTED_SUBTASKS,
    inspect_subtask,
    load_rows,
    normalize_context,
    score_prediction,
)


def fake_templates():
    return {
        "memorize": lambda *, context, time_stamp: f"MEM::{time_stamp}::{context}",
        "query": lambda *, question: f"QUERY::{question}",
    }


class FactConsolidationAdapterTest(unittest.TestCase):
    def test_normalize_preserves_chunk_and_query_order(self):
        row = {
            "context": "first. second.",
            "questions": ["q1", "q2"],
            "answers": [["a1"], ["a2"]],
            "metadata": {
                "source": "factconsolidation_sh_6k",
                "qa_pair_ids": ["pair-1", "pair-2"],
            },
        }

        payload = normalize_context(
            row,
            subtask="factconsolidation_sh_6k",
            chunker=lambda text, chunk_size: ["first.", "second."],
            templates=fake_templates(),
            chunk_size=4096,
            timestamp="2026-07-08 00:00:00",
            dataset_config={"chunk_size": 4096},
            config_hash="config-sha",
            parquet_hash="parquet-sha",
        )

        self.assertEqual(payload["subtask"], "factconsolidation_sh_6k")
        self.assertEqual(payload["chunks"], ["first.", "second."])
        self.assertEqual(payload["qa_pair_ids"], ["pair-1", "pair-2"])
        self.assertEqual(payload["config_hash"], "config-sha")
        self.assertEqual(payload["parquet_hash"], "parquet-sha")
        self.assertEqual(
            [query["question"] for query in payload["queries"]],
            ["q1", "q2"],
        )
        self.assertEqual(payload["queries"][0]["gold_answers"], ["a1"])
        self.assertEqual(payload["queries"][1]["gold_answers"], ["a2"])
        self.assertEqual(
            payload["memorization_prompts"],
            [
                "MEM::2026-07-08 00:00:00::first.",
                "MEM::2026-07-08 00:00:00::second.",
            ],
        )
        self.assertEqual(payload["queries"][0]["query_prompt"], "QUERY::q1")
        self.assertEqual(payload["queries"][1]["query_prompt"], "QUERY::q2")

    def test_invalid_source_name_fails_loudly(self):
        row = {
            "context": "x",
            "questions": ["q"],
            "answers": [["a"]],
            "metadata": {"source": "wrong"},
        }

        with self.assertRaisesRegex(ValueError, "source mismatch"):
            normalize_context(
                row,
                subtask="factconsolidation_sh_6k",
                chunker=lambda text, chunk_size: [text],
                templates=fake_templates(),
                chunk_size=4096,
                timestamp="2026-07-08 00:00:00",
                dataset_config={"chunk_size": 4096},
                config_hash="config-sha",
                parquet_hash="parquet-sha",
            )

    def test_load_rows_filters_by_supported_subtask(self):
        rows = [
            {
                "context": "a",
                "questions": ["q1"],
                "answers": [["a1"]],
                "metadata": {"source": "factconsolidation_sh_6k"},
            },
            {
                "context": "b",
                "questions": ["q2"],
                "answers": [["a2"]],
                "metadata": {"source": "factconsolidation_mh_6k"},
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "rows.parquet"
            pq.write_table(pa.Table.from_pylist(rows), parquet_path)

            filtered = load_rows(parquet_path, "factconsolidation_sh_6k")

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["metadata"]["source"], "factconsolidation_sh_6k")
        self.assertIn("factconsolidation_sh_6k", SUPPORTED_SUBTASKS)

    def test_load_rows_rejects_unsupported_subtask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = Path(tmpdir) / "rows.parquet"
            pq.write_table(pa.Table.from_pylist([]), parquet_path)

            with self.assertRaisesRegex(ValueError, "unsupported FactConsolidation"):
                load_rows(parquet_path, "factconsolidation_sh_262k")

    def test_score_prediction_delegates_to_official_post_process(self):
        captured = {}

        def post_process(prediction_payload, gold_answers, dataset_config):
            captured["prediction_payload"] = prediction_payload
            captured["gold_answers"] = gold_answers
            captured["dataset_config"] = dataset_config
            return {"substring_exact_match": 1.0}, {"trace": "ok"}

        result = score_prediction(
            "Berlin",
            ["Berlin"],
            {"sub_dataset": "factconsolidation_sh_6k"},
            post_process,
        )

        self.assertEqual(result["metrics"]["substring_exact_match"], 1.0)
        self.assertEqual(result["additional"]["trace"], "ok")
        self.assertEqual(captured["prediction_payload"], {"output": "Berlin"})
        self.assertEqual(captured["gold_answers"], ["Berlin"])
        self.assertEqual(
            captured["dataset_config"]["sub_dataset"], "factconsolidation_sh_6k"
        )

    def test_inspect_subtask_reports_hashes_and_counts(self):
        row = {
            "context": "first. second.",
            "questions": ["q1", "q2"],
            "answers": [["a1"], ["a2"]],
            "metadata": {
                "source": "factconsolidation_sh_6k",
                "qa_pair_ids": ["pair-1", "pair-2"],
            },
        }

        report = inspect_subtask(
            subtask="factconsolidation_sh_6k",
            rows=[row],
            dataset_config={"chunk_size": 4096, "context_max_length": 6000},
            chunker=lambda text, chunk_size: ["first.", "second."],
            templates=fake_templates(),
            config_hash="config-sha",
            parquet_hash="parquet-sha",
            timestamp="2026-07-08 00:00:00",
        )

        self.assertEqual(report["matched_rows"], 1)
        self.assertEqual(report["question_count"], 2)
        self.assertEqual(report["chunk_count"], 2)
        self.assertEqual(report["config_hash"], "config-sha")
        self.assertEqual(report["parquet_hash"], "parquet-sha")


if __name__ == "__main__":
    unittest.main()
