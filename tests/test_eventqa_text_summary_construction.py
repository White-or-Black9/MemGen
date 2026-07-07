import hashlib
import math
import unittest

from scripts.eval.eventqa_text_summary_construction import (
    SummaryContractError,
    build_summary_prompt,
    persist_summary,
    validate_construction_artifact,
)


class SpaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()

    def decode(self, token_ids, **kwargs):
        del kwargs
        return " ".join(token_ids)


class SummaryConstructionTest(unittest.TestCase):
    def test_prompt_contains_only_previous_summary_and_chunk(self):
        prompt = build_summary_prompt("old memory", "new event")
        self.assertIn("Previous summary:\nold memory", prompt)
        self.assertIn("New event text:\nnew event", prompt)
        self.assertNotIn("Question:", prompt)
        self.assertNotIn("gold", prompt.lower())

    def test_persistence_truncates_deterministically_to_128_tokens(self):
        tokenizer = SpaceTokenizer()
        raw = " ".join(f"t{i}" for i in range(140))
        persisted = persist_summary(tokenizer, raw, budget=128)
        self.assertEqual(persisted["raw_token_count"], 140)
        self.assertEqual(persisted["persisted_token_count"], 128)
        self.assertTrue(persisted["truncated"])
        self.assertEqual(persisted["persisted_token_ids"], [f"t{i}" for i in range(128)])

    def test_artifact_requires_ordered_complete_nonempty_trace(self):
        traces = []
        previous = ""
        for index in range(2):
            summary = f"summary {index}"
            traces.append(
                {
                    "chunk_index": index,
                    "chunk_sha256": "a" * 64,
                    "chunk_token_count": 100,
                    "previous_summary_sha256": hashlib.sha256(previous.encode()).hexdigest(),
                    "previous_summary_token_count": 0 if not previous else 2,
                    "rendered_input_sha256": "b" * 64,
                    "rendered_input_token_count": 200,
                    "raw_output_sha256": hashlib.sha256(summary.encode()).hexdigest(),
                    "raw_output_token_count": 2,
                    "persisted_summary": summary,
                    "persisted_summary_sha256": hashlib.sha256(summary.encode()).hexdigest(),
                    "persisted_summary_token_count": 2,
                    "truncated": False,
                    "latency_seconds": 1.0,
                }
            )
            previous = summary
        artifact = {
            "schema_version": "eventqa-text-summary-construction/v1",
            "scope": {"context_index": 0, "chunk_count": 2},
            "method": {"summary_token_budget": 128, "latent_memory_bank": False},
            "cost": {
                "construction_latency_seconds": 2.0,
                "baseline_gpu_memory_bytes": 100,
                "peak_gpu_memory_bytes": 200,
            },
            "traces": traces,
            "final_summary": previous,
            "final_summary_sha256": hashlib.sha256(previous.encode()).hexdigest(),
            "final_summary_token_count": 2,
        }
        validate_construction_artifact(artifact)
        artifact["traces"][1]["chunk_index"] = 2
        with self.assertRaisesRegex(SummaryContractError, "ordered"):
            validate_construction_artifact(artifact)

    def test_artifact_rejects_empty_summary_and_nonfinite_cost(self):
        artifact = {
            "schema_version": "eventqa-text-summary-construction/v1",
            "scope": {"context_index": 0, "chunk_count": 1},
            "method": {"summary_token_budget": 128, "latent_memory_bank": False},
            "cost": {
                "construction_latency_seconds": math.nan,
                "baseline_gpu_memory_bytes": 100,
                "peak_gpu_memory_bytes": 200,
            },
            "traces": [],
            "final_summary": "",
            "final_summary_sha256": hashlib.sha256(b"").hexdigest(),
            "final_summary_token_count": 0,
        }
        with self.assertRaises(SummaryContractError):
            validate_construction_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
