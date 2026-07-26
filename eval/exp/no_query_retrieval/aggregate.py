"""Aggregate five full EventQA P7 no-query-retrieval artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.eventqa_p7_no_query_retrieval import (
    NoQueryRetrievalContractError,
    validate_artifact,
)


SCHEMA_VERSION = "eventqa-p7-no-query-retrieval-aggregate/v1"
EXPECTED_CONTEXTS = list(range(5))


class NoQueryRetrievalAggregateError(ValueError):
    """Raised when full no-query-retrieval artifacts are incomplete or inconsistent."""


def _mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise NoQueryRetrievalAggregateError("aggregate values must be finite and nonempty")
    return statistics.fmean(values)


def aggregate_artifacts(paths: list[Path]) -> dict[str, Any]:
    if len(paths) != 5:
        raise NoQueryRetrievalAggregateError("full aggregate requires contexts 0-4 exactly once")

    artifacts: dict[int, tuple[dict[str, Any], Path]] = {}
    for path in map(Path, paths):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        try:
            validate_artifact(artifact)
        except NoQueryRetrievalContractError as error:
            raise NoQueryRetrievalAggregateError(str(error)) from error
        context_index = artifact["scope"]["context_index"]
        if context_index in artifacts:
            raise NoQueryRetrievalAggregateError("full aggregate requires contexts 0-4 exactly once")
        artifacts[context_index] = (artifact, path)

    if sorted(artifacts) != EXPECTED_CONTEXTS:
        raise NoQueryRetrievalAggregateError("full aggregate requires contexts 0-4 exactly once")

    records: list[dict[str, Any]] = []
    per_context: list[dict[str, Any]] = []
    for context_index in EXPECTED_CONTEXTS:
        artifact, path = artifacts[context_index]
        rows = artifact["records"]
        if any(int(row["query_invariants"]["retrieved_latent_count"]) != 0 for row in rows):
            raise NoQueryRetrievalAggregateError("retrieval disabled but a query retrieved latents")
        records.extend(rows)
        query_total = sum(float(row["cost"]["query_latency_seconds"]) for row in rows)
        per_context.append(
            {
                "context_index": context_index,
                "artifact_path": str(path),
                "substring_exact_match": _mean(
                    [float(row["substring_exact_match"]) for row in rows]
                ),
                "eventqa_recall": _mean(
                    [float(row["eventqa_recall"]) for row in rows]
                ),
                "format_failure_count": sum(
                    int(any(row["format_flags"].values())) for row in rows
                ),
                "construction_latency_seconds": float(
                    artifact["construction"]["construction_latency_seconds"]
                ),
                "query_latency_seconds": query_total,
                "end_to_end_latency_seconds": float(
                    artifact["cost"]["end_to_end_latency_seconds"]
                ),
                "incremental_peak_gpu_memory_bytes": int(
                    artifact["cost"]["peak_gpu_memory_bytes"]
                    - artifact["cost"]["baseline_gpu_memory_bytes"]
                ),
                "final_slot_count": int(artifact["construction"]["final_slot_count"]),
            }
        )

    identities = {(row["context_index"], row["query_index"]) for row in records}
    if len(records) != 500 or len(identities) != 500:
        raise NoQueryRetrievalAggregateError("aggregate requires 500 unique identities")

    construction_total = sum(row["construction_latency_seconds"] for row in per_context)
    query_total = sum(row["query_latency_seconds"] for row in per_context)
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "context_indices": EXPECTED_CONTEXTS,
            "question_count": 500,
            "questions_per_context": 100,
        },
        "method_config": {
            "retrieve_threshold": 0.05,
            "update_threshold": 0.10,
            "max_slots": 16,
            "top_k": 2,
            "decay_alpha": 0.05,
            "generation_max_length": 40,
            "eventqa_protocol": "frozen_context_bank",
            "query_retrieval_disabled": True,
        },
        "construction": {
            "final_slot_counts": sorted({row["final_slot_count"] for row in per_context}),
        },
        "effectiveness": {
            "substring_exact_match": _mean(
                [float(row["substring_exact_match"]) for row in records]
            ),
            "eventqa_recall": _mean([float(row["eventqa_recall"]) for row in records]),
            "format_failure_count": sum(
                int(any(row["format_flags"].values())) for row in records
            ),
        },
        "cost": {
            "construction_latency_seconds": construction_total,
            "query_latency_seconds": query_total,
            "end_to_end_latency_seconds": construction_total + query_total,
            "construction_amortized_seconds_per_question": construction_total / 500,
            "query_amortized_seconds_per_question": query_total / 500,
            "end_to_end_amortized_seconds_per_question": (construction_total + query_total) / 500,
            "incremental_peak_gpu_memory_bytes_max": max(
                row["incremental_peak_gpu_memory_bytes"] for row in per_context
            ),
        },
        "invariants": {
            "all_queries_disable_retrieval": all(
                int(row["query_invariants"]["retrieved_latent_count"]) == 0
                and not row["query_invariants"]["retrieved_indices"]
                and int(row["query_invariants"]["query_write_count"]) == 0
                and row["query_invariants"]["bank_snapshot_changed_after_query"] is False
                for row in records
            ),
        },
        "per_context": per_context,
    }


def _markdown(summary: dict[str, Any]) -> str:
    effect = summary["effectiveness"]
    cost = summary["cost"]
    lines = [
        "# EventQA P7 No-Query-Retrieval Full Aggregate",
        "",
        f"- Questions: {summary['scope']['question_count']}",
        f"- EM: {effect['substring_exact_match']:.6f}",
        f"- EventQA recall: {effect['eventqa_recall']:.6f}",
        f"- Format failures: {effect['format_failure_count']}",
        f"- Construction total: {cost['construction_latency_seconds']:.3f} s",
        f"- Query total: {cost['query_latency_seconds']:.3f} s",
        f"- End-to-end total: {cost['end_to_end_latency_seconds']:.3f} s",
        f"- Max incremental peak GPU: {cost['incremental_peak_gpu_memory_bytes_max']} bytes",
        "",
        "| Context | EM | Recall | Format failures | Construction s | Query s | E2E s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_context"]:
        lines.append(
            f"| {row['context_index']} | {row['substring_exact_match']:.3f} | "
            f"{row['eventqa_recall']:.3f} | {row['format_failure_count']} | "
            f"{row['construction_latency_seconds']:.3f} | {row['query_latency_seconds']:.3f} | "
            f"{row['end_to_end_latency_seconds']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        nargs=5,
        required=True,
        help="Five full artifact json paths for contexts 0-4.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)

    summary = aggregate_artifacts([Path(path) for path in args.artifacts])
    Path(args.output_json).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    Path(args.output_md).write_text(_markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
