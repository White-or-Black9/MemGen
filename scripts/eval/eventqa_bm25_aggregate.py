"""Strictly aggregate five EventQA BM25 top-2 full-context artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.eventqa_bm25_retrieved_text import (
    BM25ContractError,
    validate_artifact,
)


SCHEMA_VERSION = "eventqa-bm25-top2-aggregate/v1"
EXPECTED_CONTEXTS = list(range(5))
EXPECTED_CONFIG = {"k1": 1.5, "b": 0.75, "top_k": 2}


class BM25AggregateError(ValueError):
    """Raised when full-pass inputs are incomplete or incomparable."""


def _mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise BM25AggregateError("aggregate values must be finite and nonempty")
    return statistics.fmean(values)


def aggregate_artifacts(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    contexts = [artifact.get("scope", {}).get("context_index") for artifact in artifacts]
    if sorted(contexts) != EXPECTED_CONTEXTS or len(set(contexts)) != 5:
        raise BM25AggregateError("full aggregate requires contexts 0-4 exactly once")
    for artifact in artifacts:
        if artifact.get("bm25") != EXPECTED_CONFIG:
            raise BM25AggregateError("BM25 configuration drift")
        try:
            validate_artifact(artifact)
        except BM25ContractError as error:
            raise BM25AggregateError(str(error)) from error

    ordered = sorted(artifacts, key=lambda item: item["scope"]["context_index"])
    records = [record for artifact in ordered for record in artifact["records"]]
    identities = {(record["context_index"], record["query_index"]) for record in records}
    if len(records) != 500 or len(identities) != 500:
        raise BM25AggregateError("aggregate requires 500 unique context/question identities")

    per_context = []
    for artifact in ordered:
        context_records = artifact["records"]
        per_context.append(
            {
                "context_index": artifact["scope"]["context_index"],
                "run_id": artifact["run_id"],
                "substring_exact_match": _mean(
                    [float(record["substring_exact_match"]) for record in context_records]
                ),
                "eventqa_recall": _mean(
                    [float(record["eventqa_recall"]) for record in context_records]
                ),
                "format_failure_count": sum(
                    int(any(record["format_flags"].values())) for record in context_records
                ),
                "method_total_seconds": artifact["cost"][
                    "index_construction_latency_seconds"
                ]
                + sum(
                    record["cost"]["end_to_end_latency_seconds"]
                    for record in context_records
                ),
                "incremental_peak_gpu_memory_bytes": artifact["cost"][
                    "incremental_peak_gpu_memory_bytes"
                ],
                "max_rendered_prompt_token_count": max(
                    record["rendered_prompt_token_count"] for record in context_records
                ),
            }
        )

    retrieval_total = sum(
        record["cost"]["retrieval_latency_seconds"] for record in records
    )
    generation_total = sum(
        record["cost"]["generation_latency_seconds"] for record in records
    )
    index_total = sum(
        artifact["cost"]["index_construction_latency_seconds"] for artifact in ordered
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "subtask": "eventqa_65536",
            "context_indices": EXPECTED_CONTEXTS,
            "question_count": 500,
            "questions_per_context": 100,
        },
        "bm25": EXPECTED_CONFIG,
        "effectiveness": {
            "substring_exact_match": _mean(
                [float(record["substring_exact_match"]) for record in records]
            ),
            "eventqa_recall": _mean(
                [float(record["eventqa_recall"]) for record in records]
            ),
            "format_failure_count": sum(
                int(any(record["format_flags"].values())) for record in records
            ),
        },
        "cost": {
            "index_construction_latency_seconds": index_total,
            "retrieval_latency_seconds": retrieval_total,
            "generation_latency_seconds": generation_total,
            "method_total_seconds": index_total + retrieval_total + generation_total,
            "amortized_seconds_per_question": (
                index_total + retrieval_total + generation_total
            )
            / 500,
            "incremental_peak_gpu_memory_bytes_max": max(
                artifact["cost"]["incremental_peak_gpu_memory_bytes"]
                for artifact in ordered
            ),
        },
        "capacity": {
            "max_rendered_prompt_token_count": max(
                record["rendered_prompt_token_count"] for record in records
            ),
            "context_capacity": records[0]["context_capacity"],
            "all_capacity_ok": all(record["capacity_ok"] for record in records),
        },
        "per_context": per_context,
    }


def _markdown(summary: dict[str, Any]) -> str:
    effectiveness = summary["effectiveness"]
    cost = summary["cost"]
    lines = [
        "# EventQA BM25 Top-2 Full Aggregate",
        "",
        f"- Questions: {summary['scope']['question_count']}",
        f"- EM: {effectiveness['substring_exact_match']:.6f}",
        f"- EventQA recall: {effectiveness['eventqa_recall']:.6f}",
        f"- Format failures: {effectiveness['format_failure_count']}",
        f"- Method total: {cost['method_total_seconds']:.3f} s",
        f"- Amortized: {cost['amortized_seconds_per_question']:.3f} s/question",
        f"- Max incremental peak: {cost['incremental_peak_gpu_memory_bytes_max'] / 2**20:.1f} MiB",
        f"- Max prompt: {summary['capacity']['max_rendered_prompt_token_count']}/{summary['capacity']['context_capacity']} tokens",
        "",
        "| Context | EM | Recall | Format failures | Method seconds | Max prompt tokens |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_context"]:
        lines.append(
            f"| {row['context_index']} | {row['substring_exact_match']:.3f} | "
            f"{row['eventqa_recall']:.3f} | {row['format_failure_count']} | "
            f"{row['method_total_seconds']:.3f} | {row['max_rendered_prompt_token_count']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument(
        "--output-json", default="outputs/mab/eventqa_bm25_top2_full_aggregate.json"
    )
    parser.add_argument(
        "--output-md", default="outputs/mab/eventqa_bm25_top2_full_aggregate.md"
    )
    args = parser.parse_args(argv)
    if len(args.artifact) != 5:
        raise BM25AggregateError("exactly five --artifact paths are required")
    artifacts = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.artifact]
    summary = aggregate_artifacts(artifacts)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output_md.write_text(_markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
