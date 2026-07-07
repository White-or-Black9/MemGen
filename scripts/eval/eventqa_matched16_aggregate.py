"""Strictly aggregate five EventQA matched16 full artifacts."""

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

from scripts.eval.eventqa_matched16_retrieved_text import (
    Matched16ContractError,
    validate_artifact,
)


SCHEMA_VERSION = "eventqa-matched16-aggregate/v1"
EXPECTED_CONTEXTS = list(range(5))


class Matched16AggregateError(ValueError):
    """Raised when full matched16 artifacts are incomplete or invalid."""


def _mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise Matched16AggregateError("aggregate values must be finite")
    return statistics.fmean(values)


def aggregate_artifacts(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    contexts = [artifact.get("scope", {}).get("context_index") for artifact in artifacts]
    if sorted(contexts) != EXPECTED_CONTEXTS or len(set(contexts)) != 5:
        raise Matched16AggregateError("full aggregate requires contexts 0-4 exactly once")
    for artifact in artifacts:
        try:
            validate_artifact(artifact)
        except Matched16ContractError as error:
            raise Matched16AggregateError(str(error)) from error
    ordered = sorted(artifacts, key=lambda item: item["scope"]["context_index"])
    records = [record for artifact in ordered for record in artifact["records"]]
    identities = {(record["context_index"], record["query_index"]) for record in records}
    if len(records) != 500 or len(identities) != 500:
        raise Matched16AggregateError("aggregate requires 500 unique identities")
    source_counts = sorted({record["source_token_count"] for record in records})
    prompt_deltas = sorted({record["rendered_prompt_token_delta"] for record in records})
    if source_counts != [16] or prompt_deltas != [16]:
        raise Matched16AggregateError("all source counts and prompt deltas must equal 16")

    per_context = []
    for artifact in ordered:
        rows = artifact["records"]
        per_context.append(
            {
                "context_index": artifact["scope"]["context_index"],
                "run_id": artifact["run_id"],
                "substring_exact_match": _mean(
                    [float(row["substring_exact_match"]) for row in rows]
                ),
                "eventqa_recall": _mean([float(row["eventqa_recall"]) for row in rows]),
                "format_failure_count": sum(
                    int(any(row["format_flags"].values())) for row in rows
                ),
                "method_total_seconds": artifact["cost"][
                    "index_construction_latency_seconds"
                ]
                + sum(row["cost"]["end_to_end_latency_seconds"] for row in rows),
                "incremental_peak_gpu_memory_bytes": artifact["cost"][
                    "incremental_peak_gpu_memory_bytes"
                ],
            }
        )
    index_total = sum(
        artifact["cost"]["index_construction_latency_seconds"] for artifact in ordered
    )
    selection_total = sum(
        row["cost"]["retrieval_and_window_latency_seconds"] for row in records
    )
    generation_total = sum(row["cost"]["generation_latency_seconds"] for row in records)
    method_total = index_total + selection_total + generation_total
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "context_indices": EXPECTED_CONTEXTS,
            "question_count": 500,
            "questions_per_context": 100,
        },
        "budget": {
            "source_token_counts": source_counts,
            "rendered_prompt_token_deltas": prompt_deltas,
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
            "index_construction_latency_seconds": index_total,
            "retrieval_and_window_latency_seconds": selection_total,
            "generation_latency_seconds": generation_total,
            "method_total_seconds": method_total,
            "amortized_seconds_per_question": method_total / 500,
            "incremental_peak_gpu_memory_bytes_max": max(
                artifact["cost"]["incremental_peak_gpu_memory_bytes"]
                for artifact in ordered
            ),
        },
        "capacity": {
            "max_matched_rendered_token_count": max(
                row["matched_rendered_token_count"] for row in records
            ),
            "context_capacity": records[0]["context_capacity"],
            "all_capacity_ok": all(row["capacity_ok"] for row in records),
        },
        "per_context": per_context,
    }


def _markdown(summary: dict[str, Any]) -> str:
    effect = summary["effectiveness"]
    cost = summary["cost"]
    lines = [
        "# EventQA Strict Matched16 Full Aggregate",
        "",
        f"- Questions: {summary['scope']['question_count']}",
        f"- EM: {effect['substring_exact_match']:.6f}",
        f"- Recall: {effect['eventqa_recall']:.6f}",
        f"- Format failures: {effect['format_failure_count']}",
        f"- Method total: {cost['method_total_seconds']:.3f} s",
        f"- Amortized: {cost['amortized_seconds_per_question']:.3f} s/question",
        f"- Max incremental peak: {cost['incremental_peak_gpu_memory_bytes_max']/2**20:.1f} MiB",
        "",
        "| Context | EM | Recall | Format failures | Method seconds |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_context"]:
        lines.append(
            f"| {row['context_index']} | {row['substring_exact_match']:.3f} | "
            f"{row['eventqa_recall']:.3f} | {row['format_failure_count']} | "
            f"{row['method_total_seconds']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument(
        "--output-json", default="outputs/mab/eventqa_matched16_full_aggregate.json"
    )
    parser.add_argument(
        "--output-md", default="outputs/mab/eventqa_matched16_full_aggregate.md"
    )
    args = parser.parse_args(argv)
    if len(args.artifact) != 5:
        raise Matched16AggregateError("exactly five --artifact paths are required")
    artifacts = [json.loads(Path(path).read_text()) for path in args.artifact]
    summary = aggregate_artifacts(artifacts)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n")
    output_md.write_text(_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
