"""Strictly aggregate five provenance-linked EventQA text-summary pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.eventqa_text_summary_construction import (
    SummaryContractError,
    validate_construction_artifact,
)
from scripts.eval.eventqa_text_summary_query import (
    SummaryQueryContractError,
    validate_query_artifact,
)


SCHEMA_VERSION = "eventqa-text-summary-aggregate/v1"
EXPECTED_CONTEXTS = list(range(5))


class TextSummaryAggregateError(ValueError):
    """Raised when full text-summary inputs are incomplete or inconsistent."""


def _mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise TextSummaryAggregateError("aggregate values must be finite and nonempty")
    return statistics.fmean(values)


def _load(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    return json.loads(raw), raw


def aggregate_pairs(
    construction_paths: list[Path],
    query_paths: list[Path],
    *,
    controlled_cost_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(construction_paths) != 5 or len(query_paths) != 5:
        raise TextSummaryAggregateError("full aggregate requires contexts 0-4 exactly once")

    constructions: dict[int, tuple[dict[str, Any], Path, bytes]] = {}
    for path in map(Path, construction_paths):
        artifact, raw = _load(path)
        try:
            validate_construction_artifact(artifact)
        except SummaryContractError as error:
            raise TextSummaryAggregateError(str(error)) from error
        context_index = artifact["scope"]["context_index"]
        if context_index in constructions:
            raise TextSummaryAggregateError("full aggregate requires contexts 0-4 exactly once")
        constructions[context_index] = (artifact, path, raw)

    queries: dict[int, tuple[dict[str, Any], Path]] = {}
    for path in map(Path, query_paths):
        artifact, _ = _load(path)
        try:
            validate_query_artifact(artifact)
        except SummaryQueryContractError as error:
            raise TextSummaryAggregateError(str(error)) from error
        context_index = artifact["scope"]["context_index"]
        if context_index in queries:
            raise TextSummaryAggregateError("full aggregate requires contexts 0-4 exactly once")
        queries[context_index] = (artifact, path)

    if sorted(constructions) != EXPECTED_CONTEXTS or sorted(queries) != EXPECTED_CONTEXTS:
        raise TextSummaryAggregateError("full aggregate requires contexts 0-4 exactly once")

    records: list[dict[str, Any]] = []
    per_context: list[dict[str, Any]] = []
    for context_index in EXPECTED_CONTEXTS:
        construction, construction_path, construction_raw = constructions[context_index]
        query, query_path = queries[context_index]
        source = query["summary_source"]
        if source["artifact_sha256"] != hashlib.sha256(construction_raw).hexdigest():
            raise TextSummaryAggregateError(
                f"context {context_index} construction artifact hash mismatch"
            )
        if source["final_summary_sha256"] != construction["final_summary_sha256"]:
            raise TextSummaryAggregateError(
                f"context {context_index} final summary hash mismatch"
            )
        if source["final_summary_token_count"] != construction["final_summary_token_count"]:
            raise TextSummaryAggregateError(
                f"context {context_index} final summary token count mismatch"
            )
        if query["cost"]["construction_latency_seconds"] != construction["cost"]["construction_latency_seconds"]:
            raise TextSummaryAggregateError(
                f"context {context_index} construction latency mismatch"
            )

        rows = query["records"]
        records.extend(rows)
        prompt_deltas = sorted({row["rendered_prompt_token_delta"] for row in rows})
        per_context.append(
            {
                "context_index": context_index,
                "construction_artifact": str(construction_path),
                "query_artifact": str(query_path),
                "construction_artifact_sha256": source["artifact_sha256"],
                "final_summary_sha256": source["final_summary_sha256"],
                "final_summary_token_count": source["final_summary_token_count"],
                "rendered_prompt_token_deltas": prompt_deltas,
                "substring_exact_match": _mean(
                    [float(row["substring_exact_match"]) for row in rows]
                ),
                "eventqa_recall": _mean(
                    [float(row["eventqa_recall"]) for row in rows]
                ),
                "format_failure_count": sum(
                    int(any(row["format_flags"].values())) for row in rows
                ),
                "construction_latency_seconds": construction["cost"]["construction_latency_seconds"],
                "query_latency_seconds": query["cost"]["query_latency_seconds"],
                "end_to_end_latency_seconds": query["cost"]["end_to_end_latency_seconds"],
                "construction_incremental_peak_gpu_memory_bytes": construction["cost"].get(
                    "incremental_peak_gpu_memory_bytes",
                    construction["cost"]["peak_gpu_memory_bytes"] - construction["cost"]["baseline_gpu_memory_bytes"],
                ),
                "query_incremental_peak_gpu_memory_bytes": query["cost"].get(
                    "incremental_peak_gpu_memory_bytes",
                    query["cost"]["peak_gpu_memory_bytes"] - query["cost"]["baseline_gpu_memory_bytes"],
                ),
            }
        )

    identities = {(row["context_index"], row["query_index"]) for row in records}
    if len(records) != 500 or len(identities) != 500:
        raise TextSummaryAggregateError("aggregate requires 500 unique identities")

    construction_total = sum(row["construction_latency_seconds"] for row in per_context)
    query_total = sum(row["query_latency_seconds"] for row in per_context)
    if controlled_cost_evidence is None:
        cost_status = {
            "confounded_by_shared_gpu": True,
            "paper_facing": False,
            "caveat": "Full-pass timing and peak-memory measurements were collected under shared-GPU contention and are not paper-facing cost evidence.",
        }
    else:
        if controlled_cost_evidence.get("schema_version") != "eventqa-text-summary-controlled-cost/v1":
            raise TextSummaryAggregateError("unexpected controlled-cost evidence schema")
        if controlled_cost_evidence.get("context_indices") != EXPECTED_CONTEXTS:
            raise TextSummaryAggregateError("controlled-cost evidence must cover contexts 0-4")
        if controlled_cost_evidence.get("serialized_single_gpu") is not True:
            raise TextSummaryAggregateError("controlled-cost evidence must declare serialized single-GPU execution")
        if controlled_cost_evidence.get("all_preflight_clear") is not True:
            raise TextSummaryAggregateError("controlled-cost evidence preflight is not clear")
        gpu_index = controlled_cost_evidence.get("gpu_index")
        if not isinstance(gpu_index, int) or gpu_index < 0:
            raise TextSummaryAggregateError("controlled-cost evidence requires a nonnegative gpu_index")
        cost_status = {
            "confounded_by_shared_gpu": False,
            "paper_facing": True,
            "caveat": "Measured in serialized single-GPU processes after clear per-context occupancy preflights.",
            "controlled_cost_evidence": controlled_cost_evidence,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "context_indices": EXPECTED_CONTEXTS,
            "question_count": 500,
            "questions_per_context": 100,
        },
        "method": {"summary_token_budget": 128, "latent_memory_bank": False},
        "summary": {
            "final_summary_token_counts": sorted(
                {row["final_summary_token_count"] for row in per_context}
            ),
            "rendered_prompt_token_deltas": sorted(
                {delta for row in per_context for delta in row["rendered_prompt_token_deltas"]}
            ),
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
            "construction_incremental_peak_gpu_memory_bytes_max": max(
                row["construction_incremental_peak_gpu_memory_bytes"] for row in per_context
            ),
            "query_incremental_peak_gpu_memory_bytes_max": max(
                row["query_incremental_peak_gpu_memory_bytes"] for row in per_context
            ),
            **cost_status,
        },
        "capacity": {
            "context_capacity": records[0]["context_capacity"],
            "max_injected_rendered_token_count": max(
                row["injected_rendered_token_count"] for row in records
            ),
            "all_capacity_ok": all(row["capacity_ok"] for row in records),
        },
        "per_context": per_context,
    }


def _markdown(summary: dict[str, Any]) -> str:
    effect = summary["effectiveness"]
    cost = summary["cost"]
    lines = [
        "# EventQA Same-Model Text-Summary Full Aggregate",
        "",
        f"- Questions: {summary['scope']['question_count']}",
        f"- EM: {effect['substring_exact_match']:.6f}",
        f"- EventQA recall: {effect['eventqa_recall']:.6f}",
        f"- Format failures: {effect['format_failure_count']}",
        f"- Summary token counts: {summary['summary']['final_summary_token_counts']}",
        f"- Rendered prompt deltas: {summary['summary']['rendered_prompt_token_deltas']}",
        f"- Construction total: {cost['construction_latency_seconds']:.3f} s",
        f"- Query total: {cost['query_latency_seconds']:.3f} s",
        f"- End-to-end total: {cost['end_to_end_latency_seconds']:.3f} s",
        f"- Cost status: {cost['caveat']}",
        "",
        "| Context | Summary tokens | Prompt delta(s) | EM | Recall | Format failures | E2E seconds |",
        "|---:|---:|:---|---:|---:|---:|---:|",
    ]
    for row in summary["per_context"]:
        lines.append(
            f"| {row['context_index']} | {row['final_summary_token_count']} | "
            f"{row['rendered_prompt_token_deltas']} | {row['substring_exact_match']:.3f} | "
            f"{row['eventqa_recall']:.3f} | {row['format_failure_count']} | "
            f"{row['end_to_end_latency_seconds']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction", action="append", required=True)
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument(
        "--output-json", default="outputs/mab/eventqa_text_summary_full_aggregate.json"
    )
    parser.add_argument(
        "--output-md", default="outputs/mab/eventqa_text_summary_full_aggregate.md"
    )
    parser.add_argument(
        "--controlled-cost-evidence",
        help="JSON attestation for a serialized single-GPU cost run with clear occupancy preflights.",
    )
    args = parser.parse_args(argv)
    controlled_cost_evidence = None
    if args.controlled_cost_evidence:
        controlled_cost_evidence = json.loads(
            Path(args.controlled_cost_evidence).read_text(encoding="utf-8")
        )
    summary = aggregate_pairs(
        [Path(path) for path in args.construction],
        [Path(path) for path in args.query],
        controlled_cost_evidence=controlled_cost_evidence,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    output_md.write_text(_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
