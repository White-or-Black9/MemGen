"""Strictly aggregate five capacity-max recent-text MemGen EventQA artifacts."""

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

from scripts.eval.eventqa_memgen_recent_window import (
    SCHEMA_VERSION as ARTIFACT_SCHEMA_VERSION,
    RecentWindowContractError,
    validate_artifact,
)


SCHEMA_VERSION = "eventqa-memgen-recent-window-aggregate/v1"
EXPECTED_CONTEXTS = list(range(5))


class RecentWindowAggregateError(ValueError):
    """Raised when the capacity-max full pass is incomplete or inconsistent."""


def _mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise RecentWindowAggregateError("aggregate values must be finite and nonempty")
    return statistics.fmean(values)


def aggregate_artifacts(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    contexts = [artifact.get("scope", {}).get("context_index") for artifact in artifacts]
    if sorted(contexts) != EXPECTED_CONTEXTS or len(set(contexts)) != len(EXPECTED_CONTEXTS):
        raise RecentWindowAggregateError("full aggregate requires contexts 0-4 exactly once")
    for artifact in artifacts:
        if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise RecentWindowAggregateError("unexpected artifact schema")
        if artifact.get("scope", {}).get("measurement_scope") != "full":
            raise RecentWindowAggregateError("all inputs must be full artifacts")
        try:
            validate_artifact(artifact)
        except RecentWindowContractError as error:
            raise RecentWindowAggregateError(str(error)) from error

    ordered = sorted(artifacts, key=lambda artifact: artifact["scope"]["context_index"])
    methods = [artifact["method"] for artifact in ordered]
    expected_method = methods[0]
    comparable_keys = (
        "bank_mode",
        "history_policy",
        "requested_recent_history_token_budget",
        "resolved_recent_history_token_budget",
        "generation_reserve_tokens",
    )
    if any(any(method[key] != expected_method[key] for key in comparable_keys) for method in methods[1:]):
        raise RecentWindowAggregateError("recent-window configuration drift across contexts")

    records = [record for artifact in ordered for record in artifact["records"]]
    identities = {(record["context_index"], record["query_index"]) for record in records}
    if len(records) != 500 or len(identities) != 500:
        raise RecentWindowAggregateError("aggregate requires 500 unique context/question identities")

    per_context = []
    for artifact in ordered:
        rows = artifact["records"]
        per_context.append(
            {
                "context_index": artifact["scope"]["context_index"],
                "run_id": artifact["run_id"],
                "substring_exact_match": _mean([float(row["substring_exact_match"]) for row in rows]),
                "eventqa_recall": _mean([float(row["eventqa_recall"]) for row in rows]),
                "format_failure_count": sum(int(any(row["format_flags"].values())) for row in rows),
                "max_rendered_prompt_token_count": max(row["rendered_prompt_token_count"] for row in rows),
                "incremental_peak_gpu_memory_bytes": artifact["cost"]["incremental_peak_gpu_memory_bytes"],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": {"context_indices": EXPECTED_CONTEXTS, "question_count": 500, "questions_per_context": 100},
        "method": {**expected_method, "model_path": expected_method["model_path"]},
        "effectiveness": {
            "substring_exact_match": _mean([float(row["substring_exact_match"]) for row in records]),
            "eventqa_recall": _mean([float(row["eventqa_recall"]) for row in records]),
            "format_failure_count": sum(int(any(row["format_flags"].values())) for row in records),
        },
        "capacity": {
            "context_capacity": records[0]["context_capacity"],
            "max_rendered_prompt_token_count": max(row["rendered_prompt_token_count"] for row in records),
            "all_capacity_ok": all(row["capacity_ok"] for row in records),
        },
        "cost": {
            "paper_facing": False,
            "caveat": "Shared-GPU effect run; latency and memory are not paper-facing cost evidence.",
            "incremental_peak_gpu_memory_bytes_max": max(
                artifact["cost"]["incremental_peak_gpu_memory_bytes"] for artifact in ordered
            ),
        },
        "per_context": per_context,
    }


def _markdown(summary: dict[str, Any]) -> str:
    effect, capacity, cost = summary["effectiveness"], summary["capacity"], summary["cost"]
    lines = [
        "# EventQA MemGen Recent-Text Window Full Aggregate",
        "",
        f"- Questions: {summary['scope']['question_count']}",
        f"- EM: {effect['substring_exact_match']:.6f}",
        f"- EventQA recall: {effect['eventqa_recall']:.6f}",
        f"- Format failures: {effect['format_failure_count']}",
        f"- Recent-text budget: {summary['method']['resolved_recent_history_token_budget']} tokens",
        f"- Max prompt: {capacity['max_rendered_prompt_token_count']}/{capacity['context_capacity']} tokens",
        f"- Cost status: {cost['caveat']}",
        "",
        "| Context | EM | Recall | Format failures | Max prompt tokens |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_context"]:
        lines.append(
            f"| {row['context_index']} | {row['substring_exact_match']:.3f} | "
            f"{row['eventqa_recall']:.3f} | {row['format_failure_count']} | "
            f"{row['max_rendered_prompt_token_count']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--output-json", default="outputs/mab/eventqa_memgen_recent_window_bmax_full_aggregate.json")
    parser.add_argument("--output-md", default="outputs/mab/eventqa_memgen_recent_window_bmax_full_aggregate.md")
    args = parser.parse_args(argv)
    if len(args.artifact) != 5:
        raise RecentWindowAggregateError("exactly five --artifact paths are required")
    summary = aggregate_artifacts([json.loads(Path(path).read_text(encoding="utf-8")) for path in args.artifact])
    output_json, output_md = Path(args.output_json), Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_md.write_text(_markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
