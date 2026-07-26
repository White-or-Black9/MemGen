"""Aggregate five complete-pass EventQA diagnostic-control estimates per method."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "eventqa-explicit-controls-repeat-aggregate/v1"
METHODS = ("text_summary", "bm25_top2", "matched16")


class ExplicitControlsRepeatAggregateError(ValueError):
    """Raised when complete-pass control aggregates are not comparable."""


def _finite(value: Any, *, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ExplicitControlsRepeatAggregateError(f"{label} must be finite")
    return float(value)


def _summarize(values: list[float]) -> dict[str, float]:
    if len(values) != 5:
        raise ExplicitControlsRepeatAggregateError("exactly five complete passes are required")
    return {"mean": statistics.fmean(values), "std": statistics.stdev(values)}


def aggregate_method(paths: list[Path], *, method_id: str) -> dict[str, Any]:
    if len(paths) != 5:
        raise ExplicitControlsRepeatAggregateError(f"{method_id} requires five aggregates")
    runs: list[dict[str, Any]] = []
    for repeat_index, path in enumerate(paths, start=1):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        scope = artifact.get("scope", {})
        if scope.get("question_count") != 500 or scope.get("context_indices") != list(range(5)):
            raise ExplicitControlsRepeatAggregateError(f"{path} is not a complete 500-question pass")
        effect = artifact.get("effectiveness", {})
        runs.append(
            {
                "repeat_index": repeat_index,
                "source_aggregate": str(path),
                "substring_exact_match": _finite(effect.get("substring_exact_match"), label="EM"),
                "eventqa_recall": _finite(effect.get("eventqa_recall"), label="recall"),
                "format_failure_count": _finite(effect.get("format_failure_count"), label="format failures"),
            }
        )
    return {
        "method_id": method_id,
        "repeat_count": 5,
        "repeat_protocol": "five complete process-level passes with seed=42; repeats 2-5 may share GPUs and are effectiveness-only",
        "metrics": {
            "em": _summarize([run["substring_exact_match"] for run in runs]),
            "recall": _summarize([run["eventqa_recall"] for run in runs]),
            "format_failures": _summarize([run["format_failure_count"] for run in runs]),
        },
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-summary", action="append", required=True)
    parser.add_argument("--bm25", action="append", required=True)
    parser.add_argument("--matched16", action="append", required=True)
    parser.add_argument(
        "--output-json",
        default="outputs/mab/eventqa_explicit_controls_repeat_aggregate.json",
    )
    parser.add_argument(
        "--output-md",
        default="outputs/mab/eventqa_explicit_controls_repeat_aggregate.md",
    )
    args = parser.parse_args(argv)
    methods = [
        aggregate_method([Path(path) for path in args.text_summary], method_id="text_summary"),
        aggregate_method([Path(path) for path in args.bm25], method_id="bm25_top2"),
        aggregate_method([Path(path) for path in args.matched16], method_id="matched16"),
    ]
    summary = {"schema_version": SCHEMA_VERSION, "methods": methods}
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# EventQA Explicit-Control Repeat Aggregate", "", "| Method | Passes | EM | Recall | Format failures |", "|---|---:|---:|---:|---:|"]
    for method in methods:
        metrics = method["metrics"]
        lines.append(
            f"| {method['method_id']} | 5 | {metrics['em']['mean']:.3f}±{metrics['em']['std']:.3f} | "
            f"{metrics['recall']['mean']:.3f}±{metrics['recall']['std']:.3f} | "
            f"{metrics['format_failures']['mean']:.1f}±{metrics['format_failures']['std']:.1f} |"
        )
    lines += ["", "Effectiveness-only repeated estimates; cost is taken only from the controlled serialized measurement artifacts.", ""]
    output_md.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
