"""Build paper-facing EventQA analysis tables from the authoritative aggregate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "eventqa-analysis-tables/v1"
QUESTION_COUNT = 500


class EventQAAnalysisTablesError(ValueError):
    """Raised when the required aggregate fields are missing."""


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require(mapping: dict[str, Any], key: str, *, context: str) -> Any:
    if key not in mapping:
        raise EventQAAnalysisTablesError(f"missing key in {context}: {key}")
    return mapping[key]


def _method_by_id(paper_aggregate: dict[str, Any], method_id: str) -> dict[str, Any]:
    for method in _require(paper_aggregate, "methods", context="paper aggregate"):
        if method.get("method_id") == method_id:
            return method
    raise EventQAAnalysisTablesError(f"missing method in paper aggregate: {method_id}")


def _round4(value: float) -> float:
    return round(float(value), 4)


def _round1(value: float) -> float:
    return round(float(value), 1)


def _format_mean_std(mean: float, std: float, *, digits: int) -> str:
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def _context_row(
    *,
    context_id: str,
    bank_off: dict[str, Any],
    p6: dict[str, Any],
    p7: dict[str, Any],
) -> dict[str, Any]:
    bank_off_ctx = _require(_require(bank_off, "per_context", context="bank_off"), context_id, context=f"bank_off per_context")
    p6_ctx = _require(_require(p6, "per_context", context="p6"), context_id, context=f"p6 per_context")
    p7_ctx = _require(_require(p7, "per_context", context="p7"), context_id, context=f"p7 per_context")

    bank_off_em = float(_require(_require(bank_off_ctx, "em", context=f"bank_off ctx{context_id}"), "mean", context=f"bank_off ctx{context_id} em"))
    bank_off_recall = float(_require(_require(bank_off_ctx, "recall", context=f"bank_off ctx{context_id}"), "mean", context=f"bank_off ctx{context_id} recall"))
    p6_em = float(_require(_require(p6_ctx, "em", context=f"p6 ctx{context_id}"), "mean", context=f"p6 ctx{context_id} em"))
    p6_recall = float(_require(_require(p6_ctx, "recall", context=f"p6 ctx{context_id}"), "mean", context=f"p6 ctx{context_id} recall"))
    p7_em = float(_require(_require(p7_ctx, "em", context=f"p7 ctx{context_id}"), "mean", context=f"p7 ctx{context_id} em"))
    p7_recall = float(_require(_require(p7_ctx, "recall", context=f"p7 ctx{context_id}"), "mean", context=f"p7 ctx{context_id} recall"))
    p7_format = float(_require(_require(p7_ctx, "format_failures", context=f"p7 ctx{context_id}"), "mean", context=f"p7 ctx{context_id} format"))

    return {
        "context_id": f"ctx{context_id}",
        "bank_off_em": _round4(bank_off_em),
        "bank_off_recall": _round4(bank_off_recall),
        "p6_em_mean": _round4(p6_em),
        "p6_em_std": _round4(float(_require(_require(p6_ctx, "em", context=f"p6 ctx{context_id}"), "std", context=f"p6 ctx{context_id} em"))),
        "p6_recall_mean": _round4(p6_recall),
        "p6_recall_std": _round4(float(_require(_require(p6_ctx, "recall", context=f"p6 ctx{context_id}"), "std", context=f"p6 ctx{context_id} recall"))),
        "p7_em_mean": _round4(p7_em),
        "p7_em_std": _round4(float(_require(_require(p7_ctx, "em", context=f"p7 ctx{context_id}"), "std", context=f"p7 ctx{context_id} em"))),
        "p7_recall_mean": _round4(p7_recall),
        "p7_recall_std": _round4(float(_require(_require(p7_ctx, "recall", context=f"p7 ctx{context_id}"), "std", context=f"p7 ctx{context_id} recall"))),
        "p7_format_failures_mean": _round1(p7_format),
        "p7_format_failures_std": _round1(float(_require(_require(p7_ctx, "format_failures", context=f"p7 ctx{context_id}"), "std", context=f"p7 ctx{context_id} format"))),
        "p7_minus_bank_off_em": _round4(p7_em - bank_off_em),
        "p7_minus_bank_off_recall": _round4(p7_recall - bank_off_recall),
        "p7_minus_p6_em": _round4(p7_em - p6_em),
        "p7_minus_p6_recall": _round4(p7_recall - p6_recall),
    }


def _transition_row(*, method: dict[str, Any]) -> dict[str, Any]:
    metrics = _require(method, "metrics", context=method["method_id"])
    helpful = _require(_require(metrics, "helpful_memory_count", context=method["method_id"]), "values", context=f"{method['method_id']} helpful")
    harmful = _require(_require(metrics, "harmful_memory_count", context=method["method_id"]), "values", context=f"{method['method_id']} harmful")
    format_harm = _require(_require(metrics, "format_harm_count", context=method["method_id"]), "values", context=f"{method['method_id']} format_harm")
    unchanged = [QUESTION_COUNT - float(h) - float(r) for h, r in zip(helpful, harmful)]
    net = [float(h) - float(r) for h, r in zip(helpful, harmful)]

    helpful_mean = float(_require(_require(metrics, "helpful_memory_count", context=method["method_id"]), "mean", context=f"{method['method_id']} helpful"))
    helpful_std = float(_require(_require(metrics, "helpful_memory_count", context=method["method_id"]), "std", context=f"{method['method_id']} helpful"))
    harmful_mean = float(_require(_require(metrics, "harmful_memory_count", context=method["method_id"]), "mean", context=f"{method['method_id']} harmful"))
    harmful_std = float(_require(_require(metrics, "harmful_memory_count", context=method["method_id"]), "std", context=f"{method['method_id']} harmful"))
    format_harm_mean = float(_require(_require(metrics, "format_harm_count", context=method["method_id"]), "mean", context=f"{method['method_id']} format_harm"))
    format_harm_std = float(_require(_require(metrics, "format_harm_count", context=method["method_id"]), "std", context=f"{method['method_id']} format_harm"))

    unchanged_mean = sum(unchanged) / len(unchanged)
    unchanged_std = (sum((x - unchanged_mean) ** 2 for x in unchanged) / len(unchanged)) ** 0.5
    net_mean = sum(net) / len(net)
    net_std = (sum((x - net_mean) ** 2 for x in net) / len(net)) ** 0.5

    return {
        "method_id": method["method_id"],
        "display_name": method["display_name"],
        "repeat_count": int(method["repeat_count"]),
        "helpful_mean": _round1(helpful_mean),
        "helpful_std": _round1(helpful_std),
        "harmful_mean": _round1(harmful_mean),
        "harmful_std": _round1(harmful_std),
        "unchanged_mean": _round1(unchanged_mean),
        "unchanged_std": _round1(unchanged_std),
        "format_harm_mean": _round1(format_harm_mean),
        "format_harm_std": _round1(format_harm_std),
        "net_mean": _round1(net_mean),
        "net_std": _round1(net_std),
        "notes": "Counts are over 500 paired questions; unchanged = total - helpful - harmful; format-harm is a subset diagnostic.",
    }


def build_tables(*, paper_aggregate: dict[str, Any]) -> dict[str, Any]:
    bank_off = _method_by_id(paper_aggregate, "bank_off")
    p6 = _method_by_id(paper_aggregate, "p6")
    p7 = _method_by_id(paper_aggregate, "p7")

    context_order = ["0", "1", "2", "3", "4"]
    context_table = [_context_row(context_id=context_id, bank_off=bank_off, p6=p6, p7=p7) for context_id in context_order]
    transition_table = [_transition_row(method=method) for method in [p6, p7]]

    return {
        "schema_version": SCHEMA_VERSION,
        "contextwise_table": context_table,
        "transition_table": transition_table,
    }


def _markdown(tables: dict[str, Any]) -> str:
    lines = [
        "# EventQA Analysis Tables",
        "",
        "## Context-wise Table",
        "",
        "| Context | Bank-off EM | Bank-off Recall | P6 EM | P6 Recall | P7 EM | P7 Recall | P7 Format Failures | P7-Bank-off EM | P7-P6 EM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tables["contextwise_table"]:
        lines.append(
            f"| {row['context_id']} | {row['bank_off_em']:.3f} | {row['bank_off_recall']:.3f} | "
            f"{_format_mean_std(row['p6_em_mean'], row['p6_em_std'], digits=3)} | "
            f"{_format_mean_std(row['p6_recall_mean'], row['p6_recall_std'], digits=3)} | "
            f"{_format_mean_std(row['p7_em_mean'], row['p7_em_std'], digits=3)} | "
            f"{_format_mean_std(row['p7_recall_mean'], row['p7_recall_std'], digits=3)} | "
            f"{_format_mean_std(row['p7_format_failures_mean'], row['p7_format_failures_std'], digits=1)} | "
            f"{row['p7_minus_bank_off_em']:+.3f} | {row['p7_minus_p6_em']:+.3f} |"
        )
    lines += [
        "",
        "## Transition Table",
        "",
        "| Method | Repeats | Helpful | Harmful | Unchanged | Format-harm | Net gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tables["transition_table"]:
        lines.append(
            f"| {row['display_name']} | {row['repeat_count']} | "
            f"{_format_mean_std(row['helpful_mean'], row['helpful_std'], digits=1)} | "
            f"{_format_mean_std(row['harmful_mean'], row['harmful_std'], digits=1)} | "
            f"{_format_mean_std(row['unchanged_mean'], row['unchanged_std'], digits=1)} | "
            f"{_format_mean_std(row['format_harm_mean'], row['format_harm_std'], digits=1)} | "
            f"{_format_mean_std(row['net_mean'], row['net_std'], digits=1)} |"
        )
    lines += [
        "",
        "Format-harm is a diagnostic subset rather than an additional partition bucket.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-aggregate", default="outputs/mab/eventqa_paper_aggregate.json")
    parser.add_argument("--output-json", default="outputs/mab/eventqa_analysis_tables.json")
    parser.add_argument("--output-md", default="outputs/mab/eventqa_analysis_tables.md")
    args = parser.parse_args(argv)

    tables = build_tables(paper_aggregate=_load(args.paper_aggregate))
    Path(args.output_json).write_text(json.dumps(tables, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Path(args.output_md).write_text(_markdown(tables) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
