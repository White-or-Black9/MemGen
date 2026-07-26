"""Build the paper-facing unified EventQA final comparison package."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCHEMA_VERSION = "eventqa-final-table-package/v1"


class FinalTablePackageError(ValueError):
    """Raised when the unified EventQA package inputs are incomplete."""


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require(mapping: dict[str, Any], key: str, *, context: str) -> Any:
    if key not in mapping:
        raise FinalTablePackageError(f"missing key in {context}: {key}")
    return mapping[key]


def _method_by_id(paper_aggregate: dict[str, Any], method_id: str) -> dict[str, Any]:
    for method in paper_aggregate.get("methods", []):
        if method.get("method_id") == method_id:
            return method
    raise FinalTablePackageError(f"missing method in paper aggregate: {method_id}")


def _repeat_method(repeat_aggregate: dict[str, Any], method_id: str) -> dict[str, Any]:
    if repeat_aggregate.get("schema_version") != "eventqa-explicit-controls-repeat-aggregate/v1":
        raise FinalTablePackageError("unexpected explicit-controls repeat aggregate schema")
    for method in repeat_aggregate.get("methods", []):
        if method.get("method_id") == method_id:
            if method.get("repeat_count") != 5:
                raise FinalTablePackageError(f"{method_id} repeat aggregate must contain five passes")
            return method
    raise FinalTablePackageError(f"missing method in explicit-controls repeat aggregate: {method_id}")


def _main_row(
    *,
    method_id: str,
    display_name: str,
    representation: str,
    retrieval_form: str,
    repeat_count: int,
    em_mean: float,
    em_std: float | None,
    recall_mean: float,
    recall_std: float | None,
    format_failures: float,
    format_std: float | None,
    protocol_notes: str,
) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "display_name": display_name,
        "representation": representation,
        "retrieval_form": retrieval_form,
        "repeat_count": repeat_count,
        "em_mean": em_mean,
        "em_std": em_std,
        "recall_mean": recall_mean,
        "recall_std": recall_std,
        "format_failures_mean": format_failures,
        "format_failures_std": format_std,
        "protocol_notes": protocol_notes,
    }


def build_package(
    *,
    paper_aggregate: dict[str, Any],
    cost_aggregate: dict[str, Any],
    bm25_aggregate: dict[str, Any],
    matched16_aggregate: dict[str, Any],
    text_summary_aggregate: dict[str, Any],
    explicit_controls_repeat_aggregate: dict[str, Any],
    no_query_aggregate: dict[str, Any],
) -> dict[str, Any]:
    bank_off = _method_by_id(paper_aggregate, "bank_off")
    p6 = _method_by_id(paper_aggregate, "p6")
    p7 = _method_by_id(paper_aggregate, "p7")
    text_summary_repeat = _repeat_method(explicit_controls_repeat_aggregate, "text_summary")
    bm25_repeat = _repeat_method(explicit_controls_repeat_aggregate, "bm25_top2")
    matched16_repeat = _repeat_method(explicit_controls_repeat_aggregate, "matched16")

    disabled_cost = cost_aggregate.get("methods", {}).get("disabled")
    p7_cost = cost_aggregate.get("methods", {}).get("p7")
    if disabled_cost is None or p7_cost is None:
        raise FinalTablePackageError("cost aggregate missing method rows")

    text_summary_cost = _require(text_summary_aggregate, "cost", context="text_summary aggregate")
    bm25_cost = _require(bm25_aggregate, "cost", context="bm25 aggregate")
    matched16_cost = _require(matched16_aggregate, "cost", context="matched16 aggregate")
    no_query_cost = _require(no_query_aggregate, "cost", context="no_query aggregate")

    main_table = [
        _main_row(
            method_id="bank_off",
            display_name=bank_off["display_name"],
            representation="none",
            retrieval_form="none",
            repeat_count=int(bank_off["repeat_count"]),
            em_mean=float(bank_off["metrics"]["em"]["mean"]),
            em_std=float(bank_off["metrics"]["em"]["std"]),
            recall_mean=float(bank_off["metrics"]["recall"]["mean"]),
            recall_std=float(bank_off["metrics"]["recall"]["std"]),
            format_failures=float(bank_off["metrics"]["format_failures"]["mean"]),
            format_std=float(bank_off["metrics"]["format_failures"]["std"]),
            protocol_notes="five-repeat Bank-off row reconstructed from frozen P7 paired artifacts",
        ),
        _main_row(
            method_id="text_summary",
            display_name="Same-model text-summary memory",
            representation="persistent summary text",
            retrieval_form="full summary injection",
            repeat_count=int(text_summary_repeat["repeat_count"]),
            em_mean=float(text_summary_repeat["metrics"]["em"]["mean"]),
            em_std=float(text_summary_repeat["metrics"]["em"]["std"]),
            recall_mean=float(text_summary_repeat["metrics"]["recall"]["mean"]),
            recall_std=float(text_summary_repeat["metrics"]["recall"]["std"]),
            format_failures=float(text_summary_repeat["metrics"]["format_failures"]["mean"]),
            format_std=float(text_summary_repeat["metrics"]["format_failures"]["std"]),
            protocol_notes="five complete process-level passes (seed=42); negative same-model baseline",
        ),
        _main_row(
            method_id="bm25_top2",
            display_name="BM25 top-2 retrieved text",
            representation="explicit retrieved text",
            retrieval_form="deterministic BM25 top-2",
            repeat_count=int(bm25_repeat["repeat_count"]),
            em_mean=float(bm25_repeat["metrics"]["em"]["mean"]),
            em_std=float(bm25_repeat["metrics"]["em"]["std"]),
            recall_mean=float(bm25_repeat["metrics"]["recall"]["mean"]),
            recall_std=float(bm25_repeat["metrics"]["recall"]["std"]),
            format_failures=float(bm25_repeat["metrics"]["format_failures"]["mean"]),
            format_std=float(bm25_repeat["metrics"]["format_failures"]["std"]),
            protocol_notes="five complete process-level passes (seed=42)",
        ),
        _main_row(
            method_id="matched16",
            display_name="16-token matched-budget retrieved text",
            representation="explicit retrieved text",
            retrieval_form="deterministic 16-token matched injection",
            repeat_count=int(matched16_repeat["repeat_count"]),
            em_mean=float(matched16_repeat["metrics"]["em"]["mean"]),
            em_std=float(matched16_repeat["metrics"]["em"]["std"]),
            recall_mean=float(matched16_repeat["metrics"]["recall"]["mean"]),
            recall_std=float(matched16_repeat["metrics"]["recall"]["std"]),
            format_failures=float(matched16_repeat["metrics"]["format_failures"]["mean"]),
            format_std=float(matched16_repeat["metrics"]["format_failures"]["std"]),
            protocol_notes="five complete process-level passes (seed=42); exact 16-token visible budget",
        ),
        _main_row(
            method_id="p6",
            display_name=p6["display_name"],
            representation="latent bank",
            retrieval_form="query-time latent retrieval",
            repeat_count=int(p6["repeat_count"]),
            em_mean=float(p6["metrics"]["em"]["mean"]),
            em_std=float(p6["metrics"]["em"]["std"]),
            recall_mean=float(p6["metrics"]["recall"]["mean"]),
            recall_std=float(p6["metrics"]["recall"]["std"]),
            format_failures=float(p6["metrics"]["format_failures"]["mean"]),
            format_std=float(p6["metrics"]["format_failures"]["std"]),
            protocol_notes="five-repeat lower-update-threshold comparator",
        ),
        _main_row(
            method_id="p7_no_query_retrieval",
            display_name="P7 with query retrieval disabled",
            representation="latent bank",
            retrieval_form="construction only; retrieval disabled at QA",
            repeat_count=1,
            em_mean=float(no_query_aggregate["effectiveness"]["substring_exact_match"]),
            em_std=None,
            recall_mean=float(no_query_aggregate["effectiveness"]["eventqa_recall"]),
            recall_std=None,
            format_failures=float(no_query_aggregate["effectiveness"]["format_failure_count"]),
            format_std=None,
            protocol_notes="one complete pass; all query retrieval disabled",
        ),
        _main_row(
            method_id="p7",
            display_name=p7["display_name"],
            representation="latent bank",
            retrieval_form="query-time latent retrieval",
            repeat_count=int(p7["repeat_count"]),
            em_mean=float(p7["metrics"]["em"]["mean"]),
            em_std=float(p7["metrics"]["em"]["std"]),
            recall_mean=float(p7["metrics"]["recall"]["mean"]),
            recall_std=float(p7["metrics"]["recall"]["std"]),
            format_failures=float(p7["metrics"]["format_failures"]["mean"]),
            format_std=float(p7["metrics"]["format_failures"]["std"]),
            protocol_notes="five-repeat main result",
        ),
    ]

    explicit_controls = [
        row
        for row in main_table
        if row["method_id"] in {"bank_off", "text_summary", "bm25_top2", "matched16", "p7_no_query_retrieval", "p7"}
    ]

    cost_table = [
        {
            "method_id": "bank_off",
            "display_name": bank_off["display_name"],
            "construction_seconds_total": float(disabled_cost["construction_latency_seconds_total"]),
            "end_to_end_seconds_total": float(disabled_cost["end_to_end_latency_seconds_total"]),
            "amortized_seconds_per_question": float(disabled_cost["amortized_end_to_end_seconds_per_question"]),
            "incremental_peak_gpu_memory_bytes_max": int(disabled_cost["incremental_peak_gpu_memory_bytes_max"]),
            "paper_facing_cost": True,
            "cost_notes": "method-separable same-GPU serialized full pass",
        },
        {
            "method_id": "text_summary",
            "display_name": "Same-model text-summary memory",
            "construction_seconds_total": float(_require(text_summary_cost, "construction_latency_seconds", context="text_summary cost")),
            "end_to_end_seconds_total": float(_require(text_summary_cost, "end_to_end_latency_seconds", context="text_summary cost")),
            "amortized_seconds_per_question": float(_require(text_summary_cost, "end_to_end_amortized_seconds_per_question", context="text_summary cost")),
            "incremental_peak_gpu_memory_bytes_max": max(
                int(_require(text_summary_cost, "construction_incremental_peak_gpu_memory_bytes_max", context="text_summary construction cost")),
                int(_require(text_summary_cost, "query_incremental_peak_gpu_memory_bytes_max", context="text_summary query cost")),
            ),
            "paper_facing_cost": bool(_require(text_summary_cost, "paper_facing", context="text_summary cost")),
            "cost_notes": str(_require(text_summary_cost, "caveat", context="text_summary cost")),
        },
        {
            "method_id": "bm25_top2",
            "display_name": "BM25 top-2 retrieved text",
            "construction_seconds_total": float(_require(bm25_cost, "index_construction_latency_seconds", context="bm25 cost") + _require(bm25_cost, "retrieval_latency_seconds", context="bm25 cost")),
            "end_to_end_seconds_total": float(_require(bm25_cost, "method_total_seconds", context="bm25 cost")),
            "amortized_seconds_per_question": float(_require(bm25_cost, "amortized_seconds_per_question", context="bm25 cost")),
            "incremental_peak_gpu_memory_bytes_max": int(_require(bm25_cost, "incremental_peak_gpu_memory_bytes_max", context="bm25 cost")),
            "paper_facing_cost": True,
            "cost_notes": "one complete pass",
        },
        {
            "method_id": "matched16",
            "display_name": "16-token matched-budget retrieved text",
            "construction_seconds_total": float(_require(matched16_cost, "index_construction_latency_seconds", context="matched16 cost") + _require(matched16_cost, "retrieval_and_window_latency_seconds", context="matched16 cost")),
            "end_to_end_seconds_total": float(_require(matched16_cost, "method_total_seconds", context="matched16 cost")),
            "amortized_seconds_per_question": float(_require(matched16_cost, "amortized_seconds_per_question", context="matched16 cost")),
            "incremental_peak_gpu_memory_bytes_max": int(_require(matched16_cost, "incremental_peak_gpu_memory_bytes_max", context="matched16 cost")),
            "paper_facing_cost": True,
            "cost_notes": "one complete pass",
        },
        {
            "method_id": "p7_no_query_retrieval",
            "display_name": "P7 with query retrieval disabled",
            "construction_seconds_total": float(_require(no_query_cost, "construction_latency_seconds", context="no_query cost")),
            "end_to_end_seconds_total": float(_require(no_query_cost, "end_to_end_latency_seconds", context="no_query cost")),
            "amortized_seconds_per_question": float(_require(no_query_cost, "end_to_end_amortized_seconds_per_question", context="no_query cost")),
            "incremental_peak_gpu_memory_bytes_max": int(_require(no_query_cost, "incremental_peak_gpu_memory_bytes_max", context="no_query cost")),
            "paper_facing_cost": True,
            "cost_notes": "one complete pass",
        },
        {
            "method_id": "p7",
            "display_name": p7["display_name"],
            "construction_seconds_total": float(p7_cost["construction_latency_seconds_total"]),
            "end_to_end_seconds_total": float(p7_cost["end_to_end_latency_seconds_total"]),
            "amortized_seconds_per_question": float(p7_cost["amortized_end_to_end_seconds_per_question"]),
            "incremental_peak_gpu_memory_bytes_max": int(p7_cost["incremental_peak_gpu_memory_bytes_max"]),
            "paper_facing_cost": True,
            "cost_notes": "method-separable same-GPU serialized full pass",
        },
    ]

    p7_em = float(p7["metrics"]["em"]["mean"])
    p7_recall = float(p7["metrics"]["recall"]["mean"])
    p6_em = float(p6["metrics"]["em"]["mean"])
    p6_recall = float(p6["metrics"]["recall"]["mean"])
    bank_off_em = float(bank_off["metrics"]["em"]["mean"])
    bank_off_recall = float(bank_off["metrics"]["recall"]["mean"])

    claim_audit = {
        "claims": {
            "p7_vs_disabled": {
                "supported": p7_em > bank_off_em and p7_recall > bank_off_recall,
                "evidence": f"P7 EM/recall {p7_em:.4f}/{p7_recall:.4f} vs Disabled {bank_off_em:.4f}/{bank_off_recall:.4f}",
            },
            "p7_vs_p6": {
                "supported": p7_em > p6_em and float(p7["metrics"]["format_failures"]["mean"]) < float(p6["metrics"]["format_failures"]["mean"]) and abs(p7_recall - p6_recall) <= 0.01,
                "evidence": f"P7-P6 deltas: EM {p7_em - p6_em:+.4f}, recall {p7_recall - p6_recall:+.4f}, format failures {float(p7['metrics']['format_failures']['mean']) - float(p6['metrics']['format_failures']['mean']):+.1f}",
            },
            "p7_beats_explicit_controls": {
                "supported": all(
                    p7_em > row["em_mean"] and p7_recall > row["recall_mean"]
                    for row in explicit_controls
                    if row["method_id"] in {"text_summary", "bm25_top2", "matched16", "p7_no_query_retrieval"}
                ),
                "evidence": "P7 exceeds text-summary, BM25 top-2, matched16, and no-query-retrieval on both EM and recall.",
            },
            "query_time_retrieval_is_necessary": {
                "supported": bool(no_query_aggregate["invariants"]["all_queries_disable_retrieval"])
                and math.isclose(float(no_query_aggregate["effectiveness"]["substring_exact_match"]), bank_off_em)
                and math.isclose(float(no_query_aggregate["effectiveness"]["eventqa_recall"]), bank_off_recall),
                "evidence": "No-query-retrieval exactly matches Disabled effectiveness while preserving the constructed bank.",
            },
            "p7_cost_superiority": {
                "supported": False,
                "evidence": "P7 adds measured overhead over Disabled, while explicit-text baselines use different cost profiles; no blanket cost-superiority claim is supported.",
            },
        }
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "main_table": main_table,
        "explicit_controls": explicit_controls,
        "cost_table": cost_table,
        "claim_audit": claim_audit,
    }


def _markdown(package: dict[str, Any]) -> str:
    lines = [
        "# EventQA Final Comparison Package",
        "",
        "## Main Table",
        "",
        "| Method | Repeats | EM | Recall | Format failures | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in package["main_table"]:
        em = f"{row['em_mean']:.3f}" + (f"±{row['em_std']:.3f}" if row["em_std"] is not None else "")
        recall = f"{row['recall_mean']:.3f}" + (f"±{row['recall_std']:.3f}" if row["recall_std"] is not None else "")
        ff = f"{row['format_failures_mean']:.1f}" + (f"±{row['format_failures_std']:.1f}" if row["format_failures_std"] is not None else "")
        lines.append(f"| {row['display_name']} | {row['repeat_count']} | {em} | {recall} | {ff} | {row['protocol_notes']} |")
    lines += [
        "",
        "## Cost Table",
        "",
        "| Method | End-to-end s | s/question | Peak GPU bytes | Paper-facing | Notes |",
        "|---|---:|---:|---:|:---:|---|",
    ]
    for row in package["cost_table"]:
        lines.append(
            f"| {row['display_name']} | {row['end_to_end_seconds_total']:.3f} | "
            f"{row['amortized_seconds_per_question']:.3f} | {row['incremental_peak_gpu_memory_bytes_max']} | "
            f"{'yes' if row['paper_facing_cost'] else 'no'} | {row['cost_notes']} |"
        )
    lines += [
        "",
        "## Claim Audit",
        "",
    ]
    for claim_id, claim in package["claim_audit"]["claims"].items():
        lines.append(f"- `{claim_id}`: {'supported' if claim['supported'] else 'not supported'}; {claim['evidence']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-aggregate", default="outputs/mab/eventqa_paper_aggregate.json")
    parser.add_argument("--cost-aggregate", default="outputs/mab/eventqa_method_separable_cost_full_aggregate.json")
    parser.add_argument("--bm25-aggregate", default="outputs/mab/eventqa_bm25_top2_full_aggregate.json")
    parser.add_argument("--matched16-aggregate", default="outputs/mab/eventqa_matched16_full_aggregate.json")
    parser.add_argument("--text-summary-aggregate", default="outputs/mab/eventqa_text_summary_full_aggregate.json")
    parser.add_argument(
        "--explicit-controls-repeat-aggregate",
        default="outputs/mab/eventqa_explicit_controls_repeat_aggregate.json",
    )
    parser.add_argument("--no-query-aggregate", default="outputs/mab/eventqa_p7_no_query_retrieval_full_aggregate.json")
    parser.add_argument("--output-json", default="outputs/mab/eventqa_final_comparison_package.json")
    parser.add_argument("--output-md", default="outputs/mab/eventqa_final_comparison_package.md")
    args = parser.parse_args(argv)

    package = build_package(
        paper_aggregate=_load(args.paper_aggregate),
        cost_aggregate=_load(args.cost_aggregate),
        bm25_aggregate=_load(args.bm25_aggregate),
        matched16_aggregate=_load(args.matched16_aggregate),
        text_summary_aggregate=_load(args.text_summary_aggregate),
        explicit_controls_repeat_aggregate=_load(args.explicit_controls_repeat_aggregate),
        no_query_aggregate=_load(args.no_query_aggregate),
    )
    Path(args.output_json).write_text(
        json.dumps(package, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    Path(args.output_md).write_text(_markdown(package), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
