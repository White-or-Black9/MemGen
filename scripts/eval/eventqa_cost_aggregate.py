"""Aggregate validated full EventQA method-separable cost artifacts offline."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.eventqa_method_separable_cost import validate_cost_summary


SCHEMA_VERSION = "eventqa-full-cost-aggregate/v1"
METHODS = ("disabled", "p7")
CONTEXTS = set(range(5))


class CostAggregationError(ValueError):
    """Raised when full cost artifacts are incomplete or non-comparable."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CostAggregationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CostAggregationError(f"expected JSON object in {path}")
    return value


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load(path: Path) -> dict[str, Any]:
    summary = _read_json(path)
    try:
        validate_cost_summary(summary)
    except Exception as exc:
        raise CostAggregationError(f"invalid cost summary {path}: {exc}") from exc
    if summary["scope"].get("measurement_scope") != "full":
        raise CostAggregationError(f"expected full measurement scope in {path}")
    manifest_path = path.with_name("manifest.json")
    manifest = _read_json(manifest_path)
    gpu = manifest.get("gpu")
    if not isinstance(gpu, dict):
        raise CostAggregationError(f"missing GPU metadata in {manifest_path}")
    return {"path": path, "summary": summary, "manifest": manifest, "gpu": gpu}


def _aggregate_method(items: list[dict[str, Any]]) -> dict[str, Any]:
    query_values = [
        float(value)
        for item in items
        for value in item["summary"]["cost"]["query_latency_seconds"]["values"]
    ]
    output_values = [
        float(value)
        for item in items
        for value in item["summary"]["cost"]["output_tokens"]["values"]
    ]
    construction = [
        float(item["summary"]["cost"]["construction_latency_seconds"])
        for item in items
    ]
    end_to_end = [
        float(item["summary"]["cost"]["end_to_end_latency_seconds"])
        for item in items
    ]
    incremental_peak = [
        int(item["summary"]["cost"]["incremental_peak_gpu_memory_bytes"])
        for item in items
    ]
    effectiveness_em = [
        float(item["summary"]["effectiveness"]["substring_exact_match"])
        for item in items
    ]
    effectiveness_recall = [
        float(item["summary"]["effectiveness"]["eventqa_recall"])
        for item in items
    ]
    return {
        "context_count": len(items),
        "question_count": len(query_values),
        "construction_latency_seconds": _stats(construction),
        "construction_latency_seconds_total": sum(construction),
        "query_latency_seconds": _stats(query_values),
        "query_latency_seconds_total": sum(query_values),
        "end_to_end_latency_seconds": _stats(end_to_end),
        "end_to_end_latency_seconds_total": sum(end_to_end),
        "amortized_end_to_end_seconds_per_question": sum(end_to_end) / len(query_values),
        "incremental_peak_gpu_memory_bytes": _stats(
            [float(value) for value in incremental_peak]
        ),
        "incremental_peak_gpu_memory_bytes_max": max(incremental_peak),
        "output_tokens": _stats(output_values),
        "effectiveness": {
            "substring_exact_match": statistics.fmean(effectiveness_em),
            "eventqa_recall": statistics.fmean(effectiveness_recall),
        },
        "artifact_paths": [_relative(item["path"]) for item in items],
    }


def aggregate_cost_artifacts(paths: Iterable[str | Path]) -> dict[str, Any]:
    items = [_load(Path(path)) for path in paths]
    if len(items) != 10:
        raise CostAggregationError(
            f"expected both methods across contexts 0-4 (10 artifacts), found {len(items)}"
        )
    grouped: dict[str, dict[int, dict[str, Any]]] = {method: {} for method in METHODS}
    gpu_keys = set()
    for item in items:
        summary = item["summary"]
        method = summary.get("method")
        context = summary["scope"].get("context_index")
        if method not in grouped:
            raise CostAggregationError(f"unexpected method: {method}")
        if context in grouped[method]:
            raise CostAggregationError(f"duplicate {method} context {context}")
        grouped[method][context] = item
        gpu_keys.add(
            (item["gpu"].get("cuda_visible_devices"), item["gpu"].get("name"))
        )
    for method in METHODS:
        if set(grouped[method]) != CONTEXTS:
            raise CostAggregationError(
                f"{method} contexts must be 0-4, found {sorted(grouped[method])}"
            )
    if len(gpu_keys) != 1:
        raise CostAggregationError(f"artifacts used different GPU identities: {gpu_keys}")

    # Protocol and generation settings must match. P7-only bank fields are
    # intentionally excluded from cross-method equality.
    shared_configs = set()
    p7_configs = set()
    for method in METHODS:
        for item in grouped[method].values():
            config = item["summary"]["method_config"]
            shared_configs.add(
                (config.get("generation_max_length"), config.get("eventqa_protocol"))
            )
            if method == "p7":
                p7_configs.add(
                    (
                        config.get("retrieve_threshold"),
                        config.get("update_threshold"),
                        config.get("max_slots"),
                        config.get("top_k"),
                        config.get("decay_alpha"),
                    )
                )
                invariants = item["summary"]["invariants"]
                if not invariants.get("all_query_writes_zero"):
                    raise CostAggregationError("P7 query-write invariant failed")
                if not invariants.get("all_bank_snapshots_unchanged"):
                    raise CostAggregationError("P7 bank-snapshot invariant failed")
    if len(shared_configs) != 1:
        raise CostAggregationError(f"shared protocol config mismatch: {shared_configs}")
    if p7_configs != {(0.05, 0.1, 16, 2, 0.05)}:
        raise CostAggregationError(f"unexpected P7 config: {p7_configs}")

    methods = {
        method: _aggregate_method([grouped[method][index] for index in range(5)])
        for method in METHODS
    }
    disabled = methods["disabled"]
    p7 = methods["p7"]
    per_context = {}
    for context_index in range(5):
        per_context[str(context_index)] = {}
        for method in METHODS:
            summary = grouped[method][context_index]["summary"]
            per_context[str(context_index)][method] = {
                "construction_latency_seconds": summary["cost"][
                    "construction_latency_seconds"
                ],
                "query_latency_seconds": summary["cost"]["query_latency_seconds"],
                "end_to_end_latency_seconds": summary["cost"][
                    "end_to_end_latency_seconds"
                ],
                "incremental_peak_gpu_memory_bytes": summary["cost"][
                    "incremental_peak_gpu_memory_bytes"
                ],
                "effectiveness": summary["effectiveness"],
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "measurement_mode": "standalone_process_same_gpu_serial",
        "comparability": {
            "verdict": "comparable",
            "gpu": {
                "cuda_visible_devices": next(iter(gpu_keys))[0],
                "name": next(iter(gpu_keys))[1],
            },
            "shared_protocol": {
                "generation_max_length": next(iter(shared_configs))[0],
                "eventqa_protocol": next(iter(shared_configs))[1],
            },
            "p7_config": {
                "retrieve_threshold": 0.05,
                "update_threshold": 0.1,
                "max_slots": 16,
                "top_k": 2,
                "decay_alpha": 0.05,
            },
        },
        "methods": methods,
        "comparison": {
            "construction_latency_delta_seconds": (
                p7["construction_latency_seconds_total"]
                - disabled["construction_latency_seconds_total"]
            ),
            "query_latency_delta_seconds": (
                p7["query_latency_seconds"]["mean"]
                - disabled["query_latency_seconds"]["mean"]
            ),
            "end_to_end_latency_delta_seconds": (
                p7["end_to_end_latency_seconds_total"]
                - disabled["end_to_end_latency_seconds_total"]
            ),
            "end_to_end_latency_ratio": (
                p7["end_to_end_latency_seconds_total"]
                / disabled["end_to_end_latency_seconds_total"]
            ),
            "amortized_seconds_per_question_delta": (
                p7["amortized_end_to_end_seconds_per_question"]
                - disabled["amortized_end_to_end_seconds_per_question"]
            ),
            "incremental_peak_delta_bytes": (
                p7["incremental_peak_gpu_memory_bytes_max"]
                - disabled["incremental_peak_gpu_memory_bytes_max"]
            ),
        },
        "per_context": per_context,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# EventQA Full Method-Separable Cost Aggregate",
        "",
        "| Method | Construction total (s) | Query mean ± std (s) | End-to-end total (s) | Amortized/query (s) | Incremental peak max (MiB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = payload["methods"][method]
        lines.append(
            f"| {method} | {row['construction_latency_seconds_total']:.3f} | "
            f"{row['query_latency_seconds']['mean']:.3f} ± {row['query_latency_seconds']['std']:.3f} | "
            f"{row['end_to_end_latency_seconds_total']:.3f} | "
            f"{row['amortized_end_to_end_seconds_per_question']:.3f} | "
            f"{row['incremental_peak_gpu_memory_bytes_max'] / (1024 ** 2):.1f} |"
        )
    lines.extend(
        [
            "",
            f"P7/Disabled end-to-end ratio: `{payload['comparison']['end_to_end_latency_ratio']:.3f}`.",
            "",
            "Model loading is excluded. This is a same-GPU serialized inference-cost",
            "measurement, not a repeated-load benchmark and not a throughput claim.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = sorted(Path(args.input_root).glob("*/*/cost_summary.json"))
    payload = aggregate_cost_artifacts(paths)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
