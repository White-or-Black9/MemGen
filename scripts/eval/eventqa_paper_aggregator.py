"""Validate and aggregate EventQA paper artifacts without running inference."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "eventqa-paper-aggregate/v1"
REQUIRED_FILES = (
    "manifest.json",
    "run_config.json",
    "eventqa_per_question.jsonl",
    "bank_transition_aggregate.json",
)
CONFIG_IDENTITY_FIELDS = (
    "subtask",
    "metric",
    "optional_metric",
    "model_checkpoint_id",
    "eventqa_protocol",
    "query_phase",
    "generation_max_length",
    "retrieve_threshold",
    "update_threshold",
    "max_slots",
    "top_k",
    "decay_alpha",
)
QUESTION_REQUIRED_FIELDS = (
    "context_index",
    "context_id",
    "qa_pair_id",
    "bank_off_substring_exact_match",
    "bank_on_substring_exact_match",
    "bank_off_eventqa_recall",
    "bank_on_eventqa_recall",
    "bank_off_format_flags",
    "bank_on_format_flags",
    "bank_on_query_turn_retrieved_indices",
    "query_write_count",
    "bank_snapshot_changed_after_query",
)


class AggregationError(ValueError):
    """Raised when an artifact cannot satisfy the paper aggregation contract."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AggregationError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AggregationError(f"expected JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AggregationError(f"cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AggregationError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise AggregationError(f"expected object at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise AggregationError(f"no question rows in {path}")
    return rows


def _ensure_fields(mapping: dict[str, Any], fields: Iterable[str], label: str) -> None:
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise AggregationError(f"{label} missing required fields: {', '.join(missing)}")


def _stats(values: list[float | int]) -> dict[str, Any]:
    normalized = [float(value) for value in values]
    mean = statistics.fmean(normalized)
    std = statistics.pstdev(normalized)
    return {"mean": mean, "std": std, "values": normalized}


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_run(run_path: str | Path) -> dict[str, Any]:
    root = Path(run_path)
    if not root.is_dir():
        raise AggregationError(f"run root is not a directory: {root}")
    for filename in REQUIRED_FILES:
        if not (root / filename).is_file():
            raise AggregationError(f"required artifact missing: {root / filename}")

    manifest = _read_json(root / "manifest.json")
    config = _read_json(root / "run_config.json")
    rows = _read_jsonl(root / "eventqa_per_question.jsonl")
    transition = _read_json(root / "bank_transition_aggregate.json")
    _ensure_fields(config, CONFIG_IDENTITY_FIELDS, f"{root}/run_config.json")
    _ensure_fields(transition, ("global", "per_context"), f"{root}/bank_transition_aggregate.json")

    for field in ("run_id", "metric", "optional_metric", "eventqa_protocol"):
        if manifest.get(field) != config.get(field):
            raise AggregationError(f"manifest/run_config mismatch for {field} in {root}")
    if config["eventqa_protocol"] != "frozen_context_bank":
        raise AggregationError(
            f"eventqa_protocol must be frozen_context_bank in {root}, got {config['eventqa_protocol']!r}"
        )
    if config["query_phase"] != "read-only":
        raise AggregationError(f"query_phase must be read-only in {root}")

    identities = []
    retrieval_ids_recorded = True
    for index, row in enumerate(rows):
        _ensure_fields(row, QUESTION_REQUIRED_FIELDS, f"{root}/eventqa_per_question.jsonl row {index}")
        if row["query_write_count"] != 0:
            raise AggregationError(f"query_write_count must be zero in {root} row {index}")
        if row["bank_snapshot_changed_after_query"] is not False:
            raise AggregationError(f"bank snapshot changed after query in {root} row {index}")
        identities.append((row["context_index"], row["context_id"], row["qa_pair_id"]))
        retrieval_ids_recorded &= isinstance(row["bank_on_query_turn_retrieved_indices"], list)
    if len(set(identities)) != len(identities):
        raise AggregationError(f"duplicate question identity within {root}")

    global_metrics = transition["global"]
    _ensure_fields(
        global_metrics,
        (
            "question_count",
            "bank_off_em",
            "bank_on_em",
            "bank_off_recall",
            "bank_on_recall",
            "bank_off_format_failures",
            "bank_on_format_failures",
            "helpful_memory_count",
            "harmful_memory_count",
            "format_harm_count",
        ),
        f"{root}/bank_transition_aggregate.json global",
    )
    if global_metrics["question_count"] != len(rows):
        raise AggregationError(f"question_count mismatch in {root}")

    return {
        "root": root,
        "manifest": manifest,
        "config": config,
        "rows": rows,
        "question_identity": identities,
        "transition": transition,
        "retrieval_ids_recorded": retrieval_ids_recorded,
    }


def _check_repeat_identity(runs: list[dict[str, Any]], method_id: str) -> None:
    reference = runs[0]
    for run in runs[1:]:
        for field in CONFIG_IDENTITY_FIELDS:
            if run["config"].get(field) != reference["config"].get(field):
                raise AggregationError(
                    f"{method_id} repeat config mismatch for {field}: "
                    f"{reference['config'].get(field)!r} != {run['config'].get(field)!r}"
                )
        if run["question_identity"] != reference["question_identity"]:
            raise AggregationError(f"{method_id} repeat question identity mismatch")


def _metric_key(mode: str, suffix: str) -> str:
    if mode not in {"bank_off", "bank_on"}:
        raise AggregationError(f"unsupported method mode: {mode!r}")
    return f"{mode}_{suffix}"


def _aggregate_method(spec: dict[str, Any]) -> dict[str, Any]:
    _ensure_fields(spec, ("method_id", "mode", "runs"), "method specification")
    if not isinstance(spec["runs"], list) or not spec["runs"]:
        raise AggregationError(f"{spec['method_id']} must define at least one run")
    runs = [_load_run(path) for path in spec["runs"]]
    _check_repeat_identity(runs, spec["method_id"])
    mode = spec["mode"]
    em_key = _metric_key(mode, "em")
    recall_key = _metric_key(mode, "recall")
    format_key = _metric_key(mode, "format_failures")

    metrics = {
        "em": _stats([run["transition"]["global"][em_key] for run in runs]),
        "recall": _stats([run["transition"]["global"][recall_key] for run in runs]),
        "format_failures": _stats(
            [run["transition"]["global"][format_key] for run in runs]
        ),
        "helpful_memory_count": _stats(
            [run["transition"]["global"]["helpful_memory_count"] for run in runs]
        ),
        "harmful_memory_count": _stats(
            [run["transition"]["global"]["harmful_memory_count"] for run in runs]
        ),
        "format_harm_count": _stats(
            [run["transition"]["global"]["format_harm_count"] for run in runs]
        ),
    }

    context_ids = sorted(runs[0]["transition"]["per_context"], key=int)
    for run in runs[1:]:
        if sorted(run["transition"]["per_context"], key=int) != context_ids:
            raise AggregationError(f"{spec['method_id']} repeat context scope mismatch")
    per_context = {}
    for context_id in context_ids:
        per_context[context_id] = {
            "question_count_per_repeat": runs[0]["transition"]["per_context"][context_id][
                "question_count"
            ],
            "em": _stats(
                [run["transition"]["per_context"][context_id][em_key] for run in runs]
            ),
            "recall": _stats(
                [run["transition"]["per_context"][context_id][recall_key] for run in runs]
            ),
            "format_failures": _stats(
                [run["transition"]["per_context"][context_id][format_key] for run in runs]
            ),
        }

    config = runs[0]["config"]
    method_config = {field: config[field] for field in CONFIG_IDENTITY_FIELDS}
    return {
        "method_id": spec["method_id"],
        "display_name": spec.get("display_name", spec["method_id"]),
        "mode": mode,
        "repeat_count": len(runs),
        "question_count_per_repeat": len(runs[0]["rows"]),
        "context_count": len(context_ids),
        "method_config": method_config,
        "scorer_contract": {
            "metric": config["metric"],
            "optional_metric": config["optional_metric"],
            "subtask": config["subtask"],
            "eventqa_protocol": config["eventqa_protocol"],
        },
        "metrics": metrics,
        "per_context": per_context,
        "query_evidence": {
            "retrieval_ids_recorded": all(run["retrieval_ids_recorded"] for run in runs),
            "injected_text_tokens": None,
            "retrieved_text_ids_recorded": False,
        },
        "cost": {
            "construction_latency_seconds": None,
            "query_latency_seconds": None,
            "end_to_end_latency_seconds": None,
            "peak_gpu_memory_bytes": None,
            "status": "missing_method_separable_measurement",
        },
        "provenance": {
            "run_ids": [run["config"]["run_id"] for run in runs],
            "run_roots": [_relative(run["root"]) for run in runs],
            "artifact_paths": [
                [_relative(run["root"] / filename) for filename in REQUIRED_FILES]
                for run in runs
            ],
        },
    }


def aggregate_paper_results(method_specs: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(method_specs, list) or not method_specs:
        raise AggregationError("configuration must contain at least one method")
    methods = [_aggregate_method(spec) for spec in method_specs]
    method_ids = [method["method_id"] for method in methods]
    if len(set(method_ids)) != len(method_ids):
        raise AggregationError("method_id values must be unique")
    return {
        "schema_version": SCHEMA_VERSION,
        "generation_mode": "offline_no_inference",
        "methods": methods,
    }


def _format_stat(metric: dict[str, Any], decimals: int = 3) -> str:
    return f"{metric['mean']:.{decimals}f} ± {metric['std']:.{decimals}f}"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# EventQA Paper Aggregate",
        "",
        f"Schema: `{payload['schema_version']}`",
        "",
        "| Method | Repeats | EM | Recall | Format failures | Cost status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for method in payload["methods"]:
        metrics = method["metrics"]
        lines.append(
            f"| {method['method_id']} | {method['repeat_count']} | "
            f"{_format_stat(metrics['em'])} | {_format_stat(metrics['recall'])} | "
            f"{_format_stat(metrics['format_failures'], 1)} | {method.get('cost', {}).get('status', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "All rows are generated offline from frozen artifacts. Null cost fields are",
            "intentional until method-separable measurements are available.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON file containing a methods list.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _read_json(Path(args.config))
    methods = config.get("methods")
    payload = aggregate_paper_results(methods)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
