"""Measure standalone Disabled or frozen-P7 EventQA cost on context 0 q0-9."""

from __future__ import annotations

import json
import math
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memgen.model.latent_memory_bank import LatentMemoryBankConfig
from scripts.eval import mab6b_weaver_space_bank_eventqa_65536_n5 as eventqa


SCHEMA_VERSION = "eventqa-method-cost/v1"
EXPECTED_QUESTIONS = list(range(10))
DEFAULT_OUTPUT_ROOT = "outputs/mab/eventqa_method_separable_cost_smoke"


class CostContractError(ValueError):
    """Raised when a standalone cost artifact violates the smoke contract."""


def expected_question_indices(
    measurement_scope: str, context_index: int, question_limit: int
) -> list[int]:
    if measurement_scope == "smoke":
        if context_index != 0 or question_limit != 10:
            raise CostContractError("smoke scope must be context 0 q0-9")
        return list(range(10))
    if measurement_scope == "full":
        if context_index not in range(5) or question_limit != 100:
            raise CostContractError("full scope must be context 0-4 and q0-99")
        return list(range(100))
    raise CostContractError(f"unsupported measurement scope: {measurement_scope}")


def _finite_nonnegative(value: float | int, label: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0:
        raise CostContractError(f"{label} must be finite and nonnegative")


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def build_cost_summary(
    *,
    method: str,
    run_id: str,
    context_index: int,
    question_indices: list[int],
    construction_latency_seconds: float,
    query_latencies_seconds: list[float],
    end_to_end_latency_seconds: float,
    peak_gpu_memory_bytes: int,
    baseline_gpu_memory_bytes: int,
    output_tokens: list[int],
    query_invariants: list[dict[str, Any]],
    command: list[str],
    measurement_scope: str = "smoke",
    measurement_mode: str = "standalone_process",
) -> dict[str, Any]:
    if method not in {"disabled", "p7"}:
        raise CostContractError(f"unsupported standalone method: {method}")
    expected = expected_question_indices(
        measurement_scope, context_index, len(question_indices)
    )
    if question_indices != expected:
        raise CostContractError(f"{measurement_scope} question indices are not contiguous")
    if len(query_latencies_seconds) != len(expected) or len(output_tokens) != len(expected):
        raise CostContractError(
            f"{measurement_scope} requires {len(expected)} query measurements"
        )
    for label, value in (
        ("construction latency", construction_latency_seconds),
        ("end-to-end latency", end_to_end_latency_seconds),
        ("peak GPU memory", peak_gpu_memory_bytes),
        ("baseline GPU memory", baseline_gpu_memory_bytes),
    ):
        _finite_nonnegative(value, label)
    for value in query_latencies_seconds:
        _finite_nonnegative(value, "query latency")
    if peak_gpu_memory_bytes < baseline_gpu_memory_bytes:
        raise CostContractError("peak GPU memory cannot be below baseline allocation")
    if method == "disabled" and construction_latency_seconds != 0:
        raise CostContractError("Disabled construction latency must be zero")

    all_writes_zero = all(item.get("query_write_count") == 0 for item in query_invariants)
    all_snapshots_unchanged = all(
        item.get("bank_snapshot_changed_after_query") is False
        for item in query_invariants
    )
    if method == "p7":
        if len(query_invariants) != len(expected):
            raise CostContractError(
                f"P7 requires {len(expected)} query invariant records"
            )
        if not all_writes_zero:
            raise CostContractError("P7 query writes must remain zero")
        if not all_snapshots_unchanged:
            raise CostContractError("P7 bank snapshots must remain unchanged")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "measurement_mode": measurement_mode,
        "run_id": run_id,
        "method": method,
        "scope": {
            "measurement_scope": measurement_scope,
            "subtask": "eventqa_65536",
            "context_index": context_index,
            "question_indices": question_indices,
            "question_count": len(question_indices),
        },
        "cost": {
            "construction_latency_seconds": construction_latency_seconds,
            "query_latency_seconds": _stats(query_latencies_seconds),
            "end_to_end_latency_seconds": end_to_end_latency_seconds,
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
            "baseline_gpu_memory_bytes": baseline_gpu_memory_bytes,
            "incremental_peak_gpu_memory_bytes": (
                peak_gpu_memory_bytes - baseline_gpu_memory_bytes
            ),
            "output_tokens": _stats([float(value) for value in output_tokens]),
        },
        "invariants": {
            "all_query_writes_zero": all_writes_zero if method == "p7" else None,
            "all_bank_snapshots_unchanged": (
                all_snapshots_unchanged if method == "p7" else None
            ),
            "query_records": query_invariants,
        },
        "command": command,
    }
    validate_cost_summary(summary)
    return summary


def validate_cost_summary(summary: dict[str, Any]) -> None:
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise CostContractError("unexpected cost schema version")
    if summary.get("measurement_mode") not in {
        "standalone_process",
        "continuous_process_context_segment",
    }:
        raise CostContractError("unsupported cost measurement mode")
    scope = summary.get("scope", {})
    measurement_scope = scope.get("measurement_scope", "smoke")
    expected = expected_question_indices(
        measurement_scope,
        scope.get("context_index"),
        scope.get("question_count"),
    )
    if scope.get("question_indices") != expected:
        raise CostContractError(f"{measurement_scope} question indices are invalid")
    cost = summary.get("cost", {})
    for field in (
        "construction_latency_seconds",
        "end_to_end_latency_seconds",
        "peak_gpu_memory_bytes",
        "baseline_gpu_memory_bytes",
        "incremental_peak_gpu_memory_bytes",
    ):
        _finite_nonnegative(cost.get(field), field)
    query = cost.get("query_latency_seconds", {})
    if len(query.get("values", [])) != len(expected):
        raise CostContractError(
            f"{measurement_scope} requires {len(expected)} query measurements"
        )


def build_parser():
    parser = eventqa.build_parser()
    parser.description = __doc__
    parser.add_argument("--method", required=True, choices=("disabled", "p7"))
    parser.add_argument(
        "--measurement-scope", choices=("smoke", "full"), default="smoke"
    )
    parser.add_argument(
        "--measurement-mode",
        choices=("standalone_process", "continuous_process_context_segment"),
        default="standalone_process",
        help="Execution contract for this context artifact.",
    )
    parser.set_defaults(
        output_root=DEFAULT_OUTPUT_ROOT,
        requested_contexts=5,
        context_index=0,
        question_limit=10,
        eventqa_protocol="frozen_context_bank",
        retrieve_threshold=0.05,
        update_threshold=0.10,
        max_slots=16,
        top_k=2,
        decay_alpha=0.05,
        generation_max_length=40,
        skip_research_note=True,
        reseed_per_context=True,
    )
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _cuda_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_call(function: Callable[[], Any]) -> tuple[Any, float]:
    _cuda_sync()
    start = time.perf_counter()
    result = function()
    _cuda_sync()
    return result, time.perf_counter() - start


def build_p7_query_invariant(
    result: dict[str, Any], *, query_index: int
) -> dict[str, Any]:
    query_turn = eventqa._query_turn(result)
    return {
        "query_index": query_index,
        "query_write_count": result["query_write_count"],
        "bank_snapshot_changed_after_query": result[
            "bank_snapshot_changed_after_query"
        ],
        "retrieved_indices": list(query_turn["retrieved_indices"]),
        "retrieved_latent_count": int(query_turn["retrieved_latent_count"]),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    question_indices = expected_question_indices(
        args.measurement_scope, args.context_index, args.question_limit
    )
    if args.eventqa_protocol != "frozen_context_bank":
        raise CostContractError("cost smoke requires frozen_context_bank")

    started_at = datetime.now(timezone.utc).isoformat()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"{timestamp}-eventqa-cost-{args.method}-ctx{args.context_index}-"
        f"q0-{question_indices[-1]}-{args.measurement_scope}"
    )
    output_dir = Path(args.output_root) / args.method / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]

    rows = eventqa._load_rows(args.parquet, eventqa.SUB_DATASET)
    context_payload = eventqa.build_context_payload(
        args, rows[args.context_index], args.context_index, started_at
    )
    runtime_bank_config = eventqa._eventqa_bank_config(args)
    manifest = eventqa._build_manifest(
        run_id,
        args,
        started_at,
        git_status_before=eventqa._git("status", "--short", "--branch"),
        selected_context_indices=[args.context_index],
    )
    manifest.update(
        {
            "cost_schema_version": SCHEMA_VERSION,
            "measurement_mode": "standalone_process",
            "standalone_method": args.method,
            "measurement_scope": args.measurement_scope,
            "question_indices": question_indices,
            "exact_command": command,
        }
    )
    eventqa._assert_runtime_bank_config_matches(
        LatentMemoryBankConfig(**runtime_bank_config), manifest
    )
    _write_json(output_dir / "manifest.json", manifest)

    model, capacity = eventqa.weaver_bank._load_model(args)
    import torch

    manifest["context_capacity"] = capacity
    manifest["gpu"] = {
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline_gpu_memory = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
    else:
        baseline_gpu_memory = 0

    frozen_bank = None
    construction_latency = 0.0
    query_latencies: list[float] = []
    output_tokens: list[int] = []
    query_invariants: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    try:
        if args.method == "p7":
            construction_result, construction_latency = _measure_call(
                lambda: eventqa._run_eventqa_model(
                    args,
                    model,
                    capacity,
                    eventqa._construction_only_payload(context_payload),
                    "on",
                    runtime_bank_config,
                    preserve_bank=True,
                    construction_only=True,
                    recorded_bank_config=manifest,
                    score_trace_state={},
                )
            )
            frozen_bank = construction_result.pop("_retained_bank")

        for question_index in question_indices:
            payload = eventqa.build_question_payload(context_payload, question_index)
            query_payload = eventqa._query_only_payload(payload)
            if args.method == "disabled":
                result, latency = _measure_call(
                    lambda p=query_payload: eventqa._run_eventqa_model(
                        args, model, capacity, p, "off"
                    )
                )
            else:
                result, latency = _measure_call(
                    lambda p=query_payload: eventqa._run_eventqa_model(
                        args,
                        model,
                        capacity,
                        p,
                        "on",
                        runtime_bank_config,
                        external_bank=frozen_bank,
                        preserve_bank=True,
                        recorded_bank_config=manifest,
                        score_trace_state={},
                    )
                )
                retained = result.pop("_retained_bank")
                if retained is not frozen_bank:
                    raise CostContractError("P7 replaced the frozen bank instance")
                query_invariants.append(
                    build_p7_query_invariant(result, query_index=question_index)
                )
            query_latencies.append(latency)
            query_turn = eventqa._query_turn(result)
            output_tokens.append(int(query_turn["output_len"]))
            with tempfile.TemporaryDirectory() as tmpdir:
                score = eventqa._score_prediction(args, payload, result["prediction"], tmpdir)
            question_rows.append(
                {
                    "context_index": args.context_index,
                    "query_index": question_index,
                    "qa_pair_id": payload["qa_pair_id"],
                    "prediction": result["prediction"],
                    "substring_exact_match": eventqa._metric_value(
                        score, "substring_exact_match", default=0
                    ),
                    "eventqa_recall": eventqa._metric_value(
                        score, "eventqa_recall", default=0.0
                    ),
                    "query_latency_seconds": latency,
                    "output_tokens": int(query_turn["output_len"]),
                }
            )

        peak_gpu_memory = (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else baseline_gpu_memory
        )
        end_to_end_latency = construction_latency + sum(query_latencies)
        summary = build_cost_summary(
            method=args.method,
            run_id=run_id,
            context_index=args.context_index,
            question_indices=question_indices,
            construction_latency_seconds=construction_latency,
            query_latencies_seconds=query_latencies,
            end_to_end_latency_seconds=end_to_end_latency,
            peak_gpu_memory_bytes=peak_gpu_memory,
            baseline_gpu_memory_bytes=baseline_gpu_memory,
            output_tokens=output_tokens,
            query_invariants=query_invariants,
            command=command,
            measurement_scope=args.measurement_scope,
            measurement_mode=args.measurement_mode,
        )
        summary["effectiveness"] = {
            "substring_exact_match": statistics.fmean(
                float(row["substring_exact_match"]) for row in question_rows
            ),
            "eventqa_recall": statistics.fmean(
                float(row["eventqa_recall"]) for row in question_rows
            ),
        }
        summary["method_config"] = {
            "retrieve_threshold": args.retrieve_threshold if args.method == "p7" else None,
            "update_threshold": args.update_threshold if args.method == "p7" else None,
            "max_slots": args.max_slots if args.method == "p7" else None,
            "top_k": args.top_k if args.method == "p7" else None,
            "decay_alpha": args.decay_alpha if args.method == "p7" else None,
            "generation_max_length": args.generation_max_length,
            "eventqa_protocol": args.eventqa_protocol,
        }
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["cost_summary_path"] = str(output_dir / "cost_summary.json")
        _write_json(output_dir / "cost_summary.json", summary)
        _write_jsonl(output_dir / "per_question.jsonl", question_rows)
        _write_json(output_dir / "manifest.json", manifest)
    finally:
        if frozen_bank is not None:
            frozen_bank.reset()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
