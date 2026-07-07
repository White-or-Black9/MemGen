"""Run standalone EventQA P7 with query-time retrieval disabled."""

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


SCHEMA_VERSION = "eventqa-p7-no-query-retrieval/v1"
DEFAULT_OUTPUT_ROOT = "outputs/mab/eventqa_p7_no_query_retrieval_smoke"


class NoQueryRetrievalContractError(ValueError):
    """Raised when the no-query-retrieval artifact violates contract."""


def expected_question_indices(
    measurement_scope: str, context_index: int, question_limit: int
) -> list[int]:
    if measurement_scope == "smoke":
        if context_index != 0 or question_limit != 10:
            raise NoQueryRetrievalContractError("smoke scope must be context 0 q0-9")
        return list(range(10))
    if measurement_scope == "full":
        if context_index not in range(5) or question_limit != 100:
            raise NoQueryRetrievalContractError("full scope must be context 0-4 q0-99")
        return list(range(100))
    raise NoQueryRetrievalContractError(
        f"unsupported measurement scope: {measurement_scope}"
    )


def _finite_nonnegative(value: Any, label: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise NoQueryRetrievalContractError(f"{label} must be finite and nonnegative")


def validate_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise NoQueryRetrievalContractError("unexpected schema version")
    if artifact.get("measurement_mode") != "standalone_process":
        raise NoQueryRetrievalContractError("measurement must use a standalone process")
    scope = artifact.get("scope", {})
    measurement_scope = scope.get("measurement_scope", "smoke")
    context_index = scope.get("context_index")
    question_indices = scope.get("question_indices")
    expected = expected_question_indices(
        measurement_scope, context_index, len(question_indices or [])
    )
    records = artifact.get("records", [])
    scope_label = "q0-9" if measurement_scope == "smoke" else "q0-99"
    if question_indices != expected:
        raise NoQueryRetrievalContractError("question index scope drift")
    if [record.get("query_index") for record in records] != expected:
        raise NoQueryRetrievalContractError(
            f"{measurement_scope} records must cover context {context_index} {scope_label} exactly"
        )
    method_config = artifact.get("method_config", {})
    if method_config.get("query_retrieval_disabled") is not True:
        raise NoQueryRetrievalContractError("query_retrieval_disabled must be true")
    if method_config.get("eventqa_protocol") != "frozen_context_bank":
        raise NoQueryRetrievalContractError("eventqa_protocol must be frozen_context_bank")
    construction = artifact.get("construction", {})
    _finite_nonnegative(
        construction.get("construction_latency_seconds"),
        "construction_latency_seconds",
    )
    _finite_nonnegative(construction.get("final_slot_count"), "final_slot_count")
    cost = artifact.get("cost", {})
    for field in (
        "baseline_gpu_memory_bytes",
        "peak_gpu_memory_bytes",
        "end_to_end_latency_seconds",
    ):
        _finite_nonnegative(cost.get(field), field)
    if cost["peak_gpu_memory_bytes"] < cost["baseline_gpu_memory_bytes"]:
        raise NoQueryRetrievalContractError("peak GPU memory cannot be below baseline")
    for record in records:
        if record.get("context_index") != context_index:
            raise NoQueryRetrievalContractError("record context does not match artifact context")
        method_cost = record.get("cost", {})
        for field in ("query_latency_seconds", "output_tokens"):
            _finite_nonnegative(method_cost.get(field), field)
        invariants = record.get("query_invariants", {})
        if invariants.get("query_write_count") != 0:
            raise NoQueryRetrievalContractError("query writes must remain zero")
        if invariants.get("bank_snapshot_changed_after_query") is not False:
            raise NoQueryRetrievalContractError("bank snapshot must remain unchanged")
        if list(invariants.get("retrieved_indices", [])):
            raise NoQueryRetrievalContractError(
                "retrieval disabled but retrieved_indices is non-empty"
            )
        if int(invariants.get("retrieved_latent_count", -1)) != 0:
            raise NoQueryRetrievalContractError(
                "retrieval disabled but retrieved_latent_count is non-zero"
            )


def build_parser():
    parser = eventqa.build_parser()
    parser.description = __doc__
    parser.add_argument("--measurement-scope", choices=("smoke", "full"), default="smoke")
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    question_indices = expected_question_indices(
        args.measurement_scope, args.context_index, args.question_limit
    )
    if args.eventqa_protocol != "frozen_context_bank":
        raise NoQueryRetrievalContractError(
            "no-query-retrieval requires frozen_context_bank"
        )

    started_at = datetime.now(timezone.utc).isoformat()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"{timestamp}-eventqa-p7-no-query-retrieval-ctx{args.context_index}-"
        f"q0-{question_indices[-1]}-{args.measurement_scope}"
    )
    output_dir = Path(args.output_root) / run_id
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
            "schema_version": SCHEMA_VERSION,
            "measurement_mode": "standalone_process",
            "measurement_scope": args.measurement_scope,
            "question_indices": question_indices,
            "query_retrieval_disabled": True,
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
    artifact_records: list[dict[str, Any]] = []
    per_question_rows: list[dict[str, Any]] = []
    construction_latency = 0.0
    construction_result = None
    try:
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
                    disable_query_retrieval=True,
                    recorded_bank_config=manifest,
                    score_trace_state={},
                )
            )
            retained = result.pop("_retained_bank")
            if retained is not frozen_bank:
                raise NoQueryRetrievalContractError("query replaced the frozen bank instance")
            query_turn = eventqa._query_turn(result)
            invariants = {
                "query_write_count": int(result["query_write_count"]),
                "bank_snapshot_changed_after_query": bool(
                    result["bank_snapshot_changed_after_query"]
                ),
                "retrieved_indices": list(query_turn["retrieved_indices"]),
                "retrieved_latent_count": int(query_turn["retrieved_latent_count"]),
            }
            with tempfile.TemporaryDirectory() as tmpdir:
                score = eventqa._score_prediction(args, payload, result["prediction"], tmpdir)
            record = {
                "context_index": args.context_index,
                "query_index": question_index,
                "qa_pair_id": payload["qa_pair_id"],
                "prediction": result["prediction"],
                "substring_exact_match": eventqa._metric_value(
                    score, "substring_exact_match", default=0
                ),
                "eventqa_recall": eventqa._metric_value(score, "eventqa_recall", default=0.0),
                "format_flags": eventqa._format_flags(result["prediction"]),
                "cost": {
                    "query_latency_seconds": latency,
                    "output_tokens": int(query_turn["output_len"]),
                },
                "query_invariants": invariants,
            }
            artifact_records.append(record)
            per_question_rows.append(
                {
                    "context_index": args.context_index,
                    "query_index": question_index,
                    "qa_pair_id": payload["qa_pair_id"],
                    "prediction": result["prediction"],
                    "substring_exact_match": record["substring_exact_match"],
                    "eventqa_recall": record["eventqa_recall"],
                    "query_latency_seconds": latency,
                    "output_tokens": int(query_turn["output_len"]),
                }
            )

        artifact = {
            "schema_version": SCHEMA_VERSION,
            "measurement_mode": "standalone_process",
            "run_id": run_id,
            "scope": {
                "measurement_scope": args.measurement_scope,
                "subtask": "eventqa_65536",
                "context_index": args.context_index,
                "question_indices": question_indices,
                "question_count": len(question_indices),
            },
            "method_config": {
                "retrieve_threshold": args.retrieve_threshold,
                "update_threshold": args.update_threshold,
                "max_slots": args.max_slots,
                "top_k": args.top_k,
                "decay_alpha": args.decay_alpha,
                "generation_max_length": args.generation_max_length,
                "eventqa_protocol": args.eventqa_protocol,
                "query_retrieval_disabled": True,
            },
            "construction": {
                "construction_latency_seconds": construction_latency,
                "final_slot_count": int(
                    construction_result["pre_query_bank_summary"]["slot_count"]
                ),
            },
            "cost": {
                "baseline_gpu_memory_bytes": baseline_gpu_memory,
                "peak_gpu_memory_bytes": (
                    int(torch.cuda.max_memory_allocated())
                    if torch.cuda.is_available()
                    else baseline_gpu_memory
                ),
                "end_to_end_latency_seconds": (
                    construction_latency
                    + sum(record["cost"]["query_latency_seconds"] for record in artifact_records)
                ),
            },
            "effectiveness": {
                "substring_exact_match": statistics.fmean(
                    float(row["substring_exact_match"]) for row in artifact_records
                ),
                "eventqa_recall": statistics.fmean(
                    float(row["eventqa_recall"]) for row in artifact_records
                ),
                "format_failure_count": sum(
                    int(any(row["format_flags"].values())) for row in artifact_records
                ),
            },
            "records": artifact_records,
            "command": command,
        }
        validate_artifact(artifact)
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["artifact_path"] = str(output_dir / "smoke_artifact.json")
        _write_json(output_dir / "smoke_artifact.json", artifact)
        _write_jsonl(output_dir / "per_question.jsonl", per_question_rows)
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
