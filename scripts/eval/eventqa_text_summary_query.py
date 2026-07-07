"""Run EventQA context0 q0-9 with a frozen same-model text summary."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import mab6b_weaver_space_bank_eventqa_65536_n5 as eventqa
from scripts.eval.eventqa_bm25_retrieved_text import _rendered_prompt
from scripts.eval.eventqa_text_summary_construction import validate_construction_artifact


SCHEMA_VERSION = "eventqa-text-summary-query/v1"
DEFAULT_OUTPUT_ROOT = "outputs/mab/eventqa_text_summary_query_smoke"
DEFAULT_SUMMARY_ARTIFACT = (
    "outputs/mab/eventqa_text_summary_construction_smoke/"
    "20260706T091043Z-eventqa-text-summary-construction-ctx0/"
    "construction_artifact.json"
)


class SummaryQueryContractError(ValueError):
    pass


def expected_question_indices(scope: str, context_index: int, question_limit: int) -> list[int]:
    if scope == "smoke":
        if context_index != 0 or question_limit != 10:
            raise SummaryQueryContractError("smoke scope must be context0 q0-9")
        return list(range(10))
    if scope == "full":
        if context_index not in range(5) or question_limit != 100:
            raise SummaryQueryContractError("full scope must be context0-4 q0-99")
        return list(range(100))
    raise SummaryQueryContractError("unsupported query scope")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_summary_query_prompt(summary: str, official_query_prompt: str) -> str:
    return f"Persistent memory summary:\n{summary}\n\n{official_query_prompt}"


def _finite(value: Any, label: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise SummaryQueryContractError(f"{label} must be finite and nonnegative")


def validate_query_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise SummaryQueryContractError("unexpected schema version")
    scope = artifact.get("scope", {})
    measurement_scope = scope.get("measurement_scope", "smoke")
    expected = expected_question_indices(measurement_scope, scope.get("context_index"), len(scope.get("question_indices", [])))
    if scope.get("question_indices") != expected:
        raise SummaryQueryContractError("query scope mismatch")
    source = artifact.get("summary_source", {})
    if len(source.get("artifact_sha256", "")) != 64 or len(source.get("final_summary_sha256", "")) != 64:
        raise SummaryQueryContractError("summary provenance is incomplete")
    records = artifact.get("records", [])
    if len(records) != len(expected) or [r.get("query_index") for r in records] != expected:
        raise SummaryQueryContractError("query records mismatch")
    cost = artifact.get("cost", {})
    for field in ("construction_latency_seconds", "query_latency_seconds", "end_to_end_latency_seconds", "baseline_gpu_memory_bytes", "peak_gpu_memory_bytes"):
        _finite(cost.get(field), field)
    if cost["peak_gpu_memory_bytes"] < cost["baseline_gpu_memory_bytes"]:
        raise SummaryQueryContractError("peak memory below baseline")
    for record in records:
        if record.get("context_index") != scope.get("context_index") or record.get("summary_sha256") != source["final_summary_sha256"] or record.get("summary_token_count") != source["final_summary_token_count"]:
            raise SummaryQueryContractError("summary mutated across queries")
        if record.get("rendered_prompt_token_delta") != record.get("injected_rendered_token_count") - record.get("official_rendered_token_count"):
            raise SummaryQueryContractError("prompt delta mismatch")
        if record.get("capacity_ok") is not True or record["injected_rendered_token_count"] > record["context_capacity"]:
            raise SummaryQueryContractError("prompt capacity failure")
        _finite(record.get("cost", {}).get("query_latency_seconds"), "query latency")
        _finite(record.get("cost", {}).get("output_tokens"), "output tokens")


def build_parser():
    parser = eventqa.build_parser()
    parser.description = __doc__
    parser.add_argument("--summary-artifact", default=DEFAULT_SUMMARY_ARTIFACT)
    parser.add_argument("--measurement-scope", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--run-id")
    parser.set_defaults(output_root=DEFAULT_OUTPUT_ROOT, context_index=0, question_limit=10, generation_max_length=40, skip_research_note=True, reseed_per_context=True)
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    question_indices = expected_question_indices(args.measurement_scope, args.context_index, args.question_limit)
    source_path = Path(args.summary_artifact)
    source_bytes = source_path.read_bytes()
    construction = json.loads(source_bytes)
    validate_construction_artifact(construction)
    if construction["scope"]["context_index"] != args.context_index:
        raise SummaryQueryContractError("construction/query context mismatch")
    summary = construction["final_summary"]
    summary_hash = construction["final_summary_sha256"]
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = args.run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-eventqa-text-summary-query-ctx{args.context_index}-q0-{question_indices[-1]}")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    rows = eventqa._load_rows(args.parquet, eventqa.SUB_DATASET)
    context = eventqa.build_context_payload(args, rows[args.context_index], args.context_index, started_at)
    model, capacity = eventqa.weaver_bank._load_model(args)
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
        baseline = int(torch.cuda.memory_allocated()); torch.cuda.reset_peak_memory_stats()
    else:
        baseline = 0
    records = []
    try:
        for qi in question_indices:
            payload = eventqa.build_question_payload(context, qi)
            injected_prompt = build_summary_query_prompt(summary, payload["query_prompt"])
            _, official_count = _rendered_prompt(model, payload["query_prompt"])
            injected_rendered, injected_count = _rendered_prompt(model, injected_prompt)
            if injected_count > capacity:
                raise SummaryQueryContractError("injected prompt exceeds capacity")
            query_payload = eventqa._query_only_payload(payload)
            query_payload["query_prompt"] = injected_prompt
            query_payload["memorization_prompts"] = [injected_prompt]
            if torch.cuda.is_available(): torch.cuda.synchronize()
            start = time.perf_counter()
            result = eventqa._run_eventqa_model(args, model, capacity, query_payload, "off")
            if torch.cuda.is_available(): torch.cuda.synchronize()
            latency = time.perf_counter() - start
            if result["rendered_query_prompt"] != injected_rendered:
                raise SummaryQueryContractError("runtime prompt differs from preflight")
            with tempfile.TemporaryDirectory() as tmpdir:
                score = eventqa._score_prediction(args, payload, result["prediction"], tmpdir)
            turn = eventqa._query_turn(result)
            records.append({
                "context_index": args.context_index, "query_index": qi,
                "summary_sha256": summary_hash, "summary_token_count": construction["final_summary_token_count"],
                "official_query_sha256": _sha256(payload["query_prompt"]), "injected_prompt_sha256": _sha256(injected_prompt),
                "official_rendered_token_count": official_count, "injected_rendered_token_count": injected_count,
                "rendered_prompt_token_delta": injected_count - official_count, "context_capacity": capacity, "capacity_ok": True,
                "prediction": result["prediction"],
                "substring_exact_match": eventqa._metric_value(score, "substring_exact_match", default=0),
                "eventqa_recall": eventqa._metric_value(score, "eventqa_recall", default=0.0),
                "format_flags": eventqa._format_flags(result["prediction"]),
                "cost": {"query_latency_seconds": latency, "output_tokens": int(turn["output_len"])},
            })
        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else baseline
        query_total = sum(r["cost"]["query_latency_seconds"] for r in records)
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "scope": {"measurement_scope": args.measurement_scope, "context_index": args.context_index, "question_indices": question_indices},
            "summary_source": {"artifact_path": str(source_path), "artifact_sha256": hashlib.sha256(source_bytes).hexdigest(), "final_summary_sha256": summary_hash, "final_summary_token_count": construction["final_summary_token_count"]},
            "cost": {"construction_latency_seconds": construction["cost"]["construction_latency_seconds"], "query_latency_seconds": query_total, "end_to_end_latency_seconds": construction["cost"]["construction_latency_seconds"] + query_total, "baseline_gpu_memory_bytes": baseline, "peak_gpu_memory_bytes": peak, "incremental_peak_gpu_memory_bytes": peak-baseline},
            "records": records,
        }
        validate_query_artifact(artifact)
        _write_json(output_dir / "query_artifact.json", artifact)
        (output_dir / "per_question.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in records), encoding="utf-8")
        _write_json(output_dir / "manifest.json", {"run_id": run_id, "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(), "exact_command": [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])], "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "summary_artifact": str(source_path)})
    finally:
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
