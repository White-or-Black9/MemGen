"""EventQA P7 structural control: top-1 direct latent injection."""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memgen.model.latent_memory_bank import LatentMemoryBankConfig
from scripts.eval import mab6b_weaver_space_bank_eventqa_65536_n5 as eventqa
from scripts.eval.eventqa_p7_no_retrieved_memory_conditioning import (
    _measure,
    _write_json,
    _write_jsonl,
    expected_question_indices,
)


SCHEMA_VERSION = "eventqa-p7-direct-top1/v1"
DEFAULT_OUTPUT_ROOT = "outputs/mab/eventqa_p7_direct_top1"


class DirectTop1ContractError(ValueError):
    pass


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


def _record(result: dict, payload: dict, score: dict, latency: float) -> dict:
    turn = eventqa._query_turn(result)
    invariant = {
        "query_latent_usage": "direct_top1",
        "bank_slot_count": int(result["pre_query_bank_summary"]["slot_count"]),
        "retrieval_invocation_count": int(result["query_retrieval_invocation_count"]),
        "retrieval_requested_top_k": int(turn.get("retrieval_requested_top_k") or 0),
        "retrieval_effective_top_k": int(turn.get("retrieval_effective_top_k") or 0),
        "retrieved_slot_count": int(turn.get("retrieved_slot_count", 0)),
        "retrieved_indices": list(turn.get("retrieved_indices", [])),
        "retrieved_scores": list(turn.get("retrieved_scores", [])),
        "retrieved_latent_count": int(turn.get("retrieved_latent_count", 0)),
        "retrieved_latent_space": turn.get("retrieved_memory_space"),
        "reasoner_injected_latent_count": int(turn.get("reasoner_injected_latent_count", 0)),
        "reasoner_injected_latent_shape": turn.get("reasoner_injected_latent_shape"),
        "reasoner_injected_latent_space": turn.get("reasoner_injected_latent_space"),
        "trigger_invoke_count": int(turn["trigger_positive_count"]),
        "query_weaver_invoke_count": int(turn.get("query_weaver_invoke_count", 0)),
        "weaver_output_generated": bool(turn.get("weaver_output_generated", False)),
        "weaver_output_consumed": bool(turn.get("weaver_output_consumed", False)),
        "direct_injection_applied": bool(turn.get("direct_injection_applied", False)),
        "reasoner_invoke_count": int(turn["output_len"]),
        "query_write_count": int(result["query_write_count"]),
        "query_update_count": 0,
        "query_replace_count": 0,
        "query_read_only_enforced": bool(result["query_read_only_enforced"]),
        "bank_snapshot_changed_after_query": bool(result["bank_snapshot_changed_after_query"]),
    }
    if invariant["query_write_count"] or invariant["bank_snapshot_changed_after_query"]:
        raise DirectTop1ContractError("query mutation detected")
    if invariant["query_weaver_invoke_count"]:
        raise DirectTop1ContractError("direct_top1 invoked query-time Weaver")
    if invariant["weaver_output_generated"] or invariant["weaver_output_consumed"]:
        raise DirectTop1ContractError("direct_top1 generated or consumed Weaver output")
    if invariant["retrieved_slot_count"] > 1 or invariant["retrieval_effective_top_k"] > 1:
        raise DirectTop1ContractError("direct_top1 retrieved more than one slot")
    if invariant["retrieved_slot_count"]:
        if invariant["reasoner_injected_latent_count"] != invariant["retrieved_latent_count"]:
            raise DirectTop1ContractError("direct injection latent budget mismatch")
    elif invariant["reasoner_injected_latent_count"]:
        raise DirectTop1ContractError("empty retrieval injected latent")
    return {
        "context_id": payload["context_id"], "context_index": payload["context_index"],
        "question_id": payload["question_id"], "query_index": payload["query_id"],
        "qa_pair_id": payload["qa_pair_id"], "prediction": result["prediction"],
        "substring_exact_match": eventqa._metric_value(score, "substring_exact_match", default=0),
        "eventqa_recall": eventqa._metric_value(score, "eventqa_recall", default=0.0),
        "format_flags": eventqa._format_flags(result["prediction"]),
        "cost": {"query_latency_seconds": latency, "output_tokens": int(turn["output_len"])},
        "query_invariants": invariant,
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise DirectTop1ContractError("unexpected artifact schema")
    method = artifact.get("method_config", {})
    if method.get("query_latent_usage") != "direct_top1":
        raise DirectTop1ContractError("incorrect query latent usage")
    if method.get("query_retrieval_disabled"):
        raise DirectTop1ContractError("direct_top1 cannot disable retrieval")
    for row in artifact.get("records", []):
        invariant = row["query_invariants"]
        if invariant["retrieved_slot_count"] > 1 or invariant["query_weaver_invoke_count"]:
            raise DirectTop1ContractError("invalid direct_top1 query path")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    indices = expected_question_indices(args.measurement_scope, args.context_index, args.question_limit)
    if args.eventqa_protocol != "frozen_context_bank":
        raise DirectTop1ContractError("requires frozen_context_bank")
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-eventqa-p7-direct-top1-ctx{args.context_index}-q0-{indices[-1]}-{args.measurement_scope}"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    rows = eventqa._load_rows(args.parquet, eventqa.SUB_DATASET)
    context = eventqa.build_context_payload(args, rows[args.context_index], args.context_index, started_at)
    bank_config = eventqa._eventqa_bank_config(args)
    manifest = eventqa._build_manifest(run_id, args, started_at, git_status_before=eventqa._git("status", "--short", "--branch"), selected_context_indices=[args.context_index])
    manifest.update({
        "schema_version": SCHEMA_VERSION, "measurement_mode": "standalone_process",
        "measurement_scope": args.measurement_scope, "question_indices": indices,
        "query_retrieval_disabled": False, "query_latent_usage": "direct_top1",
        "query_direct_retrieve_top_k": 1, "exact_command": command,
        "method_fingerprint": eventqa._hash_value({"p7_variant": "direct_top1", "query_latent_usage": "direct_top1", "query_top_k": 1, "bank_config": bank_config, "generation_max_length": args.generation_max_length}),
    })
    eventqa._assert_runtime_bank_config_matches(LatentMemoryBankConfig(**bank_config), manifest)
    _write_json(output_dir / "manifest.json", manifest)
    model, capacity = eventqa.weaver_bank._load_model(args)
    baseline_memory = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    frozen_bank = None
    try:
        construction, construction_latency = _measure(lambda: eventqa._run_eventqa_model(args, model, capacity, eventqa._construction_only_payload(context), "on", bank_config, preserve_bank=True, construction_only=True, recorded_bank_config=manifest, score_trace_state={}))
        frozen_bank = construction.pop("_retained_bank")
        construction_fingerprint = eventqa._bank_state_fingerprint(frozen_bank)
        records = []
        for question_index in indices:
            payload = eventqa._query_only_payload(eventqa.build_question_payload(context, question_index))
            result, latency = _measure(lambda p=payload: eventqa._run_eventqa_model(args, model, capacity, p, "on", bank_config, external_bank=frozen_bank, preserve_bank=True, query_latent_usage="direct_top1", recorded_bank_config=manifest, score_trace_state={}))
            if result.pop("_retained_bank") is not frozen_bank:
                raise DirectTop1ContractError("query replaced frozen bank")
            with tempfile.TemporaryDirectory() as tmpdir:
                score = eventqa._score_prediction(args, payload, result["prediction"], tmpdir)
            records.append(_record(result, payload, score, latency))
        artifact = {
            "schema_version": SCHEMA_VERSION, "measurement_mode": "standalone_process", "run_id": run_id,
            "scope": {"measurement_scope": args.measurement_scope, "context_index": args.context_index, "question_indices": indices},
            "method_config": {"eventqa_protocol": args.eventqa_protocol, "query_retrieval_disabled": False, "query_latent_usage": "direct_top1", "query_direct_retrieve_top_k": 1, **bank_config},
            "construction": {"construction_latency_seconds": construction_latency, "final_slot_count": len(frozen_bank), "frozen_bank_fingerprint": construction_fingerprint},
            "effectiveness": {"substring_exact_match": statistics.fmean(float(r["substring_exact_match"]) for r in records), "eventqa_recall": statistics.fmean(float(r["eventqa_recall"]) for r in records), "format_failure_count": sum(int(any(r["format_flags"].values())) for r in records)},
            "cost": {"baseline_gpu_memory_bytes": baseline_memory, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else baseline_memory, "end_to_end_latency_seconds": construction_latency + sum(r["cost"]["query_latency_seconds"] for r in records)},
            "records": records, "command": command,
        }
        validate_artifact(artifact)
        _write_json(output_dir / "artifact.json", artifact)
        _write_jsonl(output_dir / "per_question.jsonl", records)
        manifest.update({"finished_at": datetime.now(timezone.utc).isoformat(), "artifact_path": str(output_dir / "artifact.json")})
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
