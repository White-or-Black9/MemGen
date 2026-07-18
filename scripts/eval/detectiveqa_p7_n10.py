"""Current-P7 DetectiveQA compressed-memory runner."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import mab3_bank_on_full_history as mab3
from scripts.eval import mab5a_detectiveqa_compressed_n10 as base
from scripts.eval import mab6b_weaver_space_bank_detectiveqa_n10 as weaver_bank
from scripts.eval import mab6b_weaver_space_bank_eventqa_65536_n5 as eventqa


EXPERIMENT_NAME = "DetectiveQA current-P7 compressed-memory n10"
RUN_PREFIX = "detectiveqa-current-p7-n10"
DEFAULT_OUTPUT_ROOT = "outputs/mab/detectiveqa_p7_n10"
SCHEMA_VERSION = "detectiveqa-current-p7-run/v1"
SUPPORTED_METHODS = ("disabled", "p7", "p7_no_query_retrieval")


def p7_bank_config() -> dict[str, Any]:
    config = mab3.version_a_bank_config(
        top_k=2,
        threshold=0.05,
        retrieve_policy="threshold_topk",
    )
    config.update(
        {
            "enabled": True,
            "batch_size": 1,
            "retrieve_threshold": 0.05,
            "update_threshold": 0.10,
            "max_slots": 16,
            "top_k": 2,
            "decay_alpha": 0.05,
            "retrieve_policy": "threshold_topk",
            "update_policy": "thread_update",
            "storage_space": "weaver",
            "query_phase": "read_only",
        }
    )
    return config


def runtime_bank_config() -> dict[str, Any]:
    config = dict(p7_bank_config())
    config.pop("storage_space", None)
    config.pop("query_phase", None)
    return config


def disabled_query_response_length(args) -> int:
    return int(getattr(args, "generation_max_length", eventqa.GENERATION_MAX_LENGTH))


@contextmanager
def override_disabled_query_response_length(args):
    response_length = disabled_query_response_length(args)
    original_build_config = mab3._build_config
    original_interaction_config = mab3._interaction_config

    def patched_build_config(runtime_args, context_capacity, bank_config=None):
        config = original_build_config(runtime_args, context_capacity, bank_config)
        config["run"]["interaction"]["max_response_length"] = response_length
        return config

    def patched_interaction_config(config_dict, context_capacity):
        interaction_config = original_interaction_config(config_dict, context_capacity)
        interaction_config.max_response_length = response_length
        return interaction_config

    mab3._build_config = patched_build_config
    mab3._interaction_config = patched_interaction_config
    try:
        yield
    finally:
        mab3._build_config = original_build_config
        mab3._interaction_config = original_interaction_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument(
        "--dataset-root",
        default="/mnt/18T/baishilong/datasets/MemoryAgentBench",
    )
    parser.add_argument(
        "--mab-repo", default="/mnt/18T/baishilong/benchmarks/MemoryAgentBench"
    )
    parser.add_argument(
        "--mab-python", default="/home/baishilong/miniconda3/envs/MABench/bin/python"
    )
    parser.add_argument(
        "--model-path",
        default=(
            "/home/baishilong/.cache/huggingface/hub/"
            "models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
            "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        default=(
            "/home/baishilong/.cache/huggingface/hub/"
            "models--Kana-s--MemGen/snapshots/"
            "269d9b1741130b94fffa410cdaa3d4bc74081a7f/"
            "Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model"
        ),
    )
    parser.add_argument(
        "--model-checkpoint-id",
        default=(
            "Kana-s/MemGen@269d9b1/Qwen2.5-1.5B-Instruct/triviaqa/"
            "weaver-sft/pn=8_pl=8_in=0_il=8/model"
        ),
    )
    parser.add_argument("--cfg-path", default="configs/latent_memory/triviaqa.yaml")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--methods", default="disabled,p7,p7_no_query_retrieval")
    parser.add_argument("--max-contexts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eventqa-protocol",
        default=eventqa.DEFAULT_EVENTQA_PROTOCOL,
    )
    parser.add_argument(
        "--generation-max-length",
        type=int,
        default=eventqa.GENERATION_MAX_LENGTH,
    )
    parser.add_argument("--skip-research-note", action="store_true")
    return parser


def expected_method_set(methods_spec: str) -> list[str]:
    methods = [item.strip() for item in methods_spec.split(",") if item.strip()]
    if not methods:
        raise ValueError("methods cannot be empty")
    seen = set()
    ordered = []
    for method in methods:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"unknown method: {method}")
        if method in seen:
            raise ValueError(f"duplicate method: {method}")
        seen.add(method)
        ordered.append(method)
    return ordered


def validate_query_phase_invariants(run: dict[str, Any]) -> None:
    if run["method"] == "disabled":
        return
    if int(run.get("query_write_count", 0)) != 0:
        raise ValueError("query write isolation failed")
    if bool(run.get("bank_snapshot_changed_after_query")):
        raise ValueError("query changed frozen bank")
    if run.get("query_read_only_enforced") is not True:
        raise ValueError("query read-only contract failed")


def _summary_sha256(summary: dict[str, Any]) -> str:
    payload = json.dumps(summary, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def record_from_result(
    *,
    method: str,
    payload: dict[str, Any],
    result: dict[str, Any],
    score: dict[str, Any],
) -> dict[str, Any]:
    pre_summary = result.get("pre_query_bank_summary")
    post_summary = result.get("post_query_bank_summary")
    record = {
        "method": method,
        "context_id": payload["context_id"],
        "context_index": int(payload.get("context_index", 0)),
        "query_id": int(payload.get("query_id", 0)),
        "question_count_in_context": int(payload.get("question_count_in_context", 1)),
        "question": payload.get("question"),
        "prediction": result["prediction"],
        "gold_answers": list(payload.get("gold_answers", [])),
        "metrics": score.get("metrics", {}),
        "additional": score.get("additional", {}),
        "bank_created": method != "disabled",
        "query_write_count": int(result.get("query_write_count", 0)),
        "query_write_attempt_count": int(result.get("query_write_attempt_count", 0)),
        "query_read_only_enforced": bool(
            result.get("query_read_only_enforced", method == "disabled")
        ),
        "bank_reset_after_context": bool(
            result.get("bank_reset_after_context", True)
        ),
        "cross_context_leakage_detected": bool(
            result.get("cross_context_leakage_detected", False)
        ),
        "retrieved_indices_by_turn": result.get("retrieved_indices_by_turn", []),
        "retrieved_scores_by_turn": result.get("retrieved_scores_by_turn", []),
        "bank_write_count": int(result.get("bank_write_count", 0)),
        "bank_retrieval_count": int(result.get("bank_retrieval_count", 0)),
        "bank_retrieved_latent_count": int(
            result.get("bank_retrieved_latent_count", 0)
        ),
        "bank_slot_count_final_before_reset": int(
            result.get("bank_slot_count_final_before_reset", 0)
        ),
        "pre_query_bank_summary": pre_summary,
        "post_query_bank_summary": post_summary,
    }
    if pre_summary is not None and post_summary is not None:
        record["pre_query_bank_sha256"] = _summary_sha256(pre_summary)
        record["post_query_bank_sha256"] = _summary_sha256(post_summary)
        record["bank_snapshot_changed_after_query"] = (
            record["pre_query_bank_sha256"] != record["post_query_bank_sha256"]
        )
    else:
        record["pre_query_bank_sha256"] = None
        record["post_query_bank_sha256"] = None
        record["bank_snapshot_changed_after_query"] = False
    return record


def expand_query_payloads(context_payload: dict[str, Any]) -> list[dict[str, Any]]:
    question_count = int(context_payload["question_count"])
    shared = {
        "context_id": context_payload["context_id"],
        "context_index": int(context_payload["context_index"]),
        "question_count_in_context": question_count,
        "dataset_config": context_payload["dataset_config"],
        "chunks": list(context_payload["chunks"]),
        "chunk_token_lengths": list(context_payload["chunk_token_lengths"]),
        "memorization_prompts": list(context_payload["memorization_prompts"]),
    }
    payloads = []
    for query in context_payload["queries"]:
        payloads.append(
            {
                **shared,
                "query_id": int(query["query_id"]),
                "question": query["question"],
                "query_prompt": query["query_prompt"],
                "gold_answers": list(query["gold_answers"]),
            }
        )
    return payloads


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_model(args):
    model, capacity = base._load_model(args)
    model.config.retrieved_memory_to_weaver = True
    model.config.memory_bank_storage_space = "weaver"
    return model, capacity


def _query_only_payload(payload: dict[str, Any]) -> dict[str, Any]:
    query_payload = dict(payload)
    query_payload["chunks"] = []
    query_payload["chunk_token_lengths"] = []
    query_payload["memorization_prompts"] = [payload["query_prompt"]]
    return query_payload


def _construction_only_payload(context_payload: dict[str, Any]) -> dict[str, Any]:
    first_query = expand_query_payloads(context_payload)[0]
    payload = dict(first_query)
    payload["chunks"] = list(context_payload["chunks"])
    payload["chunk_token_lengths"] = list(context_payload["chunk_token_lengths"])
    payload["memorization_prompts"] = list(context_payload["memorization_prompts"])
    return payload


def _run_model(
    args,
    model,
    capacity: int,
    payload: dict[str, Any],
    bank_mode: str,
    *,
    bank_config: dict[str, Any] | None = None,
    disable_query_retrieval: bool = False,
) -> dict[str, Any]:
    if bank_mode == "off":
        return base._run_model(args, model, capacity, payload, bank_mode)
    runtime_config = dict(bank_config or runtime_bank_config())
    original_factory = base._manager_class
    capture: dict[str, Any] = {}
    base._manager_class = eventqa._eventqa_manager_factory(
        original_factory,
        capture,
        disable_query_retrieval=disable_query_retrieval,
        recorded_bank_config=runtime_config,
    )
    try:
        result = weaver_bank._run_model(
            args,
            model,
            capacity,
            payload,
            bank_mode,
            runtime_config,
        )
    finally:
        base._manager_class = original_factory
        restore_bank_trace = capture.get("restore_bank_trace")
        if restore_bank_trace is not None:
            restore_bank_trace()
    lifecycle = capture["lifecycle"]
    result.update(
        eventqa._query_memory_diagnostics(
            lifecycle,
            eventqa._query_turn(result),
            retrieve_threshold=float(runtime_config["retrieve_threshold"]),
        )
    )
    result["query_write_count"] = int(result["query_write_count_delta"])
    result["query_write_attempt_count"] = int(result["query_write_attempt_count_delta"])
    return result


def validate_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected schema version")
    expected = set(artifact["methods"])
    by_scope: dict[tuple[str, int], set[str]] = {}
    for record in artifact["records"]:
        validate_query_phase_invariants(record)
        if not bool(record.get("bank_reset_after_context", True)):
            raise ValueError("bank did not reset after context")
        if bool(record.get("cross_context_leakage_detected", False)):
            raise ValueError("cross-context leakage detected")
        key = (record["context_id"], int(record["query_id"]))
        by_scope.setdefault(key, set()).add(record["method"])
    for key, seen in by_scope.items():
        if seen != expected:
            raise ValueError(
                f"method scope drift at {key}: expected {sorted(expected)}, got {sorted(seen)}"
            )


def _aggregate(records: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "record_count": len(records),
        "method_order": list(methods),
        "context_count": len({row["context_id"] for row in records}),
        "query_count": len(records) // len(methods) if methods else len(records),
    }
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    for record in records:
        by_method[record["method"]].append(record)
    for method, rows in by_method.items():
        exact = [int(bool(row["metrics"].get("exact_match"))) for row in rows]
        summary[f"{method}_exact_match"] = (
            sum(exact) / len(exact) if exact else None
        )
        summary[f"{method}_retrieval_count"] = sum(
            int(row.get("bank_retrieval_count", 0)) for row in rows
        )
    return summary


def main() -> int:
    args = build_parser().parse_args()
    args.parquet = str(
        Path(args.dataset_root) / "data/Long_Range_Understanding-00000-of-00001.parquet"
    )
    args.data_config = str(
        Path(args.mab_repo)
        / "configs/data_conf/Long_Range_Understanding/Detective_QA.yaml"
    )
    started_at = _utc_now()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{RUN_PREFIX}"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    methods = expected_method_set(args.methods)
    model, capacity = _load_model(args)
    records: list[dict[str, Any]] = []
    try:
        total_matches = base.count_context_matches(args.parquet, base.SUB_DATASET)
        match_indices = base.select_match_indices(total_matches, int(args.max_contexts))
        with tempfile.TemporaryDirectory(prefix="detectiveqa-p7-") as tmpdir:
            for context_index, match_index in enumerate(match_indices):
                payload_path = Path(tmpdir) / f"payload_{context_index}.json"
                context_payload = base._prepare_payload(args, payload_path, match_index, started_at)
                context_payload["context_index"] = context_index
                query_payloads = expand_query_payloads(context_payload)
                for method in methods:
                    if method == "disabled":
                        for payload in query_payloads:
                            with override_disabled_query_response_length(args):
                                result = _run_model(
                                    args,
                                    model,
                                    capacity,
                                    _query_only_payload(payload),
                                    "off",
                                )
                            with tempfile.TemporaryDirectory() as score_tmpdir:
                                score = base._score_prediction(
                                    args,
                                    payload,
                                    result["prediction"],
                                    score_tmpdir,
                                )
                            records.append(
                                record_from_result(
                                    method=method,
                                    payload=payload,
                                    result=result,
                                    score=score,
                                )
                            )
                        continue

                    frozen_bank = None
                    construction_payload = _construction_only_payload(context_payload)
                    construction_result = eventqa._run_eventqa_model(
                        args,
                        model,
                        capacity,
                        construction_payload,
                        "on",
                        runtime_bank_config(),
                        preserve_bank=True,
                        construction_only=True,
                        recorded_bank_config=runtime_bank_config(),
                    )
                    frozen_bank = construction_result.pop("_retained_bank")
                    try:
                        for payload in query_payloads:
                            result = eventqa._run_eventqa_model(
                                args,
                                model,
                                capacity,
                                _query_only_payload(payload),
                                "on",
                                runtime_bank_config(),
                                external_bank=frozen_bank,
                                preserve_bank=True,
                                disable_query_retrieval=(method == "p7_no_query_retrieval"),
                                recorded_bank_config=runtime_bank_config(),
                            )
                            retained_bank = result.pop("_retained_bank")
                            if retained_bank is not frozen_bank:
                                raise ValueError("frozen bank identity changed across queries")
                            validate_query_phase_invariants(
                                {
                                    "method": method,
                                    **result,
                                }
                            )
                            with tempfile.TemporaryDirectory() as score_tmpdir:
                                score = base._score_prediction(
                                    args,
                                    payload,
                                    result["prediction"],
                                    score_tmpdir,
                                )
                            record = record_from_result(
                                method=method,
                                payload=payload,
                                result=result,
                                score=score,
                            )
                            record["bank_reset_after_context"] = True
                            records.append(record)
                    finally:
                        if frozen_bank is not None:
                            frozen_bank.reset()
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "methods": methods,
            "context_capacity": capacity,
            "record_count": len(records),
            "records": records,
        }
        validate_artifact(artifact)
        summary = _aggregate(records, methods)
        _write_json(output_dir / "artifact.json", artifact)
        _write_json(output_dir / "summary.json", summary)
        _write_jsonl(output_dir / "records.jsonl", records)
    finally:
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
