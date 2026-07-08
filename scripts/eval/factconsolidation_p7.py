"""Paired Disabled/P7 runner for FactConsolidation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.eval import factconsolidation_adapter as adapter
from scripts.eval import mab3_bank_on_full_history as mab3
from scripts.eval import mab5a_detectiveqa_compressed_n10 as base
from scripts.eval import mab6b_weaver_space_bank_eventqa_65536_n5 as eventqa
from scripts.eval import mab6b_weaver_space_bank_detectiveqa_n10 as weaver_bank


DEFAULT_OUTPUT_ROOT = "outputs/mab/factconsolidation_p7"
DEFAULT_DATASET_ROOT = "/mnt/18T/baishilong/datasets/MemoryAgentBench"
DEFAULT_MAB_REPO = "/mnt/18T/baishilong/benchmarks/MemoryAgentBench"
DEFAULT_MAB_PYTHON = "/home/baishilong/miniconda3/envs/MABench/bin/python"
DEFAULT_MODEL_PATH = (
    "/home/baishilong/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
DEFAULT_CHECKPOINT_PATH = (
    "/home/baishilong/.cache/huggingface/hub/"
    "models--Kana-s--MemGen/snapshots/"
    "269d9b1741130b94fffa410cdaa3d4bc74081a7f/"
    "Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model"
)
DEFAULT_MODEL_CHECKPOINT_ID = (
    "Kana-s/MemGen@269d9b1/Qwen2.5-1.5B-Instruct/triviaqa/"
    "weaver-sft/pn=8_pl=8_in=0_il=8/model"
)
DEFAULT_CFG_PATH = "configs/latent_memory/triviaqa.yaml"
SCHEMA_VERSION = "factconsolidation-p7-run/v1"
SUPPORTED_METHODS = ("disabled", "p7", "p7_no_query_retrieval")


class FactConsolidationRunContractError(ValueError):
    """Raised when a FactConsolidation run violates lifecycle invariants."""


def load_matrix(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def p7_bank_config(matrix: dict) -> dict:
    config = mab3.version_a_bank_config(
        top_k=int(matrix["p7"]["top_k"]),
        threshold=float(matrix["p7"]["retrieve_threshold"]),
        retrieve_policy="threshold_topk",
    )
    config.update(dict(matrix["p7"]))
    config.update(
        {
            "enabled": True,
            "batch_size": 1,
            "update_policy": "thread_update",
            "retrieve_policy": "threshold_topk",
        }
    )
    return config


def _runtime_bank_config(matrix: dict) -> dict[str, Any]:
    config = dict(p7_bank_config(matrix))
    config.pop("storage_space", None)
    config.pop("query_phase", None)
    return config


def validate_batch_size(batch_size: int) -> None:
    if int(batch_size) != 1:
        raise FactConsolidationRunContractError(
            "FactConsolidation runner requires batch_size=1"
        )


def validate_context_start(context_state: dict[str, Any]) -> None:
    if int(context_state.get("initial_slot_count", 0)) != 0:
        raise FactConsolidationRunContractError(
            "each context must start with zero slots"
        )


def validate_disabled_no_bank(run: dict[str, Any]) -> None:
    if run["method"] == "disabled" and bool(run.get("bank_created")):
        raise FactConsolidationRunContractError("Disabled created a bank")


def validate_query_phase_invariants(run: dict[str, Any]) -> None:
    if run["method"] == "disabled":
        return
    if int(run.get("query_write_count", 0)) != 0:
        raise FactConsolidationRunContractError("query write isolation failed")
    if bool(run.get("bank_snapshot_changed_after_query")):
        raise FactConsolidationRunContractError(
            "query snapshot changed during read-only phase"
        )
    if run.get("query_read_only_enforced") is not True:
        raise FactConsolidationRunContractError(
            "query read-only contract was not enforced"
        )


def validate_no_query_retrieval_construction(
    p7_run: dict[str, Any], no_query_run: dict[str, Any]
) -> None:
    fields = (
        "construction_bank_write_count",
        "construction_final_slot_count",
        "construction_turn_count",
    )
    mismatches = [
        field
        for field in fields
        if p7_run.get(field) != no_query_run.get(field)
    ]
    if mismatches:
        raise FactConsolidationRunContractError(
            "construction mismatch between p7 and p7_no_query_retrieval: "
            + ", ".join(mismatches)
        )


def validate_run_invariants(run: dict[str, Any]) -> None:
    validate_disabled_no_bank(run)
    if run["method"] != "disabled":
        validate_query_phase_invariants(run)
    if not bool(run.get("bank_reset_after_context")):
        raise FactConsolidationRunContractError("bank did not reset after context")
    if bool(run.get("cross_context_leakage_detected")):
        raise FactConsolidationRunContractError("cross-context leakage detected after reset")


def expected_method_set(methods_spec: str) -> list[str]:
    methods = [item.strip() for item in methods_spec.split(",") if item.strip()]
    if not methods:
        raise FactConsolidationRunContractError("methods cannot be empty")
    seen = set()
    ordered = []
    for method in methods:
        if method not in SUPPORTED_METHODS:
            raise FactConsolidationRunContractError(f"unknown method: {method}")
        if method in seen:
            raise FactConsolidationRunContractError(f"duplicate method: {method}")
        seen.add(method)
        ordered.append(method)
    return ordered


def build_context_payload(normalized: dict[str, Any], *, context_index: int) -> dict[str, Any]:
    queries = list(normalized["queries"])
    return {
        "subtask": normalized["subtask"],
        "dataset_config": normalized["dataset_config"],
        "context_id": normalized["context_id"],
        "context_index": context_index,
        "chunks": list(normalized["chunks"]),
        "chunk_token_lengths": list(normalized.get("chunk_token_lengths", [])),
        "memorization_prompts": list(normalized["memorization_prompts"]),
        "questions": [query["question"] for query in queries],
        "answers": [list(query["gold_answers"]) for query in queries],
        "question_ids": list(range(len(queries))),
        "question_types": [None] * len(queries),
        "qa_pair_ids": list(normalized.get("qa_pair_ids", [])),
        "previous_events": [[] for _ in queries],
        "question_count": int(normalized["question_count"]),
        "source": normalized["subtask"],
        "config_hash": normalized.get("config_hash"),
        "parquet_hash": normalized.get("parquet_hash"),
        "queries": queries,
    }


def build_query_payload(context_payload: dict[str, Any], question_index: int) -> dict[str, Any]:
    return {
        "dataset_config": context_payload["dataset_config"],
        "context_id": context_payload["context_id"],
        "context_index": context_payload["context_index"],
        "query_id": question_index,
        "question_id": context_payload["question_ids"][question_index],
        "question_type": context_payload["question_types"][question_index],
        "qa_pair_id": context_payload["qa_pair_ids"][question_index],
        "previous_events": context_payload["previous_events"][question_index],
        "chunks": context_payload["chunks"],
        "chunk_token_lengths": context_payload["chunk_token_lengths"],
        "memorization_prompts": context_payload["memorization_prompts"],
        "query_prompt": context_payload["queries"][question_index]["query_prompt"]
        if "queries" in context_payload
        else None,
        "question": context_payload["questions"][question_index],
        "gold_answers": list(context_payload["answers"][question_index]),
    }


def _query_only_payload(payload: dict[str, Any]) -> dict[str, Any]:
    query_payload = dict(payload)
    query_payload["chunks"] = []
    query_payload["chunk_token_lengths"] = []
    query_payload["memorization_prompts"] = [payload["query_prompt"]]
    return query_payload


def _construction_only_payload(context_payload: dict[str, Any]) -> dict[str, Any]:
    payload = build_query_payload(context_payload, 0)
    payload["chunks"] = context_payload["chunks"]
    payload["chunk_token_lengths"] = context_payload["chunk_token_lengths"]
    payload["memorization_prompts"] = context_payload["memorization_prompts"]
    return payload


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def validate_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise FactConsolidationRunContractError("unexpected schema version")
    methods = artifact.get("methods") or sorted(
        {record["method"] for record in artifact.get("records", [])}
    )
    expected = set(methods)
    by_scope: dict[tuple[str, int], set[str]] = {}
    for record in artifact.get("records", []):
        validate_run_invariants(record)
        if int(record.get("post_reset_slot_count", 0)) != 0:
            raise FactConsolidationRunContractError("bank reset failed")
        scope = (record["context_id"], int(record["query_id"]))
        by_scope.setdefault(scope, set()).add(record["method"])
    for scope, seen in by_scope.items():
        if seen != expected:
            raise FactConsolidationRunContractError(
                f"method scope drift at {scope}: expected {sorted(expected)}, got {sorted(seen)}"
            )
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for construction in artifact.get("construction_runs", []):
        grouped.setdefault(construction["context_id"], {})[construction["method"]] = construction
    for context_id, runs in grouped.items():
        if "p7" in runs and "p7_no_query_retrieval" in runs:
            validate_no_query_retrieval_construction(runs["p7"], runs["p7_no_query_retrieval"])


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _subtask_config_path(mab_repo: str | Path, split: str, subtask: str) -> Path:
    return (
        Path(mab_repo)
        / "configs"
        / "data_conf"
        / split
        / (subtask.replace("factconsolidation", "Factconsolidation") + ".yaml")
    )


def _bridge_script() -> Path:
    return Path(base.__file__).with_name("mab2_mab_bridge.py")


def _mab_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    return env


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _context_payload_from_row(
    *,
    args,
    matrix: dict[str, Any],
    subtask: str,
    row: dict[str, Any],
    context_index: int,
    timestamp: str,
) -> dict[str, Any]:
    config_path = _subtask_config_path(args.mab_repo, matrix["split"], subtask)
    with tempfile.TemporaryDirectory() as tmpdir:
        prepared_path = Path(tmpdir) / f"{subtask}_{context_index}.json"
        command = [
            args.mab_python,
            str(_bridge_script()),
            "prepare",
            "--mab-repo", args.mab_repo,
            "--output", str(prepared_path),
            "--parquet", args.parquet,
            "--data-config", str(config_path),
            "--sub-dataset", subtask,
            "--chunk-size", str(args.chunk_size),
            "--timestamp", timestamp,
            "--match-index", str(context_index),
        ]
        subprocess.run(command, check=True, env=_mab_env())
        prepared = _load_json(prepared_path)
    normalized = {
        "subtask": subtask,
        "context_id": prepared["context_id"],
        "dataset_config": prepared["dataset_config"],
        "chunks": prepared["chunks"],
        "chunk_token_lengths": prepared.get("chunk_token_lengths", []),
        "memorization_prompts": prepared["memorization_prompts"],
        "queries": [
            {
                "query_id": query_id,
                "qa_pair_id": row.get("metadata", {}).get("qa_pair_ids", [None] * len(row["questions"]))[query_id]
                if query_id < len(row.get("metadata", {}).get("qa_pair_ids", []))
                else None,
                "question": question,
                "query_prompt": prepared["query_prompt"].replace(row["questions"][0], question, 1)
                if row["questions"]
                else prepared["query_prompt"],
                "gold_answers": list(answer if isinstance(answer, list) else [answer]),
            }
            for query_id, (question, answer) in enumerate(zip(row["questions"], row["answers"]))
        ],
        "qa_pair_ids": list(row.get("metadata", {}).get("qa_pair_ids", [])),
        "question_count": len(row["questions"]),
        "config_hash": adapter.sha256_file(config_path),
        "parquet_hash": adapter.sha256_file(Path(args.parquet)),
    }
    payload = build_context_payload(normalized, context_index=context_index)
    return payload


def _prompt_token_count(tokenizer, prompt: str) -> int:
    tokens, _ = base._render_chat(
        tokenizer,
        [[
            {"role": "system", "content": base.DEFAULT_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ]],
    )
    return int(tokens)


def _assert_prompt_capacity(tokenizer, capacity: int, context_payload: dict[str, Any]) -> None:
    for prompt in context_payload["memorization_prompts"]:
        prompt_tokens = _prompt_token_count(tokenizer, prompt)
        if prompt_tokens > capacity:
            raise FactConsolidationRunContractError(
                f"construction prompt exceeded capacity: {prompt_tokens}>{capacity}"
            )
    for query_index in range(context_payload["question_count"]):
        payload = build_query_payload(context_payload, query_index)
        query_payload = _query_only_payload(payload)
        compressed_tokens, _, _, _ = base.compressed_query_token_count(tokenizer, query_payload)
        if compressed_tokens > capacity:
            raise FactConsolidationRunContractError(
                f"query prompt exceeded capacity: {compressed_tokens}>{capacity}"
            )


def _construction_record(method: str, context_payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    pre = result["pre_query_bank_summary"]
    return {
        "method": method,
        "context_id": context_payload["context_id"],
        "construction_bank_write_count": int(pre["write_count"]),
        "construction_final_slot_count": int(pre["slot_count"]),
        "construction_turn_count": len(result.get("construction_turn_diagnostics", [])),
    }


def _record_from_result(
    *,
    method: str,
    payload: dict[str, Any],
    result: dict[str, Any],
    score: dict[str, Any],
    cleanup_slot_count: int,
) -> dict[str, Any]:
    if method == "disabled":
        return {
            "method": method,
            "context_id": payload["context_id"],
            "query_id": int(payload["query_id"]),
            "qa_pair_id": payload["qa_pair_id"],
            "prediction": result["prediction"],
            "metrics": score["metrics"],
            "additional": score["additional"],
            "bank_created": False,
            "query_write_count": 0,
            "bank_snapshot_changed_after_query": False,
            "query_read_only_enforced": True,
            "bank_reset_after_context": True,
            "cross_context_leakage_detected": False,
            "post_reset_slot_count": int(cleanup_slot_count),
        }
    pre = result["pre_query_bank_summary"]
    post = result["post_query_bank_summary"]
    return {
        "method": method,
        "context_id": payload["context_id"],
        "query_id": int(payload["query_id"]),
        "qa_pair_id": payload["qa_pair_id"],
        "prediction": result["prediction"],
        "metrics": score["metrics"],
        "additional": score["additional"],
        "bank_created": True,
        "query_write_count": int(result["query_write_count_delta"]),
        "bank_snapshot_changed_after_query": bool(result["bank_snapshot_changed_after_query"]),
        "query_read_only_enforced": bool(result["query_read_only_enforced"]),
        "bank_reset_after_context": bool(result["bank_reset_after_context"]),
        "cross_context_leakage_detected": bool(result["cross_context_leakage_detected"]),
        "post_reset_slot_count": int(cleanup_slot_count),
        "pre_query_bank_sha256": _json_hash(pre),
        "post_query_bank_sha256": _json_hash(post),
        "retrieved_indices_by_turn": result.get("retrieved_indices_by_turn", []),
        "retrieved_scores_by_turn": result.get("retrieved_scores_by_turn", []),
        "construction_bank_write_count": int(pre["write_count"]),
        "construction_final_slot_count": int(pre["slot_count"]),
        "construction_turn_count": len(result.get("construction_turn_diagnostics", [])),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--mab-repo", default=DEFAULT_MAB_REPO)
    parser.add_argument("--mab-python", default=DEFAULT_MAB_PYTHON)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--model-checkpoint-id", default=DEFAULT_MODEL_CHECKPOINT_ID)
    parser.add_argument("--cfg-path", default=DEFAULT_CFG_PATH)
    parser.add_argument("--matrix", default="configs/eval/factconsolidation_p7.json")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--subtask", required=True)
    parser.add_argument("--methods", default="disabled,p7,p7_no_query_retrieval")
    parser.add_argument("--max-contexts", type=int, default=1)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation-max-length", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--eventqa-protocol", default="frozen_context_bank")
    parser.add_argument("--skip-research-note", action="store_true")
    parser.add_argument("--reseed-per-context", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_batch_size(args.batch_size)
    matrix = load_matrix(args.matrix)
    if args.subtask not in matrix["subtasks"]:
        raise FactConsolidationRunContractError(
            f"subtask not present in matrix: {args.subtask}"
        )
    methods = expected_method_set(args.methods)
    args.parquet = str(Path(args.dataset_root) / "data/Conflict_Resolution-00000-of-00001.parquet")

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{args.subtask}-paired-p7"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    git_status_before = _git("status", "--short", "--branch")
    started_at = datetime.now(timezone.utc).isoformat()
    rows = adapter.load_rows(Path(args.parquet), args.subtask)
    selected_rows = rows[: min(len(rows), int(args.max_contexts))]
    runtime_bank_config = _runtime_bank_config(matrix)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "started_at": started_at,
        "subtask": args.subtask,
        "methods": methods,
        "matrix_path": str(Path(args.matrix)),
        "parquet_path": args.parquet,
        "mab_repo": args.mab_repo,
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "model_checkpoint_id": args.model_checkpoint_id,
        "batch_size": args.batch_size,
        "eventqa_protocol": args.eventqa_protocol,
        "max_contexts": int(args.max_contexts),
        "max_queries": args.max_queries,
        "latent_memory_bank_config": p7_bank_config(matrix),
        "git_status_before": git_status_before,
    }
    _write_json(output_dir / "manifest.json", manifest)

    model, capacity = weaver_bank._load_model(args)
    tokenizer = model.tokenizer
    records: list[dict[str, Any]] = []
    construction_runs: list[dict[str, Any]] = []
    try:
        for context_index, row in enumerate(selected_rows):
            context_payload = _context_payload_from_row(
                args=args,
                matrix=matrix,
                subtask=args.subtask,
                row=row,
                context_index=context_index,
                timestamp=started_at,
            )
            _assert_prompt_capacity(tokenizer, capacity, context_payload)
            question_total = context_payload["question_count"]
            if args.max_queries is not None:
                question_total = min(question_total, int(args.max_queries))
            validate_context_start({"initial_slot_count": 0})
            context_method_runs: dict[str, dict[str, Any]] = {}
            for method in methods:
                if method == "disabled":
                    for question_index in range(question_total):
                        payload = build_query_payload(context_payload, question_index)
                        with tempfile.TemporaryDirectory() as tmpdir:
                            result = eventqa._run_eventqa_model(
                                args,
                                model,
                                capacity,
                                _query_only_payload(payload),
                                "off",
                            )
                            score = eventqa._score_prediction(
                                args, payload, result["prediction"], tmpdir
                            )
                        record = _record_from_result(
                            method=method,
                            payload=payload,
                            result=result,
                            score=score,
                            cleanup_slot_count=0,
                        )
                        validate_run_invariants(record)
                        records.append(record)
                    continue

                cleanup_slot_count = -1
                frozen_bank = None
                construction_payload = _construction_only_payload(context_payload)
                construction_result = eventqa._run_eventqa_model(
                    args,
                    model,
                    capacity,
                    construction_payload,
                    "on",
                    runtime_bank_config,
                    preserve_bank=True,
                    construction_only=True,
                    recorded_bank_config=runtime_bank_config,
                )
                frozen_bank = construction_result.pop("_retained_bank")
                construction = _construction_record(method, context_payload, construction_result)
                construction_runs.append(construction)
                context_method_runs[method] = construction
                try:
                    for question_index in range(question_total):
                        payload = build_query_payload(context_payload, question_index)
                        with tempfile.TemporaryDirectory() as tmpdir:
                            result = eventqa._run_eventqa_model(
                                args,
                                model,
                                capacity,
                                _query_only_payload(payload),
                                "on",
                                runtime_bank_config,
                                external_bank=frozen_bank,
                                preserve_bank=True,
                                disable_query_retrieval=(method == "p7_no_query_retrieval"),
                                recorded_bank_config=runtime_bank_config,
                            )
                            retained = result.pop("_retained_bank")
                            if retained is not frozen_bank:
                                raise FactConsolidationRunContractError(
                                    "frozen bank identity changed across queries"
                                )
                            score = eventqa._score_prediction(
                                args, payload, result["prediction"], tmpdir
                            )
                        record = _record_from_result(
                            method=method,
                            payload=payload,
                            result=result,
                            score=score,
                            cleanup_slot_count=0,
                        )
                        records.append(record)
                finally:
                    if frozen_bank is not None:
                        frozen_bank.reset()
                        cleanup_slot_count = len(frozen_bank)
                        if cleanup_slot_count != 0:
                            raise FactConsolidationRunContractError(
                                "bank reset failed"
                            )
                        for record in records:
                            if record["context_id"] == context_payload["context_id"] and record["method"] == method:
                                record["post_reset_slot_count"] = cleanup_slot_count
                                record["bank_reset_after_context"] = cleanup_slot_count == 0
                if method == "p7_no_query_retrieval" and "p7" in context_method_runs:
                    validate_no_query_retrieval_construction(
                        context_method_runs["p7"],
                        context_method_runs["p7_no_query_retrieval"],
                    )
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "subtask": args.subtask,
            "methods": methods,
            "records": records,
            "construction_runs": construction_runs,
            "capacity": capacity,
        }
        validate_artifact(artifact)
        _write_json(output_dir / "artifact.json", artifact)
        _write_jsonl(output_dir / "records.jsonl", records)
    finally:
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
