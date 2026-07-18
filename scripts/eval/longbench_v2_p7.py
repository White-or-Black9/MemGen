#!/usr/bin/env python3
"""Frozen three-method LongBench v2 P7 smoke runner."""

from __future__ import annotations

import argparse
from copy import copy
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from scripts.eval import longbench_v2_adapter as adapter
from scripts.eval import longbench_v2_contract as contract
from scripts.eval import longbench_v2_scorer as scorer
from scripts.eval import mab6b_weaver_space_bank_eventqa_65536_n5 as eventqa
from scripts.eval import mab6b_weaver_space_bank_detectiveqa_n10 as weaver_bank


SCHEMA_VERSION = "longbench-v2-p7-smoke/v1"
P7_METHODS = ("p7", "p7_no_query_retrieval")
ALL_METHODS = ("disabled_window_fit", *P7_METHODS)
QUERY_PROMPT_VERSIONS = ("v1", "strict_format_v2", "constrained_choice_v3")
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


class LongBenchV2RunnerError(RuntimeError):
    """Raised when model-facing smoke execution violates its contract."""


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def build_runtime_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint_path=args.checkpoint_path,
        model_path=args.model_path,
        cfg_path=args.cfg_path,
        seed=args.seed,
        generation_max_length=args.generation_max_length,
        constrained_choice=args.query_prompt_version == "constrained_choice_v3",
        retrieve_threshold=0.05,
        update_threshold=0.10,
        max_slots=16,
        top_k=2,
        decay_alpha=0.05,
        eventqa_protocol="frozen_context_bank",
        strict_official_eventqa_prompt=False,
        first_line_official_eventqa_prompt=False,
        bank_transition_diagnostics=False,
        trace_score_decomposition=False,
        save_frozen_bank=False,
        reseed_per_context=False,
    )


def render_query_prompt(item: dict[str, Any], version: str) -> str:
    if version == "v1":
        return adapter.render_memory_query_prompt(item)
    if version == "strict_format_v2":
        return adapter.render_memory_query_prompt(item) + """

Return exactly one line and nothing else:
The correct answer is (A)
Replace A with exactly one of A, B, C, or D.
Do not explain, translate, add punctuation, or add any other text."""
    if version == "constrained_choice_v3":
        return adapter.render_memory_query_prompt(item)
    raise LongBenchV2RunnerError(f"unknown query prompt version: {version}")


def build_payload(
    item: dict[str, Any],
    chunks: Sequence[dict[str, Any]],
    *,
    query_prompt_version: str = "v1",
) -> dict[str, Any]:
    chunk_texts = [chunk["text"] for chunk in chunks]
    memorization_prompts = [
        adapter.render_memorization_prompt(text, chunk_index=index, chunk_count=len(chunk_texts))
        for index, text in enumerate(chunk_texts)
    ]
    return {
        "dataset_config": "longbench_v2",
        "context_id": item["item_id"],
        "context_index": 0,
        "query_id": 0,
        "question_id": item["item_id"],
        "question_type": item["sub_domain"],
        "qa_pair_id": item["item_id"],
        "previous_events": [],
        "chunks": chunk_texts,
        "chunk_token_lengths": [chunk["token_count"] for chunk in chunks],
        "memorization_prompts": memorization_prompts,
        "query_prompt": render_query_prompt(item, query_prompt_version),
        "query_prompt_version": query_prompt_version,
        "question": item["question"],
        "gold_answers": [item["gold_choice"]],
    }


def query_only_payload(payload: dict[str, Any], *, prompt: str | None = None) -> dict[str, Any]:
    result = dict(payload)
    result["chunks"] = []
    result["chunk_token_lengths"] = []
    result["query_prompt"] = prompt if prompt is not None else payload["query_prompt"]
    result["memorization_prompts"] = [result["query_prompt"]]
    return result


def construction_only_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload)


def retrieved_latent_count(result: dict[str, Any]) -> int:
    turns = result.get("retrieved_indices_by_turn", [])
    if not turns:
        return 0
    indices = turns[-1] or []
    return len(indices) * 8


def construction_record(method: str, item: dict[str, Any], payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    pre = result["pre_query_bank_summary"]
    return {
        "item_id": item["item_id"],
        "method": method,
        "query_prompt_version": payload["query_prompt_version"],
        "construction_hash": json_hash({"chunks": payload["chunks"], "prompts": payload["memorization_prompts"]}),
        "construction_bank_write_count": int(pre["write_count"]),
        "construction_final_slot_count": int(pre["slot_count"]),
        "construction_turn_count": len(payload["chunks"]),
    }


def common_record(item: dict[str, Any], method: str, payload: dict[str, Any], prediction: str) -> dict[str, Any]:
    scored = scorer.score_prediction(item, prediction)
    return {
        **scored,
        "method": method,
        "query_prompt_version": payload["query_prompt_version"],
        "capacity_class": item["capacity_class"],
        "prompt_hash": json_hash(payload["query_prompt"]),
        "question_hash": json_hash(item["question"]),
        "choices_hash": json_hash(item["choices"]),
    }


def run_disabled(runtime_args, model, capacity: int, item: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    if item["capacity_class"] != "window_fit":
        raise LongBenchV2RunnerError("Disabled full-context is invalid for over-capacity item")
    full_prompt = adapter.render_prompt(item, include_context=True)
    disabled_payload = query_only_payload(payload, prompt=full_prompt)
    result = eventqa._run_eventqa_model(runtime_args, model, capacity, disabled_payload, "off")
    record = common_record(item, "disabled_window_fit", disabled_payload, result["prediction"])
    generations = result.get("generations", [])
    constrained_choice_active = bool(
        generations and generations[-1].get("constrained_choice", False)
    )
    if getattr(runtime_args, "constrained_choice", False) and not constrained_choice_active:
        raise LongBenchV2RunnerError("constrained choice decoding was not active")
    record.update({
        "construction_hash": None,
        "query_write_count": 0,
        "bank_snapshot_changed_after_query": False,
        "post_reset_slot_count": 0,
        "retrieved_latent_count": 0,
        "bank_created": False,
        "constrained_choice_active": constrained_choice_active,
    })
    return record


def run_p7_method(
    runtime_args,
    model,
    capacity: int,
    item: dict[str, Any],
    payload: dict[str, Any],
    method: str,
    runtime_bank_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    construction_runtime_args = copy(runtime_args)
    construction_runtime_args.constrained_choice = False
    construction = eventqa._run_eventqa_model(
        construction_runtime_args,
        model,
        capacity,
        construction_only_payload(payload),
        "on",
        runtime_bank_config,
        preserve_bank=True,
        construction_only=True,
        recorded_bank_config=runtime_bank_config,
    )
    bank = construction.pop("_retained_bank")
    construction_info = construction_record(method, item, payload, construction)
    try:
        query_result = eventqa._run_eventqa_model(
            runtime_args,
            model,
            capacity,
            query_only_payload(payload),
            "on",
            runtime_bank_config,
            external_bank=bank,
            preserve_bank=True,
            disable_query_retrieval=(method == "p7_no_query_retrieval"),
            recorded_bank_config=runtime_bank_config,
        )
        retained = query_result.pop("_retained_bank")
        if retained is not bank:
            raise LongBenchV2RunnerError("frozen bank identity changed")
        record = common_record(item, method, payload, query_result["prediction"])
        generations = query_result.get("generations", [])
        constrained_choice_active = bool(
            generations and generations[-1].get("constrained_choice", False)
        )
        if getattr(runtime_args, "constrained_choice", False) and not constrained_choice_active:
            raise LongBenchV2RunnerError("constrained choice decoding was not active")
        record.update({
            "construction_hash": construction_info["construction_hash"],
            "construction_bank_write_count": construction_info["construction_bank_write_count"],
            "construction_final_slot_count": construction_info["construction_final_slot_count"],
            "construction_turn_count": construction_info["construction_turn_count"],
            "query_write_count": int(query_result["query_write_count_delta"]),
            "bank_snapshot_changed_after_query": bool(query_result["bank_snapshot_changed_after_query"]),
            "query_read_only_enforced": bool(query_result["query_read_only_enforced"]),
            "retrieved_latent_count": retrieved_latent_count(query_result),
            "retrieved_indices_by_turn": query_result.get("retrieved_indices_by_turn", []),
            "pre_query_bank_summary": query_result["pre_query_bank_summary"],
            "post_query_bank_summary": query_result["post_query_bank_summary"],
            "bank_created": True,
            "constrained_choice_active": constrained_choice_active,
        })
    finally:
        bank.reset()
    record["post_reset_slot_count"] = len(bank)
    if record["post_reset_slot_count"] != 0:
        raise LongBenchV2RunnerError("post-item bank reset failed")
    return record, construction_info


def run_p7_pair(
    runtime_args,
    model,
    capacity: int,
    item: dict[str, Any],
    payload: dict[str, Any],
    runtime_bank_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    construction_runtime_args = copy(runtime_args)
    construction_runtime_args.constrained_choice = False
    construction = eventqa._run_eventqa_model(
        construction_runtime_args,
        model,
        capacity,
        construction_only_payload(payload),
        "on",
        runtime_bank_config,
        preserve_bank=True,
        construction_only=True,
        recorded_bank_config=runtime_bank_config,
    )
    bank = construction.pop("_retained_bank")
    base_construction = construction_record("p7", item, payload, construction)
    records = []
    constructions = []
    try:
        for method in ("p7", "p7_no_query_retrieval"):
            query_result = eventqa._run_eventqa_model(
                runtime_args,
                model,
                capacity,
                query_only_payload(payload),
                "on",
                runtime_bank_config,
                external_bank=bank,
                preserve_bank=True,
                disable_query_retrieval=(method == "p7_no_query_retrieval"),
                recorded_bank_config=runtime_bank_config,
            )
            retained = query_result.pop("_retained_bank")
            if retained is not bank:
                raise LongBenchV2RunnerError("frozen bank identity changed")
            record = common_record(item, method, payload, query_result["prediction"])
            generations = query_result.get("generations", [])
            constrained_choice_active = bool(
                generations and generations[-1].get("constrained_choice", False)
            )
            if getattr(runtime_args, "constrained_choice", False) and not constrained_choice_active:
                raise LongBenchV2RunnerError("constrained choice decoding was not active")
            record.update({
                "construction_hash": base_construction["construction_hash"],
                "construction_bank_write_count": base_construction["construction_bank_write_count"],
                "construction_final_slot_count": base_construction["construction_final_slot_count"],
                "construction_turn_count": base_construction["construction_turn_count"],
                "query_write_count": int(query_result["query_write_count_delta"]),
                "bank_snapshot_changed_after_query": bool(query_result["bank_snapshot_changed_after_query"]),
                "query_read_only_enforced": bool(query_result["query_read_only_enforced"]),
                "retrieved_latent_count": retrieved_latent_count(query_result),
                "retrieved_indices_by_turn": query_result.get("retrieved_indices_by_turn", []),
                "pre_query_bank_summary": query_result["pre_query_bank_summary"],
                "post_query_bank_summary": query_result["post_query_bank_summary"],
                "bank_created": True,
                "constrained_choice_active": constrained_choice_active,
            })
            records.append(record)
            constructions.append({**base_construction, "method": method})
    finally:
        bank.reset()
    if len(bank) != 0:
        raise LongBenchV2RunnerError("post-item bank reset failed")
    for record in records:
        record["post_reset_slot_count"] = 0
    return records, constructions


def aggregate(
    records: Sequence[dict[str, Any]],
    constructions: Sequence[dict[str, Any]],
    *,
    validate_contract: bool = True,
) -> dict[str, Any]:
    methods = sorted({record["method"] for record in records})
    contract_result = (
        contract.validate_aligned_records(records)
        if validate_contract
        else {
            "contract_valid": None,
            "status": "pending_method_shard_merge",
            "item_count": len({record["item_id"] for record in records}),
            "record_count": len(records),
        }
    )
    return {
        "contract": contract_result,
        "methods": methods,
        "metrics": {
            method: scorer.aggregate_scores([row for row in records if row["method"] == method], method=method)
            for method in methods
        },
        "construction_runs": list(constructions),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", default="configs/eval/longbench_v2_p7_smoke_ids.json")
    parser.add_argument("--output-root", default="outputs/longbench_v2/smoke")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--cfg-path", default="configs/latent_memory/triviaqa.yaml")
    parser.add_argument("--chunk-token-budget", type=int, default=8192)
    parser.add_argument("--generation-max-length", type=int, default=12)
    parser.add_argument(
        "--query-prompt-version",
        choices=QUERY_PROMPT_VERSIONS,
        default="v1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        default=",".join(ALL_METHODS),
        help="Comma-separated methods. p7_no_query_retrieval requires p7.",
    )
    parser.add_argument("--item-start", type=int, default=0, help="Inclusive manifest index.")
    parser.add_argument("--item-stop", type=int, help="Exclusive manifest index.")
    return parser


def parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(part.strip() for part in value.split(",") if part.strip())
    if not methods or len(set(methods)) != len(methods):
        raise LongBenchV2RunnerError("methods must be a non-empty unique list")
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        raise LongBenchV2RunnerError(f"unknown methods: {sorted(unknown)}")
    if "p7_no_query_retrieval" in methods and "p7" not in methods:
        raise LongBenchV2RunnerError("p7_no_query_retrieval requires p7")
    return methods


def select_item_slice(items: Sequence[dict[str, Any]], start: int, stop: int | None) -> list[dict[str, Any]]:
    effective_stop = len(items) if stop is None else stop
    if start < 0 or effective_stop < 0 or start >= effective_stop or effective_stop > len(items):
        raise LongBenchV2RunnerError(
            f"invalid item slice [{start}:{effective_stop}] for {len(items)} items"
        )
    return list(items[start:effective_stop])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    methods = parse_methods(args.methods)
    runtime_args = build_runtime_args(args)
    dataset = adapter.load_dataset(args.dataset)
    manifest = adapter.load_manifest(args.manifest)
    all_items = adapter.select_manifest_rows(dataset, manifest)
    items = select_item_slice(all_items, args.item_start, args.item_stop)
    item_stop = args.item_start + len(items)
    method_label = (
        "all" if set(methods) == set(ALL_METHODS)
        else "p7pair" if set(methods) == set(P7_METHODS)
        else "p7only" if set(methods) == {"p7"}
        else "disabled"
    )
    run_id = datetime.now(timezone.utc).strftime(
        f"%Y%m%dT%H%M%SZ-longbench-v2-{method_label}-{args.item_start:02d}-{item_stop:02d}"
    )
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "dataset": adapter.DATASET_ID,
        "dataset_revision": adapter.DATASET_REVISION,
        "source_manifest": str(Path(args.manifest)),
        "methods": list(methods),
        "item_start": args.item_start,
        "item_stop": item_stop,
        "item_ids": [item["item_id"] for item in items],
        "chunk_token_budget": args.chunk_token_budget,
        "generation_max_length": args.generation_max_length,
        "query_prompt_version": args.query_prompt_version,
        "seed": args.seed,
        "git_status_before": subprocess.run(["git", "status", "--short", "--branch"], text=True, capture_output=True, check=True).stdout.strip(),
    })

    model, capacity = weaver_bank._load_model(runtime_args)
    tokenizer = model.tokenizer
    token_count = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    runtime_bank_config = eventqa._eventqa_bank_config(runtime_args)
    records: list[dict[str, Any]] = []
    constructions: list[dict[str, Any]] = []
    try:
        for item_index, item in enumerate(items):
            chunks = adapter.chunk_text(
                item["context"], token_count=token_count, token_budget=args.chunk_token_budget,
            )
            payload = build_payload(
                item,
                chunks,
                query_prompt_version=args.query_prompt_version,
            )
            if "disabled_window_fit" in methods and item["capacity_class"] == "window_fit":
                records.append(run_disabled(runtime_args, model, capacity, item, payload))
            if set(P7_METHODS).issubset(methods):
                item_records, item_constructions = run_p7_pair(
                    runtime_args, model, capacity, item, payload, runtime_bank_config,
                )
                records.extend(item_records)
                constructions.extend(item_constructions)
            elif "p7" in methods:
                item_record, item_construction = run_p7_method(
                    runtime_args,
                    model,
                    capacity,
                    item,
                    payload,
                    "p7",
                    runtime_bank_config,
                )
                records.append(item_record)
                constructions.append(item_construction)
            write_jsonl(output_dir / "records.partial.jsonl", records)
            print(f"completed item {item_index + 1}/{len(items)} {item['item_id']}", flush=True)
        summary = aggregate(
            records,
            constructions,
            validate_contract=set(methods) == set(ALL_METHODS),
        )
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "capacity": capacity,
            "records": records,
            **summary,
        }
        write_json(output_dir / "artifact.json", artifact)
        write_jsonl(output_dir / "records.jsonl", records)
        return 0
    finally:
        del model


if __name__ == "__main__":
    raise SystemExit(main())
