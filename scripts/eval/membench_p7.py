#!/usr/bin/env python3
"""Paired MemBench runner with frozen latent-bank query retrieval.

The runner consumes normalized MemBench FirstAgent trajectories.  It never
calls the source benchmark's text-memory ``recall`` API: construction turns
are written into the latent bank, then the same frozen bank is used for P7 and
the no-query-retrieval ablation.  The official target remains exact equality
of one option letter (A/B/C/D).
"""

from __future__ import annotations

import argparse
from copy import copy
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import mab6b_weaver_space_bank_detectiveqa_n10 as weaver_bank
from scripts.eval import mab6b_weaver_space_bank_eventqa_65536_n5 as eventqa
from scripts.eval import membench_adapter as adapter


SCHEMA_VERSION = "membench-p7-run/v1"
METHODS = ("no_memory", "text_full_history", "p7", "p7_no_query_retrieval")
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


class MemBenchRunContractError(RuntimeError):
    """Raised when a paired MemBench run is not comparable."""


OFFICIAL_INITIAL_INSTRUCTION = (
    "Please help me record the following information. If there are any questions "
    "within the information, please help me answer them."
)
OFFICIAL_QUERY_TEMPLATE = """Please answer the following question based on past memories of your'conversation with the user.
Past memory: {memory}
Question: (current time is {time}) {question}
Choices:
A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}
Please output the correct option for the question, only one corresponding letter, without any other messages.
Example: D
"""
MEMGEN_CHOICE_GRAMMAR = re.compile(r"^The correct answer is \(([ABCD])\)$")


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def render_construction_prompt(turn: dict[str, Any], *, is_first_turn: bool) -> str:
    """Mirror the environment's initial record instruction and message stream."""
    prefix = f"{OFFICIAL_INITIAL_INSTRUCTION}\n" if is_first_turn else ""
    return f"{prefix}{turn['content']}"


def render_query_prompt(record: dict[str, Any], *, memory_text: str = "") -> str:
    """Render the source repository's FirstAgent query template verbatim."""
    query = record["query"]
    return OFFICIAL_QUERY_TEMPLATE.format(
        memory=memory_text,
        time=query.get("question_time", ""),
        question=query["question"],
        choice_A=query["choices"]["A"],
        choice_B=query["choices"]["B"],
        choice_C=query["choices"]["C"],
        choice_D=query["choices"]["D"],
    )


def build_payload(record: dict[str, Any]) -> dict[str, Any]:
    turns = list(record["construction_turns"])
    return {
        "dataset_config": "membench_firstagent",
        "context_id": record["context_id"],
        "context_index": 0,
        "query_id": record["query"]["query_id"],
        "question_id": record["query"]["query_id"],
        "question_type": record["category"],
        "qa_pair_id": record["query"]["query_id"],
        "previous_events": [],
        "chunks": [turn["content"] for turn in turns],
        "chunk_token_lengths": [],
        "memorization_prompts": [
            render_construction_prompt(turn, is_first_turn=index == 0)
            for index, turn in enumerate(turns)
        ],
        # P7 receives the same visible template as no-memory/no-query.  Its
        # memory enters only through the frozen latent-bank path.
        "query_prompt": render_query_prompt(record),
        "question": record["query"]["question"],
        "gold_answers": [record["query"]["gold_choice"]],
    }


def full_history_memory_text(record: dict[str, Any]) -> str:
    """Explicit-text control: the same sequential messages supplied to P7."""
    return "\n\n".join(turn["content"] for turn in record["construction_turns"])


def query_only_payload(payload: dict[str, Any], *, prompt: str | None = None) -> dict[str, Any]:
    result = dict(payload)
    result["chunks"] = []
    result["chunk_token_lengths"] = []
    result["query_prompt"] = prompt if prompt is not None else payload["query_prompt"]
    result["memorization_prompts"] = [result["query_prompt"]]
    return result


def construction_record(record: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    pre = result["pre_query_bank_summary"]
    return {
        "context_id": record["context_id"],
        "construction_hash": json_hash(
            [turn["content"] for turn in record["construction_turns"]]
        ),
        "construction_bank_write_count": int(pre["write_count"]),
        "construction_final_slot_count": int(pre["slot_count"]),
        "construction_turn_count": len(record["construction_turns"]),
    }


def decode_action_response(raw_generation: str) -> tuple[str, str]:
    """Produce the official action letter from MemGen's exact choice grammar.

    The source agent applies its JSON-schema decoder before assigning
    ``action['response']``.  MemGen's deterministic constrained-choice decoder
    instead emits ``The correct answer is (X)``.  This accepts only that exact
    grammar (or an already bare letter); it does not use relaxed text matching.
    """
    if raw_generation in adapter.CHOICE_KEYS:
        return raw_generation, "bare_letter"
    match = MEMGEN_CHOICE_GRAMMAR.fullmatch(raw_generation)
    if match:
        return match.group(1), "memgen_exact_choice_grammar"
    return "", "invalid_raw_generation"


def result_record(record: dict[str, Any], method: str, raw_generation: str, *, construction: dict | None = None, query_result: dict | None = None) -> dict[str, Any]:
    action_response, action_adapter = decode_action_response(raw_generation)
    scored = adapter.score_choice(action_response, record["query"]["gold_choice"])
    output = {
        "context_id": record["context_id"],
        "trajectory_id": record["trajectory_id"],
        "query_id": record["query"]["query_id"],
        "category": record["category"],
        "method": method,
        **scored,
        "raw_generation": raw_generation,
        "action_response": action_response,
        "action_adapter": action_adapter,
        "query_prompt_hash": json_hash(render_query_prompt(record)),
        "construction_hash": None,
        "bank_created": False,
        "retrieved_latent_count": 0,
        "query_write_count": 0,
        "bank_snapshot_changed_after_query": False,
        "query_read_only_enforced": True,
        "post_reset_slot_count": 0,
    }
    if construction is not None and query_result is not None:
        generations = query_result.get("generations", [])
        final_indices = generations[-1].get("retrieved_indices", []) if generations else []
        output.update({
            **construction,
            "bank_created": True,
            "retrieved_latent_count": len(final_indices) * 8,
            "retrieved_indices": list(final_indices),
            "query_write_count": int(query_result["query_write_count_delta"]),
            "bank_snapshot_changed_after_query": bool(query_result["bank_snapshot_changed_after_query"]),
            "query_read_only_enforced": bool(query_result["query_read_only_enforced"]),
            "pre_query_bank_summary": query_result["pre_query_bank_summary"],
            "post_query_bank_summary": query_result["post_query_bank_summary"],
        })
    return output


def validate_paired_records(rows: Sequence[dict[str, Any]]) -> None:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["context_id"], {})[row["method"]] = row
    for context_id, methods in grouped.items():
        if set(methods) != set(METHODS):
            raise MemBenchRunContractError(f"incomplete method set for {context_id}")
        p7, no_query = methods["p7"], methods["p7_no_query_retrieval"]
        for field in ("construction_hash", "construction_bank_write_count", "construction_final_slot_count"):
            if p7[field] != no_query[field]:
                raise MemBenchRunContractError(f"construction mismatch for {context_id}: {field}")
        if no_query["retrieved_latent_count"] != 0:
            raise MemBenchRunContractError(f"no-query retrieval was active for {context_id}")
        for method, row in methods.items():
            if row["query_write_count"] != 0 or row["bank_snapshot_changed_after_query"]:
                raise MemBenchRunContractError(f"query mutation detected for {context_id}/{method}")
            if row["post_reset_slot_count"] != 0:
                raise MemBenchRunContractError(f"bank reset failed for {context_id}/{method}")


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        output[method] = {
            "count": len(group),
            "official_choice_exact_match": fmean(row["official_choice_exact_match"] for row in group) if group else 0.0,
            "invalid_choice_output_count": sum(not row["valid_choice_output"] for row in group),
            "raw_generation_adapter_failure_count": sum(
                row.get("action_adapter") == "invalid_raw_generation" for row in group
            ),
            "mean_retrieved_latent_count": fmean(row["retrieved_latent_count"] for row in group) if group else 0.0,
        }
    return output


def build_runtime_args(args) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint_path=args.checkpoint_path,
        model_path=args.model_path,
        cfg_path=args.cfg_path,
        seed=args.seed,
        generation_max_length=args.generation_max_length,
        constrained_choice=True,
        retrieve_threshold=args.retrieve_threshold,
        update_threshold=args.update_threshold,
        max_slots=args.max_slots,
        top_k=args.top_k,
        decay_alpha=0.05,
        eventqa_protocol="frozen_context_bank",
        strict_official_eventqa_prompt=False,
        first_line_official_eventqa_prompt=False,
        bank_transition_diagnostics=False,
        trace_score_decomposition=False,
        save_frozen_bank=False,
        reseed_per_context=False,
    )


def run_p7_pair(runtime_args, model, capacity: int, record: dict[str, Any], bank_config: dict[str, Any]) -> list[dict[str, Any]]:
    payload = build_payload(record)
    construction_args = copy(runtime_args)
    construction_args.constrained_choice = False
    construction_result = eventqa._run_eventqa_model(
        construction_args, model, capacity, payload, "on", bank_config,
        preserve_bank=True, construction_only=True, recorded_bank_config=bank_config,
    )
    bank = construction_result.pop("_retained_bank")
    construction = construction_record(record, construction_result)
    rows = []
    try:
        for method in ("p7", "p7_no_query_retrieval"):
            query_result = eventqa._run_eventqa_model(
                runtime_args, model, capacity, query_only_payload(payload), "on", bank_config,
                external_bank=bank, preserve_bank=True,
                disable_query_retrieval=(method == "p7_no_query_retrieval"),
                recorded_bank_config=bank_config,
            )
            retained = query_result.pop("_retained_bank")
            if retained is not bank:
                raise MemBenchRunContractError("frozen bank identity changed")
            rows.append(result_record(record, method, str(query_result["prediction"]), construction=construction, query_result=query_result))
    finally:
        bank.reset()
    if len(bank) != 0:
        raise MemBenchRunContractError("bank reset failed")
    for row in rows:
        row["post_reset_slot_count"] = 0
    return rows


def run_disabled(runtime_args, model, capacity: int, record: dict[str, Any]) -> dict[str, Any]:
    payload = build_payload(record)
    result = eventqa._run_eventqa_model(
        runtime_args, model, capacity, query_only_payload(payload), "off"
    )
    return result_record(record, "no_memory", str(result["prediction"]))


def run_text_full_history(runtime_args, model, capacity: int, record: dict[str, Any]) -> dict[str, Any]:
    """Same-backbone explicit-memory control using the official memory field."""
    payload = build_payload(record)
    full_text_prompt = render_query_prompt(
        record, memory_text=full_history_memory_text(record)
    )
    result = eventqa._run_eventqa_model(
        runtime_args, model, capacity, query_only_payload(payload, prompt=full_text_prompt), "off"
    )
    return result_record(record, "text_full_history", str(result["prediction"]))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--subset", required=True, choices=sorted(adapter.SUPPORTED_SUBSETS))
    parser.add_argument("--output-root", default="outputs/membench")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--cfg-path", default="configs/latent_memory/triviaqa.yaml")
    parser.add_argument("--max-items", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation-max-length", type=int, default=12)
    parser.add_argument("--retrieve-threshold", type=float, default=0.05)
    parser.add_argument("--update-threshold", type=float, default=0.10)
    parser.add_argument("--max-slots", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Write a deterministic payload manifest without loading a model.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = adapter.normalize_dataset(adapter.load_dataset(args.dataset))
    selected = records[: args.max_items]
    if not selected:
        raise MemBenchRunContractError("selected item set is empty")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-membench-" + args.subset)
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "subset": args.subset,
        "dataset_path": str(args.dataset),
        "dataset_sha256": adapter.sha256_file(args.dataset),
        "methods": list(METHODS),
        "selected_context_ids": [record["context_id"] for record in selected],
        "max_items": args.max_items,
        "metric": "official_choice_exact_match",
        "scoring_contract": "strict action_response == gold_choice; no relaxed text matching",
        "query_protocol": {
            "source_template": "MembenchAgent.py:INSTRUCTION_FIRST",
            "visible_memory_text": {
                "no_memory": "empty",
                "text_full_history": "all sequential construction messages",
                "p7": "empty; memory supplied only through latent bank",
                "p7_no_query_retrieval": "empty; constructed latent bank is not read",
            },
            "query_time_preserved": True,
            "choice_format_preserved": "A./B./C./D.",
        },
        "action_protocol": {
            "source": "official agent JSON schema extracts action['response'] as A/B/C/D",
            "memgen_adapter": "exact constrained grammar The correct answer is (X) -> X only",
        },
        "p7_protocol": "construct sequentially, freeze before one query, block query writes; latent retrieval replaces text memory.recall",
        "dry_run": args.dry_run,
        "git_status_before": subprocess.run(["git", "status", "--short", "--branch"], text=True, capture_output=True, check=True).stdout.strip(),
    }
    write_json(output_dir / "manifest.json", manifest)
    if args.dry_run:
        write_json(output_dir / "payloads.json", {record["context_id"]: build_payload(record) for record in selected})
        return 0

    runtime_args = build_runtime_args(args)
    model, capacity = weaver_bank._load_model(runtime_args)
    bank_config = eventqa._eventqa_bank_config(runtime_args)
    rows: list[dict[str, Any]] = []
    try:
        for index, record in enumerate(selected, start=1):
            rows.append(run_disabled(runtime_args, model, capacity, record))
            rows.append(run_text_full_history(runtime_args, model, capacity, record))
            rows.extend(run_p7_pair(runtime_args, model, capacity, record, bank_config))
            write_jsonl(output_dir / "records.partial.jsonl", rows)
            print(f"completed {index}/{len(selected)} {record['context_id']}", flush=True)
        validate_paired_records(rows)
        write_jsonl(output_dir / "records.jsonl", rows)
        write_json(output_dir / "artifact.json", {
            **manifest,
            "dry_run": False,
            "capacity": capacity,
            "records": rows,
            "metrics": aggregate(rows),
            "contract_valid": True,
        })
    finally:
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
