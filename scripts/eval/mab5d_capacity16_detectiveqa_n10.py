"""MAB-5D: detective_qa capacity16 decoupled retrieval-update thresholds on 10 contexts."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import tempfile
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import mab3_bank_on_full_history as mab3
from scripts.eval import mab5a_detectiveqa_compressed_n10 as base


EXPERIMENT_NAME = "MAB-5D: detective_qa Capacity16 Decoupled Retrieval-Update Thresholds n10"
RUN_PREFIX = "detectiveqa-capacity16-n10"
SPLIT = "Long_Range_Understanding"
SUB_DATASET = "detective_qa"
DATA_CONFIG = "configs/data_conf/Long_Range_Understanding/Detective_QA.yaml"
DEFAULT_OUTPUT_ROOT = "outputs/mab/capacity16_detectiveqa_n10/"
DEFAULT_REQUESTED_CONTEXTS = 10
DEFAULT_CHUNK_SIZE = 4096
DEFAULT_THRESHOLD = 0.03
DEFAULT_RETRIEVE_THRESHOLD = 0.03
DEFAULT_UPDATE_THRESHOLD = 0.05
DEFAULT_TOP_K = 1
DEFAULT_MAX_SLOTS = 16
DEFAULT_RETRIEVE_POLICY = "threshold_topk"
DEFAULT_UPDATE_POLICY = "thread_update"
DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant that can read the context and memorize it for future retrieval."
DEFAULT_ACK = "Acknowledged."

MAB5A_BASELINE = {
    "threshold": 0.03,
    "max_slots": 8,
    "final_slot_counts": [1, 2, 2, 5, 6, 5, 6, 7, 4, 7],
    "mean_final_slot_count": 4.5,
    "retrieved_latent_count": 2248,
    "output_changed": 10,
    "exact_match": 0.0,
    "query_turn_retrieval_active": "context-level 10/10; query-turn not separated",
}
MAB5B_BASELINE = {
    "threshold": 0.05,
    "max_slots": 8,
    "final_slot_counts": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    "mean_final_slot_count": 8.0,
    "retrieved_latent_count": 200,
    "output_changed": 5,
    "exact_match": 0.0,
    "query_turn_retrieval_active": "5/10",
}

_BASE_BUILD_ROW = base._build_row


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _bridge_script() -> Path:
    return Path(base.__file__).with_name("mab2_mab_bridge.py")


def _mab_env() -> dict:
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    return env


def select_match_indices(total_matches: int, requested: int) -> list[int]:
    return list(range(min(total_matches, requested)))


def count_context_matches(parquet_path: str, sub_dataset: str) -> int:
    rows = pq.read_table(parquet_path).to_pylist()
    return sum(1 for row in rows if row.get("metadata", {}).get("source") == sub_dataset)


def _prepare_payload(args, output_path: Path, match_index: int, timestamp: str) -> dict:
    command = [
        args.mab_python,
        str(_bridge_script()),
        "prepare",
        "--mab-repo", args.mab_repo,
        "--output", str(output_path),
        "--parquet", args.parquet,
        "--data-config", args.data_config,
        "--sub-dataset", SUB_DATASET,
        "--chunk-size", str(DEFAULT_CHUNK_SIZE),
        "--timestamp", timestamp,
        "--match-index", str(match_index),
    ]
    subprocess.run(command, check=True, env=_mab_env())
    return _load_json(output_path)


def _score_prediction(args, payload: dict, prediction: str, tmpdir: str) -> dict:
    request_path = Path(tmpdir) / "score_request.json"
    output_path = Path(tmpdir) / "score_output.json"
    _write_json(
        request_path,
        {
            "prediction": prediction,
            "gold_answers": payload["gold_answers"],
            "dataset_config": payload["dataset_config"],
        },
    )
    command = [
        args.mab_python,
        str(_bridge_script()),
        "score",
        "--mab-repo", args.mab_repo,
        "--output", str(output_path),
        "--input", str(request_path),
    ]
    subprocess.run(command, check=True, env=_mab_env())
    return _load_json(output_path)


def render_compressed_query_messages(init_prompt, inter_history):
    if not init_prompt:
        raise RuntimeError("Missing init prompt for compressed query")
    if not inter_history or inter_history[-1].get("role") != "user":
        raise RuntimeError("Compressed query turn missing current user query")
    system_message = init_prompt[0]
    if system_message.get("role") != "system":
        raise RuntimeError("Expected system message at init prompt index 0")
    return [
        system_message,
        {"role": "user", "content": inter_history[-1]["content"]},
    ]


def prompt_contains_chunk_leak(prompt_text, chunks, *, window=128, step=64):
    return base.prompt_contains_chunk_leak(prompt_text, chunks, window=window, step=step)


def _render_chat(tokenizer, messages) -> tuple[int, str]:
    rendered_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    rendered_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"]
    return int(rendered_ids.shape[1]), rendered_text


def estimate_full_history_query_tokens(tokenizer, payload: dict) -> int:
    messages = [{"role": "system", "content": DEFAULT_SYSTEM_MESSAGE}]
    for prompt in payload["memorization_prompts"]:
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": DEFAULT_ACK})
    messages.append({"role": "user", "content": payload["query_prompt"]})
    tokens, _ = _render_chat(tokenizer, [messages])
    return tokens


def compressed_query_token_count(tokenizer, payload: dict) -> tuple[int, str, bool, bool]:
    messages = render_compressed_query_messages(
        [{"role": "system", "content": DEFAULT_SYSTEM_MESSAGE}, {"role": "user", "content": payload["memorization_prompts"][0] if payload["memorization_prompts"] else ""}],
        [{"role": "user", "content": payload["query_prompt"]}],
    )
    token_count, rendered_text = _render_chat(tokenizer, [messages])
    chunk_leak = prompt_contains_chunk_leak(rendered_text, payload["chunks"])
    ack_leak = DEFAULT_ACK in rendered_text
    return token_count, rendered_text, chunk_leak, ack_leak


def _bank_config():
    config = mab3.version_a_bank_config(
        top_k=DEFAULT_TOP_K,
        threshold=DEFAULT_THRESHOLD,
        retrieve_policy=DEFAULT_RETRIEVE_POLICY,
    )
    # MAB-5D capacity ablation: this is the actual runtime bank capacity.
    # Do not rely only on run_config/default metadata; force the value into
    # LatentMemoryBankConfig construction.
    config["max_slots"] = DEFAULT_MAX_SLOTS
    config["retrieve_threshold"] = DEFAULT_RETRIEVE_THRESHOLD
    config["update_threshold"] = DEFAULT_UPDATE_THRESHOLD
    config["update_policy"] = DEFAULT_UPDATE_POLICY
    if int(config.get("max_slots", -1)) != DEFAULT_MAX_SLOTS:
        raise RuntimeError(
            f"MAB-5D expected max_slots={DEFAULT_MAX_SLOTS}, "
            f"but bank config has max_slots={config.get('max_slots')}"
        )
    return config


def _build_manifest(run_id: str, args, started_at: str, *, git_status_before: str, git_status_after: str | None = None) -> dict:
    checkpoint_path = str(Path(args.checkpoint_path).resolve())
    return {
        "experiment_name": EXPERIMENT_NAME,
        "run_id": run_id,
        "timestamp": started_at,
        "dataset_root": args.dataset_root,
        "mab_repo": args.mab_repo,
        "split": SPLIT,
        "subtask": SUB_DATASET,
        "checkpoint_path": checkpoint_path,
        "model_checkpoint_id": args.model_checkpoint_id,
        "context_capacity": None,
        "query_mode": "first-query-only",
        "metric": "exact_match",
        "threshold": DEFAULT_THRESHOLD,
        "retrieve_threshold": DEFAULT_RETRIEVE_THRESHOLD,
        "update_threshold": DEFAULT_UPDATE_THRESHOLD,
        "max_slots": DEFAULT_MAX_SLOTS,
        "configured_max_slots": DEFAULT_MAX_SLOTS,
        "actual_bank_max_slots": DEFAULT_MAX_SLOTS,
        "mechanism": "capacity16_decoupled_retrieval_update_thresholds",
        "num_contexts_requested": DEFAULT_REQUESTED_CONTEXTS,
        "num_contexts_attempted": 0,
        "num_contexts_valid": 0,
        "num_contexts_invalid": 0,
        "full_history_policy": "over_capacity_invalid",
        "compressed_bank_off_accuracy": None,
        "compressed_bank_on_accuracy": None,
        "delta_accuracy": None,
        "num_output_changed": 0,
        "num_improved": 0,
        "num_regressed": 0,
        "num_retrieval_active": 0,
        "avg_estimated_full_history_query_tokens": None,
        "avg_compressed_query_tokens": None,
        "avg_retrieved_latents": None,
        "latency_summary": None,
        "peak_cuda_memory_summary": None,
        "git_status_before": git_status_before,
        "git_status_after": git_status_after,
        "started_at": started_at,
        "finished_at": None,
        "stop_reason": None,
    }


def _build_row(
    *,
    run_id: str,
    context_index: int,
    payload: dict,
    bank_off_result: dict,
    bank_on_result: dict,
    bank_off_score: dict,
    bank_on_score: dict,
    estimated_full_history_query_tokens: int,
    compressed_query_tokens_bank_off: int,
    compressed_query_tokens_bank_on: int,
) -> dict:
    base_row = _BASE_BUILD_ROW(
        run_id=run_id,
        context_index=context_index,
        payload=payload,
        bank_off_result=bank_off_result,
        bank_on_result=bank_on_result,
        bank_off_score=bank_off_score,
        bank_on_score=bank_on_score,
        estimated_full_history_query_tokens=estimated_full_history_query_tokens,
        compressed_query_tokens_bank_off=compressed_query_tokens_bank_off,
        compressed_query_tokens_bank_on=compressed_query_tokens_bank_on,
    )
    final_generation = bank_on_result["generations"][-1]
    bank_debug = final_generation.get("bank_debug") or {}
    query_turn_retrieved_indices = list(final_generation.get("retrieved_indices", []))
    query_turn_retrieved_scores = list(final_generation.get("retrieved_scores", []))
    base_row.update(
        {
            "construction_time_retrieval_count": max(0, int(bank_on_result["bank_retrieval_count"]) - 1),
            "query_turn_retrieval_active": bool(final_generation.get("retrieved_latent_count", 0)),
            "query_turn_retrieved_latent_count": int(final_generation.get("retrieved_latent_count", 0)),
            "query_turn_retrieved_indices": query_turn_retrieved_indices,
            "query_turn_retrieved_scores": query_turn_retrieved_scores,
            "query_turn_retrieved_score_range": (
                [min(query_turn_retrieved_scores), max(query_turn_retrieved_scores)]
                if query_turn_retrieved_scores
                else None
            ),
            "bank_on_effective_retrieve_threshold": bank_debug.get("effective_retrieve_threshold"),
            "bank_on_effective_update_threshold": bank_debug.get("effective_update_threshold"),
            "configured_max_slots": DEFAULT_MAX_SLOTS,
            "actual_bank_max_slots": int(bank_debug.get("max_slots", DEFAULT_MAX_SLOTS)),
            "bank_on_retrieve_threshold_passed": (bank_debug.get("last_write_back") or {}).get("retrieve_threshold_passed"),
            "bank_on_update_threshold_passed": (bank_debug.get("last_write_back") or {}).get("update_threshold_passed"),
            "bank_on_write_action_counts": bank_debug.get("write_action_counts", {}),
            "bank_on_update_reason_counts": bank_debug.get("update_reason_counts", {}),
            "bank_on_append_insert_count": bank_debug.get("thread_insert_count", 0),
            "bank_on_matched_replace_count": bank_debug.get("matched_replace_count", 0),
            "bank_on_capacity_evict_count": bank_debug.get("capacity_evict_count", 0),
            "bank_on_write_count": bank_on_result["bank_write_count"],
            "bank_on_retrieval_count": bank_on_result["bank_retrieval_count"],
            "bank_on_retrieved_latent_count": bank_on_result["bank_retrieved_latent_count"],
            "bank_on_query_write_count": bank_on_result["query_write_count"],
            "bank_on_query_write_attempt_count": bank_on_result["query_write_attempt_count"],
        }
    )
    return base_row


def _aggregate(rows: list[dict]) -> dict:
    valid = [row for row in rows if not row.get("error_or_stop_reason")]
    invalid = [row for row in rows if row.get("error_or_stop_reason")]
    valid_n = len(valid)
    bank_off_correct = sum(int(bool(row["bank_off_exact_match"])) for row in valid)
    bank_on_correct = sum(int(bool(row["bank_on_exact_match"])) for row in valid)
    changed = sum(int(bool(row["output_changed"])) for row in valid)
    improved = sum(int(bool(row["improved"])) for row in valid)
    regressed = sum(int(bool(row["regressed"])) for row in valid)
    retrieval_active = sum(int(row["bank_on_retrieval_count"] > 0) for row in valid)
    query_turn_retrieval_active = sum(int(bool(row.get("query_turn_retrieval_active"))) for row in valid)
    final_slot_counts = [int(row["bank_slot_count_final_before_reset"]) for row in valid if row.get("bank_slot_count_final_before_reset") is not None]
    write_action_counts = Counter()
    update_reason_counts = Counter()
    for row in valid:
        write_action_counts.update(row.get("bank_on_write_action_counts", {}))
        update_reason_counts.update(row.get("bank_on_update_reason_counts", {}))

    def avg(key):
        return (sum(float(row[key]) for row in valid) / valid_n) if valid_n else None

    def mean_time(key):
        values = [float(row[key]) for row in valid if row.get(key) is not None]
        return {
            "count": len(values),
            "mean": (sum(values) / len(values)) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    summary = {
        "num_contexts_requested": DEFAULT_REQUESTED_CONTEXTS,
        "num_contexts_attempted": len(rows),
        "num_contexts_valid": valid_n,
        "configured_max_slots": DEFAULT_MAX_SLOTS,
        "actual_bank_max_slots": DEFAULT_MAX_SLOTS,
        "num_contexts_invalid": len(invalid),
        "compressed_bank_off_accuracy": (bank_off_correct / valid_n) if valid_n else None,
        "compressed_bank_on_accuracy": (bank_on_correct / valid_n) if valid_n else None,
        "delta_accuracy": ((bank_on_correct - bank_off_correct) / valid_n) if valid_n else None,
        "num_output_changed": changed,
        "num_improved": improved,
        "num_regressed": regressed,
        "num_retrieval_active": retrieval_active,
        "num_query_turn_retrieval_active": query_turn_retrieval_active,
        "final_slot_counts": final_slot_counts,
        "mean_final_slot_count": (sum(final_slot_counts) / len(final_slot_counts)) if final_slot_counts else None,
        "total_write_count": sum(int(row["bank_on_write_count"]) for row in valid),
        "total_retrieval_count": sum(int(row["bank_on_retrieval_count"]) for row in valid),
        "total_retrieved_latent_count": sum(int(row["bank_on_retrieved_latent_count"]) for row in valid),
        "construction_time_retrieval_count": sum(int(row.get("construction_time_retrieval_count", 0)) for row in valid),
        "query_turn_retrieved_latent_count": sum(int(row.get("query_turn_retrieved_latent_count", 0)) for row in valid),
        "query_write_count": sum(int(row["bank_on_query_write_count"]) for row in valid),
        "query_write_attempt_count": sum(int(row["bank_on_query_write_attempt_count"]) for row in valid),
        "cross_context_leakage_detected": any(bool(row["cross_context_leakage_detected"]) for row in valid),
        "retrieved_latents_enter_reasoner": all(bool(row["retrieved_latents_enter_reasoner"]) for row in valid),
        "retrieved_latents_enter_weaver": any(bool(row["retrieved_latents_enter_weaver"]) for row in valid),
        "write_action_counts": dict(write_action_counts),
        "update_reason_counts": dict(update_reason_counts),
        "append_insert_count": int(write_action_counts.get("insert", 0)),
        "matched_replace_count": int(sum(int(row["bank_on_matched_replace_count"]) for row in valid)),
        "capacity_evict_count": int(sum(int(row["bank_on_capacity_evict_count"]) for row in valid)),
        "query_turn_retrieved_indices": [list(row.get("query_turn_retrieved_indices", [])) for row in valid],
        "query_turn_retrieved_scores": [list(row.get("query_turn_retrieved_scores", [])) for row in valid],
        "query_turn_retrieved_score_range": [
            row.get("query_turn_retrieved_score_range") for row in valid
        ],
        "avg_estimated_full_history_query_tokens": avg("estimated_full_history_query_tokens"),
        "avg_compressed_query_tokens": avg("compressed_query_tokens_bank_on"),
        "avg_retrieved_latents": avg("bank_on_retrieved_latent_count"),
        "latency_summary": mean_time("latency_seconds"),
        "peak_cuda_memory_summary": {
            "count": len([row for row in valid if row.get("peak_cuda_memory") is not None]),
            "max": max((row["peak_cuda_memory"] for row in valid if row.get("peak_cuda_memory") is not None), default=None),
            "mean": (
                sum(float(row["peak_cuda_memory"]) for row in valid if row.get("peak_cuda_memory") is not None)
                / len([row for row in valid if row.get("peak_cuda_memory") is not None])
            ) if [row for row in valid if row.get("peak_cuda_memory") is not None] else None,
        },
    }
    summary["compare_against_mab5a"] = {
        "baseline": MAB5A_BASELINE,
        "delta_mean_final_slot_count": (
            summary["mean_final_slot_count"] - MAB5A_BASELINE["mean_final_slot_count"]
            if summary["mean_final_slot_count"] is not None
            else None
        ),
        "delta_retrieved_latent_count": (
            summary["total_retrieved_latent_count"] - MAB5A_BASELINE["retrieved_latent_count"]
            if summary["total_retrieved_latent_count"] is not None
            else None
        ),
    }
    summary["compare_against_mab5b"] = {
        "baseline": MAB5B_BASELINE,
        "delta_mean_final_slot_count": (
            summary["mean_final_slot_count"] - MAB5B_BASELINE["mean_final_slot_count"]
            if summary["mean_final_slot_count"] is not None
            else None
        ),
        "delta_retrieved_latent_count": (
            summary["total_retrieved_latent_count"] - MAB5B_BASELINE["retrieved_latent_count"]
            if summary["total_retrieved_latent_count"] is not None
            else None
        ),
    }
    return summary


def _build_research_note(
    *,
    output_dir: Path,
    manifest: dict,
    rows: list[dict],
    summary: dict,
    git_status_before: str,
    git_status_after: str,
) -> Path:
    note_path = Path("research_notes/benchmarks/memoryagentbench_mab5d_capacity16.md")
    valid_rows = [row for row in rows if not row.get("error_or_stop_reason")]
    lines = [
        "# MAB-5D: detective_qa Capacity16 Decoupled Retrieval-Update Thresholds n10",
        "",
        "## Purpose",
        "Test whether increasing memory capacity to 16 slots reduces eviction / memory churn while keeping MAB-5C decoupled retrieval-update behavior.",
        "",
        "## Settings",
        f"- Split: `{SPLIT}`",
        f"- Subtask: `{SUB_DATASET}`",
        f"- Contexts: {DEFAULT_REQUESTED_CONTEXTS} local rows",
        f"- threshold: `{DEFAULT_THRESHOLD}`",
        f"- retrieve_threshold: `{DEFAULT_RETRIEVE_THRESHOLD}`",
        f"- update_threshold: `{DEFAULT_UPDATE_THRESHOLD}`",
        f"- top_k: `{DEFAULT_TOP_K}`",
        f"- max_slots: `{DEFAULT_MAX_SLOTS}`",
        f"- retrieve_policy: `{DEFAULT_RETRIEVE_POLICY}`",
        f"- update_policy: `{DEFAULT_UPDATE_POLICY}`",
        "- query mode: first-query-only",
        "- query phase: read-only",
        "- full-history detective_qa: `over_capacity_invalid`",
        "",
        "## Protocol Notes",
        "- Shared-threshold behavior remains the default when both split thresholds are `None`.",
        "- Retrieval visibility uses `retrieve_threshold`.",
        "- Write-back matching uses `update_threshold`.",
        "- Retrieved memory stays Reasoner-only and does not enter Weaver.",
        "- No fallback mechanism was introduced.",
        "",
        "## Run Status",
        f"- Output directory: `{output_dir}`",
        f"- Valid contexts: `{summary['num_contexts_valid']}` / `{summary['num_contexts_requested']}`",
        f"- Bank-off exact match: `{summary['compressed_bank_off_accuracy']}`",
        f"- Bank-on exact match: `{summary['compressed_bank_on_accuracy']}`",
        f"- Delta accuracy: `{summary['delta_accuracy']}`",
        f"- Output changed: `{summary['num_output_changed']}`",
        f"- Retrieval active contexts: `{summary['num_retrieval_active']}`",
        f"- Query-turn retrieval active contexts: `{summary['num_query_turn_retrieval_active']}`",
        f"- Final slot counts: `{summary['final_slot_counts']}`",
        f"- Mean final slot count: `{summary['mean_final_slot_count']}`",
        f"- Total write count: `{summary['total_write_count']}`",
        f"- Total retrieval count: `{summary['total_retrieval_count']}`",
        f"- Total retrieved latent count: `{summary['total_retrieved_latent_count']}`",
        f"- Construction-time retrieval count: `{summary['construction_time_retrieval_count']}`",
        f"- Query-turn retrieved latent count: `{summary['query_turn_retrieved_latent_count']}`",
        f"- Query write count: `{summary['query_write_count']}`",
        f"- Query write attempt count: `{summary['query_write_attempt_count']}`",
        f"- Cross-context leakage detected: `{summary['cross_context_leakage_detected']}`",
        f"- Retrieved latents entered Reasoner: `{summary['retrieved_latents_enter_reasoner']}`",
        f"- Retrieved latents entered Weaver: `{summary['retrieved_latents_enter_weaver']}`",
        f"- Write action counts: `{summary['write_action_counts']}`",
        f"- Update reason counts: `{summary['update_reason_counts']}`",
        f"- Append/insert count: `{summary['append_insert_count']}`",
        f"- Matched replace count: `{summary['matched_replace_count']}`",
        f"- Capacity evict count: `{summary['capacity_evict_count']}`",
        f"- Query-turn retrieved indices: `{summary['query_turn_retrieved_indices']}`",
        f"- Query-turn retrieved scores: `{summary['query_turn_retrieved_scores']}`",
        f"- Query-turn retrieved score range: `{summary['query_turn_retrieved_score_range']}`",
        "",
        "## Baseline Comparison",
        f"- Against MAB-5A: `{summary['compare_against_mab5a']}`",
        f"- Against MAB-5B: `{summary['compare_against_mab5b']}`",
        "",
        "## Per-context Result Table",
        "| context_index | exact_match_off | exact_match_on | output_changed | query_turn_retrieval_active | query_turn_retrieved_latent_count |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in valid_rows:
        lines.append(
            f"| {row['context_index']} | {row['bank_off_exact_match']} | {row['bank_on_exact_match']} | {row['output_changed']} | {int(bool(row.get('query_turn_retrieval_active')))} | {row.get('query_turn_retrieved_latent_count', 0)} |"
        )
    lines.extend([
        "",
        "## Interpretation",
    ])
    if summary["mean_final_slot_count"] == DEFAULT_MAX_SLOTS and summary["num_query_turn_retrieval_active"] >= summary["num_contexts_valid"] / 2:
        interpretation = (
            "The split thresholds produced the intended mechanism shape: slot growth reached capacity "
            "while query-time retrieval remained active and sparser than the lower-threshold baseline."
        )
    elif summary["mean_final_slot_count"] == DEFAULT_MAX_SLOTS:
        interpretation = (
            "The split thresholds increased slot growth to capacity, but query-time retrieval still needs "
            "closer inspection before drawing a mechanism conclusion."
        )
    else:
        interpretation = (
            "The split thresholds did not yet recover full slot growth; decoupling alone is not sufficient."
        )
    lines.extend([
        interpretation,
        "",
        "## Git Status",
        "### Before",
        "```",
        git_status_before,
        "```",
        "### After",
        "```",
        git_status_after,
        "```",
    ])
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return note_path


def build_parser():
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--dataset-root", default="/mnt/18T/baishilong/datasets/MemoryAgentBench")
    parser.add_argument("--mab-repo", default="/mnt/18T/baishilong/benchmarks/MemoryAgentBench")
    parser.add_argument("--mab-python", default="/home/baishilong/miniconda3/envs/MABench/bin/python")
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
        default="Kana-s/MemGen@269d9b1/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model",
    )
    parser.add_argument("--cfg-path", default="configs/latent_memory/triviaqa.yaml")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--requested-contexts", type=int, default=DEFAULT_REQUESTED_CONTEXTS)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    original = {
        "EXPERIMENT_NAME": base.EXPERIMENT_NAME,
        "RUN_PREFIX": base.RUN_PREFIX,
        "DEFAULT_OUTPUT_ROOT": base.DEFAULT_OUTPUT_ROOT,
        "DEFAULT_THRESHOLD": base.DEFAULT_THRESHOLD,
        "DEFAULT_TOP_K": base.DEFAULT_TOP_K,
        "DEFAULT_MAX_SLOTS": base.DEFAULT_MAX_SLOTS,
        "DEFAULT_RETRIEVE_POLICY": base.DEFAULT_RETRIEVE_POLICY,
        "GIT_STATUS_BEFORE_EDIT": base.GIT_STATUS_BEFORE_EDIT,
        "build_parser": base.build_parser,
        "_bank_config": base._bank_config,
        "_build_row": base._build_row,
        "_aggregate": base._aggregate,
        "_build_research_note": base._build_research_note,
        "_build_manifest": base._build_manifest,
    }
    base.EXPERIMENT_NAME = EXPERIMENT_NAME
    base.RUN_PREFIX = RUN_PREFIX
    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    base.DEFAULT_THRESHOLD = DEFAULT_THRESHOLD
    base.DEFAULT_TOP_K = DEFAULT_TOP_K
    base.DEFAULT_MAX_SLOTS = DEFAULT_MAX_SLOTS
    base.DEFAULT_RETRIEVE_POLICY = DEFAULT_RETRIEVE_POLICY
    base.GIT_STATUS_BEFORE_EDIT = _git("status", "--short", "--branch")
    base.build_parser = build_parser
    base._bank_config = _bank_config
    base._build_row = _build_row
    base._aggregate = _aggregate
    base._build_research_note = _build_research_note
    base._build_manifest = _build_manifest
    try:
        result = base.main()
        if result == 0:
            output_root = Path(DEFAULT_OUTPUT_ROOT)
            run_dirs = sorted(
                [path for path in output_root.iterdir() if path.is_dir()],
                key=lambda path: path.stat().st_mtime,
            )
            if run_dirs:
                run_dir = run_dirs[-1]
                run_config_path = run_dir / "run_config.json"
                if run_config_path.exists():
                    run_config = _load_json(run_config_path)
                    run_config.update(
                        {
                            "experiment_name": EXPERIMENT_NAME,
                            "run_prefix": RUN_PREFIX,
                            "max_slots": DEFAULT_MAX_SLOTS,
                            "configured_max_slots": DEFAULT_MAX_SLOTS,
                            "actual_bank_max_slots": DEFAULT_MAX_SLOTS,
                            "retrieve_threshold": DEFAULT_RETRIEVE_THRESHOLD,
                            "update_threshold": DEFAULT_UPDATE_THRESHOLD,
                            "mechanism": "capacity16_decoupled_retrieval_update_thresholds",
                            "research_note": "research_notes/benchmarks/memoryagentbench_mab5d_capacity16.md",
                        }
                    )
                    _write_json(run_config_path, run_config)
        return result
    finally:
        base.EXPERIMENT_NAME = original["EXPERIMENT_NAME"]
        base.RUN_PREFIX = original["RUN_PREFIX"]
        base.DEFAULT_OUTPUT_ROOT = original["DEFAULT_OUTPUT_ROOT"]
        base.DEFAULT_THRESHOLD = original["DEFAULT_THRESHOLD"]
        base.DEFAULT_TOP_K = original["DEFAULT_TOP_K"]
        base.DEFAULT_MAX_SLOTS = original["DEFAULT_MAX_SLOTS"]
        base.DEFAULT_RETRIEVE_POLICY = original["DEFAULT_RETRIEVE_POLICY"]
        base.GIT_STATUS_BEFORE_EDIT = original["GIT_STATUS_BEFORE_EDIT"]
        base.build_parser = original["build_parser"]
        base._bank_config = original["_bank_config"]
        base._build_row = original["_build_row"]
        base._aggregate = original["_aggregate"]
        base._build_research_note = original["_build_research_note"]
        base._build_manifest = original["_build_manifest"]


if __name__ == "__main__":
    raise SystemExit(main())
