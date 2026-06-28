"""MAB-6B-FR: detective_qa Weaver-space bank with final-query format repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import mab5a_detectiveqa_compressed_n10 as base
from scripts.eval import mab5c_decoupled_thresholds_detectiveqa_n10 as mab5c
from scripts.eval import mab6b_weaver_space_bank_detectiveqa_n10 as parent

_BASE_PREPARE_PAYLOAD = base._prepare_payload

EXPERIMENT_NAME = "MAB-6B-FR: detective_qa Weaver-space bank + final-query format repair"
RUN_PREFIX = "detectiveqa-version-b-weaver-space-bank-format-repair-n10"
DEFAULT_OUTPUT_ROOT = "outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_format_repair"
FORMAT_REPAIR_RESEARCH_NOTE_PATH = Path(
    "research_notes/benchmarks/memoryagentbench_mab6b_fr_format_repair.md"
)
CANONICAL_MAB6B_BASELINE = (
    "outputs/mab/version_b_weaver_space_bank_detectiveqa_n10/"
    "20260625T122323Z-detectiveqa-version-b-weaver-space-bank-n10"
)
FORMAT_REPAIR_PREFIX = (
    "Answer the following question with only the final answer. "
    "Do not include reasoning, explanation, citations, or extra text. "
    "If the answer is a name, output only the name.\n\n"
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _classify_output_format(text: str | None) -> str | None:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return "other"
    lowered = stripped.lower()
    if stripped.startswith("```") or "```" in stripped:
        return "code_leak"
    if stripped.startswith("{") or stripped.startswith("[") or "\"answer\"" in lowered:
        return "json_leak"
    if "no answer provided" in lowered or "no search results found" in lowered or "missing context" in lowered:
        return "refusal"
    if any(token in stripped for token in ("输出", "答案", "不需要", "单词表", "翻译")):
        return "language_drift"
    if any(token in lowered for token in ("reasoning", "because", "this is the correct answer", "question:")):
        return "reasoning_sprawl"
    first_line = stripped.splitlines()[0].strip()
    if first_line and "\n" not in stripped and not any(ch in stripped for ch in "{}[]`"):
        return "clean_option"
    return "other"


def _dict_sum(rows: list[dict], key: str) -> dict:
    total: dict[str, int] = {}
    for row in rows:
        value = row.get(key) or {}
        for name, count in value.items():
            total[name] = total.get(name, 0) + int(count)
    return total


def _query_turn_retrieved_latent_count(bank_on_result: dict) -> int:
    generations = bank_on_result.get("generations") or []
    if generations:
        final_generation = generations[-1]
        if "memory_retrieved_latent_count" in final_generation:
            return int(final_generation["memory_retrieved_latent_count"])
        if "retrieved_latent_count" in final_generation:
            return int(final_generation["retrieved_latent_count"])
    retrieved_indices = bank_on_result.get("retrieved_indices_by_turn") or []
    if retrieved_indices:
        return len(retrieved_indices[-1])
    return 0


def _prepare_payload(args, output_path: Path, match_index: int, timestamp: str) -> dict:
    payload = _BASE_PREPARE_PAYLOAD(args, output_path, match_index, timestamp)
    original_query_prompt = payload["query_prompt"]
    repaired_query_prompt = FORMAT_REPAIR_PREFIX + original_query_prompt
    payload["query_prompt_original"] = original_query_prompt
    payload["query_prompt"] = repaired_query_prompt
    payload["format_repair"] = {
        "enabled": True,
        "applied_to_bank_off": True,
        "applied_to_bank_on": True,
        "applied_to_chunks": False,
        "prefix": FORMAT_REPAIR_PREFIX,
        "original_query_prompt_hash": _sha256_text(original_query_prompt),
        "repaired_query_prompt_hash": _sha256_text(repaired_query_prompt),
    }
    return payload


def _build_manifest(
    run_id: str,
    args,
    started_at: str,
    *,
    git_status_before: str,
    git_status_after: str | None = None,
) -> dict:
    manifest = parent._build_manifest(
        run_id,
        args,
        started_at,
        git_status_before=git_status_before,
        git_status_after=git_status_after,
    )
    manifest.update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_id": run_id,
            "output_root": DEFAULT_OUTPUT_ROOT,
            "format_repair_enabled": True,
            "format_repair_applied_to_bank_off": True,
            "format_repair_applied_to_bank_on": True,
            "format_repair_applied_to_chunks": False,
            "format_repair_prefix": FORMAT_REPAIR_PREFIX,
            "comparison_baseline_primary": CANONICAL_MAB6B_BASELINE,
        }
    )
    return manifest


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
    row = parent._build_row(
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
    format_repair = payload["format_repair"]
    row.update(
        {
            "format_repair_enabled": True,
            "format_repair_applied_to_bank_off": True,
            "format_repair_applied_to_bank_on": True,
            "format_repair_applied_to_chunks": False,
            "original_query_prompt_hash": format_repair["original_query_prompt_hash"],
            "repaired_query_prompt_hash": format_repair["repaired_query_prompt_hash"],
            "final_query_prompt_hash_bank_off": bank_off_result["prompt_trace"][-1]["rendered_prompt_hash"],
            "final_query_prompt_hash_bank_on": bank_on_result["prompt_trace"][-1]["rendered_prompt_hash"],
            "final_query_token_len_bank_off": bank_off_result["prompt_trace"][-1]["prompt_history_token_len"],
            "final_query_token_len_bank_on": bank_on_result["prompt_trace"][-1]["prompt_history_token_len"],
            "write_action_counts": _dict_sum(bank_on_result["generations"], "write_action_counts"),
            "update_reason_counts": _dict_sum(bank_on_result["generations"], "update_reason_counts"),
            "append_insert_count": sum(
                int((gen.get("write_action_counts") or {}).get("insert", 0))
                for gen in bank_on_result["generations"]
            ) if bank_on_result.get("generations") else 0,
            "matched_replace_count": sum(
                int((gen.get("write_action_counts") or {}).get("replace_matched", 0))
                for gen in bank_on_result["generations"]
            ) if bank_on_result.get("generations") else 0,
            "capacity_evict_count": sum(
                int((gen.get("write_action_counts") or {}).get("capacity_evict", 0))
                for gen in bank_on_result["generations"]
            ) if bank_on_result.get("generations") else 0,
            "query_turn_retrieved_indices": list(bank_on_result["retrieved_indices_by_turn"][-1]) if bank_on_result.get("retrieved_indices_by_turn") else [],
            "query_turn_retrieved_scores": list(bank_on_result["retrieved_scores_by_turn"][-1]) if bank_on_result.get("retrieved_scores_by_turn") else [],
            "query_turn_retrieved_latent_count": _query_turn_retrieved_latent_count(bank_on_result),
            "bank_off_primary_output_format_status": _classify_output_format(row["bank_off_prediction"]),
            "bank_on_primary_output_format_status": _classify_output_format(row["bank_on_prediction"]),
        }
    )
    return row


def _aggregate(rows: list[dict]) -> dict:
    summary = parent._aggregate(rows)
    valid = [row for row in rows if not row.get("error_or_stop_reason")]
    summary.update(
        {
            "format_repair_enabled": True,
            "format_repair_applied_to_bank_off": True,
            "format_repair_applied_to_bank_on": True,
            "format_repair_applied_to_chunks": False,
            "final_query_token_lens_bank_off": [
                int(row.get("final_query_token_len_bank_off", 0)) for row in valid
            ],
            "final_query_token_lens_bank_on": [
                int(row.get("final_query_token_len_bank_on", 0)) for row in valid
            ],
            "write_action_counts": _dict_sum(valid, "write_action_counts"),
            "update_reason_counts": _dict_sum(valid, "update_reason_counts"),
            "append_insert_count": sum(int(row.get("append_insert_count", 0)) for row in valid),
            "matched_replace_count": sum(int(row.get("matched_replace_count", 0)) for row in valid),
            "capacity_evict_count": sum(int(row.get("capacity_evict_count", 0)) for row in valid),
            "query_turn_retrieved_indices": [
                list(row.get("query_turn_retrieved_indices", [])) for row in valid
            ],
            "query_turn_retrieved_latent_count": [
                int(row.get("query_turn_retrieved_latent_count", 0)) for row in valid
            ],
        }
    )
    return summary


def build_parser():
    parser = parent.build_parser()
    parser.description = EXPERIMENT_NAME
    parser.set_defaults(output_root=DEFAULT_OUTPUT_ROOT)
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
        "_prepare_payload": base._prepare_payload,
        "_bank_config": base._bank_config,
        "_build_row": base._build_row,
        "_aggregate": base._aggregate,
        "_build_research_note": base._build_research_note,
        "_build_manifest": base._build_manifest,
        "_load_model": base._load_model,
        "_run_model": base._run_model,
        "ALLOW_RETRIEVED_LATENTS_ENTER_WEAVER": base.ALLOW_RETRIEVED_LATENTS_ENTER_WEAVER,
        "RESEARCH_NOTE_PATH": parent.RESEARCH_NOTE_PATH,
    }
    base.EXPERIMENT_NAME = EXPERIMENT_NAME
    base.RUN_PREFIX = RUN_PREFIX
    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    base.DEFAULT_THRESHOLD = parent.DEFAULT_THRESHOLD
    base.DEFAULT_TOP_K = parent.DEFAULT_TOP_K
    base.DEFAULT_MAX_SLOTS = parent.DEFAULT_MAX_SLOTS
    base.DEFAULT_RETRIEVE_POLICY = parent.DEFAULT_RETRIEVE_POLICY
    base.GIT_STATUS_BEFORE_EDIT = mab5c._git("status", "--short", "--branch")
    base.build_parser = build_parser
    base._prepare_payload = _prepare_payload
    base._bank_config = parent._bank_config
    base._build_row = _build_row
    base._aggregate = _aggregate
    base._build_research_note = parent._build_research_note
    base._build_manifest = _build_manifest
    base._load_model = parent._load_model
    base._run_model = parent._run_model
    base.ALLOW_RETRIEVED_LATENTS_ENTER_WEAVER = True
    parent.RESEARCH_NOTE_PATH = FORMAT_REPAIR_RESEARCH_NOTE_PATH
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
                            "retrieve_threshold": parent.DEFAULT_RETRIEVE_THRESHOLD,
                            "update_threshold": parent.DEFAULT_UPDATE_THRESHOLD,
                            "retrieved_memory_to_weaver": True,
                            "memory_bank_storage_space": "weaver",
                            "mechanism": "version_b_weaver_space_bank_format_repair",
                            "format_repair_enabled": True,
                            "format_repair_applied_to_bank_off": True,
                            "format_repair_applied_to_bank_on": True,
                            "format_repair_applied_to_chunks": False,
                            "format_repair_prefix": FORMAT_REPAIR_PREFIX,
                            "comparison_baseline_primary": CANONICAL_MAB6B_BASELINE,
                            "research_note": str(FORMAT_REPAIR_RESEARCH_NOTE_PATH),
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
        base._prepare_payload = original["_prepare_payload"]
        base._bank_config = original["_bank_config"]
        base._build_row = original["_build_row"]
        base._aggregate = original["_aggregate"]
        base._build_research_note = original["_build_research_note"]
        base._build_manifest = original["_build_manifest"]
        base._load_model = original["_load_model"]
        base._run_model = original["_run_model"]
        base.ALLOW_RETRIEVED_LATENTS_ENTER_WEAVER = original["ALLOW_RETRIEVED_LATENTS_ENTER_WEAVER"]
        parent.RESEARCH_NOTE_PATH = original["RESEARCH_NOTE_PATH"]


if __name__ == "__main__":
    raise SystemExit(main())
