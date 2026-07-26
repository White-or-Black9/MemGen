"""Evaluate bank-disabled MemGen with a capacity-safe recent-text window."""

from __future__ import annotations

import argparse
import hashlib
import json
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


SCHEMA_VERSION = "eventqa-memgen-recent-window/v1"
DEFAULT_OUTPUT_ROOT = "outputs/mab/eventqa_memgen_recent_window_smoke"


class RecentWindowContractError(ValueError):
    """Raised when the fixed-window MemGen comparison contract is violated."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def expected_question_indices(scope: str, context_index: int, question_limit: int) -> list[int]:
    if scope == "smoke" and context_index == 0 and question_limit == 10:
        return list(range(10))
    if scope == "full" and context_index in range(5) and question_limit == 100:
        return list(range(100))
    raise RecentWindowContractError("smoke requires ctx0/q0-9; full requires ctx0-4/q0-99")


def build_recent_window_prompt(history: str, official_query_prompt: str) -> str:
    return (
        "Use the recent source context below to answer the task.\n\n"
        "[Recent context]\n"
        + history
        + "\n\n[Task]\n"
        + official_query_prompt
    )


def _rendered_count(model, prompt: str) -> tuple[str, int]:
    messages = [
        {"role": "system", "content": eventqa.base.DEFAULT_SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]
    rendered = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return rendered, len(model.tokenizer.encode(rendered, add_special_tokens=False))


def _recent_text(tokenizer, context_text: str, budget: int) -> tuple[str, int]:
    if budget <= 0:
        raise RecentWindowContractError("recent-history-token-budget must be positive")
    token_ids = tokenizer.encode(context_text, add_special_tokens=False)
    selected = token_ids[-budget:]
    return tokenizer.decode(selected, skip_special_tokens=True, clean_up_tokenization_spaces=False), len(selected)


def resolve_budget(
    model,
    context_text: str,
    official_prompts: list[str],
    *,
    requested_budget: int,
    input_capacity: int,
) -> tuple[str, int, int]:
    """Return a common recent suffix that fits every selected question."""
    candidate = min(requested_budget, len(model.tokenizer.encode(context_text, add_special_tokens=False)))
    while candidate > 0:
        history, actual = _recent_text(model.tokenizer, context_text, candidate)
        counts = [_rendered_count(model, build_recent_window_prompt(history, prompt))[1] for prompt in official_prompts]
        if max(counts) <= input_capacity:
            return history, actual, max(counts)
        candidate -= min(256, candidate)
    raise RecentWindowContractError("no recent-text window fits after reserving generation tokens")


def validate_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise RecentWindowContractError("unexpected schema")
    method = artifact.get("method", {})
    if method.get("bank_mode") != "off" or method.get("history_policy") != "recent_text_suffix":
        raise RecentWindowContractError("baseline must be bank-disabled recent-text suffix")
    scope = artifact.get("scope", {})
    indices = scope.get("question_indices", [])
    records = artifact.get("records", [])
    if len(records) != len(indices) or [row.get("query_index") for row in records] != indices:
        raise RecentWindowContractError("question coverage is incomplete")
    if not records or not all(row.get("capacity_ok") is True for row in records):
        raise RecentWindowContractError("all prompts must fit the reserved capacity")
    if not all(row.get("bank_mode") == "off" for row in records):
        raise RecentWindowContractError("a record enabled the persistent bank")


def build_parser() -> argparse.ArgumentParser:
    parser = eventqa.build_parser()
    parser.description = __doc__
    parser.add_argument("--measurement-scope", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--recent-history-token-budget", type=int, default=8192)
    parser.add_argument("--generation-reserve-tokens", type=int, default=40)
    parser.add_argument("--run-id")
    parser.set_defaults(output_root=DEFAULT_OUTPUT_ROOT, requested_contexts=1, eventqa_protocol="frozen_context_bank", skip_research_note=True, reseed_per_context=True)
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _cuda_sync() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    question_indices = expected_question_indices(args.measurement_scope, args.context_index, args.question_limit)
    if args.generation_reserve_tokens != args.generation_max_length:
        raise RecentWindowContractError("generation reserve must equal generation-max-length")
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = args.run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-eventqa-memgen-recent-window-ctx{args.context_index}-q0-{question_indices[-1]}-{args.measurement_scope}"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = eventqa._load_rows(args.parquet, eventqa.SUB_DATASET)
    context = eventqa.build_context_payload(args, rows[args.context_index], args.context_index, started_at)
    model, capacity = eventqa.weaver_bank._load_model(args)
    import torch
    try:
        context_rng = eventqa._prepare_context_rng(
            base_seed=args.seed,
            context_index=args.context_index,
            reseed_per_context=args.reseed_per_context,
        )
        # This is the baseline contract: no session-local persistent bank and no
        # retrieval back into Weaver. The model still uses the MemGen checkpoint.
        model.config.retrieved_memory_to_weaver = False
        model.config.memory_bank_storage_space = "reasoner"
        context_text = "\n\n".join(context["chunks"])
        payloads = [eventqa.build_question_payload(context, index) for index in question_indices]
        history, history_tokens, max_rendered = resolve_budget(
            model,
            context_text,
            [payload["query_prompt"] for payload in payloads],
            requested_budget=args.recent_history_token_budget,
            input_capacity=capacity - args.generation_reserve_tokens,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); _cuda_sync()
            baseline_memory = int(torch.cuda.memory_allocated()); torch.cuda.reset_peak_memory_stats()
        else:
            baseline_memory = 0
        manifest = eventqa._build_manifest(run_id, args, started_at, git_status_before=eventqa._git("status", "--short", "--branch"), selected_context_indices=[args.context_index])
        manifest.update({
            "schema_version": SCHEMA_VERSION,
            "measurement_scope": args.measurement_scope,
            "exact_command": [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
            "context_id": context["context_id"],
            "chunk_count": len(context["chunks"]),
            "full_context_sha256": _sha256(context_text),
            "method": {"model_path": "MemGen checkpoint", "bank_mode": "off", "history_policy": "recent_text_suffix", "requested_recent_history_token_budget": args.recent_history_token_budget, "resolved_recent_history_token_budget": history_tokens, "generation_reserve_tokens": args.generation_reserve_tokens},
            "context_capacity": capacity,
            "max_rendered_prompt_token_count": max_rendered,
            "context_rng": context_rng,
            "gpu": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        })
        _write_json(output_dir / "manifest.json", manifest)
        records = []
        for query_index, payload in zip(question_indices, payloads):
            prompt = build_recent_window_prompt(history, payload["query_prompt"])
            rendered, rendered_count = _rendered_count(model, prompt)
            capacity_ok = rendered_count <= capacity - args.generation_reserve_tokens
            if not capacity_ok:
                raise RecentWindowContractError("runtime prompt exceeds reserved capacity")
            query_payload = eventqa._query_only_payload(payload)
            query_payload["query_prompt"] = prompt
            query_payload["memorization_prompts"] = [prompt]
            _cuda_sync(); start = time.perf_counter()
            result = eventqa._run_eventqa_model(args, model, capacity, query_payload, "off")
            _cuda_sync(); latency = time.perf_counter() - start
            if result["rendered_query_prompt"] != rendered:
                raise RecentWindowContractError("runtime prompt differs from preflight")
            with tempfile.TemporaryDirectory() as tmpdir:
                score = eventqa._score_prediction(args, payload, result["prediction"], tmpdir)
            turn = eventqa._query_turn(result)
            records.append({"context_index": args.context_index, "query_index": query_index, "bank_mode": "off", "history_sha256": _sha256(history), "history_token_count": history_tokens, "prompt_sha256": _sha256(prompt), "rendered_prompt_token_count": rendered_count, "context_capacity": capacity, "capacity_ok": capacity_ok, "prediction": result["prediction"], "substring_exact_match": eventqa._metric_value(score, "substring_exact_match", default=0), "eventqa_recall": eventqa._metric_value(score, "eventqa_recall", default=0.0), "format_flags": eventqa._format_flags(result["prediction"]), "cost": {"end_to_end_latency_seconds": latency, "output_tokens": int(turn["output_len"])}})
        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else baseline_memory
        artifact = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "scope": {"measurement_scope": args.measurement_scope, "context_index": args.context_index, "question_indices": question_indices}, "method": manifest["method"], "cost": {"baseline_gpu_memory_bytes": baseline_memory, "peak_gpu_memory_bytes": peak, "incremental_peak_gpu_memory_bytes": peak - baseline_memory}, "records": records}
        validate_artifact(artifact)
        name = "smoke_artifact.json" if args.measurement_scope == "smoke" else "full_artifact.json"
        _write_json(output_dir / name, artifact)
        (output_dir / "per_question.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat(); _write_json(output_dir / "manifest.json", manifest)
    finally:
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
