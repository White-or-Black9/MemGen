"""Run an EventQA BM25 top-2 strict matched16 evaluation."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import mab6b_weaver_space_bank_eventqa_65536_n5 as eventqa
from scripts.eval.eventqa_bm25_retrieved_text import (
    BM25Index,
    _rendered_prompt,
    retrieve_top_k,
    tokenize_bm25,
)


SCHEMA_VERSION = "eventqa-matched16/v1"
EXPECTED_QUESTIONS = list(range(10))
DEFAULT_OUTPUT_ROOT = "outputs/mab/eventqa_matched16_smoke"


class Matched16ContractError(ValueError):
    """Raised when the matched-source-token contract is violated."""


def expected_question_indices(
    measurement_scope: str, context_index: int, question_limit: int
) -> list[int]:
    if measurement_scope == "smoke":
        if context_index != 0 or question_limit != 10:
            raise Matched16ContractError("smoke scope must be context 0 q0-9")
        return list(range(10))
    if measurement_scope == "full":
        if context_index not in range(5) or question_limit != 100:
            raise Matched16ContractError("full scope must be context 0-4 q0-99")
        return list(range(100))
    raise Matched16ContractError(f"unsupported measurement scope: {measurement_scope}")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bm25_query_term_weights(index: BM25Index, query: str) -> dict[str, float]:
    weights = {}
    document_count = len(index.term_frequencies)
    for term in set(tokenize_bm25(query)):
        document_frequency = index.document_frequency.get(term, 0)
        weights[term] = math.log(
            1.0
            + (document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )
    return weights


def select_relevant_window(
    token_ids: list[int],
    decode: Callable[[list[int]], str],
    query: str,
    term_weights: dict[str, float],
    *,
    width: int = 8,
) -> dict[str, Any]:
    return rank_relevant_windows(
        token_ids, decode, query, term_weights, width=width
    )[0]


def rank_relevant_windows(
    token_ids: list[int],
    decode: Callable[[list[int]], str],
    query: str,
    term_weights: dict[str, float],
    *,
    width: int = 8,
) -> list[dict[str, Any]]:
    if len(token_ids) < width:
        raise Matched16ContractError(f"retrieved chunk has fewer than {width} tokens")
    query_terms = set(tokenize_bm25(query))
    candidates = []
    for start in range(len(token_ids) - width + 1):
        window_ids = token_ids[start : start + width]
        text = decode(window_ids)
        counts: dict[str, int] = {}
        for term in tokenize_bm25(text):
            if term in query_terms:
                counts[term] = counts.get(term, 0) + 1
        score = sum(term_weights.get(term, 0.0) * count for term, count in counts.items())
        candidates.append(
            {
                "token_start": start,
                "token_end": start + width,
                "token_ids": token_ids[start : start + width],
                "text": text,
                "text_sha256": _sha256(text),
                "window_score": score,
            }
        )
    candidates.sort(key=lambda item: (-item["window_score"], item["token_start"]))
    for rank, candidate in enumerate(candidates, start=1):
        candidate["candidate_rank"] = rank
    return candidates


def select_budget_constrained_pair(
    candidates_by_chunk: list[list[dict[str, Any]]],
    official_query_prompt: str,
    rendered_count: Callable[[str], int],
    *,
    official_rendered_token_count: int,
    candidate_limits: tuple[int, ...] = (32, 64, 256),
) -> dict[str, Any]:
    if len(candidates_by_chunk) != 2:
        raise Matched16ContractError("constrained selector requires two chunks")
    top_score = sum(candidates[0]["window_score"] for candidates in candidates_by_chunk)
    top_pair = (candidates_by_chunk[0][0], candidates_by_chunk[1][0])
    top_prompt = build_matched_prompt(list(top_pair), official_query_prompt)
    top_delta = rendered_count(top_prompt) - official_rendered_token_count
    if top_delta == 16:
        return {
            "windows": [dict(top_pair[0]), dict(top_pair[1])],
            "candidate_ranks": [1, 1],
            "fallback_used": False,
            "score_loss": 0.0,
            "search_limit": 1,
            "checked_pair_count": 1,
            "rendered_prompt_token_delta": top_delta,
        }
    checked: set[tuple[int, int]] = {(1, 1)}
    for limit in candidate_limits:
        left = candidates_by_chunk[0][:limit]
        right = candidates_by_chunk[1][:limit]
        pairs = sorted(
            itertools.product(left, right),
            key=lambda pair: (
                -(pair[0]["window_score"] + pair[1]["window_score"]),
                pair[0]["token_start"],
                pair[1]["token_start"],
            ),
        )
        for pair in pairs:
            identity = (pair[0]["candidate_rank"], pair[1]["candidate_rank"])
            if identity in checked:
                continue
            checked.add(identity)
            prompt = build_matched_prompt(list(pair), official_query_prompt)
            delta = rendered_count(prompt) - official_rendered_token_count
            if delta == 16:
                selected_score = pair[0]["window_score"] + pair[1]["window_score"]
                return {
                    "windows": [dict(pair[0]), dict(pair[1])],
                    "candidate_ranks": list(identity),
                    "fallback_used": identity != (1, 1),
                    "score_loss": top_score - selected_score,
                    "search_limit": limit,
                    "checked_pair_count": len(checked),
                    "rendered_prompt_token_delta": delta,
                }
    raise Matched16ContractError(
        "no exact-delta16 window pair found within candidate search limits"
    )


def build_matched_prompt(windows: list[dict[str, Any]], official_query_prompt: str) -> str:
    return "".join(window["text"] for window in windows) + official_query_prompt


def _finite_nonnegative(value: Any, label: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise Matched16ContractError(f"{label} must be finite and nonnegative")


def validate_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise Matched16ContractError("unexpected schema version")
    if artifact.get("measurement_mode") != "standalone_process":
        raise Matched16ContractError("measurement must use a standalone process")
    scope = artifact.get("scope", {})
    measurement_scope = scope.get("measurement_scope", "smoke")
    context_index = scope.get("context_index")
    question_indices = scope.get("question_indices", [])
    expected = expected_question_indices(
        measurement_scope, context_index, len(question_indices)
    )
    if question_indices != expected:
        label = "q0-9" if measurement_scope == "smoke" else "q0-99"
        raise Matched16ContractError(
            f"{measurement_scope} scope must cover context {context_index} {label}"
        )
    if artifact.get("method") != {
        "retrieval": "bm25_top2",
        "window_tokens_per_chunk": 8,
        "source_token_budget": 16,
    }:
        raise Matched16ContractError("matched16 method configuration drift")
    records = artifact.get("records", [])
    if len(records) != len(expected) or [row.get("query_index") for row in records] != expected:
        label = "q0-9" if measurement_scope == "smoke" else "q0-99"
        raise Matched16ContractError(
            f"{measurement_scope} records must cover {label} exactly"
        )
    top_cost = artifact.get("cost", {})
    for field in (
        "index_construction_latency_seconds",
        "baseline_gpu_memory_bytes",
        "peak_gpu_memory_bytes",
    ):
        _finite_nonnegative(top_cost.get(field), field)
    if top_cost["peak_gpu_memory_bytes"] < top_cost["baseline_gpu_memory_bytes"]:
        raise Matched16ContractError("peak GPU memory cannot be below baseline")
    for record in records:
        if record.get("context_index") != context_index:
            raise Matched16ContractError("record context does not match artifact context")
        windows = record.get("windows", [])
        if len(windows) != 2 or record.get("source_token_count") != 16:
            raise Matched16ContractError("each question must inject exactly 16 source tokens")
        for window in windows:
            token_ids = window.get("token_ids", [])
            if len(token_ids) != 8:
                raise Matched16ContractError("each source window must contain 8 token IDs")
            if window.get("token_end") - window.get("token_start") != 8:
                raise Matched16ContractError("source window offsets must span 8 tokens")
            for field in ("chunk_id", "text"):
                if not isinstance(window.get(field), str) or not window[field]:
                    raise Matched16ContractError(f"window {field} is missing")
            for field in ("chunk_text_sha256", "text_sha256"):
                if len(window.get(field, "")) != 64:
                    raise Matched16ContractError(f"window {field} is invalid")
            if _sha256(window["text"]) != window["text_sha256"]:
                raise Matched16ContractError("window text hash mismatch")
            _finite_nonnegative(window.get("chunk_bm25_score"), "chunk BM25 score")
            _finite_nonnegative(window.get("window_score"), "window score")
        for field in ("official_query_sha256", "matched_prompt_sha256"):
            if len(record.get(field, "")) != 64:
                raise Matched16ContractError(f"{field} is invalid")
        official_count = record.get("official_rendered_token_count")
        matched_count = record.get("matched_rendered_token_count")
        delta = record.get("rendered_prompt_token_delta")
        if delta != matched_count - official_count:
            raise Matched16ContractError("rendered prompt token delta is inconsistent")
        if delta != 16:
            raise Matched16ContractError("rendered prompt token delta must equal 16")
        if record.get("capacity_ok") is not True or matched_count > record.get("context_capacity", -1):
            raise Matched16ContractError("matched prompt capacity check failed")
        for field in (
            "official_rendered_token_count",
            "matched_rendered_token_count",
            "rendered_prompt_token_delta",
            "context_capacity",
        ):
            _finite_nonnegative(record.get(field), field)
        method_cost = record.get("cost", {})
        for field in (
            "retrieval_and_window_latency_seconds",
            "generation_latency_seconds",
            "end_to_end_latency_seconds",
            "output_tokens",
        ):
            _finite_nonnegative(method_cost.get(field), field)


def validate_smoke_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("scope", {}).get("measurement_scope", "smoke") != "smoke":
        raise Matched16ContractError("validate_smoke_artifact requires smoke scope")
    validate_artifact(artifact)


def build_parser():
    parser = eventqa.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--measurement-scope", choices=("smoke", "full"), default="smoke"
    )
    parser.set_defaults(
        output_root=DEFAULT_OUTPUT_ROOT,
        requested_contexts=1,
        context_index=0,
        question_limit=10,
        eventqa_protocol="frozen_context_bank",
        generation_max_length=40,
        skip_research_note=True,
        reseed_per_context=True,
    )
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _cuda_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    question_indices = expected_question_indices(
        args.measurement_scope, args.context_index, args.question_limit
    )
    started_at = datetime.now(timezone.utc).isoformat()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"{timestamp}-eventqa-matched16-ctx{args.context_index}-"
        f"q0-{question_indices[-1]}-{args.measurement_scope}"
    )
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]

    rows = eventqa._load_rows(args.parquet, eventqa.SUB_DATASET)
    context_payload = eventqa.build_context_payload(
        args, rows[args.context_index], args.context_index, started_at
    )
    chunks = context_payload["chunks"]
    index_start = time.perf_counter()
    index = BM25Index(chunks)
    index_latency = time.perf_counter() - index_start

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
            "exact_command": command,
            "method": {
                "retrieval": "bm25_top2",
                "window_tokens_per_chunk": 8,
                "source_token_budget": 16,
                "window_scoring": "bm25_idf_weighted_query_term_overlap",
                "tie_break": "lowest_source_token_start",
                "budget_constraint": "rendered_prompt_token_delta_equals_16",
                "candidate_limits": [32, 64, 256],
            },
            "context_id": context_payload["context_id"],
            "chunk_count": len(chunks),
            "chunk_hashes": [_sha256(chunk) for chunk in chunks],
        }
    )
    _write_json(output_dir / "manifest.json", manifest)

    model, capacity = eventqa.weaver_bank._load_model(args)
    import torch

    tokenizer = model.tokenizer
    manifest["context_capacity"] = capacity
    manifest["gpu"] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _cuda_sync()
        baseline_gpu_memory = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
    else:
        baseline_gpu_memory = 0

    records: list[dict[str, Any]] = []
    try:
        for query_index in question_indices:
            payload = eventqa.build_question_payload(context_payload, query_index)
            selection_start = time.perf_counter()
            retrieved = retrieve_top_k(
                chunks,
                payload["question"],
                context_id=context_payload["context_id"],
                index=index,
            )
            term_weights = bm25_query_term_weights(index, payload["question"])
            candidates_by_chunk = []
            for item in retrieved:
                source_ids = tokenizer.encode(item["text"], add_special_tokens=False)
                candidates = rank_relevant_windows(
                    source_ids,
                    lambda ids: tokenizer.decode(
                        ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    payload["question"],
                    term_weights,
                    width=8,
                )
                candidates_by_chunk.append(candidates)
            official_rendered, official_token_count = _rendered_prompt(
                model, payload["query_prompt"]
            )
            constrained = select_budget_constrained_pair(
                candidates_by_chunk,
                payload["query_prompt"],
                lambda prompt: _rendered_prompt(model, prompt)[1],
                official_rendered_token_count=official_token_count,
            )
            windows = constrained["windows"]
            for window, item in zip(windows, retrieved):
                window.update(
                    chunk_index=item["chunk_index"],
                    chunk_id=item["chunk_id"],
                    chunk_bm25_score=item["bm25_score"],
                    chunk_text_sha256=item["text_sha256"],
                )
            selection_latency = time.perf_counter() - selection_start
            source_token_count = sum(len(window["token_ids"]) for window in windows)
            if source_token_count != 16:
                raise Matched16ContractError("runtime source token budget is not 16")

            matched_prompt = build_matched_prompt(windows, payload["query_prompt"])
            matched_rendered, matched_token_count = _rendered_prompt(model, matched_prompt)
            capacity_ok = matched_token_count <= capacity
            if not capacity_ok:
                raise Matched16ContractError("matched prompt exceeds model capacity")
            query_payload = eventqa._query_only_payload(payload)
            query_payload["query_prompt"] = matched_prompt
            query_payload["memorization_prompts"] = [matched_prompt]

            _cuda_sync()
            generation_start = time.perf_counter()
            result = eventqa._run_eventqa_model(args, model, capacity, query_payload, "off")
            _cuda_sync()
            generation_latency = time.perf_counter() - generation_start
            if result["rendered_query_prompt"] != matched_rendered:
                raise Matched16ContractError("preflight and runtime prompts differ")
            with tempfile.TemporaryDirectory() as tmpdir:
                score = eventqa._score_prediction(args, payload, result["prediction"], tmpdir)
            query_turn = eventqa._query_turn(result)
            records.append(
                {
                    "context_index": args.context_index,
                    "query_index": query_index,
                    "qa_pair_id": payload["qa_pair_id"],
                    "windows": windows,
                    "source_token_count": source_token_count,
                    "official_query_sha256": _sha256(payload["query_prompt"]),
                    "matched_prompt_sha256": _sha256(matched_prompt),
                    "official_rendered_prompt_sha256": _sha256(official_rendered),
                    "matched_rendered_prompt_sha256": _sha256(matched_rendered),
                    "official_rendered_token_count": official_token_count,
                    "matched_rendered_token_count": matched_token_count,
                    "rendered_prompt_token_delta": matched_token_count - official_token_count,
                    "budget_constraint": {
                        "candidate_ranks": constrained["candidate_ranks"],
                        "fallback_used": constrained["fallback_used"],
                        "score_loss": constrained["score_loss"],
                        "search_limit": constrained["search_limit"],
                        "checked_pair_count": constrained["checked_pair_count"],
                    },
                    "context_capacity": capacity,
                    "capacity_ok": capacity_ok,
                    "prediction": result["prediction"],
                    "substring_exact_match": eventqa._metric_value(
                        score, "substring_exact_match", default=0
                    ),
                    "eventqa_recall": eventqa._metric_value(
                        score, "eventqa_recall", default=0.0
                    ),
                    "format_flags": eventqa._format_flags(result["prediction"]),
                    "cost": {
                        "retrieval_and_window_latency_seconds": selection_latency,
                        "generation_latency_seconds": generation_latency,
                        "end_to_end_latency_seconds": selection_latency + generation_latency,
                        "output_tokens": int(query_turn["output_len"]),
                    },
                }
            )
        peak_gpu_memory = (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else baseline_gpu_memory
        )
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "measurement_mode": "standalone_process",
            "run_id": run_id,
            "scope": {
                "measurement_scope": args.measurement_scope,
                "context_index": args.context_index,
                "question_indices": question_indices,
            },
            "method": {
                "retrieval": "bm25_top2",
                "window_tokens_per_chunk": 8,
                "source_token_budget": 16,
            },
            "cost": {
                "index_construction_latency_seconds": index_latency,
                "baseline_gpu_memory_bytes": baseline_gpu_memory,
                "peak_gpu_memory_bytes": peak_gpu_memory,
                "incremental_peak_gpu_memory_bytes": peak_gpu_memory - baseline_gpu_memory,
            },
            "records": records,
        }
        validate_artifact(artifact)
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        artifact_name = (
            "smoke_artifact.json"
            if args.measurement_scope == "smoke"
            else "full_artifact.json"
        )
        _write_json(output_dir / artifact_name, artifact)
        _write_jsonl(output_dir / "per_question.jsonl", records)
        _write_json(output_dir / "manifest.json", manifest)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
