#!/usr/bin/env python3
"""LoCoMo-QA Disabled and frozen-P7 runner over normalized adapter outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import locomo_qa_scorer as scorer


EXPERIMENT_NAME = "MAB-6B LoCoMo-QA Frozen Runner"
DEFAULT_OUTPUT_ROOT = "outputs/mab/locomo_qa_frozen_runner"
DEFAULT_CHUNK_SIZE = 4096
DEFAULT_GENERATION_MAX_LENGTH = 40
DEFAULT_MODE = "disabled"
DEFAULT_CONSTRUCTION_GRANULARITY = "token_chunk"
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
    "Kana-s/MemGen@269d9b1/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model"
)
DEFAULT_CFG_PATH = "configs/latent_memory/triviaqa.yaml"
DEFAULT_RETRIEVE_THRESHOLD = 0.05
DEFAULT_UPDATE_THRESHOLD = 0.10
DEFAULT_MAX_SLOTS = 16
DEFAULT_TOP_K = 2
DEFAULT_DECAY_ALPHA = 0.05
DEFAULT_RETRIEVE_POLICY = "threshold_topk"
DEFAULT_UPDATE_POLICY = "thread_update"
DEFAULT_SYSTEM_MESSAGE = (
    "You are a helpful assistant that can memorize conversation history for future question answering."
)
DEFAULT_ACK = "Acknowledged."
LOCOMO_QUERY_TEMPLATE = (
    "Based on the conversation history you memorized, answer the question concisely.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)
SPECIAL_TOKEN_PATTERNS = (
    r"<\|[^>]+?\|>",
    r"</?s>",
    r"</?think>",
    r"\[/?INST\]",
)
MARKER_ONLY_LINES = {
    "question",
    "question:",
    "answer",
    "answer:",
    "context",
    "context:",
    "conversation",
    "conversation:",
}
INLINE_TRUNCATION_MARKERS = (
    "Question:",
    "Conversation:",
    "Context:",
    "Answer:",
    "<|",
)
NO_CONTEXT_PATTERNS = (
    "no conversation history",
    "no context provided",
    "no conversation history or context",
    "no conversation history or question provided",
    "there is no conversation history",
)
REFUSAL_PATTERNS = (
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i can't",
    "unable to",
    "not able to",
    "cannot provide an answer",
)
META_REASONING_PATTERNS = (
    "<think>",
    "i need to",
    "to answer the question",
    "search for the answer",
    "using the search function",
    "let me think",
    "judgment required",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_normalized_conversations(path: str | Path) -> List[Dict[str, Any]]:
    return scorer.load_jsonl(path)


def load_normalized_qa_records(path: str | Path) -> List[Dict[str, Any]]:
    return scorer.load_jsonl(path)


def select_conversation(conversations: Sequence[Dict[str, Any]], conversation_id: str) -> Dict[str, Any]:
    for row in conversations:
        if row.get("conversation_id") == conversation_id:
            return row
    raise KeyError(f"Conversation not found: {conversation_id}")


def select_questions(
    qa_rows: Sequence[Dict[str, Any]],
    *,
    conversation_id: str,
    max_questions: int | None = None,
) -> List[Dict[str, Any]]:
    selected = [row for row in qa_rows if row.get("conversation_id") == conversation_id]
    selected = sorted(selected, key=lambda row: int(row.get("question_index", 0)))
    if max_questions is not None:
        selected = selected[:max_questions]
    return selected


def _render_turn(turn: Dict[str, Any]) -> str:
    speaker = turn.get("speaker") or turn.get("role") or "unknown"
    content = turn.get("content") or turn.get("raw_text") or ""
    return f"{speaker}: {content}"


def render_session_text(session_row: Dict[str, Any]) -> str:
    header = f"[Session {session_row['session_id']}"
    timestamp = session_row.get("timestamp")
    if timestamp:
        header += f" | {timestamp}"
    header += "]"
    turn_lines = [_render_turn(turn) for turn in session_row.get("turns", [])]
    return "\n".join([header, *turn_lines]).strip()


def render_conversation_blocks(conversation_row: Dict[str, Any]) -> List[str]:
    return [render_session_text(session) for session in conversation_row.get("sessions", [])]


def build_session_granularity_chunks(
    conversation_row: Dict[str, Any],
    *,
    token_counter: Callable[[str], int] | None = None,
) -> tuple[List[str], List[int], List[int]]:
    if token_counter is None:
        token_counter = lambda text: len(text.split())
    sessions = list(conversation_row.get("sessions", []))
    chunks = [render_session_text(session) for session in sessions]
    chunk_token_lengths = [token_counter(chunk) for chunk in chunks]
    session_ids = [
        int(session.get("session_id"))
        for session in sessions
        if session.get("session_id") is not None
    ]
    if not chunks:
        raise RuntimeError("No session chunks were produced from normalized conversation sessions")
    return chunks, chunk_token_lengths, session_ids


def chunk_text_by_token_budget(
    blocks: Sequence[str],
    *,
    chunk_size: int,
    token_counter: Callable[[str], int] | None = None,
) -> tuple[List[str], List[int]]:
    if token_counter is None:
        token_counter = lambda text: len(text.split())

    chunks: List[str] = []
    chunk_token_lengths: List[int] = []
    current_lines: List[str] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = token_counter(block)
        if current_lines and current_tokens + block_tokens > chunk_size:
            chunk = "\n\n".join(current_lines)
            chunks.append(chunk)
            chunk_token_lengths.append(token_counter(chunk))
            current_lines = [block]
            current_tokens = block_tokens
        else:
            current_lines.append(block)
            current_tokens += block_tokens

    if current_lines:
        chunk = "\n\n".join(current_lines)
        chunks.append(chunk)
        chunk_token_lengths.append(token_counter(chunk))

    if not chunks:
        raise RuntimeError("No chunks were produced from normalized conversation blocks")
    return chunks, chunk_token_lengths


def build_memorization_prompt(chunk: str, *, chunk_index: int, chunk_count: int) -> str:
    return (
        f"Please memorize the following conversation chunk ({chunk_index + 1}/{chunk_count}) "
        "for future question answering.\n\n"
        f"{chunk}"
    )


def build_conversation_payload(
    conversation_row: Dict[str, Any],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    construction_granularity: str = DEFAULT_CONSTRUCTION_GRANULARITY,
    token_counter: Callable[[str], int] | None = None,
) -> Dict[str, Any]:
    if construction_granularity == "session":
        chunks, chunk_token_lengths, session_ids = build_session_granularity_chunks(
            conversation_row,
            token_counter=token_counter,
        )
        construction_chunk_unit = "session"
    elif construction_granularity == "token_chunk":
        blocks = render_conversation_blocks(conversation_row)
        chunks, chunk_token_lengths = chunk_text_by_token_budget(
            blocks,
            chunk_size=chunk_size,
            token_counter=token_counter,
        )
        session_ids = [
            int(session.get("session_id"))
            for session in conversation_row.get("sessions", [])
            if session.get("session_id") is not None
        ]
        construction_chunk_unit = "token_chunk"
    else:
        raise ValueError(f"Unsupported construction_granularity: {construction_granularity}")
    memorization_prompts = [
        build_memorization_prompt(chunk, chunk_index=index, chunk_count=len(chunks))
        for index, chunk in enumerate(chunks)
    ]
    return {
        "context_id": conversation_row["conversation_id"],
        "context_index": int(conversation_row.get("sample_index", 0)),
        "chunks": chunks,
        "chunk_token_lengths": chunk_token_lengths,
        "memorization_prompts": memorization_prompts,
        "construction_granularity": construction_granularity,
        "construction_chunk_count": len(chunks),
        "construction_chunk_unit": construction_chunk_unit,
        "construction_sessions_covered": session_ids,
        "session_count": conversation_row.get("session_count"),
        "turn_count": conversation_row.get("turn_count"),
        "source_dataset": conversation_row.get("source_dataset"),
        "source_path": conversation_row.get("source_path"),
    }


def build_question_payload(
    conversation_row: Dict[str, Any],
    qa_row: Dict[str, Any],
    *,
    conversation_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = conversation_payload or {}
    return {
        "context_id": conversation_row["conversation_id"],
        "context_index": int(conversation_row.get("sample_index", 0)),
        "question_id": qa_row["question_id"],
        "query_id": int(qa_row.get("question_index", 0)),
        "question_type": qa_row.get("category_name"),
        "qa_pair_id": qa_row["question_id"],
        "chunks": list(payload.get("chunks", [])),
        "chunk_token_lengths": list(payload.get("chunk_token_lengths", [])),
        "memorization_prompts": list(payload.get("memorization_prompts", [])),
        "query_prompt": LOCOMO_QUERY_TEMPLATE.format(question=qa_row["question_text"]),
        "question": qa_row["question_text"],
        "gold_answers": list(qa_row.get("reference_answers", [])) or [qa_row.get("gold_answer")],
        "gold_answer": qa_row.get("gold_answer"),
        "category": qa_row.get("category"),
        "category_name": qa_row.get("category_name"),
        "construction_granularity": payload.get("construction_granularity"),
        "construction_chunk_count": payload.get("construction_chunk_count"),
        "construction_chunk_unit": payload.get("construction_chunk_unit"),
        "construction_sessions_covered": list(payload.get("construction_sessions_covered", [])),
    }


def query_only_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    query_payload = dict(payload)
    query_payload["chunks"] = []
    query_payload["chunk_token_lengths"] = []
    query_payload["memorization_prompts"] = [payload["query_prompt"]]
    return query_payload


def _coerce_prediction_text(value: str | None) -> str:
    if value is None:
        return ""
    return str(value)


def _strip_special_tokens(text: str) -> str:
    cleaned = text
    for pattern in SPECIAL_TOKEN_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def _strip_chat_prefixes(text: str) -> str:
    prefixes = (
        "assistant:",
        "assistant",
        "final answer:",
        "response:",
    )
    stripped = text.strip()
    lowered = stripped.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return stripped[len(prefix):].strip()
    return stripped


def _truncate_prompt_leakage(text: str) -> str:
    candidate = text.strip()
    if not candidate:
        return ""
    lowered = candidate.lower()
    if lowered in MARKER_ONLY_LINES:
        return ""
    cut_positions = []
    for marker in INLINE_TRUNCATION_MARKERS:
        index = candidate.find(marker)
        if index != -1:
            cut_positions.append(index)
    if " question" in lowered:
        cut_positions.append(lowered.find(" question"))
    if cut_positions:
        cut_index = min(index for index in cut_positions if index >= 0)
        candidate = candidate[:cut_index].strip()
    lowered = candidate.lower()
    if lowered in MARKER_ONLY_LINES:
        return ""
    return candidate


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def extract_prediction_contract(raw_prediction_text: str | None, *, question_text: str | None = None) -> Dict[str, Any]:
    raw_text = _coerce_prediction_text(raw_prediction_text)
    lowered_raw = raw_text.lower()
    prompt_leak = any(marker in raw_text for marker in ("Question:", "Conversation:", "Context:", "Answer:", "<|", "<think>", "</think>"))
    if not prompt_leak:
        raw_lines = [line.strip().lower() for line in raw_text.replace("\r\n", "\n").split("\n") if line.strip()]
        prompt_leak = any(line in MARKER_ONLY_LINES for line in raw_lines)
    source_text = raw_text.rsplit("Answer:", 1)[-1] if "Answer:" in raw_text else raw_text
    source_text = _strip_special_tokens(source_text).replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for raw_line in source_text.split("\n"):
        line = _strip_chat_prefixes(raw_line)
        line = _truncate_prompt_leakage(line)
        if line:
            lines.append(line)
    cleaned_text = lines[0].strip() if lines else ""
    cleaned_text = _truncate_prompt_leakage(cleaned_text)

    normalized_raw = _normalize_for_match(raw_text)
    normalized_question = _normalize_for_match(question_text or "")
    question_restatement = bool(
        normalized_question
        and normalized_raw
        and (
            normalized_raw.startswith(normalized_question)
            or normalized_question in normalized_raw
            or _normalize_for_match(cleaned_text) == normalized_question
        )
    )
    no_context_denial = any(pattern in lowered_raw for pattern in NO_CONTEXT_PATTERNS)
    refusal = any(pattern in lowered_raw for pattern in REFUSAL_PATTERNS)
    meta_reasoning_or_search = any(pattern in lowered_raw for pattern in META_REASONING_PATTERNS)
    answer_extraction_failed = not bool(cleaned_text)

    return {
        "prediction_text": cleaned_text,
        "raw_prediction_text": raw_text,
        "prompt_leak": bool(prompt_leak or question_restatement),
        "question_restatement": question_restatement,
        "no_context_denial": no_context_denial,
        "refusal": refusal,
        "meta_reasoning_or_search": meta_reasoning_or_search,
        "answer_extraction_failed": answer_extraction_failed,
    }


def default_diagnostics_for_mode(mode: str) -> Dict[str, Any]:
    zeroes = {
        "construction_write_count": 0,
        "construction_retrieve_count": 0,
        "query_retrieval_active_count": 0,
        "retrieved_latent_count": 0,
        "query_write_count": 0,
        "final_slot_count": 0,
        "trigger_call_count": 0,
        "weaver_call_count": 0,
        "latency_seconds": 0.0,
        "peak_gpu_memory": None,
        "output_token_count": None,
        "query_write_attempt_count": 0,
        "bank_snapshot_changed_after_query": False,
        "construction_granularity": DEFAULT_CONSTRUCTION_GRANULARITY,
        "construction_chunk_count": 0,
        "construction_chunk_unit": "token_chunk",
        "construction_sessions_covered": [],
    }
    zeroes["mode"] = mode
    return zeroes


def build_prediction_record(
    *,
    mode: str,
    conversation_id: str,
    qa_row: Dict[str, Any],
    prediction_text: str | None,
    raw_prediction_text: str | None,
    diagnostics: Dict[str, Any],
    output_contract: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    merged = default_diagnostics_for_mode(mode)
    merged.update(diagnostics)
    contract = output_contract or extract_prediction_contract(
        raw_prediction_text if raw_prediction_text is not None else prediction_text,
        question_text=qa_row.get("question_text"),
    )
    cleaned_prediction = contract["prediction_text"] if prediction_text is None else prediction_text
    raw_prediction = contract["raw_prediction_text"] if raw_prediction_text is None else raw_prediction_text
    status = "ok"
    if cleaned_prediction is None:
        status = "missing"
    elif not str(cleaned_prediction).strip():
        status = "empty"
    return {
        "conversation_id": conversation_id,
        "question_id": qa_row["question_id"],
        "category": qa_row.get("category"),
        "category_name": qa_row.get("category_name"),
        "question": qa_row.get("question_text"),
        "gold_answer": qa_row.get("gold_answer"),
        "prediction": cleaned_prediction,
        "prediction_text": cleaned_prediction,
        "raw_prediction_text": raw_prediction,
        "prediction_status": status,
        "method": mode,
        "mode": mode,
        "prompt_leak": bool(contract["prompt_leak"]),
        "question_restatement": bool(contract["question_restatement"]),
        "no_context_denial": bool(contract["no_context_denial"]),
        "refusal": bool(contract["refusal"]),
        "meta_reasoning_or_search": bool(contract["meta_reasoning_or_search"]),
        "answer_extraction_failed": bool(contract["answer_extraction_failed"]),
        "construction_granularity": merged["construction_granularity"],
        "construction_chunk_count": int(merged["construction_chunk_count"]),
        "construction_chunk_unit": merged["construction_chunk_unit"],
        "construction_sessions_covered": list(merged.get("construction_sessions_covered", [])),
        "construction_write_count": int(merged["construction_write_count"]),
        "construction_retrieve_count": int(merged["construction_retrieve_count"]),
        "query_retrieval_active_count": int(merged["query_retrieval_active_count"]),
        "retrieved_latent_count": int(merged["retrieved_latent_count"]),
        "query_write_count": int(merged["query_write_count"]),
        "final_slot_count": int(merged["final_slot_count"]),
        "trigger_call_count": int(merged["trigger_call_count"]),
        "weaver_call_count": int(merged["weaver_call_count"]),
        "latency_seconds": float(merged["latency_seconds"]),
        "peak_gpu_memory": merged["peak_gpu_memory"],
        "output_token_count": merged["output_token_count"],
        "query_write_attempt_count": int(merged["query_write_attempt_count"]),
        "bank_snapshot_changed_after_query": bool(merged["bank_snapshot_changed_after_query"]),
    }


def assert_zero_query_writes(row: Dict[str, Any]) -> None:
    if int(row.get("query_write_count", 0)) != 0:
        raise RuntimeError(f"query_write_count must be zero, got {row.get('query_write_count')}")


def score_predictions(
    qa_rows: Sequence[Dict[str, Any]],
    prediction_rows: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    scored_rows = scorer.score_prediction_records(qa_rows, prediction_rows)
    scored_by_question = {row["question_id"]: row for row in scored_rows}
    merged_rows: List[Dict[str, Any]] = []
    for prediction_row in prediction_rows:
        merged = dict(prediction_row)
        merged.update(scored_by_question[prediction_row["question_id"]])
        merged_rows.append(merged)

    aggregate_method = None
    if prediction_rows:
        aggregate_method = prediction_rows[0].get("mode", prediction_rows[0].get("method"))
    aggregate = scorer.aggregate_scores(merged_rows, method=aggregate_method)
    if prediction_rows:
        aggregate["mode"] = aggregate_method
    aggregate["cost_summary"] = build_cost_summary(prediction_rows)
    return merged_rows, aggregate


def build_cost_summary(prediction_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not prediction_rows:
        return {}

    def _mean_int(field: str) -> float:
        return fmean(int(row.get(field, 0) or 0) for row in prediction_rows)

    latencies = [float(row.get("latency_seconds", 0.0) or 0.0) for row in prediction_rows]
    peak_values = [row.get("peak_gpu_memory") for row in prediction_rows if row.get("peak_gpu_memory") is not None]
    output_tokens = [row.get("output_token_count") for row in prediction_rows if row.get("output_token_count") is not None]
    return {
        "mean_latency_seconds": fmean(latencies),
        "max_peak_gpu_memory": max(peak_values) if peak_values else None,
        "mean_output_token_count": fmean(float(value) for value in output_tokens) if output_tokens else None,
        "mean_construction_write_count": _mean_int("construction_write_count"),
        "mean_construction_retrieve_count": _mean_int("construction_retrieve_count"),
        "mean_query_retrieval_active_count": _mean_int("query_retrieval_active_count"),
        "mean_retrieved_latent_count": _mean_int("retrieved_latent_count"),
        "mean_query_write_count": _mean_int("query_write_count"),
        "mean_final_slot_count": _mean_int("final_slot_count"),
        "mean_trigger_call_count": _mean_int("trigger_call_count"),
        "mean_weaver_call_count": _mean_int("weaver_call_count"),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def build_snapshot_metadata(conversation_id: str, mode: str, *, final_slot_count: int) -> Dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "mode": mode,
        "final_slot_count": final_slot_count,
    }


def _p7_bank_config() -> Dict[str, Any]:
    from scripts.eval import mab3_bank_on_full_history as mab3

    config = mab3.version_a_bank_config(
        top_k=DEFAULT_TOP_K,
        threshold=DEFAULT_RETRIEVE_THRESHOLD,
        retrieve_policy=DEFAULT_RETRIEVE_POLICY,
    )
    config["retrieve_threshold"] = DEFAULT_RETRIEVE_THRESHOLD
    config["update_threshold"] = DEFAULT_UPDATE_THRESHOLD
    config["max_slots"] = DEFAULT_MAX_SLOTS
    config["top_k"] = DEFAULT_TOP_K
    config["decay_alpha"] = DEFAULT_DECAY_ALPHA
    config["retrieve_policy"] = DEFAULT_RETRIEVE_POLICY
    config["update_policy"] = DEFAULT_UPDATE_POLICY
    return config


def _recorded_bank_config() -> Dict[str, Any]:
    config = _p7_bank_config()
    config["threshold"] = config["retrieve_threshold"]
    return config


def _load_model_for_mode(args, mode: str):
    if mode == "disabled":
        from scripts.eval import mab5a_detectiveqa_compressed_n10 as mab5a

        return mab5a._load_model(args)

    if mode == "p7":
        from scripts.eval import mab6b_weaver_space_bank_detectiveqa_n10 as weaver_bank

        return weaver_bank._load_model(args)

    raise ValueError(f"Unsupported mode: {mode}")


def _run_disabled_query(args, model, capacity: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    from scripts.eval import mab5a_detectiveqa_compressed_n10 as mab5a

    return mab5a._run_model(args, model, capacity, payload, "off")


def _run_p7_construction(args, model, capacity: int, payload: Dict[str, Any]) -> tuple[Dict[str, Any], Any]:
    from scripts.eval import mab6b_weaver_space_bank_eventqa_65536_n5 as eventqa

    recorded = _recorded_bank_config()
    result = eventqa._run_eventqa_model(
        args,
        model,
        capacity,
        payload,
        "on",
        _p7_bank_config(),
        preserve_bank=True,
        construction_only=True,
        recorded_bank_config=recorded,
        score_trace_state=None,
    )
    frozen_bank = result.pop("_retained_bank", None)
    if frozen_bank is None:
        raise RuntimeError("Construction did not retain a frozen bank")
    return result, frozen_bank


def _run_p7_query(args, model, capacity: int, payload: Dict[str, Any], frozen_bank) -> Dict[str, Any]:
    from scripts.eval import mab6b_weaver_space_bank_eventqa_65536_n5 as eventqa

    recorded = _recorded_bank_config()
    result = eventqa._run_eventqa_model(
        args,
        model,
        capacity,
        payload,
        "on",
        _p7_bank_config(),
        external_bank=frozen_bank,
        preserve_bank=True,
        recorded_bank_config=recorded,
        score_trace_state=None,
    )
    retained_bank = result.pop("_retained_bank", None)
    if retained_bank is not frozen_bank:
        raise RuntimeError("Frozen bank object changed across query turns")
    return result


def _question_runtime_diagnostics(result: Dict[str, Any]) -> Dict[str, Any]:
    generations = list(result.get("generations", []))
    last_generation = generations[-1] if generations else {}
    query_write_count = int(
        result.get("query_write_count_delta", result.get("query_write_count", 0)) or 0
    )
    return {
        "query_retrieval_active_count": int(bool(last_generation.get("retrieved_latent_count", 0))),
        "retrieved_latent_count": int(last_generation.get("retrieved_latent_count", 0) or 0),
        "query_write_count": query_write_count,
        "trigger_call_count": int(last_generation.get("trigger_count", 0) or 0),
        "weaver_call_count": int(last_generation.get("trigger_positive_count", 0) or 0),
        "latency_seconds": float(result.get("latency_seconds", 0.0) or 0.0),
        "peak_gpu_memory": result.get("peak_cuda_memory"),
        "output_token_count": int(last_generation.get("output_len", 0) or 0),
        "query_write_attempt_count": int(
            result.get("query_write_attempt_count_delta", result.get("query_write_attempt_count", 0)) or 0
        ),
        "bank_snapshot_changed_after_query": bool(result.get("bank_snapshot_changed_after_query", False)),
    }


def _construction_diagnostics_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    pre_query = result.get("pre_query_bank_summary") or {}
    generations = list(result.get("generations", []))
    return {
        "construction_write_count": int(pre_query.get("write_count", 0) or 0),
        "construction_retrieve_count": int(pre_query.get("retrieval_count", 0) or 0),
        "final_slot_count": int(pre_query.get("slot_count", 0) or 0),
        "trigger_call_count": sum(int(gen.get("trigger_count", 0) or 0) for gen in generations),
        "weaver_call_count": sum(int(gen.get("trigger_positive_count", 0) or 0) for gen in generations),
        "construction_latency_seconds": float(result.get("latency_seconds", 0.0) or 0.0),
        "construction_peak_gpu_memory": result.get("peak_cuda_memory"),
        "construction_granularity": DEFAULT_CONSTRUCTION_GRANULARITY,
        "construction_chunk_count": 0,
        "construction_chunk_unit": "token_chunk",
        "construction_sessions_covered": [],
    }


def _disabled_zero_diagnostics() -> Dict[str, Any]:
    return {
        "construction_write_count": 0,
        "construction_retrieve_count": 0,
        "final_slot_count": 0,
        "trigger_call_count": 0,
        "weaver_call_count": 0,
        "construction_latency_seconds": 0.0,
        "construction_peak_gpu_memory": None,
        "construction_granularity": DEFAULT_CONSTRUCTION_GRANULARITY,
        "construction_chunk_count": 0,
        "construction_chunk_unit": "token_chunk",
        "construction_sessions_covered": [],
    }


def _prepare_runtime_payloads(
    conversation_row: Dict[str, Any],
    qa_rows: Sequence[Dict[str, Any]],
    *,
    chunk_size: int,
    construction_granularity: str,
    token_counter: Callable[[str], int] | None = None,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    conversation_payload = build_conversation_payload(
        conversation_row,
        chunk_size=chunk_size,
        construction_granularity=construction_granularity,
        token_counter=token_counter,
    )
    question_payloads = [
        build_question_payload(conversation_row, qa_row, conversation_payload=conversation_payload)
        for qa_row in qa_rows
    ]
    return conversation_payload, question_payloads


def _serialize_run_summary(
    *,
    output_dir: Path,
    mode: str,
    conversation_id: str,
    selected_question_ids: Sequence[str],
    aggregate_metrics: Dict[str, Any],
    query_write_zero_passed: bool,
) -> None:
    lines = [
        f"# LoCoMo-QA {mode} Smoke Summary",
        "",
        f"- conversation_id: `{conversation_id}`",
        f"- question_ids: `{list(selected_question_ids)}`",
        f"- exact_match_mean: `{aggregate_metrics['overall_micro']['exact_match_mean']}`",
        f"- token_f1_mean: `{aggregate_metrics['overall_micro']['token_f1_mean']}`",
        f"- invalid_output_count: `{aggregate_metrics['invalid_output_count']}`",
        f"- query_write_count_zero: `{query_write_zero_passed}`",
    ]
    (output_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_disabled_mode(args, output_dir: Path) -> Dict[str, Any]:
    conversations = load_normalized_conversations(args.normalized_conversations)
    qa_records = load_normalized_qa_records(args.normalized_qa_records)
    conversation_row = select_conversation(conversations, args.conversation_id)
    selected_qa = select_questions(qa_records, conversation_id=args.conversation_id, max_questions=args.max_questions)
    conversation_payload, question_payloads = _prepare_runtime_payloads(
        conversation_row,
        selected_qa,
        chunk_size=args.chunk_size,
        construction_granularity=args.construction_granularity,
    )
    model, capacity = _load_model_for_mode(args, "disabled")
    prediction_rows: List[Dict[str, Any]] = []

    for qa_row, payload in zip(selected_qa, question_payloads):
        result = _run_disabled_query(args, model, capacity, payload)
        diagnostics = dict(_disabled_zero_diagnostics())
        diagnostics.update(_question_runtime_diagnostics(result))
        diagnostics.update(
            {
                "construction_granularity": conversation_payload["construction_granularity"],
                "construction_chunk_count": conversation_payload["construction_chunk_count"],
                "construction_chunk_unit": conversation_payload["construction_chunk_unit"],
                "construction_sessions_covered": conversation_payload["construction_sessions_covered"],
            }
        )
        output_contract = extract_prediction_contract(
            result.get("prediction"),
            question_text=qa_row.get("question_text"),
        )
        prediction_rows.append(
            build_prediction_record(
                mode="disabled",
                conversation_id=conversation_row["conversation_id"],
                qa_row=qa_row,
                prediction_text=output_contract["prediction_text"],
                raw_prediction_text=output_contract["raw_prediction_text"],
                diagnostics=diagnostics,
                output_contract=output_contract,
            )
        )

    scored_rows, aggregate_metrics = score_predictions(selected_qa, prediction_rows)
    diagnostics = {
        "mode": "disabled",
        "conversation_id": conversation_row["conversation_id"],
        "selected_question_ids": [row["question_id"] for row in selected_qa],
        "chunk_count": len(conversation_payload["chunks"]),
        "chunk_token_lengths": conversation_payload["chunk_token_lengths"],
        "construction_granularity": conversation_payload["construction_granularity"],
        "construction_chunk_count": conversation_payload["construction_chunk_count"],
        "construction_chunk_unit": conversation_payload["construction_chunk_unit"],
        "construction_sessions_covered": conversation_payload["construction_sessions_covered"],
        "gpu_used": bool(any(row["peak_gpu_memory"] is not None for row in prediction_rows)),
        "query_write_count_zero": all(int(row["query_write_count"]) == 0 for row in prediction_rows),
    }

    _write_jsonl(output_dir / "prediction_records.jsonl", prediction_rows)
    _write_jsonl(output_dir / "scored_prediction_records.jsonl", scored_rows)
    _write_json(output_dir / "aggregate_metrics.json", aggregate_metrics)
    _write_json(output_dir / "run_diagnostics.json", diagnostics)
    _serialize_run_summary(
        output_dir=output_dir,
        mode="disabled",
        conversation_id=conversation_row["conversation_id"],
        selected_question_ids=diagnostics["selected_question_ids"],
        aggregate_metrics=aggregate_metrics,
        query_write_zero_passed=diagnostics["query_write_count_zero"],
    )
    return diagnostics


def _run_p7_mode(args, output_dir: Path) -> Dict[str, Any]:
    conversations = load_normalized_conversations(args.normalized_conversations)
    qa_records = load_normalized_qa_records(args.normalized_qa_records)
    conversation_row = select_conversation(conversations, args.conversation_id)
    selected_qa = select_questions(qa_records, conversation_id=args.conversation_id, max_questions=args.max_questions)
    conversation_payload, question_payloads = _prepare_runtime_payloads(
        conversation_row,
        selected_qa,
        chunk_size=args.chunk_size,
        construction_granularity=args.construction_granularity,
    )
    model, capacity = _load_model_for_mode(args, "p7")
    construction_payload = dict(question_payloads[0])
    construction_result, frozen_bank = _run_p7_construction(args, model, capacity, construction_payload)
    construction_diag = _construction_diagnostics_from_result(construction_result)
    construction_diag.update(
        {
            "construction_granularity": conversation_payload["construction_granularity"],
            "construction_chunk_count": conversation_payload["construction_chunk_count"],
            "construction_chunk_unit": conversation_payload["construction_chunk_unit"],
            "construction_sessions_covered": conversation_payload["construction_sessions_covered"],
        }
    )
    prediction_rows: List[Dict[str, Any]] = []

    for qa_row, payload in zip(selected_qa, question_payloads):
        result = _run_p7_query(args, model, capacity, query_only_payload(payload), frozen_bank)
        query_diag = _question_runtime_diagnostics(result)
        diagnostics = dict(construction_diag)
        diagnostics.update(query_diag)
        output_contract = extract_prediction_contract(
            result.get("prediction"),
            question_text=qa_row.get("question_text"),
        )
        prediction_row = build_prediction_record(
            mode="p7",
            conversation_id=conversation_row["conversation_id"],
            qa_row=qa_row,
            prediction_text=output_contract["prediction_text"],
            raw_prediction_text=output_contract["raw_prediction_text"],
            diagnostics=diagnostics,
            output_contract=output_contract,
        )
        assert_zero_query_writes(prediction_row)
        prediction_rows.append(prediction_row)

    scored_rows, aggregate_metrics = score_predictions(selected_qa, prediction_rows)
    diagnostics = {
        "mode": "p7",
        "conversation_id": conversation_row["conversation_id"],
        "selected_question_ids": [row["question_id"] for row in selected_qa],
        "chunk_count": len(conversation_payload["chunks"]),
        "chunk_token_lengths": conversation_payload["chunk_token_lengths"],
        "construction_granularity": conversation_payload["construction_granularity"],
        "construction_chunk_count": conversation_payload["construction_chunk_count"],
        "construction_chunk_unit": conversation_payload["construction_chunk_unit"],
        "construction_sessions_covered": conversation_payload["construction_sessions_covered"],
        "gpu_used": bool(any(row["peak_gpu_memory"] is not None for row in prediction_rows) or construction_diag["construction_peak_gpu_memory"] is not None),
        "query_write_count_zero": all(int(row["query_write_count"]) == 0 for row in prediction_rows),
        "snapshot_metadata": build_snapshot_metadata(
            conversation_row["conversation_id"],
            "p7",
            final_slot_count=construction_diag["final_slot_count"],
        ),
        "construction_diagnostics": construction_diag,
    }

    _write_jsonl(output_dir / "prediction_records.jsonl", prediction_rows)
    _write_jsonl(output_dir / "scored_prediction_records.jsonl", scored_rows)
    _write_json(output_dir / "aggregate_metrics.json", aggregate_metrics)
    _write_json(output_dir / "run_diagnostics.json", diagnostics)
    _serialize_run_summary(
        output_dir=output_dir,
        mode="p7",
        conversation_id=conversation_row["conversation_id"],
        selected_question_ids=diagnostics["selected_question_ids"],
        aggregate_metrics=aggregate_metrics,
        query_write_zero_passed=diagnostics["query_write_count_zero"],
    )
    return diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--mode", choices=("disabled", "p7"), default=DEFAULT_MODE)
    parser.add_argument("--normalized-conversations", required=True)
    parser.add_argument("--normalized-qa-records", required=True)
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--max-questions", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--construction-granularity",
        choices=("token_chunk", "session"),
        default=DEFAULT_CONSTRUCTION_GRANULARITY,
        help="Construction input protocol. Session-level is supported; turn-level remains a later diagnostic TODO.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--model-checkpoint-id", default=DEFAULT_MODEL_CHECKPOINT_ID)
    parser.add_argument("--cfg-path", default=DEFAULT_CFG_PATH)
    parser.add_argument("--dataset-root", default="/mnt/18T/baishilong/datasets/MemoryAgentBench")
    parser.add_argument("--mab-repo", default="/mnt/18T/baishilong/benchmarks/MemoryAgentBench")
    parser.add_argument("--mab-python", default="/home/baishilong/miniconda3/envs/MABench/bin/python")
    parser.add_argument("--parquet", default="/mnt/18T/baishilong/datasets/MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet")
    parser.add_argument("--data-config", default="configs/data_conf/Accurate_Retrieval/EventQA/Eventqa_64k.yaml")
    parser.add_argument("--generation-max-length", type=int, default=DEFAULT_GENERATION_MAX_LENGTH)
    parser.add_argument("--eventqa-protocol", default="frozen_context_bank")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "disabled":
        diagnostics = _run_disabled_mode(args, output_dir)
    else:
        diagnostics = _run_p7_mode(args, output_dir)
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
