#!/usr/bin/env python3
"""Normalize local LoCoMo-QA data into stable JSONL artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


def _load_dataset(path: str | Path) -> List[Dict[str, Any]]:
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in {dataset_path}")
    return data


def _session_ids(conversation: Dict[str, Any]) -> List[int]:
    session_ids = []
    for key, value in conversation.items():
        if key.startswith("session_") and isinstance(value, list):
            try:
                session_ids.append(int(key.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return sorted(session_ids)


def _normalize_role(speaker: Any) -> str:
    if speaker is None:
        return "unknown"
    return str(speaker)


def _evidence_session_ids(evidence_items: Sequence[Any]) -> List[int]:
    session_ids: List[int] = []
    for item in evidence_items:
        if not isinstance(item, str):
            continue
        if not item.startswith("D") or ":" not in item:
            continue
        session_part = item[1:].split(":", 1)[0]
        try:
            session_id = int(session_part)
        except ValueError:
            continue
        if session_id not in session_ids:
            session_ids.append(session_id)
    return session_ids


def _normalize_conversation(
    sample: Dict[str, Any],
    *,
    dataset_path: Path,
    sample_index: int,
) -> Dict[str, Any]:
    conversation = sample["conversation"]
    session_order = _session_ids(conversation)
    sessions: List[Dict[str, Any]] = []
    turn_count = 0

    for session_id in session_order:
        session_key = f"session_{session_id}"
        timestamp = conversation.get(f"{session_key}_date_time")
        session_turns: List[Dict[str, Any]] = []
        for turn in conversation.get(session_key, []):
            raw_text = str(turn.get("text", ""))
            normalized_turn = {
                "turn_id": turn.get("dia_id"),
                "speaker": turn.get("speaker"),
                "role": _normalize_role(turn.get("speaker")),
                "content": raw_text,
                "raw_text": raw_text,
                "timestamp": timestamp,
                "session_id": session_id,
            }
            session_turns.append(normalized_turn)
            turn_count += 1
        sessions.append({
            "session_id": session_id,
            "timestamp": timestamp,
            "turn_count": len(session_turns),
            "turns": session_turns,
        })

    return {
        "conversation_id": sample["sample_id"],
        "source_dataset": dataset_path.stem,
        "source_path": str(dataset_path),
        "sample_index": sample_index,
        "speaker_a": conversation.get("speaker_a"),
        "speaker_b": conversation.get("speaker_b"),
        "session_count": len(sessions),
        "turn_count": turn_count,
        "session_order": session_order,
        "sessions": sessions,
        "session_summary": sample.get("session_summary", {}),
    }


def _normalize_qa_rows(
    sample: Dict[str, Any],
    *,
    sample_index: int,
    max_questions: int | None = None,
) -> List[Dict[str, Any]]:
    conversation_id = sample["sample_id"]
    rows: List[Dict[str, Any]] = []

    for question_index, qa in enumerate(sample.get("qa", [])):
        if max_questions is not None and len(rows) >= max_questions:
            break
        evidence = list(qa.get("evidence", []))
        gold_answer = qa.get("answer")
        category = qa.get("category")
        rows.append({
            "question_id": f"{conversation_id}::q{question_index:03d}",
            "reference_id": f"{conversation_id}::q{question_index:03d}",
            "conversation_id": conversation_id,
            "sample_index": sample_index,
            "question_index": question_index,
            "question_text": qa.get("question"),
            "gold_answer": gold_answer,
            "reference_answers": [] if gold_answer in (None, "") else [gold_answer],
            "category": category,
            "category_name": CATEGORY_NAMES.get(category, "unknown"),
            "evidence": evidence,
            "evidence_turn_ids": evidence,
            "evidence_session_ids": _evidence_session_ids(evidence),
            "metadata": {
                "source_has_native_question_id": False,
            },
        })
    return rows


def inspect_dataset(path: str | Path) -> Dict[str, Any]:
    dataset_path = Path(path)
    samples = _load_dataset(dataset_path)
    category_counts: Counter[int] = Counter()
    session_counts: List[int] = []
    turn_counts: List[int] = []
    qa_count = 0

    for sample in samples:
        conversation = sample["conversation"]
        session_ids = _session_ids(conversation)
        session_counts.append(len(session_ids))
        turn_counts.append(sum(len(conversation[f"session_{session_id}"]) for session_id in session_ids))
        qa_count += len(sample.get("qa", []))
        for qa in sample.get("qa", []):
            category = qa.get("category")
            if category is not None:
                category_counts[int(category)] += 1

    return {
        "source_dataset": dataset_path.stem,
        "source_path": str(dataset_path),
        "conversation_count": len(samples),
        "qa_count": qa_count,
        "category_counts": {str(key): category_counts[key] for key in sorted(category_counts)},
        "session_count_range": [min(session_counts), max(session_counts)] if session_counts else [0, 0],
        "turn_count_range": [min(turn_counts), max(turn_counts)] if turn_counts else [0, 0],
    }


def extract_records(
    path: str | Path,
    *,
    conversation_ids: Sequence[str] | None = None,
    max_questions: int | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dataset_path = Path(path)
    samples = _load_dataset(dataset_path)
    requested_ids = set(conversation_ids or [])
    conversations: List[Dict[str, Any]] = []
    qa_rows: List[Dict[str, Any]] = []

    for sample_index, sample in enumerate(samples):
        conversation_id = sample.get("sample_id")
        if requested_ids and conversation_id not in requested_ids:
            continue
        conversations.append(_normalize_conversation(sample, dataset_path=dataset_path, sample_index=sample_index))
        qa_rows.extend(_normalize_qa_rows(sample, sample_index=sample_index, max_questions=max_questions))

    return conversations, qa_rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _build_summary(conversations: Sequence[Dict[str, Any]], qa_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    category_counts: Counter[str] = Counter()
    for row in qa_rows:
        category_counts[row["category_name"]] += 1
    return {
        "conversation_count": len(conversations),
        "qa_count": len(qa_rows),
        "conversation_ids": [conversation["conversation_id"] for conversation in conversations],
        "category_counts": dict(sorted(category_counts.items())),
    }


def write_smoke_subset(
    path: str | Path,
    output_dir: str | Path,
    *,
    conversation_ids: Sequence[str] | None = None,
    max_questions: int = 5,
) -> Dict[str, Path]:
    output_path = Path(output_dir)
    conversations, qa_rows = extract_records(path, conversation_ids=conversation_ids, max_questions=max_questions)
    conversations_path = output_path / "normalized_conversations.jsonl"
    qa_records_path = output_path / "normalized_qa_records.jsonl"
    summary_path = output_path / "adapter_summary.json"
    _write_jsonl(conversations_path, conversations)
    _write_jsonl(qa_records_path, qa_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(_build_summary(conversations, qa_rows), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "conversations_path": conversations_path,
        "qa_records_path": qa_records_path,
        "summary_path": summary_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoCoMo QA adapter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect the local LoCoMo dataset")
    inspect_parser.add_argument("--input", required=True, help="Path to locomo10.json")

    extract_parser = subparsers.add_parser("extract-qa", help="Extract normalized conversations and QA records")
    extract_parser.add_argument("--input", required=True, help="Path to locomo10.json")
    extract_parser.add_argument("--output-dir", required=True, help="Output directory")
    extract_parser.add_argument("--conversation-id", action="append", dest="conversation_ids")
    extract_parser.add_argument("--max-questions", type=int)

    smoke_parser = subparsers.add_parser("write-smoke-subset", help="Write a small normalized smoke subset")
    smoke_parser.add_argument("--input", required=True, help="Path to locomo10.json")
    smoke_parser.add_argument("--output-dir", required=True, help="Output directory")
    smoke_parser.add_argument("--conversation-id", action="append", dest="conversation_ids")
    smoke_parser.add_argument("--max-questions", type=int, default=5)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect":
        print(json.dumps(inspect_dataset(args.input), indent=2, ensure_ascii=False))
        return

    if args.command == "extract-qa":
        conversations, qa_rows = extract_records(
            args.input,
            conversation_ids=args.conversation_ids,
            max_questions=args.max_questions,
        )
        output_dir = Path(args.output_dir)
        conversations_path = output_dir / "normalized_conversations.jsonl"
        qa_records_path = output_dir / "normalized_qa_records.jsonl"
        summary_path = output_dir / "adapter_summary.json"
        _write_jsonl(conversations_path, conversations)
        _write_jsonl(qa_records_path, qa_rows)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(_build_summary(conversations, qa_rows), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps({
            "conversations_path": str(conversations_path),
            "qa_records_path": str(qa_records_path),
            "summary_path": str(summary_path),
        }, indent=2))
        return

    if args.command == "write-smoke-subset":
        paths = write_smoke_subset(
            args.input,
            args.output_dir,
            conversation_ids=args.conversation_ids,
            max_questions=args.max_questions,
        )
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
