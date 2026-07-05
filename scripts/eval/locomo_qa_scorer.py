#!/usr/bin/env python3
"""Deterministic offline scorer for normalized LoCoMo-QA predictions."""

from __future__ import annotations

import argparse
import json
import string
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List, Sequence


SCORER_VERSION = "locomo_qa_v1"
SURROUNDING_PUNCTUATION = string.punctuation + "“”‘’"


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    normalized = str(text).strip().lower()
    if not normalized:
        return ""
    tokens = normalized.split()
    normalized = " ".join(tokens)
    normalized = normalized.strip(SURROUNDING_PUNCTUATION)
    tokens = [token.strip(SURROUNDING_PUNCTUATION) for token in normalized.split()]
    tokens = [token for token in tokens if token]
    return " ".join(tokens)


def _token_counts(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for token in text.split():
        counts[token] = counts.get(token, 0) + 1
    return counts


def score_row(prediction_text: Any, gold_answer: Any, prediction_status: str | None) -> Dict[str, Any]:
    normalized_prediction = normalize_text(prediction_text)
    normalized_gold_answer = normalize_text(gold_answer)
    invalid_status = prediction_status in {"missing", "invalid"}
    invalid_output = int(invalid_status or not normalized_prediction)

    if invalid_output:
        return {
            "normalized_prediction": normalized_prediction,
            "normalized_gold_answer": normalized_gold_answer,
            "exact_match": 0,
            "token_f1": 0.0,
            "invalid_output": 1,
        }

    exact_match = int(normalized_prediction == normalized_gold_answer)

    prediction_counts = _token_counts(normalized_prediction)
    gold_counts = _token_counts(normalized_gold_answer)
    overlap = 0
    for token, count in prediction_counts.items():
        overlap += min(count, gold_counts.get(token, 0))

    prediction_total = sum(prediction_counts.values())
    gold_total = sum(gold_counts.values())
    if prediction_total == 0 or gold_total == 0 or overlap == 0:
        token_f1 = 0.0
    else:
        precision = overlap / prediction_total
        recall = overlap / gold_total
        token_f1 = 2 * precision * recall / (precision + recall)

    return {
        "normalized_prediction": normalized_prediction,
        "normalized_gold_answer": normalized_gold_answer,
        "exact_match": exact_match,
        "token_f1": token_f1,
        "invalid_output": 0,
    }


def aggregate_scores(rows: Sequence[Dict[str, Any]], *, method: str | None = None) -> Dict[str, Any]:
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_conversation: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        by_category[row["category_name"]].append(row)
        by_conversation[row["conversation_id"]].append(row)

    def _mean_metric(group_rows: Sequence[Dict[str, Any]], key: str) -> float:
        return fmean(float(row[key]) for row in group_rows) if group_rows else 0.0

    overall_micro = {
        "exact_match_mean": _mean_metric(rows, "exact_match"),
        "token_f1_mean": _mean_metric(rows, "token_f1"),
    }

    per_conversation = {
        conversation_id: {
            "count": len(group_rows),
            "exact_match_mean": _mean_metric(group_rows, "exact_match"),
            "token_f1_mean": _mean_metric(group_rows, "token_f1"),
        }
        for conversation_id, group_rows in sorted(by_conversation.items())
    }

    if per_conversation:
        overall_macro = {
            "exact_match_mean": fmean(metrics["exact_match_mean"] for metrics in per_conversation.values()),
            "token_f1_mean": fmean(metrics["token_f1_mean"] for metrics in per_conversation.values()),
        }
    else:
        overall_macro = {"exact_match_mean": 0.0, "token_f1_mean": 0.0}

    category_metrics = {
        category_name: {
            "count": len(group_rows),
            "exact_match_mean": _mean_metric(group_rows, "exact_match"),
            "token_f1_mean": _mean_metric(group_rows, "token_f1"),
        }
        for category_name, group_rows in sorted(by_category.items())
    }

    invalid_output_count = sum(int(row["invalid_output"]) for row in rows)
    inferred_method = method
    if inferred_method is None and rows:
        inferred_method = rows[0].get("method")

    return {
        "method": inferred_method,
        "scorer_version": SCORER_VERSION,
        "record_count": len(rows),
        "invalid_output_count": invalid_output_count,
        "overall_micro": overall_micro,
        "overall_macro_by_conversation": overall_macro,
        "by_category": category_metrics,
        "by_conversation": per_conversation,
    }


def score_prediction_records(
    qa_records: Sequence[Dict[str, Any]],
    prediction_records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    predictions_by_question = {
        row["question_id"]: row
        for row in prediction_records
    }
    scored_rows: List[Dict[str, Any]] = []

    for qa_row in qa_records:
        prediction_row = predictions_by_question.get(qa_row["question_id"])
        prediction_text = None if prediction_row is None else prediction_row.get("prediction_text")
        raw_prediction_text = None if prediction_row is None else prediction_row.get("raw_prediction_text", prediction_text)
        prediction_status = "missing" if prediction_row is None else prediction_row.get("prediction_status", "ok")
        method = None if prediction_row is None else prediction_row.get("method")

        scores = score_row(prediction_text, qa_row.get("gold_answer"), prediction_status)
        scored_rows.append({
            "question_id": qa_row["question_id"],
            "conversation_id": qa_row["conversation_id"],
            "method": method,
            "category": qa_row.get("category"),
            "category_name": qa_row.get("category_name"),
            "prediction_text": prediction_text,
            "raw_prediction_text": raw_prediction_text,
            "gold_answer": qa_row.get("gold_answer"),
            "normalized_prediction": scores["normalized_prediction"],
            "normalized_gold_answer": scores["normalized_gold_answer"],
            "exact_match": scores["exact_match"],
            "token_f1": scores["token_f1"],
            "invalid_output": scores["invalid_output"],
            "prediction_status": prediction_status,
            "scorer_version": SCORER_VERSION,
            "status": "scored",
        })

    return scored_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoCoMo QA scorer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="Score prediction_records.jsonl against normalized QA records")
    score_parser.add_argument("--qa-records", required=True, help="Path to normalized_qa_records.jsonl")
    score_parser.add_argument("--predictions", required=True, help="Path to prediction_records.jsonl")
    score_parser.add_argument("--output-dir", required=True, help="Output directory")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command != "score":
        raise ValueError(f"Unsupported command: {args.command}")

    qa_records = load_jsonl(args.qa_records)
    prediction_records = load_jsonl(args.predictions)
    scored_rows = score_prediction_records(qa_records, prediction_records)
    aggregate = aggregate_scores(scored_rows)

    output_dir = Path(args.output_dir)
    scored_path = output_dir / "scored_prediction_records.jsonl"
    aggregate_path = output_dir / "aggregate_metrics.json"
    write_jsonl(scored_path, scored_rows)
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "scored_prediction_records_path": str(scored_path),
        "aggregate_metrics_path": str(aggregate_path),
    }, indent=2))


if __name__ == "__main__":
    main()
