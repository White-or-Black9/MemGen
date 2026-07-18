"""Aggregation helpers for future RULER-QA2 evaluation artifacts."""

from __future__ import annotations


def aggregate_records(records: list[dict]) -> dict:
    total = len(records)
    correct = sum(1 for item in records if item["correct"])
    return {
        "schema_version": "ruler-qa2-aggregate/v1",
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "memory_write_count": sum(
            int(item.get("memory_write_count", 0)) for item in records
        ),
        "retrieval_count": sum(int(item.get("retrieval_count", 0)) for item in records),
    }
