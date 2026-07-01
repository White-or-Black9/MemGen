"""Pure offline diagnostics for EventQA Bank-off/Bank-on transitions."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def contains_chinese(text: str | None) -> bool:
    return bool(CHINESE_RE.search(text or ""))


def _normalize(text: str | None) -> str:
    return " ".join((text or "").casefold().split())


def contains_gold(prediction: str | None, gold_answers: Iterable[str]) -> bool:
    normalized_prediction = _normalize(prediction)
    return any(
        normalized_gold and normalized_gold in normalized_prediction
        for normalized_gold in (_normalize(answer) for answer in gold_answers)
    )


def _format_failure(flags) -> bool:
    return bool(flags and any(bool(value) for value in flags.values()))


def build_transition_diagnostic(row: dict) -> dict:
    off_raw = row.get("bank_off_prediction") or ""
    on_raw = row.get("bank_on_prediction") or ""
    off_parsed = row.get("bank_off_parsed_prediction")
    on_parsed = row.get("bank_on_parsed_prediction")
    gold_answers = list(row.get("gold_answers") or [])
    off_em = bool(row.get("bank_off_substring_exact_match"))
    on_em = bool(row.get("bank_on_substring_exact_match"))
    off_recall = row.get("bank_off_eventqa_recall")
    on_recall = row.get("bank_on_eventqa_recall")
    off_contains = contains_gold(off_raw, gold_answers)
    on_contains = contains_gold(on_raw, gold_answers)
    off_format = _format_failure(row.get("bank_off_format_flags"))
    on_format = _format_failure(row.get("bank_on_format_flags"))
    return {
        "context_index": row.get("context_index"),
        "context_id": row.get("context_id"),
        "question_index": row.get("query_id"),
        "query_id": row.get("query_id"),
        "question_id": row.get("question_id"),
        "qa_pair_id": row.get("qa_pair_id"),
        "gold_answers": gold_answers,
        "bank_off_raw_prediction": off_raw,
        "bank_off_parsed_prediction": off_parsed,
        "bank_off_exact_match": int(off_em),
        "bank_off_recall": off_recall,
        "bank_off_format_failure": off_format,
        "bank_off_chinese_output": contains_chinese(off_raw),
        "bank_off_output_length": len(off_raw),
        "bank_on_raw_prediction": on_raw,
        "bank_on_parsed_prediction": on_parsed,
        "bank_on_exact_match": int(on_em),
        "bank_on_recall": on_recall,
        "bank_on_format_failure": on_format,
        "bank_on_chinese_output": contains_chinese(on_raw),
        "bank_on_output_length": len(on_raw),
        "bank_off_contains_gold": off_contains,
        "bank_on_contains_gold": on_contains,
        "bank_off_raw_equals_bank_on_raw": off_raw == on_raw,
        "bank_off_parsed_equals_bank_on_parsed": off_parsed == on_parsed,
        "bank_off_correct_bank_on_wrong": off_em and not on_em,
        "bank_off_wrong_bank_on_correct": not off_em and on_em,
        "bank_off_recall_positive_bank_on_recall_negative": bool(off_recall)
        and not bool(on_recall),
        "bank_off_recall_negative_bank_on_recall_positive": not bool(off_recall)
        and bool(on_recall),
        "bank_on_recall_positive_em_negative": bool(on_recall) and not on_em,
        "bank_off_recall_positive_em_negative": bool(off_recall) and not off_em,
        "helpful_memory": not off_em and on_em,
        "harmful_memory": off_em and not on_em,
        "unchanged_correct": off_em and on_em,
        "unchanged_wrong": not off_em and not on_em,
        "recall_gain": not off_contains and on_contains,
        "recall_loss": off_contains and not on_contains,
        "format_harm": on_contains and not on_em and on_format,
    }


def _mean(values):
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def _aggregate_group(rows: list[dict]) -> dict:
    total = len(rows)
    off_em_count = sum(int(row["bank_off_exact_match"]) for row in rows)
    on_em_count = sum(int(row["bank_on_exact_match"]) for row in rows)
    off_recall = _mean(row["bank_off_recall"] for row in rows)
    on_recall = _mean(row["bank_on_recall"] for row in rows)
    return {
        "question_count": total,
        "bank_off_em_count": off_em_count,
        "bank_off_em": off_em_count / total if total else None,
        "bank_on_em_count": on_em_count,
        "bank_on_em": on_em_count / total if total else None,
        "bank_on_minus_bank_off_em": (
            (on_em_count - off_em_count) / total if total else None
        ),
        "bank_off_recall": off_recall,
        "bank_on_recall": on_recall,
        "bank_on_minus_bank_off_recall": (
            None
            if off_recall is None or on_recall is None
            else on_recall - off_recall
        ),
        "bank_off_format_failures": sum(
            int(row["bank_off_format_failure"]) for row in rows
        ),
        "bank_on_format_failures": sum(
            int(row["bank_on_format_failure"]) for row in rows
        ),
        "bank_off_chinese_outputs": sum(
            int(row["bank_off_chinese_output"]) for row in rows
        ),
        "bank_on_chinese_outputs": sum(
            int(row["bank_on_chinese_output"]) for row in rows
        ),
        "helpful_memory_count": sum(int(row["helpful_memory"]) for row in rows),
        "harmful_memory_count": sum(int(row["harmful_memory"]) for row in rows),
        "unchanged_correct_count": sum(
            int(row["unchanged_correct"]) for row in rows
        ),
        "unchanged_wrong_count": sum(int(row["unchanged_wrong"]) for row in rows),
        "recall_gain_count": sum(int(row["recall_gain"]) for row in rows),
        "recall_loss_count": sum(int(row["recall_loss"]) for row in rows),
        "format_harm_count": sum(int(row["format_harm"]) for row in rows),
    }


def aggregate_transition_rows(rows: list[dict]) -> dict:
    contexts = sorted({row.get("context_index") for row in rows})
    return {
        "global": _aggregate_group(rows),
        "per_context": {
            str(context_index): _aggregate_group(
                [row for row in rows if row.get("context_index") == context_index]
            )
            for context_index in contexts
        },
    }


def _resolve_run_root(path: str | Path) -> Path:
    root = Path(path)
    if (root / "eventqa_per_question.jsonl").is_file():
        return root
    candidates = sorted(
        candidate.parent
        for candidate in root.glob("*/eventqa_per_question.jsonl")
        if candidate.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No eventqa_per_question.jsonl found in run root or child: {root}"
        )
    return candidates[-1]


def _alignment_key(row: dict):
    return (row.get("context_index"), row.get("query_id"), row.get("qa_pair_id"))


def load_eventqa_records(paths: Iterable[str | Path]):
    records = []
    resolved = []
    for path in paths:
        root = _resolve_run_root(path)
        resolved.append(str(root.resolve()))
        with (root / "eventqa_per_question.jsonl").open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    records.sort(key=lambda row: tuple("" if value is None else str(value) for value in _alignment_key(row)))
    keys = [_alignment_key(row) for row in records]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate EventQA alignment keys: {duplicates[:5]}")
    return records, resolved


IDENTITY_FIELDS = (
    "context_id",
    "query_id",
    "question_id",
    "qa_pair_id",
    "question",
    "gold_answers",
    "bank_off_rendered_query_prompt",
    "bank_on_rendered_query_prompt",
)

STABILITY_FIELDS = {
    "raw_prediction": "prediction",
    "parsed_prediction": "parsed_prediction",
    "em": "substring_exact_match",
    "recall": "eventqa_recall",
    "format_failure": "format_flags",
}


def _missing_fields(rows: list[dict], fields: Iterable[str]) -> list[str]:
    return sorted({field for field in fields if any(field not in row for row in rows)})


def _stability(left: list[dict], right: list[dict], prefix: str) -> dict:
    result = {}
    for label, suffix in STABILITY_FIELDS.items():
        field = f"{prefix}_{suffix}"
        available = sum(field in a and field in b for a, b in zip(left, right))
        equal = sum(
            field in a
            and field in b
            and (
                _format_failure(a[field]) == _format_failure(b[field])
                if suffix == "format_flags"
                else a[field] == b[field]
            )
            for a, b in zip(left, right)
        )
        if label in {"raw_prediction", "parsed_prediction"}:
            result[f"{label}_equal_count"] = equal
        else:
            result[f"{label}_changed_count"] = available - equal
        result[f"{label}_available_count"] = available
    length_deltas = [
        len((b.get(f"{prefix}_prediction") or ""))
        - len((a.get(f"{prefix}_prediction") or ""))
        for a, b in zip(left, right)
        if f"{prefix}_prediction" in a and f"{prefix}_prediction" in b
    ]
    result["output_length_changed_count"] = sum(delta != 0 for delta in length_deltas)
    result["output_length_delta_min"] = min(length_deltas, default=None)
    result["output_length_delta_max"] = max(length_deltas, default=None)
    result["output_length_delta_mean"] = _mean(length_deltas)
    return result


def compare_eventqa_records(left: list[dict], right: list[dict]) -> dict:
    left_by_key = {_alignment_key(row): row for row in left}
    right_by_key = {_alignment_key(row): row for row in right}
    left_keys = set(left_by_key)
    right_keys = set(right_by_key)
    shared_keys = sorted(
        left_keys & right_keys,
        key=lambda key: tuple("" if value is None else str(value) for value in key),
    )
    aligned_left = [left_by_key[key] for key in shared_keys]
    aligned_right = [right_by_key[key] for key in shared_keys]
    identity = {}
    for field in IDENTITY_FIELDS:
        available = sum(field in a and field in b for a, b in zip(aligned_left, aligned_right))
        equal = sum(
            field in a and field in b and a[field] == b[field]
            for a, b in zip(aligned_left, aligned_right)
        )
        identity[field] = {
            "available_count": available,
            "equal_count": equal,
            "mismatch_count": available - equal,
        }
    identity["left_only_key_count"] = len(left_keys - right_keys)
    identity["right_only_key_count"] = len(right_keys - left_keys)
    identity["shared_key_count"] = len(shared_keys)
    identity["all_identity_fields_equal"] = (
        left_keys == right_keys
        and all(
            values["mismatch_count"] == 0
            for values in identity.values()
            if isinstance(values, dict)
        )
    )
    off = _stability(aligned_left, aligned_right, "bank_off")
    on = _stability(aligned_left, aligned_right, "bank_on")
    total = len(shared_keys)
    if (
        identity["all_identity_fields_equal"]
        and off["raw_prediction_equal_count"] == total
        and on["raw_prediction_equal_count"] < total
    ):
        conclusion = "instability_localized_to_bank_on"
    elif off["raw_prediction_equal_count"] < total:
        conclusion = "bank_off_also_unstable"
    else:
        conclusion = "no_bank_on_raw_instability_detected"
    compared_fields = list(IDENTITY_FIELDS) + [
        f"{prefix}_{suffix}"
        for prefix in ("bank_off", "bank_on")
        for suffix in STABILITY_FIELDS.values()
    ]
    per_context = {}
    for context_index in sorted(
        {row.get("context_index") for row in aligned_left},
        key=lambda value: "" if value is None else str(value),
    ):
        context_left = [
            row for row in aligned_left if row.get("context_index") == context_index
        ]
        context_right_by_key = {
            _alignment_key(row): row
            for row in aligned_right
            if row.get("context_index") == context_index
        }
        context_right = [context_right_by_key[_alignment_key(row)] for row in context_left]
        per_context[str(context_index)] = {
            "record_count": len(context_left),
            "bank_off_stability": _stability(
                context_left, context_right, "bank_off"
            ),
            "bank_on_stability": _stability(
                context_left, context_right, "bank_on"
            ),
        }
    return {
        "record_count_left": len(left),
        "record_count_right": len(right),
        "data_identity": identity,
        "bank_off_stability": off,
        "bank_on_stability": on,
        "missing_fields": {
            "left": _missing_fields(left, compared_fields),
            "right": _missing_fields(right, compared_fields),
        },
        "per_context": per_context,
        "conclusion": conclusion,
    }
