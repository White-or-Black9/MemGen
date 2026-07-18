#!/usr/bin/env python3
"""Model-free strict and diagnostic LongBench v2 option scoring."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence


SCORER_VERSION = "longbench_v2_option_v1"
STRICT_PATTERN = re.compile(r'^\s*The correct answer is\s*\(([ABCD])\)\.?\s*$', re.IGNORECASE)
OPTION_PATTERN = re.compile(r'(?<![A-Za-z0-9])\(?([ABCD])\)?(?![A-Za-z0-9])', re.IGNORECASE)


def extract_strict_choice(text: Any) -> str | None:
    if text is None:
        return None
    match = STRICT_PATTERN.fullmatch(str(text))
    return match.group(1).upper() if match else None


def extract_relaxed_choice(text: Any, choices: dict[str, str]) -> str | None:
    strict = extract_strict_choice(text)
    if strict:
        return strict
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return None
    matched_letters = {match.group(1).upper() for match in OPTION_PATTERN.finditer(normalized)}
    if len(matched_letters) == 1:
        return next(iter(matched_letters))
    matched_text = {
        label for label, choice_text in choices.items()
        if normalized == " ".join(choice_text.strip().lower().split())
    }
    return next(iter(matched_text)) if len(matched_text) == 1 else None


def score_prediction(item: dict[str, Any], prediction_text: Any) -> dict[str, Any]:
    strict_choice = extract_strict_choice(prediction_text)
    relaxed_choice = extract_relaxed_choice(prediction_text, item["choices"])
    gold = item["gold_choice"]
    return {
        "item_id": item["item_id"],
        "domain": item["domain"],
        "sub_domain": item["sub_domain"],
        "difficulty": item["difficulty"],
        "length": item["length"],
        "capacity_class": item["capacity_class"],
        "gold_choice": gold,
        "prediction_text": prediction_text,
        "strict_choice": strict_choice,
        "relaxed_choice": relaxed_choice,
        "strict_correct": int(strict_choice == gold),
        "relaxed_correct": int(relaxed_choice == gold),
        "invalid_output": int(strict_choice is None),
        "scorer_version": SCORER_VERSION,
    }


def aggregate_scores(rows: Sequence[dict[str, Any]], *, method: str | None = None) -> dict[str, Any]:
    def metrics(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(group),
            "strict_accuracy": fmean(row["strict_correct"] for row in group) if group else 0.0,
            "relaxed_accuracy_diagnostic": fmean(row["relaxed_correct"] for row in group) if group else 0.0,
            "invalid_output_count": sum(row["invalid_output"] for row in group),
        }

    output = {
        "method": method,
        "scorer_version": SCORER_VERSION,
        "overall": metrics(rows),
    }
    for field in ("domain", "sub_domain", "difficulty", "length", "capacity_class"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        output[f"by_{field}"] = {key: metrics(group) for key, group in sorted(groups.items())}
    return output


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score LongBench v2 predictions")
    parser.add_argument("--items", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method")
    args = parser.parse_args()
    items = load_jsonl(args.items)
    predictions = {row["item_id"]: row for row in load_jsonl(args.predictions)}
    scored = [score_prediction(item, predictions.get(item["item_id"], {}).get("prediction_text")) for item in items]
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "scored_predictions.jsonl", scored)
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate_scores(scored, method=args.method), indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
