#!/usr/bin/env python3
"""Deterministic LongBench v2 dataset and chunking contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


DATASET_ID = "zai-org/LongBench-v2"
DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
EXPECTED_FIELDS = {
    "_id", "domain", "sub_domain", "difficulty", "length", "question",
    "choice_A", "choice_B", "choice_C", "choice_D", "answer", "context",
}
CHOICES = ("A", "B", "C", "D")
PROMPT_TEMPLATE = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""


class LongBenchV2ContractError(ValueError):
    """Raised when dataset or manifest provenance is invalid."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise LongBenchV2ContractError("dataset top level must be a list")
    validate_dataset_rows(rows)
    return rows


def validate_dataset_rows(rows: Sequence[dict[str, Any]]) -> None:
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != EXPECTED_FIELDS:
            raise LongBenchV2ContractError(f"row {index} fields do not match the official schema")
        for field in EXPECTED_FIELDS:
            value = row[field]
            if value is None or (isinstance(value, str) and not value.strip()):
                raise LongBenchV2ContractError(f"row {index} has empty field {field}")
        item_id = row["_id"]
        if item_id in seen_ids:
            raise LongBenchV2ContractError(f"duplicate dataset id: {item_id}")
        seen_ids.add(item_id)
        if row["answer"] not in CHOICES:
            raise LongBenchV2ContractError(f"row {index} has invalid answer {row['answer']!r}")


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("dataset") != DATASET_ID:
        raise LongBenchV2ContractError("manifest dataset identity mismatch")
    if manifest.get("revision") != DATASET_REVISION:
        raise LongBenchV2ContractError("manifest revision mismatch")
    items = manifest.get("items")
    if not isinstance(items, list) or manifest.get("count") != len(items):
        raise LongBenchV2ContractError("manifest count does not match items")
    item_ids = [item.get("_id") for item in items]
    if any(not item_id for item_id in item_ids) or len(set(item_ids)) != len(item_ids):
        raise LongBenchV2ContractError("manifest IDs must be non-empty and unique")
    token_cap = manifest.get("rendered_chat_token_cap")
    if not isinstance(token_cap, int) or token_cap <= 0:
        raise LongBenchV2ContractError("manifest rendered token cap is invalid")
    if any(item.get("rendered_chat_token_count", token_cap + 1) > token_cap for item in items):
        raise LongBenchV2ContractError("manifest contains an item above its rendered token cap")
    return manifest


def select_manifest_rows(
    dataset_rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    rows_by_id = {row["_id"]: row for row in dataset_rows}
    selected = []
    for manifest_item in manifest["items"]:
        item_id = manifest_item["_id"]
        if item_id not in rows_by_id:
            raise LongBenchV2ContractError(f"manifest ID missing from dataset: {item_id}")
        source = rows_by_id[item_id]
        for field in ("domain", "sub_domain", "difficulty", "length", "answer"):
            if source[field] != manifest_item[field]:
                raise LongBenchV2ContractError(f"manifest metadata mismatch for {item_id}: {field}")
        selected.append(normalize_item(source, manifest_item))
    return selected


def normalize_item(source: dict[str, Any], manifest_item: dict[str, Any]) -> dict[str, Any]:
    choices = {choice: source[f"choice_{choice}"] for choice in CHOICES}
    return {
        "item_id": source["_id"],
        "domain": source["domain"],
        "sub_domain": source["sub_domain"],
        "difficulty": source["difficulty"],
        "length": source["length"],
        "question": source["question"],
        "choices": choices,
        "gold_choice": source["answer"],
        "context": source["context"],
        "context_sha256": sha256_text(source["context"]),
        "context_word_count": manifest_item["context_word_count"],
        "context_token_count": manifest_item["context_token_count"],
        "rendered_chat_token_count": manifest_item["rendered_chat_token_count"],
        "capacity_class": manifest_item["capacity_class"],
    }


def render_prompt(item: dict[str, Any], *, include_context: bool = True) -> str:
    choices = item["choices"]
    return (PROMPT_TEMPLATE
            .replace("$DOC$", item["context"] if include_context else "")
            .replace("$Q$", item["question"])
            .replace("$C_A$", choices["A"])
            .replace("$C_B$", choices["B"])
            .replace("$C_C$", choices["C"])
            .replace("$C_D$", choices["D"]))


def render_memory_query_prompt(item: dict[str, Any]) -> str:
    choices = item["choices"]
    return f"""Based on the context you memorized, answer the question below.

What is the correct answer to this question: {item['question']}
Choices:
(A) {choices['A']}
(B) {choices['B']}
(C) {choices['C']}
(D) {choices['D']}

Format your response as follows: "The correct answer is (insert answer here)"."""


def render_memorization_prompt(chunk: str, *, chunk_index: int, chunk_count: int) -> str:
    return f"""Read and memorize the following context chunk for a later question.
This is chunk {chunk_index + 1} of {chunk_count}. Do not answer a question yet.

<text>
{chunk}
</text>"""


def _units_with_offsets(text: str) -> list[tuple[int, int]]:
    pattern = re.compile(r".*?(?:\n+|(?<=[.!?])\s+|$)", re.DOTALL)
    return [(match.start(), match.end()) for match in pattern.finditer(text) if match.end() > match.start()]


def _max_prefix_end(
    text: str,
    start: int,
    end: int,
    *,
    token_count: Callable[[str], int],
    token_budget: int,
) -> int:
    low, high = start + 1, end
    best = start
    while low <= high:
        middle = (low + high) // 2
        if token_count(text[start:middle]) <= token_budget:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    if best == start:
        raise LongBenchV2ContractError("token budget cannot fit one source character")
    whitespace = max(text.rfind(" ", start, best), text.rfind("\n", start, best), text.rfind("\t", start, best))
    return whitespace + 1 if whitespace >= start else best


def chunk_text(
    text: str,
    *,
    token_count: Callable[[str], int],
    token_budget: int,
) -> list[dict[str, Any]]:
    if token_budget <= 0:
        raise LongBenchV2ContractError("token budget must be positive")
    if not text:
        return []
    chunks: list[tuple[int, int]] = []
    current_start = 0
    current_end = 0

    def flush() -> None:
        nonlocal current_start, current_end
        if current_end > current_start:
            chunks.append((current_start, current_end))
        current_start = current_end

    for unit_start, unit_end in _units_with_offsets(text):
        if current_end == current_start:
            current_start = unit_start
        if token_count(text[current_start:unit_end]) <= token_budget:
            current_end = unit_end
            continue
        flush()
        cursor = unit_start
        while cursor < unit_end:
            if token_count(text[cursor:unit_end]) <= token_budget:
                current_start, current_end = cursor, unit_end
                break
            split_end = _max_prefix_end(
                text, cursor, unit_end, token_count=token_count, token_budget=token_budget,
            )
            chunks.append((cursor, split_end))
            cursor = split_end
            current_start = current_end = cursor
    flush()

    records = []
    for index, (start, end) in enumerate(chunks):
        chunk = text[start:end]
        count = token_count(chunk)
        if count > token_budget:
            raise LongBenchV2ContractError("chunk exceeds token budget")
        records.append({
            "chunk_id": f"chunk-{index:05d}",
            "chunk_index": index,
            "start_char": start,
            "end_char": end,
            "text": chunk,
            "text_sha256": sha256_text(chunk),
            "token_count": count,
        })
    if "".join(record["text"] for record in records) != text:
        raise LongBenchV2ContractError("chunk reconstruction mismatch")
    return records


def summarize_items(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(items),
        "domain_counts": dict(sorted(Counter(item["domain"] for item in items).items())),
        "difficulty_counts": dict(sorted(Counter(item["difficulty"] for item in items).items())),
        "length_counts": dict(sorted(Counter(item["length"] for item in items).items())),
        "capacity_counts": dict(sorted(Counter(item["capacity_class"] for item in items).items())),
        "item_ids": [item["item_id"] for item in items],
    }


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LongBench v2 deterministic adapter")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_dataset(args.dataset)
    manifest = load_manifest(args.manifest)
    items = select_manifest_rows(rows, manifest)
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "normalized_items.jsonl", items)
    (output_dir / "adapter_summary.json").write_text(
        json.dumps(summarize_items(items), indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summarize_items(items), indent=2))


if __name__ == "__main__":
    main()
