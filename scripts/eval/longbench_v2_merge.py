#!/usr/bin/env python3
"""Merge LongBench v2 method shards into one contract-validated artifact."""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any, Sequence

from scripts.eval import longbench_v2_p7 as runner


class LongBenchV2MergeError(ValueError):
    """Raised when shard artifacts cannot form one aligned comparison."""


def load_artifacts(input_root: str | Path) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(Path(input_root).glob("*/artifact.json"))
    if not paths:
        raise LongBenchV2MergeError("no shard artifacts found")
    return [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]


def exact_sign_test_p_value(wins: int, losses: int) -> float | None:
    discordant = wins + losses
    if not discordant:
        return None
    lower_tail = sum(comb(discordant, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * lower_tail / (2 ** discordant))


def paired_summary(records: Sequence[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        if record["method"] not in {left, right}:
            continue
        rows.setdefault(record["item_id"], {})[record["method"]] = record
    pairs = [methods for methods in rows.values() if {left, right}.issubset(methods)]
    wins = sum(pair[left]["strict_correct"] > pair[right]["strict_correct"] for pair in pairs)
    losses = sum(pair[left]["strict_correct"] < pair[right]["strict_correct"] for pair in pairs)
    ties = len(pairs) - wins - losses
    return {
        "left_method": left,
        "right_method": right,
        "pair_count": len(pairs),
        "left_wins": wins,
        "left_losses": losses,
        "ties": ties,
        "exact_two_sided_sign_test_p_value": exact_sign_test_p_value(wins, losses),
    }


def merge(input_root: str | Path) -> dict[str, Any]:
    sources = load_artifacts(input_root)
    records: list[dict[str, Any]] = []
    constructions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path, artifact in sources:
        for record in artifact.get("records", []):
            key = (record.get("item_id"), record.get("method"))
            if None in key or key in seen:
                raise LongBenchV2MergeError(f"duplicate or invalid record in {path}: {key}")
            seen.add(key)
            records.append(record)
        constructions.extend(artifact.get("construction_runs", []))
    summary = runner.aggregate(records, constructions, validate_contract=True)
    return {
        "schema_version": "longbench-v2-merged-comparison/v1",
        "source_artifacts": [str(path) for path, _ in sources],
        "record_count": len(records),
        "construction_count": len(constructions),
        **summary,
        "paired_comparisons": {
            "p7_vs_no_query": paired_summary(records, "p7", "p7_no_query_retrieval"),
            "p7_vs_disabled_window_fit": paired_summary(records, "p7", "disabled_window_fit"),
        },
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    artifact = merge(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    runner.write_jsonl(output_dir / "records.jsonl", artifact["records"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
