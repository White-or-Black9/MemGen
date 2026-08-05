"""Aggregate five complete dense top-2 EventQA passes for paper reporting."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


EXPECTED_SEEDS = (42, 142, 242, 342, 442)


def _finite(values: list[float], label: str) -> None:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must be finite and nonempty")


def aggregate(pass_aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    if len(pass_aggregates) != len(EXPECTED_SEEDS):
        raise ValueError("exactly five complete pass aggregates are required")
    metrics: dict[str, list[float]] = {
        "substring_exact_match": [],
        "eventqa_recall": [],
        "format_failure_count": [],
    }
    dense_reference = None
    for item in pass_aggregates:
        scope = item.get("scope", {})
        if scope.get("context_indices") != list(range(5)) or scope.get("question_count") != 500:
            raise ValueError("each pass must cover contexts 0--4 and exactly 500 questions")
        dense = item.get("dense", {})
        reference = {
            key: dense.get(key)
            for key in ("encoder_model", "encoder_config_sha256", "top_k", "window_tokens", "parent_score")
        }
        if dense_reference is None:
            dense_reference = reference
        elif reference != dense_reference:
            raise ValueError("dense retrieval configuration drift across repeats")
        effectiveness = item.get("effectiveness", {})
        for key in metrics:
            metrics[key].append(float(effectiveness[key]))
    for key, values in metrics.items():
        _finite(values, key)
    return {
        "schema_version": "eventqa-dense-top2-repeat-aggregate/v1",
        "scope": {
            "subtask": "eventqa_65536",
            "base_seeds": list(EXPECTED_SEEDS),
            "complete_passes": 5,
            "context_indices": list(range(5)),
            "questions_per_pass": 500,
        },
        "baseline_contract": {
            "runtime_bank_mode": "off",
            "persistent_latent_bank_enabled": False,
            "source_context_scope": "current_context_only",
            "scoring": "unchanged_local_eventqa_official_path",
        },
        "dense": dense_reference,
        "effectiveness": {
            key: {
                "values": values,
                "mean": statistics.fmean(values),
                "population_std": statistics.pstdev(values),
            }
            for key, values in metrics.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass-aggregate", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    result = aggregate([json.loads(Path(path).read_text(encoding="utf-8")) for path in args.pass_aggregate])
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
