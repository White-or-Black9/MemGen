"""Aggregate five complete EventQA dense top-2 artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from eventqa_dense_retrieved_text import DenseContractError, validate_artifact


EXPECTED_CONTEXTS = list(range(5))


def _mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("values must be finite and nonempty")
    return statistics.fmean(values)


def aggregate(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    if len(artifacts) != 5:
        raise ValueError("exactly five context artifacts are required")
    for artifact in artifacts:
        validate_artifact(artifact)
        if artifact["scope"]["measurement_scope"] != "full":
            raise ValueError("only full artifacts may enter the paper aggregate")
    ordered = sorted(artifacts, key=lambda artifact: artifact["scope"]["context_index"])
    if [artifact["scope"]["context_index"] for artifact in ordered] != EXPECTED_CONTEXTS:
        raise ValueError("aggregate requires contexts 0--4 exactly once")
    rows = [row for artifact in ordered for row in artifact["records"]]
    if len(rows) != 500 or len({(row["context_index"], row["query_index"]) for row in rows}) != 500:
        raise ValueError("aggregate requires 500 unique EventQA questions")
    dense = ordered[0]["dense"]
    for artifact in ordered[1:]:
        candidate = artifact["dense"]
        for key in ("encoder_model", "encoder_config_sha256", "top_k", "window_tokens", "parent_score"):
            if candidate[key] != dense[key]:
                raise ValueError(f"dense configuration drift: {key}")
    index_seconds = sum(artifact["cost"]["index_construction_latency_seconds"] for artifact in ordered)
    retrieval_seconds = sum(row["cost"]["retrieval_latency_seconds"] for row in rows)
    generation_seconds = sum(row["cost"]["generation_latency_seconds"] for row in rows)
    return {
        "schema_version": "eventqa-dense-top2-aggregate/v1",
        "scope": {"subtask": "eventqa_65536", "context_indices": EXPECTED_CONTEXTS, "question_count": 500, "questions_per_context": 100},
        "dense": {key: dense[key] for key in ("encoder_model", "encoder_config_sha256", "top_k", "window_tokens", "parent_score")},
        "effectiveness": {"substring_exact_match": _mean([float(row["substring_exact_match"]) for row in rows]),
                          "eventqa_recall": _mean([float(row["eventqa_recall"]) for row in rows]),
                          "format_failure_count": sum(int(any(row["format_flags"].values())) for row in rows)},
        "cost": {"index_construction_latency_seconds": index_seconds, "retrieval_latency_seconds": retrieval_seconds,
                 "generation_latency_seconds": generation_seconds, "method_total_seconds": index_seconds + retrieval_seconds + generation_seconds,
                 "amortized_seconds_per_question": (index_seconds + retrieval_seconds + generation_seconds) / 500,
                 "incremental_peak_gpu_memory_bytes_max": max(artifact["cost"]["incremental_peak_gpu_memory_bytes"] for artifact in ordered)},
        "capacity": {"max_rendered_prompt_token_count": max(row["rendered_prompt_token_count"] for row in rows),
                     "context_capacity": rows[0]["context_capacity"], "all_capacity_ok": all(row["capacity_ok"] for row in rows)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    try:
        summary = aggregate([json.loads(Path(path).read_text(encoding="utf-8")) for path in args.artifact])
    except DenseContractError as error:
        raise ValueError(str(error)) from error
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
