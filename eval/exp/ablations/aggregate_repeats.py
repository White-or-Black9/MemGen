"""Aggregate full EventQA ablation passes and their five process-level repeats."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


EXPECTED_CONTEXTS = list(range(5))


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("metrics must be finite and nonempty")
    return statistics.fmean(values)


def aggregate_seed(variant: str, artifact_paths: list[str]) -> dict[str, Any]:
    if len(artifact_paths) != 5:
        raise ValueError("a complete pass requires exactly five context artifacts")
    artifacts = [_load(path) for path in artifact_paths]
    contexts = [int(artifact["scope"]["context_index"]) for artifact in artifacts]
    if sorted(contexts) != EXPECTED_CONTEXTS:
        raise ValueError(f"contexts must be {EXPECTED_CONTEXTS}, found {contexts}")

    rows = [row for artifact in artifacts for row in artifact["records"]]
    identities = {(int(row["context_index"]), int(row["query_index"])) for row in rows}
    if len(rows) != 500 or len(identities) != 500:
        raise ValueError("a complete pass requires 500 unique question records")

    configs = [artifact["method_config"] for artifact in artifacts]
    if variant == "no_query_retrieval":
        if not all(config.get("query_retrieval_disabled") is True for config in configs):
            raise ValueError("no-query-retrieval artifacts do not disable query retrieval")
    elif variant == "no_retrieved_memory_conditioning":
        if not all(
            config.get("query_retrieval_disabled") is False
            and config.get("query_retrieved_memory_conditioning") is False
            for config in configs
        ):
            raise ValueError("conditioning ablation config mismatch")
    elif variant == "direct_top1":
        if not all(
            config.get("query_retrieval_disabled") is False
            and config.get("query_latent_usage") == "direct_top1"
            and int(config.get("query_direct_retrieve_top_k", -1)) == 1
            for config in configs
        ):
            raise ValueError("direct-top1 config mismatch")
    else:
        raise ValueError(f"unsupported variant: {variant}")

    return {
        "schema_version": "eventqa-ablation-seed-aggregate/v1",
        "variant": variant,
        "scope": {"context_indices": EXPECTED_CONTEXTS, "question_count": 500},
        "artifacts": artifact_paths,
        "effectiveness": {
            "substring_exact_match": _mean([float(row["substring_exact_match"]) for row in rows]),
            "eventqa_recall": _mean([float(row["eventqa_recall"]) for row in rows]),
            "format_failure_count": sum(int(any(row["format_flags"].values())) for row in rows),
        },
    }


def aggregate_repeats(seed_summaries: list[str]) -> dict[str, Any]:
    if len(seed_summaries) != 5:
        raise ValueError("repeat aggregate requires five complete-pass summaries")
    summaries = [_load(path) for path in seed_summaries]
    variants = {summary["variant"] for summary in summaries}
    if len(variants) != 1:
        raise ValueError("repeat summaries must have one variant")

    metrics = {
        key: [float(summary["effectiveness"][key]) for summary in summaries]
        for key in ("substring_exact_match", "eventqa_recall", "format_failure_count")
    }
    return {
        "schema_version": "eventqa-ablation-repeat-aggregate/v1",
        "variant": variants.pop(),
        "complete_process_passes": 5,
        "seed_summaries": seed_summaries,
        "effectiveness": {
            key: {"mean": _mean(values), "population_std": statistics.pstdev(values)}
            for key, values in metrics.items()
        },
    }


def _write(path: str, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seed = subparsers.add_parser("seed")
    seed.add_argument("--variant", required=True, choices=("no_query_retrieval", "no_retrieved_memory_conditioning", "direct_top1"))
    seed.add_argument("--artifacts", nargs=5, required=True)
    seed.add_argument("--output-json", required=True)
    repeat = subparsers.add_parser("repeats")
    repeat.add_argument("--seed-summaries", nargs=5, required=True)
    repeat.add_argument("--output-json", required=True)
    args = parser.parse_args()
    payload = (
        aggregate_seed(args.variant, args.artifacts)
        if args.command == "seed"
        else aggregate_repeats(args.seed_summaries)
    )
    _write(args.output_json, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
