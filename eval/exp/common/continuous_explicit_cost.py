"""Run a five-context explicit-text control in one cached-model process.

This is a cost-only wrapper.  It calls the existing per-context evaluator
unchanged and only replaces its model loader with a cache shared across the
five context calls, so model loading is excluded consistently with P7 and
recent-text continuous-cost measurements.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


METHOD_MODULES = {
    "bm25_top2": "scripts.eval.eventqa_bm25_retrieved_text",
    "matched16": "scripts.eval.eventqa_matched16_retrieved_text",
}


def _module_for(method: str):
    if method == "bm25_top2":
        from scripts.eval import eventqa_bm25_retrieved_text as module
    elif method == "matched16":
        from scripts.eval import eventqa_matched16_retrieved_text as module
    else:
        raise ValueError(f"unsupported explicit control: {method}")
    return module


def _single_artifact(root: Path) -> Path:
    found = sorted(root.glob("*/full_artifact.json"))
    if len(found) != 1:
        raise RuntimeError(f"expected one full artifact under {root}, got {found}")
    return found[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHOD_MODULES), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    module = _module_for(args.method)
    original_loader = module.eventqa.weaver_bank._load_model
    cached_model = None
    load_count = 0

    def cached_loader(run_args):
        nonlocal cached_model, load_count
        if cached_model is None:
            cached_model = original_loader(run_args)
            load_count += 1
        return cached_model

    module.eventqa.weaver_bank._load_model = cached_loader
    try:
        started_at = datetime.now(timezone.utc).isoformat()
        artifact_paths: list[str] = []
        artifacts: list[dict] = []
        for context_index in range(5):
            context_root = output_root / f"ctx{context_index}"
            command = [
                "--measurement-scope", "full",
                "--context-index", str(context_index),
                "--question-limit", "100",
                "--seed", str(args.seed),
                "--output-root", str(context_root),
            ]
            if module.main(command) != 0:
                return 1
            artifact_path = _single_artifact(context_root)
            artifact_paths.append(str(artifact_path))
            artifacts.append(json.loads(artifact_path.read_text(encoding="utf-8")))

        records = [row for artifact in artifacts for row in artifact["records"]]
        if len(records) != 500:
            raise RuntimeError(f"expected 500 records, got {len(records)}")
        if args.method == "bm25_top2":
            retrieval_key = "retrieval_latency_seconds"
        else:
            retrieval_key = "retrieval_and_window_latency_seconds"
        summary = {
            "schema_version": "eventqa-explicit-continuous-cost/v1",
            "run_id": args.run_id,
            "method": args.method,
            "continuous_process": True,
            "model_load_count": load_count,
            "measurement_started_at": started_at,
            "measurement_finished_at": datetime.now(timezone.utc).isoformat(),
            "scope": {"context_indices": list(range(5)), "question_count": 500},
            "cost": {
                "model_loading_excluded_from_per_question_latency": True,
                "index_construction_latency_seconds_total": sum(
                    item["cost"]["index_construction_latency_seconds"] for item in artifacts
                ),
                "retrieval_latency_seconds_total": sum(
                    row["cost"][retrieval_key] for row in records
                ),
                "generation_latency_seconds_total": sum(
                    row["cost"]["generation_latency_seconds"] for row in records
                ),
                "end_to_end_latency_seconds_total": sum(
                    item["cost"]["index_construction_latency_seconds"] for item in artifacts
                ) + sum(row["cost"]["end_to_end_latency_seconds"] for row in records),
                "end_to_end_latency_seconds_per_question": (
                    sum(item["cost"]["index_construction_latency_seconds"] for item in artifacts)
                    + sum(row["cost"]["end_to_end_latency_seconds"] for row in records)
                ) / 500,
                "incremental_peak_gpu_memory_bytes_max": max(
                    item["cost"]["incremental_peak_gpu_memory_bytes"] for item in artifacts
                ),
            },
            "artifact_paths": artifact_paths,
        }
        (output_root / "continuous_cost_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        module.eventqa.weaver_bank._load_model = original_loader
        if cached_model is not None:
            del cached_model
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
