"""Run all five frozen-P7 EventQA contexts in one model process for cost measurement."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import eventqa_method_separable_cost as p7_cost


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    original_loader = p7_cost.eventqa.weaver_bank._load_model
    cached_model = None
    load_count = 0

    def cached_loader(run_args):
        nonlocal cached_model, load_count
        if cached_model is None:
            cached_model = original_loader(run_args)
            load_count += 1
        return cached_model

    p7_cost.eventqa.weaver_bank._load_model = cached_loader
    try:
        started_at = datetime.now(timezone.utc).isoformat()
        summaries = []
        artifact_paths = []
        for context_index in range(5):
            context_root = output_root / f"ctx{context_index}"
            command = [
                "--method", "p7",
                "--measurement-scope", "full",
                "--measurement-mode", "continuous_process_context_segment",
                "--context-index", str(context_index),
                "--question-limit", "100",
                "--seed", str(args.seed),
                "--output-root", str(context_root),
            ]
            if p7_cost.main(command) != 0:
                return 1
            paths = list(context_root.glob("p7/*/cost_summary.json"))
            if len(paths) != 1:
                raise RuntimeError(
                    f"expected exactly one P7 cost summary for context {context_index}, got {paths}"
                )
            artifact_paths.append(str(paths[0]))
            summaries.append(json.loads(paths[0].read_text(encoding="utf-8")))

        query_latencies = [
            latency
            for summary in summaries
            for latency in summary["cost"]["query_latency_seconds"]["values"]
        ]
        total_questions = len(query_latencies)
        summary = {
            "schema_version": "eventqa-p7-continuous-cost/v1",
            "run_id": args.run_id,
            "continuous_process": True,
            "model_load_count": load_count,
            "measurement_started_at": started_at,
            "measurement_finished_at": datetime.now(timezone.utc).isoformat(),
            "scope": {"context_indices": list(range(5)), "question_count": total_questions},
            "method": "p7",
            "cost": {
                "model_loading_excluded_from_per_question_latency": True,
                "construction_latency_seconds_total": sum(
                    item["cost"]["construction_latency_seconds"] for item in summaries
                ),
                "construction_latency_seconds_per_context": sum(
                    item["cost"]["construction_latency_seconds"] for item in summaries
                ) / len(summaries),
                "query_latency_seconds_total": sum(query_latencies),
                "query_latency_seconds_per_question": sum(query_latencies) / total_questions,
                "end_to_end_latency_seconds_total": sum(
                    item["cost"]["end_to_end_latency_seconds"] for item in summaries
                ),
                "end_to_end_latency_seconds_per_question": sum(
                    item["cost"]["end_to_end_latency_seconds"] for item in summaries
                ) / total_questions,
                "incremental_peak_gpu_memory_bytes_max": max(
                    item["cost"]["incremental_peak_gpu_memory_bytes"] for item in summaries
                ),
            },
            "invariants": {
                "all_query_writes_zero": all(
                    item["invariants"]["all_query_writes_zero"] for item in summaries
                ),
                "all_bank_snapshots_unchanged": all(
                    item["invariants"]["all_bank_snapshots_unchanged"] for item in summaries
                ),
            },
            "artifact_paths": artifact_paths,
        }
        (output_root / "continuous_cost_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        p7_cost.eventqa.weaver_bank._load_model = original_loader
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
