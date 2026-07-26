"""Run all five recent-text EventQA contexts in one model process for cost measurement."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import eventqa_memgen_recent_window as recent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recent-history-token-budget", type=int, default=32768)
    parser.add_argument("--generation-reserve-tokens", type=int, default=40)
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    original_loader = recent.eventqa.weaver_bank._load_model
    cached_model = None
    load_count = 0

    def cached_loader(run_args):
        nonlocal cached_model, load_count
        if cached_model is None:
            cached_model = original_loader(run_args)
            load_count += 1
        return cached_model

    recent.eventqa.weaver_bank._load_model = cached_loader
    try:
        measured_started_at = None
        artifacts = []
        for context_index in range(5):
            context_run_id = f"{args.run_id}-ctx{context_index}"
            context_root = output_root / f"ctx{context_index}"
            command = [
                "--measurement-scope", "full",
                "--context-index", str(context_index),
                "--question-limit", "100",
                "--recent-history-token-budget", str(args.recent_history_token_budget),
                "--generation-reserve-tokens", str(args.generation_reserve_tokens),
                "--seed", str(args.seed),
                "--output-root", str(context_root),
                "--run-id", context_run_id,
            ]
            if recent.eventqa.weaver_bank._load_model is cached_loader and measured_started_at is None:
                # The first call loads the model inside recent.main; the per-question
                # artifact timings themselves exclude loading, as in the existing runner.
                measured_started_at = datetime.now(timezone.utc).isoformat()
            if recent.main(command) != 0:
                return 1
            artifact_path = context_root / context_run_id / "full_artifact.json"
            artifacts.append(json.loads(artifact_path.read_text(encoding="utf-8")))

        records = [record for artifact in artifacts for record in artifact["records"]]
        summary = {
            "schema_version": "eventqa-memgen-recent-window-continuous-cost/v1",
            "run_id": args.run_id,
            "continuous_process": True,
            "model_load_count": load_count,
            "measurement_started_at": measured_started_at,
            "measurement_finished_at": datetime.now(timezone.utc).isoformat(),
            "scope": {"context_indices": list(range(5)), "question_count": len(records)},
            "method": artifacts[0]["method"],
            "cost": {
                "model_loading_excluded_from_per_question_latency": True,
                "query_latency_seconds_total": sum(record["cost"]["end_to_end_latency_seconds"] for record in records),
                "query_latency_seconds_per_question": sum(record["cost"]["end_to_end_latency_seconds"] for record in records) / len(records),
                "incremental_peak_gpu_memory_bytes_max": max(artifact["cost"]["incremental_peak_gpu_memory_bytes"] for artifact in artifacts),
            },
            "artifact_paths": [str(output_root / f"ctx{i}" / f"{args.run_id}-ctx{i}" / "full_artifact.json") for i in range(5)],
        }
        (output_root / "continuous_cost_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    finally:
        recent.eventqa.weaver_bank._load_model = original_loader
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
