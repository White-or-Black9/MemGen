"""Run the existing rolling-summary construction/query flow in one model process."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import eventqa_text_summary_construction as construction
from scripts.eval import eventqa_text_summary_query as query
from scripts.eval import eventqa_text_summary_aggregate as aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--controlled-cost-evidence",
        help="Preflight attestation written by the serialized GPU launcher.",
    )
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    original_loader = construction.eventqa.weaver_bank._load_model
    if query.eventqa.weaver_bank._load_model is not original_loader:
        raise RuntimeError("rolling construction/query do not share the model loader")
    cached_model = None
    load_count = 0

    def cached_loader(run_args):
        nonlocal cached_model, load_count
        if cached_model is None:
            cached_model = original_loader(run_args)
            load_count += 1
        return cached_model

    construction.eventqa.weaver_bank._load_model = cached_loader
    try:
        started_at = datetime.now(timezone.utc).isoformat()
        construction_paths: list[Path] = []
        query_paths: list[Path] = []
        for context_index in range(5):
            construction_id = f"{args.run_id}-construction-ctx{context_index}"
            query_id = f"{args.run_id}-query-ctx{context_index}"
            construction_root = output_root / "construction"
            query_root = output_root / "query"
            if construction.main([
                "--context-index", str(context_index),
                "--seed", str(args.seed),
                "--output-root", str(construction_root),
                "--run-id", construction_id,
            ]) != 0:
                return 1
            construction_path = construction_root / construction_id / "construction_artifact.json"
            if query.main([
                "--measurement-scope", "full",
                "--context-index", str(context_index),
                "--question-limit", "100",
                "--seed", str(args.seed),
                "--summary-artifact", str(construction_path),
                "--output-root", str(query_root),
                "--run-id", query_id,
            ]) != 0:
                return 1
            construction_paths.append(construction_path)
            query_paths.append(query_root / query_id / "query_artifact.json")

        evidence = None
        if args.controlled_cost_evidence:
            evidence = json.loads(
                Path(args.controlled_cost_evidence).read_text(encoding="utf-8")
            )
        summary = aggregate.aggregate_pairs(
            construction_paths, query_paths, controlled_cost_evidence=evidence
        )
        summary.update({
            "continuous_process": True,
            "model_load_count": load_count,
            "measurement_started_at": started_at,
            "measurement_finished_at": datetime.now(timezone.utc).isoformat(),
            "cost": {
                **summary["cost"],
                "model_loading_excluded_from_per_question_latency": True,
            },
        })
        (output_root / "continuous_cost_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        construction.eventqa.weaver_bank._load_model = original_loader
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
