"""Measure Dense E5 top-2 cost over five EventQA contexts in one process.

The evaluator itself is unchanged.  This wrapper reuses the generator and the
CPU E5 encoder across contexts so their loading time is excluded consistently
with the existing Table-4 continuous-cost measurements, while context indexing,
query embedding, retrieval, and generation remain timed by the evaluator.
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

from eval.exp.dense_top2 import eventqa_dense_retrieved_text as dense


def _single_artifact(root: Path) -> Path:
    artifacts = sorted(root.glob("*/full_artifact.json"))
    if len(artifacts) != 1:
        raise RuntimeError(f"expected one full artifact under {root}, got {artifacts}")
    return artifacts[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    original_loader = dense.eventqa.weaver_bank._load_model
    original_encoder = dense.E5Encoder
    cached_model = None
    cached_encoder = None
    generator_load_count = 0
    encoder_load_count = 0

    def cached_loader(run_args):
        nonlocal cached_model, generator_load_count
        if cached_model is None:
            cached_model = original_loader(run_args)
            generator_load_count += 1
        return cached_model

    class _EncoderLease:
        def __init__(self, encoder) -> None:
            self._encoder = encoder

        def __getattr__(self, name):
            return getattr(self._encoder, name)

        def close(self) -> None:
            # The owner closes the shared encoder after all five contexts.
            return None

    def cached_encoder_factory(model_path: str, *, device: str, batch_size: int):
        nonlocal cached_encoder, encoder_load_count
        if cached_encoder is None:
            cached_encoder = original_encoder(model_path, device=device, batch_size=batch_size)
            encoder_load_count += 1
        elif cached_encoder.model_path != str(Path(model_path).resolve()):
            raise RuntimeError("continuous cost run cannot change the E5 encoder")
        elif str(cached_encoder.device) != device or cached_encoder.batch_size != batch_size:
            raise RuntimeError("continuous cost run cannot change E5 device or batch size")
        return _EncoderLease(cached_encoder)

    dense.eventqa.weaver_bank._load_model = cached_loader
    dense.E5Encoder = cached_encoder_factory
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
                "--embedding-device", args.embedding_device,
                "--embedding-batch-size", str(args.embedding_batch_size),
                "--seed", str(args.seed),
                "--output-root", str(context_root),
            ]
            if dense.main(command) != 0:
                return 1
            artifact_path = _single_artifact(context_root)
            artifact_paths.append(str(artifact_path))
            artifacts.append(json.loads(artifact_path.read_text(encoding="utf-8")))

        records = [record for artifact in artifacts for record in artifact["records"]]
        if len(records) != 500:
            raise RuntimeError(f"expected 500 records, got {len(records)}")
        summary = {
            "schema_version": "eventqa-dense-top2-continuous-cost/v1",
            "run_id": args.run_id,
            "method": "dense_e5_top2",
            "continuous_process": True,
            "generator_load_count": generator_load_count,
            "e5_encoder_load_count": encoder_load_count,
            "measurement_started_at": started_at,
            "measurement_finished_at": datetime.now(timezone.utc).isoformat(),
            "scope": {"context_indices": list(range(5)), "question_count": 500},
            "dense": {
                "encoder_model": artifacts[0]["dense"]["encoder_model"],
                "embedding_device": args.embedding_device,
                "embedding_batch_size": args.embedding_batch_size,
                "top_k": 2,
                "window_tokens": 500,
                "parent_score": "max_window_cosine",
            },
            "cost": {
                "model_loading_excluded_from_per_question_latency": True,
                "e5_loading_excluded_from_per_question_latency": True,
                "index_construction_latency_seconds_total": sum(
                    artifact["cost"]["index_construction_latency_seconds"] for artifact in artifacts
                ),
                "query_embedding_and_retrieval_latency_seconds_total": sum(
                    record["cost"]["retrieval_latency_seconds"] for record in records
                ),
                "generation_latency_seconds_total": sum(
                    record["cost"]["generation_latency_seconds"] for record in records
                ),
                "end_to_end_latency_seconds_total": sum(
                    artifact["cost"]["index_construction_latency_seconds"] for artifact in artifacts
                ) + sum(record["cost"]["end_to_end_latency_seconds"] for record in records),
                "end_to_end_latency_seconds_per_question": (
                    sum(artifact["cost"]["index_construction_latency_seconds"] for artifact in artifacts)
                    + sum(record["cost"]["end_to_end_latency_seconds"] for record in records)
                ) / 500,
                "incremental_peak_gpu_memory_bytes_max": max(
                    artifact["cost"]["incremental_peak_gpu_memory_bytes"] for artifact in artifacts
                ),
            },
            "artifact_paths": artifact_paths,
        }
        (output_root / "continuous_cost_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        dense.eventqa.weaver_bank._load_model = original_loader
        dense.E5Encoder = original_encoder
        if cached_encoder is not None:
            cached_encoder.close()
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
