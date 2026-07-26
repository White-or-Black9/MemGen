"""EventQA dense top-2 retrieved-text baseline with frozen local E5 embeddings.

The baseline changes only the retrieval ranker relative to BM25: it ranks
E5-tokenized windows by exact cosine similarity, assigns each 4096-token
EventQA parent chunk its best-window score, and injects the two selected full
parent chunks into the unchanged bank-off EventQA query path.  It never reads
external corpora, answers, candidates, or the P7 latent bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.exp.bm25_top2.eventqa_bm25_retrieved_text import (
    _cuda_sync,
    _rendered_prompt,
    _sha256,
    build_retrieved_query_prompt,
    expected_question_indices,
)
from scripts.eval import mab6b_weaver_space_bank_eventqa_65536_n5 as eventqa


SCHEMA_VERSION = "eventqa-dense-top2/v1"
DEFAULT_OUTPUT_ROOT = "outputs/mab/eventqa_dense_top2_smoke"
DEFAULT_ENCODER = "/mnt/18T/baishilong/retrieval_assets/e5-base-v2"
# ``passage: `` is two E5 wordpieces and BERT adds [CLS]/[SEP].  Decoding a
# WordPiece slice can re-tokenize a few tokens longer at word boundaries, so
# reserve an additional eight tokens and reject rather than truncate.
WINDOW_TOKENS = 500
TOP_K = 2


class DenseContractError(ValueError):
    """Raised when the locked dense-retrieval comparison contract is violated."""


def _finite_nonnegative(value: Any, label: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise DenseContractError(f"{label} must be finite and nonnegative")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    masked = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None].clamp_min(1)


class E5Encoder:
    def __init__(self, model_path: str, *, device: str, batch_size: int) -> None:
        self.model_path = str(Path(model_path).resolve())
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise DenseContractError("embedding batch size must be positive")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        max_positions = int(getattr(self.model.config, "max_position_embeddings", 0))
        if max_positions != 512:
            raise DenseContractError(f"expected E5 max_position_embeddings=512, got {max_positions}")
        self.max_positions = max_positions

    def encode(self, texts: list[str], *, prefix: str) -> torch.Tensor:
        if not texts:
            raise DenseContractError("cannot encode an empty text collection")
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = [prefix + text for text in texts[start : start + self.batch_size]]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                if int(encoded["input_ids"].shape[1]) > self.max_positions:
                    raise DenseContractError("E5 input exceeded its 512-token position limit")
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded)
                vectors.append(torch.nn.functional.normalize(
                    _mean_pool(output.last_hidden_state, encoded["attention_mask"]), p=2, dim=1
                ).cpu())
        return torch.cat(vectors, dim=0)

    def encode_passage_windows(self, windows: list[dict[str, Any]]) -> torch.Tensor:
        """Encode pre-tokenized passage windows without decode/re-tokenize drift."""
        prefix_ids = self.tokenizer.encode("passage: ", add_special_tokens=False)
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(windows), self.batch_size):
                encoded_batch = []
                for window in windows[start : start + self.batch_size]:
                    token_ids = prefix_ids + list(window["e5_token_ids"])
                    encoded = self.tokenizer.prepare_for_model(
                        token_ids, add_special_tokens=True, truncation=False,
                        return_attention_mask=True,
                    )
                    if len(encoded["input_ids"]) > self.max_positions:
                        raise DenseContractError("E5 passage window exceeded its 512-token position limit")
                    encoded_batch.append(encoded)
                batch = self.tokenizer.pad(encoded_batch, padding=True, return_tensors="pt")
                batch = {key: value.to(self.device) for key, value in batch.items()}
                output = self.model(**batch)
                vectors.append(torch.nn.functional.normalize(
                    _mean_pool(output.last_hidden_state, batch["attention_mask"]), p=2, dim=1
                ).cpu())
        return torch.cat(vectors, dim=0)

    def window_texts(self, parent_chunks: list[str]) -> list[dict[str, Any]]:
        windows: list[dict[str, Any]] = []
        for parent_index, chunk in enumerate(parent_chunks):
            ids = self.tokenizer.encode(chunk, add_special_tokens=False)
            if not ids:
                raise DenseContractError(f"parent chunk {parent_index} tokenized to empty E5 text")
            for offset in range(0, len(ids), WINDOW_TOKENS):
                window_ids = ids[offset : offset + WINDOW_TOKENS]
                text = self.tokenizer.decode(
                    window_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                windows.append({
                    "parent_chunk_index": parent_index,
                    "window_index": len(windows),
                    "token_start": offset,
                    "token_end": offset + len(window_ids),
                    "text": text,
                    "text_sha256": _sha256(text),
                    "e5_token_ids": window_ids,
                })
        return windows

    def close(self) -> None:
        del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def build_dense_index(encoder: E5Encoder, chunks: list[str]) -> tuple[list[dict[str, Any]], torch.Tensor]:
    windows = encoder.window_texts(chunks)
    vectors = encoder.encode_passage_windows(windows)
    if vectors.shape[0] != len(windows):
        raise DenseContractError("window/vector count mismatch")
    return windows, vectors


def retrieve_top_k(
    *, encoder: E5Encoder, windows: list[dict[str, Any]], vectors: torch.Tensor,
    chunks: list[str], context_id: str, question: str,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    query = encoder.encode([question], prefix="query: ")[0]
    scores = torch.mv(vectors, query)
    best_by_parent: dict[int, tuple[float, dict[str, Any]]] = {}
    for window, score_tensor in zip(windows, scores):
        score = float(score_tensor)
        parent = int(window["parent_chunk_index"])
        current = best_by_parent.get(parent)
        if current is None or score > current[0] or (
            score == current[0] and int(window["token_start"]) < int(current[1]["token_start"])
        ):
            best_by_parent[parent] = (score, window)
    ranked = sorted(best_by_parent.items(), key=lambda item: (-item[1][0], item[0]))[:TOP_K]
    if len(ranked) != TOP_K:
        raise DenseContractError("dense retrieval could not select two distinct parent chunks")
    selected = []
    for parent_index, (score, window) in ranked:
        selected.append({
            "chunk_index": parent_index,
            "chunk_id": f"{context_id}-chunk-{parent_index:04d}",
            "dense_score": score,
            "best_window_index": int(window["window_index"]),
            "best_window_token_start": int(window["token_start"]),
            "best_window_token_end": int(window["token_end"]),
            "best_window_sha256": window["text_sha256"],
            "text_sha256": _sha256(chunks[parent_index]),
            "text": chunks[parent_index],
        })
    return selected, time.perf_counter() - started


def validate_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise DenseContractError("unexpected schema version")
    if artifact.get("measurement_mode") != "standalone_process":
        raise DenseContractError("measurement must use a standalone process")
    dense = artifact.get("dense", {})
    expected = {"top_k": TOP_K, "window_tokens": WINDOW_TOKENS, "parent_score": "max_window_cosine"}
    if {key: dense.get(key) for key in expected} != expected:
        raise DenseContractError("dense configuration drift")
    scope = artifact.get("scope", {})
    indices = expected_question_indices(scope.get("measurement_scope"), scope.get("context_index"), len(scope.get("question_indices") or []))
    records = artifact.get("records", [])
    if scope.get("question_indices") != indices or [row.get("query_index") for row in records] != indices:
        raise DenseContractError("question coverage mismatch")
    for field in ("index_construction_latency_seconds", "baseline_gpu_memory_bytes", "peak_gpu_memory_bytes"):
        _finite_nonnegative(artifact.get("cost", {}).get(field), field)
    for row in records:
        retrieved = row.get("retrieved_chunks", [])
        if len(retrieved) != TOP_K or len({item.get("chunk_index") for item in retrieved}) != TOP_K:
            raise DenseContractError("each query must retrieve two distinct parent chunks")
        for item in retrieved:
            _finite_nonnegative(item.get("dense_score"), "dense score")
            if len(item.get("text_sha256", "")) != 64 or len(item.get("best_window_sha256", "")) != 64:
                raise DenseContractError("retrieval provenance hash missing")
        if row.get("capacity_ok") is not True or row["rendered_prompt_token_count"] > row["context_capacity"]:
            raise DenseContractError("prompt capacity check failed")
        for field in ("query_embedding_seconds", "retrieval_latency_seconds", "generation_latency_seconds", "end_to_end_latency_seconds", "output_tokens"):
            _finite_nonnegative(row.get("cost", {}).get(field), field)


def build_parser() -> argparse.ArgumentParser:
    parser = eventqa.build_parser()
    parser.description = __doc__
    parser.add_argument("--measurement-scope", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--encoder-model", default=DEFAULT_ENCODER)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.set_defaults(output_root=DEFAULT_OUTPUT_ROOT, requested_contexts=1, context_index=0,
                        question_limit=10, eventqa_protocol="frozen_context_bank",
                        generation_max_length=40, skip_research_note=True, reseed_per_context=True)
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    question_indices = expected_question_indices(args.measurement_scope, args.context_index, args.question_limit)
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-eventqa-dense-top2-ctx{args.context_index}-q0-{question_indices[-1]}-{args.measurement_scope}"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = eventqa._load_rows(args.parquet, eventqa.SUB_DATASET)
    context = eventqa.build_context_payload(args, rows[args.context_index], args.context_index, started_at)
    encoder = E5Encoder(args.encoder_model, device=args.embedding_device, batch_size=args.embedding_batch_size)
    try:
        index_started = time.perf_counter()
        windows, vectors = build_dense_index(encoder, context["chunks"])
        index_seconds = time.perf_counter() - index_started
        model, capacity = eventqa.weaver_bank._load_model(args)
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); _cuda_sync()
            baseline_gpu_memory = int(torch.cuda.memory_allocated()); torch.cuda.reset_peak_memory_stats()
        else:
            baseline_gpu_memory = 0
        records: list[dict[str, Any]] = []
        try:
            for query_index in question_indices:
                payload = eventqa.build_question_payload(context, query_index)
                selected, retrieval_seconds = retrieve_top_k(encoder=encoder, windows=windows, vectors=vectors,
                    chunks=context["chunks"], context_id=context["context_id"], question=payload["question"])
                explicit_prompt = build_retrieved_query_prompt(selected, payload["query_prompt"])
                rendered, rendered_count = _rendered_prompt(model, explicit_prompt)
                injected = sum(len(model.tokenizer.encode(item["text"], add_special_tokens=False)) for item in selected)
                if rendered_count > capacity:
                    raise DenseContractError(f"q{query_index} prompt {rendered_count} exceeds capacity {capacity}")
                query_payload = eventqa._query_only_payload(payload)
                query_payload["query_prompt"] = explicit_prompt
                query_payload["memorization_prompts"] = [explicit_prompt]
                _cuda_sync(); generation_started = time.perf_counter()
                result = eventqa._run_eventqa_model(args, model, capacity, query_payload, "off")
                _cuda_sync(); generation_seconds = time.perf_counter() - generation_started
                if result["rendered_query_prompt"] != rendered:
                    raise DenseContractError("preflight and runtime rendered prompts differ")
                with tempfile.TemporaryDirectory() as tmpdir:
                    score = eventqa._score_prediction(args, payload, result["prediction"], tmpdir)
                turn = eventqa._query_turn(result)
                records.append({"context_index": args.context_index, "query_index": query_index,
                    "qa_pair_id": payload["qa_pair_id"], "retrieval_query": payload["question"],
                    "retrieved_chunks": [{key: value for key, value in item.items() if key != "text"} for item in selected],
                    "query_sha256": _sha256(payload["query_prompt"]), "prompt_sha256": _sha256(explicit_prompt),
                    "rendered_prompt_sha256": _sha256(rendered), "injected_token_count": injected,
                    "rendered_prompt_token_count": rendered_count, "context_capacity": capacity, "capacity_ok": True,
                    "prediction": result["prediction"],
                    "substring_exact_match": eventqa._metric_value(score, "substring_exact_match", default=0),
                    "eventqa_recall": eventqa._metric_value(score, "eventqa_recall", default=0.0),
                    "format_flags": eventqa._format_flags(result["prediction"]),
                    "cost": {"query_embedding_seconds": retrieval_seconds, "retrieval_latency_seconds": retrieval_seconds,
                             "generation_latency_seconds": generation_seconds, "end_to_end_latency_seconds": retrieval_seconds + generation_seconds,
                             "output_tokens": int(turn["output_len"])}})
            peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else baseline_gpu_memory
            config_path = Path(args.encoder_model) / "config.json"
            artifact = {"schema_version": SCHEMA_VERSION, "measurement_mode": "standalone_process", "run_id": run_id,
                "scope": {"measurement_scope": args.measurement_scope, "context_index": args.context_index, "question_indices": question_indices},
                "dense": {"encoder_model": str(Path(args.encoder_model).resolve()), "encoder_config_sha256": _file_sha256(config_path),
                          "embedding_device": args.embedding_device, "embedding_batch_size": args.embedding_batch_size,
                          "top_k": TOP_K, "window_tokens": WINDOW_TOKENS, "parent_score": "max_window_cosine",
                          "window_count": len(windows), "window_text_hashes": [row["text_sha256"] for row in windows],
                          "embedding_matrix_sha256": hashlib.sha256(vectors.numpy().tobytes()).hexdigest()},
                "cost": {"index_construction_latency_seconds": index_seconds, "baseline_gpu_memory_bytes": baseline_gpu_memory,
                         "peak_gpu_memory_bytes": peak, "incremental_peak_gpu_memory_bytes": peak - baseline_gpu_memory}, "records": records}
            validate_artifact(artifact)
            name = "smoke_artifact.json" if args.measurement_scope == "smoke" else "full_artifact.json"
            _write_json(output_dir / name, artifact)
            (output_dir / "per_question.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
            manifest = eventqa._build_manifest(run_id, args, started_at, git_status_before=eventqa._git("status", "--short", "--branch"), selected_context_indices=[args.context_index])
            manifest.update({"schema_version": SCHEMA_VERSION, "measurement_mode": "standalone_process", "measurement_scope": args.measurement_scope,
                             "exact_command": [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])], "context_id": context["context_id"],
                             "chunk_hashes": [_sha256(chunk) for chunk in context["chunks"]], "context_capacity": capacity,
                             "dense": artifact["dense"], "cost": artifact["cost"], "finished_at": datetime.now(timezone.utc).isoformat()})
            _write_json(output_dir / "manifest.json", manifest)
        finally:
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    finally:
        encoder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
