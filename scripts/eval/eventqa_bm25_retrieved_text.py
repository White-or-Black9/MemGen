"""Run a standalone EventQA BM25 top-2 explicit-text evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import mab6b_weaver_space_bank_eventqa_65536_n5 as eventqa


SCHEMA_VERSION = "eventqa-bm25-top2/v1"
EXPECTED_QUESTIONS = list(range(10))
DEFAULT_OUTPUT_ROOT = "outputs/mab/eventqa_bm25_top2_smoke"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


class BM25ContractError(ValueError):
    """Raised when retrieval or smoke output violates the locked contract."""


def expected_question_indices(
    measurement_scope: str, context_index: int, question_limit: int
) -> list[int]:
    if measurement_scope == "smoke":
        if context_index != 0 or question_limit != 10:
            raise BM25ContractError("smoke scope must be context 0 q0-9")
        return list(range(10))
    if measurement_scope == "full":
        if context_index not in range(5) or question_limit != 100:
            raise BM25ContractError("full scope must be context 0-4 q0-99")
        return list(range(100))
    raise BM25ContractError(f"unsupported measurement scope: {measurement_scope}")


def tokenize_bm25(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise BM25ContractError("BM25 requires at least one document")
        self.k1 = float(k1)
        self.b = float(b)
        self.term_frequencies = [Counter(tokenize_bm25(doc)) for doc in documents]
        self.document_lengths = [sum(counts.values()) for counts in self.term_frequencies]
        self.average_document_length = statistics.fmean(self.document_lengths)
        self.document_frequency: Counter[str] = Counter()
        for counts in self.term_frequencies:
            self.document_frequency.update(counts.keys())

    def score(self, query: str, document_index: int) -> float:
        counts = self.term_frequencies[document_index]
        document_length = self.document_lengths[document_index]
        score = 0.0
        for term in set(tokenize_bm25(query)):
            frequency = counts.get(term, 0)
            if frequency == 0:
                continue
            document_frequency = self.document_frequency[term]
            inverse_document_frequency = math.log(
                1.0
                + (len(self.term_frequencies) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            normalization = frequency + self.k1 * (
                1.0
                - self.b
                + self.b * document_length / self.average_document_length
            )
            score += inverse_document_frequency * frequency * (self.k1 + 1.0) / normalization
        return score

    def rank(self, query: str, *, top_k: int = 2) -> list[tuple[int, float]]:
        if top_k <= 0 or top_k > len(self.term_frequencies):
            raise BM25ContractError("top_k must select available documents")
        scored = [(index, self.score(query, index)) for index in range(len(self.term_frequencies))]
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def retrieve_top_k(
    chunks: list[str],
    query: str,
    *,
    context_id: str,
    top_k: int = 2,
    k1: float = 1.5,
    b: float = 0.75,
    index: BM25Index | None = None,
) -> list[dict[str, Any]]:
    bm25 = index or BM25Index(chunks, k1=k1, b=b)
    return [
        {
            "chunk_index": chunk_index,
            "chunk_id": f"{context_id}-chunk-{chunk_index:04d}",
            "bm25_score": score,
            "text_sha256": _sha256(chunks[chunk_index]),
            "text": chunks[chunk_index],
        }
        for chunk_index, score in bm25.rank(query, top_k=top_k)
    ]


def build_retrieved_query_prompt(
    retrieved_chunks: list[dict[str, Any]], official_query_prompt: str
) -> str:
    passages = []
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        passages.append(
            f"[Retrieved passage {rank}; source={chunk['chunk_id']}]\n{chunk['text']}"
        )
    return (
        "Use the retrieved source passages below to answer the question.\n\n"
        + "\n\n".join(passages)
        + "\n\n"
        + official_query_prompt
    )


def _finite_nonnegative(value: Any, label: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise BM25ContractError(f"{label} must be finite and nonnegative")


def validate_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise BM25ContractError("unexpected schema version")
    if artifact.get("measurement_mode") != "standalone_process":
        raise BM25ContractError("measurement must use a standalone process")
    scope = artifact.get("scope", {})
    records = artifact.get("records", [])
    measurement_scope = scope.get("measurement_scope", "smoke")
    context_index = scope.get("context_index")
    question_indices = scope.get("question_indices")
    expected = expected_question_indices(
        measurement_scope, context_index, len(question_indices or [])
    )
    indices = [record.get("query_index") for record in records]
    scope_label = "q0-9" if measurement_scope == "smoke" else "q0-99"
    if question_indices != expected or len(records) != len(expected) or indices != expected:
        raise BM25ContractError(
            f"{measurement_scope} records must cover context {context_index} {scope_label} exactly"
        )
    config = artifact.get("bm25", {})
    if config != {"k1": 1.5, "b": 0.75, "top_k": 2}:
        raise BM25ContractError("BM25 configuration drift")
    cost = artifact.get("cost", {})
    for field in (
        "index_construction_latency_seconds",
        "baseline_gpu_memory_bytes",
        "peak_gpu_memory_bytes",
    ):
        _finite_nonnegative(cost.get(field), field)
    if cost["peak_gpu_memory_bytes"] < cost["baseline_gpu_memory_bytes"]:
        raise BM25ContractError("peak GPU memory cannot be below baseline")
    for record in records:
        if record.get("context_index") != context_index:
            raise BM25ContractError("record context does not match artifact context")
        retrieved = record.get("retrieved_chunks", [])
        if len(retrieved) != 2:
            raise BM25ContractError("each question requires exactly two retrieved chunks")
        for item in retrieved:
            if not isinstance(item.get("chunk_index"), int) or not item.get("chunk_id"):
                raise BM25ContractError("retrieved chunk provenance is incomplete")
            _finite_nonnegative(item.get("bm25_score"), "BM25 score")
            if len(item.get("text_sha256", "")) != 64:
                raise BM25ContractError("retrieved chunk hash is invalid")
        if len(record.get("query_sha256", "")) != 64 or len(record.get("prompt_sha256", "")) != 64:
            raise BM25ContractError("query or prompt hash is invalid")
        if record.get("capacity_ok") is not True:
            raise BM25ContractError("prompt capacity check failed")
        if record.get("rendered_prompt_token_count", 0) > record.get("context_capacity", -1):
            raise BM25ContractError("rendered prompt exceeds capacity")
        for field in ("injected_token_count", "rendered_prompt_token_count", "context_capacity"):
            _finite_nonnegative(record.get(field), field)
        method_cost = record.get("cost", {})
        for field in (
            "retrieval_latency_seconds",
            "generation_latency_seconds",
            "end_to_end_latency_seconds",
            "output_tokens",
        ):
            _finite_nonnegative(method_cost.get(field), field)


def validate_smoke_artifact(artifact: dict[str, Any]) -> None:
    scope = artifact.get("scope", {})
    if scope.get("measurement_scope", "smoke") != "smoke":
        raise BM25ContractError("validate_smoke_artifact requires smoke scope")
    validate_artifact(artifact)


def build_parser():
    parser = eventqa.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--measurement-scope", choices=("smoke", "full"), default="smoke"
    )
    parser.set_defaults(
        output_root=DEFAULT_OUTPUT_ROOT,
        requested_contexts=1,
        context_index=0,
        question_limit=10,
        eventqa_protocol="frozen_context_bank",
        generation_max_length=40,
        skip_research_note=True,
        reseed_per_context=True,
    )
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _cuda_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _rendered_prompt(model, prompt: str) -> tuple[str, int]:
    messages = [
        {"role": "system", "content": eventqa.base.DEFAULT_SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]
    rendered = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokens = model.tokenizer.encode(rendered, add_special_tokens=False)
    return rendered, len(tokens)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    question_indices = expected_question_indices(
        args.measurement_scope, args.context_index, args.question_limit
    )
    started_at = datetime.now(timezone.utc).isoformat()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"{timestamp}-eventqa-bm25-top2-ctx{args.context_index}-"
        f"q0-{question_indices[-1]}-{args.measurement_scope}"
    )
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]

    rows = eventqa._load_rows(args.parquet, eventqa.SUB_DATASET)
    context_payload = eventqa.build_context_payload(
        args, rows[args.context_index], args.context_index, started_at
    )
    chunks = context_payload["chunks"]
    index_start = time.perf_counter()
    index = BM25Index(chunks)
    index_latency = time.perf_counter() - index_start

    manifest = eventqa._build_manifest(
        run_id,
        args,
        started_at,
        git_status_before=eventqa._git("status", "--short", "--branch"),
        selected_context_indices=[args.context_index],
    )
    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "measurement_mode": "standalone_process",
            "measurement_scope": args.measurement_scope,
            "exact_command": command,
            "bm25": {"k1": 1.5, "b": 0.75, "top_k": 2},
            "context_id": context_payload["context_id"],
            "chunk_count": len(chunks),
            "chunk_hashes": [_sha256(chunk) for chunk in chunks],
        }
    )
    _write_json(output_dir / "manifest.json", manifest)

    model, capacity = eventqa.weaver_bank._load_model(args)
    import torch

    manifest["context_capacity"] = capacity
    manifest["gpu"] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _cuda_sync()
        baseline_gpu_memory = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
    else:
        baseline_gpu_memory = 0

    records: list[dict[str, Any]] = []
    try:
        for query_index in question_indices:
            payload = eventqa.build_question_payload(context_payload, query_index)
            retrieval_start = time.perf_counter()
            selected = retrieve_top_k(
                chunks,
                payload["question"],
                context_id=context_payload["context_id"],
                index=index,
            )
            retrieval_latency = time.perf_counter() - retrieval_start
            explicit_prompt = build_retrieved_query_prompt(selected, payload["query_prompt"])
            rendered_prompt, rendered_token_count = _rendered_prompt(model, explicit_prompt)
            injected_token_count = sum(
                len(model.tokenizer.encode(item["text"], add_special_tokens=False))
                for item in selected
            )
            capacity_ok = rendered_token_count <= capacity
            if not capacity_ok:
                raise BM25ContractError(
                    f"q{query_index} rendered prompt {rendered_token_count} exceeds capacity {capacity}"
                )
            query_payload = eventqa._query_only_payload(payload)
            query_payload["query_prompt"] = explicit_prompt
            query_payload["memorization_prompts"] = [explicit_prompt]

            _cuda_sync()
            generation_start = time.perf_counter()
            result = eventqa._run_eventqa_model(args, model, capacity, query_payload, "off")
            _cuda_sync()
            generation_latency = time.perf_counter() - generation_start
            if result["rendered_query_prompt"] != rendered_prompt:
                raise BM25ContractError("preflight and runtime rendered prompts differ")
            with tempfile.TemporaryDirectory() as tmpdir:
                score = eventqa._score_prediction(args, payload, result["prediction"], tmpdir)
            query_turn = eventqa._query_turn(result)
            records.append(
                {
                    "context_index": args.context_index,
                    "query_index": query_index,
                    "qa_pair_id": payload["qa_pair_id"],
                    "retrieval_query": payload["question"],
                    "retrieved_chunks": [
                        {key: value for key, value in item.items() if key != "text"}
                        for item in selected
                    ],
                    "query_sha256": _sha256(payload["query_prompt"]),
                    "prompt_sha256": _sha256(explicit_prompt),
                    "rendered_prompt_sha256": _sha256(rendered_prompt),
                    "injected_token_count": injected_token_count,
                    "rendered_prompt_token_count": rendered_token_count,
                    "context_capacity": capacity,
                    "capacity_ok": capacity_ok,
                    "prediction": result["prediction"],
                    "substring_exact_match": eventqa._metric_value(
                        score, "substring_exact_match", default=0
                    ),
                    "eventqa_recall": eventqa._metric_value(
                        score, "eventqa_recall", default=0.0
                    ),
                    "format_flags": eventqa._format_flags(result["prediction"]),
                    "cost": {
                        "retrieval_latency_seconds": retrieval_latency,
                        "generation_latency_seconds": generation_latency,
                        "end_to_end_latency_seconds": retrieval_latency + generation_latency,
                        "output_tokens": int(query_turn["output_len"]),
                    },
                }
            )
        peak_gpu_memory = (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else baseline_gpu_memory
        )
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "measurement_mode": "standalone_process",
            "run_id": run_id,
            "scope": {
                "measurement_scope": args.measurement_scope,
                "context_index": args.context_index,
                "question_indices": question_indices,
            },
            "bm25": {"k1": 1.5, "b": 0.75, "top_k": 2},
            "cost": {
                "index_construction_latency_seconds": index_latency,
                "baseline_gpu_memory_bytes": baseline_gpu_memory,
                "peak_gpu_memory_bytes": peak_gpu_memory,
                "incremental_peak_gpu_memory_bytes": peak_gpu_memory - baseline_gpu_memory,
            },
            "records": records,
        }
        validate_artifact(artifact)
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        artifact_name = (
            "smoke_artifact.json"
            if args.measurement_scope == "smoke"
            else "full_artifact.json"
        )
        _write_json(output_dir / artifact_name, artifact)
        _write_jsonl(output_dir / "per_question.jsonl", records)
        _write_json(output_dir / "manifest.json", manifest)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
