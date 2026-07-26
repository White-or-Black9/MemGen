"""Construct a same-model rolling text summary for EventQA context 0."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import mab6b_weaver_space_bank_eventqa_65536_n5 as eventqa


SCHEMA_VERSION = "eventqa-text-summary-construction/v1"
DEFAULT_OUTPUT_ROOT = "outputs/mab/eventqa_text_summary_construction_smoke"
SUMMARY_SYSTEM = (
    "You maintain a compact persistent text memory for future questions. "
    "Return only the updated memory summary."
)
SUMMARY_INSTRUCTION = """Update the persistent memory summary using the new event text.

Requirements:
- Preserve named entities, relationships, locations, times, actions, promises, and causal order.
- Keep information that may help answer future questions.
- Do not answer any question.
- Do not infer facts absent from the source.
- Output only the updated memory summary.
- Maximum length: 128 tokenizer tokens.

Previous summary:
{previous_summary}

New event text:
{chunk}"""


class SummaryContractError(ValueError):
    """Raised when rolling-summary construction violates its contract."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_summary_prompt(previous_summary: str, chunk: str) -> str:
    return SUMMARY_INSTRUCTION.format(
        previous_summary=previous_summary,
        chunk=chunk,
    )


def persist_summary(tokenizer, raw_summary: str, *, budget: int = 128) -> dict[str, Any]:
    raw_ids = tokenizer.encode(raw_summary, add_special_tokens=False)
    persisted_ids = list(raw_ids[:budget])
    persisted = tokenizer.decode(
        persisted_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return {
        "raw_token_count": len(raw_ids),
        "persisted_token_count": len(persisted_ids),
        "persisted_token_ids": persisted_ids,
        "persisted_summary": persisted,
        "truncated": len(raw_ids) > budget,
    }


def _finite_nonnegative(value: Any, label: str) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise SummaryContractError(f"{label} must be finite and nonnegative")


def validate_construction_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise SummaryContractError("unexpected schema version")
    scope = artifact.get("scope", {})
    if scope.get("context_index") not in range(5) or not isinstance(scope.get("chunk_count"), int):
        raise SummaryContractError("construction context must be in 0-4")
    if artifact.get("method") != {
        "summary_token_budget": 128,
        "latent_memory_bank": False,
    }:
        raise SummaryContractError("summary method configuration drift")
    traces = artifact.get("traces", [])
    chunk_count = scope["chunk_count"]
    if len(traces) != chunk_count or [row.get("chunk_index") for row in traces] != list(range(chunk_count)):
        raise SummaryContractError("summary trace must be complete and ordered")
    cost = artifact.get("cost", {})
    for field in (
        "construction_latency_seconds",
        "baseline_gpu_memory_bytes",
        "peak_gpu_memory_bytes",
    ):
        _finite_nonnegative(cost.get(field), field)
    if cost["peak_gpu_memory_bytes"] < cost["baseline_gpu_memory_bytes"]:
        raise SummaryContractError("peak GPU memory cannot be below baseline")
    previous_summary = ""
    for trace in traces:
        if trace.get("previous_summary_sha256") != _sha256(previous_summary):
            raise SummaryContractError("previous summary hash chain is broken")
        for field in (
            "chunk_sha256",
            "previous_summary_sha256",
            "rendered_input_sha256",
            "raw_output_sha256",
            "persisted_summary_sha256",
        ):
            if len(trace.get(field, "")) != 64:
                raise SummaryContractError(f"invalid trace hash: {field}")
        for field in (
            "chunk_token_count",
            "previous_summary_token_count",
            "rendered_input_token_count",
            "raw_output_token_count",
            "persisted_summary_token_count",
            "latency_seconds",
        ):
            _finite_nonnegative(trace.get(field), field)
        summary = trace.get("persisted_summary", "")
        if not summary.strip() or trace["persisted_summary_token_count"] > 128:
            raise SummaryContractError("persisted summary must be nonempty and within 128 tokens")
        if trace["persisted_summary_sha256"] != _sha256(summary):
            raise SummaryContractError("persisted summary hash mismatch")
        previous_summary = summary
    if not artifact.get("final_summary", "").strip():
        raise SummaryContractError("final summary must be nonempty")
    if artifact["final_summary"] != previous_summary:
        raise SummaryContractError("final summary does not match trace tail")
    if artifact.get("final_summary_sha256") != _sha256(previous_summary):
        raise SummaryContractError("final summary hash mismatch")
    if artifact.get("final_summary_token_count", 129) > 128:
        raise SummaryContractError("final summary exceeds 128 tokens")


def build_parser():
    parser = eventqa.build_parser()
    parser.description = __doc__
    parser.add_argument("--summary-token-budget", type=int, default=128)
    parser.add_argument("--run-id")
    parser.set_defaults(
        output_root=DEFAULT_OUTPUT_ROOT,
        requested_contexts=1,
        context_index=0,
        question_limit=0,
        skip_research_note=True,
        reseed_per_context=True,
    )
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.context_index not in range(5) or args.summary_token_budget != 128:
        raise SummaryContractError("construction requires context 0-4 and budget 128")
    started_at = datetime.now(timezone.utc).isoformat()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{timestamp}-eventqa-text-summary-construction-ctx{args.context_index}"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]

    rows = eventqa._load_rows(args.parquet, eventqa.SUB_DATASET)
    context_payload = eventqa.build_context_payload(args, rows[args.context_index], args.context_index, started_at)
    chunks = context_payload["chunks"]
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
            "exact_command": command,
            "construction_only": True,
            "query_data_accessed": False,
            "summary_token_budget": 128,
            "latent_memory_bank": False,
            "chunk_count": len(chunks),
            "chunk_hashes": [_sha256(chunk) for chunk in chunks],
        }
    )
    _write_json(output_dir / "manifest.json", manifest)

    model, capacity = eventqa.weaver_bank._load_model(args)
    tokenizer = model.tokenizer
    import torch
    from transformers import GenerationConfig

    manifest["context_capacity"] = capacity
    manifest["gpu"] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline_gpu_memory = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
    else:
        baseline_gpu_memory = 0

    generation_config = GenerationConfig(
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generation_config.weaver_do_sample = False
    generation_config.trigger_do_sample = False
    traces = []
    summary = ""
    try:
        for chunk_index, chunk in enumerate(chunks):
            prompt = build_summary_prompt(summary, chunk)
            messages = [
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": prompt},
            ]
            start = time.perf_counter()
            rendered = tokenizer.apply_chat_template(
                [messages],
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_len = int(rendered["input_ids"].shape[1])
            if input_len > capacity:
                raise SummaryContractError("summary input exceeds model capacity")
            inputs = {key: value.to(model.device) for key, value in rendered.items()}
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=generation_config,
                latent_memory_bank=None,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - start
            generated_ids = output[0, input_len:].detach().cpu().tolist()
            raw_summary = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            persisted = persist_summary(tokenizer, raw_summary, budget=128)
            if not persisted["persisted_summary"]:
                raise SummaryContractError(f"empty summary at chunk {chunk_index}")
            previous_summary = summary
            summary = persisted["persisted_summary"]
            traces.append(
                {
                    "chunk_index": chunk_index,
                    "chunk_sha256": _sha256(chunk),
                    "chunk_token_count": len(tokenizer.encode(chunk, add_special_tokens=False)),
                    "previous_summary_sha256": _sha256(previous_summary),
                    "previous_summary_token_count": len(tokenizer.encode(previous_summary, add_special_tokens=False)),
                    "rendered_input_sha256": hashlib.sha256(rendered["input_ids"].cpu().numpy().tobytes()).hexdigest(),
                    "rendered_input_token_count": input_len,
                    "raw_output_sha256": _sha256(raw_summary),
                    "raw_output_token_count": persisted["raw_token_count"],
                    "persisted_summary": summary,
                    "persisted_summary_sha256": _sha256(summary),
                    "persisted_summary_token_count": persisted["persisted_token_count"],
                    "truncated": persisted["truncated"],
                    "latency_seconds": latency,
                }
            )
        peak_gpu_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else baseline_gpu_memory
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "scope": {"context_index": args.context_index, "chunk_count": len(chunks)},
            "method": {"summary_token_budget": 128, "latent_memory_bank": False},
            "cost": {
                "construction_latency_seconds": sum(row["latency_seconds"] for row in traces),
                "baseline_gpu_memory_bytes": baseline_gpu_memory,
                "peak_gpu_memory_bytes": peak_gpu_memory,
                "incremental_peak_gpu_memory_bytes": peak_gpu_memory - baseline_gpu_memory,
            },
            "traces": traces,
            "final_summary": summary,
            "final_summary_sha256": _sha256(summary),
            "final_summary_token_count": len(tokenizer.encode(summary, add_special_tokens=False)),
        }
        validate_construction_artifact(artifact)
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(output_dir / "construction_artifact.json", artifact)
        _write_jsonl(output_dir / "summary_trace.jsonl", traces)
        _write_json(output_dir / "manifest.json", manifest)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
