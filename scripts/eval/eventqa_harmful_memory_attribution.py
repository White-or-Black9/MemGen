"""Minimal frozen-bank replay and harmful-memory attribution smoke for EventQA.

This is an eval-only diagnostic. It does not change the official scorer/parser or
the latent-memory-bank production API. Slot-only and tuple-only conditions are
explicitly oracle diagnostics and are not deployable inference policies.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
import tempfile
import time
from typing import Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import set_seed
from memgen.model.latent_memory_bank import (
    LatentMemoryBank,
    LatentMemoryBankConfig,
    LatentMemorySlot,
)
from scripts.eval import mab6b_weaver_space_bank_eventqa_65536_n5 as eventqa
from scripts.eval.eventqa_transition_diagnostics import contains_gold


score_prediction = eventqa._score_prediction
format_flags = eventqa._format_flags

DEFAULT_OUTPUT_ROOT = Path("outputs/mab/eventqa_harmful_memory_attribution_smoke")
REQUIRED_CONFIG = {
    "enabled": True,
    "max_slots": 16,
    "top_k": 2,
    "retrieve_threshold": 0.05,
    "update_threshold": 0.10,
    "decay_alpha": 0.05,
    "retrieve_policy": "threshold_topk",
    "update_policy": "thread_update",
}


@dataclass(frozen=True)
class ConditionSpec:
    raw: str
    condition_type: str
    excluded_original_slot_ids: tuple[int, ...] = ()
    included_original_slot_ids: tuple[int, ...] = ()
    forced_original_slot_ids: tuple[int, ...] = ()
    oracle_diagnostic: bool = False


@dataclass(frozen=True)
class FrozenBankSource:
    path: Path
    file_sha256: str
    context_index: int
    context_id: str
    bank: LatentMemoryBank
    saved_snapshot_hash: str
    bank_snapshot_hash: str
    slot_tensor_hashes: tuple[dict, ...]
    raw_state: dict


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bank_state_hash(bank: LatentMemoryBank) -> str:
    """Hash all replay-relevant bank state without mutating it."""
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(bank.config), sort_keys=True).encode("utf-8"))
    digest.update(str(bank._step).encode("ascii"))
    digest.update(str(bank._retrieval_step).encode("ascii"))
    for slot in bank._slots:
        record = {
            "memory": eventqa._tensor_hash(slot.memory),
            "key": eventqa._tensor_hash(slot.key),
            "metadata": slot.metadata,
            "created_step": slot.created_step,
            "last_access_step": slot.last_access_step,
            "last_retrieved_step": slot.last_retrieved_step,
            "access_count": slot.access_count,
            "last_score": slot.last_score,
            "original_device": slot.original_device,
            "original_dtype": slot.original_dtype,
        }
        digest.update(
            json.dumps(record, sort_keys=True, default=str).encode("utf-8")
        )
    return digest.hexdigest()


def _restore_slot(saved: dict, original_slot_index: int) -> LatentMemorySlot:
    metadata = deepcopy(saved.get("metadata") or {})
    recorded = metadata.get("original_slot_index")
    if recorded is not None and int(recorded) != original_slot_index:
        raise ValueError(
            f"frozen slot {original_slot_index} has conflicting original_slot_index={recorded}"
        )
    metadata["original_slot_index"] = original_slot_index
    return LatentMemorySlot(
        memory=saved["memory"].detach().to("cpu").contiguous().clone(),
        key=saved["key"].detach().to("cpu").contiguous().clone(),
        metadata=metadata,
        created_step=int(saved["created_step"]),
        last_access_step=int(saved.get("last_access_step", 0)),
        last_retrieved_step=int(saved["last_retrieved_step"]),
        access_count=int(saved["access_count"]),
        last_score=saved.get("last_score"),
        original_device=str(saved.get("original_device", "cpu")),
        original_dtype=str(saved.get("original_dtype", saved["memory"].dtype)),
    )


def load_frozen_bank(
    path: str | Path, *, expected_context_index: int | None = None
) -> FrozenBankSource:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "format_version", "context_index", "context_id", "step",
        "retrieval_step", "slots", "bank_config", "bank_snapshot",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"frozen bank missing fields: {missing}")
    if int(state["format_version"]) != 1:
        raise ValueError(f"unsupported frozen bank format {state['format_version']}")
    context_index = int(state["context_index"])
    if expected_context_index is not None and context_index != expected_context_index:
        raise ValueError(
            f"frozen context {context_index} != expected {expected_context_index}"
        )
    for field, expected in REQUIRED_CONFIG.items():
        actual = state["bank_config"].get(field)
        if actual != expected:
            raise ValueError(f"unexpected P7 bank config {field}={actual!r}; expected {expected!r}")

    bank = LatentMemoryBank(LatentMemoryBankConfig(**state["bank_config"]))
    bank._slots = [
        _restore_slot(saved, index) for index, saved in enumerate(state["slots"])
    ]
    bank._step = int(state["step"])
    bank._retrieval_step = int(state["retrieval_step"])
    snapshot = eventqa._bank_tensor_snapshot(
        bank, context_index=context_index, context_id=str(state["context_id"])
    )
    saved_snapshot = state["bank_snapshot"]
    if snapshot["combined_frozen_bank_hash"] != saved_snapshot["combined_frozen_bank_hash"]:
        raise ValueError("reconstructed frozen bank snapshot hash mismatch")
    saved_slots = saved_snapshot.get("slots", [])
    if len(saved_slots) != len(bank._slots):
        raise ValueError("frozen bank saved slot count mismatch")
    tensor_hashes = []
    for index, (slot, recorded) in enumerate(zip(bank._slots, saved_slots)):
        actual = {
            "original_slot_index": index,
            "memory_tensor_hash": eventqa._tensor_hash(slot.memory),
            "key_tensor_hash": eventqa._tensor_hash(slot.key),
        }
        if actual["memory_tensor_hash"] != recorded["memory_tensor_hash"]:
            raise ValueError(f"slot {index} memory tensor hash mismatch")
        if actual["key_tensor_hash"] != recorded["key_tensor_hash"]:
            raise ValueError(f"slot {index} key tensor hash mismatch")
        tensor_hashes.append(actual)
    return FrozenBankSource(
        path=path.resolve(),
        file_sha256=_sha256_file(path),
        context_index=context_index,
        context_id=str(state["context_id"]),
        bank=bank,
        saved_snapshot_hash=saved_snapshot["combined_frozen_bank_hash"],
        bank_snapshot_hash=snapshot["combined_frozen_bank_hash"],
        slot_tensor_hashes=tuple(tensor_hashes),
        raw_state=state,
    )


def _parse_ids(value: str) -> tuple[int, ...]:
    try:
        ids = tuple(int(item) for item in value.split(",") if item != "")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid slot list: {value}") from exc
    if not ids or any(index < 0 for index in ids) or len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError(f"invalid slot list: {value}")
    return ids


def parse_condition(value: str) -> ConditionSpec:
    if value == "full":
        return ConditionSpec(raw=value, condition_type="full")
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"invalid condition: {value}")
    kind, raw_ids = value.split(":", 1)
    ids = _parse_ids(raw_ids)
    if kind == "drop-slot" and len(ids) == 1:
        return ConditionSpec(value, kind, excluded_original_slot_ids=ids)
    if kind == "drop-tuple" and len(ids) >= 2:
        return ConditionSpec(value, kind, excluded_original_slot_ids=ids)
    if kind == "slot-only" and len(ids) == 1:
        return ConditionSpec(
            value, kind, included_original_slot_ids=ids,
            forced_original_slot_ids=ids, oracle_diagnostic=True,
        )
    if kind == "tuple-only" and len(ids) >= 2:
        return ConditionSpec(
            value, kind, included_original_slot_ids=ids,
            forced_original_slot_ids=ids, oracle_diagnostic=True,
        )
    raise argparse.ArgumentTypeError(f"invalid condition: {value}")


def _clone_slot(slot: LatentMemorySlot) -> LatentMemorySlot:
    original_id = int(slot.metadata["original_slot_index"])
    return LatentMemorySlot(
        memory=slot.memory.detach().clone(),
        key=slot.key.detach().clone(),
        metadata=deepcopy(slot.metadata),
        created_step=slot.created_step,
        last_access_step=slot.last_access_step,
        last_retrieved_step=slot.last_retrieved_step,
        access_count=slot.access_count,
        last_score=slot.last_score,
        original_device=slot.original_device,
        original_dtype=slot.original_dtype,
    )


def clone_bank_for_condition(
    source: FrozenBankSource, condition: str | ConditionSpec
) -> tuple[LatentMemoryBank, ConditionSpec]:
    spec = parse_condition(condition) if isinstance(condition, str) else condition
    by_id = {
        int(slot.metadata["original_slot_index"]): slot for slot in source.bank._slots
    }
    referenced = set(spec.excluded_original_slot_ids + spec.included_original_slot_ids)
    missing = sorted(referenced - set(by_id))
    if missing:
        raise ValueError(f"condition references missing original slots: {missing}")
    if spec.included_original_slot_ids:
        selected = [by_id[index] for index in spec.included_original_slot_ids]
    else:
        excluded = set(spec.excluded_original_slot_ids)
        selected = [
            slot for slot in source.bank._slots
            if int(slot.metadata["original_slot_index"]) not in excluded
        ]
    bank = LatentMemoryBank(source.bank.config)
    bank._slots = [_clone_slot(slot) for slot in selected]
    bank._step = source.bank._step
    bank._retrieval_step = source.bank._retrieval_step
    return bank, spec


def assert_query_read_only(bank: LatentMemoryBank, expected_hash: str) -> None:
    actual = bank_state_hash(bank)
    if actual != expected_hash:
        raise RuntimeError(
            f"query modified attribution bank: before={expected_hash} after={actual}"
        )


def create_output_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def _prediction_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_original_rows(source_run: Path, context_index: int) -> dict[int, dict]:
    path = source_run / "eventqa_per_question.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if int(row["context_index"]) == context_index:
            rows[int(row["query_id"])] = row
    return rows


def _runner_args(cli_args, source_config: dict):
    args = eventqa.build_parser().parse_args([])
    args.dataset_root = source_config["dataset_root"]
    args.mab_repo = source_config["mab_repo"]
    args.checkpoint_path = source_config["checkpoint_path"]
    args.model_checkpoint_id = source_config["model_checkpoint_id"]
    args.context_index = cli_args.context_index
    args.requested_contexts = 1
    args.question_limit = cli_args.question_count
    args.eventqa_protocol = "frozen_context_bank"
    args.skip_research_note = True
    args.reseed_per_context = True
    args.trace_score_decomposition = True
    args.save_frozen_bank = True
    args.bank_transition_diagnostics = False
    args.generation_max_length = int(source_config["generation_max_length"])
    bank_config = source_config["latent_memory_bank_config"]
    args.retrieve_threshold = bank_config["retrieve_threshold"]
    args.update_threshold = bank_config["update_threshold"]
    args.max_slots = bank_config["max_slots"]
    args.top_k = bank_config["top_k"]
    args.decay_alpha = bank_config["decay_alpha"]
    return args


def _original_ids(bank: LatentMemoryBank, current_indices: Iterable[int]) -> list[int]:
    return [
        int(bank._slots[index].metadata["original_slot_index"])
        for index in current_indices
    ]


def _format_failure(flags: dict) -> bool:
    return any(bool(value) for value in flags.values())


def _run_query(
    *, args, model, capacity: int, payload: dict, source: FrozenBankSource,
    condition: ConditionSpec, source_row: dict, source_run_id: str,
) -> dict:
    bank, spec = clone_bank_for_condition(source, condition)
    before = bank_state_hash(bank)
    started = time.perf_counter()
    result = eventqa._run_eventqa_model(
        args, model, capacity, eventqa._query_only_payload(payload), "on",
        eventqa._eventqa_bank_config(args), external_bank=bank,
        preserve_bank=True, recorded_bank_config=source.raw_state["bank_config"],
        score_trace_state={},
    )
    latency = time.perf_counter() - started
    retained = result.pop("_retained_bank")
    if retained is not bank:
        raise RuntimeError("query replaced attribution bank instance")
    assert_query_read_only(bank, before)
    prediction = result["prediction"]
    with tempfile.TemporaryDirectory() as tmpdir:
        score = score_prediction(args, payload, prediction, tmpdir)
    parsed = eventqa._metric_value(score, "parsed_output")
    em = int(bool(eventqa._metric_value(score, eventqa.METRIC_KEY, default=False)))
    recall = eventqa._metric_value(score, eventqa.OPTIONAL_METRIC_KEY)
    flags = format_flags(prediction)
    query_turn = eventqa._query_turn(result)
    retrieved_indices = list(query_turn["retrieved_indices"])
    return {
        "source_run_id": source_run_id,
        "source_frozen_bank_path": str(source.path),
        "source_frozen_bank_hash": source.file_sha256,
        "context_index": int(payload["context_index"]),
        "question_index": int(payload["query_id"]),
        "question_id": payload.get("question_id"),
        "condition": spec.raw,
        "condition_type": spec.condition_type,
        "oracle_diagnostic": spec.oracle_diagnostic,
        "excluded_original_slot_ids": list(spec.excluded_original_slot_ids),
        "included_original_slot_ids": list(spec.included_original_slot_ids),
        "forced_original_slot_ids": list(spec.forced_original_slot_ids),
        "counterfactual_retrieved_original_slot_ids": _original_ids(bank, retrieved_indices),
        "counterfactual_retrieved_scores": list(query_turn["retrieved_scores"]),
        "original_prediction_hash": _prediction_hash(source_row["bank_on_prediction"]),
        "replay_prediction_hash": _prediction_hash(prediction),
        "baseline_replay_matched": None,
        "raw_prediction": prediction,
        "parsed_prediction": parsed,
        "official_em": em,
        "recall": recall,
        "contains_gold": contains_gold(prediction, payload["gold_answers"]),
        "no_gold": not contains_gold(prediction, payload["gold_answers"]),
        "format_failure": _format_failure(flags),
        "format_flags": flags,
        "latency_seconds": latency,
        "query_read_only": True,
    }


def _validate_replay(row: dict, original: dict) -> dict:
    expected_ids = list(original["bank_on_query_turn_retrieved_indices"])
    checks = {
        "official_em": row["official_em"] == int(original["bank_on_substring_exact_match"]),
        "recall": row["recall"] == original["bank_on_eventqa_recall"],
        "retrieved_original_slot_ids": row["counterfactual_retrieved_original_slot_ids"] == expected_ids,
        "raw_prediction_hash": row["replay_prediction_hash"] == row["original_prediction_hash"],
        "parsed_prediction": row["parsed_prediction"] == original["bank_on_parsed_prediction"],
        "format_flags": row["format_flags"] == original["bank_on_format_flags"],
    }
    return {
        "question_index": row["question_index"],
        "matched": all(checks.values()),
        "checks": checks,
        "mismatched_fields": [key for key, matched in checks.items() if not matched],
        "expected": {
            "official_em": int(original["bank_on_substring_exact_match"]),
            "recall": original["bank_on_eventqa_recall"],
            "retrieved_original_slot_ids": expected_ids,
            "raw_prediction_hash": row["original_prediction_hash"],
            "parsed_prediction": original["bank_on_parsed_prediction"],
            "format_flags": original["bank_on_format_flags"],
        },
        "actual": {
            "official_em": row["official_em"], "recall": row["recall"],
            "retrieved_original_slot_ids": row["counterfactual_retrieved_original_slot_ids"],
            "raw_prediction_hash": row["replay_prediction_hash"],
            "parsed_prediction": row["parsed_prediction"],
            "format_flags": row["format_flags"],
        },
    }


def _condition_summary(condition: str, rows: list[dict], baseline: dict[int, dict]) -> dict:
    rescues = regressions = 0
    no_gold_rescues = no_gold_regressions = 0
    format_improvements = format_regressions = 0
    for row in rows:
        base = baseline[row["question_index"]]
        rescues += int(not base["official_em"] and bool(row["official_em"]))
        regressions += int(bool(base["official_em"]) and not row["official_em"])
        no_gold_rescues += int(base["no_gold"] and not row["no_gold"])
        no_gold_regressions += int(not base["no_gold"] and row["no_gold"])
        format_improvements += int(base["format_failure"] and not row["format_failure"])
        format_regressions += int(not base["format_failure"] and row["format_failure"])
    total = len(rows)
    return {
        "condition": condition, "question_count": total,
        "em_count": sum(r["official_em"] for r in rows),
        "em": sum(r["official_em"] for r in rows) / total if total else None,
        "mean_recall": sum(float(r["recall"] or 0) for r in rows) / total if total else None,
        "no_gold_count": sum(int(r["no_gold"]) for r in rows),
        "format_failure_count": sum(int(r["format_failure"]) for r in rows),
        "rescue_count": rescues, "regression_count": regressions,
        "no_gold_rescue_count": no_gold_rescues,
        "no_gold_regression_count": no_gold_regressions,
        "format_improvement_count": format_improvements,
        "format_regression_count": format_regressions,
        "mean_latency_seconds": sum(r["latency_seconds"] for r in rows) / total if total else None,
    }


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_outputs(output_dir: Path, rows: list[dict], summaries: dict, replay: dict) -> None:
    _write_jsonl(output_dir / "attribution_per_question.jsonl", rows)
    slot = {k: v for k, v in summaries.items() if "slot" in k}
    tuples = {k: v for k, v in summaries.items() if "tuple" in k}
    _write_json(output_dir / "attribution_per_slot.json", slot)
    _write_json(output_dir / "attribution_per_tuple.json", tuples)
    context_indices = sorted({int(row["context_index"]) for row in rows})
    if len(context_indices) > 1:
        raise ValueError(
            f"attribution_per_context expects one context, got {context_indices}"
        )
    _write_json(
        output_dir / "attribution_per_context.json",
        {
            "context_index": context_indices[0] if context_indices else None,
            "question_count": len(
                {int(row["question_index"]) for row in rows}
            ),
            "replay_gate_passed": replay["passed"],
            "conditions": summaries,
        },
    )
    _write_json(
        output_dir / "attribution_summary.json",
        {"replay_gate_passed": replay["passed"], "conditions": summaries},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--context-index", type=int, required=True)
    parser.add_argument("--question-start", type=int, default=0)
    parser.add_argument("--question-count", type=int, default=10)
    parser.add_argument("--condition", action="append", required=True)
    parser.add_argument("--require-baseline-replay-match", action="store_true")
    parser.add_argument("--skip-research-note", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    conditions = [parse_condition(value) for value in cli.condition]
    if not conditions or conditions[0].condition_type != "full":
        raise ValueError("first condition must be full for the replay gate")
    if cli.question_count <= 0:
        raise ValueError("question-count must be positive")
    source_run = cli.source_run.resolve()
    source_config = json.loads((source_run / "run_config.json").read_text(encoding="utf-8"))
    frozen_path = source_run / "frozen_banks" / f"context_{cli.context_index}.pt"
    source = load_frozen_bank(frozen_path, expected_context_index=cli.context_index)
    original_rows = _load_original_rows(source_run, cli.context_index)
    question_indices = list(range(cli.question_start, cli.question_start + cli.question_count))
    missing_questions = [index for index in question_indices if index not in original_rows]
    if missing_questions:
        raise ValueError(f"source run missing questions: {missing_questions}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = create_output_directory(
        cli.output_root / f"{timestamp}-p7-context{cli.context_index}-q{question_indices[0]}-{question_indices[-1]}"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(output_dir / "run.log", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    run_config = {
        "source_run": str(source_run), "source_run_id": source_config["run_id"],
        "source_frozen_bank_path": str(source.path),
        "source_frozen_bank_hash": source.file_sha256,
        "source_bank_snapshot_hash": source.bank_snapshot_hash,
        "context_index": cli.context_index, "question_indices": question_indices,
        "conditions": [asdict(spec) for spec in conditions],
        "require_baseline_replay_match": cli.require_baseline_replay_match,
        "skip_research_note": True, "preflight_only": cli.preflight_only,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
    }
    _write_json(output_dir / "run_config.json", run_config)
    manifest = {
        **run_config, "status": "preflight" if cli.preflight_only else "running",
        "output_dir": str(output_dir.resolve()),
        "official_scorer": str(eventqa._bridge_script()),
        "official_parser_modified": False, "model_code_modified": False,
    }
    _write_json(output_dir / "manifest.json", manifest)
    logging.info("validated frozen bank %s with %d slots", source.file_sha256, len(source.bank))
    if cli.preflight_only:
        replay = {"passed": False, "status": "preflight_only", "questions": []}
        _write_json(output_dir / "replay_validation.json", replay)
        _write_outputs(output_dir, [], {}, replay)
        manifest["status"] = "preflight_complete"
        _write_json(output_dir / "manifest.json", manifest)
        print(output_dir)
        return 0

    args = _runner_args(cli, source_config)
    rows = eventqa._load_rows(args.parquet, eventqa.SUB_DATASET)
    context_payload = eventqa.build_context_payload(
        args, rows[cli.context_index], cli.context_index, source_config["timestamp"]
    )
    model, capacity = eventqa.weaver_bank._load_model(args)
    runtime_config = eventqa._eventqa_bank_config(args)
    eventqa._assert_runtime_bank_config_matches(source.bank.config, runtime_config)

    all_rows: list[dict] = []
    baseline_rows: dict[int, dict] = {}
    validation_rows: list[dict] = []
    effective_seed = int(getattr(args, "seed", 42)) + cli.context_index
    set_seed(effective_seed, use_gpu=torch.cuda.is_available())
    logging.info("running full-bank replay gate on questions %s", question_indices)
    for question_index in question_indices:
        payload = eventqa.build_question_payload(context_payload, question_index)
        row = _run_query(
            args=args, model=model, capacity=capacity, payload=payload, source=source,
            condition=conditions[0], source_row=original_rows[question_index],
            source_run_id=source_config["run_id"],
        )
        validation = _validate_replay(row, original_rows[question_index])
        row["baseline_replay_matched"] = validation["matched"]
        all_rows.append(row)
        baseline_rows[question_index] = row
        validation_rows.append(validation)
        logging.info("full q=%d matched=%s mismatches=%s", question_index, validation["matched"], validation["mismatched_fields"])
    replay = {
        "passed": all(row["matched"] for row in validation_rows),
        "question_count": len(validation_rows),
        "matched_count": sum(int(row["matched"]) for row in validation_rows),
        "mismatch_count": sum(int(not row["matched"]) for row in validation_rows),
        "questions": validation_rows,
    }
    _write_json(output_dir / "replay_validation.json", replay)
    summaries = {"full": _condition_summary("full", list(baseline_rows.values()), baseline_rows)}
    if not replay["passed"] and cli.require_baseline_replay_match:
        logging.error("baseline replay gate failed; attribution conditions skipped")
        _write_outputs(output_dir, all_rows, summaries, replay)
        manifest.update({"status": "replay_gate_failed", "finished_at": datetime.now(timezone.utc).isoformat()})
        _write_json(output_dir / "manifest.json", manifest)
        print(output_dir)
        return 2

    for spec in conditions[1:]:
        set_seed(effective_seed, use_gpu=torch.cuda.is_available())
        condition_rows = []
        logging.info("running condition %s", spec.raw)
        for question_index in question_indices:
            payload = eventqa.build_question_payload(context_payload, question_index)
            row = _run_query(
                args=args, model=model, capacity=capacity, payload=payload, source=source,
                condition=spec, source_row=original_rows[question_index],
                source_run_id=source_config["run_id"],
            )
            row["baseline_replay_matched"] = True
            condition_rows.append(row)
            all_rows.append(row)
        summaries[spec.raw] = _condition_summary(spec.raw, condition_rows, baseline_rows)
    _write_outputs(output_dir, all_rows, summaries, replay)
    manifest.update({"status": "complete", "finished_at": datetime.now(timezone.utc).isoformat()})
    _write_json(output_dir / "manifest.json", manifest)
    logging.info("smoke complete: %s", output_dir)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
