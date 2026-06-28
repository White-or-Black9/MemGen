"""MAB-6B-FR top-k diagnostic for Weaver-space memory bank on DetectiveQA n=10."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import mab6b_weaver_space_bank_detectiveqa_n10_capacity_diagnostic as cap

EXPERIMENT_NAME = "MAB-6B-FR top-k diagnostic: Weaver-space bank + final-query format repair"
RUN_PREFIX = "detectiveqa-version-b-weaver-space-bank-format-repair-topk-diagnostic-n10"
DEFAULT_OUTPUT_ROOT = "outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_topk_diagnostic"
DEFAULT_MAX_SLOTS = 16
DEFAULT_RETRIEVE_THRESHOLD = 0.03
DEFAULT_UPDATE_THRESHOLD = 0.08
DEFAULT_TOP_K_VALUES = (1, 2, 4, 8)
DEFAULT_SMOKE_TOP_K = 2
CANONICAL_FR_BASELINE = cap.CANONICAL_FR_BASELINE
FORMAT_FAILURE_CATEGORIES = {"code_leak", "json_leak", "reasoning_sprawl", "refusal", "language_drift", "other"}
DEBUG_FAILURE_TRACEBACK = Path(DEFAULT_OUTPUT_ROOT) / "debug_failure_traceback.log"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _progress(message: str) -> None:
    print(f"[topk-diagnostic] {message}", flush=True)


def _log_failure_traceback(exc: BaseException) -> None:
    DEBUG_FAILURE_TRACEBACK.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    DEBUG_FAILURE_TRACEBACK.write_text(rendered, encoding="utf-8")
    sys.stderr.write(rendered)
    sys.stderr.flush()
    sys.stdout.flush()


def _parse_int_list(value: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        number = int(part)
        if number <= 0:
            raise ValueError("top-k values must be positive integers")
        values.append(number)
    if not values:
        raise ValueError("at least one top-k value is required")
    return tuple(values)


def _suppress_research_note(*args, **kwargs):
    output_dir = kwargs.get("output_dir")
    if output_dir is not None:
        return Path(output_dir) / "suppressed_research_note.md"
    return Path("suppressed_research_note.md")


def _topk_label(top_k: int, max_slots: int, retrieve_threshold: float, update_threshold: float) -> str:
    return f"cap{max_slots}_rt{retrieve_threshold:.2f}_ut{update_threshold:.2f}_topk{top_k}"


def _load_canonical_contexts() -> list[dict] | None:
    paired = Path(CANONICAL_FR_BASELINE) / "paired_results.json"
    if not paired.exists():
        return None
    data = _load_json(paired)
    return data.get("contexts")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-slots", type=int, default=DEFAULT_MAX_SLOTS)
    parser.add_argument("--top-k-values", default="1,2,4,8")
    parser.add_argument("--retrieve-threshold", type=float, default=DEFAULT_RETRIEVE_THRESHOLD)
    parser.add_argument("--update-threshold", type=float, default=DEFAULT_UPDATE_THRESHOLD)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser


def _score_exact_match(*, mab_python: str, mab_repo: str, prediction: str | None, gold_answer: Any, dataset_config: dict | None) -> int:
    return cap._score_exact_match(
        mab_python=mab_python,
        mab_repo=mab_repo,
        prediction=prediction,
        gold_answer=gold_answer,
        dataset_config=dataset_config,
    )


def _query_turn_retrieved_indices(row: dict) -> list[int]:
    indices = row.get("query_turn_retrieved_indices")
    if indices is not None:
        return list(indices)
    by_turn = row.get("retrieved_indices_by_turn") or []
    return list(by_turn[-1]) if by_turn else []


def _query_turn_retrieved_latent_count(row: dict) -> int:
    for key in ("query_turn_retrieved_latent_count", "memory_retrieved_latent_count", "retrieved_latent_count"):
        value = row.get(key)
        if value is not None:
            return int(value)
    indices = _query_turn_retrieved_indices(row)
    if indices:
        return len(indices) * 8
    by_turn = row.get("retrieved_indices_by_turn") or []
    if by_turn:
        return len(by_turn[-1]) * 8
    return 0


def _write_counts(row: dict) -> tuple[int, int, int]:
    append_insert = row.get("bank_on_append_insert_count")
    matched_replace = row.get("bank_on_matched_replace_count")
    capacity_evict = row.get("bank_on_capacity_evict_count")
    if append_insert is not None or matched_replace is not None or capacity_evict is not None:
        return int(append_insert or 0), int(matched_replace or 0), int(capacity_evict or 0)
    action_counts = row.get("bank_on_write_action_counts") or row.get("write_action_counts") or {}
    return (
        int(action_counts.get("insert", 0)),
        int(action_counts.get("replace_matched", 0)),
        int(action_counts.get("capacity_evict", action_counts.get("evict_oldest_insert", 0))),
    )


def _classify_prediction(text: str | None) -> str | None:
    return cap._classify_prediction(text)


def _summarize_run(
    *,
    top_k: int,
    max_slots: int,
    retrieve_threshold: float,
    update_threshold: float,
    run_result: dict,
    canonical_contexts: list[dict] | None,
) -> tuple[dict, list[dict]]:
    paired = run_result["paired_results"]
    contexts = paired["contexts"]
    per_context_rows: list[dict] = []
    exact_off = 0
    exact_on = 0
    improved_count = 0
    regressed_count = 0
    output_changed_count = 0
    final_slot_counts: list[int] = []
    append_insert_total = 0
    matched_replace_total = 0
    capacity_evict_total = 0
    query_write_total = 0
    query_write_attempt_total = 0
    error_count = 0
    error_or_stop_reasons: Counter[str] = Counter()
    retrieved_latents_enter_weaver = False
    raw_retrieved_latents_enter_reasoner = False
    query_turn_retrieved_indices_by_context: list[list[int]] = []
    query_turn_retrieved_latent_count_by_context: list[int] = []
    bank_on_retrieved_latent_count_by_context: list[int] = []
    format_failure_counts_bank_off = 0
    format_failure_counts_bank_on = 0
    bank_on_write_action_counts: Counter[str] = Counter()
    bank_on_update_reason_counts: Counter[str] = Counter()

    for idx, row in enumerate(contexts):
        bank_off_exact_match = _score_exact_match(
            mab_python=run_result["run_config"]["mab_python"],
            mab_repo=run_result["run_config"]["mab_repo"],
            prediction=row.get("bank_off_prediction"),
            gold_answer=row.get("gold_answer"),
            dataset_config=run_result["run_config"].get("dataset_config"),
        )
        bank_on_exact_match = _score_exact_match(
            mab_python=run_result["run_config"]["mab_python"],
            mab_repo=run_result["run_config"]["mab_repo"],
            prediction=row.get("bank_on_prediction"),
            gold_answer=row.get("gold_answer"),
            dataset_config=run_result["run_config"].get("dataset_config"),
        )
        bank_off_status = _classify_prediction(row.get("bank_off_prediction"))
        bank_on_status = _classify_prediction(row.get("bank_on_prediction"))
        append_insert_count, matched_replace_count, capacity_evict_count = _write_counts(row)
        query_turn_indices = _query_turn_retrieved_indices(row)
        query_turn_latent_count = _query_turn_retrieved_latent_count(row)
        bank_on_retrieved_latent_count = int(row.get("bank_on_retrieved_latent_count") or 0)
        error_or_stop_reason = row.get("error_or_stop_reason")
        if error_or_stop_reason:
            error_count += 1
            error_or_stop_reasons[str(error_or_stop_reason)] += 1

        canonical_row = canonical_contexts[idx] if canonical_contexts and idx < len(canonical_contexts) else None
        output_changed_vs_canonical = None
        if canonical_row is not None and row.get("bank_on_prediction") is not None:
            output_changed_vs_canonical = row["bank_on_prediction"] != canonical_row.get("bank_on_prediction")

        row_summary = {
            "capacity_label": _topk_label(top_k, max_slots, retrieve_threshold, update_threshold),
            "max_slots": max_slots,
            "top_k": top_k,
            "retrieve_threshold": retrieve_threshold,
            "update_threshold": update_threshold,
            "context_index": row.get("context_index"),
            "context_id": row.get("context_id"),
            "gold_answer": row.get("gold_answer"),
            "bank_off_prediction": row.get("bank_off_prediction"),
            "bank_off_EM": bank_off_exact_match,
            "bank_on_prediction": row.get("bank_on_prediction"),
            "bank_on_EM": bank_on_exact_match,
            "improved": int(bank_on_exact_match > bank_off_exact_match),
            "regressed": int(bank_on_exact_match < bank_off_exact_match),
            "output_changed": int(row.get("bank_off_prediction") != row.get("bank_on_prediction")),
            "final_slot_count": int(row.get("bank_slot_count_final_before_reset") or 0),
            "query_turn_retrieved_indices": query_turn_indices,
            "query_turn_retrieved_latent_count": query_turn_latent_count,
            "bank_on_write_action_counts": {
                "insert": append_insert_count,
                "replace_matched": matched_replace_count,
                "capacity_evict": capacity_evict_count,
            },
            "bank_on_update_reason_counts": row.get("bank_on_update_reason_counts") or {},
            "error_or_stop_reason": error_or_stop_reason,
            "bank_on_exact_match_vs_canonical": output_changed_vs_canonical,
            "bank_on_retrieved_latent_count": bank_on_retrieved_latent_count,
            "bank_on_append_insert_count": append_insert_count,
            "bank_on_matched_replace_count": matched_replace_count,
            "bank_on_capacity_evict_count": capacity_evict_count,
            "bank_on_query_write_count": int(row.get("query_write_count") or 0),
            "bank_on_query_write_attempt_count": int(row.get("query_write_attempt_count") or 0),
            "cross_context_leakage_detected": bool(row.get("cross_context_leakage_detected")),
            "retrieved_latents_enter_weaver": bool(row.get("retrieved_latents_enter_weaver")),
            "raw_retrieved_latents_enter_reasoner": bool(row.get("raw_retrieved_latents_enter_reasoner")),
            "bank_off_primary_output_format_status": bank_off_status,
            "bank_on_primary_output_format_status": bank_on_status,
        }
        per_context_rows.append(row_summary)

        exact_off += bank_off_exact_match
        exact_on += bank_on_exact_match
        improved_count += row_summary["improved"]
        regressed_count += row_summary["regressed"]
        output_changed_count += row_summary["output_changed"]
        format_failure_counts_bank_off += int(bank_off_status != "clean_option")
        format_failure_counts_bank_on += int(bank_on_status != "clean_option")
        final_slot_counts.append(row_summary["final_slot_count"])
        append_insert_total += append_insert_count
        matched_replace_total += matched_replace_count
        capacity_evict_total += capacity_evict_count
        query_write_total += int(row.get("query_write_count") or 0)
        query_write_attempt_total += int(row.get("query_write_attempt_count") or 0)
        retrieved_latents_enter_weaver = retrieved_latents_enter_weaver or bool(row.get("retrieved_latents_enter_weaver"))
        raw_retrieved_latents_enter_reasoner = raw_retrieved_latents_enter_reasoner or bool(
            row.get("raw_retrieved_latents_enter_reasoner")
        )
        query_turn_retrieved_indices_by_context.append(query_turn_indices)
        query_turn_retrieved_latent_count_by_context.append(query_turn_latent_count)
        bank_on_retrieved_latent_count_by_context.append(bank_on_retrieved_latent_count)
        bank_on_write_action_counts["insert"] += append_insert_count
        bank_on_write_action_counts["replace_matched"] += matched_replace_count
        bank_on_write_action_counts["capacity_evict"] += capacity_evict_count
        for reason, count in (row.get("bank_on_update_reason_counts") or {}).items():
            bank_on_update_reason_counts[str(reason)] += int(count)

    valid_count = len(contexts)
    summary = {
        "capacity_label": _topk_label(top_k, max_slots, retrieve_threshold, update_threshold),
        "max_slots": max_slots,
        "top_k": top_k,
        "retrieve_threshold": retrieve_threshold,
        "update_threshold": update_threshold,
        "bank_off_EM": exact_off / valid_count if valid_count else None,
        "bank_on_EM": exact_on / valid_count if valid_count else None,
        "improved_count": improved_count,
        "regressed_count": regressed_count,
        "output_changed_count": output_changed_count,
        "final_slot_counts": final_slot_counts,
        "mean_final_slot_count": (sum(final_slot_counts) / len(final_slot_counts)) if final_slot_counts else None,
        "append_insert_count": append_insert_total,
        "matched_replace_count": matched_replace_total,
        "capacity_evict_count": capacity_evict_total,
        "query_write_count": query_write_total,
        "query_write_attempt_count": query_write_attempt_total,
        "cross_context_leakage_detected": any(bool(row.get("cross_context_leakage_detected")) for row in contexts),
        "retrieved_latents_enter_weaver": retrieved_latents_enter_weaver,
        "raw_retrieved_latents_enter_reasoner": raw_retrieved_latents_enter_reasoner,
        "query_turn_retrieved_indices_by_context": query_turn_retrieved_indices_by_context,
        "query_turn_retrieved_latent_count_by_context": query_turn_retrieved_latent_count_by_context,
        "bank_on_retrieved_latent_count_by_context": bank_on_retrieved_latent_count_by_context,
        "format_failure_counts_bank_off": format_failure_counts_bank_off,
        "format_failure_counts_bank_on": format_failure_counts_bank_on,
        "error_count": error_count,
        "error_or_stop_reasons": dict(error_or_stop_reasons),
        "write_action_counts": dict(bank_on_write_action_counts),
        "update_reason_counts": dict(bank_on_update_reason_counts),
    }
    return summary, per_context_rows


def _run_single_topk(
    top_k: int,
    *,
    max_slots: int,
    retrieve_threshold: float,
    update_threshold: float,
    requested_contexts: int | None = None,
    output_root: Path | None = None,
) -> dict:
    topk_label = _topk_label(top_k, max_slots, retrieve_threshold, update_threshold)
    base_output_root = Path(DEFAULT_OUTPUT_ROOT) if output_root is None else output_root
    topk_output_root = base_output_root / topk_label
    topk_output_root.mkdir(parents=True, exist_ok=True)
    _progress(f"setting={topk_label} output_root={topk_output_root}")
    _progress(f"artifact directory created for {topk_label}")

    original = {
        "EXPERIMENT_NAME": cap.EXPERIMENT_NAME,
        "RUN_PREFIX": cap.RUN_PREFIX,
        "TOP_K": cap.TOP_K,
        "RETRIEVE_THRESHOLD": cap.RETRIEVE_THRESHOLD,
        "UPDATE_THRESHOLD": cap.UPDATE_THRESHOLD,
        "DEFAULT_OUTPUT_ROOT": cap.DEFAULT_OUTPUT_ROOT,
        "DEFAULT_MAX_SLOTS": DEFAULT_MAX_SLOTS,
        "DEFAULT_REQUESTED_CONTEXTS": getattr(cap, "DEFAULT_REQUESTED_CONTEXTS", None),
        "parent_build_research_note": cap.fr.parent._build_research_note,
        "argv": list(sys.argv),
    }

    cap.EXPERIMENT_NAME = f"{EXPERIMENT_NAME} ({topk_label})"
    cap.RUN_PREFIX = f"{RUN_PREFIX}-{topk_label}"
    cap.TOP_K = top_k
    cap.RETRIEVE_THRESHOLD = retrieve_threshold
    cap.UPDATE_THRESHOLD = update_threshold
    cap.DEFAULT_OUTPUT_ROOT = str(topk_output_root)
    cap.DEFAULT_MAX_SLOTS = max_slots
    if requested_contexts is not None:
        cap.DEFAULT_REQUESTED_CONTEXTS = requested_contexts
    cap.fr.parent._build_research_note = _suppress_research_note
    sys.argv = [str(Path(__file__).resolve())]
    try:
        _progress(f"calling underlying FR runner for {topk_label}")
        result = cap._run_single_capacity(
            max_slots,
            requested_contexts=requested_contexts,
            output_root=topk_output_root,
        )
        _progress(f"underlying FR runner returned for {topk_label} status=0")
        if result is None:
            raise RuntimeError(f"FR runner returned no result for {topk_label}")
        run_dir = Path(result["run_dir"])
        run_config_path = run_dir / "run_config.json"
        if run_config_path.exists():
            run_config = _load_json(run_config_path)
            run_config.update(
                {
                    "max_slots": max_slots,
                    "top_k": top_k,
                    "retrieve_threshold": retrieve_threshold,
                    "update_threshold": update_threshold,
                    "retrieved_memory_to_weaver": True,
                    "memory_bank_storage_space": "weaver",
                    "mechanism": "version_b_weaver_space_bank_format_repair_topk_diagnostic",
                    "comparison_baseline_primary": CANONICAL_FR_BASELINE,
                }
            )
            _write_json(run_config_path, run_config)
            result["run_config"] = run_config
        return result
    finally:
        cap.EXPERIMENT_NAME = original["EXPERIMENT_NAME"]
        cap.RUN_PREFIX = original["RUN_PREFIX"]
        cap.TOP_K = original["TOP_K"]
        cap.RETRIEVE_THRESHOLD = original["RETRIEVE_THRESHOLD"]
        cap.UPDATE_THRESHOLD = original["UPDATE_THRESHOLD"]
        cap.DEFAULT_OUTPUT_ROOT = original["DEFAULT_OUTPUT_ROOT"]
        cap.DEFAULT_MAX_SLOTS = original["DEFAULT_MAX_SLOTS"]
        if original["DEFAULT_REQUESTED_CONTEXTS"] is not None:
            cap.DEFAULT_REQUESTED_CONTEXTS = original["DEFAULT_REQUESTED_CONTEXTS"]
        cap.fr.parent._build_research_note = original["parent_build_research_note"]
        sys.argv = original["argv"]


def _write_summary_markdown(aggregate: dict, output_root: Path) -> Path:
    rows = aggregate["rows"]
    lines = [
        "# MAB-6B-FR Top-k Diagnostic Summary",
        "",
        f"- Experiment: `{aggregate['experiment_name']}`",
        f"- Fixed max_slots: `{aggregate['max_slots']}`",
        f"- retrieve_threshold: `{aggregate['retrieve_threshold']}`",
        f"- update_threshold: `{aggregate['update_threshold']}`",
        f"- Output root: `{output_root}`",
        "",
        "| top_k | bank_off_EM | bank_on_EM | improved | regressed | output_changed | final_slots | append_insert | matched_replace | q_idx | q_latents |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['top_k']} | {row['bank_off_EM']:.2f} | {row['bank_on_EM']:.2f} | "
            f"{row['improved_count']} | {row['regressed_count']} | {row['output_changed_count']} | "
            f"{row['final_slot_counts']} | {row['append_insert_count']} | {row['matched_replace_count']} | "
            f"{row['query_turn_retrieved_indices_by_context']} | {row['query_turn_retrieved_latent_count_by_context']} |"
        )
    summary_path = output_root / "topk_diagnostic_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def main() -> int:
    args = build_parser().parse_args()
    smoke_test = bool(args.smoke_test)
    max_slots = int(args.max_slots)
    retrieve_threshold = float(args.retrieve_threshold)
    update_threshold = float(args.update_threshold)
    top_k_values = (DEFAULT_SMOKE_TOP_K,) if smoke_test else _parse_int_list(args.top_k_values)
    if smoke_test:
        top_k_values = (DEFAULT_SMOKE_TOP_K,)
    output_root = Path(args.output_root)
    if smoke_test:
        output_root = output_root / "smoke_test"
    output_root.mkdir(parents=True, exist_ok=True)

    _progress("script entry")
    _progress(
        f"constructed settings smoke_test={smoke_test} max_slots={max_slots} top_k_values={list(top_k_values)} "
        f"retrieve_threshold={retrieve_threshold} update_threshold={update_threshold}"
    )
    canonical_contexts = _load_canonical_contexts()
    results: list[dict] = []
    per_context_rows: list[dict] = []

    original = {
        "EXPERIMENT_NAME": cap.EXPERIMENT_NAME,
        "RUN_PREFIX": cap.RUN_PREFIX,
        "TOP_K": cap.TOP_K,
        "RETRIEVE_THRESHOLD": cap.RETRIEVE_THRESHOLD,
        "UPDATE_THRESHOLD": cap.UPDATE_THRESHOLD,
        "DEFAULT_OUTPUT_ROOT": cap.DEFAULT_OUTPUT_ROOT,
        "DEFAULT_MAX_SLOTS": DEFAULT_MAX_SLOTS,
        "DEFAULT_REQUESTED_CONTEXTS": getattr(cap, "DEFAULT_REQUESTED_CONTEXTS", None),
        "parent_build_research_note": cap.fr.parent._build_research_note,
    }

    cap.EXPERIMENT_NAME = EXPERIMENT_NAME
    cap.RUN_PREFIX = RUN_PREFIX
    cap.RETRIEVE_THRESHOLD = retrieve_threshold
    cap.UPDATE_THRESHOLD = update_threshold
    cap.DEFAULT_OUTPUT_ROOT = str(output_root)
    cap.DEFAULT_MAX_SLOTS = max_slots
    if smoke_test:
        cap.DEFAULT_REQUESTED_CONTEXTS = 1
    cap.fr.parent._build_research_note = _suppress_research_note

    try:
        requested_contexts = 1 if smoke_test else None
        for top_k in top_k_values:
            _progress(f"starting setting top_k={top_k}")
            run_result = _run_single_topk(
                top_k,
                max_slots=max_slots,
                retrieve_threshold=retrieve_threshold,
                update_threshold=update_threshold,
                requested_contexts=requested_contexts,
                output_root=output_root,
            )
            summary, topk_per_context_rows = _summarize_run(
                top_k=top_k,
                max_slots=max_slots,
                retrieve_threshold=retrieve_threshold,
                update_threshold=update_threshold,
                run_result=run_result,
                canonical_contexts=canonical_contexts,
            )
            results.append(summary)
            per_context_rows.extend(topk_per_context_rows)

        aggregate = {
            "experiment_name": EXPERIMENT_NAME,
            "topk_root": str(output_root),
            "canonical_reference": CANONICAL_FR_BASELINE,
            "max_slots": max_slots,
            "retrieve_threshold": retrieve_threshold,
            "update_threshold": update_threshold,
            "top_k_values": list(top_k_values),
            "rows": results,
        }

        _progress("before aggregation")
        _write_json(output_root / "topk_diagnostic_aggregate.json", aggregate)
        _write_csv(output_root / "topk_diagnostic_aggregate.csv", aggregate["rows"])
        _write_jsonl(output_root / "topk_diagnostic_per_context.jsonl", per_context_rows)
        _write_csv(output_root / "topk_diagnostic_per_context.csv", per_context_rows)
        _write_summary_markdown(aggregate, output_root)
        _progress("after aggregation")

        print(json.dumps({"topk_root": str(output_root), "aggregate_report": str(output_root / "topk_diagnostic_aggregate.json")}, ensure_ascii=False))
        return 0
    finally:
        cap.EXPERIMENT_NAME = original["EXPERIMENT_NAME"]
        cap.RUN_PREFIX = original["RUN_PREFIX"]
        cap.TOP_K = original["TOP_K"]
        cap.RETRIEVE_THRESHOLD = original["RETRIEVE_THRESHOLD"]
        cap.UPDATE_THRESHOLD = original["UPDATE_THRESHOLD"]
        cap.DEFAULT_OUTPUT_ROOT = original["DEFAULT_OUTPUT_ROOT"]
        cap.DEFAULT_MAX_SLOTS = original["DEFAULT_MAX_SLOTS"]
        if original["DEFAULT_REQUESTED_CONTEXTS"] is not None:
            cap.DEFAULT_REQUESTED_CONTEXTS = original["DEFAULT_REQUESTED_CONTEXTS"]
        cap.fr.parent._build_research_note = original["parent_build_research_note"]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        if not isinstance(exc, SystemExit) or exc.code not in (0, None):
            _log_failure_traceback(exc)
        raise
