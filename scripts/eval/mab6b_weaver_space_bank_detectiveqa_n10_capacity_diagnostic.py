"""MAB-6B-FR capacity diagnostic for Weaver-space memory bank on DetectiveQA n=10."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path
import sys
import traceback
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import mab6b_weaver_space_bank_detectiveqa_n10_format_repair as fr

EXPERIMENT_NAME = "MAB-6B-FR capacity diagnostic: Weaver-space bank + final-query format repair"
RUN_PREFIX = "detectiveqa-version-b-weaver-space-bank-format-repair-capacity-diagnostic-n10"
DEFAULT_OUTPUT_ROOT = "outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_capacity_diagnostic"
CANONICAL_FR_BASELINE = (
    "outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_format_repair/"
    "20260626T014628Z-detectiveqa-version-b-weaver-space-bank-format-repair-n10"
)
SWEEP_MAX_SLOTS = (8, 16, 32)
RETRIEVE_THRESHOLD = 0.03
UPDATE_THRESHOLD = 0.08
TOP_K = 1
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


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _progress(message: str) -> None:
    print(f"[capacity-diagnostic] {message}", flush=True)


def _log_failure_traceback(exc: BaseException) -> None:
    DEBUG_FAILURE_TRACEBACK.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    DEBUG_FAILURE_TRACEBACK.write_text(rendered, encoding="utf-8")
    sys.stderr.write(rendered)
    sys.stderr.flush()
    sys.stdout.flush()


def _suppress_research_note(*args, **kwargs):
    output_dir = kwargs.get("output_dir")
    if output_dir is not None:
        return Path(output_dir) / "suppressed_research_note.md"
    return Path("suppressed_research_note.md")


def _capacity_label(max_slots: int) -> str:
    return f"cap{max_slots}_rt{RETRIEVE_THRESHOLD:.2f}_ut{UPDATE_THRESHOLD:.2f}_topk{TOP_K}"


def _load_canonical_contexts() -> list[dict] | None:
    paired = Path(CANONICAL_FR_BASELINE) / "paired_results.json"
    if not paired.exists():
        return None
    data = _load_json(paired)
    return data.get("contexts")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def _score_exact_match(*, mab_python: str, mab_repo: str, prediction: str | None, gold_answer: Any, dataset_config: dict | None) -> int:
    if prediction is None:
        return 0
    answer = gold_answer if isinstance(gold_answer, list) else [gold_answer]
    request = {
        "prediction": prediction,
        "gold_answers": answer,
        "dataset_config": dataset_config or {"sub_dataset": "factconsolidation_sh_6k"},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        request_path = Path(tmpdir) / "score_request.json"
        output_path = Path(tmpdir) / "score_output.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        subprocess.run(
            [
                mab_python,
                str(REPO_ROOT / "scripts" / "eval" / "mab2_mab_bridge.py"),
                "score",
                "--mab-repo",
                mab_repo,
                "--output",
                str(output_path),
                "--input",
                str(request_path),
            ],
            check=True,
            env={**os.environ},
        )
        metrics = _load_json(output_path)["metrics"]
    return int(bool(metrics.get("exact_match")))


def _infer_write_counts(context: dict) -> tuple[int, int, int]:
    write_count = int(context.get("bank_on_write_count") or 0)
    final_slot_count = int(context.get("bank_slot_count_final_before_reset") or 0)
    capacity_evict_count = int(context.get("bank_on_capacity_evict_count") or 0)
    append_insert_count = min(write_count, final_slot_count)
    matched_replace_count = max(write_count - append_insert_count - capacity_evict_count, 0)
    return append_insert_count, matched_replace_count, capacity_evict_count


def _infer_query_turn_retrieved_latent_count(context: dict) -> int:
    retrieval_count = int(context.get("bank_on_retrieval_count") or 0)
    total_retrieved = int(context.get("bank_on_retrieved_latent_count") or 0)
    if retrieval_count <= 0:
        return 0
    if total_retrieved % retrieval_count == 0:
        return total_retrieved // retrieval_count
    return int(context.get("query_turn_retrieved_latent_count") or 0)


def _classify_prediction(text: str | None) -> str | None:
    return fr._classify_output_format(text)


def _summarize_diagnostics_rows(
    *,
    capacity_label: str,
    max_slots: int,
    diagnostics_rows: list[dict],
    run_config: dict,
    canonical_contexts: list[dict] | None,
) -> tuple[dict, list[dict]]:
    per_context_rows: list[dict] = []
    exact_off = 0
    exact_on = 0
    improved_count = 0
    regressed_count = 0
    output_changed_count = 0
    bank_off_format_failures = 0
    bank_on_format_failures = 0
    final_slot_counts: list[int] = []
    append_insert_total = 0
    matched_replace_total = 0
    capacity_evict_total = 0
    query_write_total = 0
    query_write_attempt_total = 0
    cross_context_leakage_detected = False
    retrieved_latents_enter_weaver = False
    raw_retrieved_latents_enter_reasoner = False
    query_turn_retrieved_indices_by_context: list[list[int]] = []
    query_turn_retrieved_latent_count_by_context: list[int] = []
    bank_on_retrieved_latent_count_by_context: list[int] = []
    bank_on_write_action_counts = {"insert": 0, "replace_matched": 0, "capacity_evict": 0}
    bank_on_update_reason_counts = {"empty_bank": 0, "matched_thread": 0}

    for idx, row in enumerate(diagnostics_rows):
        bank_off_exact_match = _score_exact_match(
            mab_python=run_config["mab_python"],
            mab_repo=run_config["mab_repo"],
            prediction=row.get("bank_off_prediction"),
            gold_answer=row.get("gold_answer"),
            dataset_config=run_config.get("dataset_config"),
        )
        bank_on_exact_match = _score_exact_match(
            mab_python=run_config["mab_python"],
            mab_repo=run_config["mab_repo"],
            prediction=row.get("bank_on_prediction"),
            gold_answer=row.get("gold_answer"),
            dataset_config=run_config.get("dataset_config"),
        )
        bank_off_format_status = _classify_prediction(row.get("bank_off_prediction"))
        bank_on_format_status = _classify_prediction(row.get("bank_on_prediction"))
        output_changed = row.get("bank_off_prediction") != row.get("bank_on_prediction")
        improved = int(bank_on_exact_match > bank_off_exact_match)
        regressed = int(bank_on_exact_match < bank_off_exact_match)
        append_insert_count, matched_replace_count, capacity_evict_count = _infer_write_counts(row)
        query_turn_retrieved_indices = list((row.get("retrieved_indices_by_turn") or [[]])[-1])
        query_turn_retrieved_latent_count = _infer_query_turn_retrieved_latent_count(row)
        bank_on_retrieved_latent_count = int(row.get("bank_on_retrieved_latent_count") or 0)

        canonical_row = canonical_contexts[idx] if canonical_contexts and idx < len(canonical_contexts) else None
        output_changed_vs_canonical = None
        if canonical_row is not None and row.get("bank_on_prediction") is not None:
            output_changed_vs_canonical = row["bank_on_prediction"] != canonical_row.get("bank_on_prediction")

        per_context_rows.append(
            {
                "capacity_label": capacity_label,
                "max_slots": max_slots,
                "retrieve_threshold": RETRIEVE_THRESHOLD,
                "update_threshold": UPDATE_THRESHOLD,
                "top_k": TOP_K,
                "context_index": row.get("context_index"),
                "context_id": row.get("context_id"),
                "gold_answer": row.get("gold_answer"),
                "bank_off_prediction": row.get("bank_off_prediction"),
                "bank_off_EM": bank_off_exact_match,
                "bank_on_prediction": row.get("bank_on_prediction"),
                "bank_on_EM": bank_on_exact_match,
                "bank_off_exact_match": bank_off_exact_match,
                "bank_on_exact_match": bank_on_exact_match,
                "output_changed": output_changed,
                "improved": improved,
                "regressed": regressed,
                "output_changed_vs_canonical_frf": output_changed_vs_canonical,
                "final_slot_count": int(row.get("bank_slot_count_final_before_reset") or 0),
                "query_turn_retrieved_indices": query_turn_retrieved_indices,
                "query_turn_retrieved_latent_count": query_turn_retrieved_latent_count,
                "bank_on_write_action_counts": {
                    "insert": append_insert_count,
                    "replace_matched": matched_replace_count,
                    "capacity_evict": capacity_evict_count,
                },
                "bank_on_update_reason_counts": None,
                "error_or_stop_reason": row.get("error_or_stop_reason"),
                "write_action_counts": {
                    "insert": append_insert_count,
                    "replace_matched": matched_replace_count,
                    "capacity_evict": capacity_evict_count,
                },
                "append_insert_count": append_insert_count,
                "matched_replace_count": matched_replace_count,
                "capacity_evict_count": capacity_evict_count,
                "format_failure_bank_off": bank_off_format_status != "clean_option",
                "format_failure_bank_on": bank_on_format_status != "clean_option",
                "query_write_count": int(row.get("query_write_count") or 0),
                "query_write_attempt_count": int(row.get("query_write_attempt_count") or 0),
                "cross_context_leakage_detected": bool(row.get("cross_context_leakage_detected")),
                "retrieved_latents_enter_weaver": bool(run_config.get("retrieved_memory_to_weaver")),
                "raw_retrieved_latents_enter_reasoner": False,
            }
        )

        exact_off += bank_off_exact_match
        exact_on += bank_on_exact_match
        improved_count += improved
        regressed_count += regressed
        output_changed_count += int(output_changed)
        bank_off_format_failures += int(bank_off_format_status != "clean_option")
        bank_on_format_failures += int(bank_on_format_status != "clean_option")
        final_slot_counts.append(int(row.get("bank_slot_count_final_before_reset") or 0))
        append_insert_total += append_insert_count
        matched_replace_total += matched_replace_count
        capacity_evict_total += capacity_evict_count
        query_write_total += int(row.get("query_write_count") or 0)
        query_write_attempt_total += int(row.get("query_write_attempt_count") or 0)
        cross_context_leakage_detected = cross_context_leakage_detected or bool(row.get("cross_context_leakage_detected"))
        retrieved_latents_enter_weaver = retrieved_latents_enter_weaver or bool(run_config.get("retrieved_memory_to_weaver"))
        query_turn_retrieved_indices_by_context.append(query_turn_retrieved_indices)
        query_turn_retrieved_latent_count_by_context.append(query_turn_retrieved_latent_count)
        bank_on_retrieved_latent_count_by_context.append(bank_on_retrieved_latent_count)
        bank_on_write_action_counts["insert"] += append_insert_count
        bank_on_write_action_counts["replace_matched"] += matched_replace_count
        bank_on_write_action_counts["capacity_evict"] += capacity_evict_count
        if append_insert_count > 0:
            bank_on_update_reason_counts["empty_bank"] += 1
        if matched_replace_count > 0:
            bank_on_update_reason_counts["matched_thread"] += 1

    valid_count = len(diagnostics_rows)
    summary = {
        "capacity_label": capacity_label,
        "max_slots": max_slots,
        "top_k": TOP_K,
        "retrieve_threshold": RETRIEVE_THRESHOLD,
        "update_threshold": UPDATE_THRESHOLD,
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
        "cross_context_leakage_detected": cross_context_leakage_detected,
        "retrieved_latents_enter_weaver": bool(run_config.get("retrieved_memory_to_weaver")) if valid_count else False,
        "raw_retrieved_latents_enter_reasoner": False,
        "query_turn_retrieved_indices_by_context": query_turn_retrieved_indices_by_context,
        "query_turn_retrieved_latent_count_by_context": query_turn_retrieved_latent_count_by_context,
        "bank_on_retrieved_latent_count_by_context": bank_on_retrieved_latent_count_by_context,
        "format_failure_counts_bank_off": bank_off_format_failures,
        "format_failure_counts_bank_on": bank_on_format_failures,
        "write_action_counts": bank_on_write_action_counts,
        "update_reason_counts": bank_on_update_reason_counts,
    }
    return summary, per_context_rows


def _run_single_capacity(
    max_slots: int,
    *,
    requested_contexts: int | None = None,
    output_root: Path | None = None,
) -> dict:
    capacity_label = _capacity_label(max_slots)
    base_output_root = Path(DEFAULT_OUTPUT_ROOT) if output_root is None else output_root
    capacity_output_root = base_output_root / capacity_label
    capacity_output_root.mkdir(parents=True, exist_ok=True)
    _progress(f"setting={capacity_label} output_root={capacity_output_root}")
    _progress(f"artifact directory created for {capacity_label}")

    original = {
        "EXPERIMENT_NAME": fr.EXPERIMENT_NAME,
        "RUN_PREFIX": fr.RUN_PREFIX,
        "DEFAULT_OUTPUT_ROOT": fr.DEFAULT_OUTPUT_ROOT,
        "parent_DEFAULT_MAX_SLOTS": fr.parent.DEFAULT_MAX_SLOTS,
        "parent_DEFAULT_RETRIEVE_THRESHOLD": fr.parent.DEFAULT_RETRIEVE_THRESHOLD,
        "parent_DEFAULT_UPDATE_THRESHOLD": fr.parent.DEFAULT_UPDATE_THRESHOLD,
        "parent_DEFAULT_TOP_K": fr.parent.DEFAULT_TOP_K,
        "parent_DEFAULT_REQUESTED_CONTEXTS": fr.parent.DEFAULT_REQUESTED_CONTEXTS,
        "parent_build_research_note": fr.parent._build_research_note,
        "argv": list(sys.argv),
    }

    fr.EXPERIMENT_NAME = f"{EXPERIMENT_NAME} ({capacity_label})"
    fr.RUN_PREFIX = f"{RUN_PREFIX}-{capacity_label}"
    fr.DEFAULT_OUTPUT_ROOT = str(capacity_output_root)
    fr.parent.DEFAULT_MAX_SLOTS = max_slots
    fr.parent.DEFAULT_RETRIEVE_THRESHOLD = RETRIEVE_THRESHOLD
    fr.parent.DEFAULT_UPDATE_THRESHOLD = UPDATE_THRESHOLD
    fr.parent.DEFAULT_TOP_K = TOP_K
    if requested_contexts is not None:
        fr.parent.DEFAULT_REQUESTED_CONTEXTS = requested_contexts
    fr.parent._build_research_note = _suppress_research_note
    sys.argv = [str(Path(__file__).resolve())]
    try:
        _progress(f"calling underlying FR runner for {capacity_label}")
        result = fr.main()
        _progress(f"underlying FR runner returned for {capacity_label} status={result}")
        if result != 0:
            raise RuntimeError(f"fr.main() returned non-zero status {result} for {capacity_label}")
        run_dirs = sorted(
            [path for path in capacity_output_root.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
        )
        if not run_dirs:
            raise RuntimeError(f"No run directory created under {capacity_output_root}")
        run_dir = run_dirs[-1]
        paired = _load_json(run_dir / "paired_results.json")
        manifest = _load_json(run_dir / "manifest.json")
        run_config = _load_json(run_dir / "run_config.json")
        run_config.update(
            {
                "retrieve_threshold": RETRIEVE_THRESHOLD,
                "update_threshold": UPDATE_THRESHOLD,
                "max_slots": max_slots,
                "top_k": TOP_K,
                "retrieved_memory_to_weaver": True,
                "memory_bank_storage_space": "weaver",
                "mechanism": "version_b_weaver_space_bank_format_repair_capacity_diagnostic",
                "comparison_baseline_primary": CANONICAL_FR_BASELINE,
            }
        )
        _write_json(run_dir / "run_config.json", run_config)
        return {
            "capacity_label": capacity_label,
            "max_slots": max_slots,
            "top_k": TOP_K,
            "retrieve_threshold": RETRIEVE_THRESHOLD,
            "update_threshold": UPDATE_THRESHOLD,
            "output_root": str(capacity_output_root),
            "run_dir": str(run_dir),
            "paired_results": paired,
            "manifest": manifest,
            "run_config": run_config,
        }
    finally:
        fr.EXPERIMENT_NAME = original["EXPERIMENT_NAME"]
        fr.RUN_PREFIX = original["RUN_PREFIX"]
        fr.DEFAULT_OUTPUT_ROOT = original["DEFAULT_OUTPUT_ROOT"]
        fr.parent.DEFAULT_MAX_SLOTS = original["parent_DEFAULT_MAX_SLOTS"]
        fr.parent.DEFAULT_RETRIEVE_THRESHOLD = original["parent_DEFAULT_RETRIEVE_THRESHOLD"]
        fr.parent.DEFAULT_UPDATE_THRESHOLD = original["parent_DEFAULT_UPDATE_THRESHOLD"]
        fr.parent.DEFAULT_TOP_K = original["parent_DEFAULT_TOP_K"]
        fr.parent.DEFAULT_REQUESTED_CONTEXTS = original["parent_DEFAULT_REQUESTED_CONTEXTS"]
        fr.parent._build_research_note = original["parent_build_research_note"]
        sys.argv = original["argv"]


def _aggregate_capacities(run_result: dict, canonical_contexts: list[dict] | None) -> dict:
    paired = run_result["paired_results"]
    contexts = paired["contexts"]
    summary = paired["summary"]
    capacity_label = run_result["capacity_label"]
    per_context_rows = []
    for idx, row in enumerate(contexts):
        canonical_row = canonical_contexts[idx] if canonical_contexts and idx < len(canonical_contexts) else None
        output_changed_vs_canonical = None
        if canonical_row is not None and row.get("bank_on_prediction") is not None:
            output_changed_vs_canonical = row["bank_on_prediction"] != canonical_row.get("bank_on_prediction")
        append_insert_count, matched_replace_count, capacity_evict_count = _infer_write_counts(row)
        per_context_rows.append(
            {
                "capacity_label": capacity_label,
                "max_slots": run_result["max_slots"],
                "top_k": run_result["top_k"],
                "retrieve_threshold": run_result["retrieve_threshold"],
                "update_threshold": run_result["update_threshold"],
                "context_index": row.get("context_index"),
                "context_id": row.get("context_id"),
                "gold_answer": row.get("gold_answer"),
                "bank_off_prediction": row.get("bank_off_prediction"),
                "bank_off_EM": row.get("bank_off_exact_match"),
                "bank_on_prediction": row.get("bank_on_prediction"),
                "bank_on_EM": row.get("bank_on_exact_match"),
                "improved": row.get("improved"),
                "regressed": row.get("regressed"),
                "output_changed": row.get("output_changed"),
                "final_slot_count": row.get("bank_slot_count_final_before_reset"),
                "query_turn_retrieved_indices": row.get("query_turn_retrieved_indices", []),
                "query_turn_retrieved_latent_count": _infer_query_turn_retrieved_latent_count(row),
                "bank_on_write_action_counts": {
                    "insert": append_insert_count,
                    "replace_matched": matched_replace_count,
                    "capacity_evict": capacity_evict_count,
                },
                "bank_on_update_reason_counts": None,
                "error_or_stop_reason": row.get("error_or_stop_reason"),
                "bank_on_exact_match_vs_canonical": output_changed_vs_canonical,
            }
        )
    bank_off_format_failures = sum(1 for row in contexts if row.get("bank_off_primary_output_format_status") != "clean_option")
    bank_on_format_failures = sum(1 for row in contexts if row.get("bank_on_primary_output_format_status") != "clean_option")
    return {
        "capacity_label": capacity_label,
        "max_slots": run_result["max_slots"],
        "top_k": run_result["top_k"],
        "retrieve_threshold": run_result["retrieve_threshold"],
        "update_threshold": run_result["update_threshold"],
        "bank_off_EM": summary.get("compressed_bank_off_accuracy"),
        "bank_on_EM": summary.get("compressed_bank_on_accuracy"),
        "improved_count": summary.get("num_improved"),
        "regressed_count": summary.get("num_regressed"),
        "output_changed_count": summary.get("num_output_changed"),
        "final_slot_counts": summary.get("final_slot_counts"),
        "mean_final_slot_count": summary.get("mean_final_slot_count"),
        "append_insert_count": summary.get("append_insert_count"),
        "matched_replace_count": summary.get("matched_replace_count"),
        "capacity_evict_count": summary.get("capacity_evict_count"),
        "query_write_count": summary.get("query_write_count"),
        "query_write_attempt_count": summary.get("query_write_attempt_count"),
        "cross_context_leakage_detected": summary.get("cross_context_leakage_detected"),
        "retrieved_latents_enter_weaver": summary.get("retrieved_latents_enter_weaver"),
        "raw_retrieved_latents_enter_reasoner": summary.get("raw_retrieved_latents_enter_reasoner"),
        "query_turn_retrieved_indices_by_context": summary.get("query_turn_retrieved_indices_by_context"),
        "query_turn_retrieved_latent_count_by_context": summary.get("query_turn_retrieved_latent_count_by_context"),
        "bank_on_retrieved_latent_count_by_context": summary.get("bank_on_retrieved_latent_count_by_context"),
        "format_failure_counts_bank_off": bank_off_format_failures,
        "format_failure_counts_bank_on": bank_on_format_failures,
        "write_action_counts": summary.get("write_action_counts"),
        "update_reason_counts": summary.get("update_reason_counts"),
        "per_context_rows": per_context_rows,
        "run_dir": run_result["run_dir"],
        "output_root": run_result["output_root"],
    }


def recompute_aggregate_from_existing_artifacts(capacity_root: Path | str = DEFAULT_OUTPUT_ROOT) -> dict:
    capacity_root = Path(capacity_root)
    canonical_contexts = _load_canonical_contexts()
    rows: list[dict] = []
    per_context_rows: list[dict] = []
    for cap_dir in sorted(p for p in capacity_root.iterdir() if p.is_dir() and p.name.startswith("cap")):
        run_dirs = sorted((p for p in cap_dir.iterdir() if p.is_dir()), key=lambda path: path.stat().st_mtime)
        if not run_dirs:
            continue
        run_dir = run_dirs[-1]
        run_config = _load_json(run_dir / "run_config.json")
        diagnostics_rows = _load_jsonl(run_dir / "diagnostics.jsonl")
        summary, cap_per_context_rows = _summarize_diagnostics_rows(
            capacity_label=cap_dir.name,
            max_slots=int(run_config.get("max_slots", 0)),
            diagnostics_rows=diagnostics_rows,
            run_config=run_config,
            canonical_contexts=canonical_contexts,
        )
        summary.update(
            {
                "canonical_reference_available": canonical_contexts is not None,
                "canonical_bank_off_EM": None if canonical_contexts is None else sum(
                    _score_exact_match(
                        mab_python=run_config["mab_python"],
                        mab_repo=run_config["mab_repo"],
                        prediction=row.get("bank_off_prediction"),
                        gold_answer=row.get("gold_answer"),
                        dataset_config=run_config.get("dataset_config"),
                    )
                    for row in canonical_contexts
                ) / len(canonical_contexts),
                "canonical_bank_on_EM": None if canonical_contexts is None else sum(
                    _score_exact_match(
                        mab_python=run_config["mab_python"],
                        mab_repo=run_config["mab_repo"],
                        prediction=row.get("bank_on_prediction"),
                        gold_answer=row.get("gold_answer"),
                        dataset_config=run_config.get("dataset_config"),
                    )
                    for row in canonical_contexts
                ) / len(canonical_contexts),
                "run_dir": str(run_dir),
                "output_root": str(cap_dir),
            }
        )
        rows.append(summary)
        per_context_rows.extend(cap_per_context_rows)
    aggregate = {
        "experiment_name": EXPERIMENT_NAME,
        "capacity_root": str(capacity_root),
        "canonical_reference": CANONICAL_FR_BASELINE,
        "retrieve_threshold": RETRIEVE_THRESHOLD,
        "update_threshold": UPDATE_THRESHOLD,
        "top_k": TOP_K,
        "max_slots_values": list(SWEEP_MAX_SLOTS),
        "rows": rows,
    }
    _write_json(capacity_root / "capacity_diagnostic_aggregate.json", aggregate)
    _write_csv(capacity_root / "capacity_diagnostic_aggregate.csv", aggregate["rows"])
    _write_jsonl(capacity_root / "capacity_diagnostic_per_context.jsonl", per_context_rows)
    _write_csv(capacity_root / "capacity_diagnostic_per_context.csv", per_context_rows)
    return aggregate


def _max_slots_values(smoke_test: bool) -> tuple[int, ...]:
    return (SWEEP_MAX_SLOTS[0],) if smoke_test else SWEEP_MAX_SLOTS


def _output_root_for_mode(smoke_test: bool) -> Path:
    root = Path(DEFAULT_OUTPUT_ROOT)
    return root / "smoke_test" if smoke_test else root


def main() -> int:
    args = build_parser().parse_args()
    smoke_test = bool(args.smoke_test)
    capacity_values = _max_slots_values(smoke_test)
    requested_contexts = 1 if smoke_test else None
    capacity_root = _output_root_for_mode(smoke_test)
    capacity_root.mkdir(parents=True, exist_ok=True)
    _progress("script entry")
    _progress(
        f"constructed settings smoke_test={smoke_test} max_slots={list(capacity_values)} "
        f"retrieve_threshold={RETRIEVE_THRESHOLD} update_threshold={UPDATE_THRESHOLD} top_k={TOP_K}"
    )
    canonical_contexts = _load_canonical_contexts()
    results = []
    per_context_rows: list[dict] = []
    for max_slots in capacity_values:
        _progress(f"starting setting cap{max_slots}")
        run_result = _run_single_capacity(
            max_slots,
            requested_contexts=requested_contexts,
            output_root=capacity_root,
        )
        capacity_summary = _aggregate_capacities(run_result, canonical_contexts)
        results.append(capacity_summary)
        per_context_rows.extend(capacity_summary["per_context_rows"])

    aggregate = {
        "experiment_name": EXPERIMENT_NAME,
        "capacity_root": str(capacity_root),
        "canonical_reference": CANONICAL_FR_BASELINE,
        "retrieve_threshold": RETRIEVE_THRESHOLD,
        "update_threshold": UPDATE_THRESHOLD,
        "top_k": TOP_K,
        "max_slots_values": list(capacity_values),
        "rows": [
            {
                key: value
                for key, value in row.items()
                if key not in {"per_context_rows", "paired_results"}
            }
            for row in results
        ],
    }

    _progress("before aggregation")
    _write_json(capacity_root / "capacity_diagnostic_aggregate.json", aggregate)
    _write_csv(capacity_root / "capacity_diagnostic_aggregate.csv", aggregate["rows"])
    _write_jsonl(capacity_root / "capacity_diagnostic_per_context.jsonl", per_context_rows)
    _write_csv(capacity_root / "capacity_diagnostic_per_context.csv", per_context_rows)

    summary_lines = [
        "# MAB-6B-FR Capacity Diagnostic Summary",
        "",
        "| max_slots | bank_off_EM | bank_on_EM | improved | regressed | output_changed | final_slots | append_insert | matched_replace | q_idx | q_latents |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|",
    ]
    for row in aggregate["rows"]:
        summary_lines.append(
            f"| {row['max_slots']} | {row['bank_off_EM']:.2f} | {row['bank_on_EM']:.2f} | "
            f"{row['improved_count']} | {row['regressed_count']} | {row['output_changed_count']} | "
            f"{row['final_slot_counts']} | {row['append_insert_count']} | {row['matched_replace_count']} | "
            f"{row['query_turn_retrieved_indices_by_context']} | {row['query_turn_retrieved_latent_count_by_context']} |"
        )
    (capacity_root / "capacity_diagnostic_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    _progress("after aggregation")

    print(json.dumps({"capacity_root": str(capacity_root), "aggregate_report": str(capacity_root / "capacity_diagnostic_aggregate.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        if not isinstance(exc, SystemExit) or exc.code not in (0, None):
            _log_failure_traceback(exc)
        raise
