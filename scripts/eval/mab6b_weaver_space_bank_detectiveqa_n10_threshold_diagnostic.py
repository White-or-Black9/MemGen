"""MAB-6B-FR threshold diagnostic for Weaver-space memory bank on DetectiveQA n=10."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import mab6b_weaver_space_bank_detectiveqa_n10_format_repair as fr

EXPERIMENT_NAME = "MAB-6B-FR threshold diagnostic: Weaver-space bank + final-query format repair"
RUN_PREFIX = "detectiveqa-version-b-weaver-space-bank-format-repair-threshold-diagnostic-n10"
DEFAULT_OUTPUT_ROOT = "outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_threshold_diagnostic"
CANONICAL_FR_BASELINE = (
    "outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_format_repair/"
    "20260626T014628Z-detectiveqa-version-b-weaver-space-bank-format-repair-n10"
)
SWEEP_UPDATE_THRESHOLDS = (0.05, 0.08, 0.10, 0.12)
RETRIEVE_THRESHOLD = 0.03
FORMAT_FAILURE_CATEGORIES = {"code_leak", "json_leak", "reasoning_sprawl", "refusal", "language_drift", "other"}


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


def _suppress_research_note(*args, **kwargs):
    output_dir = kwargs.get("output_dir")
    if output_dir is not None:
        return Path(output_dir) / "suppressed_research_note.md"
    return Path("suppressed_research_note.md")


def _threshold_label(retrieve_threshold: float, update_threshold: float) -> str:
    return f"rt{retrieve_threshold:.2f}_ut{update_threshold:.2f}"


def _load_canonical_contexts() -> list[dict] | None:
    paired = Path(CANONICAL_FR_BASELINE) / "paired_results.json"
    if not paired.exists():
        return None
    data = _load_json(paired)
    return data.get("contexts")


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
    threshold_label: str,
    retrieve_threshold: float,
    update_threshold: float,
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
                "threshold_label": threshold_label,
                "retrieve_threshold": retrieve_threshold,
                "update_threshold": update_threshold,
                "context_index": row.get("context_index"),
                "context_id": row.get("context_id"),
                "gold_answer": row.get("gold_answer"),
                "bank_off_prediction": row.get("bank_off_prediction"),
                "bank_off_score": bank_off_exact_match,
                "bank_on_prediction": row.get("bank_on_prediction"),
                "bank_on_score": bank_on_exact_match,
                "bank_off_exact_match": bank_off_exact_match,
                "bank_on_exact_match": bank_on_exact_match,
                "bank_off_format_status": bank_off_format_status,
                "bank_on_format_status": bank_on_format_status,
                "output_changed": output_changed,
                "improved": improved,
                "regressed": regressed,
                "output_changed_vs_canonical_frf": output_changed_vs_canonical,
                "final_slot_count": int(row.get("bank_slot_count_final_before_reset") or 0),
                "write_action_counts": {
                    "insert": append_insert_count,
                    "replace_matched": matched_replace_count,
                    "capacity_evict": capacity_evict_count,
                },
                "append_insert_count": append_insert_count,
                "matched_replace_count": matched_replace_count,
                "capacity_evict_count": capacity_evict_count,
                "query_turn_retrieved_indices": query_turn_retrieved_indices,
                "query_turn_retrieved_latent_count": query_turn_retrieved_latent_count,
                "bank_on_retrieved_latent_count": bank_on_retrieved_latent_count,
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
        retrieved_latents_enter_weaver = retrieved_latents_enter_weaver or bool(row.get("retrieved_latents_enter_weaver"))
        raw_retrieved_latents_enter_reasoner = raw_retrieved_latents_enter_reasoner or False
        query_turn_retrieved_indices_by_context.append(query_turn_retrieved_indices)
        query_turn_retrieved_latent_count_by_context.append(query_turn_retrieved_latent_count)
        bank_on_retrieved_latent_count_by_context.append(bank_on_retrieved_latent_count)

    valid_count = len(diagnostics_rows)
    return (
        {
            "threshold_label": threshold_label,
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
            "cross_context_leakage_detected": cross_context_leakage_detected,
            "retrieved_latents_enter_weaver": bool(run_config.get("retrieved_memory_to_weaver")) if valid_count else False,
            "raw_retrieved_latents_enter_reasoner": False,
            "query_turn_retrieved_indices_by_context": query_turn_retrieved_indices_by_context,
            "query_turn_retrieved_latent_count_by_context": query_turn_retrieved_latent_count_by_context,
            "bank_on_retrieved_latent_count_by_context": bank_on_retrieved_latent_count_by_context,
            "format_failure_counts_bank_off": bank_off_format_failures,
            "format_failure_counts_bank_on": bank_on_format_failures,
        },
        per_context_rows,
    )


def recompute_aggregate_from_existing_artifacts(sweep_root: Path | str = DEFAULT_OUTPUT_ROOT) -> dict:
    sweep_root = Path(sweep_root)
    canonical_contexts = _load_canonical_contexts()
    rows: list[dict] = []
    per_context_rows: list[dict] = []

    for threshold_dir in sorted(p for p in sweep_root.iterdir() if p.is_dir() and p.name.startswith("rt")):
        run_dirs = sorted((p for p in threshold_dir.iterdir() if p.is_dir()), key=lambda path: path.stat().st_mtime)
        if not run_dirs:
            continue
        run_dir = run_dirs[-1]
        run_config = _load_json(run_dir / "run_config.json")
        diagnostics_rows = _load_jsonl(run_dir / "diagnostics.jsonl")
        threshold_summary, threshold_per_context_rows = _summarize_diagnostics_rows(
            threshold_label=threshold_dir.name,
            retrieve_threshold=float(run_config.get("retrieve_threshold", RETRIEVE_THRESHOLD)),
            update_threshold=float(run_config.get("update_threshold", 0.0)),
            diagnostics_rows=diagnostics_rows,
            run_config=run_config,
            canonical_contexts=canonical_contexts,
        )
        threshold_summary.update(
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
                "output_root": str(threshold_dir),
            }
        )
        rows.append(threshold_summary)
        per_context_rows.extend(threshold_per_context_rows)

    aggregate = {
        "experiment_name": EXPERIMENT_NAME,
        "sweep_root": str(sweep_root),
        "canonical_reference": CANONICAL_FR_BASELINE,
        "retrieve_threshold": RETRIEVE_THRESHOLD,
        "thresholds": list(SWEEP_UPDATE_THRESHOLDS),
        "rows": rows,
    }

    _write_json(sweep_root / "threshold_diagnostic_aggregate.json", aggregate)
    _write_csv(sweep_root / "threshold_diagnostic_aggregate.csv", aggregate["rows"])
    _write_jsonl(sweep_root / "threshold_diagnostic_per_context.jsonl", per_context_rows)
    _write_csv(sweep_root / "threshold_diagnostic_per_context.csv", per_context_rows)

    return aggregate


def _run_single_threshold(update_threshold: float) -> dict:
    threshold_label = _threshold_label(RETRIEVE_THRESHOLD, update_threshold)
    threshold_output_root = Path(DEFAULT_OUTPUT_ROOT) / threshold_label
    threshold_output_root.mkdir(parents=True, exist_ok=True)

    original = {
        "EXPERIMENT_NAME": fr.EXPERIMENT_NAME,
        "RUN_PREFIX": fr.RUN_PREFIX,
        "DEFAULT_OUTPUT_ROOT": fr.DEFAULT_OUTPUT_ROOT,
        "parent_DEFAULT_UPDATE_THRESHOLD": fr.parent.DEFAULT_UPDATE_THRESHOLD,
        "parent_DEFAULT_RETRIEVE_THRESHOLD": fr.parent.DEFAULT_RETRIEVE_THRESHOLD,
        "parent_build_research_note": fr.parent._build_research_note,
    }

    fr.EXPERIMENT_NAME = f"{EXPERIMENT_NAME} ({threshold_label})"
    fr.RUN_PREFIX = f"{RUN_PREFIX}-{threshold_label}"
    fr.DEFAULT_OUTPUT_ROOT = str(threshold_output_root)
    fr.parent.DEFAULT_UPDATE_THRESHOLD = update_threshold
    fr.parent.DEFAULT_RETRIEVE_THRESHOLD = RETRIEVE_THRESHOLD
    fr.parent._build_research_note = _suppress_research_note
    try:
        result = fr.main()
        if result != 0:
            raise RuntimeError(f"fr.main() returned non-zero status {result} for {threshold_label}")
        run_dirs = sorted(
            [path for path in threshold_output_root.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
        )
        if not run_dirs:
            raise RuntimeError(f"No run directory created under {threshold_output_root}")
        run_dir = run_dirs[-1]
        paired = _load_json(run_dir / "paired_results.json")
        manifest = _load_json(run_dir / "manifest.json")
        run_config = _load_json(run_dir / "run_config.json")
        return {
            "threshold_label": threshold_label,
            "retrieve_threshold": RETRIEVE_THRESHOLD,
            "update_threshold": update_threshold,
            "output_root": str(threshold_output_root),
            "run_dir": str(run_dir),
            "paired_results": paired,
            "manifest": manifest,
            "run_config": run_config,
        }
    finally:
        fr.EXPERIMENT_NAME = original["EXPERIMENT_NAME"]
        fr.RUN_PREFIX = original["RUN_PREFIX"]
        fr.DEFAULT_OUTPUT_ROOT = original["DEFAULT_OUTPUT_ROOT"]
        fr.parent.DEFAULT_UPDATE_THRESHOLD = original["parent_DEFAULT_UPDATE_THRESHOLD"]
        fr.parent.DEFAULT_RETRIEVE_THRESHOLD = original["parent_DEFAULT_RETRIEVE_THRESHOLD"]
        fr.parent._build_research_note = original["parent_build_research_note"]


def _format_failure_count(rows: list[dict], key: str) -> int:
    return sum(1 for row in rows if row.get(key) != "clean_option")


def _threshold_summary(run_result: dict, canonical_contexts: list[dict] | None) -> dict:
    paired = run_result["paired_results"]
    summary = paired["summary"]
    contexts = paired["contexts"]
    threshold_label = run_result["threshold_label"]

    per_context_rows = []
    output_changed_vs_canonical_count = 0
    for idx, row in enumerate(contexts):
        canonical_row = canonical_contexts[idx] if canonical_contexts and idx < len(canonical_contexts) else None
        output_changed_vs_canonical = None
        if canonical_row is not None and row.get("bank_on_prediction") is not None:
            output_changed_vs_canonical = row["bank_on_prediction"] != canonical_row.get("bank_on_prediction")
            output_changed_vs_canonical_count += int(output_changed_vs_canonical)
        per_context_rows.append(
            {
                "threshold_label": threshold_label,
                "retrieve_threshold": run_result["retrieve_threshold"],
                "update_threshold": run_result["update_threshold"],
                "context_index": row.get("context_index"),
                "context_id": row.get("context_id"),
                "gold_answer": row.get("gold_answer"),
                "bank_off_prediction": row.get("bank_off_prediction"),
                "bank_off_score": row.get("bank_off_exact_match"),
                "bank_on_prediction": row.get("bank_on_prediction"),
                "bank_on_score": row.get("bank_on_exact_match"),
                "bank_off_exact_match": row.get("bank_off_exact_match"),
                "bank_on_exact_match": row.get("bank_on_exact_match"),
                "bank_off_format_status": row.get("bank_off_primary_output_format_status"),
                "bank_on_format_status": row.get("bank_on_primary_output_format_status"),
                "output_changed": row.get("output_changed"),
                "improved": row.get("improved"),
                "regressed": row.get("regressed"),
                "output_changed_vs_canonical_frf": output_changed_vs_canonical,
                "final_slot_count": row.get("bank_slot_count_final_before_reset"),
                "write_action_counts": row.get("write_action_counts"),
                "append_insert_count": row.get("append_insert_count"),
                "matched_replace_count": row.get("matched_replace_count"),
                "capacity_evict_count": row.get("capacity_evict_count"),
                "query_turn_retrieved_indices": row.get("query_turn_retrieved_indices", []),
                "query_turn_retrieved_latent_count": row.get("query_turn_retrieved_latent_count", 0),
                "format_failure_bank_off": row.get("bank_off_primary_output_format_status") != "clean_option",
                "format_failure_bank_on": row.get("bank_on_primary_output_format_status") != "clean_option",
                "query_write_count": row.get("query_write_count"),
                "query_write_attempt_count": row.get("query_write_attempt_count"),
                "cross_context_leakage_detected": row.get("cross_context_leakage_detected"),
                "retrieved_latents_enter_weaver": row.get("retrieved_latents_enter_weaver"),
                "raw_retrieved_latents_enter_reasoner": row.get("raw_retrieved_latents_enter_reasoner"),
            }
        )

    bank_off_format_failures = _format_failure_count(contexts, "bank_off_primary_output_format_status")
    bank_on_format_failures = _format_failure_count(contexts, "bank_on_primary_output_format_status")
    canonical_summary = None
    if canonical_contexts is not None:
        canonical_summary = {
            "bank_off_EM": sum(int(row.get("bank_off_exact_match", 0)) for row in canonical_contexts) / len(canonical_contexts),
            "bank_on_EM": sum(int(row.get("bank_on_exact_match", 0)) for row in canonical_contexts) / len(canonical_contexts),
        }

    return {
        "threshold_label": threshold_label,
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
        "query_turn_retrieved_indices": summary.get("query_turn_retrieved_indices"),
        "query_turn_retrieved_latent_count": summary.get("query_turn_retrieved_latent_count"),
        "bank_off_format_failure_count": bank_off_format_failures,
        "bank_on_format_failure_count": bank_on_format_failures,
        "output_changed_vs_canonical_frf_count": output_changed_vs_canonical_count,
        "canonical_reference_available": canonical_summary is not None,
        "canonical_bank_off_EM": None if canonical_summary is None else canonical_summary["bank_off_EM"],
        "canonical_bank_on_EM": None if canonical_summary is None else canonical_summary["bank_on_EM"],
        "per_context_rows": per_context_rows,
        "run_dir": run_result["run_dir"],
        "output_root": run_result["output_root"],
    }


def main():
    sweep_root = Path(DEFAULT_OUTPUT_ROOT)
    sweep_root.mkdir(parents=True, exist_ok=True)
    canonical_contexts = _load_canonical_contexts()
    results = []
    per_context_rows: list[dict] = []
    for update_threshold in SWEEP_UPDATE_THRESHOLDS:
        run_result = _run_single_threshold(update_threshold)
        threshold_summary = _threshold_summary(run_result, canonical_contexts)
        results.append(threshold_summary)
        per_context_rows.extend(threshold_summary["per_context_rows"])

    aggregate = {
        "experiment_name": EXPERIMENT_NAME,
        "sweep_root": str(sweep_root),
        "canonical_reference": CANONICAL_FR_BASELINE,
        "retrieve_threshold": RETRIEVE_THRESHOLD,
        "thresholds": list(SWEEP_UPDATE_THRESHOLDS),
        "rows": [
            {
                key: value
                for key, value in row.items()
                if key not in {"per_context_rows", "paired_results"}
            }
            for row in results
        ],
    }

    _write_json(sweep_root / "threshold_diagnostic_aggregate.json", aggregate)
    _write_csv(
        sweep_root / "threshold_diagnostic_aggregate.csv",
        aggregate["rows"],
    )
    _write_jsonl(sweep_root / "threshold_diagnostic_per_context.jsonl", per_context_rows)
    _write_csv(sweep_root / "threshold_diagnostic_per_context.csv", per_context_rows)

    print(json.dumps({"sweep_root": str(sweep_root), "aggregate_report": str(sweep_root / "threshold_diagnostic_aggregate.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
