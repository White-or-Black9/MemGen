"""MAB-6B-FR retrieve-threshold relaxation diagnostic on DetectiveQA n=10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import mab6b_weaver_space_bank_detectiveqa_n10_capacity_diagnostic as cap
from scripts.eval import mab6b_weaver_space_bank_detectiveqa_n10_topk_diagnostic as topk

EXPERIMENT_NAME = "MAB-6B-FR retrieve-threshold relaxation diagnostic"
RUN_PREFIX = "detectiveqa-version-b-weaver-space-bank-format-repair-retrieve-threshold-relaxation-n10"
DEFAULT_OUTPUT_ROOT = "outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_retrieve_threshold_relaxation"
BENCHMARK_NOTE_PATH = Path(
    "research_notes/benchmarks/memoryagentbench_mab6b_fr_retrieve_threshold_relaxation.md"
)
DEFAULT_MAX_SLOTS = 16
DEFAULT_UPDATE_THRESHOLD = 0.08
DEFAULT_RETRIEVE_THRESHOLDS = (0.03, 0.02, 0.01, 0.005)
DEFAULT_TOP_K_VALUES = (1, 4)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    print(f"[retrieve-threshold-relaxation] {message}", flush=True)


def _threshold_token(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "")


def _setting_label(retrieve_threshold: float, top_k: int) -> str:
    return f"rt{_threshold_token(retrieve_threshold)}_topk{top_k}"


def _suppress_research_note(*args, **kwargs):
    output_dir = kwargs.get("output_dir")
    if output_dir is not None:
        return Path(output_dir) / "suppressed_research_note.md"
    return Path("suppressed_research_note.md")


def _iter_json_artifacts(root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl"}
    )


def _peak_cuda_memory_from_contexts(contexts: list[dict]) -> int | None:
    values = [row.get("peak_cuda_memory") for row in contexts if row.get("peak_cuda_memory") is not None]
    return max(values) if values else None


def _run_single_setting(
    *,
    retrieve_threshold: float,
    top_k: int,
    max_slots: int,
    update_threshold: float,
    requested_contexts: int | None,
    output_root: Path,
) -> dict:
    setting_label = _setting_label(retrieve_threshold, top_k)
    setting_root = output_root / setting_label
    setting_root.mkdir(parents=True, exist_ok=True)
    _progress(
        f"running setting={setting_label} retrieve_threshold={retrieve_threshold} top_k={top_k} "
        f"max_slots={max_slots} update_threshold={update_threshold}"
    )

    canonical_contexts = topk._load_canonical_contexts()
    run_result = topk._run_single_topk(
        top_k,
        max_slots=max_slots,
        retrieve_threshold=retrieve_threshold,
        update_threshold=update_threshold,
        requested_contexts=requested_contexts,
        output_root=setting_root,
    )
    summary, per_context_rows = topk._summarize_run(
        top_k=top_k,
        max_slots=max_slots,
        retrieve_threshold=retrieve_threshold,
        update_threshold=update_threshold,
        run_result=run_result,
        canonical_contexts=canonical_contexts,
    )
    summary.update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "setting_label": setting_label,
            "requested_contexts": requested_contexts,
            "run_dir": run_result["run_dir"],
            "output_root": str(setting_root),
            "paired_results_path": str(Path(run_result["run_dir"]) / "paired_results.json"),
            "diagnostics_path": str(Path(run_result["run_dir"]) / "diagnostics.jsonl"),
            "manifest_path": str(Path(run_result["run_dir"]) / "manifest.json"),
            "run_config_path": str(Path(run_result["run_dir"]) / "run_config.json"),
            "peak_cuda_memory": _peak_cuda_memory_from_contexts(run_result["paired_results"]["contexts"]),
            "failed": False,
            "traceback_path": None,
        }
    )
    worker_result_path = setting_root / "worker_result.json"
    per_context_path = setting_root / "worker_per_context.jsonl"
    _write_json(worker_result_path, summary)
    _write_jsonl(per_context_path, per_context_rows)
    summary["files_written"] = _iter_json_artifacts(setting_root)
    _write_json(worker_result_path, summary)
    return summary


def _write_failure_result(
    *,
    output_root: Path,
    retrieve_threshold: float,
    top_k: int,
    max_slots: int,
    update_threshold: float,
    requested_contexts: int | None,
    exc: BaseException,
) -> Path:
    setting_label = _setting_label(retrieve_threshold, top_k)
    setting_root = output_root / setting_label
    setting_root.mkdir(parents=True, exist_ok=True)
    traceback_path = setting_root / "debug_failure_traceback.log"
    traceback_path.write_text(
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        encoding="utf-8",
    )
    worker_result = {
        "experiment_name": EXPERIMENT_NAME,
        "setting_label": setting_label,
        "retrieve_threshold": retrieve_threshold,
        "top_k": top_k,
        "max_slots": max_slots,
        "update_threshold": update_threshold,
        "requested_contexts": requested_contexts,
        "failed": True,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback_path": str(traceback_path),
        "output_root": str(setting_root),
        "files_written": _iter_json_artifacts(setting_root),
    }
    worker_result_path = setting_root / "worker_result.json"
    _write_json(worker_result_path, worker_result)
    worker_result["files_written"] = _iter_json_artifacts(setting_root)
    _write_json(worker_result_path, worker_result)
    return worker_result_path


def aggregate_existing_artifacts(output_root: Path | str = DEFAULT_OUTPUT_ROOT) -> dict:
    output_root = Path(output_root)
    rows: list[dict] = []
    per_context_rows: list[dict] = []
    failed_jobs: list[dict] = []

    for setting_root in sorted(p for p in output_root.iterdir() if p.is_dir() and p.name.startswith("rt")):
        worker_result_path = setting_root / "worker_result.json"
        if not worker_result_path.exists():
            continue
        worker = _load_json(worker_result_path)
        worker["files_written"] = _iter_json_artifacts(setting_root)
        worker["top_k_4_reached_32_latents"] = bool(
            worker.get("top_k") == 4
            and all(int(value) == 32 for value in worker.get("query_turn_retrieved_latent_count_by_context", []))
        )
        worker["output_format_failures"] = {
            "bank_off": int(worker.get("format_failure_counts_bank_off", 0)),
            "bank_on": int(worker.get("format_failure_counts_bank_on", 0)),
        }
        rows.append(worker)
        if worker.get("failed"):
            failed_jobs.append(
                {
                    "setting_label": worker.get("setting_label"),
                    "error_type": worker.get("error_type"),
                    "error_message": worker.get("error_message"),
                    "traceback_path": worker.get("traceback_path"),
                }
            )
            continue
        per_context_rows.extend(_load_jsonl(setting_root / "worker_per_context.jsonl"))

    aggregate = {
        "experiment_name": EXPERIMENT_NAME,
        "output_root": str(output_root),
        "settings": [
            {"retrieve_threshold": rt, "top_k": top_k}
            for rt in DEFAULT_RETRIEVE_THRESHOLDS
            for top_k in DEFAULT_TOP_K_VALUES
        ],
        "rows": rows,
        "failed_jobs": failed_jobs,
    }
    return aggregate


def _write_benchmark_note(aggregate: dict, note_path: Path = BENCHMARK_NOTE_PATH) -> Path:
    lines = [
        "# MAB-6B-FR Retrieve-Threshold Relaxation Diagnostic",
        "",
        f"- Output root: `{aggregate['output_root']}`",
        f"- Fixed max_slots: `{DEFAULT_MAX_SLOTS}`",
        f"- Fixed update_threshold: `{DEFAULT_UPDATE_THRESHOLD}`",
        "- Dataset: `detective_qa`",
        "- Sample size: `n=10` unless noted as smoke",
        "- Guardrail: do not interpret top_k=4 quality unless query-turn retrieved latent count reaches 32.",
        "",
        "| setting | rt | top_k | bank_off_EM | bank_on_EM | final_slot_counts | q_indices | q_latents | reached_32 | bank_on_format_failures | failed |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | --- |",
    ]
    for row in aggregate["rows"]:
        lines.append(
            f"| {row.get('setting_label')} | {row.get('retrieve_threshold')} | {row.get('top_k')} | "
            f"{row.get('bank_off_EM')} | {row.get('bank_on_EM')} | {row.get('final_slot_counts')} | "
            f"{row.get('query_turn_retrieved_indices_by_context')} | {row.get('query_turn_retrieved_latent_count_by_context')} | "
            f"{row.get('top_k_4_reached_32_latents')} | {row.get('output_format_failures', {}).get('bank_on')} | "
            f"{row.get('failed')} |"
        )
    if aggregate["failed_jobs"]:
        lines.extend(["", "## Failed Jobs", ""])
        for row in aggregate["failed_jobs"]:
            lines.append(
                f"- `{row['setting_label']}`: `{row['error_type']}` at `{row['traceback_path']}`"
            )
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return note_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run-setting")
    run_parser.add_argument("--retrieve-threshold", type=float, required=True)
    run_parser.add_argument("--top-k", type=int, required=True)
    run_parser.add_argument("--max-slots", type=int, default=DEFAULT_MAX_SLOTS)
    run_parser.add_argument("--update-threshold", type=float, default=DEFAULT_UPDATE_THRESHOLD)
    run_parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    run_parser.add_argument("--requested-contexts", type=int)
    run_parser.add_argument("--smoke-test", action="store_true")

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    aggregate_parser.add_argument("--write-note", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "run-setting":
        requested_contexts = 1 if args.smoke_test else args.requested_contexts
        output_root = Path(args.output_root)
        if args.smoke_test:
            output_root = output_root / "smoke_test"
        try:
            summary = _run_single_setting(
                retrieve_threshold=float(args.retrieve_threshold),
                top_k=int(args.top_k),
                max_slots=int(args.max_slots),
                update_threshold=float(args.update_threshold),
                requested_contexts=requested_contexts,
                output_root=output_root,
            )
        except BaseException as exc:
            _write_failure_result(
                output_root=output_root,
                retrieve_threshold=float(args.retrieve_threshold),
                top_k=int(args.top_k),
                max_slots=int(args.max_slots),
                update_threshold=float(args.update_threshold),
                requested_contexts=requested_contexts,
                exc=exc,
            )
            raise
        print(json.dumps(summary, ensure_ascii=False))
        return 0

    output_root = Path(args.output_root)
    aggregate = aggregate_existing_artifacts(output_root)
    aggregate_path = output_root / "retrieve_threshold_relaxation_aggregate.json"
    per_context_path = output_root / "retrieve_threshold_relaxation_per_context.jsonl"
    _write_json(aggregate_path, aggregate)
    all_per_context_rows: list[dict] = []
    for setting_root in sorted(p for p in output_root.iterdir() if p.is_dir() and p.name.startswith("rt")):
        all_per_context_rows.extend(_load_jsonl(setting_root / "worker_per_context.jsonl"))
    _write_jsonl(per_context_path, all_per_context_rows)
    note_path = None
    if args.write_note:
        note_path = _write_benchmark_note(aggregate)
    payload = {
        "aggregate_path": str(aggregate_path),
        "per_context_path": str(per_context_path),
        "note_path": None if note_path is None else str(note_path),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
