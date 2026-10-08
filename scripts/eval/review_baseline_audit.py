"""Offline audit of the submitted EventQA paper; standard library only.

Historical artifacts are read-only. Outputs go to research_notes/next_version.
The frozen campaign paths are intentional: never select a latest/best run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_notes/next_version"
SEEDS = (42, 142, 242, 342, 442)
PDF = "review/nference_Time_Latent_Me.pdf"
PDF_HASH = "6b04a3c49b9bb68cccf282059be34f7dadc516e4069e42b204fcae99f68ffed3"
FORMAL = "outputs/mab/formal_p7_effect_repeats/20260722T120000Z-formal-p7-effect-repeats"
CONTROL = "outputs/mab/paper_baseline_effects/20260722T012259Z-eventqa-paper-baseline-effects"
RECENT = "outputs/mab/p7_recent_text_32256_effect_repeats/20260722T153000Z-p7-recent-text-32256-effect-repeats/recent_text_32256"
DENSE = "outputs/mab/dense_top2_effect_repeats/20260726T_dense_top2_paper_repeats"
ABLATION = "outputs/mab/ablation_effect_repeats/20260722T143500Z-eventqa-ablation-effect-repeats"
DECAY = "outputs/mab/no_decay_effect_repeats/20260725T_no_decay_alpha0_full"
COST = "outputs/mab/eventqa_comparable_cost/20260722T013124Z-eventqa-comparable-cost"
DENSE_COST = "outputs/mab/eventqa_dense_top2_comparable_cost/20260726T100413Z-eventqa-dense-top2-comparable-cost"
# Values transcribed from submitted Figures 2/3 (pages 5/6), not paper/*.md.
METHODS = {
    "p7": (FORMAL, "latent", (0.188, 0.231, 118.6)),
    "recent_text": (RECENT, "control", (0.034, 0.096, 211.0)),
    "rolling_summary": (CONTROL + "/rolling_summary", "summary", (0.012, 0.078, 267.0)),
    "bm25_top2": (CONTROL + "/bm25_top2", "control", (0.030, 0.226, 265.0)),
    "dense_top2": (DENSE, "dense", (0.030, 0.240, 285.0)),
    "matched16": (CONTROL + "/matched16", "control", (0.068, 0.180, 347.0)),
    "no_retrieved_memory_conditioning": (ABLATION + "/no_retrieved_memory_conditioning", "ablation", (0.008, 0.178, 377.0)),
    "direct_top1": (ABLATION + "/direct_top1", "ablation", (0.047, 0.186, 354.6)),
    "no_decay": (DECAY, "latent", (0.163, 0.251, 189.0)),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stats(values):
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("non-finite or missing metric")
    return {"values": values, "mean": statistics.fmean(values), "population_std": statistics.pstdev(values)}


def one(paths, label):
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f"{label}: expected exactly one artifact, found {len(paths)}")
    return paths[0]


def metric_rows(rows, latent=False):
    expected = {(c, q) for c in range(5) for q in range(100)}
    identities = [(r["context_index"], r.get("query_id") if latent else r["query_index"]) for r in rows]
    if len(rows) != 500 or len(set(identities)) != 500 or set(identities) != expected:
        raise ValueError("missing/duplicate question identities; expected 5 x 100")
    prefix = "bank_on_" if latent else ""
    for r in rows:
        if r[prefix + "substring_exact_match"] not in (0, 1, False, True):
            raise ValueError("invalid EM value")
        if r[prefix + "eventqa_recall"] not in (0, 1, False, True):
            raise ValueError("invalid recall value")
        if latent and (r["query_write_count"] != 0 or r["bank_snapshot_changed_after_query"] is not False or r.get("cross_context_leakage_detected", False)):
            raise ValueError("frozen-bank lifecycle violation")
        if "query_invariants" in r:
            inv = r["query_invariants"]
            for key, value in inv.items():
                if "write_count" in key and "attempt" not in key and value != 0:
                    raise ValueError(f"ablation lifecycle violation: {key}")
                if "snapshot_changed" in key and value is not False:
                    raise ValueError("ablation snapshot changed")
    return [statistics.fmean(float(r[prefix + "substring_exact_match"]) for r in rows),
            statistics.fmean(float(r[prefix + "eventqa_recall"]) for r in rows),
            sum(int(any(r[prefix + "format_flags"].values())) for r in rows)]


def audit_method(method, definition):
    root, kind, expected = definition
    runs = []
    for seed in SEEDS:
        seedroot = ROOT / root / f"seed{seed}"
        if kind == "latent":
            ap = one(seedroot.glob("*/eventqa_aggregate.json"), str(seedroot))
            rowfiles = [ap.parent / "eventqa_per_question.jsonl"]
            rows = jsonl(rowfiles[0])
            aggregate = read(ap)
            summaries = aggregate["summaries"]
            recorded = [sum(r["bank_on_accuracy"] * r["question_count"] for r in summaries) / 500,
                        sum(r["bank_on_eventqa_recall_mean"] * r["question_count"] for r in summaries) / 500,
                        sum(r["bank_on_format_failure_count"] for r in summaries)]
            manifest = read(ap.parent / "manifest.json")
            for k, v in {"retrieve_threshold": .05, "update_threshold": .10, "max_slots": 16, "top_k": 2,
                         "decay_alpha": 0.0 if method == "no_decay" else .05,
                         "generation_max_length": 40, "eventqa_protocol": "frozen_context_bank"}.items():
                if manifest.get(k) != v:
                    raise ValueError(f"{method} seed{seed}: config drift {k}")
            repro = manifest.get("reproducibility_diagnostics", {})
            if (ap.parent / "reproducibility_diagnostics.json").exists():
                repro = read(ap.parent / "reproducibility_diagnostics.json").get("runtime_metadata", {})
            metadata = {k: repro.get(k) for k in ["git_commit", "git_dirty", "seed", "gpu", "python_version", "torch_version", "transformers_version", "dataset_path", "checkpoint_snapshot_hash", "argv"]}
            if metadata["seed"] != seed:
                raise ValueError("runtime seed mismatch")
            metadata["checkpoint_path"] = manifest.get("checkpoint_path")
            metadata["bank_config"] = manifest.get("latent_memory_bank_config")
            metadata["context_ids"] = aggregate.get("context_ids")
            metadata["question_identity_sha256"] = hashlib.sha256(json.dumps(sorted(
                [(r["context_id"], r["query_id"], r["question"], r["gold_answers"]) for r in rows]),
                ensure_ascii=False).encode()).hexdigest()
        else:
            ap = seedroot / "aggregate.json"
            aggregate = read(ap)
            recorded = [aggregate["effectiveness"][k] for k in ["substring_exact_match", "eventqa_recall", "format_failure_count"]]
            if kind == "ablation":
                rowfiles = [ROOT / p for p in aggregate["artifacts"]]
            elif kind == "summary":
                rowfiles = sorted(seedroot.glob("query/*/query_artifact.json"))
            elif kind == "dense":
                rowfiles = sorted(seedroot.glob("*/full_artifact.json"))
            else:
                rowfiles = sorted(seedroot.glob("ctx*/*/full_artifact.json"))
            if len(rowfiles) != 5:
                raise ValueError(f"{method} seed{seed}: expected five context artifacts")
            rows = [r for p in rowfiles for r in read(p)["records"]]
            metadata = {"seed_directory": seed, "runtime_seed_verified": False,
                        "method_contract": {k: aggregate[k] for k in ["method", "dense", "bm25", "budget"] if k in aggregate}}
        computed = metric_rows(rows, latent=kind == "latent")
        if any(abs(a - b) > 1e-9 for a, b in zip(computed, recorded)):
            raise ValueError(f"{method} seed{seed}: raw/aggregate mismatch {computed} vs {recorded}")
        files = [ap, *rowfiles]
        for p in rowfiles:
            for name in ["manifest.json", "run_config.json", "reproducibility_diagnostics.json"]:
                candidate = p.parent / name
                if candidate.exists() and candidate not in files:
                    files.append(candidate)
        runs.append({"seed": seed, "metrics": computed, "metadata": metadata,
                     "files": [{"path": str(p.relative_to(ROOT)), "resolved_path": str(p.resolve()), "sha256": digest(p)} for p in files]})
    metrics = {k: stats([run["metrics"][i] for run in runs]) for i, k in enumerate(["em", "recall", "format_flags_count"])}
    for k, target, digits in zip(metrics, expected, [3, 3, 1]):
        if round(metrics[k]["mean"], digits) != target:
            raise ValueError(f"{method}: submitted mean mismatch {k} {metrics[k]} vs {target}")
    if kind == "latent" and len({run["metadata"]["question_identity_sha256"] for run in runs}) != 1:
        raise ValueError(f"{method}: question text/gold identity drift across seeds")
    submitted_std = {"p7": (.047, .042, 19.1), "no_decay": (.047, .030, 52.8), "direct_top1": (.008, .009, 21.3)}
    if method in submitted_std:
        for k, target, digits in zip(metrics, submitted_std[method], [3, 3, 1]):
            if round(metrics[k]["population_std"], digits) != target:
                raise ValueError(f"{method}: submitted population std mismatch {k}")
    return {"campaign": root, "kind": kind, "trust": "raw_metrics_verified_provenance_partial", "metrics": metrics, "runs": runs}


def audit_cost():
    result = {}
    for method in ["p7", "recent_text", "rolling_summary", "bm25_top2", "matched16", "dense_top2"]:
        campaign = ROOT / (DENSE_COST if method == "dense_top2" else COST)
        evidence = read(campaign / "campaign_evidence.json")
        values, files = [], []
        for rep in range(1, 4):
            path = campaign / f"rep{rep}"
            if method != "dense_top2":
                path = path / method
            path = path / "continuous_cost_summary.json"
            item = read(path)
            if item["scope"]["question_count"] != 500 or item["scope"]["context_indices"] != list(range(5)):
                raise ValueError("incomplete cost run")
            cost = item["cost"]
            if cost.get("model_loading_excluded_from_per_question_latency") is not True:
                raise ValueError("cost boundary drift")
            total = cost.get("end_to_end_latency_seconds_total", cost.get("end_to_end_latency_seconds"))
            if method == "recent_text":
                # No separate bank/index construction; this schema records query total.
                total = cost["query_latency_seconds_total"]
            peak = cost.get("incremental_peak_gpu_memory_bytes_max", max(cost.get("construction_incremental_peak_gpu_memory_bytes_max", 0), cost.get("query_incremental_peak_gpu_memory_bytes_max", 0)))
            values.append((total, peak / 2**30))
            if not math.isfinite(total) or total <= 0 or peak <= 0:
                raise ValueError("invalid cost value")
            for artifact in item.get("artifact_paths", []):
                if not (ROOT / artifact).is_file():
                    raise ValueError(f"missing cost source: {artifact}")
            files.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path), "continuous_process": item.get("continuous_process"), "model_load_count": item.get("model_load_count")})
        result[method] = {"total_seconds": stats([v[0] for v in values]), "seconds_per_question": stats([v[0] / 500 for v in values]), "max_peak_incremental_gib": max(v[1] for v in values),
                          "campaign_evidence": evidence, "campaign_evidence_sha256": digest(campaign / "campaign_evidence.json"), "files": files}
    # Submitted page 7 explicitly reports these precise timing numbers.
    for method, target in [("p7", 377.06), ("recent_text", 2688.90)]:
        if round(result[method]["total_seconds"]["mean"], 2) != target:
            raise ValueError(f"submitted cost mismatch: {method}")
    for method, target in {"p7": .160, "recent_text": 13.094, "rolling_summary": 1.782,
                           "bm25_top2": 3.505, "dense_top2": 3.505, "matched16": .159}.items():
        if round(result[method]["max_peak_incremental_gib"], 3) != target:
            raise ValueError(f"submitted peak allocation mismatch: {method}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write only new-version audit JSON and result table")
    args = parser.parse_args()
    if digest(ROOT / PDF) != PDF_HASH:
        raise ValueError("submitted PDF changed; re-audit paper identity")
    methods = {}
    for name, definition in METHODS.items():
        methods[name] = audit_method(name, definition)
        print(f"verified {name}: five seeds, 500 questions per seed", flush=True)
    if methods["p7"]["runs"][0]["metadata"]["question_identity_sha256"] != methods["no_decay"]["runs"][0]["metadata"]["question_identity_sha256"]:
        raise ValueError("P7/no-decay question text or gold differs")
    cost = audit_cost()
    provenance_paths = [ROOT / PDF, ROOT / "review/review", ROOT / "review/gpt_review_suggest.md",
                        ROOT / "memgen/model/modeling_memgen.py", ROOT / "memgen/model/latent_memory_bank.py",
                        ROOT / "eval/exp/common/eventqa_base.py", ROOT / "eval/exp/common/triviaqa.yaml",
                        Path("/mnt/18T/baishilong/benchmarks/MemoryAgentBench/utils/eval_other_utils.py"),
                        Path("/mnt/18T/baishilong/benchmarks/MemoryAgentBench/configs/data_conf/Accurate_Retrieval/EventQA/Eventqa_64k.yaml")]
    provenance = [{"path": str(p), "sha256": digest(p)} for p in provenance_paths]
    payload = {"schema_version": "submitted-review-baseline/v1", "paper": {"path": PDF, "sha256": PDF_HASH, "pages": 8},
               "audit_code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "audit_script_sha256": digest(Path(__file__)), "seeds": list(SEEDS), "unique_questions": 500,
               "methods": methods, "cost": cost, "current_source_identity": provenance,
               "gate": "PASS_WITH_LIMITATIONS", "limitations": [
                   "Published format failures are heuristic format flags, not parser failures.",
                   "Historical P7 runs record dirty worktrees: commit plus config does not reconstruct exact original code.",
                   "Control seed directories alone do not establish actual runtime seed provenance.",
                   "Dense cost campaign is GPU 4; other methods use GPU 5, contradicting an all-method same-GPU claim.",
                   "Frozen-bank tensors and historical source diffs are not guaranteed persisted.",
               ]}
    lines = ["# Submitted-paper baseline: independently recomputed results", "", "Source: `review/nference_Time_Latent_Me.pdf`, pages 5–7. Population standard deviation; 500 unique questions repeated five times.", "", "| Method | EM | Raw recall | Heuristic format flags |", "|---|---:|---:|---:|"]
    for name, item in methods.items():
        metrics = item["metrics"]
        lines.append("| " + name + " | " + " | ".join(f"{v['mean']:.4f} ± {v['population_std']:.4f}" for v in metrics.values()) + " |")
    lines += ["", "Format flags are not benchmark parser failures. No parsed-only accuracy is inferred from these counts.", "", "| Method | End-to-end seconds | Seconds/question | Max incremental GiB | Campaign GPU |", "|---|---:|---:|---:|---:|"]
    for name, item in cost.items():
        total, per = item["total_seconds"], item["seconds_per_question"]
        lines.append(f"| {name} | {total['mean']:.2f} ± {total['population_std']:.2f} | {per['mean']:.3f} ± {per['population_std']:.3f} | {item['max_peak_incremental_gib']:.3f} | {item['campaign_evidence']['gpu_index']} |")
    lines += ["", "Timing includes method preparation and queries, excludes model loading. Peak allocation is incremental GPU allocation, not total VRAM or CPU bank storage.", "", "## Gate", "", "PASS_WITH_LIMITATIONS: existing predictions can anchor offline reanalysis; exact historical executable reconstruction is incomplete.", ""]
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "baseline_evidence.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (OUT / "RESULT_TABLE.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"gate": payload["gate"], "methods": len(methods), "effect_runs": 45, "cost_runs": 18, "raw_records": 22500, "limitations": payload["limitations"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
