"""Bounded M2 smoke: one construction, 10 questions, 5 policies + references.

Run with the MemGen environment. --preflight-only never loads a model.
Snapshots are locally generated trusted pickle files; never load third-party ones.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from scripts.eval import mab6b_weaver_space_bank_eventqa_65536_n5 as eventqa
from scripts.eval import review_eventqa_decomposition as metrics
from scripts.eval.review_retrieval_policies import (
    POLICIES, capture_bank, capture_rng, derive_last_write, fingerprint,
    load_snapshot, policy_adapter, restore_bank, restore_rng, save_snapshot,
)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def parser():
    p = eventqa.build_parser()
    p.description = __doc__
    p.set_defaults(context_index=0, question_limit=10, requested_contexts=5, seed=42,
                   top_k=2, retrieve_threshold=.05, update_threshold=.10, max_slots=16,
                   decay_alpha=.05, trace_score_decomposition=True, reseed_per_context=True,
                   save_frozen_bank=True, output_root="outputs/mab/review_m2_smoke")
    p.add_argument("--snapshot", help="Replay a trusted M2 snapshot instead of constructing")
    p.add_argument("--policy", choices=POLICIES + ("all",), default="all")
    p.add_argument("--run-id")
    p.add_argument("--preflight-only", action="store_true")
    return p


def preflight(args):
    # The submitted config path is resolved explicitly against official checkout.
    paths = {"checkpoint": Path(args.checkpoint_path), "backbone": Path(args.model_path),
             "dataset": Path(args.parquet), "data_config": Path(args.mab_repo) / args.data_config,
             "model_config": Path(args.cfg_path), "scorer": Path(args.mab_repo) / "utils/eval_other_utils.py",
             "scorer_python": Path(args.mab_python)}
    if args.snapshot:
        paths["snapshot"] = Path(args.snapshot)
    missing = [name for name, path in paths.items() if not path.exists()]
    report = {"paths": {k: str(v.resolve()) for k, v in paths.items()}, "missing": missing,
              "cuda_available": torch.cuda.is_available(), "python": sys.executable,
              "torch": torch.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if report["cuda_available"]:
        free, total = torch.cuda.mem_get_info()
        report.update(gpu_name=torch.cuda.get_device_name(), free_vram_bytes=free, total_vram_bytes=total)
    report["gate"] = "PASS" if not missing and report["cuda_available"] else "BLOCKED"
    return report


def score_records(args, records, output):
    request = output / "score_request.json"
    response = output / "score_response.json"
    write_json(request, records)
    code = """
import json, sys
from pathlib import Path
from scripts.eval import review_eventqa_decomposition as m
official = m.load_official(sys.argv[3])
rows = json.loads(Path(sys.argv[1]).read_text())
for row in rows:
    info = m.candidate_info(row['question'], row['gold_answers'], official)
    if info['errors']: raise ValueError(info['errors'])
    row.update(m.evaluate(row['prediction'], row['gold_answers'], info, row['format_flags'], official))
policies = sorted({r['policy'] for r in rows})
# Official parser can return None. Preserve the raw metric values in records;
# aggregate absent scores as zero, not by silently dropping failed questions.
aggregate_rows = [dict(r, official_em=r['official_em'] or 0,
                       official_raw_recall=r['official_raw_recall'] or 0) for r in rows]
summary = {p: m.summarize([r for r in aggregate_rows if r['policy'] == p]) for p in policies}
nonempty = {p: m.summarize([r for r in aggregate_rows if r['policy'] == p and r['selection'][0]['n'] > 0]) for p in policies}
full = {r['query_id']: r for r in rows if r['policy'] == 'full'}
paired = {}
for p in policies:
    if p == 'full' or not full: continue
    other = [r for r in rows if r['policy'] == p]
    paired[p] = {'wins': sum(full[r['query_id']]['strict_candidate_correct'] and not r['strict_candidate_correct'] for r in other),
                 'losses': sum(r['strict_candidate_correct'] and not full[r['query_id']]['strict_candidate_correct'] for r in other),
                 'ties': sum(r['strict_candidate_correct'] == full[r['query_id']]['strict_candidate_correct'] for r in other)}
Path(sys.argv[2]).write_text(json.dumps({'records': rows, 'summary': summary, 'n_positive': nonempty,
                                      'paired_full': paired, 'scorer_sha256': m.sha(sys.argv[3])}, indent=2) + '\\n')
"""
    subprocess.run([args.mab_python, "-c", code, str(request), str(response),
                    str(Path(args.mab_repo) / "utils/eval_other_utils.py")], cwd=ROOT, check=True)
    return metrics.read(response)


def verify_query(result, trace, before, bank):
    if result["query_write_count_delta"] != 0 or not result["query_read_only_enforced"]:
        raise RuntimeError("query write contract violated")
    if result["bank_snapshot_changed_after_query"] or fingerprint(capture_bank(bank)) != before:
        raise RuntimeError("query changed frozen bank")
    # Multiple query retrievals would change the query representation and count;
    # fail rather than pretend this is a single-slot-selection intervention.
    if len(trace) != 1 or result["query_retrieval_invocation_count"] != 1:
        raise RuntimeError("expected exactly one query retrieval")


def results_text(scored, manifest):
    lines = ["# M2 限定 smoke 结果", "", "seed42/context0/q0–9；仅验证协议，不推断检索有效性。", "",
             "| 策略 | 题数 | 官方 EM | 严格候选正确率 | 候选有效率 | n>0 题数 |",
             "|---|---:|---:|---:|---:|---:|"]
    for policy, row in scored["summary"].items():
        lines.append(f"| {policy} | {row['question_count']} | {row['em']['value']:.3f} | "
                     f"{row['strict_candidate_accuracy']['value']:.3f} | "
                     f"{row['candidate_valid_rate']['value']:.3f} | "
                     f"{scored['n_positive'][policy]['question_count']} |")
    lines += ["", f"原生排序改变答案的题目：{manifest['order_changed_queries']}。",
              f"正式实验是否必须增加原生排序参考：{manifest['formal_native_reference_required']}。",
              "", "全部逐题选择、条件指标和配对胜负见 results.json。",
              "官方 parser 无输出时，聚合按零分计入全部题目，原始空指标保留在 records。",
              "这是当前源码的新 campaign，不能替代历史 dirty-code P7 的精确复现。", ""]
    return "\n".join(lines)


def run(args):
    if (args.seed, args.context_index, args.question_limit) != (42, 0, 10):
        raise ValueError("this phase is bounded to seed42/context0/q0-9")
    if args.strict_official_eventqa_prompt or args.first_line_official_eventqa_prompt or args.construction_only:
        raise ValueError("M2 must retain the submitted unconstrained query protocol")
    if args.eventqa_protocol != "frozen_context_bank":
        raise ValueError("M2 requires frozen_context_bank")
    if (args.top_k, args.max_slots, args.retrieve_threshold, args.update_threshold,
        args.decay_alpha, args.generation_max_length, args.chunk_size) != (2, 16, .05, .10, .05, 40, 4096):
        raise ValueError("M2 must retain M0 submitted bank/generation configuration")
    check = preflight(args)
    print(json.dumps(check, indent=2), flush=True)
    if args.preflight_only:
        return 0 if check["gate"] == "PASS" else 2
    if check["gate"] != "PASS":
        raise RuntimeError("preflight blocked; no model loaded")
    # Shared GPU reserve. Never kill, move or modify another user's process.
    if check["free_vram_bytes"] < 12 * 1024**3:
        raise RuntimeError("less than 12 GiB free VRAM; no model loaded")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-m2-smoke")
    output = Path(args.output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    config = eventqa._eventqa_bank_config(args)
    selected_policies = list(POLICIES) if args.policy == "all" else [args.policy]
    manifest = {"schema": "m2-smoke/v1", "run_id": run_id, "args": vars(args), "bank_config": config,
                "review_items": ["R03", "R07"], "parent": "research_notes/next_version/m1/M1_REPORT.md",
                "paper_sha256": metrics.sha(ROOT / "review/nference_Time_Latent_Me.pdf"),
                "preflight": check, "policies": selected_policies, "status": "RUNNING",
                "evidence_role": "protocol_verification_only", "formal_campaign_authorized": False}
    source_files = sorted(set(ROOT.glob("memgen/**/*.py")) | set(ROOT.glob("interactions/**/*.py")) |
                          set(ROOT.glob("scripts/eval/*.py")) | {ROOT / "main.py",
                          ROOT / "scripts/eval/run_review_m2_smoke.sh"})
    manifest["source_sha256"] = {str(p.relative_to(ROOT)): metrics.sha(p) for p in source_files}
    manifest["input_sha256"] = {key: metrics.sha(path) for key, path in check["paths"].items()
                                if Path(path).is_file()}
    manifest["git_head"] = eventqa._git("rev-parse", "HEAD")
    manifest["git_status"] = eventqa._git("status", "--short")
    (output / "tracked_source.diff").write_text(eventqa._git("diff", "HEAD"))
    write_json(output / "manifest.json", manifest)
    records = []
    try:
        rows = eventqa._load_rows(args.parquet, eventqa.SUB_DATASET)
        context = eventqa.build_context_payload(args, rows[0], 0, eventqa._utc_now())
        write_json(output / "context_identity.json", {
            "context_id": context["context_id"], "question_identity": eventqa._question_identity_records(context),
            "chunk_sha256": [metrics.text_sha(c) for c in context["chunks"]]})
        model, capacity = eventqa.weaver_bank._load_model(args)
        runtime = eventqa._runtime_reproducibility_metadata(args, selected_context_indices=[0], model=model)
        write_json(output / "runtime.json", runtime)
        eventqa._prepare_context_rng(base_seed=42, context_index=0, reseed_per_context=True)
        if args.snapshot:
            saved = load_snapshot(args.snapshot)
            expected = {"context_id": context["context_id"], "seed": 42, "bank_config": config,
                        "checkpoint": str(Path(args.checkpoint_path).resolve()),
                        "source_sha256": manifest["source_sha256"],
                        "chunk_sha256": [metrics.text_sha(c) for c in context["chunks"]]}
            for key, value in expected.items():
                if saved["provenance"].get(key) != value:
                    raise ValueError(f"snapshot provenance mismatch: {key}")
            snapshot_path = Path(args.snapshot)
        else:
            construction = eventqa._run_eventqa_model(
                args, model, capacity, eventqa._construction_only_payload(context), "on", config,
                preserve_bank=True, construction_only=True, recorded_bank_config=config)
            bank = construction.pop("_retained_bank")
            write_json(output / "construction.json", construction)
            last_write = derive_last_write(construction["construction_turn_diagnostics"], len(bank))
            bank_state, rng_state = capture_bank(bank), capture_rng()
            snapshot_path = output / "frozen_bank.pt"
            save_snapshot(snapshot_path, bank_state, rng_state, last_write,
                          {"context_id": context["context_id"], "seed": 42, "bank_config": config,
                           "checkpoint": str(Path(args.checkpoint_path).resolve()), "runtime": runtime,
                           "source_sha256": manifest["source_sha256"],
                           "chunk_sha256": [metrics.text_sha(c) for c in context["chunks"]]})
            saved = load_snapshot(snapshot_path)
            bank.reset()
            del bank
        manifest["snapshot_path"] = str(snapshot_path.resolve())
        if any(slot.memory.shape[0] != 8 for slot in saved["bank"]["_slots"]):
            raise ValueError("submitted protocol requires 8 latent tokens per slot")
        manifest["snapshot_file_sha256"] = metrics.sha(snapshot_path)
        manifest["snapshot_bank_sha256"] = saved["bank_sha256"]
        write_json(output / "manifest.json", manifest)
        order_changed = []
        policies = selected_policies + (["native_order", "full_repeat"] if "full" in selected_policies else [])
        for query_id in range(10):
            question = eventqa.build_question_payload(context, query_id)
            payload = eventqa._query_only_payload(question)
            query_records = {}
            for label in policies:
                bank = restore_bank(saved["bank"])
                restore_rng(saved["rng"])
                before = fingerprint(capture_bank(bank))
                trace = []
                actual = "full" if label == "full_repeat" else label
                with policy_adapter(bank, actual, last_write=saved["last_write"], seed=42,
                                    context=0, query=query_id, trace=trace):
                    result = eventqa._run_eventqa_model(args, model, capacity, payload, "on", config,
                                                      external_bank=bank, preserve_bank=True,
                                                      recorded_bank_config=config)
                result.pop("_retained_bank", None)
                verify_query(result, trace, before, bank)
                record = {"policy": label, "seed": 42, "context_id": context["context_id"],
                          "query_id": query_id, "question": question["question"],
                          "gold_answers": question["gold_answers"], "prediction": result["prediction"],
                          "query_sha256": metrics.text_sha(payload["query_prompt"]), "selection": trace,
                          "format_flags": eventqa._format_flags(result["prediction"]),
                          "query_write_count_delta": result["query_write_count_delta"],
                          "snapshot_unchanged": True}
                artifact = output / "queries" / f"q{query_id:02d}_{label}.json"
                artifact.parent.mkdir(exist_ok=True)
                write_json(artifact, {"record": record, "runtime_result": result})
                records.append(record)
                query_records[label] = record
                print(f"QUERY_COMPLETE q={query_id} policy={label} n={trace[0]['n']} slots={trace[0]['selected_indices']}", flush=True)
                bank.reset()
            if "full" in query_records:
                full, repeat, native = (query_records[p] for p in ("full", "full_repeat", "native_order"))
                if full["prediction"] != repeat["prediction"] or full["selection"] != repeat["selection"]:
                    raise RuntimeError(f"non-reproducible full output/selection at q{query_id}")
                if full["prediction"] != native["prediction"]:
                    order_changed.append(query_id)
                budgets = {(r["selection"][0]["n"], r["selection"][0]["latent_count"]) for r in query_records.values()}
                if len(budgets) != 1:
                    raise RuntimeError(f"cross-policy count/budget mismatch q{query_id}")
        if metrics.sha(snapshot_path) != manifest["snapshot_file_sha256"]:
            raise RuntimeError("snapshot file changed")
        scored = score_records(args, records, output)
        write_json(output / "results.json", scored)
        manifest.update(status="PASS", query_count=len(records), main_query_count=10 * len(selected_policies),
                        order_changed_queries=order_changed, formal_native_reference_required=bool(order_changed),
                        finished_at=eventqa._utc_now())
        manifest["n_positive_questions"] = sum(r["selection"][0]["n"] > 0 for r in records
                                               if r["policy"] == selected_policies[0])
        manifest["selection_changed_questions"] = {
            p: sum(r["selection"][0]["selected_indices"] != r["selection"][0]["native_indices"]
                   and set(r["selection"][0]["selected_indices"]) != set(r["selection"][0]["native_indices"])
                   for r in records if r["policy"] == p) for p in selected_policies}
        write_json(output / "manifest.json", manifest)
        (output / "RESULTS.md").write_text(results_text(scored, manifest))
        print(json.dumps({"status": "PASS", "output": str(output), "order_changed": order_changed,
                          "summary": scored["summary"]}, indent=2), flush=True)
        return 0
    except BaseException as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", completed_queries=len(records))
        write_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    raise SystemExit(run(parser().parse_args()))
