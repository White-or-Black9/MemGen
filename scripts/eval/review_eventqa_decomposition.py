"""Offline M1 diagnostics for the frozen submitted-paper predictions.

Run with the existing MABench Python environment. The official module is loaded
directly, without importing MemGen or loading any model. Default is read-only.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from functools import lru_cache
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import statistics
import subprocess
import tokenize

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "research_notes/next_version/baseline_evidence.json"
OUTPUT = ROOT / "research_notes/next_version/m1"
OFFICIAL = Path("/mnt/18T/baishilong/benchmarks/MemoryAgentBench/utils/eval_other_utils.py")
CATEGORIES = ("correct_valid", "gold_mentioned_wrong_valid", "gold_mentioned_no_valid", "gold_absent")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_official(path=OFFICIAL):
    spec = importlib.util.spec_from_file_location("m1_official_eventqa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def candidates(question):
    """Extract the literal list, allowing bracket characters inside strings."""
    marker = "Below is a list of possible subsequent events:"
    tail = question.split(marker, 1)[1].lstrip()
    if not tail.startswith("["):
        raise ValueError("candidate list missing")
    depth = 0
    for token in tokenize.generate_tokens(io.StringIO(tail).readline):
        if token.type == tokenize.OP:
            if token.string == "[":
                depth += 1
            elif token.string == "]":
                depth -= 1
                if depth == 0:
                    line, column = token.end
                    lines = tail.splitlines(keepends=True)
                    literal = "".join(lines[:line - 1]) + lines[line - 1][:column]
                    value = ast.literal_eval(literal)
                    if not isinstance(value, list) or not value or any(not isinstance(v, str) or not v.strip() for v in value):
                        raise ValueError("invalid candidate list")
                    return value
    raise ValueError("unterminated candidate list")


def candidate_info(question, gold, official):
    errors = []
    try:
        original = candidates(question)
        unique = list(dict.fromkeys(v.strip() for v in original))
        normalized = [official.normalize_answer(v) for v in unique]
        gold_norm = {official.normalize_answer(v) for v in gold}
        collisions = len(set(normalized)) != len(normalized)
        if collisions:
            errors.append("normalization_collision")
        if any(not n for n in normalized) or any(not g for g in gold_norm):
            errors.append("empty_normalized_text")
        covered = gold_norm.issubset(set(normalized))
        if not covered:
            errors.append("gold_not_covered")
        chance = sum(n in gold_norm for n in normalized) / len(unique) if not errors else None
        return {"candidates": unique, "normalized": normalized, "gold_normalized": sorted(gold_norm),
                "original_count": len(original), "duplicate_count": len(original) - len(unique),
                "multi_gold": len(gold) > 1, "chance": chance, "errors": errors}
    except (ValueError, SyntaxError, IndexError, tokenize.TokenError, IndentationError) as exc:
        return {"candidates": [], "normalized": [], "gold_normalized": [], "original_count": None,
                "duplicate_count": None, "multi_gold": len(gold) > 1, "chance": None,
                "errors": ["candidate_extraction_failed:" + type(exc).__name__]}


def evaluate(prediction, gold, info, flags, official, score=None):
    if not isinstance(prediction, str):
        raise ValueError("prediction must be a string")
    parsed = official.parse_output(prediction)
    if score is None:
        if parsed is None:
            score = {"substring_exact_match": None, "eventqa_recall": None}
        else:
            score = official._process_eventqa_dataset({"output": prediction}, gold)[0]
    nonempty = parsed is not None and bool(parsed.strip())
    normalized = official.normalize_answer(parsed) if nonempty else ""
    matches = [i for i, n in enumerate(info["normalized"]) if normalized and n == normalized]
    valid = len(matches) == 1
    correct = valid and normalized in info["gold_normalized"]
    mentioned = any(g.lower() in prediction.lower() for g in gold)
    category = ("correct_valid" if correct else "gold_mentioned_wrong_valid" if mentioned and valid
                else "gold_mentioned_no_valid" if mentioned else "gold_absent")
    return {"parsed": parsed, "parser_none": parsed is None, "parser_empty": parsed is not None and not parsed.strip(),
            "parsed_nonempty": nonempty, "candidate_valid": valid, "candidate_match_count": len(matches),
            "strict_candidate_correct": correct, "official_em": score["substring_exact_match"],
            "substring_only_correct": bool(score["substring_exact_match"]) and not correct,
            "official_raw_recall": score["eventqa_recall"], "raw_gold_any": mentioned,
            "heuristic_flag": any(flags.values()), "format_flags": flags, "category": category,
            "candidate_errors": info["errors"]}


def fraction(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def summarize(rows):
    n = len(rows)
    counts = {k: sum(bool(r[k]) for r in rows) for k in
              ["parser_none", "parser_empty", "parsed_nonempty", "candidate_valid", "strict_candidate_correct", "substring_only_correct", "heuristic_flag"]}
    categories = {k: sum(r["category"] == k for r in rows) for k in CATEGORIES}
    assert sum(categories.values()) == n
    assert counts["parser_none"] + counts["parser_empty"] + counts["parsed_nonempty"] == n
    em = sum(float(r["official_em"]) for r in rows)
    assert em == counts["strict_candidate_correct"] + counts["substring_only_correct"]
    parsed = [r for r in rows if r["parsed_nonempty"]]
    valid = [r for r in rows if r["candidate_valid"]]
    return {"question_count": n, "counts": counts, "categories": categories,
            "em": fraction(em, n), "raw_recall": fraction(sum(r["official_raw_recall"] for r in rows), n),
            "parsed_nonempty_rate": fraction(len(parsed), n), "candidate_valid_rate": fraction(len(valid), n),
            "strict_candidate_accuracy": fraction(counts["strict_candidate_correct"], n),
            "substring_only_accuracy": fraction(counts["substring_only_correct"], n),
            "heuristic_flag_rate": fraction(counts["heuristic_flag"], n),
            "parsed_conditional_em": fraction(sum(r["official_em"] for r in parsed), len(parsed)),
            "parsed_conditional_strict_accuracy": fraction(sum(r["strict_candidate_correct"] for r in parsed), len(parsed)),
            "candidate_conditional_em": fraction(sum(r["official_em"] for r in valid), len(valid)),
            "candidate_conditional_strict_accuracy": fraction(sum(r["strict_candidate_correct"] for r in valid), len(valid))}


def collect(run, kind):
    files = run["files"]
    if kind == "latent":
        selected = [f for f in files if f["path"].endswith("eventqa_per_question.jsonl")]
    else:
        selected = [f for f in files if Path(f["path"]).name in ("full_artifact.json", "query_artifact.json", "artifact.json")]
    rows = []
    for source in selected:
        p = ROOT / source["path"]
        if kind == "latent":
            with p.open(encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
        else:
            records = read(p)["records"]
        manifest = read(p.parent / "manifest.json") if (p.parent / "manifest.json").exists() else {}
        for r in records:
            rows.append((r, manifest, source["path"]))
    return rows


def identity(row, latent=False):
    return row["context_index"], row["query_id" if latent else "query_index"]


def canonical_map(method):
    rows = collect(method["runs"][0], "latent")
    result = {}
    for row, _, _ in rows:
        key = identity(row, True)
        messages = row["bank_on_rendered_query_messages"]
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        result[key] = {"question": row["question"], "gold": row["gold_answers"], "context_id": row["context_id"],
                       "qa_pair_id": row["qa_pair_id"], "query_sha256": text_sha(user)}
    if len(rows) != 500 or len(result) != 500:
        raise ValueError("canonical map incomplete or duplicate")
    return result


def check_identity(row, manifest, canonical, latent):
    context_id = row.get("context_id") or manifest.get("context_id")
    if context_id is not None and context_id != canonical["context_id"]:
        raise ValueError("context ID mismatch")
    if row.get("qa_pair_id") is not None and row["qa_pair_id"] != canonical["qa_pair_id"]:
        raise ValueError("question ID mismatch")
    if latent and (row["question"] != canonical["question"] or row["gold_answers"] != canonical["gold"]):
        raise ValueError("question/gold mismatch")
    query_hash = row.get("query_sha256") or row.get("official_query_sha256")
    if query_hash is not None and query_hash != canonical["query_sha256"]:
        raise ValueError("query hash mismatch")
    if latent:
        return "question_gold_context"
    if query_hash:
        return "official_query_hash"
    if context_id:
        return "context_id_query_index"
    raise ValueError("only positional identity available")


def paired(bank, other):
    b = {r["key"]: r for r in bank}
    t = {r["key"]: r for r in other}
    if set(b) != set(t):
        raise ValueError("paired identities differ")
    result = {}
    for subset, keys in [("all", list(b)), ("both_parsed_nonempty", [k for k in b if b[k]["parsed_nonempty"] and t[k]["parsed_nonempty"]]),
                         ("both_candidate_valid", [k for k in b if b[k]["candidate_valid"] and t[k]["candidate_valid"]])]:
        result[subset] = {}
        for metric in ("official_em", "strict_candidate_correct"):
            wins = sum(bool(b[k][metric]) and not bool(t[k][metric]) for k in keys)
            losses = sum(bool(t[k][metric]) and not bool(b[k][metric]) for k in keys)
            result[subset][metric] = {"n": len(keys), "p7_accuracy": fraction(sum(b[k][metric] for k in keys), len(keys)),
                                       "control_accuracy": fraction(sum(t[k][metric] for k in keys), len(keys)),
                                       "wins": wins, "losses": losses, "ties": len(keys) - wins - losses}
    bs, ts = summarize(bank), summarize(other)
    sb, st = bs["parsed_nonempty_rate"]["value"], ts["parsed_nonempty_rate"]["value"]
    ab, at = bs["parsed_conditional_em"]["value"], ts["parsed_conditional_em"]["value"]
    result["em_product_decomposition"] = {"delta_em": bs["em"]["value"] - ts["em"]["value"],
        "parsed_rate_term": (sb - st) * (ab + at) / 2 if ab is not None and at is not None else None,
        "conditional_em_term": (ab - at) * (sb + st) / 2 if ab is not None and at is not None else None}
    result["em_answer_partition"] = {"delta_em": bs["em"]["value"] - ts["em"]["value"],
        "strict_candidate_accuracy_delta": bs["strict_candidate_accuracy"]["value"] - ts["strict_candidate_accuracy"]["value"],
        "substring_only_accuracy_delta": bs["substring_only_accuracy"]["value"] - ts["substring_only_accuracy"]["value"]}
    return result


def across(summaries):
    keys = [k for k, v in summaries[0].items() if isinstance(v, dict) and "value" in v]
    result = {}
    for k in keys:
        values = [s[k]["value"] for s in summaries]
        defined = [v for v in values if v is not None]
        result[k] = {"values": values, "defined_runs": len(defined), "mean": statistics.fmean(defined) if defined else None,
                     "population_std": statistics.pstdev(defined) if defined else None}
    return result


def report_text(result):
    lines = ["# M1：官方评分、候选选择与格式分解", "", f"Gate: **{result['gate']}**。九种方法，五次重复；500 个唯一问题，不是 2,500 个独立样本。", "",
             "条件指标按每次运行计算再取均值；完整分子/分母、逐 context 和配对统计见 summary.json。", "",
             "| Method | Official EM | Raw recall | Parsed nonempty | Candidate valid | Strict candidate accuracy | Parsed conditional EM | Valid conditional strict accuracy | Heuristic flags |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in result["methods"].items():
        if item["status"] != "PASS":
            lines.append(f"| {name} | BLOCKED | | | | | | | |")
            continue
        keys = ["em", "raw_recall", "parsed_nonempty_rate", "candidate_valid_rate", "strict_candidate_accuracy", "parsed_conditional_em", "candidate_conditional_strict_accuracy", "heuristic_flag_rate"]
        values = [item["across_runs"][k]["mean"] for k in keys]
        lines.append("| " + name + " | " + " | ".join(f"{v:.4f}" if v is not None else "null" for v in values) + " |")
    chance = result["dataset"]["chance"]
    lines += ["", f"Candidate chance：{chance['value']:.4f}，有效题数 {chance['denominator']}/500。异常题及重复/多 gold 见 summary.json。这是始终输出合法候选的均匀选择理论参考，不是实测同协议模型 baseline。", "",
              "## 错误分解（每次 500 题的平均数量）", "", "| Method | Correct valid | Gold mentioned, wrong valid | Gold mentioned, no valid | Gold absent |", "|---|---:|---:|---:|---:|"]
    for name, item in result["methods"].items():
        if item["status"] == "PASS":
            lines.append("| " + name + " | " + " | ".join(f"{statistics.fmean(r['summary']['categories'][k] for r in item['runs']):.1f}" for k in CATEGORIES) + " |")
    lines += ["", "## P7 与对照的共同有效题目", "", "每个 seed 独立配对。下面是五次配对的平均分母及准确率，不作显著性检验。", "",
              "| Control | Both candidate-valid n | P7 strict acc | Control strict acc | P7 wins / losses |", "|---|---:|---:|---:|---:|"]
    for name, pair in result["paired"].items():
        items = [r["comparison"]["both_candidate_valid"]["strict_candidate_correct"] for r in pair]
        bp = [i["p7_accuracy"]["value"] for i in items if i["n"]]
        tp = [i["control_accuracy"]["value"] for i in items if i["n"]]
        lines.append(f"| {name} | {statistics.fmean(i['n'] for i in items):.1f} | {statistics.fmean(bp):.4f} | {statistics.fmean(tp):.4f} | {statistics.fmean(i['wins'] for i in items):.1f} / {statistics.fmean(i['losses'] for i in items):.1f} |" if bp else f"| {name} | 0 | null | null | 0 / 0 |")
    if "matched16" in result["paired"]:
        partitions = [p["comparison"]["em_answer_partition"] for p in result["paired"]["matched16"]]
        lines += ["", "## P7−Matched16 的 EM 记账分解", "",
                  f"平均 ΔEM = {statistics.fmean(p['delta_em'] for p in partitions):.4f} = "
                  f"Δ严格候选正确率 {statistics.fmean(p['strict_candidate_accuracy_delta'] for p in partitions):.4f} + "
                  f"Δ仅 substring 命中率 {statistics.fmean(p['substring_only_accuracy_delta'] for p in partitions):.4f}。",
                  "这是互斥正确输出类别的算术分解，不是检索机制或格式变化的因果归因。"]
    lines += ["", "## 解释边界", "", "- 官方 EM 是 substring EM；strict candidate accuracy 是新增诊断。", "- parser 能提取非空文本不代表答案是合法候选。heuristic flags 不是 parser failure。", "- raw recall 按官方多 gold 规则；分类中的 gold mentioned 指任一 gold，二者分别保留。", "- 共同有效集合仍是条件子集，不自动证明历史证据检索相关或公平控制了全部难度。", "- 种子目录不保证 controls 的实际随机性；沿用 M0 的历史来源限制。", "- 格式和有效选择的关联不能解释为因果。retrieval utility 待 M2–M4 验证。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    baseline = read(BASELINE)
    official_hash = sha(OFFICIAL)
    expected = next(s["sha256"] for s in baseline["current_source_identity"] if s["path"] == str(OFFICIAL))
    if official_hash != expected:
        raise ValueError("official scorer changed since M0")
    if sha(ROOT / baseline["paper"]["path"]) != baseline["paper"]["sha256"]:
        raise ValueError("submitted PDF changed")
    seen = set()
    for method in baseline["methods"].values():
        for run in method["runs"]:
            for f in run["files"]:
                if f["path"] not in seen:
                    if sha(ROOT / f["path"]) != f["sha256"]:
                        raise ValueError("M0 artifact changed: " + f["path"])
                    seen.add(f["path"])
    print(f"verified {len(seen)} frozen input hashes", flush=True)
    official = load_official()
    canonical = canonical_map(baseline["methods"]["p7"])
    infos = {key: candidate_info(c["question"], c["gold"], official) for key, c in canonical.items()}
    chance_values = [i["chance"] for i in infos.values() if i["chance"] is not None]
    dataset = {"chance": fraction(sum(chance_values), len(chance_values)),
               "candidate_count_distribution": dict(Counter(i["original_count"] for i in infos.values())),
               "duplicate_question_count": sum(bool(i["duplicate_count"]) for i in infos.values()),
               "multi_gold_question_count": sum(i["multi_gold"] for i in infos.values()),
               "anomalies": [{"key": list(k), "errors": i["errors"]} for k, i in infos.items() if i["errors"]]}

    @lru_cache(maxsize=None)
    def score(prediction, gold):
        metrics, additional = official._process_eventqa_dataset({"output": prediction}, list(gold))
        return metrics, additional["parsed_output"]

    methods, all_rows, mismatches = {}, {}, []
    for name, method in baseline["methods"].items():
        runs, method_rows, errors = [], {}, []
        latent = method["kind"] == "latent"
        prefix = "bank_on_" if latent else ""
        for run in method["runs"]:
            seed = run["seed"]
            rows = []
            identities = set()
            for raw, manifest, source in collect(run, method["kind"]):
                key = identity(raw, latent)
                try:
                    if key in identities or key not in canonical:
                        raise ValueError("duplicate/unknown question")
                    identities.add(key)
                    verification = check_identity(raw, manifest, canonical[key], latent)
                    prediction = raw[prefix + "prediction"]
                    metrics, parsed = score(prediction, tuple(canonical[key]["gold"]))
                    for metric, official_key in [("substring_exact_match", "substring_exact_match"), ("eventqa_recall", "eventqa_recall")]:
                        if float(metrics[official_key]) != float(raw[prefix + metric]):
                            raise ValueError(f"score mismatch {metric}: new={metrics[official_key]} saved={raw[prefix + metric]}")
                    if latent and parsed != raw["bank_on_parsed_prediction"]:
                        raise ValueError("parsed output mismatch")
                    row = evaluate(prediction, canonical[key]["gold"], infos[key], raw[prefix + "format_flags"], official, metrics)
                    row.update({"method": name, "seed": seed, "key": key, "context_id": canonical[key]["context_id"],
                                "prediction": prediction, "gold_answers": canonical[key]["gold"],
                                "selected_candidate": infos[key]["candidates"][infos[key]["normalized"].index(official.normalize_answer(parsed))] if row["candidate_valid"] else None,
                                "candidate_count": len(infos[key]["candidates"]), "chance": infos[key]["chance"],
                                "identity_verification": verification, "source": source})
                    rows.append(row)
                except (ValueError, TypeError, KeyError) as exc:
                    errors.append({"method": name, "seed": seed, "key": list(key), "source": source, "error": str(exc)})
            if identities != set(canonical) or len(rows) != 500:
                errors.append({"method": name, "seed": seed, "error": "incomplete scored/aligned question set", "scored": len(rows)})
            if len(rows) == 500:
                summary = summarize(rows)
                # Check official EM product identity, not causal attribution.
                conditional = summary["parsed_conditional_em"]["value"]
                if conditional is not None and abs(summary["em"]["value"] - summary["parsed_nonempty_rate"]["value"] * conditional) > 1e-12:
                    errors.append({"method": name, "seed": seed, "error": "EM product identity failed"})
                runs.append({"seed": seed, "summary": summary,
                             "per_context": {str(c): summarize([r for r in rows if r["key"][0] == c]) for c in range(5)}})
                method_rows[seed] = rows
        mismatches.extend(errors)
        methods[name] = {"status": "BLOCKED" if errors else "PASS", "runs": runs,
                         "across_runs": across([r["summary"] for r in runs]) if not errors else None,
                         "identity_verification": dict(Counter(r["identity_verification"] for rows in method_rows.values() for r in rows)), "errors": errors}
        if not errors:
            all_rows[name] = method_rows
        print(f"{name}: {methods[name]['status']} ({sum(len(r) for r in method_rows.values())} scores)", flush=True)
    pairs = {}
    if "p7" in all_rows:
        for name, seedrows in all_rows.items():
            if name != "p7":
                pairs[name] = [{"seed": seed, "comparison": paired(all_rows["p7"][seed], seedrows[seed]),
                                "per_context": {str(c): paired([r for r in all_rows['p7'][seed] if r['key'][0] == c], [r for r in seedrows[seed] if r['key'][0] == c]) for c in range(5)}} for seed in baseline["seeds"]]
    result = {"schema_version": "eventqa-review-m1/v1", "gate": "BLOCKED" if mismatches else "PASS_WITH_LIMITATIONS",
              "baseline_sha256": sha(BASELINE), "analyzer_sha256": sha(Path(__file__)), "official_scorer_sha256": official_hash,
              "benchmark_git_revision": subprocess.check_output(["git", "-C", str(OFFICIAL.parents[1]), "rev-parse", "HEAD"], text=True).strip(),
              "unique_questions": 500, "seeds": baseline["seeds"], "dataset": dataset, "methods": methods, "paired": pairs,
              "mismatches": mismatches, "score_cache": score.cache_info()._asdict()}
    if args.write:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (OUTPUT / "RESULTS.md").write_text(report_text(result), encoding="utf-8")
        with (OUTPUT / "per_question.jsonl").open("w", encoding="utf-8") as f:
            for seedrows in all_rows.values():
                for rows in seedrows.values():
                    for r in sorted(rows, key=lambda x: x["key"]):
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        examples = []
        for name, seedrows in all_rows.items():
            for category in CATEGORIES:
                pool = [r for r in seedrows[baseline['seeds'][0]] if r["category"] == category]
                examples.extend(sorted(pool, key=lambda r: r["key"])[:2])
        (OUTPUT / "examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"gate": result["gate"], "methods": len(all_rows), "mismatches": len(mismatches), "candidate_anomalies": len(dataset["anomalies"]), "chance": dataset["chance"]}, indent=2))
    return 2 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
