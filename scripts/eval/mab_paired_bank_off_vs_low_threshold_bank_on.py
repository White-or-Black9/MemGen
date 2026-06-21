"""Paired MAB Bank-off vs low-threshold Bank-on evaluation on deterministic contexts."""

import argparse
import gc
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from scripts.eval import mab2_bank_off as mab2
from scripts.eval import mab2_mab_bridge as bridge
from scripts.eval import mab3_bank_on_full_history as mab3
from scripts.eval import mab3a_threshold_ablation as mab3a


BASELINE_A = "Original MemGen Full-history Rebuild Bank-off"
BASELINE_B = "MemGen + LatentBank V-A Full-history Rebuild Bank-on"
SUB_DATASET = "factconsolidation_sh_6k"
THRESHOLD = 0.03


def select_match_indices(total_matches, requested):
    return list(range(min(total_matches, requested)))


def aggregate_results(rows, *, requested, attempted):
    valid = [row for row in rows if not row.get("error_or_stop_reason")]
    invalid = [row for row in rows if row.get("error_or_stop_reason")]
    valid_n = len(valid)
    bank_off_correct = sum(int(bool(row["bank_off_substring_exact_match"])) for row in valid)
    bank_on_correct = sum(int(bool(row["bank_on_substring_exact_match"])) for row in valid)
    retrieval_active = sum(int(row["bank_on_retrieved_latent_count"] > 0) for row in valid)
    changed = sum(int(bool(row["output_changed"])) for row in valid)
    improved = sum(
        int(row["bank_on_substring_exact_match"] > row["bank_off_substring_exact_match"])
        for row in valid
    )
    regressed = sum(
        int(row["bank_on_substring_exact_match"] < row["bank_off_substring_exact_match"])
        for row in valid
    )
    same_score = sum(
        int(row["bank_on_substring_exact_match"] == row["bank_off_substring_exact_match"])
        for row in valid
    )
    avg = lambda key: (sum(float(row[key]) for row in valid) / valid_n) if valid_n else None
    return {
        "num_contexts_requested": requested,
        "num_contexts_attempted": attempted,
        "num_contexts_valid": valid_n,
        "num_contexts_invalid": len(invalid),
        "bank_off_correct": bank_off_correct,
        "bank_on_correct": bank_on_correct,
        "bank_off_accuracy": (bank_off_correct / valid_n) if valid_n else None,
        "bank_on_accuracy": (bank_on_correct / valid_n) if valid_n else None,
        "delta_accuracy": ((bank_on_correct - bank_off_correct) / valid_n) if valid_n else None,
        "num_bank_on_retrieval_active": retrieval_active,
        "num_bank_on_output_changed_vs_bank_off": changed,
        "num_bank_on_improved": improved,
        "num_bank_on_regressed": regressed,
        "num_bank_on_same_score": same_score,
        "average_full_history_query_tokens": avg("full_history_query_tokens"),
        "average_chunk_count": avg("chunk_count"),
        "average_retrieved_latents": avg("bank_on_retrieved_latent_count"),
        "average_latency": avg("latency_total"),
        "peak_cuda_memory": max((row["peak_cuda_memory"] for row in valid if row["peak_cuda_memory"] is not None), default=None),
    }


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _count_matches(parquet_path, sub_dataset):
    rows = pq.read_table(parquet_path).to_pylist()
    return sum(1 for row in rows if row.get("metadata", {}).get("source") == sub_dataset)


def _prepare_payload(args, output_path, match_index):
    command = [
        args.mab_python,
        str(Path(bridge.__file__)),
        "prepare",
        "--mab-repo", args.mab_repo,
        "--output", str(output_path),
        "--parquet", args.parquet,
        "--data-config", args.data_config,
        "--sub-dataset", SUB_DATASET,
        "--chunk-size", "4096",
        "--timestamp", mab3.PINNED_MAB2_TIMESTAMP,
        "--match-index", str(match_index),
    ]
    import os
    import subprocess

    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    subprocess.run(command, check=True, env=env)
    return _load_json(output_path)


def _score_prediction(args, payload, prediction, tmpdir):
    score_request = Path(tmpdir) / "score_request.json"
    score_output = Path(tmpdir) / "score_output.json"
    _write_json(score_request, {
        "prediction": prediction,
        "gold_answers": payload["gold_answers"],
        "dataset_config": payload["dataset_config"],
    })
    return mab2._bridge(args, "score", score_request, score_output, score_output)


def _pair_row_result(context_index, payload, bank_off_result, bank_on_result, bank_off_score, bank_on_score):
    bank_on_query = bank_on_result["model_trace"]["generations"][-1]
    bank_on_debug = bank_on_query["bank_debug"]
    bank_off_query = bank_off_result["generation_trace"]["generations"][-1]
    parity = mab3.prompt_parity_summary(
        bank_on_hashes=[item["rendered_prompt_hash"] for item in bank_on_result["prompt_trace"]],
        bank_off_hashes=[item["rendered_prompt_hash"] for item in bank_off_result["prompt_trace"]],
    )
    return {
        "context_id": payload["context_id"],
        "query_id": 0,
        "context_index": context_index,
        "chunk_count": len(payload["chunks"]),
        "chunk_token_lengths": payload["chunk_token_lengths"],
        "full_history_query_tokens": bank_off_result["prompt_trace"][-1]["prompt_history_token_len"],
        "bank_off_prediction": bank_off_result["prediction"],
        "bank_off_substring_exact_match": int(bool(mab3a.extract_substring_exact_match(bank_off_score))),
        "bank_on_prediction": bank_on_result["prediction"],
        "bank_on_substring_exact_match": int(bool(mab3a.extract_substring_exact_match(bank_on_score))),
        "bank_on_threshold": THRESHOLD,
        "bank_on_write_count": bank_on_debug["memory_write_count"],
        "bank_on_retrieval_count": bank_on_debug["memory_retrieve_count"],
        "bank_on_retrieved_latent_count": sum(
            int(item["retrieved_latent_count"]) for item in bank_on_result["model_trace"]["generations"]
        ),
        "bank_on_slots_final_before_reset": bank_on_result["lifecycle"]["final_debug_before_reset"]["slot_count"],
        "bank_on_retrieved_indices_by_turn": [
            list(item["retrieved_indices"]) for item in bank_on_result["model_trace"]["generations"]
        ],
        "bank_on_retrieved_scores_by_turn": [
            list(item["retrieved_scores"]) for item in bank_on_result["model_trace"]["generations"]
        ],
        "bank_on_retrieved_latents_enter_reasoner": all(
            (not item["retrieved_latent_count"]) or item["retrieved_latents_enter_reasoner"]
            for item in bank_on_result["model_trace"]["generations"]
        ),
        "bank_on_retrieved_latents_enter_weaver": any(
            item["retrieved_latents_enter_weaver"] for item in bank_on_result["model_trace"]["generations"]
        ),
        "prompt_hash_parity": parity,
        "output_changed": bank_off_result["prediction"] != bank_on_result["prediction"],
        "improved": int(bool(mab3a.extract_substring_exact_match(bank_on_score))) > int(bool(mab3a.extract_substring_exact_match(bank_off_score))),
        "regressed": int(bool(mab3a.extract_substring_exact_match(bank_on_score))) < int(bool(mab3a.extract_substring_exact_match(bank_off_score))),
        "latency_total": (
            sum(item["latency_sec"] for item in bank_off_result["generation_trace"]["generations"]) +
            sum(item["latency_sec"] for item in bank_on_result["model_trace"]["generations"])
        ),
        "peak_cuda_memory": max(
            [item["peak_cuda_memory"] for item in bank_off_result["generation_trace"]["generations"] if item["peak_cuda_memory"] is not None] +
            [item["peak_cuda_memory"] for item in bank_on_result["model_trace"]["generations"] if item["peak_cuda_memory"] is not None],
            default=None,
        ),
        "error_or_stop_reason": None,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Paired Bank-off vs low-threshold Bank-on")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--mab-repo", required=True)
    parser.add_argument("--mab-python", required=True)
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-checkpoint-id", required=True)
    parser.add_argument("--cfg-path", default="configs/latent_memory/triviaqa.yaml")
    parser.add_argument("--output-root", default="outputs/mab/paired_bank_off_vs_low_threshold_bank_on")
    parser.add_argument("--requested-contexts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.parquet = str(Path(args.dataset_root) / "data/Conflict_Resolution-00000-of-00001.parquet")
    args.data_config = str(Path(args.mab_repo) / "configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml")
    return args


def main():
    args = build_parser()
    started_at = mab2._utc_now()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-factconsolidation-sh-6k-n10"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "run_id": run_id,
        "status": "running",
        "baseline_a": BASELINE_A,
        "baseline_b": BASELINE_B,
        "bank_on_threshold": THRESHOLD,
        "history_policy": "full_rebuild",
        "cross_turn_kv_reuse": False,
        "compressed_memory_used": False,
        "dataset_path": args.dataset_root,
        "split": mab2.SPLIT,
        "sub_dataset": SUB_DATASET,
        "memgen_branch": mab2._git("branch", "--show-current"),
        "memgen_git_status": mab2._git("status", "--short", "--branch"),
        "started_at": started_at,
        "finished_at": None,
        "stop_reason": None,
    }
    diagnostics = []
    paired_rows = []
    attempted = 0
    try:
        total_matches = _count_matches(args.parquet, SUB_DATASET)
        match_indices = select_match_indices(total_matches, args.requested_contexts)
        manifest["selection_policy"] = {
            "requested": args.requested_contexts,
            "matched_available": total_matches,
            "selected_match_indices": match_indices,
            "deterministic_order": "parquet_row_order_filtered_by_metadata.source",
        }
        with tempfile.TemporaryDirectory(prefix="mab-paired-") as tmpdir:
            for context_index, match_index in enumerate(match_indices):
                attempted += 1
                payload_path = Path(tmpdir) / f"payload_{context_index}.json"
                payload = _prepare_payload(args, payload_path, match_index)
                bank_off_result = None
                bank_on_result = None
                try:
                    bank_off_result = mab2._run_model(args, payload)
                    preflight_query_tokens = bank_off_result["prompt_trace"][-1]["prompt_history_token_len"]
                    context_capacity = bank_off_result["context_capacity"]
                    if preflight_query_tokens + 8 + 10 > context_capacity:
                        raise RuntimeError("full-history query exceeds context capacity")
                    try:
                        import torch
                    except Exception:
                        torch = None
                    if torch is not None:
                        mab3a.release_cuda_cache(torch, gc)
                    bank_on_result = mab3._run_model(
                        args,
                        payload,
                        mab3.version_a_bank_config(threshold=THRESHOLD, retrieve_policy="threshold_topk", top_k=1),
                    )
                    bank_off_score = _score_prediction(args, payload, bank_off_result["prediction"], tmpdir)
                    bank_on_score = _score_prediction(args, payload, bank_on_result["prediction"], tmpdir)
                    row = _pair_row_result(
                        context_index,
                        payload,
                        bank_off_result,
                        bank_on_result,
                        bank_off_score,
                        bank_on_score,
                    )
                    paired_rows.append(row)
                    for turn_index, generation in enumerate(bank_on_result["model_trace"]["generations"]):
                        diagnostics.append({
                            "run_id": run_id,
                            "context_id": payload["context_id"],
                            "context_index": context_index,
                            "query_id": 0 if turn_index == len(payload["chunks"]) else None,
                            "turn_index": turn_index,
                            "turn_type": "query" if turn_index == len(payload["chunks"]) else "memorize_chunk",
                            "full_history_query_tokens": preflight_query_tokens,
                            "context_capacity": context_capacity,
                            "bank_off_prompt_hash": bank_off_result["prompt_trace"][turn_index]["rendered_prompt_hash"],
                            "bank_on_prompt_hash": bank_on_result["prompt_trace"][turn_index]["rendered_prompt_hash"],
                            "bank_on_retrieved_indices": list(generation["retrieved_indices"]),
                            "bank_on_retrieved_scores": list(generation["retrieved_scores"]),
                            "bank_on_retrieved_latent_count": generation["retrieved_latent_count"],
                            "bank_on_retrieved_latents_enter_reasoner": generation["retrieved_latents_enter_reasoner"],
                            "bank_on_retrieved_latents_enter_weaver": generation["retrieved_latents_enter_weaver"],
                            "error": None,
                        })
                except Exception as error:
                    paired_rows.append({
                        "context_id": payload.get("context_id", f"match-{match_index}"),
                        "query_id": 0,
                        "context_index": context_index,
                        "chunk_count": len(payload.get("chunks", [])),
                        "chunk_token_lengths": payload.get("chunk_token_lengths"),
                        "full_history_query_tokens": None,
                        "bank_off_prediction": None,
                        "bank_off_substring_exact_match": None,
                        "bank_on_prediction": None,
                        "bank_on_substring_exact_match": None,
                        "bank_on_threshold": THRESHOLD,
                        "bank_on_write_count": None,
                        "bank_on_retrieval_count": None,
                        "bank_on_retrieved_latent_count": None,
                        "bank_on_slots_final_before_reset": None,
                        "bank_on_retrieved_indices_by_turn": None,
                        "bank_on_retrieved_scores_by_turn": None,
                        "bank_on_retrieved_latents_enter_reasoner": None,
                        "bank_on_retrieved_latents_enter_weaver": None,
                        "prompt_hash_parity": None,
                        "output_changed": None,
                        "improved": None,
                        "regressed": None,
                        "latency_total": None,
                        "peak_cuda_memory": None,
                        "error_or_stop_reason": f"{type(error).__name__}: {error}",
                    })
                    diagnostics.append({
                        "run_id": run_id,
                        "context_index": context_index,
                        "error": f"{type(error).__name__}: {error}",
                    })
                finally:
                    if bank_off_result is not None:
                        del bank_off_result
                    if bank_on_result is not None:
                        del bank_on_result
                    try:
                        import torch
                    except Exception:
                        torch = None
                    if torch is not None:
                        mab3a.release_cuda_cache(torch, gc)
        aggregate = aggregate_results(paired_rows, requested=args.requested_contexts, attempted=attempted)
        manifest.update({
            "status": "success",
            "aggregate": aggregate,
            "memoryagentbench_commit_or_path": mab2._run_capture_mab_commit(args.mab_repo),
        })
    except Exception as error:
        manifest["status"] = "invalid"
        manifest["stop_reason"] = f"{type(error).__name__}: {error}"
        diagnostics.append({"run_id": run_id, "error": manifest["stop_reason"]})
    finally:
        manifest["finished_at"] = mab2._utc_now()
        _write_json(output_dir / "manifest.json", manifest)
        _write_json(output_dir / "paired_results.json", {
            "aggregate": aggregate_results(paired_rows, requested=args.requested_contexts, attempted=attempted),
            "rows": paired_rows,
        })
        _write_jsonl(output_dir / "diagnostics.jsonl", diagnostics)
        _write_json(output_dir / "run_config.json", {
            "requested_contexts": args.requested_contexts,
            "sub_dataset": SUB_DATASET,
            "bank_on_threshold": THRESHOLD,
            "compressed_memory_used": False,
            "external_api_used": False,
        })
    print(str(output_dir))
    return 0 if manifest["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
