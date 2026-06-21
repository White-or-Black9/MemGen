"""MAB-3A: full-history low-threshold retrieval ablation on one MAB context."""

import argparse
import gc
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from scripts.eval import mab2_bank_off as mab2
from scripts.eval import mab3_bank_on_full_history as mab3


BASELINE_NAME = "MAB-3A: LatentBank Full-history Low-threshold Retrieval Ablation"
PAIRED_MAB2_ARTIFACT = "outputs/mab/memgen_bank_off/20260620T034034Z-factconsolidation-sh-6k-onectx"
PAIRED_MAB3_ARTIFACT = "outputs/mab/memgen_bank_on_full_history/20260620T085407Z-factconsolidation-sh-6k-onectx"
TOP_K_ONLY_IGNORED_THRESHOLD = 0.7


def build_threshold_cases():
    cases = [
        {
            "label": "top_k_only",
            "threshold": None,
            "top_k_only": True,
            "retrieve_policy": "topk",
            "config_threshold": TOP_K_ONLY_IGNORED_THRESHOLD,
        }
    ]
    for value in [0.00, 0.01, 0.02, 0.03, 0.035, 0.04, 0.045, 0.05, 0.07, 0.10, 0.70]:
        cases.append({
            "label": f"{value:.3f}".rstrip("0").rstrip(".") if value not in {0.00, 0.10, 0.70} else f"{value:.2f}",
            "threshold": value,
            "top_k_only": False,
            "retrieve_policy": "threshold_topk",
            "config_threshold": value,
        })
    return cases


def bank_config_for_case(case):
    return mab3.version_a_bank_config(
        top_k=1,
        threshold=case["config_threshold"],
        retrieve_policy=case["retrieve_policy"],
    )


def candidate_score_pairs(scores):
    return [
        {"slot_index": index, "score": float(score)}
        for index, score in enumerate(scores)
    ]


def count_slots_passing_threshold(scores, threshold):
    if threshold is None:
        return None
    return sum(float(score) >= threshold for score in scores)


def summarize_threshold_result(*, case, diagnostics, prediction, gold_answers, score_value):
    turn2 = diagnostics[1]
    turn3 = diagnostics[2]
    notes = []
    if case["top_k_only"]:
        notes.append(
            f"retrieve_policy=topk; config threshold kept at {case['config_threshold']:.2f} but ignored by selection"
        )
    if any(diag["retrieved_latent_count"] > 0 for diag in diagnostics):
        notes.append("retrieved_latent_injection_active")
    if prediction != "":
        notes.append("scoreable_output")
    return {
        "threshold": case["threshold"],
        "top_k_only": case["top_k_only"],
        "max_score_turn2": turn2["max_score"],
        "max_score_turn3": turn3["max_score"],
        "slots_passing_threshold_turn2": count_slots_passing_threshold(
            turn2["candidate_raw_scores"], case["threshold"]
        ),
        "slots_passing_threshold_turn3": count_slots_passing_threshold(
            turn3["candidate_raw_scores"], case["threshold"]
        ),
        "retrieved_latent_count_total": sum(
            int(diag["retrieved_latent_count"]) for diag in diagnostics
        ),
        "retrieved_latent_count_by_turn": [
            int(diag["retrieved_latent_count"]) for diag in diagnostics
        ],
        "retrieved_indices_by_turn": [
            list(diag["retrieved_indices"]) for diag in diagnostics
        ],
        "retrieved_scores_by_turn": [
            list(diag["retrieved_scores"]) for diag in diagnostics
        ],
        "prediction": prediction,
        "gold_answers": list(gold_answers),
        "substring_exact_match": int(bool(score_value)),
        "notes": "; ".join(notes),
    }


def extract_substring_exact_match(score_payload):
    if "metrics" in score_payload and "substring_exact_match" in score_payload["metrics"]:
        return int(bool(score_payload["metrics"]["substring_exact_match"]))
    if "additional" in score_payload and "substring_exact_match" in score_payload["additional"]:
        return int(bool(score_payload["additional"]["substring_exact_match"]))
    raise KeyError("substring_exact_match")


def release_cuda_cache(torch_module, gc_module=gc):
    gc_module.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _build_case_diagnostics(
    *, run_id, case, payload, prompt_trace, model_trace, paired_mab2_hashes, paired_mab3_hashes
):
    diagnostics = []
    for index, (prompt, generation) in enumerate(zip(prompt_trace, model_trace["generations"])):
        debug = generation["bank_debug"]
        prompt_hash = prompt["rendered_prompt_hash"]
        diagnostics.append({
            "run_id": run_id,
            "threshold_label": case["label"],
            "threshold": case["threshold"],
            "top_k_only": case["top_k_only"],
            "retrieve_policy": case["retrieve_policy"],
            "context_id": payload["context_id"],
            "query_id": 0 if index == len(payload["chunks"]) else None,
            "turn_index": index,
            "turn_type": "query" if index == len(payload["chunks"]) else "memorize_chunk",
            "input_len": generation["input_len"],
            "output_len": generation["output_len"],
            **prompt,
            "full_history_included": bool(prompt["full_history_included"]),
            "prompt_hash_matches_mab2": prompt_hash == paired_mab2_hashes[index],
            "prompt_hash_matches_mab3": prompt_hash == paired_mab3_hashes[index],
            "bank_enabled": True,
            "bank_created": True,
            "bank_write_count": debug["memory_write_count"],
            "bank_retrieval_count": debug["memory_retrieve_count"],
            "bank_slot_count": debug["slot_count"],
            "replacement_count": debug["replace_count"],
            "candidate_slot_indices": list(range(len(generation["scores"]))),
            "candidate_raw_scores": [float(score) for score in generation["scores"]],
            "candidate_score_pairs": candidate_score_pairs(generation["scores"]),
            "max_score": generation["max_score"],
            "matched_slot_index": generation["argmax_index"],
            "threshold_passed": generation["threshold_passed"],
            "retrieved_slot_count": generation["retrieved_slot_count"],
            "retrieved_indices": list(generation["retrieved_indices"]),
            "retrieved_scores": [float(score) for score in generation["retrieved_scores"]],
            "retrieved_latent_count": generation["retrieved_latent_count"],
            "top_retrieval_scores": mab3.top_retrieval_scores(generation["scores"]),
            "retrieved_latents_enter_reasoner": generation["retrieved_latents_enter_reasoner"],
            "retrieved_latents_enter_weaver": generation["retrieved_latents_enter_weaver"],
            "trigger_count": generation["trigger_count"],
            "trigger_positive_count": generation["trigger_positive_count"],
            "weaver_call_count": generation["trigger_positive_count"],
            "latency_sec": generation["latency_sec"],
            "peak_cuda_memory": generation["peak_cuda_memory"],
            "error": None,
        })
    return diagnostics


def _build_manifest(run_id, args, started_at):
    return {
        "run_id": run_id,
        "status": "running",
        "baseline_name": BASELINE_NAME,
        "paired_mab2_artifact": args.paired_mab2_artifact,
        "paired_mab3_artifact": args.paired_mab3_artifact,
        "history_policy": "full_rebuild",
        "cross_turn_kv_reuse": False,
        "intra_generation_kv_cache": False,
        "batch_size": 1,
        "compressed_memory": False,
        "dataset_path": args.dataset_root,
        "split": mab2.SPLIT,
        "sub_dataset": mab2.SUB_DATASET,
        "num_contexts": 1,
        "num_queries": 1,
        "model_checkpoint": args.model_checkpoint_id,
        "memgen_branch": mab2._git("branch", "--show-current"),
        "memgen_git_status": mab2._git("status", "--short", "--branch"),
        "started_at": started_at,
        "finished_at": None,
        "stop_reason": None,
    }


def _score_prediction(args, payload, prediction, tmpdir):
    score_request = Path(tmpdir) / "score_request.json"
    score_output = Path(tmpdir) / "score_output.json"
    mab2._write_json(score_request, {
        "prediction": prediction,
        "gold_answers": payload["gold_answers"],
        "dataset_config": payload["dataset_config"],
    })
    return mab2._bridge(args, "score", score_request, score_output, score_output)


def _validate_payload(payload, paired_results, paired_manifest):
    if payload["context_id"] != paired_results["context_id"]:
        raise RuntimeError("MAB-3A context differs from MAB-2/MAB-3")
    if payload["gold_answers"] != paired_results["gold_answers"]:
        raise RuntimeError("MAB-3A query/answer target differs from MAB-2/MAB-3")
    if payload["chunk_token_lengths"] != paired_manifest["chunk_token_lengths"]:
        raise RuntimeError("MAB-3A official chunks differ from paired artifacts")


def build_parser():
    parser = argparse.ArgumentParser(description=BASELINE_NAME)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--mab-repo", required=True)
    parser.add_argument("--mab-python", required=True)
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-checkpoint-id", required=True)
    parser.add_argument("--cfg-path", default="configs/latent_memory/triviaqa.yaml")
    parser.add_argument("--output-root", default="outputs/mab/memgen_bank_on_threshold_ablation")
    parser.add_argument("--paired-mab2-artifact", default=PAIRED_MAB2_ARTIFACT)
    parser.add_argument("--paired-mab3-artifact", default=PAIRED_MAB3_ARTIFACT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.parquet = str(Path(args.dataset_root) / "data/Conflict_Resolution-00000-of-00001.parquet")
    args.data_config = str(Path(args.mab_repo) / "configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml")
    return args


def main():
    args = build_parser()
    started_at = mab2._utc_now()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-factconsolidation-sh-6k-onectx"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = _build_manifest(run_id, args, started_at)
    threshold_results = []
    diagnostics = []
    try:
        paired_mab2_dir = Path(args.paired_mab2_artifact)
        paired_mab2_manifest = _load_json(paired_mab2_dir / "manifest.json")
        paired_mab2_results = _load_json(paired_mab2_dir / "results.json")
        paired_mab2_diagnostics = _read_jsonl(paired_mab2_dir / "diagnostics.jsonl")
        paired_mab2_hashes = [item["rendered_prompt_hash"] for item in paired_mab2_diagnostics]

        paired_mab3_dir = Path(args.paired_mab3_artifact)
        paired_mab3_manifest = _load_json(paired_mab3_dir / "manifest.json")
        paired_mab3_diagnostics = _read_jsonl(paired_mab3_dir / "diagnostics.jsonl")
        paired_mab3_hashes = [item["rendered_prompt_hash"] for item in paired_mab3_diagnostics]

        with tempfile.TemporaryDirectory(prefix="mab3a-threshold-ablation-") as tmpdir:
            payload_path = Path(tmpdir) / "payload.json"
            payload = mab3._prepare_payload(args, payload_path)
            _validate_payload(payload, paired_mab2_results, paired_mab2_manifest)
            _validate_payload(payload, _load_json(paired_mab3_dir / "results.json"), paired_mab3_manifest)

            for case in build_threshold_cases():
                bank_config = bank_config_for_case(case)
                model_result = None
                try:
                    model_result = mab3._run_model(args, payload, bank_config)
                    mab3.prompt_parity_summary(
                        bank_on_hashes=[item["rendered_prompt_hash"] for item in model_result["prompt_trace"]],
                        bank_off_hashes=paired_mab2_hashes,
                    )
                    score = _score_prediction(args, payload, model_result["prediction"], tmpdir)
                    case_diagnostics = _build_case_diagnostics(
                        run_id=run_id,
                        case=case,
                        payload=payload,
                        prompt_trace=model_result["prompt_trace"],
                        model_trace=model_result["model_trace"],
                        paired_mab2_hashes=paired_mab2_hashes,
                        paired_mab3_hashes=paired_mab3_hashes,
                    )
                    diagnostics.extend(case_diagnostics)
                    threshold_results.append(
                        summarize_threshold_result(
                            case=case,
                            diagnostics=case_diagnostics,
                            prediction=model_result["prediction"],
                            gold_answers=payload["gold_answers"],
                            score_value=extract_substring_exact_match(score),
                        )
                    )
                finally:
                    if model_result is not None:
                        del model_result
                    try:
                        import torch
                    except Exception:
                        torch = None
                    if torch is not None:
                        release_cuda_cache(torch)

            manifest.update({
                "status": "success",
                "chunk_count": len(payload["chunks"]),
                "chunk_token_lengths": payload["chunk_token_lengths"],
                "context_capacity": paired_mab3_manifest["context_capacity"],
                "full_history_query_tokens": paired_mab3_manifest["full_history_query_tokens"],
                "memoryagentbench_commit_or_path": mab2._run_capture_mab_commit(args.mab_repo),
                "threshold_cases": build_threshold_cases(),
                "prompt_hash_reference": {
                    "paired_mab2_first_turn": paired_mab2_hashes[0],
                    "paired_mab3_first_turn": paired_mab3_hashes[0],
                },
                "candidate_score_logging": "enabled_pre_threshold_raw_scores",
                "same_context_as_mab2": True,
                "same_context_as_mab3": True,
            })
    except Exception as error:
        manifest["status"] = "invalid"
        manifest["stop_reason"] = f"{type(error).__name__}: {error}"
        diagnostics.append({
            "run_id": run_id,
            "turn_index": len(diagnostics),
            "error": manifest["stop_reason"],
        })
    finally:
        manifest["finished_at"] = mab2._utc_now()
        mab2._write_json(output_dir / "manifest.json", manifest)
        mab2._write_json(output_dir / "threshold_results.json", threshold_results)
        mab2._write_diagnostics(output_dir / "diagnostics.jsonl", diagnostics)
        mab2._write_json(output_dir / "run_config.json", {
            "baseline_name": BASELINE_NAME,
            "paired_mab2_artifact": args.paired_mab2_artifact,
            "paired_mab3_artifact": args.paired_mab3_artifact,
            "threshold_cases": build_threshold_cases(),
            "fixed_controls": {
                "top_k": 1,
                "max_slots": 8,
                "decay_alpha": 0.05,
                "pool_last_n": 64,
                "update_policy": "thread_update",
                "history_policy": "full_rebuild",
                "cross_turn_kv_reuse": False,
                "batch_size": 1,
            },
            "pinned_mab2_timestamp": mab3.PINNED_MAB2_TIMESTAMP,
            "external_api_used": False,
        })
    print(str(output_dir))
    return 0 if manifest["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
