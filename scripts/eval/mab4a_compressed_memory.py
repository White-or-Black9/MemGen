"""MAB-4A: exploratory compressed-memory LatentBank run on one MAB context."""

import argparse
import gc
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from scripts.eval import mab2_bank_off as mab2
from scripts.eval import mab3_bank_on_full_history as mab3
from scripts.eval import mab3a_threshold_ablation as mab3a


BASELINE_NAME = "MAB-4A: LatentBank Compressed-memory Bank-on Exploratory"
PAIRED_MAB2_ARTIFACT = "outputs/mab/memgen_bank_off/20260620T034034Z-factconsolidation-sh-6k-onectx"
PAIRED_MAB3_ARTIFACT = "outputs/mab/memgen_bank_on_full_history/20260620T085407Z-factconsolidation-sh-6k-onectx"
PAIRED_MAB3A_ARTIFACT = "outputs/mab/memgen_bank_on_threshold_ablation/20260620T103852Z-factconsolidation-sh-6k-onectx"
LEAK_WINDOW = 128
LEAK_STEP = 64


def build_threshold_cases():
    return [
        {
            "label": "top_k_only",
            "threshold": None,
            "top_k_only": True,
            "retrieve_policy": "topk",
            "config_threshold": mab3a.TOP_K_ONLY_IGNORED_THRESHOLD,
        },
        {
            "label": "0.00",
            "threshold": 0.0,
            "top_k_only": False,
            "retrieve_policy": "threshold_topk",
            "config_threshold": 0.0,
        },
        {
            "label": "0.03",
            "threshold": 0.03,
            "top_k_only": False,
            "retrieve_policy": "threshold_topk",
            "config_threshold": 0.03,
        },
        {
            "label": "0.035",
            "threshold": 0.035,
            "top_k_only": False,
            "retrieve_policy": "threshold_topk",
            "config_threshold": 0.035,
        },
        {
            "label": "0.70",
            "threshold": 0.7,
            "top_k_only": False,
            "retrieve_policy": "threshold_topk",
            "config_threshold": 0.7,
        },
    ]


def bank_config_for_case(case):
    return mab3.version_a_bank_config(
        top_k=1,
        threshold=case["config_threshold"],
        retrieve_policy=case["retrieve_policy"],
    )


def build_compressed_query_messages(init_prompt, inter_history):
    if not init_prompt:
        raise RuntimeError("Missing init prompt for compressed query")
    if not inter_history or inter_history[-1].get("role") != "user":
        raise RuntimeError("Compressed query turn missing current user query")
    system_message = init_prompt[0]
    if system_message.get("role") != "system":
        raise RuntimeError("Expected system message at init prompt index 0")
    return [system_message, {"role": "user", "content": inter_history[-1]["content"]}]


def prompt_contains_chunk_leak(prompt_text, chunks, *, window=LEAK_WINDOW, step=LEAK_STEP):
    if not prompt_text:
        return False
    if len(prompt_text) < window:
        return False
    starts = range(0, len(prompt_text) - window + 1)
    for start in starts:
        snippet = prompt_text[start : start + window]
        for chunk in chunks:
            if len(chunk) >= window and snippet in chunk:
                return True
    return False


def summarize_threshold_result(*, case, diagnostics, prediction, gold_answers, score_value):
    query_diag = diagnostics[-1]
    return {
        "threshold": case["threshold"],
        "top_k_only": case["top_k_only"],
        "query_prompt_token_len": query_diag["prompt_history_token_len"],
        "query_prompt_contains_chunk_text": query_diag["query_prompt_contains_chunk_text"],
        "retrieved_latent_count_total": sum(int(item["retrieved_latent_count"]) for item in diagnostics),
        "retrieved_latent_count_by_turn": [int(item["retrieved_latent_count"]) for item in diagnostics],
        "retrieved_indices_by_turn": [list(item["retrieved_indices"]) for item in diagnostics],
        "retrieved_scores_by_turn": [list(item["retrieved_scores"]) for item in diagnostics],
        "prediction": prediction,
        "gold_answers": list(gold_answers),
        "substring_exact_match": int(bool(score_value)),
    }


def _build_manifest(run_id, args, started_at):
    return {
        "run_id": run_id,
        "status": "running",
        "baseline_name": BASELINE_NAME,
        "paired_mab2_artifact": args.paired_mab2_artifact,
        "paired_mab3_artifact": args.paired_mab3_artifact,
        "paired_mab3a_artifact": args.paired_mab3a_artifact,
        "history_policy": "compressed",
        "cross_turn_kv_reuse": False,
        "intra_generation_kv_cache": False,
        "batch_size": 1,
        "compressed_memory": True,
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


def _validate_payload(payload, paired_results, paired_manifest):
    if payload["context_id"] != paired_results["context_id"]:
        raise RuntimeError("MAB-4A context differs from paired artifacts")
    if payload["gold_answers"] != paired_results["gold_answers"]:
        raise RuntimeError("MAB-4A query/answer target differs from paired artifacts")
    if payload["chunk_token_lengths"] != paired_manifest["chunk_token_lengths"]:
        raise RuntimeError("MAB-4A official chunks differ from paired artifacts")


def _manager_class(chunks, query, capacity, prompt_trace, lifecycle, bank_trace):
    from interactions.multiturn_interaction import MultiTurnInteractionManager

    class CompressedManager(MultiTurnInteractionManager):
        def _create_session_memory_bank(self, actual_batch_size):
            bank = super()._create_session_memory_bank(actual_batch_size)
            lifecycle["create_count"] += 1
            if bank is None:
                raise RuntimeError("Enabled LatentMemoryBank was not created")
            lifecycle["bank"] = bank
            lifecycle["created_bank_id"] = id(bank)
            lifecycle["initial_slot_count"] = len(bank)
            if len(bank) != 0:
                raise RuntimeError("Bank contains nonzero slots at session start")
            mab3._install_bank_trace(bank, bank_trace)
            return bank

        def _build_chat_history(self, rollings):
            messages = super()._build_chat_history(rollings)
            turn = len(prompt_trace)
            if turn < len(chunks):
                required_chunks = chunks[: turn + 1]
                contents = [item.get("content", "") for item in messages[0]]
                for index, chunk in enumerate(required_chunks, start=1):
                    if not any(chunk in content for content in contents):
                        raise RuntimeError(f"Full-history audit failed on memorize turn {index}")
                rendered_messages = messages
                full_history_included = True
                leak_detected = None
            else:
                rendered_messages = [
                    build_compressed_query_messages(
                        rollings["init_prompts"][0],
                        rollings["inter_histories"][0],
                    )
                ]
                rendered_text = self.tokenizer.apply_chat_template(
                    rendered_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                leak_detected = prompt_contains_chunk_leak(rendered_text, chunks)
                if leak_detected:
                    raise RuntimeError("Compressed query prompt contains chunk text")
                full_history_included = False
            rendered = self.tokenizer.apply_chat_template(
                rendered_messages,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
                return_dict=True,
            )["input_ids"]
            length = int(rendered.shape[1])
            if length + 8 + self.config.max_response_length > capacity:
                raise RuntimeError(
                    f"Rendered history exceeds capacity: {length}+8+{self.config.max_response_length}>{capacity}"
                )
            prompt_trace.append({
                "prompt_history_token_len": length,
                "full_history_included": full_history_included,
                "query_prompt_contains_chunk_text": leak_detected,
                "rendered_prompt_hash": hashlib.sha256(rendered.cpu().numpy().tobytes()).hexdigest(),
            })
            return rendered_messages

        def run_agent_loop(self, gen_batch):
            try:
                return super().run_agent_loop(gen_batch)
            finally:
                bank = lifecycle.get("bank")
                if bank is not None:
                    lifecycle["final_debug_before_reset"] = bank.debug_summary()
                    bank.reset()
                    lifecycle["post_reset_slot_count"] = len(bank)

    return CompressedManager


def _run_model(args, payload, bank_config):
    import torch
    from interactions.base_interaction import InteractionDataProto
    from main import set_seed
    from memgen.model import MemGenModel

    set_seed(args.seed, use_gpu=True)
    preliminary = mab3._build_config(args, 32768, bank_config)
    model = MemGenModel.from_config(preliminary["model"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MAB-4A")
    model = model.to(device=torch.device("cuda"), dtype=torch.bfloat16)
    model.eval()
    capacity = int(getattr(model.reasoner.config, "max_position_embeddings", 0))
    if capacity <= 0:
        raise RuntimeError("Could not determine Reasoner context capacity")
    config_dict = mab3._build_config(args, capacity, bank_config)
    prompt_trace = []
    lifecycle = {"create_count": 0}
    model_trace = {
        "active": {},
        "generations": [],
        "generation_bank_ids": [],
        "weaver_prompt_calls": 0,
        "weaver_inference_calls": 0,
    }
    mab3._install_model_trace(model, model_trace)
    manager = _manager_class(
        payload["chunks"],
        payload["query_prompt"],
        capacity,
        prompt_trace,
        lifecycle,
        model_trace,
    )(
        model.tokenizer,
        model,
        mab3._interaction_config(config_dict, capacity),
    )
    env = mab2.MABEpisodeEnv(
        payload["memorization_prompts"][1:] + [payload["query_prompt"]],
        expected_turns=len(payload["chunks"]) + 1,
    )
    proto = InteractionDataProto()
    proto.no_tensor_batch["init_prompts"] = [[
        {
            "role": "system",
            "content": "You are a helpful assistant that can read the context and memorize it for future retrieval.",
        },
        {"role": "user", "content": payload["memorization_prompts"][0]},
    ]]
    proto.no_tensor_batch["envs"] = [env]
    manager.run_agent_loop(proto)
    mab3.assert_bank_lifecycle(
        create_count=lifecycle["create_count"],
        initial_slot_count=lifecycle["initial_slot_count"],
        generation_bank_ids=model_trace["generation_bank_ids"],
        created_bank_id=lifecycle["created_bank_id"],
        post_reset_slot_count=lifecycle["post_reset_slot_count"],
    )
    if env.final_answer is None:
        raise RuntimeError("Final answer could not be separated")
    return {
        "prediction": env.final_answer,
        "prompt_trace": prompt_trace,
        "model_trace": model_trace,
        "lifecycle": lifecycle,
        "context_capacity": capacity,
        "trigger_active_flag": bool(model.config.trigger_active),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=BASELINE_NAME)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--mab-repo", required=True)
    parser.add_argument("--mab-python", required=True)
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-checkpoint-id", required=True)
    parser.add_argument("--cfg-path", default="configs/latent_memory/triviaqa.yaml")
    parser.add_argument("--output-root", default="outputs/mab/memgen_bank_on_compressed_memory")
    parser.add_argument("--paired-mab2-artifact", default=PAIRED_MAB2_ARTIFACT)
    parser.add_argument("--paired-mab3-artifact", default=PAIRED_MAB3_ARTIFACT)
    parser.add_argument("--paired-mab3a-artifact", default=PAIRED_MAB3A_ARTIFACT)
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
    diagnostics = []
    results = {"threshold_results": []}
    try:
        paired_mab2_dir = Path(args.paired_mab2_artifact)
        paired_mab2_manifest = mab3a._load_json(paired_mab2_dir / "manifest.json")
        paired_mab2_results = mab3a._load_json(paired_mab2_dir / "results.json")
        paired_mab2_diagnostics = mab3a._read_jsonl(paired_mab2_dir / "diagnostics.jsonl")
        paired_mab2_hashes = [item["rendered_prompt_hash"] for item in paired_mab2_diagnostics]

        paired_mab3_dir = Path(args.paired_mab3_artifact)
        paired_mab3_results = mab3a._load_json(paired_mab3_dir / "results.json")
        paired_mab3a_dir = Path(args.paired_mab3a_artifact)
        paired_mab3a_results = mab3a._load_json(paired_mab3a_dir / "threshold_results.json")

        with tempfile.TemporaryDirectory(prefix="mab4a-compressed-") as tmpdir:
            payload_path = Path(tmpdir) / "payload.json"
            payload = mab3._prepare_payload(args, payload_path)
            _validate_payload(payload, paired_mab2_results, paired_mab2_manifest)
            _validate_payload(payload, paired_mab3_results, paired_mab2_manifest)
            for case in build_threshold_cases():
                bank_config = bank_config_for_case(case)
                model_result = None
                try:
                    model_result = _run_model(args, payload, bank_config)
                    if model_result["prompt_trace"][-1]["full_history_included"]:
                        raise RuntimeError("Compressed query unexpectedly included full history")
                    if model_result["prompt_trace"][-1]["query_prompt_contains_chunk_text"]:
                        raise RuntimeError("Compressed query prompt leaked chunk text")
                    if model_result["model_trace"]["generations"][0]["bank_debug"]["memory_write_count"] < 1:
                        raise RuntimeError("Chunk turns did not write memories")
                    if not case["top_k_only"] and case["threshold"] in {0.0, 0.03, 0.035}:
                        if model_result["model_trace"]["generations"][-1]["retrieved_latent_count"] == 0:
                            raise RuntimeError("Low-threshold compressed query did not retrieve")
                    score = mab3a._score_prediction(args, payload, model_result["prediction"], tmpdir)
                    case_diags = []
                    for index, (prompt, generation) in enumerate(zip(
                        model_result["prompt_trace"],
                        model_result["model_trace"]["generations"],
                    )):
                        debug = generation["bank_debug"]
                        case_diags.append({
                            "run_id": run_id,
                            "threshold_label": case["label"],
                            "threshold": case["threshold"],
                            "top_k_only": case["top_k_only"],
                            "history_policy": "compressed",
                            "context_id": payload["context_id"],
                            "query_id": 0 if index == len(payload["chunks"]) else None,
                            "turn_index": index,
                            "turn_type": "query" if index == len(payload["chunks"]) else "memorize_chunk",
                            "input_len": generation["input_len"],
                            "output_len": generation["output_len"],
                            **prompt,
                            "query_prompt_matches_full_history_mab2": (
                                prompt["rendered_prompt_hash"] == paired_mab2_hashes[index]
                            ),
                            "bank_created": True,
                            "bank_write_count": debug["memory_write_count"],
                            "bank_retrieval_count": debug["memory_retrieve_count"],
                            "bank_slot_count": debug["slot_count"],
                            "retrieved_indices": list(generation["retrieved_indices"]),
                            "retrieved_scores": list(generation["retrieved_scores"]),
                            "retrieved_latent_count": generation["retrieved_latent_count"],
                            "retrieved_latents_enter_reasoner": generation["retrieved_latents_enter_reasoner"],
                            "retrieved_latents_enter_weaver": generation["retrieved_latents_enter_weaver"],
                            "prediction": model_result["prediction"] if index == len(payload["chunks"]) else None,
                            "gold_answers": payload["gold_answers"] if index == len(payload["chunks"]) else None,
                            "latency_sec": generation["latency_sec"],
                            "peak_cuda_memory": generation["peak_cuda_memory"],
                            "error": None,
                        })
                        if generation["retrieved_latents_enter_weaver"]:
                            raise RuntimeError("Retrieved latents entered Weaver")
                    diagnostics.extend(case_diags)
                    results["threshold_results"].append(
                        summarize_threshold_result(
                            case=case,
                            diagnostics=case_diags,
                            prediction=model_result["prediction"],
                            gold_answers=payload["gold_answers"],
                            score_value=mab3a.extract_substring_exact_match(score),
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
                        mab3a.release_cuda_cache(torch, gc)
            manifest.update({
                "status": "success",
                "chunk_count": len(payload["chunks"]),
                "chunk_token_lengths": payload["chunk_token_lengths"],
                "full_history_query_tokens_reference": 7677,
                "context_capacity": results["threshold_results"][0]["query_prompt_token_len"] if results["threshold_results"] else None,
                "memoryagentbench_commit_or_path": mab2._run_capture_mab_commit(args.mab_repo),
                "threshold_cases": build_threshold_cases(),
            })
            results.update({
                "baseline_name": BASELINE_NAME,
                "context_id": payload["context_id"],
                "query_id": 0,
                "gold_answers": payload["gold_answers"],
                "paired_mab2_prediction": paired_mab2_results["prediction"],
                "paired_mab2_score": paired_mab2_results["substring_exact_match"],
                "paired_mab3_prediction": paired_mab3_results["prediction"],
                "paired_mab3_score": paired_mab3_results["substring_exact_match"],
                "paired_mab3a_threshold_results": paired_mab3a_results,
                "manifest_path": str(output_dir / "manifest.json"),
                "diagnostics_path": str(output_dir / "diagnostics.jsonl"),
            })
            manifest["context_capacity"] = paired_mab2_manifest.get("context_capacity")
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
        mab2._write_json(output_dir / "results.json", results)
        mab2._write_diagnostics(output_dir / "diagnostics.jsonl", diagnostics)
        mab2._write_json(output_dir / "run_config.json", {
            "baseline_name": BASELINE_NAME,
            "threshold_cases": build_threshold_cases(),
            "history_policy": "compressed",
            "batch_size": 1,
            "full_history_query_tokens_reference": 7677,
            "external_api_used": False,
        })
    print(str(output_dir))
    return 0 if manifest["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
