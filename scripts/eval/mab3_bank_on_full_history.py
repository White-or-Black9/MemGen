"""MAB-3: LatentBank Version A with full visible-history rebuild."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType, SimpleNamespace

from scripts.eval import mab2_bank_off as mab2


BASELINE_NAME = "MemGen + LatentBank V-A Full-history Rebuild Bank-on"
PAIRED_BASELINE_NAME = "Original MemGen Full-history Rebuild Bank-off"
PAIRED_ARTIFACT = "outputs/mab/memgen_bank_off/20260620T034034Z-factconsolidation-sh-6k-onectx"
PINNED_MAB2_TIMESTAMP = "2026-06-20 11:40:37"


def version_a_bank_config(
    *,
    top_k=1,
    threshold=0.7,
    retrieve_policy="threshold_topk",
):
    return {
        "enabled": True,
        "batch_size": 1,
        "max_slots": 8,
        "top_k": top_k,
        "threshold": threshold,
        "decay_alpha": 0.05,
        "pool_last_n": 64,
        "retrieve_policy": retrieve_policy,
        "update_policy": "thread_update",
        "storage_device": "cpu",
        "debug": True,
    }


def assert_bank_lifecycle(
    *, create_count, initial_slot_count, generation_bank_ids,
    created_bank_id, post_reset_slot_count,
):
    if create_count != 1:
        raise RuntimeError(f"Expected one bank creation, observed {create_count}")
    if initial_slot_count != 0:
        raise RuntimeError("Bank contained nonzero slots at session start")
    if not generation_bank_ids or any(
        bank_id != created_bank_id for bank_id in generation_bank_ids
    ):
        raise RuntimeError("All turns must share the same bank object")
    if post_reset_slot_count != 0:
        raise RuntimeError("Bank was not empty after session reset")


def assert_memory_boundary(
    *, retrieved_latent_count, retrieved_latents_enter_reasoner,
    retrieved_latents_enter_weaver, stored_latent_reasoner_space,
    stored_latent_detached_cloned,
):
    if retrieved_latents_enter_weaver:
        raise RuntimeError("Retrieved latents entered Weaver")
    if retrieved_latent_count and not retrieved_latents_enter_reasoner:
        raise RuntimeError("Retrieved latents did not enter Reasoner")
    if not stored_latent_reasoner_space:
        raise RuntimeError("Stored latent was not reasoner-space")
    if not stored_latent_detached_cloned:
        raise RuntimeError("Stored latent was not detached/cloned")


def prompt_parity_summary(*, bank_on_hashes, bank_off_hashes):
    if not bank_on_hashes or not bank_off_hashes or bank_on_hashes[0] != bank_off_hashes[0]:
        raise RuntimeError("MAB-3 initial visible prompt does not match MAB-2")
    return {
        "initial_prompt_exact_match": True,
        "all_turns_exact_match": bank_on_hashes == bank_off_hashes,
        "per_turn_exact_match": [
            left == right for left, right in zip(bank_on_hashes, bank_off_hashes)
        ],
        "later_difference_reason": "generated_acknowledgements_may_differ",
    }


def top_retrieval_scores(scores, limit=3):
    return sorted((float(score) for score in scores), reverse=True)[:limit]


def build_manifest_skeleton(
    *, run_id, dataset_path, model_checkpoint, memgen_branch,
    memgen_git_status, started_at,
):
    return {
        "run_id": run_id,
        "status": "running",
        "baseline_name": BASELINE_NAME,
        "paired_baseline_name": PAIRED_BASELINE_NAME,
        "paired_baseline_artifact": PAIRED_ARTIFACT,
        "history_policy": "full_rebuild",
        "cross_turn_kv_reuse": False,
        "intra_generation_kv_cache": False,
        "bank_enabled": True,
        "bank_created": False,
        "dataset_path": dataset_path,
        "split": mab2.SPLIT,
        "sub_dataset": mab2.SUB_DATASET,
        "num_contexts": 1,
        "num_queries": 1,
        "chunk_count": None,
        "chunk_token_lengths": None,
        "full_history_query_tokens": None,
        "context_capacity": None,
        "model_checkpoint": model_checkpoint,
        "memgen_branch": memgen_branch,
        "memgen_git_status": memgen_git_status,
        "started_at": started_at,
        "finished_at": None,
        "stop_reason": None,
    }


def _build_config(args, context_capacity, bank_config=None):
    from common.config import Config

    bank_config = bank_config or version_a_bank_config()
    options = [
        "model.model_name", args.model_path,
        "model.load_model_path", args.checkpoint_path,
        "model.weaver.model_name", args.model_path,
        "model.trigger.model_name", args.model_path,
        "run.mode", "evaluate",
        "run.seed", str(args.seed),
        "run.interaction.max_turns", "3",
        "run.interaction.max_start_length", str(context_capacity),
        "run.interaction.max_prompt_length", str(context_capacity),
        "run.interaction.max_obs_length", str(context_capacity),
        "run.interaction.max_response_length", "10",
        "run.interaction.batch_size", "1",
        "run.interaction.temperature", "0.0",
        "run.interaction.weaver_do_sample", "False",
        "run.interaction.trigger_do_sample", "False",
    ]
    for key, value in bank_config.items():
        options.extend([f"run.latent_memory_bank.{key}", str(value)])
    return Config(SimpleNamespace(cfg_path=args.cfg_path, options=options)).to_dict()


def _interaction_config(config_dict, context_capacity):
    from interactions.base_interaction import InteractionConfig

    return InteractionConfig(
        max_turns=3,
        max_start_length=context_capacity,
        max_prompt_length=context_capacity,
        max_response_length=10,
        max_obs_length=context_capacity,
        temperature=0.0,
        batch_size=1,
        output_dir=None,
        weaver_do_sample=False,
        trigger_do_sample=False,
        latent_memory_bank=config_dict["run"]["latent_memory_bank"],
    )


def _prepare_payload(args, output_path):
    command = [
        args.mab_python,
        str(Path(mab2.__file__).with_name("mab2_mab_bridge.py")),
        "prepare",
        "--mab-repo", args.mab_repo,
        "--output", str(output_path),
        "--parquet", args.parquet,
        "--data-config", args.data_config,
        "--sub-dataset", mab2.SUB_DATASET,
        "--chunk-size", "4096",
        "--timestamp", PINNED_MAB2_TIMESTAMP,
    ]
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    subprocess.run(command, check=True, env=env)
    return json.loads(output_path.read_text(encoding="utf-8"))


def _install_bank_trace(bank, trace):
    original_retrieve = bank.retrieve_with_context
    original_write_back = bank.write_back

    def tracked_retrieve(self, *args, **kwargs):
        result = original_retrieve(*args, **kwargs)
        trace["last_retrieval"] = {
            "scores": list(result.scores),
            "max_score": None if result.max_score is None else float(result.max_score),
            "argmax_index": result.argmax_index,
            "threshold_passed": bool(result.threshold_passed),
            "retrieved_indices": list(result.retrieved_indices),
            "retrieved_scores": list(result.retrieved_scores),
            "retrieved_slot_count": len(result.slots),
            "retrieved_latent_count": sum(int(slot.memory.shape[0]) for slot in result.slots),
            "retrieval_step": int(result.retrieval_step),
        }
        return result

    def tracked_write_back(self, memory, retrieval_result, *args, **kwargs):
        trace["last_write_input"] = memory
        trace["last_write_reasoner_space"] = (
            trace.get("last_weaver_to_reasoner") is not None
            and memory.shape == trace["last_weaver_to_reasoner"].shape
            and memory.data_ptr() == trace["last_weaver_to_reasoner"].data_ptr()
        )
        result = original_write_back(memory, retrieval_result, *args, **kwargs)
        normalized = memory.squeeze(0).detach().to("cpu")
        matching = [slot for slot in self._slots if slot.memory.shape == normalized.shape and slot.memory.equal(normalized)]
        trace["last_write_detached_cloned"] = bool(
            matching
            and not matching[-1].memory.requires_grad
            and matching[-1].memory.data_ptr() != normalized.data_ptr()
        )
        return result

    bank.retrieve_with_context = MethodType(tracked_retrieve, bank)
    bank.write_back = MethodType(tracked_write_back, bank)


def _install_model_trace(model, trace):
    import torch

    original_generate = model.generate
    original_r2w = model.reasoner_to_weaver.forward
    original_w2r = model.weaver_to_reasoner.forward
    original_prompt = model.weaver.augment_prompt
    original_inference = model.weaver.augment_inference
    original_reasoner_generate = model.reasoner.generate

    def tracked_r2w(module, tensor):
        trace["active"]["reasoner_to_weaver_input_len"] = int(tensor.shape[1])
        return original_r2w(tensor)

    def tracked_w2r(module, tensor):
        output = original_w2r(tensor)
        trace["last_weaver_to_reasoner"] = output
        return output

    def tracked_prompt(module, *args, **kwargs):
        trace["active"]["weaver_input_len"] = int(args[0].shape[1])
        trace["weaver_prompt_calls"] += 1
        return original_prompt(*args, **kwargs)

    def tracked_inference(module, *args, **kwargs):
        trace["weaver_inference_calls"] += 1
        return original_inference(*args, **kwargs)

    def tracked_reasoner_generate(module, *args, **kwargs):
        embeds = kwargs.get("inputs_embeds")
        trace["active"]["reasoner_augmented_input_len"] = int(embeds.shape[1])
        return original_reasoner_generate(*args, **kwargs)

    def tracked_generate(module, *args, **kwargs):
        bank = kwargs.get("latent_memory_bank")
        if bank is None:
            raise RuntimeError("Bank-on run called model without a bank")
        trace["generation_bank_ids"].append(id(bank))
        trace["active"] = {}
        trace["last_retrieval"] = {
            "scores": [],
            "max_score": None,
            "argmax_index": None,
            "threshold_passed": False,
            "retrieved_indices": [],
            "retrieved_scores": [],
            "retrieved_slot_count": 0,
            "retrieved_latent_count": 0,
            "retrieval_step": 0,
        }
        kwargs["return_augmentation_mask"] = True
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        output_ids, mask = original_generate(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = int(torch.cuda.max_memory_allocated())
        else:
            peak = None
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        retrieval = dict(trace["last_retrieval"])
        active = trace["active"]
        retrieved_count = retrieval["retrieved_latent_count"]
        expected_reasoner_len = (
            active["weaver_input_len"]
            + int(model.weaver.prompt_latents_num)
            + retrieved_count
        )
        enters_reasoner = bool(
            retrieved_count
            and active.get("reasoner_augmented_input_len") == expected_reasoner_len
        )
        enters_weaver = (
            active.get("weaver_input_len")
            != active.get("reasoner_to_weaver_input_len")
        )
        debug = bank.debug_summary()
        record = {
            "input_len": int(input_ids.shape[1]),
            "output_len": int(output_ids.shape[1] - input_ids.shape[1]),
            "trigger_count": int(mask.ne(-100).sum().item()),
            "trigger_positive_count": int(mask.eq(1).sum().item()),
            "latency_sec": time.perf_counter() - start,
            "peak_cuda_memory": peak,
            "bank_debug": debug,
            **retrieval,
            "retrieved_latents_enter_reasoner": enters_reasoner,
            "retrieved_latents_enter_weaver": enters_weaver,
            "stored_latent_reasoner_space": bool(trace.get("last_write_reasoner_space")),
            "stored_latent_detached_cloned": bool(trace.get("last_write_detached_cloned")),
        }
        assert_memory_boundary(
            retrieved_latent_count=retrieved_count,
            retrieved_latents_enter_reasoner=enters_reasoner,
            retrieved_latents_enter_weaver=enters_weaver,
            stored_latent_reasoner_space=record["stored_latent_reasoner_space"],
            stored_latent_detached_cloned=record["stored_latent_detached_cloned"],
        )
        trace["generations"].append(record)
        return output_ids

    model.reasoner_to_weaver.forward = MethodType(tracked_r2w, model.reasoner_to_weaver)
    model.weaver_to_reasoner.forward = MethodType(tracked_w2r, model.weaver_to_reasoner)
    model.weaver.augment_prompt = MethodType(tracked_prompt, model.weaver)
    model.weaver.augment_inference = MethodType(tracked_inference, model.weaver)
    model.reasoner.generate = MethodType(tracked_reasoner_generate, model.reasoner)
    model.generate = MethodType(tracked_generate, model)


def _manager_class(chunks, query, capacity, prompt_trace, lifecycle, bank_trace):
    base = mab2._manager_class(chunks, query, capacity - 8, prompt_trace, {})

    class BankOnManager(base):
        def _create_session_memory_bank(self, actual_batch_size):
            from interactions.base_interaction import InteractionManager

            bank = InteractionManager._create_session_memory_bank(self, actual_batch_size)
            lifecycle["create_count"] += 1
            if bank is None:
                raise RuntimeError("Enabled LatentMemoryBank was not created")
            lifecycle["bank"] = bank
            lifecycle["created_bank_id"] = id(bank)
            lifecycle["initial_slot_count"] = len(bank)
            if len(bank) != 0:
                raise RuntimeError("Bank contains nonzero slots at session start")
            _install_bank_trace(bank, bank_trace)
            return bank

        def run_agent_loop(self, gen_batch):
            try:
                return super().run_agent_loop(gen_batch)
            finally:
                bank = lifecycle.get("bank")
                if bank is not None:
                    lifecycle["final_debug_before_reset"] = bank.debug_summary()
                    bank.reset()
                    lifecycle["post_reset_slot_count"] = len(bank)

    return BankOnManager


def _run_model(args, payload, bank_config=None):
    import torch
    from interactions.base_interaction import InteractionDataProto
    from main import set_seed
    from memgen.model import MemGenModel

    set_seed(args.seed, use_gpu=True)
    bank_config = bank_config or version_a_bank_config()
    preliminary = _build_config(args, 32768, bank_config)
    model = MemGenModel.from_config(preliminary["model"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MAB-3")
    model = model.to(device=torch.device("cuda"), dtype=torch.bfloat16)
    model.eval()
    capacity = int(getattr(model.reasoner.config, "max_position_embeddings", 0))
    if capacity <= 0:
        raise RuntimeError("Could not determine Reasoner context capacity")
    config_dict = _build_config(args, capacity, bank_config)
    prompt_trace = []
    lifecycle = {"create_count": 0}
    bank_trace = {}
    model_trace = {
        "active": {},
        "generations": [],
        "generation_bank_ids": [],
        "weaver_prompt_calls": 0,
        "weaver_inference_calls": 0,
    }
    model_trace.update(bank_trace)
    _install_model_trace(model, model_trace)
    manager_cls = _manager_class(
        payload["chunks"], payload["query_prompt"], capacity,
        prompt_trace, lifecycle, model_trace,
    )
    manager = manager_cls(
        model.tokenizer, model, _interaction_config(config_dict, capacity)
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
    assert_bank_lifecycle(
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
    parser.add_argument("--output-root", default="outputs/mab/memgen_bank_on_full_history")
    parser.add_argument("--paired-artifact", default=PAIRED_ARTIFACT)
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
    manifest = build_manifest_skeleton(
        run_id=run_id,
        dataset_path=args.dataset_root,
        model_checkpoint=args.model_checkpoint_id,
        memgen_branch=mab2._git("branch", "--show-current"),
        memgen_git_status=mab2._git("status", "--short", "--branch"),
        started_at=started_at,
    )
    diagnostics = []
    results = None
    try:
        paired_dir = Path(args.paired_artifact)
        paired_manifest = json.loads((paired_dir / "manifest.json").read_text(encoding="utf-8"))
        paired_results = json.loads((paired_dir / "results.json").read_text(encoding="utf-8"))
        paired_diagnostics = [json.loads(line) for line in (paired_dir / "diagnostics.jsonl").read_text(encoding="utf-8").splitlines()]
        with tempfile.TemporaryDirectory(prefix="mab3-bank-on-") as tmpdir:
            payload_path = Path(tmpdir) / "payload.json"
            payload = _prepare_payload(args, payload_path)
            if payload["context_id"] != paired_results["context_id"]:
                raise RuntimeError("MAB-3 context differs from MAB-2")
            if payload["gold_answers"] != paired_results["gold_answers"]:
                raise RuntimeError("MAB-3 query/answer target differs from MAB-2")
            if payload["chunk_token_lengths"] != paired_manifest["chunk_token_lengths"]:
                raise RuntimeError("MAB-3 official chunks differ from MAB-2")
            bank_config = version_a_bank_config()
            model_result = _run_model(args, payload, bank_config)
            parity = prompt_parity_summary(
                bank_on_hashes=[item["rendered_prompt_hash"] for item in model_result["prompt_trace"]],
                bank_off_hashes=[item["rendered_prompt_hash"] for item in paired_diagnostics],
            )
            score_request = Path(tmpdir) / "score_request.json"
            score_output = Path(tmpdir) / "score_output.json"
            mab2._write_json(score_request, {
                "prediction": model_result["prediction"],
                "gold_answers": payload["gold_answers"],
                "dataset_config": payload["dataset_config"],
            })
            score = mab2._bridge(args, "score", score_request, score_output, score_output)
            for index, (prompt, generation) in enumerate(zip(
                model_result["prompt_trace"], model_result["model_trace"]["generations"]
            )):
                debug = generation["bank_debug"]
                diagnostics.append({
                    "run_id": run_id,
                    "context_id": payload["context_id"],
                    "query_id": 0 if index == len(payload["chunks"]) else None,
                    "turn_index": index,
                    "turn_type": "query" if index == len(payload["chunks"]) else "memorize_chunk",
                    "input_len": generation["input_len"],
                    "output_len": generation["output_len"],
                    **prompt,
                    "bank_enabled": True,
                    "bank_created": True,
                    "bank_write_count": debug["memory_write_count"],
                    "bank_retrieval_count": debug["memory_retrieve_count"],
                    "bank_slot_count": debug["slot_count"],
                    "replacement_count": debug["replace_count"],
                    "threshold": bank_config["threshold"],
                    "retrieved_slot_count": generation["retrieved_slot_count"],
                    "retrieved_latent_count": generation["retrieved_latent_count"],
                    "candidate_slot_indices": list(range(len(generation["scores"]))),
                    "candidate_raw_scores": list(generation["scores"]),
                    "top_retrieval_scores": top_retrieval_scores(generation["scores"]),
                    "max_score": generation["max_score"],
                    "matched_slot_index": generation["argmax_index"],
                    "threshold_passed": generation["threshold_passed"],
                    "retrieved_indices": list(generation["retrieved_indices"]),
                    "retrieved_scores": list(generation["retrieved_scores"]),
                    "retrieved_latents_enter_reasoner": generation["retrieved_latents_enter_reasoner"],
                    "retrieved_latents_enter_weaver": generation["retrieved_latents_enter_weaver"],
                    "trigger_count": generation["trigger_count"],
                    "trigger_positive_count": generation["trigger_positive_count"],
                    "weaver_call_count": generation["trigger_positive_count"],
                    "latency_sec": generation["latency_sec"],
                    "peak_cuda_memory": generation["peak_cuda_memory"],
                    "error": None,
                })
            query_diag = diagnostics[-1]
            results = {
                "context_id": payload["context_id"],
                "query_id": 0,
                "prediction": model_result["prediction"],
                "gold_answers": payload["gold_answers"],
                **score["additional"],
                **score["metrics"],
                "input_len": query_diag["input_len"],
                "output_len": query_diag["output_len"],
                "memory_construction_time": sum(item["latency_sec"] for item in diagnostics[:-1]),
                "query_time_len": query_diag["latency_sec"],
                "paired_bank_off_prediction": paired_results["prediction"],
                "paired_bank_off_score": paired_results["substring_exact_match"],
                "manifest_path": str(output_dir / "manifest.json"),
                "diagnostics_path": str(output_dir / "diagnostics.jsonl"),
            }
            final_debug = model_result["lifecycle"]["final_debug_before_reset"]
            manifest.update({
                "status": "success",
                "bank_created": True,
                "chunk_count": len(payload["chunks"]),
                "chunk_token_lengths": payload["chunk_token_lengths"],
                "full_history_query_tokens": query_diag["prompt_history_token_len"],
                "context_capacity": model_result["context_capacity"],
                "memoryagentbench_commit_or_path": mab2._run_capture_mab_commit(args.mab_repo),
                "prompt_parity": parity,
                "bank_final_before_reset": final_debug,
                "bank_post_reset_slot_count": model_result["lifecycle"]["post_reset_slot_count"],
                "same_bank_across_turns": True,
                "trigger_module_present": True,
                "trigger_active_flag": model_result["trigger_active_flag"],
                "weaver_prompt_call_count": model_result["model_trace"]["weaver_prompt_calls"],
                "weaver_inference_call_count": model_result["model_trace"]["weaver_inference_calls"],
            })
    except Exception as error:
        manifest["status"] = "invalid"
        manifest["stop_reason"] = f"{type(error).__name__}: {error}"
        diagnostics.append({
            "run_id": run_id,
            "turn_index": len(diagnostics),
            "error": manifest["stop_reason"],
            "bank_enabled": True,
            "bank_created": manifest.get("bank_created", False),
        })
    finally:
        manifest["finished_at"] = mab2._utc_now()
        mab2._write_json(output_dir / "manifest.json", manifest)
        mab2._write_diagnostics(output_dir / "diagnostics.jsonl", diagnostics)
        if results is not None:
            mab2._write_json(output_dir / "results.json", results)
        mab2._write_json(output_dir / "run_config.json", {
            "baseline_name": BASELINE_NAME,
            "paired_baseline_artifact": args.paired_artifact,
            "bank_config": version_a_bank_config(),
            "batch_size": 1,
            "generation_max_length": 10,
            "pinned_mab2_timestamp": PINNED_MAB2_TIMESTAMP,
            "external_api_used": False,
        })
    print(str(output_dir))
    return 0 if manifest["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
