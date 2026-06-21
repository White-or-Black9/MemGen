"""MAB-2: Original MemGen Full-history Rebuild Bank-off, one context/query."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType, SimpleNamespace


BASELINE_NAME = "Original MemGen Full-history Rebuild Bank-off"
SUB_DATASET = "factconsolidation_sh_6k"
SPLIT = "Conflict_Resolution"


class MABEpisodeEnv:
    def __init__(self, subsequent_prompts, expected_turns):
        self.subsequent_prompts = list(subsequent_prompts)
        self.expected_turns = expected_turns
        self.turn = 0
        self.acknowledgements = []
        self.final_answer = None

    def preprocess_action(self, response):
        return response.strip()

    def step(self, response):
        if self.turn >= self.expected_turns:
            raise RuntimeError("step called after episode completion")
        if self.turn < self.expected_turns - 1:
            self.acknowledgements.append(response)
            observation = self.subsequent_prompts[self.turn]
            done = False
        else:
            self.final_answer = response
            observation = ""
            done = True
        self.turn += 1
        return observation, 0.0, done


def assert_full_history(messages, chunks, query):
    contents = [message.get("content", "") for message in messages]
    for index, chunk in enumerate(chunks, start=1):
        if not any(chunk in content for content in contents):
            raise RuntimeError(f"Full-history audit failed: missing prior chunk {index}")
    if not any(query in content for content in contents):
        raise RuntimeError("Full-history audit failed: missing query")


def assert_bank_off_invariants(**state):
    expected = {
        "bank_enabled": False,
        "bank_created": False,
        "bank_write_count": 0,
        "bank_retrieval_count": 0,
        "bank_slot_count": 0,
    }
    if any(state.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Bank-off invariant violated: {state}")


def build_manifest_skeleton(
    *, run_id, dataset_path, model_checkpoint, memgen_branch,
    memgen_git_status, started_at,
):
    return {
        "run_id": run_id,
        "status": "running",
        "baseline_name": BASELINE_NAME,
        "history_policy": "full_rebuild",
        "cross_turn_kv_reuse": False,
        "intra_generation_kv_cache": False,
        "bank_enabled": False,
        "bank_created": False,
        "bank_write_count": 0,
        "bank_retrieval_count": 0,
        "bank_slot_count": 0,
        "dataset_path": dataset_path,
        "split": SPLIT,
        "sub_dataset": SUB_DATASET,
        "num_contexts": 1,
        "num_queries": 1,
        "chunk_count": None,
        "chunk_token_lengths": None,
        "model_checkpoint": model_checkpoint,
        "memgen_branch": memgen_branch,
        "memgen_git_status": memgen_git_status,
        "memoryagentbench_commit_or_path": None,
        "started_at": started_at,
        "finished_at": None,
        "stop_reason": None,
    }


def _run(command, *, cwd=None):
    subprocess.run(command, cwd=cwd, check=True)


def _git(*args):
    return subprocess.run(
        ["git", *args], check=True, text=True, capture_output=True
    ).stdout.strip()


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_diagnostics(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _bridge(args, mode, bridge_input, bridge_output, payload_path):
    command = [
        args.mab_python,
        str(Path(__file__).with_name("mab2_mab_bridge.py")),
        mode,
        "--mab-repo", args.mab_repo,
        "--output", str(bridge_output),
    ]
    if mode == "prepare":
        command.extend([
            "--parquet", args.parquet,
            "--data-config", args.data_config,
            "--sub-dataset", SUB_DATASET,
            "--chunk-size", "4096",
        ])
    else:
        command.extend(["--input", str(bridge_input)])
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    subprocess.run(command, check=True, env=env)
    return json.loads(Path(payload_path).read_text(encoding="utf-8"))


def _build_config(args, context_capacity):
    from common.config import Config

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
        "run.latent_memory_bank.enabled", "False",
        "run.latent_memory_bank.batch_size", "1",
    ]
    return Config(SimpleNamespace(cfg_path=args.cfg_path, options=options)).to_dict()


def _interaction_config(config_dict, context_capacity):
    from interactions.base_interaction import InteractionConfig

    interaction = config_dict["run"]["interaction"]
    return InteractionConfig(
        max_turns=3,
        max_start_length=context_capacity,
        max_prompt_length=context_capacity,
        max_response_length=10,
        max_obs_length=context_capacity,
        temperature=0.0,
        batch_size=1,
        output_dir=None,
        weaver_do_sample=interaction.get("weaver_do_sample", False),
        trigger_do_sample=interaction.get("trigger_do_sample", False),
        latent_memory_bank=config_dict["run"].get("latent_memory_bank"),
    )


def _install_generation_trace(model, trace):
    import torch

    original_generate = model.generate
    original_prompt = model.weaver.augment_prompt
    original_inference = model.weaver.augment_inference

    def tracked_prompt(module, *args, **kwargs):
        trace["weaver_prompt_calls"] += 1
        return original_prompt(*args, **kwargs)

    def tracked_inference(module, *args, **kwargs):
        trace["weaver_inference_calls"] += 1
        return original_inference(*args, **kwargs)

    def tracked_generate(module, *args, **kwargs):
        if kwargs.get("latent_memory_bank") is not None:
            raise RuntimeError("Bank-off invariant violated: model received a bank")
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
        trace["generations"].append({
            "input_len": int(input_ids.shape[1]),
            "output_len": int(output_ids.shape[1] - input_ids.shape[1]),
            "trigger_count": int(mask.ne(-100).sum().item()),
            "trigger_positive_count": int(mask.eq(1).sum().item()),
            "latency_sec": time.perf_counter() - start,
            "peak_cuda_memory": peak,
        })
        return output_ids

    model.weaver.augment_prompt = MethodType(tracked_prompt, model.weaver)
    model.weaver.augment_inference = MethodType(tracked_inference, model.weaver)
    model.generate = MethodType(tracked_generate, model)


def _manager_class(chunks, query, capacity, prompt_trace, session_trace):
    from interactions.multiturn_interaction import MultiTurnInteractionManager

    class AuditedManager(MultiTurnInteractionManager):
        def _create_session_memory_bank(self, actual_batch_size):
            bank = super()._create_session_memory_bank(actual_batch_size)
            session_trace["session_count"] += 1
            session_trace["bank_created"] = bank is not None
            if bank is not None:
                raise RuntimeError("Bank-off invariant violated: bank was created")
            return None

        def _postprocess_observations(self, observations):
            lengths = [
                len(self.tokenizer.encode(obs, add_special_tokens=False))
                for obs in observations
            ]
            if any(length > self.config.max_obs_length for length in lengths):
                raise RuntimeError("Observation would be silently truncated")
            return super()._postprocess_observations(observations)

        def _build_chat_history(self, rollings):
            messages = super()._build_chat_history(rollings)
            turn = len(prompt_trace)
            required_chunks = chunks[: min(turn + 1, len(chunks))]
            required_query = query if turn == len(chunks) else None
            contents = [item.get("content", "") for item in messages[0]]
            for index, chunk in enumerate(required_chunks, start=1):
                if not any(chunk in content for content in contents):
                    raise RuntimeError(f"Full-history audit failed: missing prior chunk {index}")
            if required_query is not None:
                assert_full_history(messages[0], chunks, query)
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
                return_dict=True,
            )["input_ids"]
            length = int(rendered.shape[1])
            if length + 8 + self.config.max_response_length > capacity:
                raise RuntimeError(
                    f"Rendered history exceeds capacity: {length}+8+"
                    f"{self.config.max_response_length}>{capacity}"
                )
            prompt_trace.append({
                "prompt_history_token_len": length,
                "full_history_included": True,
                "rendered_prompt_hash": hashlib.sha256(
                    rendered.cpu().numpy().tobytes()
                ).hexdigest(),
            })
            return messages

    return AuditedManager


def _run_model(args, payload):
    import torch
    from interactions.base_interaction import InteractionDataProto
    from main import set_seed
    from memgen.model import MemGenModel

    set_seed(args.seed, use_gpu=True)
    preliminary = _build_config(args, 32768)
    model = MemGenModel.from_config(preliminary["model"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the MAB-2 MemGen run")
    model = model.to(device=torch.device("cuda"), dtype=torch.bfloat16)
    model.eval()
    capacity = int(getattr(model.reasoner.config, "max_position_embeddings", 0))
    if capacity <= 0:
        raise RuntimeError("Could not determine Reasoner context capacity")
    config_dict = _build_config(args, capacity)

    generation_trace = {
        "generations": [],
        "weaver_prompt_calls": 0,
        "weaver_inference_calls": 0,
    }
    prompt_trace = []
    session_trace = {"session_count": 0, "bank_created": False}
    _install_generation_trace(model, generation_trace)
    manager_cls = _manager_class(
        payload["chunks"], payload["query_prompt"], capacity,
        prompt_trace, session_trace,
    )
    manager = manager_cls(
        model.tokenizer,
        model,
        _interaction_config(config_dict, capacity),
    )
    env = MABEpisodeEnv(
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
    outputs = manager.run_agent_loop(proto)
    if session_trace["session_count"] != 1:
        raise RuntimeError("One MAB context was not mapped to exactly one session")
    if env.final_answer is None:
        raise RuntimeError("Final answer could not be separated from acknowledgements")
    if len(prompt_trace) != len(payload["chunks"]) + 1:
        raise RuntimeError("Unexpected number of rendered turn prompts")
    assert_bank_off_invariants(
        bank_enabled=False,
        bank_created=session_trace["bank_created"],
        bank_write_count=0,
        bank_retrieval_count=0,
        bank_slot_count=0,
    )
    return {
        "prediction": env.final_answer,
        "acknowledgements": env.acknowledgements,
        "prompt_trace": prompt_trace,
        "generation_trace": generation_trace,
        "session_trace": session_trace,
        "context_capacity": capacity,
        "conversation_turn_count": len(outputs.no_tensor_batch["inter_histories"][0]),
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
    parser.add_argument("--output-root", default="outputs/mab/memgen_bank_off")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.parquet = str(Path(args.dataset_root) / "data/Conflict_Resolution-00000-of-00001.parquet")
    args.data_config = str(Path(args.mab_repo) / "configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml")
    return args


def main():
    args = build_parser()
    started_at = _utc_now()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-factconsolidation-sh-6k-onectx"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    git_status = _git("status", "--short", "--branch")
    manifest = build_manifest_skeleton(
        run_id=run_id,
        dataset_path=args.dataset_root,
        model_checkpoint=args.model_checkpoint_id,
        memgen_branch=_git("branch", "--show-current"),
        memgen_git_status=git_status,
        started_at=started_at,
    )
    diagnostics = []
    results = None
    try:
        with tempfile.TemporaryDirectory(prefix="mab2-bank-off-") as tmpdir:
            payload_path = Path(tmpdir) / "payload.json"
            payload = _bridge(args, "prepare", None, payload_path, payload_path)
            manifest.update({
                "chunk_count": len(payload["chunks"]),
                "chunk_token_lengths": payload["chunk_token_lengths"],
                "memoryagentbench_commit_or_path": _run_capture_mab_commit(args.mab_repo),
            })
            model_result = _run_model(args, payload)
            score_request = Path(tmpdir) / "score_request.json"
            score_output = Path(tmpdir) / "score_output.json"
            _write_json(score_request, {
                "prediction": model_result["prediction"],
                "gold_answers": payload["gold_answers"],
                "dataset_config": payload["dataset_config"],
            })
            score = _bridge(args, "score", score_request, score_output, score_output)

            for index, (prompt, generation) in enumerate(zip(
                model_result["prompt_trace"],
                model_result["generation_trace"]["generations"],
            )):
                diagnostics.append({
                    "run_id": run_id,
                    "context_id": payload["context_id"],
                    "query_id": 0 if index == len(payload["chunks"]) else None,
                    "turn_index": index,
                    "turn_type": "query" if index == len(payload["chunks"]) else "memorize_chunk",
                    "input_len": generation["input_len"],
                    "output_len": generation["output_len"],
                    **prompt,
                    "bank_enabled": False,
                    "bank_created": False,
                    "bank_write_count": 0,
                    "bank_retrieval_count": 0,
                    "bank_slot_count": 0,
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
                "manifest_path": str(output_dir / "manifest.json"),
                "diagnostics_path": str(output_dir / "diagnostics.jsonl"),
            }
            manifest.update({
                "status": "success",
                "context_capacity": model_result["context_capacity"],
                "final_query_prompt_history_token_len": query_diag["prompt_history_token_len"],
                "full_history_included": True,
                "trigger_module_present": True,
                "trigger_active_flag": model_result["trigger_active_flag"],
                "weaver_prompt_call_count": model_result["generation_trace"]["weaver_prompt_calls"],
                "weaver_inference_call_count": model_result["generation_trace"]["weaver_inference_calls"],
            })
    except Exception as error:
        manifest["status"] = "invalid"
        manifest["stop_reason"] = f"{type(error).__name__}: {error}"
        diagnostics.append({
            "run_id": run_id,
            "turn_index": len(diagnostics),
            "error": manifest["stop_reason"],
            "bank_enabled": False,
            "bank_created": False,
            "bank_write_count": 0,
            "bank_retrieval_count": 0,
            "bank_slot_count": 0,
        })
    finally:
        manifest["finished_at"] = _utc_now()
        _write_json(output_dir / "manifest.json", manifest)
        _write_diagnostics(output_dir / "diagnostics.jsonl", diagnostics)
        if results is not None:
            _write_json(output_dir / "results.json", results)
        _write_json(output_dir / "run_config.json", {
            "baseline_name": BASELINE_NAME,
            "command": [Path(sys.executable).name, str(Path(__file__).relative_to(Path.cwd())), "<arguments redacted; see research note>"],
            "external_api_used": False,
            "batch_size": 1,
            "generation_max_length": 10,
        })
    print(str(output_dir))
    return 0 if manifest["status"] == "success" else 1


def _run_capture_mab_commit(repo):
    result = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True,
    )
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
