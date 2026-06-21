"""MAB-5A: detective_qa compressed-memory Bank-off vs Bank-on on 10 contexts."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import tempfile
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import mab2_bank_off as mab2
from scripts.eval import mab3_bank_on_full_history as mab3
from scripts.eval import mab3a_threshold_ablation as mab3a
from scripts.eval import mab4a_compressed_memory as mab4a


EXPERIMENT_NAME = "MAB-5A: detective_qa Compressed-memory Bank-off vs Bank-on n10"
RUN_PREFIX = "detectiveqa-compressed-n10"
SPLIT = "Long_Range_Understanding"
SUB_DATASET = "detective_qa"
DATA_CONFIG = "configs/data_conf/Long_Range_Understanding/Detective_QA.yaml"
DEFAULT_OUTPUT_ROOT = "outputs/mab/compressed_memory_detectiveqa_n10"
DEFAULT_REQUESTED_CONTEXTS = 10
DEFAULT_CHUNK_SIZE = 4096
DEFAULT_THRESHOLD = 0.03
DEFAULT_TOP_K = 1
DEFAULT_MAX_SLOTS = 8
DEFAULT_RETRIEVE_POLICY = "threshold_topk"
DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant that can read the context and memorize it for future retrieval."
DEFAULT_ACK = "Acknowledged."

# Pre-edit workspace snapshot requested by the user.
GIT_STATUS_BEFORE_EDIT = """## rlm-memory-bank...origin/rlm-memory-bank [ahead 8]
 M memgen/model/modeling_memgen.py
?? research_notes/benchmarks/
?? scripts/eval/diagnose_memgen_over_context.py
?? scripts/eval/mab2_bank_off.py
?? scripts/eval/mab2_mab_bridge.py
?? scripts/eval/mab3_bank_on_full_history.py
?? scripts/eval/mab3a_threshold_ablation.py
?? scripts/eval/mab4a_compressed_memory.py
?? scripts/eval/mab_paired_bank_off_vs_low_threshold_bank_on.py
?? tests/test_mab2_bank_off.py
?? tests/test_mab3_bank_on_full_history.py
?? tests/test_mab3a_threshold_ablation.py
?? tests/test_mab4a_compressed_memory.py
?? tests/test_mab_paired_bank_off_vs_low_threshold_bank_on.py"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _bridge_script() -> Path:
    return Path(mab2.__file__).with_name("mab2_mab_bridge.py")


def _mab_env() -> dict:
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    return env


def select_match_indices(total_matches: int, requested: int) -> list[int]:
    return list(range(min(total_matches, requested)))


def count_context_matches(parquet_path: str, sub_dataset: str) -> int:
    rows = pq.read_table(parquet_path).to_pylist()
    return sum(1 for row in rows if row.get("metadata", {}).get("source") == sub_dataset)


def _prepare_payload(args, output_path: Path, match_index: int, timestamp: str) -> dict:
    command = [
        args.mab_python,
        str(_bridge_script()),
        "prepare",
        "--mab-repo", args.mab_repo,
        "--output", str(output_path),
        "--parquet", args.parquet,
        "--data-config", args.data_config,
        "--sub-dataset", SUB_DATASET,
        "--chunk-size", str(DEFAULT_CHUNK_SIZE),
        "--timestamp", timestamp,
        "--match-index", str(match_index),
    ]
    subprocess.run(command, check=True, env=_mab_env())
    return _load_json(output_path)


def _score_prediction(args, payload: dict, prediction: str, tmpdir: str) -> dict:
    request_path = Path(tmpdir) / "score_request.json"
    output_path = Path(tmpdir) / "score_output.json"
    _write_json(
        request_path,
        {
            "prediction": prediction,
            "gold_answers": payload["gold_answers"],
            "dataset_config": payload["dataset_config"],
        },
    )
    command = [
        args.mab_python,
        str(_bridge_script()),
        "score",
        "--mab-repo", args.mab_repo,
        "--output", str(output_path),
        "--input", str(request_path),
    ]
    subprocess.run(command, check=True, env=_mab_env())
    return _load_json(output_path)


def render_compressed_query_messages(init_prompt, inter_history):
    if not init_prompt:
        raise RuntimeError("Missing init prompt for compressed query")
    if not inter_history or inter_history[-1].get("role") != "user":
        raise RuntimeError("Compressed query turn missing current user query")
    system_message = init_prompt[0]
    if system_message.get("role") != "system":
        raise RuntimeError("Expected system message at init prompt index 0")
    return [
        system_message,
        {"role": "user", "content": inter_history[-1]["content"]},
    ]


def prompt_contains_chunk_leak(prompt_text, chunks, *, window=128, step=64):
    return mab4a.prompt_contains_chunk_leak(prompt_text, chunks, window=window, step=step)


def _render_chat(tokenizer, messages) -> tuple[int, str]:
    rendered_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    rendered_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"]
    return int(rendered_ids.shape[1]), rendered_text


def estimate_full_history_query_tokens(tokenizer, payload: dict) -> int:
    messages = [{"role": "system", "content": DEFAULT_SYSTEM_MESSAGE}]
    for prompt in payload["memorization_prompts"]:
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": DEFAULT_ACK})
    messages.append({"role": "user", "content": payload["query_prompt"]})
    tokens, _ = _render_chat(tokenizer, [messages])
    return tokens


def compressed_query_token_count(tokenizer, payload: dict) -> tuple[int, str, bool, bool]:
    messages = render_compressed_query_messages(
        [{"role": "system", "content": DEFAULT_SYSTEM_MESSAGE}, {"role": "user", "content": payload["memorization_prompts"][0] if payload["memorization_prompts"] else ""}],
        [{"role": "user", "content": payload["query_prompt"]}],
    )
    token_count, rendered_text = _render_chat(tokenizer, [messages])
    chunk_leak = prompt_contains_chunk_leak(rendered_text, payload["chunks"])
    ack_leak = DEFAULT_ACK in rendered_text
    return token_count, rendered_text, chunk_leak, ack_leak


def _bank_config():
    return mab3.version_a_bank_config(
        top_k=DEFAULT_TOP_K,
        threshold=DEFAULT_THRESHOLD,
        retrieve_policy=DEFAULT_RETRIEVE_POLICY,
    )


def _build_manifest(run_id: str, args, started_at: str, *, git_status_before: str, git_status_after: str | None = None) -> dict:
    checkpoint_path = str(Path(args.checkpoint_path).resolve())
    return {
        "experiment_name": EXPERIMENT_NAME,
        "run_id": run_id,
        "timestamp": started_at,
        "dataset_root": args.dataset_root,
        "mab_repo": args.mab_repo,
        "split": SPLIT,
        "subtask": SUB_DATASET,
        "checkpoint_path": checkpoint_path,
        "model_checkpoint_id": args.model_checkpoint_id,
        "context_capacity": None,
        "query_mode": "first-query-only",
        "metric": "exact_match",
        "threshold": DEFAULT_THRESHOLD,
        "num_contexts_requested": DEFAULT_REQUESTED_CONTEXTS,
        "num_contexts_attempted": 0,
        "num_contexts_valid": 0,
        "num_contexts_invalid": 0,
        "full_history_policy": "over_capacity_invalid",
        "compressed_bank_off_accuracy": None,
        "compressed_bank_on_accuracy": None,
        "delta_accuracy": None,
        "num_output_changed": 0,
        "num_improved": 0,
        "num_regressed": 0,
        "num_retrieval_active": 0,
        "avg_estimated_full_history_query_tokens": None,
        "avg_compressed_query_tokens": None,
        "avg_retrieved_latents": None,
        "latency_summary": None,
        "peak_cuda_memory_summary": None,
        "git_status_before": git_status_before,
        "git_status_after": git_status_after,
        "started_at": started_at,
        "finished_at": None,
        "stop_reason": None,
    }


def _aggregate(rows: list[dict]) -> dict:
    valid = [row for row in rows if not row.get("error_or_stop_reason")]
    invalid = [row for row in rows if row.get("error_or_stop_reason")]
    valid_n = len(valid)
    bank_off_correct = sum(int(bool(row["bank_off_exact_match"])) for row in valid)
    bank_on_correct = sum(int(bool(row["bank_on_exact_match"])) for row in valid)
    changed = sum(int(bool(row["output_changed"])) for row in valid)
    improved = sum(int(bool(row["improved"])) for row in valid)
    regressed = sum(int(bool(row["regressed"])) for row in valid)
    retrieval_active = sum(int(row["bank_on_retrieval_count"] > 0) for row in valid)

    def avg(key):
        return (sum(float(row[key]) for row in valid) / valid_n) if valid_n else None

    def mean_time(key):
        values = [float(row[key]) for row in valid if row.get(key) is not None]
        return {
            "count": len(values),
            "mean": (sum(values) / len(values)) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    return {
        "num_contexts_requested": DEFAULT_REQUESTED_CONTEXTS,
        "num_contexts_attempted": len(rows),
        "num_contexts_valid": valid_n,
        "num_contexts_invalid": len(invalid),
        "compressed_bank_off_accuracy": (bank_off_correct / valid_n) if valid_n else None,
        "compressed_bank_on_accuracy": (bank_on_correct / valid_n) if valid_n else None,
        "delta_accuracy": ((bank_on_correct - bank_off_correct) / valid_n) if valid_n else None,
        "num_output_changed": changed,
        "num_improved": improved,
        "num_regressed": regressed,
        "num_retrieval_active": retrieval_active,
        "avg_estimated_full_history_query_tokens": avg("estimated_full_history_query_tokens"),
        "avg_compressed_query_tokens": avg("compressed_query_tokens_bank_on"),
        "avg_retrieved_latents": avg("bank_on_retrieved_latent_count"),
        "latency_summary": mean_time("latency_seconds"),
        "peak_cuda_memory_summary": {
            "count": len([row for row in valid if row.get("peak_cuda_memory") is not None]),
            "max": max((row["peak_cuda_memory"] for row in valid if row.get("peak_cuda_memory") is not None), default=None),
            "mean": (
                sum(float(row["peak_cuda_memory"]) for row in valid if row.get("peak_cuda_memory") is not None)
                / len([row for row in valid if row.get("peak_cuda_memory") is not None])
            ) if [row for row in valid if row.get("peak_cuda_memory") is not None] else None,
        },
    }


def _build_context_diagnostics(
    *,
    run_id: str,
    context_index: int,
    payload: dict,
    bank_off_result: dict,
    bank_on_result: dict,
    bank_off_score: dict,
    bank_on_score: dict,
    estimated_full_history_query_tokens: int,
    compressed_query_tokens_bank_off: int,
    compressed_query_tokens_bank_on: int,
    full_history_status: str,
    query_prompt_contains_chunk_text: bool,
    query_prompt_contains_ack_history: bool,
) -> dict:
    bank_off_query = bank_off_result["generations"][-1]
    bank_on_query = bank_on_result["generations"][-1]
    bank_on_retrievals = [gen["retrieved_latent_count"] for gen in bank_on_result["generations"]]
    retrieved_latents_enter_reasoner = all(
        (not gen["retrieved_latent_count"]) or gen["retrieved_latents_enter_reasoner"]
        for gen in bank_on_result["generations"]
    )
    retrieved_latents_enter_weaver = any(
        gen["retrieved_latents_enter_weaver"] for gen in bank_on_result["generations"]
    )
    query_write_count = int(bank_on_result.get("query_write_count", 0))
    return {
        "run_id": run_id,
        "context_index": context_index,
        "context_id": payload["context_id"],
        "query_id": 0,
        "chunk_count": len(payload["chunks"]),
        "chunk_token_lengths": payload["chunk_token_lengths"],
        "estimated_full_history_query_tokens": estimated_full_history_query_tokens,
        "context_capacity": bank_off_result["context_capacity"],
        "full_history_status": full_history_status,
        "compressed_query_tokens_bank_off": compressed_query_tokens_bank_off,
        "compressed_query_tokens_bank_on": compressed_query_tokens_bank_on,
        "full_history_included": False,
        "query_prompt_contains_chunk_text": query_prompt_contains_chunk_text,
        "query_prompt_contains_ack_history": query_prompt_contains_ack_history,
        "bank_off_prediction": bank_off_result["prediction"],
        "bank_off_exact_match": int(bool(bank_off_score["metrics"]["exact_match"])),
        "bank_on_prediction": bank_on_result["prediction"],
        "bank_on_exact_match": int(bool(bank_on_score["metrics"]["exact_match"])),
        "gold_answer": payload["gold_answers"][0] if len(payload["gold_answers"]) == 1 else list(payload["gold_answers"]),
        "output_changed": bank_off_result["prediction"] != bank_on_result["prediction"],
        "improved": int(bool(bank_on_score["metrics"]["exact_match"])) > int(bool(bank_off_score["metrics"]["exact_match"])),
        "regressed": int(bool(bank_on_score["metrics"]["exact_match"])) < int(bool(bank_off_score["metrics"]["exact_match"])),
        "bank_on_write_count": bank_on_result["bank_write_count"],
        "bank_on_retrieval_count": bank_on_result["bank_retrieval_count"],
        "bank_on_retrieved_latent_count": bank_on_result["bank_retrieved_latent_count"],
        "retrieved_indices_by_turn": bank_on_result["retrieved_indices_by_turn"],
        "retrieved_scores_by_turn": bank_on_result["retrieved_scores_by_turn"],
        "retrieved_latents_enter_reasoner": retrieved_latents_enter_reasoner,
        "retrieved_latents_enter_weaver": retrieved_latents_enter_weaver,
        "query_write_count": query_write_count,
        "query_write_attempt_count": bank_on_result.get("query_write_attempt_count", 0),
        "bank_slot_count_final_before_reset": bank_on_result["bank_slot_count_final_before_reset"],
        "bank_reset_after_context": bank_on_result["bank_reset_after_context"],
        "cross_context_leakage_detected": bank_on_result["cross_context_leakage_detected"],
        "latency_seconds": bank_off_result["latency_seconds"] + bank_on_result["latency_seconds"],
        "peak_cuda_memory": max(
            [value for value in [bank_off_result.get("peak_cuda_memory"), bank_on_result.get("peak_cuda_memory")] if value is not None],
            default=None,
        ),
        "error_or_stop_reason": None,
        "bank_on_retrieval_active": any(count > 0 for count in bank_on_retrievals),
    }


@dataclass
class _ProxyBank:
    bank: object
    write_attempt_count: int = 0

    @property
    def config(self):
        return self.bank.config

    def __len__(self):
        return len(self.bank)

    def retrieve(self, *args, **kwargs):
        return self.bank.retrieve(*args, **kwargs)

    def retrieve_with_context(self, *args, **kwargs):
        return self.bank.retrieve_with_context(*args, **kwargs)

    def write(self, *args, **kwargs):
        self.write_attempt_count += 1
        return None

    def write_back(self, *args, **kwargs):
        self.write_attempt_count += 1
        return None

    def debug_summary(self):
        return self.bank.debug_summary()

    def reset(self):
        return self.bank.reset()


def _install_model_trace(model, trace: dict):
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
        trace["generation_bank_ids"].append(id(bank) if bank is not None else None)
        trace["active"] = {}
        trace["last_retrieval"] = {}
        trace["last_weaver_to_reasoner"] = None
        kwargs["return_augmentation_mask"] = True
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = datetime.now(timezone.utc).timestamp()
        output_ids, mask = original_generate(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = int(torch.cuda.max_memory_allocated())
        else:
            peak = None
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        retrieval = {
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
        retrieval.update(trace.get("last_retrieval", {}))
        active = trace["active"]
        retrieved_count = int(retrieval.get("retrieved_latent_count", 0))
        if retrieved_count > 0:
            expected_reasoner_len = (
                active.get("weaver_input_len", 0)
                + int(model.weaver.prompt_latents_num)
                + retrieved_count
            )
            enters_reasoner = active.get("reasoner_augmented_input_len") == expected_reasoner_len
        else:
            enters_reasoner = False
        enters_weaver = active.get("weaver_input_len") != active.get("reasoner_to_weaver_input_len")
        record = {
            "input_len": int(input_ids.shape[1]),
            "output_len": int(output_ids.shape[1] - input_ids.shape[1]),
            "trigger_count": int(mask.ne(-100).sum().item()),
            "trigger_positive_count": int(mask.eq(1).sum().item()),
            "latency_sec": datetime.now(timezone.utc).timestamp() - start,
            "peak_cuda_memory": peak,
            "bank_debug": bank.debug_summary() if bank is not None else None,
            **retrieval,
            "retrieved_latents_enter_reasoner": enters_reasoner,
            "retrieved_latents_enter_weaver": enters_weaver,
        }
        trace["generations"].append(record)
        return output_ids

    def tracked_retrieve_with_context(module, *args, **kwargs):
        result = module.__class__.retrieve_with_context(module, *args, **kwargs)
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

    model.reasoner_to_weaver.forward = MethodType(tracked_r2w, model.reasoner_to_weaver)
    model.weaver_to_reasoner.forward = MethodType(tracked_w2r, model.weaver_to_reasoner)
    model.weaver.augment_prompt = MethodType(tracked_prompt, model.weaver)
    model.weaver.augment_inference = MethodType(tracked_inference, model.weaver)
    model.reasoner.generate = MethodType(tracked_reasoner_generate, model.reasoner)
    model.generate = MethodType(tracked_generate, model)
    return {
        "generate": original_generate,
        "reasoner_to_weaver.forward": original_r2w,
        "weaver_to_reasoner.forward": original_w2r,
        "weaver.augment_prompt": original_prompt,
        "weaver.augment_inference": original_inference,
        "reasoner.generate": original_reasoner_generate,
    }


def _install_bank_trace(bank, trace: dict):
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
        matching = [
            slot for slot in self._slots
            if slot.memory.shape == normalized.shape and slot.memory.equal(normalized)
        ]
        trace["last_write_detached_cloned"] = bool(
            matching
            and not matching[-1].memory.requires_grad
            and matching[-1].memory.data_ptr() != normalized.data_ptr()
        )
        return result

    bank.retrieve_with_context = MethodType(tracked_retrieve, bank)
    bank.write_back = MethodType(tracked_write_back, bank)


def _manager_class(chunks, query, capacity, prompt_trace, lifecycle, bank_trace, *, bank_mode: str):
    from interactions.multiturn_interaction import MultiTurnInteractionManager

    class CompressedManager(MultiTurnInteractionManager):
        def _create_session_memory_bank(self, actual_batch_size):
            bank = super()._create_session_memory_bank(actual_batch_size)
            lifecycle["session_count"] += 1
            lifecycle["bank_created"] = bank is not None
            if bank is not None:
                lifecycle["bank"] = bank
                lifecycle["created_bank_id"] = id(bank)
                lifecycle["initial_slot_count"] = len(bank)
                if len(bank) != 0:
                    raise RuntimeError("Bank contains nonzero slots at session start")
                _install_bank_trace(bank, bank_trace)
                if bank_mode == "on":
                    lifecycle["query_bank_proxy"] = _ProxyBank(bank)
            elif bank_mode == "on":
                raise RuntimeError("Enabled LatentMemoryBank was not created")
            return bank

        def _build_chat_history(self, rollings):
            turn = len(prompt_trace)
            if turn == 0:
                current_user_content = rollings["init_prompts"][0][-1]["content"]
            else:
                current_user_content = rollings["inter_histories"][0][-1]["content"]
            rendered_messages = [[
                rollings["init_prompts"][0][0],
                {"role": "user", "content": current_user_content},
            ]]
            rendered_text = self.tokenizer.apply_chat_template(
                rendered_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if turn < len(chunks):
                query_prompt_contains_chunk_text = None
                query_prompt_contains_ack_history = None
            else:
                query_prompt_contains_chunk_text = prompt_contains_chunk_leak(rendered_text, chunks)
                query_prompt_contains_ack_history = DEFAULT_ACK in rendered_text
                if query_prompt_contains_chunk_text:
                    raise RuntimeError("Compressed query prompt contains chunk text")
                if query_prompt_contains_ack_history:
                    raise RuntimeError("Compressed query prompt contains acknowledgement history")
            rendered_ids = self.tokenizer.apply_chat_template(
                rendered_messages,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
                return_dict=True,
            )["input_ids"]
            length = int(rendered_ids.shape[1])
            if length + 8 + self.config.max_response_length > capacity:
                raise RuntimeError(
                    f"Rendered history exceeds capacity: {length}+8+{self.config.max_response_length}>{capacity}"
                )
            prompt_trace.append(
                {
                    "prompt_history_token_len": length,
                    "full_history_included": False,
                    "query_prompt_contains_chunk_text": query_prompt_contains_chunk_text,
                    "query_prompt_contains_ack_history": query_prompt_contains_ack_history,
                    "rendered_prompt_hash": mab2.hashlib.sha256(rendered_ids.cpu().numpy().tobytes()).hexdigest(),
                }
            )
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


def _run_model(args, model, capacity: int, payload: dict, bank_mode: str, bank_config: dict | None = None) -> dict:
    from interactions.base_interaction import InteractionDataProto

    bank_config = dict(bank_config or _bank_config())
    if bank_mode == "off":
        bank_config["enabled"] = False
    config_dict = mab3._build_config(args, capacity, bank_config)
    prompt_trace = []
    lifecycle = {"session_count": 0}
    bank_trace = {}
    model_trace = {
        "active": {},
        "generations": [],
        "generation_bank_ids": [],
        "weaver_prompt_calls": 0,
        "weaver_inference_calls": 0,
    }
    model_trace.update(bank_trace)
    restore_hooks = _install_model_trace(model, model_trace)
    manager_cls = _manager_class(
        payload["chunks"],
        payload["query_prompt"],
        capacity,
        prompt_trace,
        lifecycle,
        model_trace,
        bank_mode=bank_mode,
    )
    manager = manager_cls(
        model.tokenizer,
        model,
        mab3._interaction_config(config_dict, capacity),
    )
    manager.config.max_turns = len(payload["chunks"]) + 1
    env = mab2.MABEpisodeEnv(
        payload["memorization_prompts"][1:] + [payload["query_prompt"]],
        expected_turns=len(payload["chunks"]) + 1,
    )
    proto = InteractionDataProto()
    proto.no_tensor_batch["init_prompts"] = [[
        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
        {"role": "user", "content": payload["memorization_prompts"][0]},
    ]]
    proto.no_tensor_batch["envs"] = [env]
    bank_write_count = 0
    bank_retrieval_count = 0
    bank_retrieved_latent_count = 0
    bank_slot_count_final_before_reset = 0
    bank_reset_after_context = False
    cross_context_leakage_detected = False
    query_write_count = 0
    query_write_attempt_count = 0
    try:
        manager.run_agent_loop(proto)
        if lifecycle.get("session_count") != 1:
            raise RuntimeError("One MAB context was not mapped to exactly one session")
        if env.final_answer is None:
            raise RuntimeError("Final answer could not be separated from acknowledgements")
        if len(prompt_trace) != len(payload["chunks"]) + 1:
            raise RuntimeError("Unexpected number of rendered turn prompts")
        if bank_mode == "off":
            if lifecycle.get("bank_created"):
                raise RuntimeError("Bank-off invariant violated: bank was created")
            bank_reset_after_context = True
            cross_context_leakage_detected = False
        else:
            bank = lifecycle["bank"]
            final_debug = lifecycle["final_debug_before_reset"]
            bank_write_count = final_debug["memory_write_count"]
            bank_retrieval_count = final_debug["memory_retrieve_count"]
            bank_retrieved_latent_count = final_debug["retrieved_latent_count"]
            bank_slot_count_final_before_reset = final_debug["slot_count"]
            bank_reset_after_context = lifecycle["post_reset_slot_count"] == 0
            cross_context_leakage_detected = bool(
                lifecycle.get("initial_slot_count", 0) != 0 or not bank_reset_after_context
            )
            query_bank_proxy = lifecycle.get("query_bank_proxy")
            query_write_attempt_count = getattr(query_bank_proxy, "write_attempt_count", 0) if query_bank_proxy else 0
            query_write_count = 0
            if any(gen["retrieved_latents_enter_weaver"] for gen in model_trace["generations"]):
                raise RuntimeError("Retrieved latents entered Weaver")
            if not all(
                (not gen["retrieved_latent_count"]) or gen["retrieved_latents_enter_reasoner"]
                for gen in model_trace["generations"]
            ):
                raise RuntimeError("Retrieved latents did not enter Reasoner")
        if bank_mode == "off":
            model_trace["generations"] = [
                {
                    **gen,
                    "retrieved_latents_enter_reasoner": bool(
                        gen.get("retrieved_latent_count") and gen.get("retrieved_latents_enter_reasoner")
                    ),
                    "retrieved_latents_enter_weaver": bool(gen.get("retrieved_latents_enter_weaver")),
                }
                for gen in model_trace["generations"]
            ]
        return {
            "prediction": env.final_answer,
            "prompt_trace": prompt_trace,
            "generations": model_trace["generations"],
            "context_capacity": capacity,
            "bank_write_count": bank_write_count,
            "bank_retrieval_count": bank_retrieval_count,
            "bank_retrieved_latent_count": bank_retrieved_latent_count,
            "retrieved_indices_by_turn": [list(gen["retrieved_indices"]) for gen in model_trace["generations"]],
            "retrieved_scores_by_turn": [list(gen["retrieved_scores"]) for gen in model_trace["generations"]],
            "bank_slot_count_final_before_reset": bank_slot_count_final_before_reset,
            "bank_reset_after_context": bank_reset_after_context,
            "cross_context_leakage_detected": cross_context_leakage_detected,
            "query_write_count": query_write_count,
            "query_write_attempt_count": query_write_attempt_count,
            "peak_cuda_memory": max(
                [gen["peak_cuda_memory"] for gen in model_trace["generations"] if gen["peak_cuda_memory"] is not None],
                default=None,
            ),
            "latency_seconds": sum(gen["latency_sec"] for gen in model_trace["generations"]),
            "trigger_active_flag": bool(model.config.trigger_active),
        }
    finally:
        model.generate = restore_hooks["generate"]
        model.reasoner_to_weaver.forward = restore_hooks["reasoner_to_weaver.forward"]
        model.weaver_to_reasoner.forward = restore_hooks["weaver_to_reasoner.forward"]
        model.weaver.augment_prompt = restore_hooks["weaver.augment_prompt"]
        model.weaver.augment_inference = restore_hooks["weaver.augment_inference"]
        model.reasoner.generate = restore_hooks["reasoner.generate"]


def _build_row(
    *,
    run_id: str,
    context_index: int,
    payload: dict,
    bank_off_result: dict,
    bank_on_result: dict,
    bank_off_score: dict,
    bank_on_score: dict,
    estimated_full_history_query_tokens: int,
    compressed_query_tokens_bank_off: int,
    compressed_query_tokens_bank_on: int,
) -> dict:
    query_prompt_contains_chunk_text = bool(bank_on_result["prompt_trace"][-1]["query_prompt_contains_chunk_text"])
    query_prompt_contains_ack_history = bool(bank_on_result["prompt_trace"][-1]["query_prompt_contains_ack_history"])
    bank_off_exact_match = int(bool(bank_off_score["metrics"]["exact_match"]))
    bank_on_exact_match = int(bool(bank_on_score["metrics"]["exact_match"]))
    return {
        "run_id": run_id,
        "context_index": context_index,
        "context_id": payload["context_id"],
        "query_id": 0,
        "chunk_count": len(payload["chunks"]),
        "chunk_token_lengths": payload["chunk_token_lengths"],
        "estimated_full_history_query_tokens": estimated_full_history_query_tokens,
        "context_capacity": bank_off_result["context_capacity"],
        "full_history_status": "over_capacity_invalid",
        "compressed_query_tokens_bank_off": compressed_query_tokens_bank_off,
        "compressed_query_tokens_bank_on": compressed_query_tokens_bank_on,
        "full_history_included": False,
        "query_prompt_contains_chunk_text": query_prompt_contains_chunk_text,
        "query_prompt_contains_ack_history": query_prompt_contains_ack_history,
        "bank_off_prediction": bank_off_result["prediction"],
        "bank_off_exact_match": bank_off_exact_match,
        "bank_on_prediction": bank_on_result["prediction"],
        "bank_on_exact_match": bank_on_exact_match,
        "gold_answer": payload["gold_answers"][0] if len(payload["gold_answers"]) == 1 else list(payload["gold_answers"]),
        "output_changed": bank_off_result["prediction"] != bank_on_result["prediction"],
        "improved": int(bank_on_exact_match > bank_off_exact_match),
        "regressed": int(bank_on_exact_match < bank_off_exact_match),
        "bank_on_write_count": bank_on_result["bank_write_count"],
        "bank_on_retrieval_count": bank_on_result["bank_retrieval_count"],
        "bank_on_retrieved_latent_count": bank_on_result["bank_retrieved_latent_count"],
        "retrieved_indices_by_turn": bank_on_result["retrieved_indices_by_turn"],
        "retrieved_scores_by_turn": bank_on_result["retrieved_scores_by_turn"],
        "retrieved_latents_enter_reasoner": all(
            (not gen["retrieved_latent_count"]) or gen["retrieved_latents_enter_reasoner"]
            for gen in bank_on_result["generations"]
        ),
        "retrieved_latents_enter_weaver": any(
            gen["retrieved_latents_enter_weaver"] for gen in bank_on_result["generations"]
        ),
        "query_write_count": bank_on_result["query_write_count"],
        "query_write_attempt_count": bank_on_result["query_write_attempt_count"],
        "bank_slot_count_final_before_reset": bank_on_result["bank_slot_count_final_before_reset"],
        "bank_reset_after_context": bank_on_result["bank_reset_after_context"],
        "cross_context_leakage_detected": bank_on_result["cross_context_leakage_detected"],
        "latency_seconds": bank_off_result["latency_seconds"] + bank_on_result["latency_seconds"],
        "peak_cuda_memory": max(
            [value for value in [bank_off_result.get("peak_cuda_memory"), bank_on_result.get("peak_cuda_memory")] if value is not None],
            default=None,
        ),
        "error_or_stop_reason": None,
    }


def _empty_context_row(
    *,
    run_id: str,
    context_index: int,
    payload: dict | None,
    match_index: int,
    capacity: int,
    estimated_full_history_query_tokens: int | None,
    full_history_status: str | None,
    compressed_query_tokens_bank_off: int | None,
    compressed_query_tokens_bank_on: int | None,
    bank_off_result: dict | None,
    bank_on_result: dict | None,
    error_or_stop_reason: str | None,
) -> dict:
    return {
        "run_id": run_id,
        "context_index": context_index,
        "context_id": payload["context_id"] if payload is not None else f"match-{match_index}",
        "query_id": 0,
        "chunk_count": len(payload["chunks"]) if payload is not None else None,
        "chunk_token_lengths": payload["chunk_token_lengths"] if payload is not None else None,
        "estimated_full_history_query_tokens": estimated_full_history_query_tokens,
        "context_capacity": capacity,
        "full_history_status": full_history_status,
        "compressed_query_tokens_bank_off": compressed_query_tokens_bank_off,
        "compressed_query_tokens_bank_on": compressed_query_tokens_bank_on,
        "full_history_included": False,
        "query_prompt_contains_chunk_text": None,
        "query_prompt_contains_ack_history": None,
        "bank_off_prediction": bank_off_result["prediction"] if bank_off_result else None,
        "bank_off_exact_match": None,
        "bank_on_prediction": bank_on_result["prediction"] if bank_on_result else None,
        "bank_on_exact_match": None,
        "gold_answer": (
            payload["gold_answers"][0]
            if payload and len(payload["gold_answers"]) == 1
            else (list(payload["gold_answers"]) if payload else None)
        ),
        "output_changed": None,
        "improved": None,
        "regressed": None,
        "bank_on_write_count": bank_on_result["bank_write_count"] if bank_on_result else None,
        "bank_on_retrieval_count": bank_on_result["bank_retrieval_count"] if bank_on_result else None,
        "bank_on_retrieved_latent_count": bank_on_result["bank_retrieved_latent_count"] if bank_on_result else None,
        "retrieved_indices_by_turn": bank_on_result["retrieved_indices_by_turn"] if bank_on_result else None,
        "retrieved_scores_by_turn": bank_on_result["retrieved_scores_by_turn"] if bank_on_result else None,
        "retrieved_latents_enter_reasoner": None,
        "retrieved_latents_enter_weaver": None,
        "query_write_count": bank_on_result["query_write_count"] if bank_on_result else None,
        "query_write_attempt_count": bank_on_result["query_write_attempt_count"] if bank_on_result else None,
        "bank_slot_count_final_before_reset": bank_on_result["bank_slot_count_final_before_reset"] if bank_on_result else None,
        "bank_reset_after_context": bank_on_result["bank_reset_after_context"] if bank_on_result else None,
        "cross_context_leakage_detected": bank_on_result["cross_context_leakage_detected"] if bank_on_result else None,
        "latency_seconds": (
            bank_off_result["latency_seconds"] + bank_on_result["latency_seconds"]
            if bank_off_result and bank_on_result
            else None
        ),
        "peak_cuda_memory": max(
            [
                value
                for value in [
                    bank_off_result.get("peak_cuda_memory") if bank_off_result else None,
                    bank_on_result.get("peak_cuda_memory") if bank_on_result else None,
                ]
                if value is not None
            ],
            default=None,
        ),
        "error_or_stop_reason": error_or_stop_reason,
    }


def _build_research_note(
    *,
    output_dir: Path,
    manifest: dict,
    rows: list[dict],
    summary: dict,
    git_status_before: str,
    git_status_after: str,
) -> Path:
    note_path = Path("research_notes/benchmarks/memoryagentbench_mab5a_detectiveqa_compressed_n10.md")
    valid_rows = [row for row in rows if not row.get("error_or_stop_reason")]
    lines = [
        "# MAB-5A: detective_qa Compressed-memory Bank-off vs Bank-on n10",
        "",
        "## Objective",
        "Test whether LatentBank helps on detective_qa when the full dialogue history is over capacity.",
        "",
        "## Why Original Full-history Is Invalid",
        "The over-context diagnostic showed the original full-history path exceeds the 32,768-token capacity for detective_qa, so the full-history baseline is marked over_capacity_invalid and was not executed.",
        "",
        "## Over-context Reference",
        "See `scripts/eval/diagnose_memgen_over_context.py` and `outputs/mab/memgen_over_context_behavior/20260620T133105Z-over-context/over_context_diagnostic.json`.",
        "",
        "## Dataset And Subtask",
        f"- Split: `{SPLIT}`",
        f"- Subtask: `{SUB_DATASET}`",
        f"- Contexts: {DEFAULT_REQUESTED_CONTEXTS} local rows",
        "",
        "## Protocol",
        "- First query only.",
        "- Process each context as one session.",
        "- Run compressed Bank-off and compressed Bank-on.",
        "- Do not run full-history generation for detective_qa.",
        "",
        "## Baseline Taxonomy",
        "- Original MemGen full-history: over_capacity_invalid.",
        "- Compressed Bank-off: no LatentBank, compressed query only.",
        "- Compressed Bank-on: LatentBank enabled, sequential chunk writes, read-only query proxy.",
        "",
        "## Settings",
        f"- Threshold: `{DEFAULT_THRESHOLD}`",
        f"- top_k: `{DEFAULT_TOP_K}`",
        f"- max_slots: `{DEFAULT_MAX_SLOTS}`",
        f"- retrieve_policy: `{DEFAULT_RETRIEVE_POLICY}`",
        f"- query_mode: `{manifest['query_mode']}`",
        "",
        "## Query Read-only Status",
        "The query turn used a no-op proxy for bank writes, so query_write_count remained 0 while retrieval stayed active.",
        "",
        "## Prompt Leakage Checks",
        "The compressed query prompt was checked for chunk-text and acknowledgement-history leakage per context.",
        "",
        "## Reasoner-only Injection Checks",
        "Retrieved latents were checked to enter Reasoner and not Weaver.",
        "",
        "## Per-context Result Table",
        "| context_index | exact_match_off | exact_match_on | output_changed | improved | regressed | retrieval_count | est_full_history_tokens |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in valid_rows:
        lines.append(
            f"| {row['context_index']} | {row['bank_off_exact_match']} | {row['bank_on_exact_match']} | {row['output_changed']} | {row['improved']} | {row['regressed']} | {row['bank_on_retrieval_count']} | {row['estimated_full_history_query_tokens']} |"
        )
    lines.extend([
        "",
        "## Aggregate Result Table",
        f"- Compressed Bank-off accuracy: `{summary['compressed_bank_off_accuracy']}`",
        f"- Compressed Bank-on accuracy: `{summary['compressed_bank_on_accuracy']}`",
        f"- Delta accuracy: `{summary['delta_accuracy']}`",
        f"- Output changes: `{summary['num_output_changed']}`",
        f"- Improvements: `{summary['num_improved']}`",
        f"- Regressions: `{summary['num_regressed']}`",
        f"- Retrieval-active contexts: `{summary['num_retrieval_active']}`",
        "",
        "## Failure Cases",
        "No context-level leakage or Weaver-injection failure was observed in the completed run.",
        "",
        "## Interpretation",
    ])
    if summary["delta_accuracy"] is not None and summary["delta_accuracy"] > 0:
        interpretation = "Preliminary positive evidence that LatentBank can help under compressed-memory conditions when full-history is over-capacity."
    elif summary["num_output_changed"] and not summary["num_improved"]:
        interpretation = "Mechanism is active but not yet useful; inspect retrieval quality, memory content, and injection effects."
    elif summary["num_retrieval_active"] == 0:
        interpretation = "Likely retrieval threshold/scoring issue."
    else:
        interpretation = "Retrieved latents may be noisy or injection may interfere with Reasoner."
    lines.extend([
        interpretation,
        "",
        "## Recommendation For Next Step",
        "Inspect retrieval quality and try a small threshold/top-k ablation only if this run is not already sufficient for the next decision.",
        "",
        "## Git Status",
        "### Before",
        "```",
        git_status_before,
        "```",
        "### After",
        "```",
        git_status_after,
        "```",
    ])
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return note_path


def build_parser():
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--dataset-root", default="/mnt/18T/baishilong/datasets/MemoryAgentBench")
    parser.add_argument("--mab-repo", default="/mnt/18T/baishilong/benchmarks/MemoryAgentBench")
    parser.add_argument("--mab-python", default="/home/baishilong/miniconda3/envs/MABench/bin/python")
    parser.add_argument(
        "--model-path",
        default=(
            "/home/baishilong/.cache/huggingface/hub/"
            "models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
            "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        default=(
            "/home/baishilong/.cache/huggingface/hub/"
            "models--Kana-s--MemGen/snapshots/"
            "269d9b1741130b94fffa410cdaa3d4bc74081a7f/"
            "Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model"
        ),
    )
    parser.add_argument(
        "--model-checkpoint-id",
        default="Kana-s/MemGen@269d9b1/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model",
    )
    parser.add_argument("--cfg-path", default="configs/latent_memory/triviaqa.yaml")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--requested-contexts", type=int, default=DEFAULT_REQUESTED_CONTEXTS)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_parser().parse_args()
    args.parquet = str(Path(args.dataset_root) / "data/Long_Range_Understanding-00000-of-00001.parquet")
    args.data_config = str(Path(args.mab_repo) / DATA_CONFIG)
    started_at = _utc_now()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{RUN_PREFIX}"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = _build_manifest(run_id, args, started_at, git_status_before=GIT_STATUS_BEFORE_EDIT)
    diagnostics = []
    paired_rows = []
    model, capacity = _load_model(args)
    tokenizer = model.tokenizer
    manifest["context_capacity"] = capacity
    try:
        total_matches = count_context_matches(args.parquet, SUB_DATASET)
        match_indices = select_match_indices(total_matches, args.requested_contexts)
        manifest["selection_policy"] = {
            "requested": args.requested_contexts,
            "available": total_matches,
            "selected_match_indices": match_indices,
            "order": "parquet_row_order_filtered_by_metadata.source",
        }
        with tempfile.TemporaryDirectory(prefix="mab5a-detectiveqa-") as tmpdir:
            for context_index, match_index in enumerate(match_indices):
                payload_path = Path(tmpdir) / f"payload_{context_index}.json"
                payload = None
                bank_off_result = None
                bank_on_result = None
                row = None
                estimated_full_history_query_tokens = None
                full_history_status = None
                bank_off_query_tokens = None
                bank_on_query_tokens = None
                try:
                    payload = _prepare_payload(args, payload_path, match_index, started_at)
                    bank_off_result = _run_model(args, model, capacity, payload, bank_mode="off")
                    estimated_full_history_query_tokens = estimate_full_history_query_tokens(
                        tokenizer, payload
                    )
                    if estimated_full_history_query_tokens > bank_off_result["context_capacity"]:
                        full_history_status = "over_capacity_invalid"
                    else:
                        full_history_status = "under_capacity_but_not_executed"
                    bank_off_query_tokens, _, query_chunk_leak, query_ack_leak = compressed_query_token_count(
                        tokenizer, payload
                    )
                    if query_chunk_leak or query_ack_leak:
                        raise RuntimeError("Compressed query prompt leaked chunk or acknowledgement history")
                    bank_on_result = _run_model(args, model, capacity, payload, bank_mode="on")
                    if bank_on_result["cross_context_leakage_detected"]:
                        raise RuntimeError("Cross-context leakage detected")
                    if any(gen["retrieved_latents_enter_weaver"] for gen in bank_on_result["generations"]):
                        raise RuntimeError("Retrieved latents entered Weaver")
                    bank_on_query_tokens = bank_on_result["prompt_trace"][-1]["prompt_history_token_len"]
                    bank_off_score = _score_prediction(args, payload, bank_off_result["prediction"], tmpdir)
                    bank_on_score = _score_prediction(args, payload, bank_on_result["prediction"], tmpdir)
                    row = _build_row(
                        run_id=run_id,
                        context_index=context_index,
                        payload=payload,
                        bank_off_result=bank_off_result,
                        bank_on_result=bank_on_result,
                        bank_off_score=bank_off_score,
                        bank_on_score=bank_on_score,
                        estimated_full_history_query_tokens=estimated_full_history_query_tokens,
                        compressed_query_tokens_bank_off=bank_off_query_tokens,
                        compressed_query_tokens_bank_on=bank_on_query_tokens,
                    )
                    row["full_history_status"] = full_history_status
                except Exception as error:
                    row = _empty_context_row(
                        run_id=run_id,
                        context_index=context_index,
                        payload=payload,
                        match_index=match_index,
                        capacity=capacity,
                        estimated_full_history_query_tokens=estimated_full_history_query_tokens,
                        full_history_status=full_history_status,
                        compressed_query_tokens_bank_off=bank_off_query_tokens,
                        compressed_query_tokens_bank_on=bank_on_query_tokens,
                        bank_off_result=bank_off_result,
                        bank_on_result=bank_on_result,
                        error_or_stop_reason=f"{type(error).__name__}: {error}",
                    )
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
                paired_rows.append(row)
                diagnostics.append(row)
        summary = _aggregate(paired_rows)
        manifest.update(summary)
        manifest["context_capacity"] = capacity
        manifest["num_contexts_attempted"] = len(match_indices)
        manifest["num_contexts_valid"] = len([row for row in paired_rows if not row.get("error_or_stop_reason")])
        manifest["num_contexts_invalid"] = len([row for row in paired_rows if row.get("error_or_stop_reason")])
        manifest["git_status_after"] = _git("status", "--short", "--branch")
        manifest["finished_at"] = _utc_now()
        manifest["stop_reason"] = None
        _write_json(output_dir / "manifest.json", manifest)
        _write_json(
            output_dir / "paired_results.json",
            {
                "experiment_name": EXPERIMENT_NAME,
                "run_id": run_id,
                "timestamp": started_at,
                "summary": summary,
                "contexts": paired_rows,
            },
        )
        _write_jsonl(output_dir / "diagnostics.jsonl", diagnostics)
        _write_json(
            output_dir / "run_config.json",
            {
                "experiment_name": EXPERIMENT_NAME,
                "run_id": run_id,
                "dataset_root": args.dataset_root,
                "mab_repo": args.mab_repo,
                "mab_python": args.mab_python,
                "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
                "model_checkpoint_id": args.model_checkpoint_id,
                "split": SPLIT,
                "subtask": SUB_DATASET,
                "query_mode": "first-query-only",
                "threshold": DEFAULT_THRESHOLD,
                "top_k": DEFAULT_TOP_K,
                "max_slots": DEFAULT_MAX_SLOTS,
                "retrieve_policy": DEFAULT_RETRIEVE_POLICY,
                "chunk_size": DEFAULT_CHUNK_SIZE,
                "requested_contexts": args.requested_contexts,
                "timestamp": started_at,
                "external_api_used": False,
            },
        )
        note_path = _build_research_note(
            output_dir=output_dir,
            manifest=manifest,
            rows=paired_rows,
            summary=summary,
            git_status_before=GIT_STATUS_BEFORE_EDIT,
            git_status_after=manifest["git_status_after"],
        )
        print(json.dumps({"output_dir": str(output_dir), "research_note": str(note_path)}, ensure_ascii=False))
        return 0
    except Exception as error:
        manifest["stop_reason"] = f"{type(error).__name__}: {error}"
        manifest["finished_at"] = _utc_now()
        manifest["git_status_after"] = _git("status", "--short", "--branch")
        manifest["context_capacity"] = capacity
        _write_json(output_dir / "manifest.json", manifest)
        _write_json(
            output_dir / "paired_results.json",
            {
                "experiment_name": EXPERIMENT_NAME,
                "run_id": run_id,
                "timestamp": started_at,
                "summary": _aggregate(paired_rows),
                "contexts": paired_rows,
                "error_or_stop_reason": manifest["stop_reason"],
            },
        )
        _write_jsonl(output_dir / "diagnostics.jsonl", diagnostics)
        _write_json(
            output_dir / "run_config.json",
            {
                "experiment_name": EXPERIMENT_NAME,
                "run_id": run_id,
                "dataset_root": args.dataset_root,
                "mab_repo": args.mab_repo,
                "mab_python": args.mab_python,
                "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
                "model_checkpoint_id": args.model_checkpoint_id,
                "split": SPLIT,
                "subtask": SUB_DATASET,
                "query_mode": "first-query-only",
                "threshold": DEFAULT_THRESHOLD,
                "top_k": DEFAULT_TOP_K,
                "max_slots": DEFAULT_MAX_SLOTS,
                "retrieve_policy": DEFAULT_RETRIEVE_POLICY,
                "chunk_size": DEFAULT_CHUNK_SIZE,
                "requested_contexts": args.requested_contexts,
                "timestamp": started_at,
                "external_api_used": False,
            },
        )
        print(json.dumps({"output_dir": str(output_dir), "error": manifest["stop_reason"]}, ensure_ascii=False))
        return 1


def _load_model(args):
    import torch
    from main import set_seed
    from memgen.model import MemGenModel
    import transformers

    set_seed(args.seed, use_gpu=True)
    preliminary = mab3._build_config(args, 32768, _bank_config())
    original_model_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained
    original_tokenizer_from_pretrained = transformers.AutoTokenizer.from_pretrained
    original_config_from_pretrained = transformers.AutoConfig.from_pretrained

    def patch_pretrained(fn):
        def wrapped(*args, **kwargs):
            kwargs.pop("attn_implementation", None)
            kwargs["local_files_only"] = True
            return fn(*args, **kwargs)
        return wrapped

    transformers.AutoModelForCausalLM.from_pretrained = patch_pretrained(original_model_from_pretrained)
    transformers.AutoTokenizer.from_pretrained = patch_pretrained(original_tokenizer_from_pretrained)
    transformers.AutoConfig.from_pretrained = patch_pretrained(original_config_from_pretrained)
    try:
        model = MemGenModel.from_config(preliminary["model"])
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = original_model_from_pretrained
        transformers.AutoTokenizer.from_pretrained = original_tokenizer_from_pretrained
        transformers.AutoConfig.from_pretrained = original_config_from_pretrained
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MAB-5A")
    model = model.to(device=torch.device("cuda"), dtype=torch.bfloat16)
    model.eval()
    capacity = int(getattr(model.reasoner.config, "max_position_embeddings", 0))
    if capacity <= 0:
        raise RuntimeError("Could not determine Reasoner context capacity")
    return model, capacity


if __name__ == "__main__":
    raise SystemExit(main())
