"""MAB-6B trigger trace: detective_qa Version B Weaver-space bank on 10 contexts."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
from types import MethodType

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import mab3_bank_on_full_history as mab3
from scripts.eval import mab3a_threshold_ablation as mab3a
from scripts.eval import mab5a_detectiveqa_compressed_n10 as base
from scripts.eval import mab6b_weaver_space_bank_detectiveqa_n10 as parent

EXPERIMENT_NAME = "MAB-6B: detective_qa Version B Weaver-space Bank n10 Trigger Trace"
RUN_PREFIX = "detectiveqa-version-b-weaver-space-bank-n10-trigger-trace"
DEFAULT_OUTPUT_ROOT = "outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_trigger_trace"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _git(*args: str) -> str:
    return base.subprocess.run(
        ["git", *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _dict_delta(after: dict | None, before: dict | None) -> dict:
    after = dict(after or {})
    before = dict(before or {})
    keys = set(after) | set(before)
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(keys)
        if int(after.get(key, 0)) - int(before.get(key, 0))
    }


def _aggregate_turns(turn_rows: list[dict], *, turn_type: str, field: str) -> int:
    return sum(
        int(row.get(field, 0))
        for row in turn_rows
        if row.get("turn_type") == turn_type
    )


def _make_trace_context_summary(
    *,
    context_index: int,
    context_id: str,
    bank_on_result: dict,
    bank_on_exact_match: int,
) -> dict:
    turn_rows = bank_on_result["trigger_trace_rows"]
    query_rows = [row for row in turn_rows if row["turn_type"] == "query"]
    query_row = query_rows[-1] if query_rows else None
    return {
        "context_index": context_index,
        "context_id": context_id,
        "construction_trigger_forward_count": _aggregate_turns(
            turn_rows, turn_type="chunk", field="trigger_forward_count"
        ),
        "construction_trigger_invoke_count": _aggregate_turns(
            turn_rows, turn_type="chunk", field="trigger_invoke_count"
        ),
        "query_trigger_forward_count": _aggregate_turns(
            turn_rows, turn_type="query", field="trigger_forward_count"
        ),
        "query_trigger_invoke_count": _aggregate_turns(
            turn_rows, turn_type="query", field="trigger_invoke_count"
        ),
        "construction_weaver_call_count": _aggregate_turns(
            turn_rows, turn_type="chunk", field="weaver_call_count"
        ),
        "query_weaver_call_count": _aggregate_turns(
            turn_rows, turn_type="query", field="weaver_call_count"
        ),
        "construction_memory_retrieve_count": _aggregate_turns(
            turn_rows, turn_type="chunk", field="memory_retrieve_count"
        ),
        "query_memory_retrieve_count": _aggregate_turns(
            turn_rows, turn_type="query", field="memory_retrieve_count"
        ),
        "construction_memory_write_count": _aggregate_turns(
            turn_rows, turn_type="chunk", field="memory_write_count"
        ),
        "query_memory_write_count": _aggregate_turns(
            turn_rows, turn_type="query", field="memory_write_count"
        ),
        "final_slot_count": int(bank_on_result["bank_slot_count_final_before_reset"]),
        "prediction": bank_on_result["prediction"],
        "exact_match": int(bank_on_exact_match),
        "query_write_count": 0 if query_row is None else int(query_row.get("query_write_count", 0)),
        "query_write_attempt_count": 0 if query_row is None else int(query_row.get("query_write_attempt_count", 0)),
        "prompt_history_token_lens": list(bank_on_result.get("prompt_history_token_lens", [])),
        "prompt_history_hashes": list(bank_on_result.get("prompt_history_hashes", [])),
    }


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

        weaver_prompt_before = int(trace.get("weaver_prompt_calls", 0))
        weaver_inference_before = int(trace.get("weaver_inference_calls", 0))
        bank_before = None if bank is None else bank.debug_summary()
        write_attempt_before = getattr(bank, "write_attempt_count", 0) if bank is not None else 0

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

        bank_after = None if bank is None else bank.debug_summary()
        write_attempt_after = getattr(bank, "write_attempt_count", 0) if bank is not None else 0
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
        generation_debug = dict(getattr(module, "_last_generation_debug", {}) or {})
        retrieved_count = int(retrieval.get("retrieved_latent_count", 0))
        if retrieved_count > 0:
            expected_reasoner_len = (
                active.get("weaver_input_len", 0)
                + int(model.weaver.prompt_latents_num)
                + retrieved_count
            )
            heuristic_enters_reasoner = (
                active.get("reasoner_augmented_input_len") == expected_reasoner_len
            )
        else:
            heuristic_enters_reasoner = False
        heuristic_enters_weaver = (
            active.get("weaver_input_len") != active.get("reasoner_to_weaver_input_len")
        )
        enters_reasoner = bool(
            generation_debug.get(
                "retrieved_latents_enter_reasoner",
                heuristic_enters_reasoner,
            )
        )
        enters_weaver = bool(
            generation_debug.get(
                "retrieved_latents_enter_weaver",
                heuristic_enters_weaver,
            )
        )
        trigger_forward_count = int(mask.ne(-100).sum().item())
        trigger_invoke_count = int(mask.eq(1).sum().item())
        record = {
            "input_len": int(input_ids.shape[1]),
            "output_len": int(output_ids.shape[1] - input_ids.shape[1]),
            "trigger_count": trigger_forward_count,
            "trigger_positive_count": trigger_invoke_count,
            "trigger_forward_count": trigger_forward_count,
            "trigger_invoke_count": trigger_invoke_count,
            "trigger_skip_count": trigger_forward_count - trigger_invoke_count,
            "weaver_call_count": (
                int(trace.get("weaver_prompt_calls", 0)) - weaver_prompt_before
                + int(trace.get("weaver_inference_calls", 0)) - weaver_inference_before
            ),
            "latency_sec": datetime.now(timezone.utc).timestamp() - start,
            "peak_cuda_memory": peak,
            "bank_debug": bank_after,
            "bank_slot_count_before_generate": None if bank_before is None else int(bank_before["slot_count"]),
            "bank_slot_count_after_generate": None if bank_after is None else int(bank_after["slot_count"]),
            "memory_retrieve_count": 0 if bank_after is None else int(bank_after["memory_retrieve_count"]) - int(bank_before["memory_retrieve_count"]),
            "memory_retrieved_latent_count": 0 if bank_after is None else int(bank_after["retrieved_latent_count"]) - int(bank_before["retrieved_latent_count"]),
            "memory_write_count": 0 if bank_after is None else int(bank_after["memory_write_count"]) - int(bank_before["memory_write_count"]),
            "memory_write_attempt_count": int(write_attempt_after) - int(write_attempt_before),
            "write_action_counts": {} if bank_after is None else _dict_delta(bank_after.get("write_action_counts"), bank_before.get("write_action_counts")),
            "update_reason_counts": {} if bank_after is None else _dict_delta(bank_after.get("update_reason_counts"), bank_before.get("update_reason_counts")),
            **retrieval,
            **generation_debug,
            "retrieved_latents_enter_reasoner": enters_reasoner,
            "retrieved_latents_enter_weaver": enters_weaver,
        }
        trace["generations"].append(record)
        return output_ids

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
    original_retrieve_with_context = bank.retrieve_with_context
    original_retrieve = bank.retrieve
    original_write_back = bank.write_back
    original_write = bank.write
    bank.write_attempt_count = 0

    def tracked_retrieve_with_context(self, *args, **kwargs):
        result = original_retrieve_with_context(*args, **kwargs)
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

    def tracked_retrieve(self, *args, **kwargs):
        result = original_retrieve(*args, **kwargs)
        trace["last_retrieval"] = {
            "scores": [],
            "max_score": None,
            "argmax_index": None,
            "threshold_passed": bool(result),
            "retrieved_indices": [],
            "retrieved_scores": [],
            "retrieved_slot_count": len(result),
            "retrieved_latent_count": sum(int(slot.memory.shape[0]) for slot in result),
            "retrieval_step": int(getattr(self, "_retrieval_step", 0)),
        }
        return result

    def tracked_write_back(self, memory, retrieval_result, *args, **kwargs):
        self.write_attempt_count += 1
        return original_write_back(memory, retrieval_result, *args, **kwargs)

    def tracked_write(self, memory, *args, **kwargs):
        self.write_attempt_count += 1
        return original_write(memory, *args, **kwargs)

    bank.retrieve_with_context = MethodType(tracked_retrieve_with_context, bank)
    bank.retrieve = MethodType(tracked_retrieve, bank)
    bank.write_back = MethodType(tracked_write_back, bank)
    bank.write = MethodType(tracked_write, bank)


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
                    lifecycle["query_bank_proxy"] = base._ProxyBank(bank)
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
                query_prompt_contains_chunk_text = base.prompt_contains_chunk_leak(rendered_text, chunks)
                query_prompt_contains_ack_history = base.DEFAULT_ACK in rendered_text
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
                    "rendered_prompt_hash": base.mab2.hashlib.sha256(rendered_ids.cpu().numpy().tobytes()).hexdigest(),
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


def _load_model(args):
    return parent._load_model(args)


def _bank_config():
    return parent._bank_config()


def _build_manifest(run_id: str, args, started_at: str, *, git_status_before: str, git_status_after: str | None = None) -> dict:
    manifest = parent._build_manifest(
        run_id,
        args,
        started_at,
        git_status_before=git_status_before,
        git_status_after=git_status_after,
    )
    manifest.update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "run_id": run_id,
            "output_root": DEFAULT_OUTPUT_ROOT,
            "trigger_trace_enabled": True,
            "parent_canonical_run": (
                "outputs/mab/version_b_weaver_space_bank_detectiveqa_n10/"
                "20260625T122323Z-detectiveqa-version-b-weaver-space-bank-n10"
            ),
        }
    )
    return manifest


def _run_model(args, model, capacity: int, payload: dict, bank_mode: str, bank_config: dict | None = None) -> dict:
    from interactions.base_interaction import InteractionDataProto

    previous_flag = bool(getattr(model.config, "retrieved_memory_to_weaver", False))
    previous_storage = getattr(model.config, "memory_bank_storage_space", "reasoner")
    model.config.retrieved_memory_to_weaver = bank_mode == "on"
    model.config.memory_bank_storage_space = "weaver" if bank_mode == "on" else "reasoner"
    try:
        if bank_mode == "off":
            return parent._run_model(args, model, capacity, payload, bank_mode, bank_config)

        bank_config = dict(bank_config or _bank_config())
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
        env = base.mab2.MABEpisodeEnv(
            payload["memorization_prompts"][1:] + [payload["query_prompt"]],
            expected_turns=len(payload["chunks"]) + 1,
        )
        proto = InteractionDataProto()
        proto.no_tensor_batch["init_prompts"] = [[
            {"role": "system", "content": base.DEFAULT_SYSTEM_MESSAGE},
            {"role": "user", "content": payload["memorization_prompts"][0]},
        ]]
        proto.no_tensor_batch["envs"] = [env]
        try:
            manager.run_agent_loop(proto)
            if lifecycle.get("session_count") != 1:
                raise RuntimeError("One MAB context was not mapped to exactly one session")
            if env.final_answer is None:
                raise RuntimeError("Final answer could not be separated from acknowledgements")
            if len(prompt_trace) != len(payload["chunks"]) + 1:
                raise RuntimeError("Unexpected number of rendered turn prompts")

            final_debug = lifecycle["final_debug_before_reset"]
            query_bank_proxy = lifecycle.get("query_bank_proxy")
            query_write_attempt_count = getattr(query_bank_proxy, "write_attempt_count", 0) if query_bank_proxy else 0
            generations = model_trace["generations"]
            if not generations:
                raise RuntimeError("No generation trace recorded")

            trigger_trace_rows = []
            for turn_index, gen in enumerate(generations):
                turn_type = "chunk" if turn_index < len(payload["chunks"]) else "query"
                row = {
                    "context_index": None,
                    "context_id": payload["context_id"],
                    "turn_index": turn_index,
                    "turn_type": turn_type,
                    "generated_token_count": int(gen.get("output_len", 0)),
                    "trigger_forward_count": int(gen.get("trigger_forward_count", 0)),
                    "trigger_invoke_count": int(gen.get("trigger_invoke_count", 0)),
                    "trigger_skip_count": int(gen.get("trigger_skip_count", 0)),
                    "weaver_call_count": int(gen.get("weaver_call_count", 0)),
                    "memory_retrieve_count": int(gen.get("memory_retrieve_count", 0)),
                    "memory_retrieved_latent_count": int(gen.get("memory_retrieved_latent_count", 0)),
                    "retrieved_slot_indices": list(gen.get("retrieved_indices", [])),
                    "retrieved_scores": list(gen.get("retrieved_scores", [])),
                    "memory_write_attempt_count": int(gen.get("memory_write_attempt_count", 0)),
                    "memory_write_count": int(gen.get("memory_write_count", 0)),
                    "write_action_counts": dict(gen.get("write_action_counts", {})),
                    "update_reason_counts": dict(gen.get("update_reason_counts", {})),
                    "bank_slot_count_before_generate": gen.get("bank_slot_count_before_generate"),
                    "bank_slot_count_after_generate": gen.get("bank_slot_count_after_generate"),
                    "query_write_count": 0,
                    "query_write_attempt_count": query_write_attempt_count if turn_type == "query" else 0,
                    "query_write_blocked": bool(turn_type == "query" and query_write_attempt_count == 0),
                }
                trigger_trace_rows.append(row)

            prompt_history_token_lens = [
                int(item["prompt_history_token_len"]) for item in prompt_trace
            ]
            prompt_history_hashes = [
                str(item["rendered_prompt_hash"]) for item in prompt_trace
            ]

            return {
                "prediction": env.final_answer,
                "prompt_trace": prompt_trace,
                "prompt_history_token_lens": prompt_history_token_lens,
                "prompt_history_hashes": prompt_history_hashes,
                "generations": generations,
                "trigger_trace_rows": trigger_trace_rows,
                "context_capacity": capacity,
                "bank_write_count": final_debug["memory_write_count"],
                "bank_retrieval_count": final_debug["memory_retrieve_count"],
                "bank_retrieved_latent_count": final_debug["retrieved_latent_count"],
                "retrieved_indices_by_turn": [list(gen["retrieved_indices"]) for gen in generations],
                "retrieved_scores_by_turn": [list(gen["retrieved_scores"]) for gen in generations],
                "bank_slot_count_final_before_reset": final_debug["slot_count"],
                "bank_reset_after_context": lifecycle["post_reset_slot_count"] == 0,
                "cross_context_leakage_detected": bool(
                    lifecycle.get("initial_slot_count", 0) != 0 or lifecycle["post_reset_slot_count"] != 0
                ),
                "query_write_count": 0,
                "query_write_attempt_count": query_write_attempt_count,
                "peak_cuda_memory": max(
                    [gen["peak_cuda_memory"] for gen in generations if gen["peak_cuda_memory"] is not None],
                    default=None,
                ),
                "latency_seconds": sum(gen["latency_sec"] for gen in generations),
                "trigger_active_flag": bool(model.config.trigger_active),
            }
        finally:
            model.generate = restore_hooks["generate"]
            model.reasoner_to_weaver.forward = restore_hooks["reasoner_to_weaver.forward"]
            model.weaver_to_reasoner.forward = restore_hooks["weaver_to_reasoner.forward"]
            model.weaver.augment_prompt = restore_hooks["weaver.augment_prompt"]
            model.weaver.augment_inference = restore_hooks["weaver.augment_inference"]
            model.reasoner.generate = restore_hooks["reasoner.generate"]
    finally:
        model.config.retrieved_memory_to_weaver = previous_flag
        model.config.memory_bank_storage_space = previous_storage


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
    row = parent._build_row(
        run_id=run_id,
        context_index=context_index,
        payload=payload,
        bank_off_result=bank_off_result,
        bank_on_result=bank_on_result,
        bank_off_score=bank_off_score,
        bank_on_score=bank_on_score,
        estimated_full_history_query_tokens=estimated_full_history_query_tokens,
        compressed_query_tokens_bank_off=compressed_query_tokens_bank_off,
        compressed_query_tokens_bank_on=compressed_query_tokens_bank_on,
    )
    trace_summary = _make_trace_context_summary(
        context_index=context_index,
        context_id=payload["context_id"],
        bank_on_result=bank_on_result,
        bank_on_exact_match=row["bank_on_exact_match"],
    )
    row.update(trace_summary)
    row["prompt_history_hashes"] = list(bank_on_result.get("prompt_history_hashes", []))
    row["prompt_history_token_lens"] = list(bank_on_result.get("prompt_history_token_lens", []))
    return row


def _aggregate(rows: list[dict]) -> dict:
    summary = parent._aggregate(rows)
    valid = [row for row in rows if not row.get("error_or_stop_reason")]
    summary.update(
        {
            "construction_trigger_forward_counts": [
                int(row.get("construction_trigger_forward_count", 0)) for row in valid
            ],
            "construction_trigger_invoke_counts": [
                int(row.get("construction_trigger_invoke_count", 0)) for row in valid
            ],
            "query_trigger_forward_counts": [
                int(row.get("query_trigger_forward_count", 0)) for row in valid
            ],
            "query_trigger_invoke_counts": [
                int(row.get("query_trigger_invoke_count", 0)) for row in valid
            ],
            "construction_weaver_call_counts": [
                int(row.get("construction_weaver_call_count", 0)) for row in valid
            ],
            "query_weaver_call_counts": [
                int(row.get("query_weaver_call_count", 0)) for row in valid
            ],
            "construction_memory_retrieve_counts": [
                int(row.get("construction_memory_retrieve_count", 0)) for row in valid
            ],
            "query_memory_retrieve_counts": [
                int(row.get("query_memory_retrieve_count", 0)) for row in valid
            ],
            "construction_memory_write_counts": [
                int(row.get("construction_memory_write_count", 0)) for row in valid
            ],
            "query_memory_write_counts": [
                int(row.get("query_memory_write_count", 0)) for row in valid
            ],
        }
    )
    return summary


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
    parser.add_argument("--requested-contexts", type=int, default=parent.DEFAULT_REQUESTED_CONTEXTS)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _write_trigger_trace_artifacts(output_dir: Path, paired_rows: list[dict], summary: dict) -> None:
    trace_rows = []
    context_summaries = []
    for row in paired_rows:
        if row.get("error_or_stop_reason"):
            continue
        context_summary = {
            "context_index": row["context_index"],
            "context_id": row["context_id"],
            "construction_trigger_forward_count": row["construction_trigger_forward_count"],
            "construction_trigger_invoke_count": row["construction_trigger_invoke_count"],
            "query_trigger_forward_count": row["query_trigger_forward_count"],
            "query_trigger_invoke_count": row["query_trigger_invoke_count"],
            "construction_weaver_call_count": row["construction_weaver_call_count"],
            "query_weaver_call_count": row["query_weaver_call_count"],
            "construction_memory_retrieve_count": row["construction_memory_retrieve_count"],
            "query_memory_retrieve_count": row["query_memory_retrieve_count"],
            "construction_memory_write_count": row["construction_memory_write_count"],
            "query_memory_write_count": row["query_memory_write_count"],
            "final_slot_count": row["final_slot_count"],
            "prediction": row["bank_on_prediction"],
            "exact_match": row["bank_on_exact_match"],
            "prompt_history_hashes": row.get("prompt_history_hashes", []),
            "prompt_history_token_lens": row.get("prompt_history_token_lens", []),
        }
        context_summaries.append(context_summary)
        prompt_hashes = row.get("prompt_history_hashes", [])
        prompt_lens = row.get("prompt_history_token_lens", [])
        bank_on_trace = row.get("bank_on_trigger_trace_rows", [])
        for turn_row in bank_on_trace:
            turn_copy = dict(turn_row)
            turn_index = int(turn_copy["turn_index"])
            if turn_index < len(prompt_hashes):
                turn_copy["rendered_prompt_hash"] = prompt_hashes[turn_index]
            if turn_index < len(prompt_lens):
                turn_copy["prompt_history_token_len"] = prompt_lens[turn_index]
            trace_rows.append(turn_copy)

    trigger_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "num_contexts": len(context_summaries),
        "contexts": context_summaries,
        "aggregate": {
            "bank_off_exact_match": summary.get("compressed_bank_off_accuracy"),
            "bank_on_exact_match": summary.get("compressed_bank_on_accuracy"),
            "output_changed": summary.get("num_output_changed"),
            "num_improved": summary.get("num_improved"),
            "num_regressed": summary.get("num_regressed"),
            "final_slot_counts": summary.get("final_slot_counts"),
            "construction_trigger_forward_counts": summary.get("construction_trigger_forward_counts"),
            "construction_trigger_invoke_counts": summary.get("construction_trigger_invoke_counts"),
            "query_trigger_forward_counts": summary.get("query_trigger_forward_counts"),
            "query_trigger_invoke_counts": summary.get("query_trigger_invoke_counts"),
            "construction_weaver_call_counts": summary.get("construction_weaver_call_counts"),
            "query_weaver_call_counts": summary.get("query_weaver_call_counts"),
            "construction_memory_retrieve_counts": summary.get("construction_memory_retrieve_counts"),
            "query_memory_retrieve_counts": summary.get("query_memory_retrieve_counts"),
            "construction_memory_write_counts": summary.get("construction_memory_write_counts"),
            "query_memory_write_counts": summary.get("query_memory_write_counts"),
            "query_write_count": summary.get("query_write_count"),
            "query_write_attempt_count": summary.get("query_write_attempt_count"),
        },
    }
    _write_jsonl(output_dir / "trigger_trace.jsonl", trace_rows)
    _write_json(output_dir / "trigger_trace_summary.json", trigger_summary)


def main():
    args = build_parser().parse_args()
    args.parquet = str(Path(args.dataset_root) / "data/Long_Range_Understanding-00000-of-00001.parquet")
    args.data_config = str(Path(args.mab_repo) / base.DATA_CONFIG)
    started_at = _utc_now()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{RUN_PREFIX}"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    git_status_before = _git("status", "--short", "--branch")
    manifest = _build_manifest(run_id, args, started_at, git_status_before=git_status_before)
    diagnostics = []
    paired_rows = []
    model, capacity = _load_model(args)
    tokenizer = model.tokenizer
    manifest["context_capacity"] = capacity
    try:
        total_matches = base.count_context_matches(args.parquet, base.SUB_DATASET)
        match_indices = base.select_match_indices(total_matches, args.requested_contexts)
        manifest["selection_policy"] = {
            "requested": args.requested_contexts,
            "available": total_matches,
            "selected_match_indices": match_indices,
            "order": "parquet_row_order_filtered_by_metadata.source",
        }
        with tempfile.TemporaryDirectory(prefix="mab6b-trigger-trace-") as tmpdir:
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
                    payload = base._prepare_payload(args, payload_path, match_index, started_at)
                    bank_off_result = _run_model(args, model, capacity, payload, bank_mode="off")
                    estimated_full_history_query_tokens = base.estimate_full_history_query_tokens(
                        tokenizer, payload
                    )
                    if estimated_full_history_query_tokens > bank_off_result["context_capacity"]:
                        full_history_status = "over_capacity_invalid"
                    else:
                        full_history_status = "under_capacity_but_not_executed"
                    bank_off_query_tokens, _, query_chunk_leak, query_ack_leak = base.compressed_query_token_count(
                        tokenizer, payload
                    )
                    if query_chunk_leak or query_ack_leak:
                        raise RuntimeError("Compressed query prompt leaked chunk or acknowledgement history")
                    bank_on_result = _run_model(args, model, capacity, payload, bank_mode="on")
                    if bank_on_result["cross_context_leakage_detected"]:
                        raise RuntimeError("Cross-context leakage detected")
                    bank_on_query_tokens = bank_on_result["prompt_trace"][-1]["prompt_history_token_len"]
                    bank_off_score = base._score_prediction(args, payload, bank_off_result["prediction"], tmpdir)
                    bank_on_score = base._score_prediction(args, payload, bank_on_result["prediction"], tmpdir)
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
                    row["bank_on_trigger_trace_rows"] = [
                        {**trace_row, "context_index": context_index}
                        for trace_row in bank_on_result["trigger_trace_rows"]
                    ]
                except Exception as error:
                    row = base._empty_context_row(
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
                "split": base.SPLIT,
                "subtask": base.SUB_DATASET,
                "query_mode": "first-query-only",
                "threshold": parent.DEFAULT_THRESHOLD,
                "top_k": parent.DEFAULT_TOP_K,
                "max_slots": parent.DEFAULT_MAX_SLOTS,
                "retrieve_policy": parent.DEFAULT_RETRIEVE_POLICY,
                "chunk_size": base.DEFAULT_CHUNK_SIZE,
                "requested_contexts": args.requested_contexts,
                "timestamp": started_at,
                "external_api_used": False,
                "retrieve_threshold": parent.DEFAULT_RETRIEVE_THRESHOLD,
                "update_threshold": parent.DEFAULT_UPDATE_THRESHOLD,
                "retrieved_memory_to_weaver": True,
                "memory_bank_storage_space": "weaver",
                "instrumentation": "trigger_trace",
            },
        )
        _write_trigger_trace_artifacts(output_dir, paired_rows, summary)
        print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))
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
                "split": base.SPLIT,
                "subtask": base.SUB_DATASET,
                "query_mode": "first-query-only",
                "threshold": parent.DEFAULT_THRESHOLD,
                "top_k": parent.DEFAULT_TOP_K,
                "max_slots": parent.DEFAULT_MAX_SLOTS,
                "retrieve_policy": parent.DEFAULT_RETRIEVE_POLICY,
                "chunk_size": base.DEFAULT_CHUNK_SIZE,
                "requested_contexts": args.requested_contexts,
                "timestamp": started_at,
                "external_api_used": False,
                "retrieve_threshold": parent.DEFAULT_RETRIEVE_THRESHOLD,
                "update_threshold": parent.DEFAULT_UPDATE_THRESHOLD,
                "retrieved_memory_to_weaver": True,
                "memory_bank_storage_space": "weaver",
                "instrumentation": "trigger_trace",
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
