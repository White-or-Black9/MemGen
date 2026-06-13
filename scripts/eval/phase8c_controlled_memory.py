import argparse
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
from transformers import GenerationConfig

from common.config import Config
from main import set_seed
from memgen.model import MemGenModel
from memgen.model.latent_memory_bank import (
    LatentMemoryBank,
    LatentMemoryBankConfig,
)


DEFAULT_CONFIG_PATH = "configs/latent_memory/gsm8k.yaml"
SYSTEM_PROMPT = (
    "Follow the current instruction exactly. Information may be introduced "
    "earlier in the same session. Use <continue> for acknowledgement turns "
    "and <answer>...</answer> for the final answer."
)

GROUP_CONFIGS = {
    "G0_disabled": {
        "memory_mode": "disabled",
        "memory_enabled": False,
        "update_policy": None,
        "oracle_visible": False,
    },
    "G1_vA_simple": {
        "memory_mode": "vA_simple",
        "memory_enabled": True,
        "update_policy": "replace_oldest",
        "oracle_visible": False,
    },
    "G2_vA_thread_update": {
        "memory_mode": "vA_thread_update",
        "memory_enabled": True,
        "update_policy": "thread_update",
        "oracle_visible": False,
    },
    "G3_oracle_visible": {
        "memory_mode": "disabled",
        "memory_enabled": False,
        "update_policy": None,
        "oracle_visible": True,
    },
}


@dataclass(frozen=True)
class ControlledEpisode:
    episode_id: str
    episode_type: str
    entity: str
    attribute: str
    value: str
    early_fact_text: str
    distractor_text: str
    query_text: str
    gold_answer: str


@dataclass(frozen=True)
class ParsedAnswer:
    answer: Optional[str]
    parser_mode: str
    success: bool


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Phase 8C controlled multi-turn mechanism study."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group", choices=sorted(GROUP_CONFIGS), required=True)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-response-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--memory-mode",
        choices=("disabled", "vA_simple", "vA_thread_update"),
        required=True,
    )
    return parser.parse_args()


def validate_args(args) -> Dict[str, Any]:
    if args.batch_size != 1:
        raise ValueError("Controlled memory evaluation supports batch_size=1 only")
    if args.sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if args.seed != 42:
        raise ValueError("Phase 8C controlled evaluation requires seed=42")
    if args.max_response_length <= 0:
        raise ValueError("max_response_length must be positive")
    group_config = get_group_config(args.group)
    if args.memory_mode != group_config["memory_mode"]:
        raise ValueError(
            f"group {args.group} requires memory_mode="
            f"{group_config['memory_mode']}"
        )
    return group_config


def get_group_config(group: str) -> Dict[str, Any]:
    if group not in GROUP_CONFIGS:
        raise ValueError(f"Unsupported group: {group}")
    return dict(GROUP_CONFIGS[group])


def generate_episodes(sample_count: int, seed: int = 42) -> List[ControlledEpisode]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    rng = random.Random(seed)
    project_names = [
        "Project Lumen",
        "Project Sable",
        "Project Vireo",
        "Project Kestrel",
        "Project Nacre",
        "Project Solace",
    ]
    distractor_names = [
        "Project Amber",
        "Project Cobalt",
        "Project Willow",
        "Project Onyx",
        "Project Harbor",
        "Project Quartz",
    ]
    semantic_values = [
        "Archive Cedar",
        "Room Juniper",
        "Vault Meridian",
        "Folder Nimbus",
        "Cabinet Orion",
        "Desk Topaz",
    ]
    episodes = []
    for index in range(sample_count):
        entity = project_names[index % len(project_names)]
        distractor_entity = distractor_names[index % len(distractor_names)]
        if index % 2 == 0:
            episode_type = "exact_code"
            attribute = "access code"
            value = f"{rng.randint(100000, 999999)}"
        else:
            episode_type = "semantic_relation"
            attribute = "assigned archive"
            value = semantic_values[index % len(semantic_values)]
        early_fact_text = f"The {attribute} for {entity} is {value}."
        distractor_text = (
            f"{distractor_entity} uses the blue folder and meets on Tuesday."
        )
        query_text = f"What is the {attribute} for {entity}?"
        if value in distractor_text:
            raise RuntimeError("Distractor unexpectedly contains the gold answer")
        episodes.append(
            ControlledEpisode(
                episode_id=f"early_fact_{index:04d}",
                episode_type=episode_type,
                entity=entity,
                attribute=attribute,
                value=value,
                early_fact_text=early_fact_text,
                distractor_text=distractor_text,
                query_text=query_text,
                gold_answer=value,
            )
        )
    return episodes


def build_user_turns(
    episode: ControlledEpisode,
    *,
    oracle_visible: bool = False,
) -> List[str]:
    turn1 = (
        "Remember this session-specific fact:\n"
        f"{episode.early_fact_text}\n"
        "Reply only with <continue>."
    )
    turn2 = (
        "Unrelated update:\n"
        f"{episode.distractor_text}\n"
        "Reply only with <continue>."
    )
    turn3_parts = []
    if oracle_visible:
        turn3_parts.append(
            "Visible reference for this oracle control:\n"
            f"{episode.early_fact_text}\n"
        )
    turn3_parts.append(
        f"{episode.query_text}\n"
        "Return exactly one line:\n"
        "<answer>VALUE</answer>\n"
        "Do not include any other text."
    )
    return [turn1, turn2, "".join(turn3_parts)]


def build_turn_prompts(
    episode: ControlledEpisode,
    *,
    oracle_visible: bool = False,
) -> List[List[Dict[str, str]]]:
    user_turns = build_user_turns(episode, oracle_visible=oracle_visible)
    return [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_turn},
        ]
        for user_turn in user_turns
    ]


def render_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if isinstance(rendered, list):
        if len(rendered) != 1:
            raise ValueError("Expected one rendered prompt")
        return rendered[0]
    return rendered


def leakage_flags(
    turn3_prompt_text: str,
    episode: ControlledEpisode,
    *,
    oracle_visible: bool = False,
) -> Dict[str, bool]:
    normalized = turn3_prompt_text.lower()
    user_turns = build_user_turns(episode, oracle_visible=False)
    flags = {
        "prompt_contains_gold_answer": episode.gold_answer.lower() in normalized,
        "prompt_contains_early_fact": episode.early_fact_text.lower() in normalized,
        "prompt_contains_distractor": episode.distractor_text.lower() in normalized,
        "prompt_contains_turn1_text": user_turns[0].lower() in normalized,
        "prompt_contains_turn2_text": user_turns[1].lower() in normalized,
    }
    flags["leakage_passed"] = (
        oracle_visible
        or not any(value for key, value in flags.items() if key != "leakage_passed")
    )
    return flags


def parse_strict_answer(response: str) -> Optional[str]:
    matches = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if not matches:
        return None
    return matches[-1].strip()


def _strip_semantic_response(response: str) -> str:
    candidate = response.strip()
    quote_pairs = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"))
    for opening, closing in quote_pairs:
        if candidate.startswith(opening) and candidate.endswith(closing):
            candidate = candidate[len(opening) : -len(closing)].strip()
            break
    candidate = re.sub(r"[.!?。！？]\s*$", "", candidate, count=1).strip()
    return candidate


def parse_relaxed_answer(response: str, episode_type: str) -> ParsedAnswer:
    strict_answer = parse_strict_answer(response)
    if strict_answer is not None:
        return ParsedAnswer(strict_answer, "strict_tag", True)

    if episode_type == "exact_code":
        candidates = re.findall(r"(?<!\d)\d{6}(?!\d)", response)
        if len(candidates) == 1:
            return ParsedAnswer(
                candidates[0],
                "exact_code_single_candidate",
                True,
            )
        if len(candidates) > 1:
            return ParsedAnswer(None, "ambiguous", False)
        return ParsedAnswer(None, "none", False)

    if episode_type == "semantic_relation":
        candidate = _strip_semantic_response(response)
        if candidate:
            return ParsedAnswer(candidate, "semantic_full_response", True)
        return ParsedAnswer(None, "none", False)

    raise ValueError(f"Unsupported episode_type: {episode_type}")


def normalize_answer(answer: Optional[str]) -> str:
    if answer is None:
        return ""
    return " ".join(answer.strip().lower().split())


def exact_match(answer: Optional[str], gold_answer: str) -> bool:
    return normalize_answer(answer) == normalize_answer(gold_answer)


def compute_metrics(response: str, episode: ControlledEpisode) -> Dict[str, Any]:
    strict_answer = parse_strict_answer(response)
    relaxed = parse_relaxed_answer(response, episode.episode_type)
    return {
        "parsed_answer_strict": strict_answer,
        "parsed_answer_relaxed": relaxed.answer,
        "strict_parser_success": strict_answer is not None,
        "relaxed_parser_success": relaxed.success,
        "parser_mode": relaxed.parser_mode,
        "strict_exact_match": exact_match(strict_answer, episode.gold_answer),
        "relaxed_exact_match": exact_match(relaxed.answer, episode.gold_answer),
    }


# Deprecated compatibility alias. New code should use parse_strict_answer().
parse_answer = parse_strict_answer


def build_config_args(args, model_path: str, checkpoint_path: str):
    options = [
        "model.model_name",
        model_path,
        "model.load_model_path",
        checkpoint_path,
        "model.max_prompt_aug_num",
        "1",
        "model.max_inference_aug_num",
        "3",
        "model.weaver.model_name",
        model_path,
        "model.weaver.prompt_latents_len",
        "8",
        "model.weaver.inference_latents_len",
        "8",
        "model.trigger.model_name",
        model_path,
        "model.trigger.active",
        "False",
        "run.seed",
        str(args.seed),
    ]
    return SimpleNamespace(cfg_path=DEFAULT_CONFIG_PATH, options=options)


def create_memory_bank(group_config: Dict[str, Any]) -> Optional[LatentMemoryBank]:
    if not group_config["memory_enabled"]:
        return None
    bank = LatentMemoryBank(
        LatentMemoryBankConfig(
            enabled=True,
            batch_size=1,
            max_slots=8,
            top_k=1,
            threshold=0.7,
            decay_alpha=0.05,
            pool_last_n=64,
            update_policy=group_config["update_policy"],
            retrieve_policy="threshold_topk",
            storage_device="cpu",
            debug=True,
        )
    )
    bank.reset()
    return bank


def install_model_trace(model, trace: Dict[str, Any]):
    original_should_augment = model._should_augment
    original_augment_prompt = model.weaver.augment_prompt
    original_augment_inference = model.weaver.augment_inference
    original_reasoner_to_weaver = model.reasoner_to_weaver.forward

    def tracked_should_augment(self, *args, **kwargs):
        trace["trigger_calls"] += 1
        return original_should_augment(*args, **kwargs)

    def tracked_augment_prompt(self, *args, **kwargs):
        trace["weaver_prompt_calls"] += 1
        trace["weaver_input_token_counts"].append(int(args[0].shape[1]))
        return original_augment_prompt(*args, **kwargs)

    def tracked_augment_inference(self, *args, **kwargs):
        trace["weaver_inference_calls"] += 1
        trace["weaver_input_token_counts"].append(int(args[0].shape[1]))
        return original_augment_inference(*args, **kwargs)

    def tracked_reasoner_to_weaver(module, tensor):
        trace["reasoner_to_weaver_input_token_counts"].append(
            int(tensor.shape[1])
        )
        return original_reasoner_to_weaver(tensor)

    model._should_augment = MethodType(tracked_should_augment, model)
    model.weaver.augment_prompt = MethodType(
        tracked_augment_prompt,
        model.weaver,
    )
    model.weaver.augment_inference = MethodType(
        tracked_augment_inference,
        model.weaver,
    )
    model.reasoner_to_weaver.forward = MethodType(
        tracked_reasoner_to_weaver,
        model.reasoner_to_weaver,
    )

    def restore():
        model._should_augment = original_should_augment
        model.weaver.augment_prompt = original_augment_prompt
        model.weaver.augment_inference = original_augment_inference
        model.reasoner_to_weaver.forward = original_reasoner_to_weaver

    return restore


def install_retrieval_trace(
    bank: Optional[LatentMemoryBank],
    trace: List[Dict[str, Any]],
) -> None:
    if bank is None:
        return
    original_retrieve_with_context = bank.retrieve_with_context

    def tracked_retrieve_with_context(self, *args, **kwargs):
        result = original_retrieve_with_context(*args, **kwargs)
        trace.append(
            {
                "scores": list(result.scores),
                "max_score": result.max_score,
                "argmax_index": result.argmax_index,
                "threshold_passed": result.threshold_passed,
                "retrieved_indices": list(result.retrieved_indices),
                "retrieved_scores": list(result.retrieved_scores),
                "bank_step": result.bank_step,
            }
        )
        return result

    bank.retrieve_with_context = MethodType(tracked_retrieve_with_context, bank)


def memory_boundary_summary(
    bank_debug: Optional[Dict[str, Any]],
    model_trace: Dict[str, Any],
) -> Dict[str, Any]:
    stored_hidden_sizes = []
    if bank_debug is not None:
        stored_hidden_sizes = [
            slot["memory_shape"][-1]
            for slot in bank_debug.get("slots", [])
            if slot.get("memory_shape")
        ]
    weaver_inputs_unchanged = (
        model_trace["weaver_input_token_counts"]
        == model_trace["reasoner_to_weaver_input_token_counts"]
    )
    return {
        "retrieved_memory_reasoner_only": {
            "available": True,
            "passed": weaver_inputs_unchanged,
            "evidence": "Weaver input token counts equal reasoner-to-Weaver counts",
        },
        "weaver_input_token_counts": list(
            model_trace["weaver_input_token_counts"]
        ),
        "reasoner_to_weaver_input_token_counts": list(
            model_trace["reasoner_to_weaver_input_token_counts"]
        ),
        "stored_latent_hidden_sizes": stored_hidden_sizes,
        "stored_latent_reasoner_space": {
            "available": bank_debug is not None,
            "passed": (
                all(size == 1536 for size in stored_hidden_sizes)
                if stored_hidden_sizes
                else None
            ),
        },
        "fallback_top1_implemented": False,
        "last_retrieved_decay_implemented": False,
        "version_b": False,
    }


def required_episode_fields_present(record: Dict[str, Any]) -> bool:
    required = {
        "episode_id",
        "group",
        "entity",
        "attribute",
        "gold_answer",
        "turns",
        "final_answer",
        "exact_match",
        "exact_match_metric",
        "parsed_answer_strict",
        "parsed_answer_relaxed",
        "strict_parser_success",
        "relaxed_parser_success",
        "parser_mode",
        "strict_exact_match",
        "relaxed_exact_match",
        "reward",
        "valid_episode",
        "invalid_reason",
        "memory_bank_debug",
        "bank_lifecycle",
        "trigger_calls",
        "weaver_prompt_calls",
        "weaver_inference_calls",
        "latency",
        "errors",
    }
    return required.issubset(record)


def run_turn(
    model,
    tokenizer,
    messages,
    generation_config,
    bank,
) -> Dict[str, Any]:
    prompt_text = render_prompt(tokenizer, messages)
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    start = time.perf_counter()
    output_ids, augmentation_mask = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        latent_memory_bank=bank,
        return_augmentation_mask=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - start
    response_ids = output_ids[:, input_ids.size(1) :]
    response = tokenizer.batch_decode(
        response_ids,
        skip_special_tokens=True,
    )[0]
    return {
        "visible_prompt": prompt_text,
        "response": response,
        "prompt_token_length": int(input_ids.size(1)),
        "augmentation_mask": augmentation_mask.detach().cpu().tolist(),
        "latency": latency,
    }


def run_episode(
    model,
    tokenizer,
    episode: ControlledEpisode,
    group: str,
    group_config: Dict[str, Any],
    generation_config: GenerationConfig,
) -> Dict[str, Any]:
    bank = create_memory_bank(group_config)
    bank_id = id(bank) if bank is not None else None
    retrieval_trace: List[Dict[str, Any]] = []
    install_retrieval_trace(bank, retrieval_trace)
    prompts = build_turn_prompts(
        episode,
        oracle_visible=group_config["oracle_visible"],
    )
    model_trace = {
        "trigger_calls": 0,
        "weaver_prompt_calls": 0,
        "weaver_inference_calls": 0,
        "weaver_input_token_counts": [],
        "reasoner_to_weaver_input_token_counts": [],
    }
    restore_model_trace = install_model_trace(model, model_trace)
    turns = []
    errors = []
    episode_start = time.perf_counter()
    slots_after_turns = []
    try:
        for turn_index, messages in enumerate(prompts, start=1):
            slots_before = len(bank) if bank is not None else None
            turn = run_turn(
                model,
                tokenizer,
                messages,
                generation_config,
                bank,
            )
            turn["turn_index"] = turn_index
            turn["bank_id"] = bank_id
            turn["slots_before"] = slots_before
            turn["slots_after"] = len(bank) if bank is not None else None
            turn_metrics = (
                compute_metrics(turn["response"], episode)
                if turn_index == 3
                else {
                    "parsed_answer_strict": None,
                    "parsed_answer_relaxed": None,
                    "strict_parser_success": False,
                    "relaxed_parser_success": False,
                    "parser_mode": "none",
                    "strict_exact_match": False,
                    "relaxed_exact_match": False,
                }
            )
            turn.update(turn_metrics)
            turn["parsed_answer"] = turn["parsed_answer_strict"]
            turn["leakage_flags"] = (
                leakage_flags(
                    turn["visible_prompt"],
                    episode,
                    oracle_visible=group_config["oracle_visible"],
                )
                if turn_index == 3
                else {}
            )
            turns.append(turn)
            slots_after_turns.append(len(bank) if bank is not None else None)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        restore_model_trace()
    latency = time.perf_counter() - episode_start
    final_turn = turns[-1] if len(turns) == 3 else None
    parsed_answer_strict = (
        final_turn["parsed_answer_strict"] if final_turn is not None else None
    )
    parsed_answer_relaxed = (
        final_turn["parsed_answer_relaxed"] if final_turn is not None else None
    )
    strict_parser_success = bool(
        final_turn["strict_parser_success"] if final_turn is not None else False
    )
    relaxed_parser_success = bool(
        final_turn["relaxed_parser_success"] if final_turn is not None else False
    )
    parser_mode = final_turn["parser_mode"] if final_turn is not None else "none"
    turn3_flags = turns[-1]["leakage_flags"] if len(turns) == 3 else {}
    leakage_passed = bool(turn3_flags.get("leakage_passed", False))
    valid_episode = len(turns) == 3 and leakage_passed and not errors
    invalid_reason = None
    if errors:
        invalid_reason = errors[0]
    elif len(turns) != 3:
        invalid_reason = "episode_did_not_complete_three_turns"
    elif not leakage_passed:
        invalid_reason = "turn3_prompt_leakage"
    strict_is_exact = bool(
        valid_episode and final_turn and final_turn["strict_exact_match"]
    )
    relaxed_is_exact = bool(
        valid_episode and final_turn and final_turn["relaxed_exact_match"]
    )
    bank_debug = bank.debug_summary() if bank is not None else None
    boundary_summary = memory_boundary_summary(bank_debug, model_trace)
    return {
        "episode_id": episode.episode_id,
        "episode_type": episode.episode_type,
        "group": group,
        "entity": episode.entity,
        "attribute": episode.attribute,
        "gold_answer": episode.gold_answer,
        "turns": turns,
        "turn3_prompt_text": (
            turns[-1]["visible_prompt"] if len(turns) == 3 else None
        ),
        "final_answer": parsed_answer_strict,
        "parsed_answer_strict": parsed_answer_strict,
        "parsed_answer_relaxed": parsed_answer_relaxed,
        "strict_parser_success": strict_parser_success,
        "relaxed_parser_success": relaxed_parser_success,
        "parser_mode": parser_mode,
        "strict_exact_match": strict_is_exact,
        "relaxed_exact_match": relaxed_is_exact,
        "exact_match": strict_is_exact,
        "exact_match_metric": "strict_exact_match_deprecated_alias",
        "reward": float(strict_is_exact),
        "valid_episode": valid_episode,
        "invalid_reason": invalid_reason,
        "memory_bank_debug": bank_debug,
        "retrieval_trace": retrieval_trace,
        "bank_lifecycle": {
            "bank_created": bank is not None,
            "bank_id": bank_id,
            "initial_slots": 0 if bank is not None else None,
            "slots_after_turns": slots_after_turns,
            "final_slots": len(bank) if bank is not None else None,
            "same_bank_across_turns": (
                bool(turns)
                and all(turn["bank_id"] == bank_id for turn in turns)
                if bank is not None
                else None
            ),
        },
        "memory_boundary_checks": boundary_summary,
        "trigger_calls": model_trace["trigger_calls"],
        "weaver_prompt_calls": model_trace["weaver_prompt_calls"],
        "weaver_inference_calls": model_trace["weaver_inference_calls"],
        "latency": latency,
        "errors": errors,
    }


def build_summary(
    group: str,
    records: List[Dict[str, Any]],
    total_latency: float,
) -> Dict[str, Any]:
    valid_records = [record for record in records if record["valid_episode"]]
    strict_exact_count = sum(
        record["strict_exact_match"] for record in valid_records
    )
    relaxed_exact_count = sum(
        record["relaxed_exact_match"] for record in valid_records
    )
    strict_parser_success_count = sum(
        record["strict_parser_success"] for record in valid_records
    )
    relaxed_parser_success_count = sum(
        record["relaxed_parser_success"] for record in valid_records
    )
    valid_count = len(valid_records)
    leakage_pass_count = sum(
        bool(record["turns"][-1]["leakage_flags"].get("leakage_passed"))
        for record in records
        if len(record["turns"]) == 3
    )
    crash_count = sum(bool(record["errors"]) for record in records)
    latencies = [record["latency"] for record in records]
    return {
        "summary": {
            "group": group,
            "sample_count": len(records),
            "valid_episode_count": valid_count,
            "leakage_pass_count": leakage_pass_count,
            "strict_exact_match_count": strict_exact_count,
            "strict_exact_match_rate": (
                strict_exact_count / valid_count if valid_count else 0.0
            ),
            "relaxed_exact_match_count": relaxed_exact_count,
            "relaxed_exact_match_rate": (
                relaxed_exact_count / valid_count if valid_count else 0.0
            ),
            "strict_parser_success_count": strict_parser_success_count,
            "relaxed_parser_success_count": relaxed_parser_success_count,
            "exact_match_count": strict_exact_count,
            "exact_match_rate": (
                strict_exact_count / valid_count if valid_count else 0.0
            ),
            "exact_match_metric": "strict_exact_match_deprecated_alias",
            "mean_reward": (
                sum(record["reward"] for record in valid_records)
                / valid_count
                if valid_count
                else 0.0
            ),
            "mean_latency": sum(latencies) / len(latencies) if latencies else 0.0,
            "total_latency": total_latency,
            "crash_count": crash_count,
            "memory_boundary_summary": {
                "all_disabled_banks_absent": all(
                    not record["bank_lifecycle"]["bank_created"]
                    for record in records
                ),
                "all_reasoner_only_checks_passed": all(
                    record["memory_boundary_checks"][
                        "retrieved_memory_reasoner_only"
                    ]["passed"]
                    for record in records
                ),
                "version_b": False,
            },
        }
    }


def build_verification(
    records: List[Dict[str, Any]],
    summary_record: Dict[str, Any],
    group_config: Dict[str, Any],
    peak_cuda_memory_bytes: Optional[int],
) -> Dict[str, Any]:
    summary = summary_record["summary"]
    return {
        "sample_count": summary["sample_count"],
        "group_config": group_config,
        "leakage_pass_count": summary["leakage_pass_count"],
        "valid_episode_count": summary["valid_episode_count"],
        "strict_exact_match_count": summary["strict_exact_match_count"],
        "strict_exact_match_rate": summary["strict_exact_match_rate"],
        "relaxed_exact_match_count": summary["relaxed_exact_match_count"],
        "relaxed_exact_match_rate": summary["relaxed_exact_match_rate"],
        "strict_parser_success_count": summary["strict_parser_success_count"],
        "relaxed_parser_success_count": summary["relaxed_parser_success_count"],
        "exact_match_count": summary["exact_match_count"],
        "exact_match_rate": summary["exact_match_rate"],
        "exact_match_metric": "strict_exact_match_deprecated_alias",
        "memory_boundary_checks": [
            record["memory_boundary_checks"] for record in records
        ],
        "crash_count": summary["crash_count"],
        "non_finite_metric": any(
            not math.isfinite(float(record["reward"])) for record in records
        ),
        "disabled_path_bank_created": any(
            record["bank_lifecycle"]["bank_created"] for record in records
        )
        if not group_config["memory_enabled"]
        else None,
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
        "fallback_top1": False,
        "last_retrieved_decay": False,
        "version_b": False,
    }


def write_artifacts(
    output_dir: Path,
    records: List[Dict[str, Any]],
    summary_record: Dict[str, Any],
    run_config: Dict[str, Any],
    verification: Dict[str, Any],
) -> None:
    evaluate_dir = output_dir / "evaluate"
    evaluate_dir.mkdir(parents=True, exist_ok=True)
    answer_path = evaluate_dir / "answer.json"
    with answer_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.write(json.dumps(summary_record, ensure_ascii=False) + "\n")

    conversation_lines = []
    for record in records:
        conversation_lines.append(f"Episode: {record['episode_id']}")
        for turn in record["turns"]:
            conversation_lines.extend(
                [
                    f"Turn {turn['turn_index']} Prompt:",
                    turn["visible_prompt"],
                    f"Turn {turn['turn_index']} Response:",
                    turn["response"],
                ]
            )
        conversation_lines.append(f"Reward: {record['reward']:.4f}")
        conversation_lines.append("-" * 40)
    (evaluate_dir / "conversations.txt").write_text(
        "\n".join(conversation_lines) + "\n",
        encoding="utf-8",
    )
    (output_dir / "verification.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "memory_trace.json").write_text(
        json.dumps(
            [
                {
                    "episode_id": record["episode_id"],
                    "retrieval_trace": record["retrieval_trace"],
                    "memory_bank_debug": record["memory_bank_debug"],
                }
                for record in records
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    group_config = validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(Path(args.model_path).resolve())
    checkpoint_path = str(Path(args.checkpoint_path).resolve())
    set_seed(args.seed, use_gpu=True)

    config = Config(build_config_args(args, model_path, checkpoint_path))
    config_dict = config.to_dict()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Controlled memory model smoke requires CUDA for FlashAttention"
        )
    model = MemGenModel.from_config(config_dict["model"]).to(
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    model.eval()
    tokenizer = model.tokenizer
    generation_config = GenerationConfig(
        max_new_tokens=args.max_response_length,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generation_config.weaver_do_sample = False
    generation_config.trigger_do_sample = False

    episodes = generate_episodes(args.sample_count, args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    records = [
        run_episode(
            model,
            tokenizer,
            episode,
            args.group,
            group_config,
            generation_config,
        )
        for episode in episodes
    ]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_latency = time.perf_counter() - start
    summary_record = build_summary(args.group, records, total_latency)

    run_config = {
        "phase": "Phase 8C-alt",
        "purpose": "controlled multi-turn mechanism study",
        "command": " ".join(sys.argv),
        "group": args.group,
        "group_config": group_config,
        "sample_count": args.sample_count,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_response_length": args.max_response_length,
        "model_path": model_path,
        "checkpoint_path": checkpoint_path,
        "base_config": DEFAULT_CONFIG_PATH,
        "write_age_decay": True,
        "fallback_top1": False,
        "version_b": False,
    }
    verification = build_verification(
        records,
        summary_record,
        group_config,
        (
            torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else None
        ),
    )
    write_artifacts(
        output_dir,
        records,
        summary_record,
        run_config,
        verification,
    )
    print(json.dumps(verification, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
