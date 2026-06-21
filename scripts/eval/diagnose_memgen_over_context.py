"""Controlled Bank-off diagnostic for MemGen over-context behavior near capacity."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import site
import sys
import warnings
from contextlib import redirect_stderr
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAB_REPO = Path("/mnt/18T/baishilong/benchmarks/MemoryAgentBench")
MABENCH_SITE_PACKAGES = "/home/baishilong/miniconda3/envs/MABench/lib/python3.10/site-packages"
DEFAULT_QWEN_PATH = (
    "/home/baishilong/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)
DEFAULT_CHECKPOINT_PATH = (
    "/home/baishilong/.cache/huggingface/hub/"
    "models--Kana-s--MemGen/snapshots/"
    "269d9b1741130b94fffa410cdaa3d4bc74081a7f/"
    "Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model"
)
DEFAULT_CHECKPOINT_ID = (
    "Kana-s/MemGen@269d9b1/"
    "Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model"
)
DEFAULT_CFG_PATH = "configs/latent_memory/triviaqa.yaml"
DEFAULT_OUTPUT_ROOT = "outputs/mab/memgen_over_context_behavior"
DEFAULT_TEST_LENGTHS = [32000, 32760, 32800, 35000]
DEFAULT_MAB_DATASET_ROOT = "/mnt/18T/baishilong/datasets/MemoryAgentBench"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _git(*args: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _build_config(args, context_capacity: int) -> dict:
    from common.config import Config
    from types import SimpleNamespace

    options = [
        "model.model_name", args.model_path,
        "model.load_model_path", args.checkpoint_path,
        "model.weaver.model_name", args.model_path,
        "model.trigger.model_name", args.model_path,
        "run.mode", "evaluate",
        "run.seed", str(args.seed),
        "run.interaction.max_turns", "1",
        "run.interaction.max_start_length", str(context_capacity),
        "run.interaction.max_prompt_length", str(context_capacity),
        "run.interaction.max_obs_length", "16",
        "run.interaction.max_response_length", str(args.max_new_tokens),
        "run.interaction.batch_size", "1",
        "run.interaction.temperature", "0.0",
        "run.interaction.weaver_do_sample", "False",
        "run.interaction.trigger_do_sample", "False",
        "run.latent_memory_bank.enabled", "False",
        "run.latent_memory_bank.batch_size", "1",
    ]
    return Config(SimpleNamespace(cfg_path=args.cfg_path, options=options)).to_dict()


def _ensure_mabench_packages() -> None:
    if MABENCH_SITE_PACKAGES not in sys.path:
        site.addsitedir(MABENCH_SITE_PACKAGES)


def _make_messages(tokenizer, target_tokens: int) -> tuple[list[dict], int]:
    system_text = "You are a helpful assistant that can read the context and memorize it for future retrieval."
    unit = " alpha"

    def render_len(repetitions: int) -> int:
        content = "Synthetic context:" + (unit * repetitions)
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": content},
        ]
        rendered = tokenizer.apply_chat_template(
            [messages],
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )["input_ids"]
        return int(rendered.shape[1])

    low = 0
    high = 1
    while render_len(high) < target_tokens:
        high *= 2

    while low + 1 < high:
        mid = (low + high) // 2
        if render_len(mid) < target_tokens:
            low = mid
        else:
            high = mid

    repetitions = high
    content = "Synthetic context:" + (unit * repetitions)
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": content},
    ]
    actual = render_len(repetitions)
    return messages, actual


def _collect_warnings(func):
    stream = io.StringIO()
    logger = logging.getLogger()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.WARNING)
    logger.addHandler(handler)
    old_level = logger.level
    if old_level > logging.WARNING:
        logger.setLevel(logging.WARNING)
    caught_warning_messages = []
    try:
        with warnings.catch_warnings(record=True) as caught, redirect_stderr(stream):
            warnings.simplefilter("always")
            try:
                result = func()
            except Exception as exc:
                messages = [str(item.message) for item in caught]
                text = stream.getvalue().strip()
                if text:
                    messages.extend(line for line in text.splitlines() if line.strip())
                exc._captured_warning_messages = messages
                raise
            caught_warning_messages = [str(item.message) for item in caught]
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    text = stream.getvalue().strip()
    log_lines = [line for line in text.splitlines() if line.strip()]
    return result, caught_warning_messages + log_lines


def _run_single_case(model, tokenizer, requested_tokens: int, context_capacity: int, max_new_tokens: int):
    import torch
    from transformers import GenerationConfig

    case = {
        "requested_tokens": requested_tokens,
        "actual_input_tokens": None,
        "tokenization_succeeded": False,
        "truncation_detected": False,
        "generation_called": False,
        "generation_succeeded": False,
        "exception_type": None,
        "exception_message": None,
        "warning_messages": [],
        "output_token_count": None,
        "peak_cuda_memory": None,
        "rendered_prompt_hash": None,
        "stopped_after_case": False,
    }

    try:
        messages, actual_tokens = _make_messages(tokenizer, requested_tokens)
        rendered = tokenizer.apply_chat_template(
            [messages],
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )
        case["tokenization_succeeded"] = True
        case["actual_input_tokens"] = int(rendered["input_ids"].shape[1])
        case["truncation_detected"] = False
        case["rendered_prompt_hash"] = hashlib.sha256(
            rendered["input_ids"].cpu().numpy().tobytes()
        ).hexdigest()
    except Exception as exc:
        case["exception_type"] = type(exc).__name__
        case["exception_message"] = str(exc)
        return case, False

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generation_config.weaver_do_sample = False
    generation_config.trigger_do_sample = False

    def _call_generate():
        import torch

        inputs = {key: value.to(model.device) for key, value in rendered.items()}
        case["generation_called"] = True
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=generation_config,
            latent_memory_bank=None,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            case["peak_cuda_memory"] = int(torch.cuda.max_memory_allocated())
        return output

    try:
        output, warning_messages = _collect_warnings(_call_generate)
        case["warning_messages"] = warning_messages
        case["generation_succeeded"] = True
        case["output_token_count"] = int(output.shape[1] - case["actual_input_tokens"])
        return case, False
    except Exception as exc:
        case["warning_messages"] = getattr(exc, "_captured_warning_messages", []) or case["warning_messages"]
        case["exception_type"] = type(exc).__name__
        case["exception_message"] = str(exc)
        if torch.cuda.is_available():
            case["peak_cuda_memory"] = int(torch.cuda.max_memory_allocated())
        stop = case["actual_input_tokens"] is not None and case["actual_input_tokens"] > context_capacity
        case["stopped_after_case"] = stop
        return case, stop


def _detective_preflight(dataset_root: str) -> dict:
    import hashlib
    _ensure_mabench_packages()
    import nltk
    import pyarrow.parquet as pq
    import tiktoken

    sys.path.insert(0, str(MAB_REPO.resolve()))
    from utils.templates import get_template

    parquet = Path(dataset_root) / "data/Long_Range_Understanding-00000-of-00001.parquet"
    rows = pq.read_table(parquet).to_pylist()
    row = next(item for item in rows if item.get("metadata", {}).get("source") == "detective_qa")
    context = row["context"]
    question = row["questions"][0]
    memorize_template = get_template("detective_qa", "memorize", "Long_context_agent")
    query_template = get_template("detective_qa", "query", "Long_context_agent")
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    sentences = nltk.sent_tokenize(context)
    chunks = []
    current = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence, allowed_special={"<|endoftext|>"}))
        if current and current_tokens + sentence_tokens > 4096:
            chunks.append(" ".join(current))
            current = [sentence]
            current_tokens = sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens
    if current:
        chunks.append(" ".join(current))

    prompts = [
        memorize_template.format(context=chunk, time_stamp="2026-06-20 00:00:00")
        for chunk in chunks
    ]
    query_prompt = query_template.format(question=question)
    parts = ["<system> You are a helpful assistant that can read the context and memorize it for future retrieval."]
    for prompt in prompts:
        parts.append(f"<user> {prompt}")
        parts.append("<assistant> Acknowledged.")
    parts.append(f"<user> {query_prompt}")
    parts.append("<assistant>")
    estimated_tokens = len(encoding.encode("\n".join(parts)))

    return {
        "split": "Long_Range_Understanding",
        "sub_dataset": "detective_qa",
        "selected_context_id": f"lru-{hashlib.sha256(context.encode('utf-8')).hexdigest()[:16]}",
        "estimated_full_history_query_tokens": estimated_tokens,
        "context_capacity": 32768,
        "marked_over_capacity": estimated_tokens > 32768,
        "generation_called": False,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_QWEN_PATH)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--model-checkpoint-id", default=DEFAULT_CHECKPOINT_ID)
    parser.add_argument("--cfg-path", default=DEFAULT_CFG_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-root", default=DEFAULT_MAB_DATASET_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--test-lengths", nargs="+", type=int, default=DEFAULT_TEST_LENGTHS)
    return parser


def main():
    import torch
    from main import set_seed
    from memgen.model import MemGenModel

    args = build_parser().parse_args()
    started_at = _utc_now()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-over-context"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    git_status = _git("status", "--short", "--branch")
    set_seed(args.seed, use_gpu=True)
    preliminary = _build_config(args, 32768)
    model = MemGenModel.from_config(preliminary["model"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the over-context diagnostic")
    model = model.to(device=torch.device("cuda"), dtype=torch.bfloat16)
    model.eval()
    context_capacity = int(getattr(model.reasoner.config, "max_position_embeddings", 0))
    if context_capacity <= 0:
        raise RuntimeError("Could not determine reasoner context capacity")
    tokenizer = model.tokenizer

    results = {
        "run_id": run_id,
        "checkpoint_path": args.checkpoint_path,
        "model_checkpoint_id": args.model_checkpoint_id,
        "context_capacity": context_capacity,
        "test_cases": [],
        "source_inspection_summary": {
            "full_chat_history_rendered_in": "interactions/multiturn_interaction.py::_build_chat_history + apply_chat_template in run_agent_loop",
            "apply_chat_template_receives_full_history": True,
            "tokenization_uses_truncation_true": False,
            "explicit_full_history_truncation_helper": False,
            "observation_truncation_exists": True,
            "model_generate_checks_capacity_explicitly": False,
            "model_generate_checks_input_plus_new_tokens": False,
            "latent_injection_changes_effective_length": True,
            "current_mab_runners_preflight_capacity": True,
        },
        "detective_qa_preflight": _detective_preflight(args.dataset_root),
        "final_recommendation": None,
        "git_status_before_after": git_status,
        "started_at": started_at,
        "finished_at": None,
    }

    stop_after = False
    for requested in args.test_lengths:
        if stop_after:
            break
        case, stop_after = _run_single_case(
            model=model,
            tokenizer=tokenizer,
            requested_tokens=requested,
            context_capacity=context_capacity,
            max_new_tokens=args.max_new_tokens,
        )
        results["test_cases"].append(case)

    over_capacity_failed = any(
        (case["actual_input_tokens"] or 0) > context_capacity and case["exception_type"]
        for case in results["test_cases"]
    )
    over_capacity_succeeded = any(
        (case["actual_input_tokens"] or 0) > context_capacity and case["generation_succeeded"]
        for case in results["test_cases"]
    )
    if over_capacity_failed:
        final = (
            "Over-capacity prompts should be marked invalid before generation. "
            "Do not use silent or runtime-failed over-context samples as the full-history baseline."
        )
    elif over_capacity_succeeded:
        final = (
            "Over-capacity generation continued without an explicit guard. Treat this path as unsupported "
            "for benchmark baselines until an explicit preflight guard is enforced."
        )
    else:
        final = (
            "No over-capacity call was completed safely in this diagnostic. Add an explicit preflight guard "
            "before any full-history MAB sample near or over capacity."
        )
    results["final_recommendation"] = final
    results["finished_at"] = _utc_now()
    _write_json(output_dir / "over_context_diagnostic.json", results)
    print(json.dumps({"output_dir": str(output_dir), "result_path": str(output_dir / 'over_context_diagnostic.json')}, ensure_ascii=False))


if __name__ == "__main__":
    main()
