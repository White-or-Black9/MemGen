import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


PHASE = "R4-1D"
DEFAULT_RETRIEVAL_ENDPOINT = "http://127.0.0.1:8001/retrieve"
DEFAULT_RETRIEVAL_TOPK = 3
CANNOT_FIND_PAGES = "Cannot find corresponding pages."


@dataclass(frozen=True)
class ParsedAnswer:
    answer: Optional[str]
    parser_success: bool
    parser_mode: str


@dataclass
class RetrievalAccounting:
    endpoint: str = DEFAULT_RETRIEVAL_ENDPOINT
    topk: int = DEFAULT_RETRIEVAL_TOPK
    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    saw_cannot_find_pages: bool = False
    failures: List[Dict[str, str]] = field(default_factory=list)

    def record_success(self) -> None:
        self.call_count += 1
        self.success_count += 1

    def record_failure(self, error: BaseException) -> None:
        self.call_count += 1
        self.failure_count += 1
        self.failures.append(
            {
                "type": type(error).__name__,
                "message": str(error),
            }
        )

    def observe(self, observation: Optional[str]) -> None:
        if observation and CANNOT_FIND_PAGES in observation:
            self.saw_cannot_find_pages = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "topk": self.topk,
            "call_count": self.call_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "saw_cannot_find_pages": self.saw_cannot_find_pages,
            "failures": list(self.failures),
        }


@dataclass(frozen=True)
class RunMetadata:
    batch_size: int
    memory_mode: str
    memory_enabled: bool
    checkpoint_path: str
    config_overrides: List[str]
    memory_threshold: Optional[float]
    memory_top_k: Optional[int]
    temperature: float
    max_response_length: int
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AccountingRetriever:
    def __init__(self, retriever: Any, accounting: RetrievalAccounting):
        self.retriever = retriever
        self.accounting = accounting
        if hasattr(self.retriever, "config"):
            self.retriever.config["search_url"] = accounting.endpoint
            self.retriever.config["topk"] = accounting.topk

    def batch_search(self, queries: List[str]) -> List[str]:
        try:
            result = self.retriever.batch_search(queries)
        except Exception as error:
            self.accounting.record_failure(error)
            raise
        self.accounting.record_success()
        return result


def parse_strict_answer(response: Optional[str]) -> ParsedAnswer:
    if not response:
        return ParsedAnswer(None, False, "none")
    matches = re.findall(
        r"<answer>(.*?)</answer>",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if not matches:
        return ParsedAnswer(None, False, "none")
    return ParsedAnswer(matches[-1].strip(), True, "strict_answer_tag")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="R4-1D single-sample TriviaQA dynamic structured harness."
    )
    parser.add_argument("--cfg-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--memory-mode",
        choices=("disabled", "version_a_aligned"),
        default="disabled",
    )
    parser.add_argument("--require-retrieval-ok", action="store_true")
    parser.add_argument(
        "--retrieval-endpoint",
        default=DEFAULT_RETRIEVAL_ENDPOINT,
    )
    parser.add_argument("--retrieval-topk", type=int, default=DEFAULT_RETRIEVAL_TOPK)
    parser.add_argument("--memory-threshold", type=float, default=0.7)
    parser.add_argument("--memory-top-k", type=int, default=1)
    parser.add_argument("--max-response-length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size != 1:
        raise ValueError("R4-1D harness requires batch_size=1")
    if args.sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if args.sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    if args.max_response_length <= 0:
        raise ValueError("max_response_length must be positive")
    if args.retrieval_topk <= 0:
        raise ValueError("retrieval_topk must be positive")
    if args.memory_top_k <= 0:
        raise ValueError("memory_top_k must be positive")
    if args.memory_threshold < 0:
        raise ValueError("memory_threshold must be non-negative")
    if args.memory_mode not in {"disabled", "version_a_aligned"}:
        raise ValueError(f"Unsupported memory_mode: {args.memory_mode}")


def build_config_overrides(args: argparse.Namespace) -> List[str]:
    return [
        "model.load_model_path",
        str(args.checkpoint_path),
        "run.mode",
        "evaluate",
        "run.seed",
        str(args.seed),
        "run.interaction.batch_size",
        "1",
        "run.interaction.temperature",
        str(args.temperature),
        "run.interaction.max_response_length",
        str(args.max_response_length),
        "run.interaction.weaver_do_sample",
        "False",
        "run.interaction.trigger_do_sample",
        "False",
    ]


def build_memory_bank_config(args: argparse.Namespace) -> Dict[str, Any]:
    if args.memory_mode == "disabled":
        return {
            "enabled": False,
            "batch_size": 1,
        }
    if args.memory_mode == "version_a_aligned":
        return {
            "enabled": True,
            "batch_size": 1,
            "max_slots": 8,
            "top_k": int(args.memory_top_k),
            "threshold": float(args.memory_threshold),
            "decay_alpha": 0.05,
            "pool_last_n": 64,
            "retrieve_policy": "threshold_topk",
            "update_policy": "thread_update",
            "storage_device": "cpu",
            "debug": True,
        }
    raise ValueError(f"Unsupported memory_mode: {args.memory_mode}")


def build_run_metadata(args: argparse.Namespace) -> RunMetadata:
    memory_enabled = args.memory_mode == "version_a_aligned"
    return RunMetadata(
        batch_size=args.batch_size,
        memory_mode=args.memory_mode,
        memory_enabled=memory_enabled,
        checkpoint_path=str(args.checkpoint_path),
        config_overrides=build_config_overrides(args),
        memory_threshold=(float(args.memory_threshold) if memory_enabled else None),
        memory_top_k=(int(args.memory_top_k) if memory_enabled else None),
        temperature=args.temperature,
        max_response_length=args.max_response_length,
        seed=args.seed,
    )


def build_sample_record(
    *,
    sample: Dict[str, Any],
    sample_index: int,
    sample_id: str,
    conversation: List[Dict[str, str]],
    final_response: Optional[str],
    parsed: ParsedAnswer,
    reward: Optional[float],
    retrieval: RetrievalAccounting,
    run: RunMetadata,
    memory_bank_debug: Optional[Dict[str, Any]],
    valid_run: bool,
    invalid_reason: Optional[str],
) -> Dict[str, Any]:
    return {
        "phase": PHASE,
        "sample_index": sample_index,
        "sample_id": sample_id,
        "question": sample.get("prompt") or sample.get("question"),
        "gold_answers": list(sample.get("answer") or []),
        "conversation": conversation,
        "final_response": final_response,
        "parsed_answer": parsed.answer,
        "parser_success": parsed.parser_success,
        "parser_mode": parsed.parser_mode,
        "reward": reward,
        "retrieval": retrieval.to_dict(),
        "run": run.to_dict(),
        "memory_bank_debug": memory_bank_debug,
        "valid_run": valid_run,
        "invalid_reason": invalid_reason,
    }


def build_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_count = sum(bool(record["valid_run"]) for record in records)
    invalid_count = len(records) - valid_count
    retrieval_blocked_count = sum(
        bool(record.get("invalid_reason"))
        and "retrieval" in str(record.get("invalid_reason"))
        for record in records
    )
    return {
        "summary": {
            "sample_count": len(records),
            "valid_run_count": valid_count,
            "invalid_run_count": invalid_count,
            "retrieval_blocked_count": retrieval_blocked_count,
        }
    }


def conversation_to_text(record: Dict[str, Any]) -> str:
    lines = [
        f"Sample: {record['sample_id']}",
        f"Question: {record['question']}",
    ]
    for turn in record["conversation"]:
        lines.append(f"{turn.get('role', 'unknown')}: {turn.get('content', '')}")
    lines.append(f"Reward: {record['reward']}")
    lines.append(f"Valid run: {record['valid_run']}")
    lines.append(f"Invalid reason: {record['invalid_reason']}")
    return "\n".join(lines)


def write_artifacts(
    output_dir: Path,
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
    run_config: Dict[str, Any],
) -> None:
    evaluate_dir = output_dir / "evaluate"
    evaluate_dir.mkdir(parents=True, exist_ok=True)
    with (evaluate_dir / "answer.json").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    (evaluate_dir / "conversations.txt").write_text(
        "\n\n".join(conversation_to_text(record) for record in records) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
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
                    "sample_id": record["sample_id"],
                    "memory_bank_debug": record["memory_bank_debug"],
                }
                for record in records
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _sample_from_raw(raw_sample: Dict[str, Any]) -> Dict[str, Any]:
    answer = raw_sample.get("answer", {})
    if isinstance(answer, dict):
        aliases = answer.get("normalized_aliases") or answer.get("aliases") or []
    else:
        aliases = answer or []
    return {
        "prompt": raw_sample.get("question") or raw_sample.get("prompt"),
        "answer": list(aliases),
    }


def _load_sample(sample_index: int) -> Dict[str, Any]:
    from datasets import load_dataset

    dataset = load_dataset(
        "mandarjoshi/trivia_qa",
        "rc.wikipedia.nocontext",
        split="validation",
    )
    if sample_index >= len(dataset):
        raise ValueError(
            f"sample_index {sample_index} out of range for validation split"
        )
    return _sample_from_raw(dataset[sample_index])


def _build_config(args: argparse.Namespace) -> Dict[str, Any]:
    from common.config import Config

    config_args = SimpleNamespace(
        cfg_path=args.cfg_path,
        options=build_config_overrides(args),
    )
    config = Config(config_args)
    config_dict = config.to_dict()
    config_dict["latent_memory_bank"] = build_memory_bank_config(args)
    return config_dict


def _build_interaction_config(config_dict: Dict[str, Any]):
    from interactions.base_interaction import InteractionConfig

    interaction = config_dict["run"].get("interaction", {})
    return InteractionConfig(
        max_turns=interaction.get("max_turns", 5),
        max_start_length=interaction.get("max_start_length", 1024),
        max_prompt_length=interaction.get("max_prompt_length", 4096),
        max_response_length=interaction.get("max_response_length", 1024),
        max_obs_length=interaction.get("max_obs_length", 512),
        temperature=interaction.get("temperature", 0.0),
        batch_size=interaction.get("batch_size", 1),
        output_dir=None,
        weaver_do_sample=interaction.get("weaver_do_sample", False),
        trigger_do_sample=interaction.get("trigger_do_sample", False),
        latent_memory_bank=config_dict.get("latent_memory_bank"),
    )


def _build_data_proto(system_prompt: str, user_prompt: str, env: Any):
    from interactions.base_interaction import InteractionDataProto

    data_proto = InteractionDataProto()
    data_proto.no_tensor_batch["init_prompts"] = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    ]
    data_proto.no_tensor_batch["envs"] = [env]
    return data_proto


def _final_assistant_response(conversation: List[Dict[str, str]]) -> Optional[str]:
    for message in reversed(conversation):
        if message.get("role") == "assistant":
            return message.get("content")
    return None


def _invalid_reason(
    *,
    errors: List[str],
    retrieval: RetrievalAccounting,
    require_retrieval_ok: bool,
) -> Optional[str]:
    if errors:
        return "crashed_run"
    if retrieval.failure_count or retrieval.saw_cannot_find_pages:
        return "retrieval_endpoint_unavailable"
    if require_retrieval_ok and retrieval.call_count == 0:
        return "retrieval_not_exercised"
    return None


def run_single_sample(
    args: argparse.Namespace,
    *,
    config_dict: Dict[str, Any],
    model: Any,
    interaction_manager: Any,
) -> Dict[str, Any]:
    from data.triviaqa.env import TriviaQAEnv
    import torch

    sample = _load_sample(args.sample_index)
    env = TriviaQAEnv(config_dict["dataset"])
    system_prompt, user_prompt = env.set_env(sample)
    retrieval = RetrievalAccounting(
        endpoint=args.retrieval_endpoint,
        topk=args.retrieval_topk,
    )
    env.explorer = AccountingRetriever(env.explorer, retrieval)
    data_proto = _build_data_proto(system_prompt, user_prompt, env)

    errors: List[str] = []
    start = time.perf_counter()
    try:
        outputs = interaction_manager.run_agent_loop(data_proto)
        inter_history = outputs.no_tensor_batch["inter_histories"][0]
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
        inter_history = []
    latency = time.perf_counter() - start

    conversation = data_proto.no_tensor_batch["init_prompts"][0] + inter_history
    for message in conversation:
        if message.get("role") == "user":
            retrieval.observe(message.get("content"))
    final_response = _final_assistant_response(conversation)
    parsed = parse_strict_answer(final_response)
    invalid_reason = _invalid_reason(
        errors=errors,
        retrieval=retrieval,
        require_retrieval_ok=args.require_retrieval_ok,
    )
    memory_bank_debug = getattr(
        interaction_manager,
        "latest_memory_bank_debug",
        None,
    )
    record = build_sample_record(
        sample=sample,
        sample_index=args.sample_index,
        sample_id=f"triviaqa-validation-{args.sample_index}",
        conversation=conversation,
        final_response=final_response,
        parsed=parsed,
        reward=None if errors else float(env.feedback()),
        retrieval=retrieval,
        run=build_run_metadata(args),
        memory_bank_debug=memory_bank_debug,
        valid_run=invalid_reason is None,
        invalid_reason=invalid_reason,
    )
    record["latency"] = latency
    record["errors"] = errors
    return record


def build_preflight_record(args: argparse.Namespace) -> Dict[str, Any]:
    retrieval = RetrievalAccounting(
        endpoint=args.retrieval_endpoint,
        topk=args.retrieval_topk,
    )
    parsed = parse_strict_answer(None)
    return build_sample_record(
        sample={
            "prompt": None,
            "answer": [],
        },
        sample_index=args.sample_index,
        sample_id=f"triviaqa-validation-{args.sample_index}",
        conversation=[],
        final_response=None,
        parsed=parsed,
        reward=None,
        retrieval=retrieval,
        run=build_run_metadata(args),
        memory_bank_debug=None,
        valid_run=False,
        invalid_reason="preflight_only",
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from interactions.multiturn_interaction import MultiTurnInteractionManager
    from main import set_seed
    from memgen.model import MemGenModel
    import torch

    config_dict = _build_config(args)
    set_seed(args.seed, use_gpu=torch.cuda.is_available())
    model = MemGenModel.from_config(config_dict["model"])
    if torch.cuda.is_available():
        model = model.to(device=torch.device("cuda"), dtype=torch.bfloat16)
    else:
        model = model.to(torch.bfloat16)
    model.eval()
    interaction_manager = MultiTurnInteractionManager(
        model.tokenizer,
        model,
        _build_interaction_config(config_dict),
    )
    interaction_manager.actor_rollout_wg = model

    if args.preflight_only or args.dry_run:
        records = [build_preflight_record(args)]
    else:
        records = []
        for sample_index in range(args.sample_index, args.sample_index + args.sample_count):
            sample_args = argparse.Namespace(**vars(args))
            sample_args.sample_index = sample_index
            records.append(
                run_single_sample(
                    sample_args,
                    config_dict=config_dict,
                    model=model,
                    interaction_manager=interaction_manager,
                )
            )

    summary = build_summary(records)
    run_config = {
        "phase": PHASE,
        "command": " ".join(sys.argv),
        "cfg_path": args.cfg_path,
        "sample_index": args.sample_index,
        "sample_count": args.sample_count,
        "memory_mode": args.memory_mode,
        "memory_threshold": args.memory_threshold,
        "memory_top_k": args.memory_top_k,
        "require_retrieval_ok": args.require_retrieval_ok,
        "retrieval_endpoint": args.retrieval_endpoint,
        "retrieval_topk": args.retrieval_topk,
        "dry_run": args.dry_run,
        "preflight_only": args.preflight_only,
        "version_b": False,
        "fallback_top1": False,
    }
    write_artifacts(output_dir, records, summary, run_config)
    return 0 if records[0]["valid_run"] or args.preflight_only or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
