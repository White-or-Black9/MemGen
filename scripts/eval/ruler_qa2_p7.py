"""Dry-run skeleton for a future RULER-QA2 P7 runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCHEMA_VERSION = "ruler-qa2-dryrun/v1"
SUPPORTED_METHODS = ("disabled", "p7", "p7_no_query_retrieval")
SUPPORTED_RUN_MODES = (
    "stub",
    "predictions",
    "disabled_query_only",
    "adapted_p7_query_only",
)


class RulerQA2RunContractError(ValueError):
    """Raised when aligned RULER-QA2 mode comparison is invalid."""


def normalize_text(value: str) -> str:
    return " ".join(value.lower().strip().split())


def substring_exact_match(prediction: str, gold_answers: list[str]) -> bool:
    normalized_prediction = normalize_text(prediction)
    return any(normalize_text(answer) in normalized_prediction for answer in gold_answers)


def expected_method_set(methods_spec: str) -> list[str]:
    methods = [item.strip() for item in methods_spec.split(",") if item.strip()]
    if not methods:
        raise RulerQA2RunContractError("methods cannot be empty")
    seen = set()
    ordered = []
    for method in methods:
        if method not in SUPPORTED_METHODS:
            raise RulerQA2RunContractError(f"unknown method: {method}")
        if method in seen:
            raise RulerQA2RunContractError(f"duplicate method: {method}")
        seen.add(method)
        ordered.append(method)
    return ordered


def validate_prepared_payload(
    payload: dict, *, expected_sub_dataset: str
) -> dict[str, int | str]:
    dataset_config = payload.get("dataset_config") or {}
    sub_dataset = dataset_config.get("sub_dataset")
    if sub_dataset != expected_sub_dataset:
        raise RulerQA2RunContractError(
            f"sub_dataset mismatch: {sub_dataset} != {expected_sub_dataset}"
        )
    queries = list(payload.get("queries") or [])
    question_count = int(payload.get("question_count", -1))
    if question_count != len(queries):
        raise RulerQA2RunContractError(
            f"question_count mismatch: {question_count} != {len(queries)}"
        )
    chunks = list(payload.get("chunks") or [])
    prompts = list(payload.get("memorization_prompts") or [])
    if len(chunks) != len(prompts):
        raise RulerQA2RunContractError(
            f"chunk/prompt mismatch: {len(chunks)} != {len(prompts)}"
        )
    for expected_query_id, query in enumerate(queries):
        if query.get("query_id") != expected_query_id:
            raise RulerQA2RunContractError(
                f"query_id mismatch at position {expected_query_id}: {query.get('query_id')}"
            )
        if not query.get("query_prompt"):
            raise RulerQA2RunContractError(f"missing query_prompt at {expected_query_id}")
        gold_answers = query.get("gold_answers") or []
        if not gold_answers:
            raise RulerQA2RunContractError(
                f"missing gold_answers at {expected_query_id}"
            )
    return {
        "context_id": str(payload["context_id"]),
        "sub_dataset": str(sub_dataset),
        "question_count": question_count,
        "chunk_count": len(chunks),
    }


def load_prepared_payload(
    path: str | Path, *, expected_sub_dataset: str
) -> tuple[dict[str, Any], dict[str, int | str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    summary = validate_prepared_payload(
        payload, expected_sub_dataset=expected_sub_dataset
    )
    return payload, summary


def slice_prepared_payload(payload: dict[str, Any], *, max_queries: int | None) -> dict[str, Any]:
    if max_queries is None:
        return payload
    sliced = dict(payload)
    sliced_queries = list(payload["queries"])[: int(max_queries)]
    sliced["queries"] = sliced_queries
    sliced["question_count"] = len(sliced_queries)
    return sliced


def validate_same_queries(reference: list[dict], candidate: list[dict]) -> None:
    reference_ids = [item["query_id"] for item in reference]
    candidate_ids = [item["query_id"] for item in candidate]
    if reference_ids != candidate_ids:
        raise RulerQA2RunContractError(
            f"query identity mismatch: {reference_ids} != {candidate_ids}"
        )


def build_query_jobs(payload: dict[str, Any], *, method: str) -> list[dict[str, Any]]:
    return [
        {
            "method": method,
            "context_id": payload["context_id"],
            "query_id": query["query_id"],
            "question": query.get("question"),
            "query_prompt": query.get("query_prompt"),
            "gold_answers": list(query.get("gold_answers") or []),
        }
        for query in payload["queries"]
    ]


def build_single_query_payload(payload: dict[str, Any], *, query_id: int) -> dict[str, Any]:
    query = payload["queries"][query_id]
    return {
        "context_id": payload["context_id"],
        "dataset_config": payload.get("dataset_config"),
        "chunks": list(payload.get("chunks") or []),
        "chunk_token_lengths": list(payload.get("chunk_token_lengths") or []),
        "memorization_prompts": list(payload.get("memorization_prompts") or []),
        "query_id": query["query_id"],
        "question": query.get("question"),
        "query_prompt": query.get("query_prompt"),
        "gold_answers": list(query.get("gold_answers") or []),
    }


def build_ruler_context_payload(prepared_payload: dict[str, Any]) -> dict[str, Any]:
    queries = list(prepared_payload.get("queries") or [])
    return {
        "dataset_config": prepared_payload.get("dataset_config"),
        "context_id": prepared_payload["context_id"],
        "context_index": 0,
        "chunks": list(prepared_payload.get("chunks") or []),
        "chunk_token_lengths": list(prepared_payload.get("chunk_token_lengths") or []),
        "memorization_prompts": list(prepared_payload.get("memorization_prompts") or []),
        "questions": [query.get("question") for query in queries],
        "answers": [list(query.get("gold_answers") or []) for query in queries],
        "question_ids": [query.get("query_id") for query in queries],
        "question_types": [None] * len(queries),
        "qa_pair_ids": [query.get("query_id") for query in queries],
        "previous_events": [[] for _ in queries],
        "queries": queries,
        "question_count": len(queries),
    }


def build_ruler_question_payload(
    context_payload: dict[str, Any], question_index: int
) -> dict[str, Any]:
    query = context_payload["queries"][question_index]
    return {
        "dataset_config": context_payload["dataset_config"],
        "context_id": context_payload["context_id"],
        "context_index": context_payload["context_index"],
        "query_id": query["query_id"],
        "question_id": context_payload["question_ids"][question_index],
        "question_type": context_payload["question_types"][question_index],
        "qa_pair_id": context_payload["qa_pair_ids"][question_index],
        "previous_events": context_payload["previous_events"][question_index],
        "chunks": context_payload["chunks"],
        "chunk_token_lengths": context_payload["chunk_token_lengths"],
        "memorization_prompts": context_payload["memorization_prompts"],
        "query_prompt": query["query_prompt"],
        "question": context_payload["questions"][question_index],
        "gold_answers": list(context_payload["answers"][question_index]),
    }


def execute_query_jobs(
    jobs: list[dict[str, Any]], *, executor
) -> list[dict[str, Any]]:
    predictions = []
    for job in jobs:
        result = executor(job)
        predictions.append(
            {
                "query_id": result["query_id"],
                "prediction": str(result.get("prediction", "")),
            }
        )
    validate_same_queries(jobs, predictions)
    return predictions


def run_disabled_query_predictions(
    payload: dict[str, Any], *, query_runner
) -> list[dict[str, Any]]:
    predictions = []
    for query_index, _query in enumerate(payload["queries"]):
        single_payload = build_single_query_payload(payload, query_id=query_index)
        result = query_runner(single_payload)
        predictions.append(
            {
                "query_id": single_payload["query_id"],
                "prediction": str(result.get("prediction", "")),
            }
        )
    expected_jobs = build_query_jobs(payload, method="disabled")
    validate_same_queries(expected_jobs, predictions)
    return predictions


def disabled_expected_turns(payload: dict[str, Any]) -> int:
    return len(payload.get("chunks") or []) + 1


def _dynamic_interaction_config(config_dict: dict[str, Any], context_capacity: int, *, max_turns: int):
    from interactions.base_interaction import InteractionConfig

    interaction = config_dict["run"]["interaction"]
    return InteractionConfig(
        max_turns=max_turns,
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


def run_mab2_disabled_query_model(args, payload: dict[str, Any]) -> dict[str, Any]:
    import torch
    from interactions.base_interaction import InteractionDataProto
    from main import set_seed
    from memgen.model import MemGenModel
    from scripts.eval import mab2_bank_off

    set_seed(args.seed, use_gpu=True)
    preliminary = mab2_bank_off._build_config(args, 32768)
    model = MemGenModel.from_config(preliminary["model"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the RULER-QA2 disabled query run")
    model = model.to(device=torch.device("cuda"), dtype=torch.bfloat16)
    model.eval()
    capacity = int(getattr(model.reasoner.config, "max_position_embeddings", 0))
    if capacity <= 0:
        raise RuntimeError("Could not determine Reasoner context capacity")
    config_dict = mab2_bank_off._build_config(args, capacity)

    generation_trace = {
        "generations": [],
        "weaver_prompt_calls": 0,
        "weaver_inference_calls": 0,
    }
    prompt_trace = []
    session_trace = {"session_count": 0, "bank_created": False}
    mab2_bank_off._install_generation_trace(model, generation_trace)
    manager_cls = mab2_bank_off._manager_class(
        payload["chunks"], payload["query_prompt"], capacity, prompt_trace, session_trace
    )
    manager = manager_cls(
        model.tokenizer,
        model,
        _dynamic_interaction_config(
            config_dict, capacity, max_turns=disabled_expected_turns(payload)
        ),
    )
    env = mab2_bank_off.MABEpisodeEnv(
        payload["memorization_prompts"][1:] + [payload["query_prompt"]],
        expected_turns=disabled_expected_turns(payload),
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
        raise RuntimeError("One RULER-QA2 context was not mapped to exactly one session")
    if env.final_answer is None:
        raise RuntimeError("Final answer could not be separated from acknowledgements")
    if len(prompt_trace) != disabled_expected_turns(payload):
        raise RuntimeError("Unexpected number of rendered turn prompts")
    mab2_bank_off.assert_bank_off_invariants(
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


def make_mab2_disabled_query_runner(args, *, model_runner=None):
    if model_runner is None:
        model_runner = run_mab2_disabled_query_model

    def runner(single_payload: dict[str, Any]) -> dict[str, Any]:
        return model_runner(args, single_payload)

    return runner


def build_mab2_disabled_args(
    *, seed: int, model_path: str, checkpoint_path: str, cfg_path: str
):
    return SimpleNamespace(
        seed=seed,
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        cfg_path=cfg_path,
    )


def make_mode_stub_records(
    payload: dict[str, Any], *, methods: list[str]
) -> dict[str, list[dict[str, Any]]]:
    records = {}
    for method in methods:
        method_records = []
        for query in payload["queries"]:
            method_records.append(
                {
                    "method": method,
                    "context_id": payload["context_id"],
                    "query_id": query["query_id"],
                    "question": query.get("question"),
                    "gold_answers": list(query.get("gold_answers") or []),
                    "status": "not_run",
                }
            )
        records[method] = method_records
    reference = records[methods[0]]
    for method in methods[1:]:
        validate_same_queries(reference, records[method])
    return records


def score_predictions_for_method(
    payload: dict[str, Any], *, method: str, predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    expected_records = make_mode_stub_records(payload, methods=[method])[method]
    validate_same_queries(expected_records, predictions)
    scored = []
    for expected, predicted in zip(expected_records, predictions):
        prediction_text = str(predicted.get("prediction", ""))
        gold_answers = list(expected["gold_answers"])
        scored.append(
            {
                **expected,
                "prediction": prediction_text,
                "correct": substring_exact_match(prediction_text, gold_answers),
                "status": "scored",
            }
        )
    return scored


def build_dry_run_artifact(
    prepared_summary: dict[str, int | str], *, methods: list[str]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": prepared_summary["sub_dataset"],
        "context_id": prepared_summary["context_id"],
        "question_count": prepared_summary["question_count"],
        "chunk_count": prepared_summary["chunk_count"],
        "methods": list(methods),
    }


def build_failure_artifact(
    prepared_summary: dict[str, int | str], *, methods: list[str], error: Exception
) -> dict[str, Any]:
    return {
        **build_dry_run_artifact(prepared_summary, methods=methods),
        "status": "failed",
        "error": f"{type(error).__name__}: {error}",
    }


def run_adapted_p7_queries(
    prepared_payload: dict[str, Any],
    *,
    args,
    method: str,
    runtime_bank_config: dict[str, Any],
    recorded_bank_config: dict[str, Any],
    model_loader=None,
    eventqa_runner=None,
) -> dict[str, Any]:
    from scripts.eval import mab6b_weaver_space_bank_eventqa_65536_n5 as eventqa

    if method not in {"p7", "p7_no_query_retrieval"}:
        raise RulerQA2RunContractError(f"unsupported adapted p7 method: {method}")
    if model_loader is None:
        model_loader = eventqa.weaver_bank._load_model
    if eventqa_runner is None:
        eventqa_runner = eventqa._run_eventqa_model

    context_payload = build_ruler_context_payload(prepared_payload)
    model, capacity = model_loader(args)
    construction_result = eventqa_runner(
        args,
        model,
        capacity,
        eventqa._construction_only_payload(context_payload),
        "on",
        runtime_bank_config,
        preserve_bank=True,
        construction_only=True,
        recorded_bank_config=recorded_bank_config,
    )
    frozen_bank = construction_result.pop("_retained_bank")
    predictions = []
    query_diagnostics = []
    for question_index in range(context_payload["question_count"]):
        question_payload = build_ruler_question_payload(context_payload, question_index)
        query_payload = eventqa._query_only_payload(question_payload)
        result = eventqa_runner(
            args,
            model,
            capacity,
            query_payload,
            "on",
            runtime_bank_config,
            external_bank=frozen_bank,
            preserve_bank=True,
            disable_query_retrieval=(method == "p7_no_query_retrieval"),
            recorded_bank_config=recorded_bank_config,
        )
        predictions.append(
            {
                "query_id": question_payload["query_id"],
                "prediction": str(result.get("prediction", "")),
            }
        )
        query_diagnostics.append(
            {
                "query_id": question_payload["query_id"],
                "retrieved_latent_count": int(result.get("retrieved_latent_count", 0)),
                "query_write_count_delta": int(result.get("query_write_count_delta", 0)),
                "query_read_only_enforced": bool(result.get("query_read_only_enforced", False)),
                "bank_snapshot_changed_after_query": bool(
                    result.get("bank_snapshot_changed_after_query", False)
                ),
            }
        )
    records = score_predictions_for_method(
        {"context_id": prepared_payload["context_id"], "queries": prepared_payload["queries"]},
        method=method,
        predictions=predictions,
    )
    for record, diagnostic in zip(records, query_diagnostics):
        record.update(diagnostic)
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": prepared_payload["dataset_config"]["sub_dataset"],
        "context_id": prepared_payload["context_id"],
        "question_count": context_payload["question_count"],
        "chunk_count": len(context_payload["chunks"]),
        "methods": [method],
        "construction": {
            "final_slot_count": int(
                construction_result.get("pre_query_bank_summary", {}).get("slot_count", 0)
            )
        },
        "mode_records": {method: records},
    }


def build_adapted_p7_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint_path=args.checkpoint_path,
        model_path=args.model_path,
        cfg_path=args.cfg_path,
        seed=args.seed,
        generation_max_length=getattr(args, "generation_max_length", 40),
        retrieve_threshold=args.retrieve_threshold,
        update_threshold=args.update_threshold,
        max_slots=args.max_slots,
        top_k=args.top_k,
        decay_alpha=args.decay_alpha,
        eventqa_protocol="frozen_context_bank",
        strict_official_eventqa_prompt=False,
        first_line_official_eventqa_prompt=False,
        bank_transition_diagnostics=False,
        trace_score_decomposition=False,
        save_frozen_bank=False,
        reseed_per_context=False,
    )


def build_recorded_bank_config(runtime_bank_config: dict[str, Any]) -> dict[str, Any]:
    return {"latent_memory_bank_config": dict(runtime_bank_config)}


def build_runner_stub_artifact(
    payload: dict[str, Any], *, prepared_summary: dict[str, int | str], methods: list[str]
) -> dict[str, Any]:
    return {
        **build_dry_run_artifact(prepared_summary, methods=methods),
        "mode_records": make_mode_stub_records(payload, methods=methods),
    }


def build_scored_artifact(
    payload: dict[str, Any],
    *,
    prepared_summary: dict[str, int | str],
    method: str,
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        **build_dry_run_artifact(prepared_summary, methods=[method]),
        "mode_records": {
            method: score_predictions_for_method(
                payload, method=method, predictions=predictions
            )
        },
    }


def build_disabled_query_scored_artifact(
    payload: dict[str, Any],
    *,
    prepared_summary: dict[str, int | str],
    runner_args,
    model_runner=None,
) -> dict[str, Any]:
    query_runner = make_mab2_disabled_query_runner(
        runner_args, model_runner=model_runner
    )
    predictions = run_disabled_query_predictions(payload, query_runner=query_runner)
    return build_scored_artifact(
        payload,
        prepared_summary=prepared_summary,
        method="disabled",
        predictions=predictions,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-mode", choices=SUPPORTED_RUN_MODES, default="stub")
    parser.add_argument("--methods", default="disabled,p7,p7_no_query_retrieval")
    parser.add_argument("--expected-sub-dataset", default="ruler_qa2_421K")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--predictions-input")
    parser.add_argument("--prediction-method", default="disabled")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--cfg-path", default="configs/latent_memory/triviaqa.yaml")
    parser.add_argument("--generation-max-length", type=int, default=40)
    parser.add_argument("--retrieve-threshold", type=float, default=0.05)
    parser.add_argument("--update-threshold", type=float, default=0.10)
    parser.add_argument("--max-slots", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--decay-alpha", type=float, default=0.05)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    methods = expected_method_set(args.methods)
    payload, summary = load_prepared_payload(
        args.prepared_input, expected_sub_dataset=args.expected_sub_dataset
    )
    payload = slice_prepared_payload(payload, max_queries=args.max_queries)
    summary = validate_prepared_payload(
        payload, expected_sub_dataset=args.expected_sub_dataset
    )
    try:
        if args.run_mode == "predictions":
            if not args.predictions_input:
                raise RulerQA2RunContractError(
                    "--predictions-input is required for run-mode=predictions"
                )
            predictions = json.loads(Path(args.predictions_input).read_text(encoding="utf-8"))
            artifact = build_scored_artifact(
                payload,
                prepared_summary=summary,
                method=args.prediction_method,
                predictions=predictions,
            )
        elif args.run_mode == "disabled_query_only":
            if not args.checkpoint_path:
                raise RulerQA2RunContractError(
                    "--checkpoint-path is required for run-mode=disabled_query_only"
                )
            runner_args = build_mab2_disabled_args(
                seed=args.seed,
                model_path=args.model_path,
                checkpoint_path=args.checkpoint_path,
                cfg_path=args.cfg_path,
            )
            artifact = build_disabled_query_scored_artifact(
                payload,
                prepared_summary=summary,
                runner_args=runner_args,
            )
        elif args.run_mode == "adapted_p7_query_only":
            if not args.checkpoint_path:
                raise RulerQA2RunContractError(
                    "--checkpoint-path is required for run-mode=adapted_p7_query_only"
                )
            if args.prediction_method not in {"p7", "p7_no_query_retrieval"}:
                raise RulerQA2RunContractError(
                    "--prediction-method must be p7 or p7_no_query_retrieval "
                    "for run-mode=adapted_p7_query_only"
                )
            from scripts.eval import mab6b_weaver_space_bank_eventqa_65536_n5 as eventqa

            adapted_args = build_adapted_p7_args(args)
            runtime_bank_config = eventqa._eventqa_bank_config(adapted_args)
            recorded_bank_config = build_recorded_bank_config(runtime_bank_config)
            artifact = run_adapted_p7_queries(
                payload,
                args=adapted_args,
                method=args.prediction_method,
                runtime_bank_config=runtime_bank_config,
                recorded_bank_config=recorded_bank_config,
            )
        else:
            artifact = build_runner_stub_artifact(
                payload, prepared_summary=summary, methods=methods
            )
        exit_code = 0
    except Exception as error:
        artifact = build_failure_artifact(summary, methods=methods, error=error)
        exit_code = 1
    Path(args.output).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
