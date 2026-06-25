"""MAB-6A: detective_qa Version B Weaver-conditioned memory on 10 contexts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import mab3_bank_on_full_history as mab3
from scripts.eval import mab5a_detectiveqa_compressed_n10 as base
from scripts.eval import mab5c_decoupled_thresholds_detectiveqa_n10 as mab5c

_BASE_LOAD_MODEL = base._load_model
_BASE_RUN_MODEL = base._run_model


EXPERIMENT_NAME = "MAB-6A: detective_qa Version B Weaver-conditioned Memory n10"
RUN_PREFIX = "detectiveqa-version-b-weaver-conditioned-n10"
DEFAULT_OUTPUT_ROOT = "outputs/mab/version_b_weaver_conditioned_detectiveqa_n10"
DEFAULT_THRESHOLD = 0.03
DEFAULT_RETRIEVE_THRESHOLD = 0.03
DEFAULT_UPDATE_THRESHOLD = 0.05
DEFAULT_TOP_K = 1
DEFAULT_MAX_SLOTS = 8
DEFAULT_RETRIEVE_POLICY = "threshold_topk"
DEFAULT_UPDATE_POLICY = "thread_update"
DEFAULT_REQUESTED_CONTEXTS = 10
RESEARCH_NOTE_PATH = Path(
    "research_notes/benchmarks/memoryagentbench_mab6a_version_b_weaver_conditioning.md"
)
MAB5C_CANONICAL_BASELINE = (
    "outputs/mab/decoupled_thresholds_detectiveqa_n10/"
    "20260622T140741Z-detectiveqa-decoupled-thresholds-n10"
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _bank_config():
    config = mab3.version_a_bank_config(
        top_k=DEFAULT_TOP_K,
        threshold=DEFAULT_THRESHOLD,
        retrieve_policy=DEFAULT_RETRIEVE_POLICY,
    )
    config["retrieve_threshold"] = DEFAULT_RETRIEVE_THRESHOLD
    config["update_threshold"] = DEFAULT_UPDATE_THRESHOLD
    config["max_slots"] = DEFAULT_MAX_SLOTS
    config["top_k"] = DEFAULT_TOP_K
    config["retrieve_policy"] = DEFAULT_RETRIEVE_POLICY
    config["update_policy"] = DEFAULT_UPDATE_POLICY
    return config


def _build_manifest(run_id: str, args, started_at: str, *, git_status_before: str, git_status_after: str | None = None) -> dict:
    manifest = mab5c._build_manifest(
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
            "threshold": DEFAULT_THRESHOLD,
            "retrieve_threshold": DEFAULT_RETRIEVE_THRESHOLD,
            "update_threshold": DEFAULT_UPDATE_THRESHOLD,
            "top_k": DEFAULT_TOP_K,
            "max_slots": DEFAULT_MAX_SLOTS,
            "retrieve_policy": DEFAULT_RETRIEVE_POLICY,
            "update_policy": DEFAULT_UPDATE_POLICY,
            "query_mode": "first-query-only",
            "query_phase": "read-only",
            "full_history_policy": "over_capacity_invalid",
            "retrieved_memory_to_weaver": True,
            "comparison_baseline": MAB5C_CANONICAL_BASELINE,
            "research_note": str(RESEARCH_NOTE_PATH),
        }
    )
    return manifest


def _load_model(args):
    model, capacity = _BASE_LOAD_MODEL(args)
    model.config.retrieved_memory_to_weaver = True
    return model, capacity


def _run_model(args, model, capacity: int, payload: dict, bank_mode: str, bank_config: dict | None = None) -> dict:
    from interactions.base_interaction import InteractionDataProto

    previous_flag = bool(getattr(model.config, "retrieved_memory_to_weaver", False))
    model.config.retrieved_memory_to_weaver = bank_mode == "on"
    try:
        if bank_mode == "off":
            return _BASE_RUN_MODEL(args, model, capacity, payload, bank_mode, bank_config)

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
        restore_hooks = base._install_model_trace(model, model_trace)
        manager_cls = base._manager_class(
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

            bank = lifecycle["bank"]
            final_debug = lifecycle["final_debug_before_reset"]
            query_bank_proxy = lifecycle.get("query_bank_proxy")
            query_write_attempt_count = getattr(query_bank_proxy, "write_attempt_count", 0) if query_bank_proxy else 0
            query_write_count = 0
            bank_reset_after_context = lifecycle["post_reset_slot_count"] == 0
            cross_context_leakage_detected = bool(
                lifecycle.get("initial_slot_count", 0) != 0 or not bank_reset_after_context
            )
            generations = model_trace["generations"]
            if not generations:
                raise RuntimeError("No generation trace recorded")
            return {
                "prediction": env.final_answer,
                "prompt_trace": prompt_trace,
                "generations": generations,
                "context_capacity": capacity,
                "bank_write_count": final_debug["memory_write_count"],
                "bank_retrieval_count": final_debug["memory_retrieve_count"],
                "bank_retrieved_latent_count": final_debug["retrieved_latent_count"],
                "retrieved_indices_by_turn": [list(gen["retrieved_indices"]) for gen in generations],
                "retrieved_scores_by_turn": [list(gen["retrieved_scores"]) for gen in generations],
                "bank_slot_count_final_before_reset": final_debug["slot_count"],
                "bank_reset_after_context": bank_reset_after_context,
                "cross_context_leakage_detected": cross_context_leakage_detected,
                "query_write_count": query_write_count,
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
    row = mab5c._build_row(
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
    final_generation = bank_on_result["generations"][-1]
    row.update(
        {
            "retrieved_memory_to_weaver": bool(final_generation.get("retrieved_memory_to_weaver")),
            "raw_retrieved_latents_enter_reasoner": bool(
                final_generation.get("raw_retrieved_latents_enter_reasoner")
            ),
            "weaver_conditioned_on_retrieved_memory": bool(
                final_generation.get("weaver_conditioned_on_retrieved_memory")
            ),
            "weaver_conditioning_token_count": int(
                final_generation.get("weaver_conditioning_token_count", 0)
            ),
            "fused_latent_generated": bool(final_generation.get("fused_latent_generated")),
        }
    )
    return row


def _aggregate(rows: list[dict]) -> dict:
    summary = mab5c._aggregate(rows)
    valid = [row for row in rows if not row.get("error_or_stop_reason")]
    summary.update(
        {
            "retrieved_memory_to_weaver": all(
                bool(row.get("retrieved_memory_to_weaver")) for row in valid
            ) if valid else True,
            "raw_retrieved_latents_enter_reasoner": any(
                bool(row.get("raw_retrieved_latents_enter_reasoner")) for row in valid
            ),
            "weaver_conditioned_on_retrieved_memory": any(
                bool(row.get("weaver_conditioned_on_retrieved_memory")) for row in valid
            ),
            "weaver_conditioning_token_count": sum(
                int(row.get("weaver_conditioning_token_count", 0)) for row in valid
            ),
            "fused_latent_generated": any(
                bool(row.get("fused_latent_generated")) for row in valid
            ),
        }
    )
    summary["compare_against_mab5c"] = {
        "baseline_artifact": MAB5C_CANONICAL_BASELINE,
        "exact_match_delta": (
            summary["compressed_bank_on_accuracy"] - summary["compressed_bank_off_accuracy"]
            if summary["compressed_bank_on_accuracy"] is not None
            and summary["compressed_bank_off_accuracy"] is not None
            else None
        ),
        "mechanism_change": "retrieved memory routed into Weaver instead of direct Reasoner injection",
    }
    return summary


def _build_research_note(
    *,
    output_dir: Path,
    manifest: dict,
    rows: list[dict],
    summary: dict,
    git_status_before: str,
    git_status_after: str,
) -> Path:
    valid_rows = [row for row in rows if not row.get("error_or_stop_reason")]
    lines = [
        "# MAB-6A: detective_qa Version B Weaver-conditioned Memory n10",
        "",
        "## Purpose",
        "Exploratory diagnostic of Version B routing: retrieved reasoner-space memory conditions Weaver, and only the fused latent is injected into Reasoner.",
        "",
        "## Settings",
        f"- Comparison baseline: `{MAB5C_CANONICAL_BASELINE}`",
        f"- threshold: `{DEFAULT_THRESHOLD}`",
        f"- retrieve_threshold: `{DEFAULT_RETRIEVE_THRESHOLD}`",
        f"- update_threshold: `{DEFAULT_UPDATE_THRESHOLD}`",
        f"- top_k: `{DEFAULT_TOP_K}`",
        f"- max_slots: `{DEFAULT_MAX_SLOTS}`",
        f"- retrieve_policy: `{DEFAULT_RETRIEVE_POLICY}`",
        f"- update_policy: `{DEFAULT_UPDATE_POLICY}`",
        "- query mode: first-query-only",
        "- query phase: read-only",
        "- full-history detective_qa: `over_capacity_invalid`",
        "- retrieved_memory_to_weaver: `True`",
        "",
        "## Guardrails",
        "- MAB-6A is exploratory.",
        "- Weaver was not trained for this input distribution.",
        "- Version A remains the default.",
        "- MAB-6A differs from MAB-5C primarily by routing retrieved memory into Weaver.",
        "- Do not claim performance improvement unless official exact_match improves.",
        "- If exact_match remains 0 but outputs change, call it mechanism-active but not a performance win.",
        "- If outputs degrade, this supports keeping Version A as default.",
        "",
        "## Run Status",
        f"- Output directory: `{output_dir}`",
        f"- Bank-off exact match: `{summary['compressed_bank_off_accuracy']}`",
        f"- Bank-on exact match: `{summary['compressed_bank_on_accuracy']}`",
        f"- Output changed: `{summary['num_output_changed']}`",
        f"- Final slot counts: `{summary['final_slot_counts']}`",
        f"- Retrieved memory to Weaver: `{summary['retrieved_memory_to_weaver']}`",
        f"- Retrieved latents entered Weaver: `{summary['retrieved_latents_enter_weaver']}`",
        f"- Raw retrieved latents entered Reasoner: `{summary['raw_retrieved_latents_enter_reasoner']}`",
        f"- Weaver conditioned on retrieved memory: `{summary['weaver_conditioned_on_retrieved_memory']}`",
        f"- Weaver conditioning token count: `{summary['weaver_conditioning_token_count']}`",
        f"- Fused latent generated: `{summary['fused_latent_generated']}`",
        f"- Query write count: `{summary['query_write_count']}`",
        f"- Query write attempt count: `{summary['query_write_attempt_count']}`",
        f"- Cross-context leakage detected: `{summary['cross_context_leakage_detected']}`",
        f"- Write action counts: `{summary['write_action_counts']}`",
        f"- Update reason counts: `{summary['update_reason_counts']}`",
        "",
        "## Comparison",
        f"- Against MAB-5C canonical: `{summary['compare_against_mab5c']}`",
        "",
        "## Per-context Result Table",
        "| context_index | exact_match_off | exact_match_on | output_changed | raw_retrieved_latents_enter_reasoner | weaver_conditioned_on_retrieved_memory |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in valid_rows:
        lines.append(
            f"| {row['context_index']} | {row['bank_off_exact_match']} | {row['bank_on_exact_match']} | {row['output_changed']} | {row.get('raw_retrieved_latents_enter_reasoner')} | {row.get('weaver_conditioned_on_retrieved_memory')} |"
        )
    lines.extend(
        [
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
        ]
    )
    RESEARCH_NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESEARCH_NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RESEARCH_NOTE_PATH


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
    original = {
        "EXPERIMENT_NAME": base.EXPERIMENT_NAME,
        "RUN_PREFIX": base.RUN_PREFIX,
        "DEFAULT_OUTPUT_ROOT": base.DEFAULT_OUTPUT_ROOT,
        "DEFAULT_THRESHOLD": base.DEFAULT_THRESHOLD,
        "DEFAULT_TOP_K": base.DEFAULT_TOP_K,
        "DEFAULT_MAX_SLOTS": base.DEFAULT_MAX_SLOTS,
        "DEFAULT_RETRIEVE_POLICY": base.DEFAULT_RETRIEVE_POLICY,
        "GIT_STATUS_BEFORE_EDIT": base.GIT_STATUS_BEFORE_EDIT,
        "build_parser": base.build_parser,
        "_bank_config": base._bank_config,
        "_build_row": base._build_row,
        "_aggregate": base._aggregate,
        "_build_research_note": base._build_research_note,
        "_build_manifest": base._build_manifest,
        "_load_model": base._load_model,
        "_run_model": base._run_model,
        "ALLOW_RETRIEVED_LATENTS_ENTER_WEAVER": base.ALLOW_RETRIEVED_LATENTS_ENTER_WEAVER,
    }
    base.EXPERIMENT_NAME = EXPERIMENT_NAME
    base.RUN_PREFIX = RUN_PREFIX
    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    base.DEFAULT_THRESHOLD = DEFAULT_THRESHOLD
    base.DEFAULT_TOP_K = DEFAULT_TOP_K
    base.DEFAULT_MAX_SLOTS = DEFAULT_MAX_SLOTS
    base.DEFAULT_RETRIEVE_POLICY = DEFAULT_RETRIEVE_POLICY
    base.GIT_STATUS_BEFORE_EDIT = mab5c._git("status", "--short", "--branch")
    base.build_parser = build_parser
    base._bank_config = _bank_config
    base._build_row = _build_row
    base._aggregate = _aggregate
    base._build_research_note = _build_research_note
    base._build_manifest = _build_manifest
    base._load_model = _load_model
    base._run_model = _run_model
    base.ALLOW_RETRIEVED_LATENTS_ENTER_WEAVER = True
    try:
        result = base.main()
        if result == 0:
            output_root = Path(DEFAULT_OUTPUT_ROOT)
            run_dirs = sorted(
                [path for path in output_root.iterdir() if path.is_dir()],
                key=lambda path: path.stat().st_mtime,
            )
            if run_dirs:
                run_dir = run_dirs[-1]
                run_config_path = run_dir / "run_config.json"
                if run_config_path.exists():
                    run_config = _load_json(run_config_path)
                    run_config.update(
                        {
                            "retrieve_threshold": DEFAULT_RETRIEVE_THRESHOLD,
                            "update_threshold": DEFAULT_UPDATE_THRESHOLD,
                            "retrieved_memory_to_weaver": True,
                            "mechanism": "version_b_weaver_conditioned_retrieval",
                            "comparison_baseline": MAB5C_CANONICAL_BASELINE,
                        }
                    )
                    _write_json(run_config_path, run_config)
        return result
    finally:
        base.EXPERIMENT_NAME = original["EXPERIMENT_NAME"]
        base.RUN_PREFIX = original["RUN_PREFIX"]
        base.DEFAULT_OUTPUT_ROOT = original["DEFAULT_OUTPUT_ROOT"]
        base.DEFAULT_THRESHOLD = original["DEFAULT_THRESHOLD"]
        base.DEFAULT_TOP_K = original["DEFAULT_TOP_K"]
        base.DEFAULT_MAX_SLOTS = original["DEFAULT_MAX_SLOTS"]
        base.DEFAULT_RETRIEVE_POLICY = original["DEFAULT_RETRIEVE_POLICY"]
        base.GIT_STATUS_BEFORE_EDIT = original["GIT_STATUS_BEFORE_EDIT"]
        base.build_parser = original["build_parser"]
        base._bank_config = original["_bank_config"]
        base._build_row = original["_build_row"]
        base._aggregate = original["_aggregate"]
        base._build_research_note = original["_build_research_note"]
        base._build_manifest = original["_build_manifest"]
        base._load_model = original["_load_model"]
        base._run_model = original["_run_model"]
        base.ALLOW_RETRIEVED_LATENTS_ENTER_WEAVER = original["ALLOW_RETRIEVED_LATENTS_ENTER_WEAVER"]


if __name__ == "__main__":
    raise SystemExit(main())
