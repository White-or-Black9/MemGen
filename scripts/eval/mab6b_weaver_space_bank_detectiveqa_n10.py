"""MAB-6B: detective_qa Version B Weaver-space bank on 10 contexts."""

from __future__ import annotations

import argparse
import json
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


EXPERIMENT_NAME = "MAB-6B: detective_qa Version B Weaver-space Bank n10"
RUN_PREFIX = "detectiveqa-version-b-weaver-space-bank-n10"
DEFAULT_OUTPUT_ROOT = "outputs/mab/version_b_weaver_space_bank_detectiveqa_n10"
DEFAULT_THRESHOLD = 0.03
DEFAULT_RETRIEVE_THRESHOLD = 0.03
DEFAULT_UPDATE_THRESHOLD = 0.05
DEFAULT_TOP_K = 1
DEFAULT_MAX_SLOTS = 8
DEFAULT_RETRIEVE_POLICY = "threshold_topk"
DEFAULT_UPDATE_POLICY = "thread_update"
DEFAULT_REQUESTED_CONTEXTS = 10
RESEARCH_NOTE_PATH = Path(
    "research_notes/benchmarks/memoryagentbench_mab6b_weaver_space_bank.md"
)
MAB6A_CANONICAL_BASELINE = (
    "outputs/mab/version_b_weaver_conditioned_detectiveqa_n10/"
    "20260625T023822Z-detectiveqa-version-b-weaver-conditioned-n10"
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


def _build_manifest(
    run_id: str,
    args,
    started_at: str,
    *,
    git_status_before: str,
    git_status_after: str | None = None,
) -> dict:
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
            "memory_bank_storage_space": "weaver",
            "comparison_baseline_primary": MAB6A_CANONICAL_BASELINE,
            "comparison_baseline_secondary": MAB5C_CANONICAL_BASELINE,
            "research_note": str(RESEARCH_NOTE_PATH),
        }
    )
    return manifest


def _load_model(args):
    model, capacity = _BASE_LOAD_MODEL(args)
    model.config.retrieved_memory_to_weaver = True
    model.config.query_retrieved_memory_conditioning = True
    model.config.query_latent_usage = "weaver_integrated"
    model.config.memory_bank_storage_space = "weaver"
    return model, capacity


def _run_model(args, model, capacity: int, payload: dict, bank_mode: str, bank_config: dict | None = None) -> dict:
    from interactions.base_interaction import InteractionDataProto

    previous_flag = bool(getattr(model.config, "retrieved_memory_to_weaver", False))
    previous_storage = getattr(model.config, "memory_bank_storage_space", "reasoner")
    model.config.retrieved_memory_to_weaver = bank_mode == "on"
    model.config.memory_bank_storage_space = "weaver" if bank_mode == "on" else "reasoner"
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
            "memory_bank_storage_space": final_generation.get("memory_bank_storage_space"),
            "stored_latent_space": final_generation.get("stored_latent_space"),
            "retrieval_query_space": final_generation.get("retrieval_query_space"),
            "retrieved_memory_space": final_generation.get("retrieved_memory_space"),
            "stored_weaver_latents_in_bank": bool(
                final_generation.get("stored_weaver_latents_in_bank")
            ),
            "retrieved_weaver_latents_from_bank": bool(
                final_generation.get("retrieved_weaver_latents_from_bank")
            ),
            "retrieved_memory_projected_to_weaver": bool(
                final_generation.get("retrieved_memory_projected_to_weaver")
            ),
            "retrieved_latents_enter_weaver": bool(
                final_generation.get("retrieved_latents_enter_weaver")
            ),
            "raw_retrieved_latents_enter_reasoner": bool(
                final_generation.get("raw_retrieved_latents_enter_reasoner")
            ),
            "retrieved_latents_enter_reasoner": bool(
                final_generation.get("retrieved_latents_enter_reasoner")
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


def _load_baseline_summary(artifact_path: str) -> dict | None:
    paired_results = Path(artifact_path) / "paired_results.json"
    if not paired_results.exists():
        return None
    return _load_json(paired_results).get("summary")


def _aggregate(rows: list[dict]) -> dict:
    summary = mab5c._aggregate(rows)
    valid = [row for row in rows if not row.get("error_or_stop_reason")]
    summary.update(
        {
            "retrieved_memory_to_weaver": all(
                bool(row.get("retrieved_memory_to_weaver")) for row in valid
            ) if valid else True,
            "memory_bank_storage_space": "weaver",
            "stored_latent_space": "weaver" if valid else None,
            "retrieval_query_space": "weaver" if valid else None,
            "retrieved_memory_space": "weaver" if valid else None,
            "stored_weaver_latents_in_bank": any(
                bool(row.get("stored_weaver_latents_in_bank")) for row in valid
            ),
            "retrieved_weaver_latents_from_bank": any(
                bool(row.get("retrieved_weaver_latents_from_bank")) for row in valid
            ),
            "retrieved_memory_projected_to_weaver": any(
                bool(row.get("retrieved_memory_projected_to_weaver")) for row in valid
            ),
            "retrieved_latents_enter_weaver": any(
                bool(row.get("retrieved_latents_enter_weaver")) for row in valid
            ),
            "raw_retrieved_latents_enter_reasoner": any(
                bool(row.get("raw_retrieved_latents_enter_reasoner")) for row in valid
            ),
            "retrieved_latents_enter_reasoner": any(
                bool(row.get("retrieved_latents_enter_reasoner")) for row in valid
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
    mab6a_summary = _load_baseline_summary(MAB6A_CANONICAL_BASELINE)
    summary["compare_against_mab6a"] = {
        "baseline_artifact": MAB6A_CANONICAL_BASELINE,
        "baseline_summary_available": mab6a_summary is not None,
        "bank_on_exact_match_delta": (
            summary["compressed_bank_on_accuracy"] - mab6a_summary["compressed_bank_on_accuracy"]
            if mab6a_summary is not None
            and summary["compressed_bank_on_accuracy"] is not None
            and mab6a_summary.get("compressed_bank_on_accuracy") is not None
            else None
        ),
        "output_changed_delta": (
            summary["num_output_changed"] - mab6a_summary["num_output_changed"]
            if mab6a_summary is not None and mab6a_summary.get("num_output_changed") is not None
            else None
        ),
        "retrieved_memory_projection_change": (
            "reasoner_to_weaver projection removed for retrieved memory"
        ),
    }
    summary["compare_against_mab5c"] = {
        "baseline_artifact": MAB5C_CANONICAL_BASELINE,
        "mechanism_change": (
            "bank stores Weaver-space memory and queries in Weaver space instead of "
            "storing reasoner-space memory and re-projecting retrieved memory"
        ),
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
        "# MAB-6B: detective_qa Version B Weaver-space Bank n10",
        "",
        "## Purpose",
        "Exploratory diagnostic of Weaver-space bank routing: store raw Weaver hidden states in the bank, query in Weaver space, condition Weaver directly on retrieved Weaver memory, and inject only the fused latent into Reasoner.",
        "",
        "## Settings",
        f"- Primary comparison baseline: `{MAB6A_CANONICAL_BASELINE}`",
        f"- Secondary comparison baseline: `{MAB5C_CANONICAL_BASELINE}`",
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
        "- memory_bank_storage_space: `weaver`",
        "",
        "## Guardrails",
        "- MAB-6B is exploratory.",
        "- MAB-6B tests whether storing Weaver-space memory avoids Weaver->Reasoner->Weaver information loss.",
        "- Version A remains the default.",
        "- MAB-6A remains reproducible.",
        "- MAB-6B differs from MAB-6A primarily by memory bank storage space and retrieval query space.",
        "- MAB-6B uses Weaver-space query and Weaver-space stored memory.",
        "- Do not claim performance improvement unless official exact_match improves.",
        "- If exact_match remains 0, call it mechanism-active but not a performance win.",
        "- Record whether retrieved memory avoided reasoner_to_weaver projection.",
        "",
        "## Run Status",
        f"- Output directory: `{output_dir}`",
        f"- Bank-off exact match: `{summary['compressed_bank_off_accuracy']}`",
        f"- Bank-on exact match: `{summary['compressed_bank_on_accuracy']}`",
        f"- Output changed: `{summary['num_output_changed']}`",
        f"- Final slot counts: `{summary['final_slot_counts']}`",
        f"- Memory bank storage space: `{summary['memory_bank_storage_space']}`",
        f"- Stored latent space: `{summary['stored_latent_space']}`",
        f"- Retrieval query space: `{summary['retrieval_query_space']}`",
        f"- Retrieved memory space: `{summary['retrieved_memory_space']}`",
        f"- Stored Weaver latents in bank: `{summary['stored_weaver_latents_in_bank']}`",
        f"- Retrieved Weaver latents from bank: `{summary['retrieved_weaver_latents_from_bank']}`",
        f"- Retrieved memory projected to Weaver: `{summary['retrieved_memory_projected_to_weaver']}`",
        f"- Retrieved latents entered Weaver: `{summary['retrieved_latents_enter_weaver']}`",
        f"- Raw retrieved latents entered Reasoner: `{summary['raw_retrieved_latents_enter_reasoner']}`",
        f"- Fused latent generated: `{summary['fused_latent_generated']}`",
        f"- Query write count: `{summary['query_write_count']}`",
        f"- Query write attempt count: `{summary['query_write_attempt_count']}`",
        f"- Cross-context leakage detected: `{summary['cross_context_leakage_detected']}`",
        "",
        "## Comparison",
        f"- Against MAB-6A canonical: `{summary['compare_against_mab6a']}`",
        f"- Against MAB-5C canonical: `{summary['compare_against_mab5c']}`",
        "",
        "## Per-context Result Table",
        "| context_index | exact_match_off | exact_match_on | output_changed | retrieval_query_space | retrieved_memory_projected_to_weaver | raw_retrieved_latents_enter_reasoner |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in valid_rows:
        lines.append(
            f"| {row['context_index']} | {row['bank_off_exact_match']} | {row['bank_on_exact_match']} | {row['output_changed']} | {row.get('retrieval_query_space')} | {row.get('retrieved_memory_projected_to_weaver')} | {row.get('raw_retrieved_latents_enter_reasoner')} |"
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
                            "memory_bank_storage_space": "weaver",
                            "mechanism": "version_b_weaver_space_bank",
                            "comparison_baseline_primary": MAB6A_CANONICAL_BASELINE,
                            "comparison_baseline_secondary": MAB5C_CANONICAL_BASELINE,
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
