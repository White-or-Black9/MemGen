"""MAB-6B-FR EventQA 65536: Weaver-space bank on 5 contexts with substring EM."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MethodType

import pyarrow.parquet as pq
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import mab5a_detectiveqa_compressed_n10 as base
from scripts.eval import mab3_bank_on_full_history as mab3
from scripts.eval import mab6b_weaver_space_bank_detectiveqa_n10 as weaver_bank
from memgen.model.latent_memory_bank import LatentMemoryBankConfig


EXPERIMENT_NAME = "MAB-6B-FR EventQA 65536 Weaver-space Bank n5"
RUN_PREFIX = "eventqa-65536-version-b-weaver-space-bank-n5"
SPLIT = "Accurate_Retrieval"
SUB_DATASET = "eventqa_65536"
DATA_CONFIG = "configs/data_conf/Accurate_Retrieval/EventQA/Eventqa_64k.yaml"
DEFAULT_OUTPUT_ROOT = "outputs/mab/version_b_weaver_space_bank_eventqa_65536_n5"
RESEARCH_NOTE_PATH = Path(
    "research_notes/benchmarks/memoryagentbench_mab6b_fr_eventqa_65536_n5.md"
)
CANONICAL_DETECTIVE_NOTE_PATH = Path(
    "research_notes/benchmarks/memoryagentbench_mab6b_weaver_space_bank.md"
)
DEFAULT_REQUESTED_CONTEXTS = 5
DEFAULT_CHUNK_SIZE = 4096
GENERATION_MAX_LENGTH = 40
DEFAULT_RETRIEVE_THRESHOLD = 0.005
DEFAULT_UPDATE_THRESHOLD = 0.08
DEFAULT_TOP_K = 1
DEFAULT_MAX_SLOTS = 16
DEFAULT_DECAY_ALPHA = 0.05
DEFAULT_RETRIEVE_POLICY = "threshold_topk"
DEFAULT_UPDATE_POLICY = "thread_update"
METRIC_KEY = "substring_exact_match"
OPTIONAL_METRIC_KEY = "eventqa_recall"
EVENTQA_PROTOCOLS = ("independent_episode", "frozen_context_bank")
DEFAULT_EVENTQA_PROTOCOL = "frozen_context_bank"
EMPTY_OUTPUT_TOKENS = {"", "none", "n/a"}
EVENTQA_QUERY_TEMPLATE = (
    "Based on the context you memorized, complete the task below:\n\n"
    "{question}\n\n The event that happens next is:"
)
BANK_CONFIG_FIELDS = (
    "enabled",
    "batch_size",
    "max_slots",
    "top_k",
    "threshold",
    "decay_alpha",
    "pool_last_n",
    "retrieve_policy",
    "update_policy",
    "storage_device",
    "debug",
    "retrieve_threshold",
    "update_threshold",
)


class _ConstructionOnlyStop(RuntimeError):
    """Signal that construction finished and query generation was skipped."""


def _eventqa_bank_config(args) -> dict:
    retrieve_threshold = float(
        getattr(args, "retrieve_threshold", DEFAULT_RETRIEVE_THRESHOLD)
    )
    config = mab3.version_a_bank_config(
        top_k=int(getattr(args, "top_k", DEFAULT_TOP_K)),
        threshold=retrieve_threshold,
        retrieve_policy=DEFAULT_RETRIEVE_POLICY,
    )
    config.update(
        {
            "retrieve_threshold": retrieve_threshold,
            "update_threshold": float(
                getattr(args, "update_threshold", DEFAULT_UPDATE_THRESHOLD)
            ),
            "max_slots": int(getattr(args, "max_slots", DEFAULT_MAX_SLOTS)),
            "top_k": int(getattr(args, "top_k", DEFAULT_TOP_K)),
            "decay_alpha": float(
                getattr(args, "decay_alpha", DEFAULT_DECAY_ALPHA)
            ),
            "retrieve_policy": DEFAULT_RETRIEVE_POLICY,
            "update_policy": DEFAULT_UPDATE_POLICY,
        }
    )
    return config


def _assert_runtime_bank_config_matches(actual_config, recorded: dict) -> None:
    actual = {
        field: (
            getattr(actual_config, field)
            if hasattr(actual_config, field)
            else actual_config[field]
        )
        for field in BANK_CONFIG_FIELDS
    }
    recorded_config = {
        field: recorded.get(
            field,
            recorded.get("latent_memory_bank_config", {}).get(field),
        )
        for field in BANK_CONFIG_FIELDS
    }
    mismatches = {
        field: {"actual": actual[field], "recorded": recorded_config[field]}
        for field in BANK_CONFIG_FIELDS
        if actual[field] != recorded_config[field]
    }
    if mismatches:
        details = ", ".join(
            f"{field}: actual={values['actual']!r}, recorded={values['recorded']!r}"
            for field, values in mismatches.items()
        )
        raise RuntimeError(f"EventQA runtime bank config mismatch: {details}")


def _bank_state_fingerprint(bank) -> str:
    if not hasattr(bank, "state_dict"):
        return hashlib.sha256(
            json.dumps(
                bank.debug_summary(), sort_keys=True, default=str
            ).encode("utf-8")
        ).hexdigest()
    state = bank.state_dict()
    digest = hashlib.sha256()
    digest.update(str(state["step"]).encode("ascii"))
    digest.update(str(state["retrieval_step"]).encode("ascii"))
    digest.update(str(bank.debug_summary()["memory_retrieve_count"]).encode("ascii"))
    digest.update(str(bank.debug_summary()["retrieved_latent_count"]).encode("ascii"))
    for slot in state["slots"]:
        for name in ("memory", "key"):
            tensor = slot[name].detach().to("cpu").contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            if tensor.dtype == torch.bfloat16:
                # NumPy has no native bfloat16 support; cast only for hashing.
                tensor = tensor.to(torch.float32)
            digest.update(tensor.numpy().tobytes())
        for name in (
            "metadata",
            "created_step",
            "last_access_step",
            "last_retrieved_step",
            "access_count",
            "last_score",
            "original_device",
            "original_dtype",
        ):
            digest.update(
                json.dumps(slot[name], sort_keys=True, default=str).encode("utf-8")
            )
    return digest.hexdigest()


def _capture_retrieval_state(bank) -> dict:
    return {
        "retrieval_step": bank._retrieval_step,
        "memory_retrieve_count": bank._memory_retrieve_count,
        "retrieved_latent_count": bank._retrieved_latent_count,
        "slots": [
            {
                "last_access_step": slot.last_access_step,
                "last_retrieved_step": slot.last_retrieved_step,
                "access_count": slot.access_count,
                "last_score": slot.last_score,
            }
            for slot in bank._slots
        ],
    }


def _restore_retrieval_state(bank, state: dict) -> None:
    bank._retrieval_step = state["retrieval_step"]
    bank._memory_retrieve_count = state["memory_retrieve_count"]
    bank._retrieved_latent_count = state["retrieved_latent_count"]
    if len(bank._slots) != len(state["slots"]):
        raise RuntimeError("Frozen EventQA bank slot count changed during retrieval")
    for slot, slot_state in zip(bank._slots, state["slots"]):
        slot.last_access_step = slot_state["last_access_step"]
        slot.last_retrieved_step = slot_state["last_retrieved_step"]
        slot.access_count = slot_state["access_count"]
        slot.last_score = slot_state["last_score"]


def _install_eventqa_bank_trace(bank, trace: dict, lifecycle: dict):
    original_retrieve = bank.retrieve_with_context
    original_write_back = bank.write_back

    def tracked_retrieve(self, *args, **kwargs):
        result = original_retrieve(*args, **kwargs)
        trace["last_retrieval"] = {
            "scores": list(result.scores),
            "max_score": (
                None if result.max_score is None else float(result.max_score)
            ),
            "argmax_index": result.argmax_index,
            "threshold_passed": bool(result.threshold_passed),
            "retrieved_indices": list(result.retrieved_indices),
            "retrieved_scores": list(result.retrieved_scores),
            "retrieved_slot_count": len(result.slots),
            "retrieved_latent_count": sum(
                int(slot.memory.shape[0]) for slot in result.slots
            ),
            "retrieval_step": int(result.retrieval_step),
        }
        return result

    def tracked_write_back(self, memory, retrieval_result, *args, **kwargs):
        result = original_write_back(memory, retrieval_result, *args, **kwargs)
        debug = self.debug_summary()
        write_event = debug.get("last_write_back") or {}
        lifecycle.setdefault("construction_turn_diagnostics", []).append(
            {
                "construction_turn_index": len(
                    lifecycle.get("construction_turn_diagnostics", [])
                ),
                "write_action": write_event.get("write_action"),
                "best_matched_score": (
                    None
                    if retrieval_result.max_score is None
                    else float(retrieval_result.max_score)
                ),
                "candidate_scores": [float(score) for score in retrieval_result.scores],
                "slot_count_after_write": int(debug["slot_count"]),
                "write_count_after_write": int(debug["memory_write_count"]),
            }
        )
        return result

    bank.retrieve_with_context = MethodType(tracked_retrieve, bank)
    bank.write_back = MethodType(tracked_write_back, bank)

    def restore() -> None:
        bank.retrieve_with_context = original_retrieve
        bank.write_back = original_write_back

    return restore


def _compact_bank_summary(summary: dict) -> dict:
    slots = list(summary.get("slots", []))
    action_counts = dict(summary.get("write_action_counts", {}))
    return {
        "slot_count": int(summary.get("slot_count", len(slots))),
        "write_count": int(summary.get("memory_write_count", 0)),
        "retrieval_count": int(summary.get("memory_retrieve_count", 0)),
        "slot_indices": list(range(len(slots))),
        "slots": [
            {
                "slot_index": index,
                "created_step": slot.get("created_step"),
                "last_retrieved_step": slot.get("last_retrieved_step"),
                "last_retrieved_age": slot.get("last_retrieved_age"),
                "access_count": slot.get("access_count"),
                "last_score": slot.get("last_score"),
                "memory_shape": slot.get("memory_shape"),
            }
            for index, slot in enumerate(slots)
        ],
        "true_insert_count": int(action_counts.get("insert", 0)),
        "true_matched_replace_count": int(summary.get("matched_replace_count", 0)),
        "true_capacity_evict_count": int(summary.get("capacity_evict_count", 0)),
        "true_replace_old_slot_count": int(
            action_counts.get("replace_oldest", 0)
            + action_counts.get("replace_old_slot", 0)
        ),
        "write_action_counts": action_counts,
    }


class _QueryReadOnlyBank:
    """Delegate construction writes and block all writes after query begins."""

    def __init__(
        self,
        bank,
        lifecycle: dict,
        *,
        freeze_retrieval_state: bool = False,
        preserve_on_reset: bool = False,
    ):
        self.bank = bank
        self.lifecycle = lifecycle
        self.freeze_retrieval_state = freeze_retrieval_state
        self.preserve_on_reset = preserve_on_reset
        self.read_only = False
        self.write_attempt_count = 0
        self._query_attempt_count_start = 0
        self._query_snapshot_finalized = False

    @property
    def config(self):
        return self.bank.config

    def __len__(self):
        return len(self.bank)

    def __getattr__(self, name):
        return getattr(self.bank, name)

    def retrieve(self, *args, **kwargs):
        return self.retrieve_with_context(*args, **kwargs).slots

    def retrieve_with_context(self, *args, **kwargs):
        if not self.read_only or not self.freeze_retrieval_state:
            return self.bank.retrieve_with_context(*args, **kwargs)
        retrieval_state = _capture_retrieval_state(self.bank)
        try:
            return self.bank.retrieve_with_context(*args, **kwargs)
        finally:
            _restore_retrieval_state(self.bank, retrieval_state)

    def reset(self):
        if self.preserve_on_reset:
            self.lifecycle["suppressed_bank_reset_count"] = (
                self.lifecycle.get("suppressed_bank_reset_count", 0) + 1
            )
            return None
        return self.bank.reset()

    def write(self, *args, **kwargs):
        if not self.read_only:
            return self.bank.write(*args, **kwargs)
        self.write_attempt_count += 1
        return None

    def write_back(self, *args, **kwargs):
        if not self.read_only:
            return self.bank.write_back(*args, **kwargs)
        self.write_attempt_count += 1
        return None

    def begin_query(self) -> None:
        if self.read_only:
            raise RuntimeError("EventQA query read-only mode was entered twice")
        self.lifecycle["pre_query_bank_summary"] = _compact_bank_summary(
            self.bank.debug_summary()
        )
        self.lifecycle["pre_query_bank_fingerprint"] = _bank_state_fingerprint(
            self.bank
        )
        self._query_attempt_count_start = self.write_attempt_count
        self.read_only = True

    def capture_post_query(self) -> dict:
        if self._query_snapshot_finalized:
            return self.lifecycle["post_query_bank_summary"]
        post_summary = _compact_bank_summary(self.bank.debug_summary())
        self.lifecycle["post_query_bank_summary"] = post_summary
        pre_summary = self.lifecycle["pre_query_bank_summary"]
        write_delta = post_summary["write_count"] - pre_summary["write_count"]
        attempt_delta = self.write_attempt_count - self._query_attempt_count_start
        self.lifecycle["query_write_count_delta"] = write_delta
        self.lifecycle["query_write_attempt_count_delta"] = attempt_delta
        self.lifecycle["query_read_only_enforced"] = write_delta == 0
        post_fingerprint = _bank_state_fingerprint(self.bank)
        self.lifecycle["post_query_bank_fingerprint"] = post_fingerprint
        self.lifecycle["bank_snapshot_changed_after_query"] = (
            post_fingerprint != self.lifecycle["pre_query_bank_fingerprint"]
        )
        if write_delta != 0:
            raise RuntimeError(
                f"EventQA query wrote to the real bank: write_count delta={write_delta}"
            )
        self._query_snapshot_finalized = True
        return post_summary

    def debug_summary(self):
        if self.read_only:
            self.capture_post_query()
        return self.bank.debug_summary()


def _query_memory_diagnostics(
    lifecycle: dict,
    query_turn: dict,
    *,
    retrieve_threshold: float,
) -> dict:
    candidate_scores = [float(score) for score in query_turn.get("scores", [])]
    retrieved_indices = list(query_turn.get("retrieved_indices", []))
    slot_1_existed = len(candidate_scores) > 1
    slot_1_passed_threshold = bool(
        slot_1_existed and candidate_scores[1] >= retrieve_threshold
    )
    post_summary = lifecycle["post_query_bank_summary"]
    return {
        "pre_query_bank_summary": lifecycle["pre_query_bank_summary"],
        "post_query_bank_summary": post_summary,
        "pre_query_write_count": lifecycle["pre_query_bank_summary"]["write_count"],
        "post_query_write_count": post_summary["write_count"],
        "query_write_count_delta": lifecycle["query_write_count_delta"],
        "query_write_attempt_count_delta": lifecycle[
            "query_write_attempt_count_delta"
        ],
        "query_read_only_enforced": lifecycle["query_read_only_enforced"],
        "bank_snapshot_changed_after_query": lifecycle.get(
            "bank_snapshot_changed_after_query", False
        ),
        "true_insert_count": post_summary["true_insert_count"],
        "true_matched_replace_count": post_summary["true_matched_replace_count"],
        "true_capacity_evict_count": post_summary["true_capacity_evict_count"],
        "true_replace_old_slot_count": post_summary[
            "true_replace_old_slot_count"
        ],
        "query_candidate_scores": candidate_scores,
        "query_candidate_slot_count": len(candidate_scores),
        "query_slot_1_existed": slot_1_existed,
        "query_slot_1_passed_retrieve_threshold": slot_1_passed_threshold,
        "query_slot_1_lost_top_k1_ranking": bool(
            slot_1_passed_threshold and 1 not in retrieved_indices
        ),
    }


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


def _mab_env() -> dict:
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    return env


def _bridge_script() -> Path:
    return Path(base.__file__).with_name("mab2_mab_bridge.py")


def _load_rows(parquet_path: str, sub_dataset: str) -> list[dict]:
    rows = pq.read_table(parquet_path).to_pylist()
    return [row for row in rows if row.get("metadata", {}).get("source") == sub_dataset]


def count_context_matches(parquet_path: str, sub_dataset: str) -> int:
    return len(_load_rows(parquet_path, sub_dataset))


def select_context_indices(
    total_matches: int, requested: int, *, context_index: int | None = None
) -> list[int]:
    if context_index is not None:
        if not 0 <= context_index < total_matches:
            raise ValueError(
                f"context-index {context_index} is out of range for "
                f"{total_matches} matched contexts"
            )
        return [context_index]
    return list(range(min(total_matches, requested)))


def _question_metadata_list(row: dict, key: str, total_questions: int):
    values = row.get("metadata", {}).get(key, [])
    if not isinstance(values, list):
        values = [values]
    if len(values) < total_questions:
        values = list(values) + [None] * (total_questions - len(values))
    return values


def build_context_payload(args, row: dict, context_index: int, timestamp: str) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        prepared_path = Path(tmpdir) / f"eventqa_prepare_{context_index}.json"
        command = [
            args.mab_python,
            str(_bridge_script()),
            "prepare",
            "--mab-repo", args.mab_repo,
            "--output", str(prepared_path),
            "--parquet", args.parquet,
            "--data-config", str(Path(args.mab_repo, args.data_config)),
            "--sub-dataset", SUB_DATASET,
            "--chunk-size", str(args.chunk_size),
            "--timestamp", timestamp,
            "--match-index", str(context_index),
        ]
        subprocess.run(command, check=True, env=_mab_env())
        prepared = _load_json(prepared_path)
    chunks = prepared["chunks"]
    chunk_token_lengths = prepared["chunk_token_lengths"]
    memorization_prompts = prepared["memorization_prompts"]
    dataset_config = prepared["dataset_config"]
    questions = list(row["questions"])
    answers = [
        answer if isinstance(answer, list) else [answer]
        for answer in row["answers"]
    ]
    question_ids = _question_metadata_list(row, "question_ids", len(questions))
    question_types = _question_metadata_list(row, "question_types", len(questions))
    qa_pair_ids = _question_metadata_list(row, "qa_pair_ids", len(questions))
    previous_events = _question_metadata_list(row, "previous_events", len(questions))
    context_sha = hashlib.sha256(row["context"].encode("utf-8")).hexdigest()
    return {
        "dataset_config": dataset_config,
        "context_id": f"eventqa-{context_sha[:16]}",
        "context_index": context_index,
        "chunks": chunks,
        "chunk_token_lengths": chunk_token_lengths,
        "memorization_prompts": memorization_prompts,
        "questions": questions,
        "answers": answers,
        "question_ids": question_ids,
        "question_types": question_types,
        "qa_pair_ids": qa_pair_ids,
        "previous_events": previous_events,
        "question_count": len(questions),
        "source": row.get("metadata", {}).get("source"),
        "timestamp": timestamp,
        "template": prepared.get("template"),
    }


def build_question_payload(context_payload: dict, question_index: int) -> dict:
    question = context_payload["questions"][question_index]
    return {
        "dataset_config": context_payload["dataset_config"],
        "context_id": context_payload["context_id"],
        "context_index": context_payload["context_index"],
        "query_id": question_index,
        "question_id": context_payload["question_ids"][question_index],
        "question_type": context_payload["question_types"][question_index],
        "qa_pair_id": context_payload["qa_pair_ids"][question_index],
        "previous_events": context_payload["previous_events"][question_index],
        "chunks": context_payload["chunks"],
        "chunk_token_lengths": context_payload["chunk_token_lengths"],
        "memorization_prompts": context_payload["memorization_prompts"],
        "query_prompt": EVENTQA_QUERY_TEMPLATE.format(question=question),
        "question": question,
        "gold_answers": list(context_payload["answers"][question_index]),
    }


def _query_only_payload(payload: dict) -> dict:
    query_payload = dict(payload)
    query_payload["chunks"] = []
    query_payload["chunk_token_lengths"] = []
    query_payload["memorization_prompts"] = [payload["query_prompt"]]
    return query_payload


def _construction_only_payload(context_payload: dict) -> dict:
    if context_payload["questions"]:
        return build_question_payload(context_payload, 0)
    return {
        "context_id": context_payload["context_id"],
        "context_index": context_payload["context_index"],
        "query_id": None,
        "question_id": None,
        "question_type": None,
        "qa_pair_id": None,
        "previous_events": [],
        "chunks": context_payload["chunks"],
        "chunk_token_lengths": context_payload["chunk_token_lengths"],
        "memorization_prompts": context_payload["memorization_prompts"],
        "query_prompt": "",
        "question": None,
        "gold_answers": [],
    }


def _eventqa_manager_factory(
    original_factory,
    capture: dict,
    *,
    external_bank=None,
    preserve_bank: bool = False,
    construction_only: bool = False,
    generation_max_length: int = GENERATION_MAX_LENGTH,
    recorded_bank_config: dict | None = None,
):
    def factory(
        chunks,
        query,
        capacity,
        prompt_trace,
        lifecycle,
        bank_trace,
        *,
        bank_mode: str,
    ):
        capture["lifecycle"] = lifecycle
        parent = original_factory(
            chunks,
            query,
            capacity,
            prompt_trace,
            lifecycle,
            bank_trace,
            bank_mode=bank_mode,
        )

        class EventQAReadOnlyQueryManager(parent):
            def __init__(self, tokenizer, actor_rollout_wg, config, is_validation=False):
                config.max_response_length = generation_max_length
                super().__init__(
                    tokenizer,
                    actor_rollout_wg,
                    config,
                    is_validation=is_validation,
                )
                self.generation_config.max_new_tokens = generation_max_length

            def _create_session_memory_bank(self, actual_batch_size):
                from interactions.multiturn_interaction import (
                    MultiTurnInteractionManager,
                )

                if external_bank is None:
                    bank = MultiTurnInteractionManager._create_session_memory_bank(
                        self, actual_batch_size
                    )
                    lifecycle["context_initial_reset_count"] = int(bank is not None)
                else:
                    bank = external_bank
                    lifecycle["context_initial_reset_count"] = 0
                lifecycle["session_count"] += 1
                lifecycle["bank_created"] = bank is not None
                if bank is None:
                    return None
                if recorded_bank_config is None:
                    raise RuntimeError("EventQA recorded bank config is required")
                _assert_runtime_bank_config_matches(
                    bank.config,
                    recorded_bank_config,
                )
                lifecycle["bank"] = bank
                lifecycle["created_bank_id"] = id(bank)
                lifecycle["initial_slot_count"] = len(bank)
                if external_bank is None and len(bank) != 0:
                    raise RuntimeError("New EventQA context bank was not empty")
                lifecycle["construction_turn_diagnostics"] = []
                capture["restore_bank_trace"] = _install_eventqa_bank_trace(
                    bank, bank_trace, lifecycle
                )
                proxy = _QueryReadOnlyBank(
                    bank,
                    lifecycle,
                    freeze_retrieval_state=True,
                    preserve_on_reset=preserve_bank,
                )
                lifecycle["query_bank_proxy"] = proxy
                lifecycle["eventqa_query_bank_proxy"] = proxy
                lifecycle["bank"] = proxy
                capture["bank"] = bank
                return proxy

            def _build_chat_history(self, rollings):
                turn = len(prompt_trace)
                messages = super()._build_chat_history(rollings)
                if turn == len(chunks):
                    proxy = lifecycle.get("eventqa_query_bank_proxy")
                    if proxy is not None:
                        proxy.begin_query()
                    lifecycle["rendered_query_messages"] = messages[0]
                    lifecycle["rendered_query_prompt"] = self.tokenizer.apply_chat_template(
                        messages[0],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    if construction_only:
                        lifecycle["construction_only_stopped_before_query"] = True
                        raise _ConstructionOnlyStop()
                return messages

        return EventQAReadOnlyQueryManager

    return factory


def _run_eventqa_model(
    args,
    model,
    capacity: int,
    payload: dict,
    bank_mode: str,
    bank_config: dict | None = None,
    *,
    external_bank=None,
    preserve_bank: bool = False,
    construction_only: bool = False,
    recorded_bank_config: dict | None = None,
) -> dict:
    if bank_mode == "on" and bank_config is None:
        raise RuntimeError("EventQA bank-on runtime config is required")
    if bank_mode == "on" and recorded_bank_config is None:
        raise RuntimeError("EventQA bank-on recorded config is required")
    original_factory = base._manager_class
    capture = {}
    base._manager_class = _eventqa_manager_factory(
        original_factory,
        capture,
        external_bank=external_bank,
        preserve_bank=preserve_bank,
        construction_only=construction_only,
        generation_max_length=int(
            getattr(args, "generation_max_length", GENERATION_MAX_LENGTH)
        ),
        recorded_bank_config=recorded_bank_config,
    )
    try:
        try:
            result = weaver_bank._run_model(
                args,
                model,
                capacity,
                payload,
                bank_mode,
                bank_config,
            )
        except _ConstructionOnlyStop:
            if not construction_only or bank_mode != "on":
                raise
            lifecycle = capture["lifecycle"]
            pre_query = lifecycle.get("pre_query_bank_summary")
            if pre_query is None:
                raise RuntimeError(
                    "Construction-only EventQA run did not capture pre-query bank summary"
                )
            result = {
                "prediction": None,
                "pre_query_bank_summary": pre_query,
                "post_query_bank_summary": pre_query,
                "query_write_count_delta": 0,
                "query_write_attempt_count_delta": 0,
                "query_read_only_enforced": True,
                "bank_snapshot_changed_after_query": False,
                "true_insert_count": pre_query["true_insert_count"],
                "true_matched_replace_count": pre_query[
                    "true_matched_replace_count"
                ],
                "true_capacity_evict_count": pre_query[
                    "true_capacity_evict_count"
                ],
                "true_replace_old_slot_count": pre_query[
                    "true_replace_old_slot_count"
                ],
                "query_write_count": 0,
                "query_write_attempt_count": 0,
                "bank_reset_after_context": lifecycle.get("post_reset_slot_count") == 0,
                "construction_only": True,
                "construction_only_stopped_before_query": bool(
                    lifecycle.get("construction_only_stopped_before_query")
                ),
                "peak_cuda_memory": None,
            }
            if preserve_bank:
                result["_retained_bank"] = capture["bank"]
    finally:
        base._manager_class = original_factory
        restore_bank_trace = capture.get("restore_bank_trace")
        if restore_bank_trace is not None:
            restore_bank_trace()

    lifecycle = capture["lifecycle"]
    result["eventqa_protocol"] = args.eventqa_protocol
    result["effective_generation_max_length"] = int(
        getattr(args, "generation_max_length", GENERATION_MAX_LENGTH)
    )
    result["rendered_query_messages"] = lifecycle["rendered_query_messages"]
    result["rendered_query_prompt"] = lifecycle["rendered_query_prompt"]
    result["bank_instance_id"] = (
        None if capture.get("bank") is None else id(capture["bank"])
    )
    result["context_initial_reset_count"] = lifecycle.get(
        "context_initial_reset_count", 0
    )
    result["context_memorization_performed"] = bool(
        bank_mode == "on" and external_bank is None and payload["chunks"]
    )
    result["construction_turn_diagnostics"] = list(
        lifecycle.get("construction_turn_diagnostics", [])
    )
    if bank_mode == "on":
        if not construction_only and "post_query_bank_summary" not in lifecycle:
            raise RuntimeError("EventQA query post-state was not captured before bank reset")
        if not construction_only:
            result.update(
                _query_memory_diagnostics(
                    lifecycle,
                    _query_turn(result),
                    retrieve_threshold=float(bank_config["retrieve_threshold"]),
                )
            )
            if not result["query_read_only_enforced"]:
                raise RuntimeError("EventQA query read-only assertion failed")
        result["bank_preserved_for_context_queries"] = preserve_bank
        result["cross_context_leakage_detected"] = False
        if preserve_bank:
            result["bank_reset_after_context"] = False
            result["_retained_bank"] = capture["bank"]
    return result


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


def _metric_value(score_payload: dict, key: str, *, default=None):
    if key in score_payload.get("metrics", {}):
        return score_payload["metrics"][key]
    if key in score_payload.get("additional", {}):
        return score_payload["additional"][key]
    return default


def _query_turn(result: dict) -> dict:
    generations = result["generations"]
    if not generations:
        raise RuntimeError("Missing generation trace")
    return generations[-1]


def _format_flags(prediction: str) -> dict:
    stripped = prediction.strip()
    lowered = stripped.lower()
    return {
        "empty_output": lowered in EMPTY_OUTPUT_TOKENS,
        "contains_json_brace": "{" in stripped or "}" in stripped,
        "contains_answer_prefix": "answer:" in lowered,
        "multiline_output": "\n" in stripped,
        "verbose_output": len(stripped.split()) > 24,
    }


def _retrieved_score_range(scores: list[float]) -> dict | None:
    if not scores:
        return None
    return {"min": min(scores), "max": max(scores)}


def _build_question_row(
    *,
    run_id: str,
    payload: dict,
    bank_off_result: dict,
    bank_on_result: dict,
    bank_off_score: dict,
    bank_on_score: dict,
    estimated_full_history_query_tokens: int,
    compressed_query_tokens_bank_off: int,
    compressed_query_tokens_bank_on: int,
) -> dict:
    bank_off_query = _query_turn(bank_off_result)
    bank_on_query = _query_turn(bank_on_result)
    bank_off_prediction = bank_off_result["prediction"]
    bank_on_prediction = bank_on_result["prediction"]
    bank_off_flags = _format_flags(bank_off_prediction)
    bank_on_flags = _format_flags(bank_on_prediction)
    return {
        "run_id": run_id,
        "context_index": payload["context_index"],
        "context_id": payload["context_id"],
        "query_id": payload["query_id"],
        "question_id": payload["question_id"],
        "question_type": payload["question_type"],
        "qa_pair_id": payload["qa_pair_id"],
        "previous_events": payload["previous_events"],
        "question": payload["question"],
        "gold_answers": payload["gold_answers"],
        "eventqa_protocol": bank_on_result["eventqa_protocol"],
        "bank_off_mode": "compressed_bridge_no_persistent_bank",
        "bank_off_is_official_long_context_baseline": False,
        "effective_generation_max_length": bank_on_result[
            "effective_generation_max_length"
        ],
        "bank_off_rendered_query_prompt": bank_off_result["rendered_query_prompt"],
        "bank_on_rendered_query_prompt": bank_on_result["rendered_query_prompt"],
        "bank_off_rendered_query_messages": bank_off_result[
            "rendered_query_messages"
        ],
        "bank_on_rendered_query_messages": bank_on_result[
            "rendered_query_messages"
        ],
        "estimated_full_history_query_tokens": estimated_full_history_query_tokens,
        "compressed_query_tokens_bank_off": compressed_query_tokens_bank_off,
        "compressed_query_tokens_bank_on": compressed_query_tokens_bank_on,
        "full_history_status": (
            "over_capacity_invalid"
            if estimated_full_history_query_tokens > bank_off_result["context_capacity"]
            else "within_capacity"
        ),
        "bank_off_prediction": bank_off_prediction,
        "bank_on_prediction": bank_on_prediction,
        "bank_off_parsed_prediction": _metric_value(
            bank_off_score, "parsed_output"
        ),
        "bank_on_parsed_prediction": _metric_value(
            bank_on_score, "parsed_output"
        ),
        "bank_off_substring_exact_match": int(bool(_metric_value(bank_off_score, METRIC_KEY, default=False))),
        "bank_on_substring_exact_match": int(bool(_metric_value(bank_on_score, METRIC_KEY, default=False))),
        "bank_off_eventqa_recall": _metric_value(bank_off_score, OPTIONAL_METRIC_KEY),
        "bank_on_eventqa_recall": _metric_value(bank_on_score, OPTIONAL_METRIC_KEY),
        "output_changed": bank_off_prediction != bank_on_prediction,
        "improved": int(
            bool(_metric_value(bank_on_score, METRIC_KEY, default=False))
            > bool(_metric_value(bank_off_score, METRIC_KEY, default=False))
        ),
        "regressed": int(
            bool(_metric_value(bank_on_score, METRIC_KEY, default=False))
            < bool(_metric_value(bank_off_score, METRIC_KEY, default=False))
        ),
        "bank_off_retrieval_count": bank_off_result["bank_retrieval_count"],
        "bank_on_retrieval_count": bank_on_result["bank_retrieval_count"],
        "bank_on_write_count": bank_on_result["bank_write_count"],
        "bank_on_retrieved_latent_count": bank_on_result["bank_retrieved_latent_count"],
        "bank_on_final_slot_count": bank_on_result["post_query_bank_summary"][
            "slot_count"
        ],
        "bank_on_query_turn_retrieved_indices": list(bank_on_query["retrieved_indices"]),
        "bank_on_query_turn_retrieved_scores": list(bank_on_query["retrieved_scores"]),
        "bank_on_query_turn_candidate_scores": bank_on_result[
            "query_candidate_scores"
        ],
        "bank_on_query_turn_candidate_slot_count": bank_on_result[
            "query_candidate_slot_count"
        ],
        "candidate_slots_before_topk": bank_on_result[
            "query_candidate_slot_count"
        ],
        "bank_on_query_slot_1_existed": bank_on_result["query_slot_1_existed"],
        "bank_on_query_slot_1_passed_retrieve_threshold": bank_on_result[
            "query_slot_1_passed_retrieve_threshold"
        ],
        "bank_on_query_slot_1_lost_top_k1_ranking": bank_on_result[
            "query_slot_1_lost_top_k1_ranking"
        ],
        "bank_on_query_turn_retrieved_latent_count": int(
            bank_on_query.get("retrieved_latent_count", 0)
        ),
        "bank_on_query_turn_retrieval_active": bool(
            bank_on_query.get("retrieved_latent_count", 0)
        ),
        "bank_on_query_turn_score_range": _retrieved_score_range(
            list(bank_on_query["retrieved_scores"])
        ),
        "pre_query_bank_summary": bank_on_result["pre_query_bank_summary"],
        "post_query_bank_summary": bank_on_result["post_query_bank_summary"],
        "pre_query_slot_count": bank_on_result["pre_query_bank_summary"][
            "slot_count"
        ],
        "post_query_slot_count": bank_on_result["post_query_bank_summary"][
            "slot_count"
        ],
        "pre_query_write_count": bank_on_result["pre_query_write_count"],
        "post_query_write_count": bank_on_result["post_query_write_count"],
        "query_write_count_delta": bank_on_result["query_write_count_delta"],
        "query_write_attempt_count_delta": bank_on_result[
            "query_write_attempt_count_delta"
        ],
        "blocked_query_write_attempts": bank_on_result[
            "query_write_attempt_count_delta"
        ],
        "query_read_only_enforced": bank_on_result["query_read_only_enforced"],
        "bank_snapshot_changed_after_query": bank_on_result[
            "bank_snapshot_changed_after_query"
        ],
        "bank_instance_id": bank_on_result["bank_instance_id"],
        "context_memorization_performed": bank_on_result[
            "context_memorization_performed"
        ],
        "construction_turn_diagnostics": bank_on_result[
            "construction_turn_diagnostics"
        ],
        "true_insert_count": bank_on_result["true_insert_count"],
        "true_matched_replace_count": bank_on_result[
            "true_matched_replace_count"
        ],
        "true_capacity_evict_count": bank_on_result[
            "true_capacity_evict_count"
        ],
        "true_replace_old_slot_count": bank_on_result[
            "true_replace_old_slot_count"
        ],
        "bank_reset_after_context": bank_on_result["bank_reset_after_context"],
        "cross_context_leakage_detected": bank_on_result["cross_context_leakage_detected"],
        "query_write_count": bank_on_result["query_write_count_delta"],
        "query_write_attempt_count": bank_on_result[
            "query_write_attempt_count_delta"
        ],
        "retrieved_latents_enter_weaver": bool(
            bank_on_query.get("retrieved_latents_enter_weaver")
        ),
        "raw_retrieved_latents_enter_reasoner": bool(
            bank_on_query.get("raw_retrieved_latents_enter_reasoner")
        ),
        "retrieved_latents_enter_reasoner": bool(
            bank_on_query.get("retrieved_latents_enter_reasoner")
        ),
        "weaver_conditioned_on_retrieved_memory": bool(
            bank_on_query.get("weaver_conditioned_on_retrieved_memory")
        ),
        "bank_off_format_flags": bank_off_flags,
        "bank_on_format_flags": bank_on_flags,
        "bank_off_empty_output": bank_off_flags["empty_output"],
        "bank_on_empty_output": bank_on_flags["empty_output"],
        "latency_seconds": bank_off_result["latency_seconds"] + bank_on_result["latency_seconds"],
        "peak_cuda_memory": max(
            [
                value
                for value in (
                    bank_off_result.get("peak_cuda_memory"),
                    bank_on_result.get("peak_cuda_memory"),
                )
                if value is not None
            ],
            default=None,
        ),
        "error_or_stop_reason": None,
    }


def _mean(values: list[float | int]) -> float | None:
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _aggregate_question_rows(rows: list[dict]) -> dict:
    valid = [row for row in rows if not row.get("error_or_stop_reason")]
    invalid = [row for row in rows if row.get("error_or_stop_reason")]
    bank_off_correct = sum(int(row["bank_off_substring_exact_match"]) for row in valid)
    bank_on_correct = sum(int(row["bank_on_substring_exact_match"]) for row in valid)
    return {
        "metric": METRIC_KEY,
        "optional_metric": OPTIONAL_METRIC_KEY,
        "num_questions_attempted": len(rows),
        "num_questions_valid": len(valid),
        "num_questions_invalid": len(invalid),
        "bank_off_accuracy": (bank_off_correct / len(valid)) if valid else None,
        "bank_on_accuracy": (bank_on_correct / len(valid)) if valid else None,
        "num_improved": sum(int(row["improved"]) for row in valid),
        "num_regressed": sum(int(row["regressed"]) for row in valid),
        "num_output_changed": sum(int(bool(row["output_changed"])) for row in valid),
        "bank_on_final_slot_counts": [row["bank_on_final_slot_count"] for row in valid],
        "bank_on_query_turn_retrieved_indices": [
            row["bank_on_query_turn_retrieved_indices"] for row in valid
        ],
        "bank_on_query_turn_retrieved_latent_counts": [
            row["bank_on_query_turn_retrieved_latent_count"] for row in valid
        ],
        "bank_on_query_turn_score_ranges": [
            row["bank_on_query_turn_score_range"] for row in valid
        ],
        "true_insert_counts": [row["true_insert_count"] for row in valid],
        "true_matched_replace_counts": [
            row["true_matched_replace_count"] for row in valid
        ],
        "true_capacity_evict_counts": [
            row["true_capacity_evict_count"] for row in valid
        ],
        "true_replace_old_slot_counts": [
            row["true_replace_old_slot_count"] for row in valid
        ],
        "query_candidate_scores": [
            row["bank_on_query_turn_candidate_scores"] for row in valid
        ],
        "query_candidate_slot_counts": [
            row["bank_on_query_turn_candidate_slot_count"] for row in valid
        ],
        "query_slot_1_lost_top_k1_ranking": [
            row["bank_on_query_slot_1_lost_top_k1_ranking"] for row in valid
        ],
        "query_write_count": sum(int(row["query_write_count"]) for row in valid),
        "query_write_attempt_count": sum(int(row["query_write_attempt_count"]) for row in valid),
        "query_read_only_enforced": all(
            bool(row["query_read_only_enforced"]) for row in valid
        ) if valid else True,
        "bank_snapshot_unchanged_across_queries": all(
            not bool(row["bank_snapshot_changed_after_query"]) for row in valid
        ) if valid else True,
        "cross_context_leakage_detected": any(
            bool(row["cross_context_leakage_detected"]) for row in valid
        ),
        "retrieved_latents_enter_weaver": all(
            bool(row["retrieved_latents_enter_weaver"]) for row in valid
        ) if valid else True,
        "raw_retrieved_latents_enter_reasoner": any(
            bool(row["raw_retrieved_latents_enter_reasoner"]) for row in valid
        ),
        "eventqa_recall_available": any(
            row["bank_off_eventqa_recall"] is not None or row["bank_on_eventqa_recall"] is not None
            for row in valid
        ),
        "bank_off_eventqa_recall_mean": _mean(
            [row["bank_off_eventqa_recall"] for row in valid if row["bank_off_eventqa_recall"] is not None]
        ),
        "bank_on_eventqa_recall_mean": _mean(
            [row["bank_on_eventqa_recall"] for row in valid if row["bank_on_eventqa_recall"] is not None]
        ),
        "bank_off_empty_output_count": sum(int(row["bank_off_empty_output"]) for row in valid),
        "bank_on_empty_output_count": sum(int(row["bank_on_empty_output"]) for row in valid),
        "bank_off_format_failure_count": sum(
            int(any(row["bank_off_format_flags"].values())) for row in valid
        ),
        "bank_on_format_failure_count": sum(
            int(any(row["bank_on_format_flags"].values())) for row in valid
        ),
        "peak_cuda_memory_max": max(
            [row["peak_cuda_memory"] for row in valid if row.get("peak_cuda_memory") is not None],
            default=None,
        ),
        "peak_cuda_memory_mean": _mean(
            [row["peak_cuda_memory"] for row in valid if row.get("peak_cuda_memory") is not None]
        ),
    }


def _build_context_summary(
    context_payload: dict,
    question_rows: list[dict],
    *,
    eventqa_protocol: str = "independent_episode",
    cleanup_slot_count: int | None = None,
    construction_result: dict | None = None,
) -> dict:
    aggregate = _aggregate_question_rows(question_rows)
    valid = [row for row in question_rows if not row.get("error_or_stop_reason")]
    aggregate.update(
        {
            "context_index": context_payload["context_index"],
            "context_id": context_payload["context_id"],
            "eventqa_protocol": eventqa_protocol,
            "question_count": aggregate["num_questions_attempted"],
            "question_count_available": context_payload["question_count"],
            "chunk_count": len(context_payload["chunks"]),
            "chunk_token_lengths": context_payload["chunk_token_lengths"],
            "total_construction_tokens": sum(context_payload["chunk_token_lengths"]),
        }
    )
    if eventqa_protocol == "frozen_context_bank":
        memorization_rows = [
            row for row in valid if row["context_memorization_performed"]
        ]
        if not memorization_rows and construction_result is not None:
            memorization_rows = [construction_result]
        construction = (
            memorization_rows[0]["construction_turn_diagnostics"]
            if memorization_rows
            else []
        )
        bank_ids = [row["bank_instance_id"] for row in valid]
        aggregate.update(
            {
                "context_memorization_count": len(memorization_rows),
                "bank_instance_ids_by_query": bank_ids,
                "same_frozen_bank_reused_across_queries": bool(bank_ids)
                and len(set(bank_ids)) == 1,
                "all_query_write_deltas_zero": all(
                    row["query_write_count_delta"] == 0 for row in valid
                ),
                "bank_snapshot_unchanged_across_queries": all(
                    not row["bank_snapshot_changed_after_query"] for row in valid
                ),
                "final_construction_slot_count": (
                    memorization_rows[0]["pre_query_bank_summary"]["slot_count"]
                    if memorization_rows
                    else None
                ),
                "true_insert_count": (
                    memorization_rows[0]["true_insert_count"]
                    if memorization_rows
                    else None
                ),
                "true_matched_replace_count": (
                    memorization_rows[0]["true_matched_replace_count"]
                    if memorization_rows
                    else None
                ),
                "true_capacity_evict_count": (
                    memorization_rows[0]["true_capacity_evict_count"]
                    if memorization_rows
                    else None
                ),
                "true_replace_old_slot_count": (
                    memorization_rows[0]["true_replace_old_slot_count"]
                    if memorization_rows
                    else None
                ),
                "construction_turn_diagnostics": construction,
                "construction_write_action_sequence": [
                    turn["write_action"] for turn in construction
                ],
                "cleanup_slot_count": cleanup_slot_count,
                "bank_reset_after_context": cleanup_slot_count == 0,
            }
        )
    return aggregate


def _build_manifest(
    run_id: str,
    args,
    started_at: str,
    *,
    git_status_before: str,
    selected_context_indices: list[int] | None = None,
) -> dict:
    checkpoint_path = str(Path(args.checkpoint_path).resolve())
    protocol = args.eventqa_protocol
    frozen_protocol = protocol == "frozen_context_bank"
    bank_config = _eventqa_bank_config(args)
    generation_max_length = int(
        getattr(args, "generation_max_length", GENERATION_MAX_LENGTH)
    )
    return {
        "experiment_name": EXPERIMENT_NAME,
        "run_id": run_id,
        "timestamp": started_at,
        "dataset_root": args.dataset_root,
        "mab_repo": args.mab_repo,
        "split": SPLIT,
        "subtask": SUB_DATASET,
        "metric": METRIC_KEY,
        "optional_metric": OPTIONAL_METRIC_KEY,
        "checkpoint_path": checkpoint_path,
        "model_checkpoint_id": args.model_checkpoint_id,
        **bank_config,
        "latent_memory_bank_config": dict(bank_config),
        "generation_max_length": generation_max_length,
        "effective_generation_max_length": generation_max_length,
        "eventqa_protocol": protocol,
        "context_bank_rebuilt_per_question": not frozen_protocol,
        "context_bank_reused_across_questions": frozen_protocol,
        "protocol_limitation": (
            None
            if frozen_protocol
            else (
                "Each question is an independent episode that rebuilds the same "
                "context bank."
            )
        ),
        "query_mode": protocol,
        "query_phase": "read-only",
        "bank_off_mode": "compressed_bridge_no_persistent_bank",
        "bank_off_is_official_long_context_baseline": False,
        "bank_off_contract": (
            "No persistent Memory Bank; official EventQA query template; no full "
            "long-context baseline because rendered full history exceeds capacity."
        ),
        "full_history_policy": "over_capacity_invalid",
        "retrieved_memory_to_weaver": True,
        "memory_bank_storage_space": "weaver",
        "research_note": str(RESEARCH_NOTE_PATH),
        "research_note_write_enabled": not getattr(
            args, "skip_research_note", False
        ),
        "canonical_note_guard": str(CANONICAL_DETECTIVE_NOTE_PATH),
        "requested_contexts": args.requested_contexts,
        "context_index": getattr(args, "context_index", None),
        "selected_context_indices": list(selected_context_indices or []),
        "question_limit": args.question_limit,
        "construction_only": bool(getattr(args, "construction_only", False)),
        "context_capacity": None,
        "git_status_before": git_status_before,
        "started_at": started_at,
        "finished_at": None,
    }


def _build_note(
    *,
    output_dir: Path,
    manifest: dict,
    context_summaries: list[dict],
    git_status_after: str,
) -> None:
    lines = [
        "# MAB-6B-FR EventQA 65536 n5",
        "",
        "Exploratory EventQA expansion using an explicit Weaver-space bank configuration.",
        "",
        "## Settings",
        f"- retrieve_threshold: `{manifest['retrieve_threshold']}`",
        f"- update_threshold: `{manifest['update_threshold']}`",
        f"- top_k: `{manifest['top_k']}`",
        f"- max_slots: `{manifest['max_slots']}`",
        f"- decay_alpha: `{manifest['decay_alpha']}`",
        f"- generation_max_length: `{manifest['generation_max_length']}`",
        f"- EventQA protocol: `{manifest['eventqa_protocol']}`",
        f"- Bank-off mode: `{manifest['bank_off_mode']}`",
        f"- metric: `{METRIC_KEY}`",
        f"- optional metric: `{OPTIONAL_METRIC_KEY}`",
        f"- output_dir: `{output_dir}`",
        f"- canonical detective note protected: `{CANONICAL_DETECTIVE_NOTE_PATH}`",
        "",
        "## Context Summaries",
    ]
    for summary in context_summaries:
        lines.extend(
            [
                f"### Context {summary['context_index']}",
                f"- context_id: `{summary['context_id']}`",
                f"- question_count: `{summary['question_count']}`",
                f"- bank_off_{METRIC_KEY}: `{summary['bank_off_accuracy']}`",
                f"- bank_on_{METRIC_KEY}: `{summary['bank_on_accuracy']}`",
                f"- final_slot_counts: `{summary['bank_on_final_slot_counts']}`",
                f"- query_turn_retrieved_latent_counts: `{summary['bank_on_query_turn_retrieved_latent_counts']}`",
                f"- query_write_count: `{summary['query_write_count']}`",
                f"- query_write_attempt_count: `{summary['query_write_attempt_count']}`",
                f"- cross_context_leakage_detected: `{summary['cross_context_leakage_detected']}`",
                f"- retrieved_latents_enter_weaver: `{summary['retrieved_latents_enter_weaver']}`",
                f"- raw_retrieved_latents_enter_reasoner: `{summary['raw_retrieved_latents_enter_reasoner']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Git Status",
            "```",
            git_status_after,
            "```",
        ]
    )
    RESEARCH_NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESEARCH_NOTE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    parser.add_argument("--data-config", default=DATA_CONFIG)
    parser.add_argument(
        "--parquet",
        default="/mnt/18T/baishilong/datasets/MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet",
    )
    parser.add_argument("--requested-contexts", type=int, default=DEFAULT_REQUESTED_CONTEXTS)
    parser.add_argument("--context-index", type=int)
    parser.add_argument("--question-limit", type=int)
    parser.add_argument("--construction-only", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--retrieve-threshold", type=float, default=DEFAULT_RETRIEVE_THRESHOLD
    )
    parser.add_argument(
        "--update-threshold", type=float, default=DEFAULT_UPDATE_THRESHOLD
    )
    parser.add_argument("--max-slots", type=int, default=DEFAULT_MAX_SLOTS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--decay-alpha", type=float, default=DEFAULT_DECAY_ALPHA)
    parser.add_argument(
        "--generation-max-length", type=int, default=GENERATION_MAX_LENGTH
    )
    parser.add_argument(
        "--eventqa-protocol",
        choices=EVENTQA_PROTOCOLS,
        default=DEFAULT_EVENTQA_PROTOCOL,
    )
    parser.add_argument("--skip-research-note", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started_at = _utc_now()
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{run_timestamp}-{RUN_PREFIX}"
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    git_status_before = _git("status", "--short", "--branch")
    rows = _load_rows(args.parquet, SUB_DATASET)
    selected_indices = select_context_indices(
        len(rows), args.requested_contexts, context_index=args.context_index
    )
    if not selected_indices:
        raise RuntimeError(f"No rows found for {SUB_DATASET}")
    runtime_bank_config = _eventqa_bank_config(args)
    manifest = _build_manifest(
        run_id,
        args,
        started_at,
        git_status_before=git_status_before,
        selected_context_indices=selected_indices,
    )
    _assert_runtime_bank_config_matches(
        LatentMemoryBankConfig(**runtime_bank_config),
        manifest,
    )
    _write_json(output_dir / "manifest.json", manifest)

    model, capacity = weaver_bank._load_model(args)
    manifest["context_capacity"] = capacity
    _write_json(output_dir / "run_config.json", manifest)

    all_question_rows: list[dict] = []
    per_context_rows: list[dict] = []
    try:
        for context_index in selected_indices:
            context_payload = build_context_payload(args, rows[context_index], context_index, started_at)
            question_total = context_payload["question_count"]
            if args.question_limit is not None:
                question_total = min(question_total, args.question_limit)
            context_question_rows = []
            frozen_bank = None
            cleanup_slot_count = None
            construction_result = None
            try:
                if args.construction_only:
                    construction_result = _run_eventqa_model(
                        args,
                        model,
                        capacity,
                        _construction_only_payload(context_payload),
                        "on",
                        runtime_bank_config,
                        preserve_bank=args.eventqa_protocol == "frozen_context_bank",
                        construction_only=True,
                        recorded_bank_config=manifest,
                    )
                    if args.eventqa_protocol == "frozen_context_bank":
                        frozen_bank = construction_result.pop("_retained_bank", None)
                for question_index in range(question_total):
                    if args.construction_only:
                        break
                    payload = build_question_payload(context_payload, question_index)
                    frozen_protocol = args.eventqa_protocol == "frozen_context_bank"
                    query_payload = _query_only_payload(payload)
                    bank_off_payload = query_payload if frozen_protocol else payload
                    bank_on_payload = (
                        payload
                        if not frozen_protocol or question_index == 0
                        else query_payload
                    )
                    with tempfile.TemporaryDirectory() as tmpdir:
                        estimated_full_history_query_tokens = base.estimate_full_history_query_tokens(
                            model.tokenizer, payload
                        )
                        compressed_query_tokens_bank_off, _, _, _ = base.compressed_query_token_count(
                            model.tokenizer, bank_off_payload
                        )
                        compressed_query_tokens_bank_on, _, _, _ = base.compressed_query_token_count(
                            model.tokenizer, bank_on_payload
                        )
                        bank_off_result = _run_eventqa_model(
                            args, model, capacity, bank_off_payload, "off"
                        )
                        bank_on_result = _run_eventqa_model(
                            args,
                            model,
                            capacity,
                            bank_on_payload,
                            "on",
                            runtime_bank_config,
                            external_bank=frozen_bank,
                            preserve_bank=frozen_protocol,
                            recorded_bank_config=manifest,
                        )
                        if frozen_protocol:
                            retained_bank = bank_on_result.pop("_retained_bank")
                            if frozen_bank is None:
                                frozen_bank = retained_bank
                            elif retained_bank is not frozen_bank:
                                raise RuntimeError(
                                    "EventQA frozen protocol replaced the context bank"
                                )
                        bank_off_score = _score_prediction(
                            args, payload, bank_off_result["prediction"], tmpdir
                        )
                        bank_on_score = _score_prediction(
                            args, payload, bank_on_result["prediction"], tmpdir
                        )
                    row = _build_question_row(
                        run_id=run_id,
                        payload=payload,
                        bank_off_result=bank_off_result,
                        bank_on_result=bank_on_result,
                        bank_off_score=bank_off_score,
                        bank_on_score=bank_on_score,
                        estimated_full_history_query_tokens=estimated_full_history_query_tokens,
                        compressed_query_tokens_bank_off=compressed_query_tokens_bank_off,
                        compressed_query_tokens_bank_on=compressed_query_tokens_bank_on,
                    )
                    context_question_rows.append(row)
                    all_question_rows.append(row)
            finally:
                if frozen_bank is not None:
                    frozen_bank.reset()
                    cleanup_slot_count = len(frozen_bank)
                    if cleanup_slot_count != 0:
                        raise RuntimeError(
                            "EventQA context bank was not empty after context cleanup"
                        )
            per_context_rows.append(
                _build_context_summary(
                    context_payload,
                    context_question_rows,
                    eventqa_protocol=args.eventqa_protocol,
                    cleanup_slot_count=cleanup_slot_count,
                    construction_result=construction_result,
                )
            )
        aggregate = {
            "run_id": run_id,
            "context_count": len(per_context_rows),
            "context_ids": [row["context_id"] for row in per_context_rows],
            "question_count_total": sum(row["question_count"] for row in per_context_rows),
            "summaries": per_context_rows,
        }
        _write_json(output_dir / "paired_results.json", aggregate)
        _write_json(output_dir / "eventqa_aggregate.json", aggregate)
        _write_jsonl(output_dir / "diagnostics.jsonl", all_question_rows)
        _write_jsonl(output_dir / "eventqa_per_question.jsonl", all_question_rows)
        _write_jsonl(output_dir / "eventqa_per_context.jsonl", per_context_rows)
        manifest["finished_at"] = _utc_now()
        manifest["context_count"] = len(per_context_rows)
        manifest["question_count_total"] = sum(row["question_count"] for row in per_context_rows)
        _write_json(output_dir / "manifest.json", manifest)
        _write_json(output_dir / "run_config.json", manifest)
        git_status_after = _git("status", "--short", "--branch")
        if not args.skip_research_note:
            _build_note(
                output_dir=output_dir,
                manifest=manifest,
                context_summaries=per_context_rows,
                git_status_after=git_status_after,
            )
    finally:
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
