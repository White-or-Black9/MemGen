"""Normalize the deterministic multiple-choice MemBench subsets for MemGen.

This adapter intentionally does not reuse MemBench's text-memory ``recall``
interface.  It preserves the official trajectories, questions, choices, and
strict letter-exact scoring so a later P7 runner can construct a latent bank
during the message stream and retrieve only at question time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


# Official FirstAgent JSON files verified locally against the four-choice
# contract.  The runner still records the exact filename/subset in each
# artifact; this set only guards against an accidental arbitrary JSON input.
SUPPORTED_SUBSETS = {
    "simple",
    "knowledge_update",
    "highlevel",
    "lowlevel_rec",
    "highlevel_rec",
    "RecMultiSession",
}
CHOICE_KEYS = ("A", "B", "C", "D")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("MemBench dataset must be a non-empty JSON object")
    return payload


def _iter_trajectories(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and "message_list" in item:
                yield item
            elif isinstance(item, (list, dict)):
                yield from _iter_trajectories(item)
    elif isinstance(value, dict):
        if "message_list" in value:
            yield value
        else:
            for item in value.values():
                yield from _iter_trajectories(item)


def _flatten_messages(messages: Any) -> Iterator[dict[str, Any]]:
    if isinstance(messages, list):
        for item in messages:
            yield from _flatten_messages(item)
    elif isinstance(messages, dict):
        yield messages
    else:
        raise ValueError(f"unexpected message type: {type(messages).__name__}")


def render_message(message: dict[str, Any]) -> str:
    """Render one official message without dropping speaker, time, or place."""
    user = message.get("user_message", message.get("user"))
    assistant = message.get("assistant_message", message.get("assistant", message.get("agent")))
    if user is None and assistant is None:
        raw = message.get("message")
        if raw is None:
            raise ValueError("MemBench message lacks user/assistant/message content")
        return str(raw)

    lines = []
    if user is not None:
        lines.append(f"User: {user}")
    if assistant is not None:
        lines.append(f"Assistant: {assistant}")
    if message.get("time") is not None:
        lines.append(f"Time: {message['time']}")
    if message.get("place") is not None:
        lines.append(f"Place: {message['place']}")
    return "\n".join(lines)


def normalize_trajectory(trajectory: dict[str, Any], *, category: str) -> dict[str, Any]:
    qa = trajectory.get("QA")
    if not isinstance(qa, dict):
        raise ValueError("MemBench trajectory lacks QA metadata")
    choices = qa.get("choices")
    if not isinstance(choices, dict) or tuple(sorted(choices)) != CHOICE_KEYS:
        raise ValueError("MemBench QA choices must contain exactly A/B/C/D")
    ground_truth = qa.get("ground_truth")
    if ground_truth not in CHOICE_KEYS:
        raise ValueError("MemBench QA ground_truth must be one of A/B/C/D")
    if not isinstance(qa.get("question"), str) or not qa["question"].strip():
        raise ValueError("MemBench QA question must be non-empty text")

    turns = []
    for turn_index, message in enumerate(_flatten_messages(trajectory.get("message_list"))):
        turns.append(
            {
                "turn_index": turn_index,
                "source_step_id": message.get("sid", message.get("mid")),
                "content": render_message(message),
            }
        )

    if not turns:
        raise ValueError("MemBench trajectory has no construction turns")
    trajectory_id = trajectory.get("tid")
    if trajectory_id is None:
        raise ValueError("MemBench trajectory lacks tid")
    query_id = qa.get("qid")
    if query_id is None:
        raise ValueError("MemBench QA lacks qid")

    return {
        "category": category,
        "context_id": f"membench-{category}-{trajectory_id}",
        "trajectory_id": trajectory_id,
        "construction_turns": turns,
        "query": {
            "query_id": query_id,
            "question": qa["question"],
            "question_time": qa.get("time"),
            "choices": {key: choices[key] for key in CHOICE_KEYS},
            "gold_choice": ground_truth,
            "gold_answer": qa.get("answer"),
            "target_step_id": qa.get("target_step_id", []),
        },
    }


def normalize_dataset(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for category, category_payload in payload.items():
        for trajectory in _iter_trajectories(category_payload):
            records.append(normalize_trajectory(trajectory, category=str(category)))
    if not records:
        raise ValueError("MemBench dataset contains no trajectories")
    identifiers = [(record["category"], record["trajectory_id"]) for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("MemBench trajectory identifiers are not unique within category")
    return records


def score_choice(prediction: str, gold_choice: str) -> dict[str, Any]:
    """Mirror the official environment's strict action-response equality."""
    if gold_choice not in CHOICE_KEYS:
        raise ValueError("gold_choice must be one of A/B/C/D")
    if not isinstance(prediction, str):
        raise TypeError("prediction must be a string")
    return {
        "official_choice_exact_match": float(prediction == gold_choice),
        "prediction": prediction,
        "gold_choice": gold_choice,
        "valid_choice_output": prediction in CHOICE_KEYS,
    }


def inspect_dataset(path: Path, *, subset: str) -> dict[str, Any]:
    if subset not in SUPPORTED_SUBSETS:
        raise ValueError(f"unsupported MemBench subset: {subset}")
    records = normalize_dataset(load_dataset(path))
    category_counts: dict[str, int] = {}
    for record in records:
        category_counts[record["category"]] = category_counts.get(record["category"], 0) + 1
    return {
        "schema_version": "membench-latent-adapter-audit/v1",
        "subset": subset,
        "dataset_path": str(path),
        "dataset_sha256": sha256_file(path),
        "trajectory_count": len(records),
        "category_counts": category_counts,
        "question_count": len(records),
        "primary_metric": "official_choice_exact_match",
        "scoring_contract": "strict prediction == gold_choice; no normalization",
        "p7_protocol": "construct sequentially, freeze before one query, block query writes",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--subset", required=True, choices=sorted(SUPPORTED_SUBSETS))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(inspect_dataset(args.dataset, subset=args.subset), indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
