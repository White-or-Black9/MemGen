"""Normalize and inspect FactConsolidation data using official MAB helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pyarrow.parquet as pq
import yaml


SUPPORTED_SUBTASKS = {
    "factconsolidation_sh_6k",
    "factconsolidation_mh_6k",
    "factconsolidation_sh_32k",
    "factconsolidation_mh_32k",
    "factconsolidation_sh_64k",
    "factconsolidation_mh_64k",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_mab_modules(mab_repo: Path):
    sys.path.insert(0, str(mab_repo.resolve()))
    from utils.eval_other_utils import chunk_text_into_sentences, post_process
    from utils.templates import get_template

    return chunk_text_into_sentences, post_process, get_template


def _render_template(template: str | Callable[..., str], **kwargs) -> str:
    if callable(template):
        return template(**kwargs)
    return template.format(**kwargs)


def _question_metadata_list(row: dict, key: str, total_questions: int) -> list[Any]:
    values = row.get("metadata", {}).get(key, [])
    if not isinstance(values, list):
        values = [values]
    if len(values) < total_questions:
        values = list(values) + [None] * (total_questions - len(values))
    return values


def _subtask_config_path(mab_repo: Path, split: str, subtask: str) -> Path:
    config_name = subtask.replace("factconsolidation", "Factconsolidation") + ".yaml"
    return mab_repo / "configs" / "data_conf" / split / config_name


def load_rows(parquet_path: Path, subtask: str) -> list[dict]:
    if subtask not in SUPPORTED_SUBTASKS:
        raise ValueError(f"unsupported FactConsolidation subtask: {subtask}")
    rows = pq.read_table(parquet_path).to_pylist()
    return [row for row in rows if row.get("metadata", {}).get("source") == subtask]


def normalize_context(
    row: dict,
    *,
    subtask: str,
    chunker,
    templates: dict[str, str | Callable[..., str]],
    chunk_size: int,
    timestamp: str,
    dataset_config: dict[str, Any],
    config_hash: str,
    parquet_hash: str,
) -> dict[str, Any]:
    source = row.get("metadata", {}).get("source")
    if source != subtask:
        raise ValueError(f"source mismatch: expected {subtask}, got {source}")

    chunks = chunker(row["context"], chunk_size=chunk_size)
    questions = list(row["questions"])
    answers = [answer if isinstance(answer, list) else [answer] for answer in row["answers"]]
    qa_pair_ids = _question_metadata_list(row, "qa_pair_ids", len(questions))
    context_sha = hashlib.sha256(row["context"].encode("utf-8")).hexdigest()
    memorization_prompts = [
        _render_template(templates["memorize"], context=chunk, time_stamp=timestamp)
        for chunk in chunks
    ]
    queries = []
    for query_id, question in enumerate(questions):
        queries.append(
            {
                "query_id": query_id,
                "qa_pair_id": qa_pair_ids[query_id],
                "question": question,
                "query_prompt": _render_template(templates["query"], question=question),
                "gold_answers": list(answers[query_id]),
            }
        )
    return {
        "subtask": subtask,
        "context_id": f"factconsolidation-{context_sha[:16]}",
        "dataset_config": dataset_config,
        "config_hash": config_hash,
        "parquet_hash": parquet_hash,
        "chunks": chunks,
        "memorization_prompts": memorization_prompts,
        "queries": queries,
        "gold_answers": answers,
        "qa_pair_ids": qa_pair_ids,
        "question_count": len(questions),
    }


def score_prediction(
    prediction: str,
    gold_answers: list[str],
    dataset_config: dict,
    post_process,
) -> dict[str, Any]:
    metrics, additional = post_process({"output": prediction}, gold_answers, dataset_config)
    return {"metrics": metrics, "additional": additional}


def inspect_subtask(
    *,
    subtask: str,
    rows: list[dict],
    dataset_config: dict[str, Any],
    chunker,
    templates: dict[str, str | Callable[..., str]],
    config_hash: str,
    parquet_hash: str,
    timestamp: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"no rows found for {subtask}")
    payload = normalize_context(
        rows[0],
        subtask=subtask,
        chunker=chunker,
        templates=templates,
        chunk_size=int(dataset_config["chunk_size"]),
        timestamp=timestamp,
        dataset_config=dataset_config,
        config_hash=config_hash,
        parquet_hash=parquet_hash,
    )
    return {
        "subtask": subtask,
        "matched_rows": len(rows),
        "question_count": payload["question_count"],
        "chunk_count": len(payload["chunks"]),
        "context_id": payload["context_id"],
        "qa_pair_ids": payload["qa_pair_ids"],
        "config_hash": config_hash,
        "parquet_hash": parquet_hash,
        "dataset_config": dataset_config,
    }


def _inspect(args: argparse.Namespace) -> int:
    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    parquet_path = Path(args.parquet)
    mab_repo = Path(args.mab_repo)
    chunker, _, get_template = _load_mab_modules(mab_repo)
    parquet_hash = sha256_file(parquet_path)
    timestamp = args.timestamp
    records = []
    for subtask in matrix["subtasks"]:
        config_path = _subtask_config_path(mab_repo, matrix["split"], subtask)
        dataset_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config_hash = sha256_file(config_path)
        templates = {
            "memorize": get_template(subtask, "memorize", "Long_context_agent"),
            "query": get_template(subtask, "query", "Long_context_agent"),
        }
        rows = load_rows(parquet_path, subtask)
        records.append(
            inspect_subtask(
                subtask=subtask,
                rows=rows,
                dataset_config=dataset_config,
                chunker=chunker,
                templates=templates,
                config_hash=config_hash,
                parquet_hash=parquet_hash,
                timestamp=timestamp,
            )
        )
    output = {
        "schema_version": "factconsolidation-dataset-audit/v2",
        "matrix_path": str(Path(args.matrix)),
        "split": matrix["split"],
        "parquet_path": str(parquet_path),
        "parquet_hash": parquet_hash,
        "subtasks": records,
    }
    _json_write(Path(args.output), output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--mab-repo", required=True)
    inspect_parser.add_argument("--parquet", required=True)
    inspect_parser.add_argument("--matrix", required=True)
    inspect_parser.add_argument("--output", required=True)
    inspect_parser.add_argument("--timestamp", default="2026-07-08 00:00:00")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "inspect":
        return _inspect(args)
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
