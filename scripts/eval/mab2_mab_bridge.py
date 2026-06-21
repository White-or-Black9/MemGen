"""Local-only bridge to MemoryAgentBench data, templates, chunking, and metrics."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def _json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _load_mab_modules(repo):
    sys.path.insert(0, str(Path(repo).resolve()))
    from utils.eval_other_utils import chunk_text_into_sentences, post_process
    from utils.templates import get_template

    return chunk_text_into_sentences, post_process, get_template


def resolve_timestamp(pinned_timestamp=None):
    return pinned_timestamp or time.strftime("%Y-%m-%d %H:%M:%S")


def select_match(rows, sub_dataset, match_index=None):
    matches = [
        row for row in rows
        if row.get("metadata", {}).get("source") == sub_dataset
    ]
    if match_index is None:
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one matching context, found {len(matches)}")
        return matches[0], len(matches), 0
    if match_index < 0 or match_index >= len(matches):
        raise RuntimeError(
            f"Requested match_index {match_index} is out of range for {len(matches)} matches"
        )
    return matches[match_index], len(matches), match_index


def prepare(args):
    import pyarrow.parquet as pq
    import tiktoken
    import yaml

    chunker, _, get_template = _load_mab_modules(args.mab_repo)
    dataset_config = yaml.safe_load(Path(args.data_config).read_text(encoding="utf-8"))
    rows = pq.read_table(args.parquet).to_pylist()
    row, matched_count, resolved_match_index = select_match(
        rows,
        args.sub_dataset,
        args.match_index,
    )
    for field in ("context", "questions", "answers", "metadata"):
        if field not in row:
            raise RuntimeError(f"Dataset row is missing required field: {field}")

    chunks = chunker(row["context"], chunk_size=args.chunk_size)
    if not chunks:
        raise RuntimeError("Official chunker returned no chunks")
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    chunk_token_lengths = [len(encoding.encode(chunk)) for chunk in chunks]
    timestamp = resolve_timestamp(args.timestamp)
    memorize_template = get_template(args.sub_dataset, "memorize", "Long_context_agent")
    prompts = [
        memorize_template.format(context=chunk, time_stamp=timestamp)
        for chunk in chunks
    ]
    question = row["questions"][0]
    query_template = get_template(args.sub_dataset, "query", "Long_context_agent")
    query_prompt = query_template.format(question=question)
    answers = row["answers"][0]
    gold_answers = answers if isinstance(answers, list) else [answers]
    context_sha = hashlib.sha256(row["context"].encode("utf-8")).hexdigest()
    _json_write(
        args.output,
        {
            "dataset_config": dataset_config,
            "total_rows": len(rows),
            "matched_rows": matched_count,
            "match_index": resolved_match_index,
            "context_id": f"conflict-resolution-{context_sha[:16]}",
            "query_id": 0,
            "chunks": chunks,
            "chunk_token_lengths": chunk_token_lengths,
            "memorization_prompts": prompts,
            "query_prompt": query_prompt,
            "gold_answers": gold_answers,
            "template": "factconsolidation/long_context_agent",
        },
    )


def score(args):
    _, post_process, _ = _load_mab_modules(args.mab_repo)
    request = json.loads(Path(args.input).read_text(encoding="utf-8"))
    metrics, additional = post_process(
        {"output": request["prediction"]},
        request["gold_answers"],
        request["dataset_config"],
    )
    _json_write(args.output, {"metrics": metrics, "additional": additional})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "score"))
    parser.add_argument("--mab-repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input")
    parser.add_argument("--parquet")
    parser.add_argument("--data-config")
    parser.add_argument("--sub-dataset", default="factconsolidation_sh_6k")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--timestamp")
    parser.add_argument("--match-index", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
