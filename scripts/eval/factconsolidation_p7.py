"""Paired Disabled/P7 runner contracts for FactConsolidation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = "outputs/mab/factconsolidation_p7"


class FactConsolidationRunContractError(ValueError):
    """Raised when a FactConsolidation run violates lifecycle invariants."""


def load_matrix(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def p7_bank_config(matrix: dict) -> dict:
    config = dict(matrix["p7"])
    config.update(
        {
            "enabled": True,
            "batch_size": 1,
            "update_policy": "thread_update",
            "retrieve_policy": "threshold_topk",
        }
    )
    return config


def validate_batch_size(batch_size: int) -> None:
    if int(batch_size) != 1:
        raise FactConsolidationRunContractError(
            "FactConsolidation runner requires batch_size=1"
        )


def validate_context_start(context_state: dict[str, Any]) -> None:
    if int(context_state.get("initial_slot_count", 0)) != 0:
        raise FactConsolidationRunContractError(
            "each context must start with zero slots"
        )


def validate_disabled_no_bank(run: dict[str, Any]) -> None:
    if run["method"] == "disabled" and bool(run.get("bank_created")):
        raise FactConsolidationRunContractError("Disabled created a bank")


def validate_query_phase_invariants(run: dict[str, Any]) -> None:
    if run["method"] == "disabled":
        return
    if int(run.get("query_write_count", 0)) != 0:
        raise FactConsolidationRunContractError("query write isolation failed")
    if bool(run.get("bank_snapshot_changed_after_query")):
        raise FactConsolidationRunContractError(
            "query snapshot changed during read-only phase"
        )
    if run.get("query_read_only_enforced") is not True:
        raise FactConsolidationRunContractError(
            "query read-only contract was not enforced"
        )


def validate_no_query_retrieval_construction(
    p7_run: dict[str, Any], no_query_run: dict[str, Any]
) -> None:
    fields = (
        "construction_bank_write_count",
        "construction_final_slot_count",
        "construction_turn_count",
    )
    mismatches = [
        field
        for field in fields
        if p7_run.get(field) != no_query_run.get(field)
    ]
    if mismatches:
        raise FactConsolidationRunContractError(
            "construction mismatch between p7 and p7_no_query_retrieval: "
            + ", ".join(mismatches)
        )


def validate_run_invariants(run: dict[str, Any]) -> None:
    validate_disabled_no_bank(run)
    if run["method"] != "disabled":
        validate_query_phase_invariants(run)
    if not bool(run.get("bank_reset_after_context")):
        raise FactConsolidationRunContractError("bank did not reset after context")
    if bool(run.get("cross_context_leakage_detected")):
        raise FactConsolidationRunContractError("cross-context leakage detected after reset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        default="configs/eval/factconsolidation_p7.json",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--method",
        choices=("disabled", "p7", "p7_no_query_retrieval"),
        default="p7",
    )
    parser.add_argument(
        "--subtask",
        default=None,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_batch_size(args.batch_size)
    matrix = load_matrix(args.matrix)
    if args.method != "disabled":
        _ = p7_bank_config(matrix)
    if args.subtask is not None and args.subtask not in matrix["subtasks"]:
        raise FactConsolidationRunContractError(
            f"subtask not present in matrix: {args.subtask}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
