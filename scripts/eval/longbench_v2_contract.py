"""No-model comparison and frozen-bank lifecycle validators."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence


REQUIRED_METHODS = ("disabled_window_fit", "p7", "p7_no_query_retrieval")


class LongBenchV2RunContractError(ValueError):
    """Raised when a comparison violates the frozen evaluation contract."""


def validate_aligned_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        item_id = record.get("item_id")
        method = record.get("method")
        if not item_id or method not in REQUIRED_METHODS:
            raise LongBenchV2RunContractError("record has invalid item_id or method")
        if method in by_item[item_id]:
            raise LongBenchV2RunContractError(f"duplicate method {method} for {item_id}")
        by_item[item_id][method] = record

    for item_id, methods in by_item.items():
        required = {"p7", "p7_no_query_retrieval"}
        if not required.issubset(methods):
            raise LongBenchV2RunContractError(f"missing P7 comparison method for {item_id}")
        p7 = methods["p7"]
        no_query = methods["p7_no_query_retrieval"]
        prompt_versions = {record.get("query_prompt_version") for record in methods.values()}
        if len(prompt_versions) != 1:
            raise LongBenchV2RunContractError(f"query prompt version mismatch for {item_id}")
        if "constrained_choice_v3" in prompt_versions:
            if not all(record.get("constrained_choice_active") is True for record in methods.values()):
                raise LongBenchV2RunContractError(
                    f"constrained choice was not active for every method on {item_id}"
                )
        for field in ("prompt_hash", "question_hash", "choices_hash", "construction_hash"):
            if p7.get(field) != no_query.get(field):
                raise LongBenchV2RunContractError(f"P7 construction mismatch for {item_id}: {field}")
        if p7.get("query_write_count") != 0 or no_query.get("query_write_count") != 0:
            raise LongBenchV2RunContractError(f"query write detected for {item_id}")
        if p7.get("bank_snapshot_changed_after_query") is not False:
            raise LongBenchV2RunContractError(f"P7 snapshot changed for {item_id}")
        if no_query.get("bank_snapshot_changed_after_query") is not False:
            raise LongBenchV2RunContractError(f"no-query snapshot changed for {item_id}")
        if p7.get("post_reset_slot_count") != 0 or no_query.get("post_reset_slot_count") != 0:
            raise LongBenchV2RunContractError(f"post-item reset failed for {item_id}")
        if no_query.get("retrieved_latent_count") != 0:
            raise LongBenchV2RunContractError(f"no-query retrieval was not disabled for {item_id}")
        if p7.get("capacity_class") == "window_fit":
            if "disabled_window_fit" not in methods:
                raise LongBenchV2RunContractError(f"window-fit item lacks Disabled comparator: {item_id}")
        elif "disabled_window_fit" in methods:
            raise LongBenchV2RunContractError(f"over-capacity item has invalid Disabled comparator: {item_id}")

    return {
        "item_count": len(by_item),
        "record_count": len(records),
        "retrieval_positive_items": sum(
            int(methods["p7"].get("retrieved_latent_count", 0) > 0)
            for methods in by_item.values()
        ),
        "contract_valid": True,
    }
