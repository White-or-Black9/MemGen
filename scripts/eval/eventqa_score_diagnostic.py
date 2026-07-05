#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


BUCKET_LABELS = [
    "<0.03",
    "0.03-0.04",
    "0.04-0.05",
    "0.05-0.06",
    "0.06-0.07",
    "0.07-0.08",
    "0.08-0.09",
    ">=0.09",
]

RANGES = {
    "0.085-0.090": (0.085, 0.090, False),
    "0.090-0.095": (0.090, 0.095, False),
    "0.080-0.100": (0.080, 0.100, True),
}

DEFAULT_RUN_ROOTS = [
    "outputs/mab/eventqa_controlled_A_rep1",
    "outputs/mab/eventqa_controlled_A_rep2",
    "outputs/mab/eventqa_controlled_B2_rep1",
    "outputs/mab/eventqa_controlled_B2_rep2",
    "outputs/mab/eventqa_controlled_D_rep1",
    "outputs/mab/eventqa_controlled_D_rep2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline EventQA score diagnostic analyzer")
    parser.add_argument(
        "--run-root",
        action="append",
        dest="run_roots",
        help="Controlled run root to analyze. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/mab/eventqa_score_diagnostics",
        help="Directory for JSON and Markdown outputs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_numeric(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    present = [float(v) for v in values if v is not None]
    missing = len(values) - len(present)
    if not present:
        return {
            "count": 0,
            "missing_count": missing,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "count": len(present),
        "missing_count": missing,
        "min": min(present),
        "p10": quantile(present, 0.10),
        "p25": quantile(present, 0.25),
        "median": quantile(present, 0.50),
        "p75": quantile(present, 0.75),
        "p90": quantile(present, 0.90),
        "max": max(present),
        "mean": statistics.fmean(present),
        "std": statistics.pstdev(present) if len(present) > 1 else 0.0,
    }


def mean_rate(values: Sequence[int]) -> Optional[float]:
    if not values:
        return None
    return statistics.fmean(values)


def histogram(values: Sequence[Any]) -> Dict[str, int]:
    counts = Counter(values)
    return {str(key): counts[key] for key in sorted(counts, key=lambda item: (str(type(item)), item))}


def assign_score_bucket(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    if score < 0.03:
        return "<0.03"
    if score < 0.04:
        return "0.03-0.04"
    if score < 0.05:
        return "0.04-0.05"
    if score < 0.06:
        return "0.05-0.06"
    if score < 0.07:
        return "0.06-0.07"
    if score < 0.08:
        return "0.07-0.08"
    if score < 0.09:
        return "0.08-0.09"
    return ">=0.09"


def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def pearson_correlation(xs: Sequence[Optional[float]], ys: Sequence[Optional[int]]) -> Dict[str, Any]:
    paired = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(paired) < 2:
        return {"count": len(paired), "pearson": None}
    x_values = [pair[0] for pair in paired]
    y_values = [pair[1] for pair in paired]
    mean_x = statistics.fmean(x_values)
    mean_y = statistics.fmean(y_values)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in paired)
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_values))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_values))
    if denom_x == 0.0 or denom_y == 0.0:
        return {"count": len(paired), "pearson": None}
    return {"count": len(paired), "pearson": numerator / (denom_x * denom_y)}


def bool_int(value: Any) -> int:
    return int(bool(value))


def run_key(config: str, repeat: str) -> str:
    return f"{config}/{repeat}"


def discover_run_dir(run_root: Path) -> Path:
    candidates = sorted(path for path in run_root.iterdir() if path.is_dir())
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one run directory under {run_root}, found {len(candidates)}")
    return candidates[0]


def slot_tuple(record: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
    retrieved_indices = record.get("retrieved_indices")
    if retrieved_indices is None:
        return None
    return tuple(int(index) for index in retrieved_indices)


def second_best_score(slots: Sequence[Dict[str, Any]]) -> Optional[float]:
    if len(slots) < 2:
        return None
    ranked = sorted((float(slot["final_score"]) for slot in slots), reverse=True)
    return ranked[1]


def selected_top1_score(slots: Sequence[Dict[str, Any]], retrieved_indices: Sequence[int]) -> Optional[float]:
    if not slots:
        return None
    if retrieved_indices:
        selected = [slot for slot in slots if slot.get("selected_by_topk")]
        if selected:
            return max(float(slot["final_score"]) for slot in selected)
    passing = [slot for slot in slots if slot.get("threshold_passed")]
    if passing:
        return max(float(slot["final_score"]) for slot in passing)
    return None


def build_query_record(
    config: str,
    repeat: str,
    score_record: Dict[str, Any],
    transition_record: Dict[str, Any],
    retrieve_threshold: float,
) -> Dict[str, Any]:
    slots = score_record.get("slots", [])
    retrieved_indices = [int(index) for index in score_record.get("retrieved_indices", [])]
    selected_score = selected_top1_score(slots, retrieved_indices)
    top2_score = second_best_score(slots)
    margin = None if selected_score is None or top2_score is None else selected_score - top2_score
    passing_count = sum(1 for slot in slots if slot.get("threshold_passed"))
    return {
        "config": config,
        "repeat": repeat,
        "context_index": int(score_record["context_index"]),
        "question_index": int(score_record["question_index"]),
        "query_id": score_record.get("query_id"),
        "retrieve_threshold": retrieve_threshold,
        "selected_top1_score": selected_score,
        "selected_top2_score": top2_score,
        "score_margin_top1_top2": margin,
        "candidate_count": len(slots),
        "candidate_count_passing_threshold": passing_count,
        "retrieved_slot_count": len(retrieved_indices),
        "retrieved_latent_count": score_record.get("retrieved_latent_count"),
        "selected_slot_indices": retrieved_indices,
        "selected_slot_tuple": retrieved_indices,
        "query_cosine_to_first": score_record.get("query_cosine_to_first"),
        "query_cosine_to_previous": score_record.get("query_cosine_to_previous"),
        "bank_on_exact_match": bool_int(transition_record.get("bank_on_exact_match")),
        "bank_on_recall": bool_int(transition_record.get("bank_on_recall")),
        "bank_on_format_failure": bool_int(transition_record.get("bank_on_format_failure")),
        "bank_on_chinese_output": bool_int(transition_record.get("bank_on_chinese_output")),
        "helpful_memory": bool_int(transition_record.get("helpful_memory")),
        "harmful_memory": bool_int(transition_record.get("harmful_memory")),
        "format_harm": bool_int(transition_record.get("format_harm")),
    }


def load_run(run_root: Path) -> Dict[str, Any]:
    run_dir = discover_run_dir(run_root)
    parts = run_root.name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected run root naming: {run_root}")
    config = parts[2]
    repeat = parts[3]
    run_config = load_json(run_dir / "run_config.json")
    retrieve_threshold = float(run_config["retrieve_threshold"])
    transitions = load_jsonl(run_dir / "bank_transition_diagnostics.jsonl")
    scores = load_jsonl(run_dir / "score_decomposition.jsonl")
    provenance = load_jsonl(run_dir / "construction_provenance.jsonl")

    transition_index = {
        (int(record["context_index"]), int(record["question_index"])): record for record in transitions
    }
    query_records: List[Dict[str, Any]] = []
    for score_record in scores:
        key = (int(score_record["context_index"]), int(score_record["question_index"]))
        transition_record = transition_index[key]
        query_records.append(
            build_query_record(config, repeat, score_record, transition_record, retrieve_threshold)
        )

    return {
        "config": config,
        "repeat": repeat,
        "run_root": str(run_root),
        "run_dir": str(run_dir),
        "run_config": run_config,
        "query_records": query_records,
        "construction_records": provenance,
    }


def summarize_score_distribution(query_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in query_records:
        grouped[run_key(record["config"], record["repeat"])].append(record)

    runs: Dict[str, Any] = {}
    for key, records in sorted(grouped.items()):
        runs[key] = {
            "query_count": len(records),
            "selected_top1_score": summarize_numeric([record["selected_top1_score"] for record in records]),
            "selected_top2_score": summarize_numeric([record["selected_top2_score"] for record in records]),
            "score_margin_top1_top2": summarize_numeric([record["score_margin_top1_top2"] for record in records]),
            "candidate_count_before_threshold": summarize_numeric([record["candidate_count"] for record in records]),
            "candidate_count_passing_threshold": summarize_numeric(
                [record["candidate_count_passing_threshold"] for record in records]
            ),
            "retrieved_slot_count_distribution": histogram(
                [record["retrieved_slot_count"] for record in records]
            ),
            "retrieved_latent_count_distribution": histogram(
                [record["retrieved_latent_count"] for record in records]
            ),
        }
    return {"runs": runs}


def empty_bucket() -> Dict[str, Any]:
    return {
        "query_count": 0,
        "bank_on_em_rate": None,
        "bank_on_recall_rate": None,
        "format_failure_rate": None,
        "chinese_output_rate": None,
        "helpful_memory_count": 0,
        "helpful_memory_rate": None,
        "harmful_memory_count": 0,
        "harmful_memory_rate": None,
        "format_harm_count": 0,
        "format_harm_rate": None,
    }


def bucket_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return empty_bucket()
    return {
        "query_count": len(records),
        "bank_on_em_rate": mean_rate([record["bank_on_exact_match"] for record in records]),
        "bank_on_recall_rate": mean_rate([record["bank_on_recall"] for record in records]),
        "format_failure_rate": mean_rate([record["bank_on_format_failure"] for record in records]),
        "chinese_output_rate": mean_rate([record["bank_on_chinese_output"] for record in records]),
        "helpful_memory_count": sum(record["helpful_memory"] for record in records),
        "helpful_memory_rate": mean_rate([record["helpful_memory"] for record in records]),
        "harmful_memory_count": sum(record["harmful_memory"] for record in records),
        "harmful_memory_rate": mean_rate([record["harmful_memory"] for record in records]),
        "format_harm_count": sum(record["format_harm"] for record in records),
        "format_harm_rate": mean_rate([record["format_harm"] for record in records]),
    }


def summarize_score_buckets(query_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in query_records:
        grouped[run_key(record["config"], record["repeat"])].append(record)

    runs: Dict[str, Any] = {}
    for key, records in sorted(grouped.items()):
        bucketed: Dict[str, List[Dict[str, Any]]] = {label: [] for label in BUCKET_LABELS}
        missing_count = 0
        for record in records:
            label = assign_score_bucket(record["selected_top1_score"])
            if label is None:
                missing_count += 1
                continue
            bucketed[label].append(record)
        runs[key] = {
            "missing_selected_top1_score_count": missing_count,
            "buckets": {label: bucket_stats(bucketed[label]) for label in BUCKET_LABELS},
        }
    return {"runs": runs}


def in_range(score: Optional[float], lower: float, upper: float, inclusive_upper: bool) -> bool:
    if score is None:
        return False
    if inclusive_upper:
        return lower <= score <= upper
    return lower <= score < upper


def summarize_construction_threshold_sensitivity(
    paired_turns: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> Dict[str, Any]:
    configs: Dict[str, Any] = {}
    for config, repeats in sorted(paired_turns.items()):
        rep_items = {rep: records for rep, records in repeats.items()}
        all_records = [record for records in rep_items.values() for record in records]
        borderline_counts = {}
        for label, (lower, upper, inclusive_upper) in RANGES.items():
            borderline_counts[label] = sum(
                1 for record in all_records if in_range(record.get("best_matched_score"), lower, upper, inclusive_upper)
            )

        paired_divergent = 0
        divergent_examples: List[Dict[str, Any]] = []
        if "rep1" in rep_items and "rep2" in rep_items:
            rep1_index = {
                (int(record["context_index"]), int(record["construction_turn_index"])): record
                for record in rep_items["rep1"]
            }
            rep2_index = {
                (int(record["context_index"]), int(record["construction_turn_index"])): record
                for record in rep_items["rep2"]
            }
            for pair_key in sorted(set(rep1_index) & set(rep2_index)):
                first = rep1_index[pair_key]
                second = rep2_index[pair_key]
                pair_in_band = any(
                    in_range(first.get("best_matched_score"), *RANGES["0.080-0.100"])
                    or in_range(second.get("best_matched_score"), *RANGES["0.080-0.100"])
                    for _ in [0]
                )
                if pair_in_band and first.get("write_action") != second.get("write_action"):
                    paired_divergent += 1
                    if not divergent_examples:
                        divergent_examples.append(
                            {
                                "context_index": pair_key[0],
                                "construction_turn_index": pair_key[1],
                                "rep1_best_matched_score": first.get("best_matched_score"),
                                "rep2_best_matched_score": second.get("best_matched_score"),
                                "rep1_write_action": first.get("write_action"),
                                "rep2_write_action": second.get("write_action"),
                            }
                        )

        total = len(all_records)
        configs[config] = {
            "total_construction_decisions": total,
            "borderline_counts": borderline_counts,
            "borderline_fraction_0.085-0.090": borderline_counts["0.085-0.090"] / total if total else None,
            "borderline_fraction_0.090-0.095": borderline_counts["0.090-0.095"] / total if total else None,
            "borderline_fraction_0.080-0.100": borderline_counts["0.080-0.100"] / total if total else None,
            "paired_divergent_write_actions_in_0.080-0.100": paired_divergent,
            "first_divergent_examples": divergent_examples,
        }
    return {"configs": configs}


def summarize_selected_slot_utility(query_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped_slots: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    routing_by_context: Dict[str, Any] = {}

    for record in query_records:
        key = f"{run_key(record['config'], record['repeat'])}/context_{record['context_index']}"
        grouped_slots[key].append(record)

    slot_summary_by_context: Dict[str, Any] = {}
    for key, records in sorted(grouped_slots.items()):
        tuple_counter = Counter(tuple(record.get("selected_slot_tuple", [])) for record in records)
        most_common_tuple, most_common_count = tuple_counter.most_common(1)[0]
        routing_by_context[key] = {
            "query_count": len(records),
            "unique_selected_slot_tuple_count": len(tuple_counter),
            "selected_slot_entropy_bits": shannon_entropy(tuple_counter),
            "top_selected_slot_tuple": list(most_common_tuple),
            "top_selected_slot_tuple_frequency": most_common_count,
            "fixed_routing": len(tuple_counter) == 1,
        }
        per_tuple = {}
        for slot_tuple_key, count in sorted(tuple_counter.items()):
            slot_records = [
                record for record in records if tuple(record.get("selected_slot_tuple", [])) == slot_tuple_key
            ]
            per_tuple[str(list(slot_tuple_key))] = {
                "count": count,
                "bank_on_em_rate": mean_rate([record["bank_on_exact_match"] for record in slot_records]),
                "bank_on_recall_rate": mean_rate([record["bank_on_recall"] for record in slot_records]),
                "format_failure_rate": mean_rate(
                    [record["bank_on_format_failure"] for record in slot_records]
                ),
                "chinese_output_rate": mean_rate(
                    [record["bank_on_chinese_output"] for record in slot_records]
                ),
                "helpful_memory_rate": mean_rate([record["helpful_memory"] for record in slot_records]),
                "harmful_memory_rate": mean_rate([record["harmful_memory"] for record in slot_records]),
                "format_harm_rate": mean_rate([record["format_harm"] for record in slot_records]),
            }
        slot_summary_by_context[key] = per_tuple

    return {
        "selected_slot_utility_by_context": slot_summary_by_context,
        "routing_by_context": routing_by_context,
    }


def summarize_query_cosines(query_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in query_records:
        key = f"{run_key(record['config'], record['repeat'])}/context_{record['context_index']}"
        grouped[key].append(record)

    result = {}
    for key, records in sorted(grouped.items()):
        first_values = [record["query_cosine_to_first"] for record in records if record["query_cosine_to_first"] is not None]
        previous_values = [
            record["query_cosine_to_previous"] for record in records if record["query_cosine_to_previous"] is not None
        ]
        result[key] = {
            "query_cosine_to_first": summarize_numeric(first_values),
            "query_cosine_to_previous": summarize_numeric(previous_values),
            "all_pair_query_query_cosine_available": False,
        }
    return result


def summarize_correlations(query_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in query_records:
        grouped[record["config"]].append(record)
    grouped["overall"] = list(query_records)

    metrics = [
        "bank_on_exact_match",
        "bank_on_recall",
        "bank_on_format_failure",
        "bank_on_chinese_output",
        "helpful_memory",
        "harmful_memory",
        "format_harm",
    ]

    summary: Dict[str, Any] = {}
    for group_name, records in sorted(grouped.items()):
        xs = [record["selected_top1_score"] for record in records]
        group_summary = {}
        for metric in metrics:
            ys = [record[metric] for record in records]
            group_summary[f"selected_top1_score_vs_{metric}"] = pearson_correlation(xs, ys)
        summary[group_name] = group_summary
    return summary


def flatten_query_records(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [record for run in runs for record in run["query_records"]]


def collect_paired_turns(runs: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    paired: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
    for run in runs:
        paired[run["config"]][run["repeat"]] = run["construction_records"]
    return paired


def healthiest_config(query_records: Sequence[Dict[str, Any]], correlations: Dict[str, Any]) -> Optional[str]:
    config_scores = {}
    for config in sorted({record["config"] for record in query_records}):
        config_records = [record for record in query_records if record["config"] == config]
        helpful = mean_rate([record["helpful_memory"] for record in config_records]) or 0.0
        harmful = mean_rate([record["harmful_memory"] for record in config_records]) or 0.0
        em = mean_rate([record["bank_on_exact_match"] for record in config_records]) or 0.0
        recall = mean_rate([record["bank_on_recall"] for record in config_records]) or 0.0
        corr = correlations.get(config, {}).get("selected_top1_score_vs_helpful_memory", {}).get("pearson") or 0.0
        config_scores[config] = em + recall + helpful + corr - harmful
    if not config_scores:
        return None
    return max(config_scores, key=config_scores.get)


def summarize_overall_evidence(
    query_records: Sequence[Dict[str, Any]],
    distribution_summary: Dict[str, Any],
    bucket_summary: Dict[str, Any],
    selected_slot_summary: Dict[str, Any],
    correlations: Dict[str, Any],
) -> Dict[str, Any]:
    top1_values = [record["selected_top1_score"] for record in query_records if record["selected_top1_score"] is not None]
    retrieve_threshold_filter_rate = 1.0 - (
        sum(record["candidate_count_passing_threshold"] > 0 for record in query_records) / len(query_records)
        if query_records
        else 0.0
    )
    dense_count = sum(1 for value in top1_values if 0.08 <= value <= 0.10)
    high_score_records = [record for record in query_records if record["selected_top1_score"] is not None and record["selected_top1_score"] >= 0.08]
    top2_available = [record for record in query_records if record["selected_top2_score"] is not None]
    routing_summaries = list(selected_slot_summary["routing_by_context"].values())
    fixed_count = sum(1 for routing in routing_summaries if routing["fixed_routing"])
    return {
        "top1_score_range": {"min": min(top1_values) if top1_values else None, "max": max(top1_values) if top1_values else None},
        "top1_score_dense_band_fraction": dense_count / len(top1_values) if top1_values else None,
        "retrieve_threshold_filter_rate": retrieve_threshold_filter_rate,
        "high_score_harm_rate": mean_rate([record["harmful_memory"] for record in high_score_records]) if high_score_records else None,
        "top2_available_fraction": len(top2_available) / len(query_records) if query_records else None,
        "dominant_config": healthiest_config(query_records, correlations),
        "retrieval_slot_pressure_fraction": fixed_count / len(routing_summaries) if routing_summaries else None,
        "distribution_summary": distribution_summary,
        "bucket_summary": bucket_summary,
    }


def recommend_parameter_ranges(
    overall_summary: Dict[str, Any],
    correlation_summary: Dict[str, Any],
    construction_summary: Dict[str, Any],
) -> Dict[str, Any]:
    em_corr = (
        correlation_summary.get("overall", {})
        .get("selected_top1_score_vs_bank_on_exact_match", {})
        .get("pearson")
    )
    recall_corr = (
        correlation_summary.get("overall", {})
        .get("selected_top1_score_vs_bank_on_recall", {})
        .get("pearson")
    )
    format_corr = (
        correlation_summary.get("overall", {})
        .get("selected_top1_score_vs_bank_on_format_failure", {})
        .get("pearson")
    )
    dense_band_fraction = overall_summary.get("top1_score_dense_band_fraction") or 0.0
    filter_rate = overall_summary.get("retrieve_threshold_filter_rate") or 0.0
    high_score_harm = overall_summary.get("high_score_harm_rate") or 0.0

    retrieve_worth = [0.02, 0.04, 0.05]
    if filter_rate < 0.05:
        retrieve_worth = [0.02, 0.03, 0.04, 0.05]
    retrieve_not_worth = [0.005, 0.01]
    if dense_band_fraction < 0.20:
        retrieve_not_worth.append(0.06)

    update_avoid = [0.09] if dense_band_fraction >= 0.25 else []
    if high_score_harm >= 0.15:
        update_avoid.append(0.085)

    return {
        "A": {
            "immediate_next_experiment": {
                "priority": "high",
                "reason": "Current score ranges are narrow, retrieve filtering is weak, and the 0.08-0.10 band is dense enough that fixed thresholds are brittle.",
                "expected_effect": "Separates threshold effects from routing instability with the smallest informative sweep.",
                "risk": "May still mix score quality and construction nondeterminism if run on more than the minimum diagnostic slice.",
                "smallest_useful_experiment": "Run the next diagnostic on the existing controlled setup with `retrieve_threshold` in {0.02, 0.03, 0.04, 0.05} and `update_threshold` in {0.095, 0.10} on the smallest bounded context set.",
            }
        },
        "B": {
            "secondary_parameter_sweep": {
                "retrieve_threshold": {
                    "priority": "high",
                    "worth_testing": retrieve_worth,
                    "not_worth_testing_yet": retrieve_not_worth,
                    "reason": "The current 0.03 gate admits almost everything, so lower values are unlikely to help and moderate increases are the meaningful check.",
                    "expected_effect": "Tests whether pruning weak matches improves utility and routing specificity.",
                    "risk": "Too-high thresholds can collapse retrieval and reduce recall.",
                    "smallest_useful_experiment": "A tiny bounded sweep over `retrieve_threshold` only, holding all other parameters fixed.",
                },
                "update_threshold": {
                    "priority": "high",
                    "worth_testing": [0.095, 0.10],
                    "keep_as_reference": [0.085, 0.09],
                    "reason": "Scores cluster near 0.09, making current write decisions sensitive to small score drift.",
                    "expected_effect": "Moves construction decisions out of the densest borderline region.",
                    "risk": "Raising too far can suppress useful updates and underfill the bank.",
                    "smallest_useful_experiment": "One bounded construction-focused repeat at `0.095`, then `0.10` if the 0.095 run remains borderline-heavy.",
                },
                "max_slots": {
                    "priority": "medium",
                    "worth_testing": [16, 20],
                    "not_worth_testing_yet": [12, 24],
                    "reason": "Evidence should first distinguish score quality from capacity; 16 is the first practical check if 8 looks saturated.",
                    "expected_effect": "Shows whether slot pressure is hurting routing without jumping to an oversized bank.",
                    "risk": "Larger banks can mask poor scoring by increasing nearly-tied candidates.",
                    "smallest_useful_experiment": "Compare 8 versus 16 under the same thresholds after threshold behavior is clearer.",
                },
                "top_k": {
                    "priority": "medium",
                    "worth_testing": [1, 2],
                    "avoid": [4],
                    "reason": "Top-k=4 would likely increase mixed and potentially harmful retrievals before score utility is understood.",
                    "expected_effect": "Checks whether a second memory helps without making routing too diffuse.",
                    "risk": "Top-k=2 can still amplify format harm if scores are not utility-aligned.",
                    "smallest_useful_experiment": "Hold thresholds fixed and compare `top_k=1` versus `top_k=2` on the same bounded artifacts path.",
                },
                "decay_alpha": {
                    "priority": "medium",
                    "worth_testing": [0.0, 0.02, 0.05],
                    "not_worth_testing_yet": [0.08],
                    "reason": "Need to check whether recency is dominating routing before trying stronger decay.",
                    "expected_effect": "Separates semantic similarity from recency bias in slot selection.",
                    "risk": "Removing decay entirely can overfavor stale generic slots.",
                    "smallest_useful_experiment": "Compare `decay_alpha=0.0` and `0.02` against the current 0.05 on a one-context diagnostic.",
                },
            }
        },
        "C": {
            "parameters_to_avoid_for_now": {
                "update_threshold": {
                    "avoid": sorted(set(update_avoid)),
                    "reason": "These values sit inside the most brittle decision band or are too close to it to resolve the ambiguity.",
                },
                "retrieve_threshold": {
                    "avoid": [0.005, 0.01],
                    "reason": "Lowering the already-weak gate is unlikely to improve discrimination.",
                },
                "top_k": {
                    "avoid": [4],
                    "reason": "Too many retrieved slots would blur utility attribution before the score function is fixed.",
                },
            }
        },
        "D": {
            "mechanism_changes_to_consider": {
                "options": [
                    "margin gate: top1 - top2 > margin",
                    "update hysteresis: low/high thresholds with gray-zone skip",
                    "question-only query representation",
                    "fp32 scoring only",
                    "deterministic EventQA mode",
                ],
                "priority": "low" if (em_corr or 0.0) > 0.15 and (recall_corr or 0.0) > 0.15 else "medium",
                "reason": "If score correlations stay weak or high-score harm remains visible, threshold tuning alone is unlikely to fix routing utility.",
                "expected_effect": "Adds utility-aware or numerically-stable guards once plain threshold diagnostics are exhausted.",
                "risk": "Mechanism changes broaden scope and make attribution harder.",
                "smallest_useful_experiment": "Introduce one guard at a time only after the bounded threshold diagnostics are complete.",
            }
        },
        "evidence_snapshot": {
            "score_vs_em_pearson": em_corr,
            "score_vs_recall_pearson": recall_corr,
            "score_vs_format_failure_pearson": format_corr,
            "dense_band_fraction_0.08_0.10": dense_band_fraction,
            "high_score_harm_rate": high_score_harm,
            "dominant_config": overall_summary.get("dominant_config"),
        },
    }


def build_report(
    overall_summary: Dict[str, Any],
    score_distribution_summary: Dict[str, Any],
    bucket_summary: Dict[str, Any],
    construction_summary: Dict[str, Any],
    selected_slot_summary: Dict[str, Any],
    correlation_summary: Dict[str, Any],
    query_cosine_summary: Dict[str, Any],
    recommendations: Dict[str, Any],
) -> str:
    healthiest = overall_summary.get("dominant_config")
    em_corr = correlation_summary["overall"]["selected_top1_score_vs_bank_on_exact_match"]["pearson"]
    recall_corr = correlation_summary["overall"]["selected_top1_score_vs_bank_on_recall"]["pearson"]
    format_corr = correlation_summary["overall"]["selected_top1_score_vs_bank_on_format_failure"]["pearson"]
    lines = [
        "# EventQA Score Diagnostic Report",
        "",
        "## Direct Answers",
        "",
        f"- Higher selected score vs EM correlation: {em_corr}",
        f"- Higher selected score vs recall correlation: {recall_corr}",
        f"- Higher selected score vs format failure correlation: {format_corr}",
        f"- Retrieve threshold filter rate at current setting: {overall_summary['retrieve_threshold_filter_rate']}",
        f"- Fraction of selected top-1 scores in 0.08-0.10: {overall_summary['top1_score_dense_band_fraction']}",
        f"- Harm rate among >=0.08 selected scores: {overall_summary['high_score_harm_rate']}",
        f"- Healthiest observed config: {healthiest}",
        "",
        "## Interpretation",
        "",
        "- Higher score does not automatically imply better utility unless EM/recall correlations are clearly positive and harmful-memory rates stay low.",
        "- If format-failure correlation is positive, higher scores are not acting as a clean quality signal.",
        "- Weak retrieve-threshold filtering implies `retrieve_threshold=0.03` is mostly permissive rather than selective.",
        "- A dense 0.08-0.10 construction band means `update_threshold=0.09` is borderline and likely brittle.",
        "",
        "## Routing",
        "",
    ]
    for key, routing in sorted(selected_slot_summary["routing_by_context"].items()):
        lines.append(
            f"- {key}: unique_tuples={routing['unique_selected_slot_tuple_count']}, entropy_bits={routing['selected_slot_entropy_bits']}, fixed_routing={routing['fixed_routing']}, top_tuple={routing['top_selected_slot_tuple']}, top_freq={routing['top_selected_slot_tuple_frequency']}"
        )
    lines.extend(
        [
            "",
            "## Construction Threshold Sensitivity",
            "",
        ]
    )
    for config, summary in sorted(construction_summary["configs"].items()):
        lines.append(
            f"- {config}: total={summary['total_construction_decisions']}, band_0.085_0.090={summary['borderline_counts']['0.085-0.090']}, band_0.090_0.095={summary['borderline_counts']['0.090-0.095']}, band_0.080_0.100={summary['borderline_counts']['0.080-0.100']}, paired_divergent_write_actions={summary['paired_divergent_write_actions_in_0.080-0.100']}"
        )
    lines.extend(
        [
            "",
            "## Query Cosine Availability",
            "",
            "- `query_cosine_to_first` and `query_cosine_to_previous` are available from the saved artifacts.",
            "- Full all-pair query-query cosine is not present in the artifacts and is reported as unavailable rather than zero.",
            "",
            "## Recommendation",
            "",
            f"- Immediate next experiment: {recommendations['A']['immediate_next_experiment']['smallest_useful_experiment']}",
            f"- Secondary sweep focus: retrieve thresholds {recommendations['B']['secondary_parameter_sweep']['retrieve_threshold']['worth_testing']} and update thresholds {recommendations['B']['secondary_parameter_sweep']['update_threshold']['worth_testing']}.",
            f"- Avoid for now: update thresholds {recommendations['C']['parameters_to_avoid_for_now']['update_threshold']['avoid']} and top_k {recommendations['C']['parameters_to_avoid_for_now']['top_k']['avoid']}.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def analyze(run_roots: Sequence[str], output_dir: str) -> Dict[str, Any]:
    runs = [load_run(Path(root)) for root in run_roots]
    query_records = flatten_query_records(runs)
    score_distribution_summary = summarize_score_distribution(query_records)
    score_bucket_summary = summarize_score_buckets(query_records)
    construction_threshold_sensitivity = summarize_construction_threshold_sensitivity(collect_paired_turns(runs))
    selected_slot_utility_summary = summarize_selected_slot_utility(query_records)
    query_cosine_summary = summarize_query_cosines(query_records)
    score_utility_correlations = summarize_correlations(query_records)
    overall_evidence = summarize_overall_evidence(
        query_records,
        score_distribution_summary,
        score_bucket_summary,
        selected_slot_utility_summary,
        score_utility_correlations,
    )
    parameter_recommendations = recommend_parameter_ranges(
        overall_evidence,
        score_utility_correlations,
        construction_threshold_sensitivity,
    )
    report = build_report(
        overall_evidence,
        score_distribution_summary,
        score_bucket_summary,
        construction_threshold_sensitivity,
        selected_slot_utility_summary,
        score_utility_correlations,
        query_cosine_summary,
        parameter_recommendations,
    )

    output = Path(output_dir)
    write_json(output / "score_distribution_summary.json", score_distribution_summary)
    write_json(output / "score_bucket_summary.json", score_bucket_summary)
    write_json(output / "construction_threshold_sensitivity.json", construction_threshold_sensitivity)
    write_json(output / "selected_slot_utility_summary.json", selected_slot_utility_summary)
    write_json(output / "parameter_range_recommendations.json", parameter_recommendations)
    write_json(output / "score_utility_correlations.json", score_utility_correlations)
    write_json(output / "query_representation_summary.json", query_cosine_summary)
    (output / "score_diagnostic_report.md").write_text(report, encoding="utf-8")
    return {
        "score_distribution_summary": score_distribution_summary,
        "score_bucket_summary": score_bucket_summary,
        "construction_threshold_sensitivity": construction_threshold_sensitivity,
        "selected_slot_utility_summary": selected_slot_utility_summary,
        "query_cosine_summary": query_cosine_summary,
        "score_utility_correlations": score_utility_correlations,
        "parameter_recommendations": parameter_recommendations,
        "report": report,
    }


def main() -> None:
    args = parse_args()
    run_roots = args.run_roots or DEFAULT_RUN_ROOTS
    result = analyze(run_roots=run_roots, output_dir=args.output_dir)
    print(result["report"], end="")


if __name__ == "__main__":
    main()
