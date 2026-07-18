# Benchmark Trial Order Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate one additional benchmark path after EventQA while preserving the current EventQA-centered paper fallback.

**Architecture:** Use a gated sequence: RULER-QA2 first, then LongBench v2 subset, then MemBench smoke, then LongBench v1 selective sanity check. Each gate produces a self-contained artifact and a decision record before the next benchmark is opened. No task modifies MemGen training, Weaver/Trigger training paths, or the disabled-mode inference path.

**Tech Stack:** Python, pytest, JSON/JSONL artifacts, existing MemGen P7 evaluation scripts, MemoryAgentBench-converted local data where available, Hugging Face datasets only for external benchmark audit/loading after explicit execution approval, tmux for detached GPU runs.

---

## Current Constraints

- EventQA remains the paper main evidence anchor.
- FactConsolidation is dropped under the current frozen-P7 setup.
- DetectiveQA is negative appendix-only diagnostic evidence after comparator alignment.
- New benchmark results must not make the existing paper conditional on success.
- Disabled mode must remain equivalent to the original MemGen path.
- Query-time writes must stay blocked for formal comparisons unless a task explicitly documents a diagnostic deviation.
- Each context/sample must reset the session-local memory bank.
- Batch size remains `1`.
- No large experiment starts during plan implementation without an explicit user execution approval.

## Trial Order

1. `RULER-QA2`: primary next attempt.
2. `LongBench v2 subset`: second attempt only after RULER-QA2 smoke is valid.
3. `MemBench smoke`: exploratory memory-agent compatibility check only.
4. `LongBench v1 selective`: final sanity check only if the earlier options fail or need a traditional long-context baseline.

## File Structure

Create during execution:

- `research_notes/plans/p7_ruler_qa2_trial_plan.md`: RULER-QA2 audit, protocol, and run gates.
- `scripts/eval/ruler_qa2_p7.py`: minimal P7/disabled/no-query runner for local RULER-QA2.
- `tests/test_ruler_qa2_p7.py`: dataset-contract, scorer, reset, and disabled-equivalence tests for the RULER runner.
- `scripts/eval/ruler_qa2_aggregate.py`: aggregation and evidence-gate logic.
- `tests/test_ruler_qa2_aggregate.py`: aggregation tests.
- `research_notes/plans/p7_longbench_v2_subset_trial_plan.md`: LongBench v2 subset audit and gate.
- `scripts/eval/longbench_v2_subset_adapter.py`: subset loader and prompt/scorer contract.
- `tests/test_longbench_v2_subset_adapter.py`: loader and multiple-choice extraction tests.
- `scripts/eval/longbench_v2_subset_p7.py`: optional runner after RULER gate.
- `tests/test_longbench_v2_subset_p7.py`: reset and mode-comparison tests.
- `research_notes/plans/p7_membench_smoke_trial_plan.md`: MemBench compatibility audit and stop criteria.
- `research_notes/plans/p7_longbench_v1_selective_trial_plan.md`: LongBench v1 selective sanity-check protocol.

Modify only after validated results exist:

- `research_notes/EXPERIMENTS.md`: append experiment records with artifact paths and metrics.
- `research_notes/DECISIONS.md`: append go/stop/promotion decisions.
- `research_notes/PROGRESS.md`: append status checkpoints.
- `research_notes/TODO.md`: update remaining benchmark queue.

Do not modify:

- `memgen/model/modeling_memgen.py`
- `memgen/model/latent_memory_bank.py`
- `memgen/model/weaver.py`
- `memgen/model/trigger.py`
- `memgen/trainer/`
- paper main text files unless the user separately requests paper updates.

---

### Task 1: Lock the Benchmark Queue and Baseline Invariants

**Files:**
- Create: `research_notes/plans/p7_benchmark_trial_order_gate.md`

- [ ] **Step 1: Record the frozen trial order**

Create `research_notes/plans/p7_benchmark_trial_order_gate.md` with:

```markdown
# P7 Benchmark Trial Order Gate

Date: 2026-07-10

## Trial order

1. RULER-QA2
2. LongBench v2 subset
3. MemBench smoke
4. LongBench v1 selective

## Non-negotiable invariants

- EventQA remains the main paper evidence anchor.
- Failed new benchmarks do not change the paper mainline.
- Disabled mode must preserve original MemGen behavior.
- Session-local bank reset occurs per context/sample.
- Query-time writes are blocked in formal comparisons.
- Batch size is 1.
- No Weaver/Trigger training path changes.

## Promotion gate

A benchmark can enter an appendix/table-ready state only if:

- the runner has tests for dataset identity, scorer contract, reset boundary, and mode comparability;
- disabled, P7, and P7-no-query-retrieval use the same question set;
- P7 is not worse than disabled on the primary metric in the smoke;
- memory logs show nonzero construction writes and query retrieval attempts for enabled variants;
- failure cases can be explained without invoking hidden protocol mismatch.
```

- [ ] **Step 2: Validate the document contains the queue and invariants**

Run:

```bash
rg -n "RULER-QA2|LongBench v2|MemBench|LongBench v1|Disabled mode|Session-local" research_notes/plans/p7_benchmark_trial_order_gate.md
```

Expected: every required phrase appears.

- [ ] **Step 3: Check markdown diff**

Run:

```bash
git diff --check -- research_notes/plans/p7_benchmark_trial_order_gate.md
git diff -- research_notes/plans/p7_benchmark_trial_order_gate.md
```

Expected: no whitespace errors; only the new gate document is changed.

---

### Task 2: RULER-QA2 Read-Only Dataset and Scorer Audit

**Files:**
- Create: `research_notes/plans/p7_ruler_qa2_trial_plan.md`
- Runtime output: `outputs/mab/ruler_qa2_audit.json`

- [ ] **Step 1: Locate local RULER-QA2 inputs**

Run:

```bash
rg -n "ruler_qa2_421K|ruler_qa2|RULER-QA2" research_notes scripts tests configs outputs -S
```

Expected: local MemoryAgentBench-converted references and prior audit notes are visible.

- [ ] **Step 2: Inspect bridge support without running inference**

Run:

```bash
/home/baishilong/miniconda3/envs/MABench/bin/python scripts/eval/mab2_mab_bridge.py --help
```

Expected: command prints bridge usage and does not load a model.

- [ ] **Step 3: Write the RULER trial protocol**

Create `research_notes/plans/p7_ruler_qa2_trial_plan.md` with:

```markdown
# P7 RULER-QA2 Trial Plan

## Role

RULER-QA2 is the first additional benchmark attempt after EventQA.

## Claim scope

RULER-QA2 may support only a long-context retrieval/reuse stress claim. It does not support a conversational or multi-session memory claim.

## Protocol

- One RULER-QA2 context maps to one MemGen session.
- Construction ingests the context sequentially.
- The memory bank is frozen before answering questions.
- Query-time writes are blocked.
- The bank resets after the context finishes.
- Compare disabled, P7, and P7-no-query-retrieval on the identical query list.

## Metrics

- substring exact match
- total questions
- correct count
- memory write count
- retrieval count
- retrieved latent count
- latency
- peak GPU memory if available from runtime logs

## Smoke gate

The smoke passes only if:

- the scorer is deterministic;
- all modes answer the same question IDs;
- enabled modes have nonzero construction writes;
- P7 does not regress versus disabled on the smoke;
- output JSON includes the exact config and artifact hashes.

## Stop gate

Stop RULER-QA2 if:

- local data is unavailable;
- scoring cannot be reproduced without manual judgment;
- disabled and enabled modes cannot be aligned on the same query set;
- context construction exceeds available GPU memory in smoke.
```

- [ ] **Step 4: Record audit output**

If the bridge can enumerate RULER-QA2 without inference, save:

```json
{
  "schema_version": "ruler-qa2-audit/v1",
  "benchmark": "ruler_qa2_421K",
  "local_data_found": true,
  "requires_model_inference": false,
  "planned_primary_metric": "substring_exact_match",
  "paper_role": "appendix_or_stress_evidence"
}
```

to `outputs/mab/ruler_qa2_audit.json`.

- [ ] **Step 5: Diff-check the audit and protocol**

Run:

```bash
git diff --check -- research_notes/plans/p7_ruler_qa2_trial_plan.md outputs/mab/ruler_qa2_audit.json
```

Expected: no whitespace errors.

---

### Task 3: Implement RULER-QA2 Runner and Aggregator Only After Task 2 Passes

**Files:**
- Create: `scripts/eval/ruler_qa2_p7.py`
- Create: `tests/test_ruler_qa2_p7.py`
- Create: `scripts/eval/ruler_qa2_aggregate.py`
- Create: `tests/test_ruler_qa2_aggregate.py`

- [ ] **Step 1: Write tests for query identity and exact-match scoring**

`tests/test_ruler_qa2_p7.py` must include:

```python
from scripts.eval.ruler_qa2_p7 import substring_exact_match, validate_same_queries


def test_substring_exact_match_accepts_gold_inside_prediction():
    assert substring_exact_match("The answer is alpha-17.", ["alpha-17"]) is True


def test_substring_exact_match_rejects_missing_gold():
    assert substring_exact_match("The answer is alpha-18.", ["alpha-17"]) is False


def test_validate_same_queries_rejects_mismatched_ids():
    disabled = [{"query_id": "q1"}, {"query_id": "q2"}]
    enabled = [{"query_id": "q1"}, {"query_id": "q3"}]
    try:
        validate_same_queries(disabled, enabled)
    except ValueError as exc:
        assert "query identity mismatch" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest -q tests/test_ruler_qa2_p7.py
```

Expected: FAIL because `scripts.eval.ruler_qa2_p7` does not exist.

- [ ] **Step 3: Implement the non-model utility layer**

`scripts/eval/ruler_qa2_p7.py` must provide:

```python
def normalize_text(value: str) -> str:
    return " ".join(value.lower().strip().split())


def substring_exact_match(prediction: str, gold_answers: list[str]) -> bool:
    normalized_prediction = normalize_text(prediction)
    return any(normalize_text(answer) in normalized_prediction for answer in gold_answers)


def validate_same_queries(reference: list[dict], candidate: list[dict]) -> None:
    reference_ids = [item["query_id"] for item in reference]
    candidate_ids = [item["query_id"] for item in candidate]
    if reference_ids != candidate_ids:
        raise ValueError(f"query identity mismatch: {reference_ids} != {candidate_ids}")
```

The model execution code must be added only after these tests pass and after the user approves RULER-QA2 execution.

- [ ] **Step 4: Write aggregation tests**

`tests/test_ruler_qa2_aggregate.py` must include:

```python
from scripts.eval.ruler_qa2_aggregate import aggregate_records


def test_aggregate_records_counts_accuracy_and_memory_usage():
    records = [
        {"correct": True, "memory_write_count": 2, "retrieval_count": 1},
        {"correct": False, "memory_write_count": 2, "retrieval_count": 0},
    ]
    summary = aggregate_records(records)
    assert summary["total"] == 2
    assert summary["correct"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["memory_write_count"] == 4
    assert summary["retrieval_count"] == 1
```

- [ ] **Step 5: Implement aggregation**

`scripts/eval/ruler_qa2_aggregate.py` must provide:

```python
def aggregate_records(records: list[dict]) -> dict:
    total = len(records)
    correct = sum(1 for item in records if item["correct"])
    return {
        "schema_version": "ruler-qa2-aggregate/v1",
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "memory_write_count": sum(int(item.get("memory_write_count", 0)) for item in records),
        "retrieval_count": sum(int(item.get("retrieval_count", 0)) for item in records),
    }
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest -q tests/test_ruler_qa2_p7.py tests/test_ruler_qa2_aggregate.py
```

Expected: PASS.

---

### Task 4: RULER-QA2 Smoke and Full Gate

**Files:**
- Runtime output: `outputs/mab/ruler_qa2_p7_smoke/<timestamp>/`
- Runtime output after smoke passes: `outputs/mab/ruler_qa2_p7_full/<timestamp>/`
- Modify: `research_notes/EXPERIMENTS.md`
- Modify: `research_notes/DECISIONS.md`

- [ ] **Step 1: Check GPU state before smoke**

Run:

```bash
nvidia-smi
```

Expected: identify one usable GPU and avoid occupying GPUs already running user jobs.

- [ ] **Step 2: Launch only a smoke after explicit user approval**

Run the smoke with:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> PYTHONPATH=/mnt/18T/baishilong/MemGen \
/home/baishilong/miniconda3/envs/memgen/bin/python scripts/eval/ruler_qa2_p7.py \
  --benchmark ruler_qa2_421K \
  --modes disabled,p7,p7_no_query_retrieval \
  --max-contexts 1 \
  --max-queries 5 \
  --output-root outputs/mab/ruler_qa2_p7_smoke
```

Expected: JSONL predictions and summary JSON for all three modes.

- [ ] **Step 3: Aggregate smoke**

Run:

```bash
PYTHONPATH=/mnt/18T/baishilong/MemGen \
/home/baishilong/miniconda3/envs/memgen/bin/python scripts/eval/ruler_qa2_aggregate.py \
  --input-root outputs/mab/ruler_qa2_p7_smoke \
  --output outputs/mab/ruler_qa2_p7_smoke/aggregate.json
```

Expected: aggregate includes identical query IDs across modes.

- [ ] **Step 4: Decide full run**

Record a `DECISION`:

- pass: full run allowed as appendix/stress candidate;
- fail: stop RULER-QA2 and do not change paper mainline.

- [ ] **Step 5: Launch full only if smoke passes and user approves**

Run:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> PYTHONPATH=/mnt/18T/baishilong/MemGen \
/home/baishilong/miniconda3/envs/memgen/bin/python scripts/eval/ruler_qa2_p7.py \
  --benchmark ruler_qa2_421K \
  --modes disabled,p7,p7_no_query_retrieval \
  --max-contexts 1 \
  --output-root outputs/mab/ruler_qa2_p7_full
```

Expected: one complete local RULER-QA2 context evaluation.

---

### Task 5: LongBench v2 Subset Audit

**Files:**
- Create: `research_notes/plans/p7_longbench_v2_subset_trial_plan.md`
- Create after approval: `scripts/eval/longbench_v2_subset_adapter.py`
- Create after approval: `tests/test_longbench_v2_subset_adapter.py`

- [ ] **Step 1: Write the subset protocol**

Create `research_notes/plans/p7_longbench_v2_subset_trial_plan.md` with:

```markdown
# P7 LongBench v2 Subset Trial Plan

## Role

LongBench v2 subset is the second benchmark attempt, opened only after RULER-QA2 smoke is valid or explicitly abandoned.

## Selected subsets

Priority order:

1. long-dialogue history understanding
2. multi-document QA
3. single-document QA

Do not include code repository understanding or structured data in the first pass.

## Claim scope

This benchmark can support only long-context robustness and multiple-choice reasoning under long contexts. It does not prove session-level conversational memory by itself.

## Protocol

- One LongBench v2 item maps to one MemGen session.
- Context is construction input.
- Question and choices form the query.
- Answer extraction must return one option label.
- Formal comparison uses disabled, P7, and P7-no-query-retrieval.
- Session reset occurs after each item.

## Smoke gate

- 10 to 20 samples only.
- Multiple-choice extraction accuracy can be computed without manual judging.
- All modes use identical item IDs.
- P7 does not regress clearly against disabled.
```

- [ ] **Step 2: Verify no full benchmark is scheduled**

Run:

```bash
rg -n "10 to 20 samples|Do not include code repository|Session reset" research_notes/plans/p7_longbench_v2_subset_trial_plan.md
```

Expected: bounded subset and reset rules appear.

- [ ] **Step 3: Add adapter tests only after user approves external dataset work**

`tests/test_longbench_v2_subset_adapter.py` must include:

```python
from scripts.eval.longbench_v2_subset_adapter import extract_option_label


def test_extract_option_label_accepts_final_answer_pattern():
    assert extract_option_label("Reasoning... Final answer: C") == "C"


def test_extract_option_label_accepts_plain_letter():
    assert extract_option_label("B") == "B"


def test_extract_option_label_returns_none_for_missing_choice():
    assert extract_option_label("The answer is unknown.") is None
```

- [ ] **Step 4: Implement only the answer extraction utility first**

`scripts/eval/longbench_v2_subset_adapter.py` must include:

```python
import re


def extract_option_label(text: str) -> str | None:
    stripped = text.strip().upper()
    if stripped in {"A", "B", "C", "D"}:
        return stripped
    match = re.search(r"FINAL ANSWER\s*:\s*([ABCD])", stripped)
    if match:
        return match.group(1)
    match = re.search(r"\bANSWER\s*:\s*([ABCD])\b", stripped)
    if match:
        return match.group(1)
    return None
```

- [ ] **Step 5: Run adapter tests**

Run:

```bash
pytest -q tests/test_longbench_v2_subset_adapter.py
```

Expected: PASS.

---

### Task 6: LongBench v2 Smoke Gate

**Files:**
- Create after Task 5 passes: `scripts/eval/longbench_v2_subset_p7.py`
- Create after Task 5 passes: `tests/test_longbench_v2_subset_p7.py`
- Runtime output: `outputs/mab/longbench_v2_subset_p7_smoke/<timestamp>/`

- [ ] **Step 1: Implement loader dry-run before model execution**

The loader dry-run must output:

```json
{
  "schema_version": "longbench-v2-subset-dryrun/v1",
  "selected_domains": ["long-dialogue", "multi-document-qa", "single-document-qa"],
  "max_samples": 20,
  "all_items_have_choices": true,
  "all_items_have_gold_labels": true
}
```

- [ ] **Step 2: Run dry-run**

Run:

```bash
PYTHONPATH=/mnt/18T/baishilong/MemGen \
/home/baishilong/miniconda3/envs/memgen/bin/python scripts/eval/longbench_v2_subset_p7.py \
  --dry-run \
  --max-samples 20 \
  --output outputs/mab/longbench_v2_subset_dryrun.json
```

Expected: dry-run JSON exists and no model is loaded.

- [ ] **Step 3: Launch smoke only if dry-run passes and user approves**

Run:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> PYTHONPATH=/mnt/18T/baishilong/MemGen \
/home/baishilong/miniconda3/envs/memgen/bin/python scripts/eval/longbench_v2_subset_p7.py \
  --modes disabled,p7,p7_no_query_retrieval \
  --max-samples 20 \
  --output-root outputs/mab/longbench_v2_subset_p7_smoke
```

Expected: per-item predictions and mode summaries exist.

- [ ] **Step 4: Decide whether to stop**

Stop LongBench v2 if:

- answer extraction fails on more than 10% of predictions;
- P7 underperforms disabled on the smoke;
- context lengths make smoke unstable;
- memory metrics show no construction writes or no retrieval attempts.

If it passes, keep it appendix/stress only unless the user explicitly requests paper promotion.

---

### Task 7: MemBench Smoke Compatibility Check

**Files:**
- Create: `research_notes/plans/p7_membench_smoke_trial_plan.md`
- Runtime output after approval: `outputs/mab/membench_compatibility_audit.json`

- [ ] **Step 1: Write the compatibility plan**

Create `research_notes/plans/p7_membench_smoke_trial_plan.md` with:

```markdown
# P7 MemBench Smoke Trial Plan

## Role

MemBench is a compatibility smoke only, not a current paper benchmark.

## Stop conditions

Stop immediately if the task requires:

- persistent cross-session memory;
- explicit textual memory objects as the evaluated output;
- environment actions that cannot be represented as MemGen construction/query turns;
- reflective memory updates that require changing Weaver or Trigger training;
- evaluator-specific APIs that would dominate the engineering effort.

## Allowed smoke

Only one minimal factual-memory case may be adapted:

- construction observes facts;
- query asks about stored facts;
- memory resets after the case;
- metric is automatic accuracy;
- disabled and enabled modes use the same query.
```

- [ ] **Step 2: Perform only repository/dataset audit after user approves external access**

Record:

```json
{
  "schema_version": "membench-compatibility-audit/v1",
  "can_run_without_protocol_change": false,
  "requires_text_memory_output": null,
  "requires_cross_session_memory": null,
  "recommended_status": "future_work_unless_minimal_factual_case_is_clean"
}
```

Update the `null` fields with audited facts before any smoke run.

- [ ] **Step 3: Decide**

If MemBench requires explicit memory outputs or cross-session persistence, record `not recommended for current frozen P7` in `research_notes/DECISIONS.md`.

---

### Task 8: LongBench v1 Selective Sanity Check

**Files:**
- Create: `research_notes/plans/p7_longbench_v1_selective_trial_plan.md`

- [ ] **Step 1: Write the selective plan**

Create `research_notes/plans/p7_longbench_v1_selective_trial_plan.md` with:

```markdown
# P7 LongBench v1 Selective Trial Plan

## Role

LongBench v1 is a final sanity check only.

## Selected tasks

Priority order:

1. passage_retrieval_en
2. passage_count
3. hotpotqa or 2wikimqa

Excluded from first pass:

- summarization
- code completion
- full suite

## Claim scope

LongBench v1 can support only traditional long-context baseline discussion. It does not prove memory-bank reasoning.

## Stop gate

Stop if the selected subset duplicates RULER-QA2 evidence or requires heterogeneous metrics that make the result harder to interpret than RULER-QA2 or LongBench v2.
```

- [ ] **Step 2: Do not implement unless prior gates fail**

Verify the plan explicitly says sanity check:

```bash
rg -n "sanity check|Excluded from first pass|does not prove memory-bank reasoning" research_notes/plans/p7_longbench_v1_selective_trial_plan.md
```

Expected: all phrases appear.

---

### Task 9: Notes Update and Paper-Safety Decision

**Files:**
- Modify: `research_notes/EXPERIMENTS.md`
- Modify: `research_notes/DECISIONS.md`
- Modify: `research_notes/PROGRESS.md`
- Modify: `research_notes/TODO.md`

- [ ] **Step 1: Append experiment records after each completed run**

Each experiment record must include:

```markdown
### EXP-YYYYMMDD-XXX: Benchmark trial name

- Type: gated benchmark trial.
- Benchmark:
- Modes:
- Sample count:
- Artifact:
- Primary metric:
- Memory metrics:
- Protocol invariants:
  - disabled-equivalent:
  - same-query-set:
  - query-time-writes-blocked:
  - session-reset:
- Result:
- Interpretation:
- Paper effect:
```

- [ ] **Step 2: Append a decision after each gate**

Each decision must use:

```markdown
### DEC-XXXX: Benchmark gate decision

- Status: accepted / rejected / deferred.
- Decision:
- Evidence:
- Risk:
- Paper effect:
- Next action:
```

- [ ] **Step 3: Preserve paper fallback**

Every decision must include one of:

- `Paper mainline unchanged; EventQA remains the main evidence anchor.`
- `Appendix-only; no main-table promotion.`
- `Internal evidence only; do not cite in paper.`

- [ ] **Step 4: Hygiene check**

Run:

```bash
git diff --check -- research_notes/EXPERIMENTS.md research_notes/DECISIONS.md research_notes/PROGRESS.md research_notes/TODO.md
git status --short
```

Expected: no whitespace errors; changed files match the completed gate.

---

## Execution Gates

### Gate A: Before any code

Required:

- Task 1 completed.
- RULER-QA2 local data and scorer audit completed.
- User confirms execution may proceed.

### Gate B: Before any GPU run

Required:

- focused tests pass;
- `nvidia-smi` checked;
- output directory specified;
- run is smoke-sized;
- no full run is launched before smoke evidence is reviewed.

### Gate C: Before paper/table update

Required:

- benchmark has full aligned run;
- disabled/P7/no-query modes are comparable;
- primary result is non-negative for P7;
- memory metrics show the bank was actually used;
- user explicitly requests paper update.

## Recommended Immediate Next Step

Do Task 1 and Task 2 only. Stop after the RULER-QA2 audit and ask whether to implement the RULER-QA2 runner.
