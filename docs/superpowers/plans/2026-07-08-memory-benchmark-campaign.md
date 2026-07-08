# Memory Benchmark Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FactConsolidation and bounded stress-test evidence to the frozen P7 paper package without making the existing EventQA paper conditional on new positive results.

**Architecture:** Preserve the current EventQA manuscript and canonical aggregates as an immutable campaign baseline. Add benchmark-specific code only under `scripts/eval/`, reuse the current P7 Weaver-space runner path, emit self-validating JSON/JSONL artifacts, and promote new results into the paper only through an explicit evidence gate. FactConsolidation is the only new main-table candidate; DetectiveQA remains stress evidence and BABILong is conditional.

**Tech Stack:** Python 3.10, PyTorch, PyArrow, existing MemGen and MemoryAgentBench environments, `unittest`/`pytest`, JSON/JSONL artifacts, Markdown paper sources, tmux for detached GPU runs.

---

## File Structure

New files:

- `scripts/eval/memory_benchmark_campaign_baseline.py`: freeze and validate the pre-campaign EventQA paper baseline.
- `tests/test_memory_benchmark_campaign_baseline.py`: baseline manifest and no-change validation.
- `configs/eval/factconsolidation_p7.json`: accepted benchmark matrix and frozen P7 configuration.
- `scripts/eval/factconsolidation_adapter.py`: normalize MAB SH/MH contexts, queries, answers, and official config metadata.
- `tests/test_factconsolidation_adapter.py`: adapter, ordering, and scorer-contract tests.
- `scripts/eval/factconsolidation_p7.py`: paired Disabled/P7/no-query evaluation runner.
- `tests/test_factconsolidation_p7.py`: lifecycle, prompt, bank-space, and query-isolation tests.
- `scripts/eval/factconsolidation_aggregate.py`: strict per-run and cross-repeat aggregation.
- `tests/test_factconsolidation_aggregate.py`: schema, identity, config, and metric validation tests.
- `scripts/eval/factconsolidation_campaign_decision.py`: main-table/mechanism/internal promotion decision.
- `tests/test_factconsolidation_campaign_decision.py`: evidence-gate tests.
- `scripts/eval/detectiveqa_stress_aggregate.py`: aggregate existing/new compressed DetectiveQA stress runs.
- `tests/test_detectiveqa_stress_aggregate.py`: capacity and invalid-full-history checks.
- `scripts/eval/memory_benchmark_paper_package.py`: combine EventQA baseline with accepted additive evidence.
- `tests/test_memory_benchmark_paper_package.py`: fallback and additive-promotion tests.

Modified files only after evidence gates pass:

- `paper/draft_v0.md`: add accepted benchmark results; unchanged on fallback.
- `paper/main_table_blueprint.md`: add FactConsolidation only if promoted.
- `paper/experiment_gap_to_table_mapping.md`: record campaign outcome.
- `research_notes/EXPERIMENTS.md`: append validated experiment records.
- `research_notes/PROGRESS.md`: append phase checkpoints.
- `research_notes/DECISIONS.md`: record promotion or fallback decision.

The following are not modified by this campaign:

- `memgen/model/modeling_memgen.py`
- `memgen/model/latent_memory_bank.py`
- `memgen/model/weaver.py`
- `memgen/model/trigger.py`
- `memgen/trainer/`
- training scripts and training configs

---

### Task 1: Freeze the EventQA Paper Baseline

**Files:**
- Create: `scripts/eval/memory_benchmark_campaign_baseline.py`
- Create: `tests/test_memory_benchmark_campaign_baseline.py`
- Create at runtime: `outputs/mab/memory_benchmark_campaign_baseline.json`

- [ ] **Step 1: Write the failing baseline-manifest test**

```python
def test_build_manifest_hashes_required_eventqa_and_paper_files(tmp_path):
    paper = tmp_path / "draft.md"
    aggregate = tmp_path / "eventqa.json"
    paper.write_text("EventQA baseline", encoding="utf-8")
    aggregate.write_text('{"schema_version":"eventqa-final-table-package/v1"}', encoding="utf-8")
    manifest = build_manifest(
        repo_root=tmp_path,
        required_paths=[paper, aggregate],
        accepted_commit="14767eb",
    )
    assert manifest["accepted_commit"] == "14767eb"
    assert manifest["files"]["draft.md"]["sha256"]
    assert manifest["files"]["eventqa.json"]["sha256"]
```

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `pytest -q tests/test_memory_benchmark_campaign_baseline.py`

Expected: FAIL because `memory_benchmark_campaign_baseline` does not exist.

- [ ] **Step 3: Implement baseline creation and validation**

Implement:

```python
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def build_manifest(repo_root: Path, required_paths: list[Path], accepted_commit: str) -> dict:
    return {
        "schema_version": "memory-benchmark-campaign-baseline/v1",
        "accepted_commit": accepted_commit,
        "files": {
            str(path.relative_to(repo_root)): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in required_paths
        },
    }

def validate_manifest(manifest: dict, repo_root: Path) -> None:
    for relative, expected in manifest["files"].items():
        path = repo_root / relative
        if not path.is_file():
            raise ValueError(f"missing baseline file: {relative}")
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"baseline hash mismatch: {relative}")
```

The CLI must hash:

- `paper/draft_v0.md`
- `paper/outline.md`
- `paper/main_table_blueprint.md`
- `research_notes/PAPER_SCOPE.md`
- `outputs/mab/eventqa_final_comparison_package.json`
- `outputs/mab/eventqa_paper_artifact_manifest_sha256.txt`

It must record `git rev-parse HEAD`, `git status --short`, and the protected EventQA package schema without editing any input.

- [ ] **Step 4: Run tests and create the read-only baseline manifest**

Run:

```bash
pytest -q tests/test_memory_benchmark_campaign_baseline.py
python scripts/eval/memory_benchmark_campaign_baseline.py \
  --repo-root /mnt/18T/baishilong/MemGen \
  --accepted-paper-commit 14767eb \
  --output outputs/mab/memory_benchmark_campaign_baseline.json
```

Expected: tests PASS; manifest validation reports all required files present and hashed.

- [ ] **Step 5: Verify scope and commit**

Run:

```bash
git diff --check -- scripts/eval/memory_benchmark_campaign_baseline.py tests/test_memory_benchmark_campaign_baseline.py
git status --short
git add scripts/eval/memory_benchmark_campaign_baseline.py tests/test_memory_benchmark_campaign_baseline.py
git commit -m "test: freeze EventQA campaign baseline"
```

Do not stage `paper/references.bib` or `paper/introduction.md`.

---

### Task 2: Audit and Lock the FactConsolidation Evaluation Matrix

**Files:**
- Create: `configs/eval/factconsolidation_p7.json`
- Create at runtime: `outputs/mab/factconsolidation_dataset_audit.json`
- Modify: `research_notes/PLANS.md`

- [ ] **Step 1: Run a read-only dataset inventory**

Use the local parquet and official configs to count contexts, questions, answers, chunk counts, and estimated prompt sizes for:

- `factconsolidation_sh_6k`
- `factconsolidation_mh_6k`
- `factconsolidation_sh_32k`
- `factconsolidation_mh_32k`
- `factconsolidation_sh_64k`
- `factconsolidation_mh_64k`

Run:

```bash
/home/baishilong/miniconda3/envs/MABench/bin/python scripts/eval/mab2_mab_bridge.py prepare \
  --mab-repo /mnt/18T/baishilong/benchmarks/MemoryAgentBench \
  --parquet /mnt/18T/baishilong/datasets/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet \
  --data-config /mnt/18T/baishilong/benchmarks/MemoryAgentBench/configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml \
  --sub-dataset factconsolidation_sh_6k \
  --output /tmp/factconsolidation_sh_6k_audit.json \
  --match-index 0
```

Repeat through a small audit script for the six fixed configs; do not run model inference.

- [ ] **Step 2: Write the frozen config matrix**

`configs/eval/factconsolidation_p7.json` must contain:

```json
{
  "schema_version": "factconsolidation-p7-matrix/v1",
  "split": "Conflict_Resolution",
  "subtasks": [
    "factconsolidation_sh_6k",
    "factconsolidation_mh_6k",
    "factconsolidation_sh_32k",
    "factconsolidation_mh_32k",
    "factconsolidation_sh_64k",
    "factconsolidation_mh_64k"
  ],
  "smoke_subtasks": ["factconsolidation_sh_6k", "factconsolidation_mh_6k"],
  "methods": ["disabled", "p7", "p7_no_query_retrieval"],
  "p7": {
    "retrieve_threshold": 0.05,
    "update_threshold": 0.10,
    "max_slots": 16,
    "top_k": 2,
    "decay_alpha": 0.05,
    "storage_space": "weaver",
    "query_phase": "read_only"
  },
  "primary_metric": "substring_exact_match"
}
```

- [ ] **Step 3: Record the scale decision inputs**

Append a bounded plan entry to `research_notes/PLANS.md` stating the actual independent-context and query counts. Do not classify FactConsolidation as a main-table benchmark yet.

- [ ] **Step 4: Validate and commit**

Run:

```bash
python -m json.tool configs/eval/factconsolidation_p7.json >/dev/null
git diff --check -- configs/eval/factconsolidation_p7.json research_notes/PLANS.md
git add configs/eval/factconsolidation_p7.json research_notes/PLANS.md
git commit -m "docs: lock FactConsolidation evaluation matrix"
```

---

### Task 3: Implement the FactConsolidation Adapter

**Files:**
- Create: `scripts/eval/factconsolidation_adapter.py`
- Create: `tests/test_factconsolidation_adapter.py`
- Reuse: `scripts/eval/mab2_mab_bridge.py`

- [ ] **Step 1: Write failing adapter tests**

Cover:

```python
def test_normalize_preserves_chunk_and_query_order():
    row = {"context": "first. second.", "questions": ["q1", "q2"], "answers": [["a1"], ["a2"]], "metadata": {"source": "factconsolidation_sh_6k"}}
    payload = normalize_context(row, subtask="factconsolidation_sh_6k", chunker=lambda text, chunk_size: ["first.", "second."], templates=fake_templates())
    assert payload["chunks"] == ["first.", "second."]
    assert [query["question"] for query in payload["queries"]] == ["q1", "q2"]

def test_invalid_source_name_fails_loudly():
    row = {"context": "x", "questions": ["q"], "answers": [["a"]], "metadata": {"source": "wrong"}}
    with pytest.raises(ValueError, match="source mismatch"):
        normalize_context(row, subtask="factconsolidation_sh_6k", chunker=lambda text, chunk_size: [text], templates=fake_templates())
```

The normalized payload schema must include `subtask`, `context_id`, ordered `chunks`, ordered `queries`, `gold_answers`, `qa_pair_ids`, official config hash, and source parquet hash.

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest -q tests/test_factconsolidation_adapter.py`

Expected: FAIL because the adapter module is absent.

- [ ] **Step 3: Implement the minimal adapter**

Implement:

```python
SUPPORTED_SUBTASKS = {
    "factconsolidation_sh_6k", "factconsolidation_mh_6k",
    "factconsolidation_sh_32k", "factconsolidation_mh_32k",
    "factconsolidation_sh_64k", "factconsolidation_mh_64k",
}

def load_rows(parquet_path: Path, subtask: str) -> list[dict]:
    if subtask not in SUPPORTED_SUBTASKS:
        raise ValueError(f"unsupported FactConsolidation subtask: {subtask}")
    rows = pyarrow.parquet.read_table(parquet_path).to_pylist()
    return [row for row in rows if row.get("metadata", {}).get("source") == subtask]

def score_prediction(prediction: str, gold_answers: list[str], dataset_config: dict, post_process) -> dict:
    metrics, additional = post_process({"output": prediction}, gold_answers, dataset_config)
    return {"metrics": metrics, "additional": additional}
```

Use official MemoryAgentBench chunking, templates, and `post_process`; do not reproduce their scoring logic locally.

- [ ] **Step 4: Run adapter tests and a no-model real-data smoke**

Run:

```bash
pytest -q tests/test_factconsolidation_adapter.py
/home/baishilong/miniconda3/envs/MABench/bin/python scripts/eval/factconsolidation_adapter.py inspect \
  --mab-repo /mnt/18T/baishilong/benchmarks/MemoryAgentBench \
  --parquet /mnt/18T/baishilong/datasets/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet \
  --matrix configs/eval/factconsolidation_p7.json \
  --output outputs/mab/factconsolidation_dataset_audit.json
```

Expected: all six subtask records validate; no model or GPU is loaded.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/factconsolidation_adapter.py tests/test_factconsolidation_adapter.py
git commit -m "feat: add FactConsolidation adapter"
```

---

### Task 4: Implement the Paired Frozen-P7 Runner

**Files:**
- Create: `scripts/eval/factconsolidation_p7.py`
- Create: `tests/test_factconsolidation_p7.py`
- Reuse: `scripts/eval/mab6b_weaver_space_bank_eventqa_65536_n5.py`
- Reuse: `scripts/eval/mab6b_weaver_space_bank_detectiveqa_n10.py`

- [ ] **Step 1: Write failing lifecycle tests**

Required tests:

```python
REQUIRED_LIFECYCLE_TESTS = {
    "test_disabled_passes_no_bank_to_generate",
    "test_p7_uses_weaver_storage_and_weaver_retrieval_query",
    "test_each_context_starts_with_zero_slots",
    "test_query_phase_blocks_write_and_preserves_snapshot",
    "test_no_query_retrieval_keeps_identical_construction",
    "test_batch_size_above_one_is_rejected",
    "test_exception_path_resets_bank",
}
```

- [ ] **Step 2: Run tests and confirm failure**

Run: `pytest -q tests/test_factconsolidation_p7.py`

- [ ] **Step 3: Implement runner interfaces**

Implement:

```python
def p7_bank_config(matrix: dict) -> dict:
    config = dict(matrix["p7"])
    config.update({"enabled": True, "batch_size": 1, "update_policy": "thread_update", "retrieve_policy": "threshold_topk"})
    return config

def validate_run_invariants(run: dict) -> None:
    if run["method"] == "disabled" and run["bank_created"]:
        raise ValueError("Disabled created a bank")
    if run["method"] != "disabled" and run["query_write_count"] != 0:
        raise ValueError("query write isolation failed")
    if run["method"] != "disabled" and run["pre_query_bank_sha256"] != run["post_query_bank_sha256"]:
        raise ValueError("query changed frozen bank")
    if run["post_reset_slot_count"] != 0:
        raise ValueError("bank reset failed")
```

The runner must:

- construct the bank once from ordered context chunks;
- snapshot it before questions;
- restore the same snapshot for every question;
- block query writes;
- reset after each context and on exceptions;
- emit construction and query timing separately;
- record slot provenance, retrieval indices/scores, insert/matched-replace/eviction counts;
- reject prompt over-capacity rather than truncate silently.

- [ ] **Step 4: Run unit tests**

Run:

```bash
pytest -q tests/test_factconsolidation_p7.py tests/test_latent_memory_bank.py tests/test_latent_memory_bank_integration.py
```

Expected: all PASS and no training files changed.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/factconsolidation_p7.py tests/test_factconsolidation_p7.py
git commit -m "feat: add frozen-P7 FactConsolidation runner"
```

---

### Task 5: Execute SH/MH 6K Smoke and Make the First Gate Decision

**Files:**
- Runtime artifacts: `outputs/mab/factconsolidation_p7_smoke/`
- Modify after validation: `research_notes/EXPERIMENTS.md`
- Modify after validation: `research_notes/PROGRESS.md`

- [ ] **Step 1: Inspect GPU occupancy and launch detached serialized smokes**

Use one available A6000 and separate method processes. Launch through tmux; do not run methods concurrently on the same GPU.

Run pattern:

```bash
tmux new-session -d -s fc_p7_smoke \
  "cd /mnt/18T/baishilong/MemGen && CUDA_VISIBLE_DEVICES=0 /home/baishilong/miniconda3/envs/memgen/bin/python scripts/eval/factconsolidation_p7.py --matrix configs/eval/factconsolidation_p7.json --subtask factconsolidation_sh_6k --methods disabled,p7,p7_no_query_retrieval --max-contexts 1 --max-queries 5 --output-root outputs/mab/factconsolidation_p7_smoke"
```

If GPU 0 is occupied, substitute the least-loaded GPU identified by `nvidia-smi`; record the physical GPU in the manifest.

- [ ] **Step 2: Verify startup once**

Check tmux, process, log growth, output directory, and GPU process. Report session, GPU, PID, and log path; then stop monitoring unless requested.

- [ ] **Step 3: Run the MH smoke after SH completes**

Use identical settings with `--subtask factconsolidation_mh_6k`.

- [ ] **Step 4: Validate artifacts**

Require:

- all selected questions scored by the official scorer;
- Disabled bank absent;
- P7 construction writes present;
- P7 query writes zero;
- pre/post query bank snapshot identical;
- reset slot count zero;
- no cross-context leakage;
- P7 config exactly matches the matrix;
- no prompt capacity failure.

- [ ] **Step 5: Record GO/NO-GO without paper edits**

Append smoke results to `EXPERIMENTS.md` and `PROGRESS.md`. GO means pipeline validity only. Commit notes and runner corrections separately; do not edit `paper/`.

---

### Task 6: Implement Strict Aggregation and Promotion Gates

**Files:**
- Create: `scripts/eval/factconsolidation_aggregate.py`
- Create: `tests/test_factconsolidation_aggregate.py`
- Create: `scripts/eval/factconsolidation_campaign_decision.py`
- Create: `tests/test_factconsolidation_campaign_decision.py`

- [ ] **Step 1: Write failing aggregate-validation tests**

Cover missing rows, duplicate question IDs, method/config mismatch, nonzero query writes, changed snapshots, reset failure, unequal method scopes, and non-finite metrics.

- [ ] **Step 2: Implement strict aggregation**

Emit:

```json
{
  "schema_version": "factconsolidation-aggregate/v1",
  "scope": {},
  "methods": {},
  "paired_transitions": {},
  "memory_metrics": {},
  "cost": {},
  "invariants": {}
}
```

Aggregate by subtask, SH/MH, context length, context identity, question identity, method, and repeat.

- [ ] **Step 3: Write failing promotion-decision tests**

Required outcomes:

```python
assert decide(insufficient_contexts) == "mechanism_only"
assert decide(invalid_protocol) == "internal_only"
assert decide(valid_but_no_effect_and_no_mechanism_signal) == "internal_only"
assert decide(valid_mechanism_signal_without_broad_scope) == "mechanism_only"
assert decide(valid_repeated_broad_evidence) == "main_table_candidate"
```

- [ ] **Step 4: Implement explicit evidence gates**

The decision payload must cite actual counts and reasons. It must never promote based only on a positive point estimate. Required inputs are independent context count, question count, repeat count, protocol validity, paired transition counts, variance, and mechanism activity.

- [ ] **Step 5: Test and commit**

```bash
pytest -q tests/test_factconsolidation_aggregate.py tests/test_factconsolidation_campaign_decision.py
git add scripts/eval/factconsolidation_aggregate.py tests/test_factconsolidation_aggregate.py scripts/eval/factconsolidation_campaign_decision.py tests/test_factconsolidation_campaign_decision.py
git commit -m "feat: add FactConsolidation evidence gates"
```

---

### Task 7: Scale to 32K/64K and Run Accepted Full Evaluations

**Files:**
- Runtime artifacts: `outputs/mab/factconsolidation_p7_full/`
- Runtime aggregate: `outputs/mab/factconsolidation_p7_aggregate.json`
- Runtime decision: `outputs/mab/factconsolidation_campaign_decision.json`

- [ ] **Step 1: Preflight every accepted context**

Measure chunk counts and rendered query capacity before generation. Mark over-capacity samples invalid; do not truncate.

- [ ] **Step 2: Execute SH/MH 32K**

Run each subtask and repeat in an independent detached process. Use fixed seeds `42`, `43`, `44`, `45`, and `46` only when stochastic generation is active; otherwise record a single deterministic pass and do not mislabel it as repeated evidence.

- [ ] **Step 3: Gate 64K execution**

Proceed only if 32K artifacts pass all invariants and estimated runtime/capacity remain viable. Otherwise record a 64K NO-GO with the exact reason.

- [ ] **Step 4: Aggregate and decide role**

Run:

```bash
python scripts/eval/factconsolidation_aggregate.py \
  --input-root outputs/mab/factconsolidation_p7_full \
  --output outputs/mab/factconsolidation_p7_aggregate.json
python scripts/eval/factconsolidation_campaign_decision.py \
  --aggregate outputs/mab/factconsolidation_p7_aggregate.json \
  --output outputs/mab/factconsolidation_campaign_decision.json
```

- [ ] **Step 5: Stop for explicit review**

Report one of `main_table_candidate`, `mechanism_only`, or `internal_only`. Do not launch ablations or edit the paper until the user accepts this role decision.

---

### Task 8: Run FactConsolidation Mechanism Ablations

**Files:**
- Runtime artifacts: `outputs/mab/factconsolidation_p7_ablations/`
- Runtime aggregate: `outputs/mab/factconsolidation_p7_ablation_aggregate.json`

- [ ] **Step 1: Freeze the ablation matrix**

Use one-factor-at-a-time conditions around P7:

- capacity: `4`, `8`, `16`, `32`;
- top-k remains `2` for this benchmark;
- update threshold: `0.05`, `0.10`, `0.15`;
- decay alpha: `0.0`, `0.05`, `0.10`;
- replacement: `thread_update`, `replace_oldest`;
- retrieval removal: P7 construction with query retrieval disabled.

Do not combine all factors factorially.

- [ ] **Step 2: Select the smallest valid task scope**

Use the shortest SH/MH scope that passed Task 7 and still exercises matched replacement and capacity eviction. If no condition reaches the relevant mechanism path, stop and classify the ablation as non-informative.

- [ ] **Step 3: Execute serialized runs**

Record exact config identity, method-separated latency, peak allocation, and transition counts.

- [ ] **Step 4: Aggregate curves**

Report task metric, insert/replace/evict counts, occupancy, retrieved latent count, and helpful/harmful paired transitions for each factor value.

- [ ] **Step 5: Decide paper visibility**

Promote only ablations with valid mechanism activation and stable interpretation. Otherwise retain internally without changing the paper.

---

### Task 9: Verify EventQA Is Complete Without Rerunning It

**Files:**
- Reuse: `outputs/mab/eventqa_final_comparison_package.json`
- Reuse: `outputs/mab/eventqa_paper_artifact_manifest_sha256.txt`
- Create at runtime: `outputs/mab/eventqa_campaign_revalidation.json`

- [ ] **Step 1: Revalidate checksums and package schema**

Run the existing manifest checks and final table packager in validation-only mode.

- [ ] **Step 2: Confirm planned retrieval ablations already exist**

Map no-query retrieval, top-k diagnostics, threshold diagnostics, capacity evidence, cost, and explicit-text controls to canonical artifacts.

- [ ] **Step 3: Produce a missing-row decision**

If all required EventQA rows exist, record `no_rerun_required`. A rerun is allowed only for a named missing or invalid row and requires a separate execution decision.

- [ ] **Step 4: Recheck baseline manifest**

Validate `outputs/mab/memory_benchmark_campaign_baseline.json`. Existing paper baseline hashes must still match unless the user intentionally changed paper files.

---

### Task 10: Package DetectiveQA as a Stress Test

**Files:**
- Create: `scripts/eval/detectiveqa_stress_aggregate.py`
- Create: `tests/test_detectiveqa_stress_aggregate.py`
- Reuse: existing `mab5*` and `mab6*` DetectiveQA artifacts

- [ ] **Step 1: Write aggregate tests**

Require the aggregator to reject scored over-capacity full-history rows and to preserve compressed-protocol labels.

- [ ] **Step 2: Implement artifact reuse first**

Aggregate existing MAB-5A/5B/5C/5D/6A/6B results by capacity, storage/injection path, exact match, retrieval activity, slot occupancy, and eviction count.

- [ ] **Step 3: Determine whether new runs are necessary**

Run new DetectiveQA evaluations only if the existing artifacts cannot produce a valid capacity/mechanism stress table. Do not rerun merely to seek a positive result.

- [ ] **Step 4: Decide BABILong activation**

Activate BABILong only if DetectiveQA cannot provide a valid length/capacity stress conclusion. Record the decision in `research_notes/DECISIONS.md` before any BABILong adapter work.

- [ ] **Step 5: Commit aggregator and tests**

```bash
pytest -q tests/test_detectiveqa_stress_aggregate.py
git add scripts/eval/detectiveqa_stress_aggregate.py tests/test_detectiveqa_stress_aggregate.py
git commit -m "analysis: package DetectiveQA stress evidence"
```

---

### Task 11: Build the Additive Paper Package and Fallback Gate

**Files:**
- Create: `scripts/eval/memory_benchmark_paper_package.py`
- Create: `tests/test_memory_benchmark_paper_package.py`
- Create at runtime: `outputs/mab/memory_benchmark_paper_package.json`

- [ ] **Step 1: Write fallback-first tests**

```python
def test_internal_only_factconsolidation_returns_unchanged_eventqa_package():
    package = build_package(eventqa=eventqa_fixture(), factconsolidation=decision_fixture("internal_only"), detectiveqa=None)
    assert package["paper_action"] == "no_change"
    assert package["main_table_additions"] == []

def test_mechanism_only_adds_appendix_without_main_table_change():
    package = build_package(eventqa=eventqa_fixture(), factconsolidation=decision_fixture("mechanism_only"), detectiveqa=None)
    assert package["main_table_additions"] == []
    assert package["ablation_additions"]

def test_main_table_candidate_adds_second_benchmark_table():
    package = build_package(eventqa=eventqa_fixture(), factconsolidation=decision_fixture("main_table_candidate"), detectiveqa=None)
    assert package["paper_action"] == "additive_update"
    assert package["main_table_additions"][0]["benchmark"] == "FactConsolidation"
```

- [ ] **Step 2: Implement package builder**

The output must contain:

```json
{
  "eventqa_baseline": {},
  "factconsolidation_role": "internal_only",
  "main_table_additions": [],
  "ablation_additions": [],
  "appendix_additions": [],
  "paper_action": "no_change",
  "claim_delta": []
}
```

- [ ] **Step 3: Run full test suite for paper packaging**

```bash
pytest -q tests/test_memory_benchmark_paper_package.py tests/test_eventqa_paper_aggregator.py tests/test_eventqa_final_table_package.py
```

- [ ] **Step 4: Build the package from validated aggregates**

The default output must be `paper_action=no_change`. Promotion requires the accepted Task 7 decision file.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/memory_benchmark_paper_package.py tests/test_memory_benchmark_paper_package.py
git commit -m "feat: gate additive benchmark paper evidence"
```

---

### Task 12: Update Research Records and Paper Only After Promotion

**Files:**
- Modify: `research_notes/EXPERIMENTS.md`
- Modify: `research_notes/PROGRESS.md`
- Modify: `research_notes/DECISIONS.md`
- Conditionally modify: `paper/draft_v0.md`
- Conditionally modify: `paper/main_table_blueprint.md`
- Modify: `paper/experiment_gap_to_table_mapping.md`

- [ ] **Step 1: Record the final campaign decision**

Append one decision with the exact evidence role:

- main-table extension accepted;
- mechanism/appendix-only accepted;
- EventQA-only fallback retained.

- [ ] **Step 2: Apply the package action**

If `paper_action=no_change`, do not edit `paper/draft_v0.md` or its existing tables. Update only experiment/decision records and the gap mapping.

If promotion is accepted, add FactConsolidation as a separate benchmark table or subsection. Do not recompute or rewrite existing EventQA values.

- [ ] **Step 3: Run evidence and prose checks**

Verify every new number against the machine-readable package. Check that the abstract and conclusion do not claim benchmark-general improvement unless both benchmark evidence and the accepted decision support it.

- [ ] **Step 4: Run final validation**

```bash
pytest -q tests/test_memory_benchmark_campaign_baseline.py tests/test_factconsolidation_adapter.py tests/test_factconsolidation_p7.py tests/test_factconsolidation_aggregate.py tests/test_factconsolidation_campaign_decision.py tests/test_detectiveqa_stress_aggregate.py tests/test_memory_benchmark_paper_package.py
git diff --check
git status --short --branch
```

Expected: all campaign tests PASS. Any pre-existing unrelated whitespace issue is reported separately and not silently edited.

- [ ] **Step 5: Commit the bounded paper decision**

Stage only the approved research-note and paper files. Inspect `git diff --cached --name-only` before committing.

Use one of:

```bash
git commit -m "paper: add validated memory benchmark evidence"
git commit -m "docs: retain EventQA-only paper after benchmark audit"
```

---

## Campaign Stop Points

Stop and request review after:

1. Task 2 dataset audit;
2. Task 5 SH/MH smoke;
3. Task 7 promotion decision;
4. Task 8 ablation aggregate;
5. Task 10 DetectiveQA/BABILong decision;
6. Task 11 paper package decision;
7. Task 12 final paper diff.

At every stop point, the existing EventQA paper remains usable. No later task is required to preserve the current paper claim.
