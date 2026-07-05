# P7 LongMemEval Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimum MemGen LongMemEval adapter/runner for frozen P7 so that one local `longmemeval_s*` context and two selected questions can be executed in prediction-only mode, with judge-compatible outputs and zero query-time writes.

**Architecture:** Reuse the external MemoryAgentBench loader, chunker, templates, and judge script contract, but keep MemGen generation inside a new repository-local runner modeled after the current EventQA and detective_qa MAB runners. Separate generation from GPT-4o judging by emitting a stable judge-ready export and a thin wrapper command, rather than hard-wiring API-dependent judging into the generation path.

**Tech Stack:** Python, existing `scripts/eval/*mab*` runners, local MemoryAgentBench clone under `/mnt/18T/baishilong/benchmarks/MemoryAgentBench`, HuggingFace dataset cache, JSON/JSONL artifacts, unittest.

---

## Scope

This is a planning document only. It does **not** implement the runner.

Constraints carried into implementation:

- no P7 method changes
- no model-code edits
- no paper edits
- no GPU runs in the planning phase

## Reuse Plan

### Existing external MAB files to reuse

Loader / dataset plumbing:

- `/mnt/18T/baishilong/benchmarks/MemoryAgentBench/utils/eval_data_utils.py`
  - `load_data_huggingface(...)`
  - `_load_and_filter_dataset(...)`
  - local source filtering by `metadata.source`
- `/mnt/18T/baishilong/benchmarks/MemoryAgentBench/conversation_creator.py`
  - `ConversationCreator`
  - `get_chunks()`
  - `get_query_and_answers()`
  - question / answer / `qa_pair_id` extraction
- `/mnt/18T/baishilong/benchmarks/MemoryAgentBench/utils/templates.py`
  - LongMemEval memorize/query template family

Metric / post-process references:

- `/mnt/18T/baishilong/benchmarks/MemoryAgentBench/utils/eval_other_utils.py`
  - use as reference only for output compatibility
  - do **not** treat its default LongMemEval route as final scoring

Judge path:

- `/mnt/18T/baishilong/benchmarks/MemoryAgentBench/llm_based_eval/longmem_qa_evaluate.py`
  - reuse the prompt logic and reference-loading logic
  - do **not** call it raw from the new runner without a wrapper/export step

Reason `longmem_qa_evaluate.py` needs wrapping:

- it assumes benchmark output discovery under `./outputs/{evaluated_method}/Accurate_Retrieval`
- it searches for filenames containing `longmemeval_s*_*`
- it expects a JSON file with a top-level `"data"` list aligned to the benchmark references
- it owns the output/result file layout and should not dictate MemGen generation output layout

Decision:

- reuse `longmem_qa_evaluate.py` logic indirectly via a thin MemGen-side export/wrapper
- do not rewrite the judge prompts unless forced by incompatibility

### Existing MemGen files to reuse

Primary runner patterns:

- `scripts/eval/mab6b_weaver_space_bank_eventqa_65536_n5.py`
  - strongest reference for frozen-bank multi-question protocol
  - strongest reference for `query_write_count == 0`
  - strongest reference for per-question latency / peak-memory / slot diagnostics
- `scripts/eval/mab6b_weaver_space_bank_detectiveqa_n10.py`
  - strongest reference for Weaver-space bank wiring and Version B/P7-aligned bank settings
- `scripts/eval/mab5a_detectiveqa_compressed_n10.py`
  - strongest reference for benchmark-bridge payload preparation, manifest layout, and paired row aggregation
- `scripts/eval/mab3_bank_on_full_history.py`
  - bank config and interaction wiring
- `scripts/eval/mab2_mab_bridge.py`
  - reference for controlled benchmark-side helper invocation pattern

Helpful tests/patterns:

- `tests/test_mab6b_weaver_space_bank.py`
- `tests/test_eventqa_transition_diagnostics.py`
- `tests/test_mab5a_detectiveqa_compressed_n10.py`
- `tests/test_mab5c_decoupled_thresholds_detectiveqa_n10.py`

## Proposed New Files

### 1. `scripts/eval/longmemeval_mab_adapter.py`

Purpose:

- isolate all MemoryAgentBench-specific LongMemEval loading, context/question selection, and judge-export formatting from the MemGen runner

Expected inputs:

- local MAB repo path
- local parquet path or HF loader configuration
- `sub_dataset` such as `longmemeval_s*`
- explicit context-index list
- explicit question filters:
  - `max_questions`
  - `question_types`
  - `question_id_whitelist`
  - optional abstention selector

Expected outputs:

- selected contexts with:
  - `context_id`
  - raw `context`
  - chunk list
  - selected question records
  - `qa_pair_id`
  - `question_id`
  - `question_type`
  - `question_date`
- judge-ready reference export in stable order

Capability:

- formal-run capable

### 2. `scripts/eval/mab6b_weaver_space_bank_longmemeval.py`

Purpose:

- main MemGen LongMemEval runner for frozen P7
- perform construction-time chunk ingestion once per context
- freeze bank
- answer multiple selected questions under the same frozen bank
- block query-time writes
- emit prediction-only and judge-ready artifacts

Expected inputs:

- checkpoint path / model checkpoint id
- MAB repo path
- MAB python path if any helper call is retained
- local parquet path
- `sub_dataset` defaulting to `longmemeval_s*`
- `context_index`
- `requested_contexts`
- `max_questions`
- question-type filters
- explicit P7 bank config defaults
- `--prediction-only`

Expected outputs:

- output root with manifest, per-question JSONL, construction diagnostics, aggregate summary, judge-ready export, run log

Capability:

- formal-run capable

### 3. `scripts/eval/longmemeval_judge_export.py`

Purpose:

- convert runner output into the exact structure needed by `longmem_qa_evaluate.py`
- emit a separate judge command record
- keep generation and judging decoupled

Expected inputs:

- MemGen LongMemEval run root
- target evaluated-method label
- target dataset label

Expected outputs:

- benchmark-compatible JSON with top-level `"data"` list
- stable question ordering aligned with selected references
- `judge_command.sh` or `judge_command.txt`

Capability:

- formal-run capable

### 4. `tests/test_longmemeval_mab_adapter.py`

Purpose:

- validate local `longmemeval_s*` selection
- validate context/question filtering
- validate preservation of `question_id`, `qa_pair_id`, `question_type`, `question_date`

Expected inputs:

- mocked or small fixture LongMemEval sample structures

Expected outputs:

- unittest pass/fail only

Capability:

- smoke-only support for development, not a benchmark artifact

### 5. `tests/test_mab6b_weaver_space_bank_longmemeval.py`

Purpose:

- validate runner protocol and output schema
- assert query-time writes remain zero
- assert one context maps to one bank reset cycle
- assert judge-ready export order is stable

Expected inputs:

- mocked runner payloads / patched model hooks

Expected outputs:

- unittest pass/fail only

Capability:

- smoke-only support for development, not a benchmark artifact

### 6. `tests/test_longmemeval_judge_export.py`

Purpose:

- validate exported JSON shape against `longmem_qa_evaluate.py` expectations
- validate no API keys are required to perform export

Expected inputs:

- synthetic MemGen run output

Expected outputs:

- unittest pass/fail only

Capability:

- smoke-only support for development, not a benchmark artifact

## Protocol Mapping to Frozen P7

### Session reset boundary

- one LongMemEval context = one session-local latent bank
- create/reset bank before construction for that context
- hard reset after all selected questions for that context
- hard reset on any exception before moving to the next context

### Construction-time ingestion loop

- use MAB `ConversationCreator` or adapter-equivalent chunking
- preserve upstream LongMemEval memorize template
- process chunks sequentially in original order
- run full P7 construction-time write / retrieve / update / replacement behavior
- record construction diagnostics per chunk

### Bank freezing before questions

- after final memorization chunk, store a construction-complete bank snapshot
- all selected evaluation questions for the same context reuse that same frozen bank root
- each question starts from the same frozen-bank snapshot, not from the previous question's post-query state

Reason:

- strongest alignment with inject-once / query-multiple-times benchmark logic
- strongest protection against question-to-question contamination

### Query-time retrieval

- retrieval remains enabled
- use frozen P7 settings unchanged
- log selected slots / scores / retrieved latent count per question

### Query-time writes blocked

- required invariant: `query_write_count == 0`
- also log `query_write_attempt_count`
- if a question path attempts a write, count and block it rather than silently mutating the bank

### Multi-question handling under the same frozen bank

- select question subset explicitly for the smoke
- for each selected question:
  - restore or clone the frozen-bank snapshot
  - run one query
  - collect result row
- do not let one question mutate the frozen base used by later questions

### Output collection

Per question:

- one row in `per_question.jsonl`

Per context:

- one row in `construction_diagnostics.jsonl`
- one context summary in aggregate output

Per run:

- `manifest.json`
- `aggregate.json`
- judge-ready export

## Metrics and Logging Contract

Each per-question row must include at least:

- `run_id`
- `context_index`
- `context_id`
- `question_index`
- `question_id`
- `qa_pair_id`
- `question_type`
- `question_date`
- `sub_dataset`
- raw question text
- gold answer
- raw prediction
- parsed prediction if any parser is used
- prediction-only status
- evaluator status
- judged label placeholder / final judged label
- construction write count
- construction retrieve count if exposed by the bank
- query retrieval active count / boolean
- retrieved latent count
- retrieved slot indices
- retrieved slot scores
- query write count
- query write attempt count
- blocked query write attempts
- final slot count of the frozen construction bank
- Trigger call count
- Weaver call count
- latency seconds
- peak GPU memory
- output token count

Each per-context construction record must include at least:

- `context_index`
- `context_id`
- chunk count
- chunk token lengths
- total construction writes
- total construction retrievals
- total true inserts
- total matched replacements
- total capacity evictions
- final slot count
- frozen-bank snapshot hash
- bank reset confirmation

Each aggregate summary must include at least:

- total contexts attempted
- total questions attempted
- total questions completed
- counts by `question_type`
- retrieval-active question count
- total blocked query writes
- mean / max latency
- mean / max peak GPU memory
- evaluator status summary

## GPT-4o Judge Handling

### Required design

- generation and judging must be separate steps
- runner must support prediction-only output with no API dependency
- judge command must be emitted as a recorded artifact, not hidden in code paths

### Recommended flow

1. run MemGen generation in `--prediction-only` mode
2. export benchmark-compatible prediction JSON
3. emit a standalone judge command artifact
4. run judge later in an environment with API keys set externally

### Judge wrapper decision

- do not embed API keys in code
- do not require keys for generation
- do not require judge success for prediction-only smoke completion

### Reuse decision for `longmem_qa_evaluate.py`

- reuse with wrapping/export
- do not reuse raw path discovery and output assumptions directly

## Local Subset Mismatch Handling

Rules:

- default smoke subset must be `longmemeval_s*`
- do not assume `longmemeval_s_-1_500` exists locally
- make subset, context indices, and question IDs explicit in manifest

Implementation requirement:

- runner CLI must accept `--sub-dataset`
- runner defaults to `longmemeval_s*`
- runner manifest must record:
  - available local sources detected
  - selected source
  - selected context indices
  - selected question IDs

## Minimum Smoke Command

Candidate command shape is inferable once the new runner exists:

```bash
cd /mnt/18T/baishilong/MemGen
CUDA_VISIBLE_DEVICES=<GPU_ID> /home/baishilong/miniconda3/envs/memgen/bin/python \
  scripts/eval/mab6b_weaver_space_bank_longmemeval.py \
  --mab-repo /mnt/18T/baishilong/benchmarks/MemoryAgentBench \
  --mab-python /home/baishilong/miniconda3/envs/MABench/bin/python \
  --parquet /mnt/18T/baishilong/datasets/MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet \
  --checkpoint-path /home/baishilong/.cache/huggingface/hub/models--Kana-s--MemGen/snapshots/269d9b1741130b94fffa410cdaa3d4bc74081a7f/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model \
  --model-checkpoint-id <checkpoint-id> \
  --sub-dataset 'longmemeval_s*' \
  --context-index 0 \
  --max-questions 2 \
  --question-type multi-session \
  --question-type single-session-preference \
  --prediction-only
```

What is still missing before this becomes executable:

- the runner file itself
- the exact `--question-type` / question-selection flag names
- final decision on whether `--mab-python` is still needed after adapter extraction

## Acceptance Criteria for Implementation Readiness

Implementation is ready to code only when all items below are explicit and stable:

- [ ] no P7 method changes are required
- [ ] exact external loader path is fixed
- [ ] exact local subset default is fixed to `longmemeval_s*`
- [ ] smoke context index selection rule is fixed
- [ ] smoke question selection rule is fixed
- [ ] output schema is defined for per-question, per-context, and aggregate artifacts
- [ ] judge dependency is isolated from generation
- [ ] `query_write_count == 0` can be asserted
- [ ] frozen-bank restore/clone behavior across multiple questions is specified
- [ ] Trigger / Weaver call counting strategy is specified
- [ ] prediction-only smoke can finish without API keys

## Task Breakdown

### Task 1: Define the adapter boundary

**Files:**
- Create: `scripts/eval/longmemeval_mab_adapter.py`
- Test: `tests/test_longmemeval_mab_adapter.py`

- [ ] **Step 1: Write the failing adapter-selection tests**

Tests should cover:

```python
def test_selects_longmemeval_sstar_from_local_source_map(): ...
def test_preserves_question_metadata_fields(): ...
def test_filters_two_questions_by_type_and_order(): ...
def test_records_abs_question_ids_when_requested(): ...
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python -m unittest tests.test_longmemeval_mab_adapter -v
```

Expected: import or symbol failures because adapter does not exist yet.

- [ ] **Step 3: Implement minimal adapter**

Implement:

- local source inspection
- LongMemEval context/question extraction
- stable question ordering
- smoke subset filtering

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python -m unittest tests.test_longmemeval_mab_adapter -v
```

Expected: PASS

### Task 2: Build the prediction-only runner

**Files:**
- Create: `scripts/eval/mab6b_weaver_space_bank_longmemeval.py`
- Test: `tests/test_mab6b_weaver_space_bank_longmemeval.py`

- [ ] **Step 1: Write the failing runner tests**

Tests should cover:

```python
def test_one_context_maps_to_one_session_reset_cycle(): ...
def test_query_write_count_is_zero_for_all_selected_questions(): ...
def test_runner_emits_required_per_question_fields(): ...
def test_runner_restores_frozen_bank_between_questions(): ...
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python -m unittest tests.test_mab6b_weaver_space_bank_longmemeval -v
```

Expected: FAIL because runner does not exist yet.

- [ ] **Step 3: Implement minimal runner**

Implement:

- manifest generation
- construction-time ingestion loop
- frozen-bank snapshot / restore
- multi-question query loop
- prediction-only artifact writing

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python -m unittest tests.test_mab6b_weaver_space_bank_longmemeval -v
```

Expected: PASS

### Task 3: Isolate judge export

**Files:**
- Create: `scripts/eval/longmemeval_judge_export.py`
- Test: `tests/test_longmemeval_judge_export.py`

- [ ] **Step 1: Write the failing export tests**

Tests should cover:

```python
def test_exports_top_level_data_list_for_judge(): ...
def test_preserves_reference_order_for_selected_questions(): ...
def test_emits_standalone_judge_command_without_api_keys(): ...
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python -m unittest tests.test_longmemeval_judge_export -v
```

Expected: FAIL because export script does not exist yet.

- [ ] **Step 3: Implement minimal export wrapper**

Implement:

- transform runner output into judge-ready JSON
- emit standalone judge command artifact
- no network use during export

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python -m unittest tests.test_longmemeval_judge_export -v
```

Expected: PASS

### Task 4: Verify the combined contract

**Files:**
- Modify: `scripts/eval/mab6b_weaver_space_bank_longmemeval.py`
- Modify: `scripts/eval/longmemeval_judge_export.py`
- Test: `tests/test_longmemeval_mab_adapter.py`
- Test: `tests/test_mab6b_weaver_space_bank_longmemeval.py`
- Test: `tests/test_longmemeval_judge_export.py`

- [ ] **Step 1: Run the full targeted test set**

Run:

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python -m unittest \
  tests.test_longmemeval_mab_adapter \
  tests.test_mab6b_weaver_space_bank_longmemeval \
  tests.test_longmemeval_judge_export -v
```

Expected: PASS

- [ ] **Step 2: Run syntax verification**

Run:

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python -m py_compile \
  scripts/eval/longmemeval_mab_adapter.py \
  scripts/eval/mab6b_weaver_space_bank_longmemeval.py \
  scripts/eval/longmemeval_judge_export.py
```

Expected: exit 0

- [ ] **Step 3: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: no whitespace or conflict-marker errors

## Self-Review

Spec coverage check:

- reuse paths: covered in Reuse Plan
- new files with inputs/outputs/smoke-vs-formal: covered in Proposed New Files
- P7 protocol mapping: covered in Protocol Mapping
- logging metrics: covered in Metrics and Logging Contract
- GPT-4o judge handling: covered in Judge Handling
- subset mismatch: covered in Local Subset Mismatch Handling
- minimum smoke command: covered in Minimum Smoke Command
- acceptance criteria: covered in Acceptance Criteria

Placeholder scan:

- no `TODO` / `TBD` placeholders left in the implementation plan

Type/contract consistency:

- one adapter file
- one runner file
- one judge-export file
- three corresponding test files

Implementation should start only after the user approves this plan.
