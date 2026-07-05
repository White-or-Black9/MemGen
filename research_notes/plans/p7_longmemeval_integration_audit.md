# P7 LongMemEval Integration Audit

Date: 2026-07-04

## Purpose

Audit whether LongMemEval can serve as the long-term conversational-memory benchmark for the frozen P7 formal experiment without changing the method.

Frozen P7 boundary:

- `retrieve_threshold=0.05`
- `update_threshold=0.10`
- `max_slots=16`
- `top_k=2`
- `decay_alpha=0.05`
- Weaver-space bank path / MAB-6B-style mechanism
- session-local latent memory bank
- no Trigger / Weaver retraining
- no utility gate
- no tuple suppression
- no top-1 fallback

## Executive finding

LongMemEval is available in the local MemoryAgentBench benchmark path, not as a MemGen-native runner in this repository.

- MemGen repo status:
  - no `scripts/eval/*longmemeval*` runner
  - no `tests/*longmemeval*`
  - no MemGen-local LongMemEval adapter implementation
- External benchmark status:
  - local MemoryAgentBench clone contains LongMemEval data configs, templates, loader integration, and an LLM-judge script
  - local cached dataset contains `longmemeval_s*` with 5 benchmark items
  - local cached dataset does **not** currently expose `longmemeval_s_-1_500`, even though an upstream config file exists for it

Conclusion:

- usable benchmark support exists, but only through the external MAB-converted path
- P7 adaptation is feasible in principle, but not with a drop-in runner that already exists in this repo

## Repository search result

### In this repo

LongMemEval references appear only in notes:

- `research_notes/benchmarks/memoryagentbench_runbook.md`
- `research_notes/benchmarks/memoryagentbench_feasibility_assessment.md`
- `research_notes/benchmarks/memoryagentbench_local_task_availability.md`
- `research_notes/benchmarks/memoryagentbench_configuration_plan.md`

These notes consistently say:

- LongMemEval is treated as a MemoryAgentBench task
- it requires an LLM judge for paper-faithful scoring
- it was previously deferred

No MemGen-native LongMemEval runner or adapter was found under:

- `scripts/eval/`
- `tests/`
- repo-level README references

### In local MemoryAgentBench clone

Found:

- config files:
  - `/mnt/18T/baishilong/benchmarks/MemoryAgentBench/configs/data_conf/Accurate_Retrieval/LongMemEval/Longmemeval_s.yaml`
  - `/mnt/18T/baishilong/benchmarks/MemoryAgentBench/configs/data_conf/Accurate_Retrieval/LongMemEval/Longmemeval_s_star.yaml`
- template support:
  - `/mnt/18T/baishilong/benchmarks/MemoryAgentBench/utils/templates.py`
- post-processing route:
  - `/mnt/18T/baishilong/benchmarks/MemoryAgentBench/utils/eval_other_utils.py`
- LLM judge:
  - `/mnt/18T/baishilong/benchmarks/MemoryAgentBench/llm_based_eval/longmem_qa_evaluate.py`
- main benchmark harness:
  - `/mnt/18T/baishilong/benchmarks/MemoryAgentBench/main.py`
  - `/mnt/18T/baishilong/benchmarks/MemoryAgentBench/conversation_creator.py`
  - `/mnt/18T/baishilong/benchmarks/MemoryAgentBench/initialization.py`

## Original vs MAB-converted status

The usable local path is MAB-converted LongMemEval inside MemoryAgentBench.

Evidence:

- task config is under `Accurate_Retrieval/LongMemEval`
- templates normalize `longmemeval_*` to benchmark template family `longmemeval`
- loader filters by `metadata.source == sub_dataset`
- judge script loads `ai-hyz/MemoryAgentBench`, split `Accurate_Retrieval`

This means the current usable path is not an original standalone LongMemEval pipeline inside MemGen. It is LongMemEval reformulated as a MemoryAgentBench task.

## Dataset contract

### Dataset path

Local cached parquet:

- `/mnt/18T/baishilong/datasets/MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet`

Benchmark loader:

- `load_dataset("ai-hyz/MemoryAgentBench", split="Accurate_Retrieval", revision="main")`
- then filter by `metadata.source`

### Available local LongMemEval subset

Local source counts:

- `longmemeval_s*`: 5
- `longmemeval_s_-1_500`: 0 found in local parquet

Important mismatch:

- upstream config `Longmemeval_s.yaml` expects `sub_dataset: longmemeval_s_-1_500`
- local cache currently appears to contain only `longmemeval_s*`

### File/sample structure

Per local LongMemEval row:

- one `context` string
- one array of `questions`
- one array of `answers`
- metadata arrays for:
  - `question_ids`
  - `question_types`
  - `question_dates`
  - `qa_pair_ids`
- extra metadata:
  - `haystack_sessions`
  - `demo`
  - `keypoints` appears absent in the inspected local rows

Observed local shape:

- 5 benchmark contexts total
- 60 questions per context
- 300 question instances total
- context length about `1.59M` to `1.72M` characters per context

### Session / chunk / turn structure

MemoryAgentBench contract:

- one dataset item -> one benchmark context
- one context -> chunked sequentially by `chunk_text_into_sentences(...)`
- default chunk size in config: `4096`
- no overlap in the standard chunker
- then multiple questions are asked after memorization

`ConversationCreator` and `main.py` implement:

1. load one context
2. split into chunks
3. call `agent.send_message(chunk, memorizing=True)` for each chunk
4. then call `agent.send_message(query, memorizing=False, query_id=..., context_id=...)` for each question

This matches the benchmark's inject-once / query-multiple-times contract.

### Question format

LongMemEval template family in `utils/templates.py`:

- memorize prompt:
  - "The following context is the conversation between the user and the assistant ..."
- query prompt:
  - "The history chats are between you and a user. Based on the relevant chat history, answer the question as concisely as you can, using a single phrase if possible."

Observed question instances include:

- `Current Date: ...`
- then `Now Answer the Question: ...`

So the benchmark explicitly encodes current-date grounding into each query.

### Answer format

Observed answers are heterogeneous:

- short factual values: numbers, names, locations
- temporal answers with off-by-one tolerance text
- preference/rubric-style answers
- abstention/unanswerable explanations
- ordering / multi-step textual answers

### Scoring / evaluator

Benchmark post-process path:

- `eval_other_utils.post_process(...)` routes `longmemeval` to `_process_infbench_longmemeval_dataset(...)`
- for LongMemEval this falls back to `default_post_process(...)`
- README clarifies paper-faithful LongMemEval metric is **LLM-as-judge**

Judge implementation:

- `llm_based_eval/longmem_qa_evaluate.py`
- uses `gpt-4o` as metric model
- evaluates each hypothesis against question, reference answer, and question type
- writes per-question judge labels

### LLM-as-judge requirement

Yes, for paper-faithful evaluation.

There is no local automatic metric in the benchmark that should be treated as the final LongMemEval score. The benchmark README explicitly labels LongMemEval as `LLM-as-judge`.

### Ability-type labels available

Yes, from `metadata.question_types`.

Observed unique local types:

- `knowledge-update`
- `multi-session`
- `single-session-assistant`
- `single-session-preference`
- `single-session-user`
- `temporal-reasoning`

Observed abstention support:

- `13/300` inspected `question_ids` contain `_abs`

So the required conversational-memory dimensions are present, including knowledge updates, temporal reasoning, multi-session questions, preference-style questions, and abstention-like unanswerable cases.

## Mapping to frozen P7

### Session-local reset

Reset boundary should be:

- one LongMemEval context = one P7 session-local bank
- reset after the final query of that context
- reset on any failure path before the next context starts

This is cleanly aligned with frozen P7.

### Is `frozen_context_bank` applicable?

Not directly.

Reason:

- EventQA `frozen_context_bank` means build one frozen bank from a static context, then reuse it across multiple query turns with query-time writes blocked
- LongMemEval is conversational and sessional; the benchmark already expects one full conversation context to be ingested before multiple queries

Recommended adaptation:

- use a LongMemEval-specific analog of EventQA's frozen-bank query regime:
  - ingest the full context once
  - freeze the constructed bank for all evaluation questions of that context
  - keep query-time writes blocked

This is benchmark-faithful and consistent with current EventQA formal conventions, but it is not literally the same named protocol.

### Construction-time ingestion

Recommended construction protocol:

- use the benchmark chunker and LongMemEval memorize template unchanged
- sequentially process each chunk as one memorization turn
- allow normal P7 write / retrieve / update / replacement during construction
- preserve the session-local Weaver-space latent bank

### Final query-time retrieval

Recommended query protocol:

- after construction, answer each LongMemEval question with the same frozen bank
- enable retrieval at query time
- block query-time writes so that one question does not contaminate later questions within the same context

This matches the benchmark's inject-once / query-multiple-times logic and avoids conflating memory quality with online within-evaluation bank mutation.

### Memory metrics to log

Minimum required metrics:

- context index / question index / `qa_pair_id`
- `question_type`
- `question_date`
- raw prediction
- parsed prediction if any local parser is used
- judge label placeholder / final judge label
- bank write count during construction
- bank retrieval count during queries
- retrieved latent count per query
- selected retrieved slot indices / scores
- final slot count after construction
- query write count delta
- query write attempts blocked
- bank reset confirmation after context
- latency seconds
- peak CUDA memory

Strongly recommended diagnostics:

- construction chunk count
- chunk token lengths
- estimated full-history query tokens
- compressed/frozen query token count
- bank snapshot hash after construction
- per-slot access counts
- true insert / matched replace / capacity evict counts
- question-type grouped judge accuracy

## Existing runner-convention reuse

### Can current conventions support Disabled / P6 / P7?

Conceptually yes, but not with an existing drop-in script.

What already exists:

- Bank-off / Bank-on paired pattern in MAB scripts
- session-local memory instrumentation
- latency / peak-memory logging
- query-write blocking conventions in EventQA and detective_qa paths

What is missing:

- generic MemoryAgentBench adapter for MemGen
- LongMemEval-specific runner
- LongMemEval-specific result packaging
- LongMemEval judge integration into MemGen output workflow

### Can current conventions support text-summary memory baseline?

Not cleanly from current scripts.

- no existing MemGen LongMemEval text-summary baseline runner was found
- would require a separate baseline implementation path

### Can current conventions support RAG / retrieved-text baseline?

Only externally via MemoryAgentBench's built-in RAG agents, not via current MemGen scripts.

- benchmark repo already has RAG agent configs
- MemGen repo does not contain a matched LongMemEval retrieved-text baseline runner

### Can current conventions support matched-budget baseline?

No existing matched-budget baseline runner was found.

- would need explicit budget accounting and a new baseline wrapper

## Engineering feasibility judgment

### Clean adaptation level

P7 can be adapted **moderately cleanly** at the protocol level:

- session-local bank fits naturally
- sequential context ingestion fits naturally
- multi-query frozen-bank evaluation fits naturally
- question-type metadata is already present

### Why it is not fully clean yet

Three blockers remain before a GPU run:

1. No MemGen-native LongMemEval runner exists.
2. Paper-faithful scoring depends on GPT-4o judge integration.
3. The local cache exposes `longmemeval_s*`, while one upstream config targets `longmemeval_s_-1_500`.

## Recommended minimum smoke test

Goal:

- verify MemGen/MAB engineering fit on one local LongMemEval context without changing P7

Recommended scope:

- contexts: `1`
- questions: `2`
- source subset: local `longmemeval_s*`
- question types:
  - one factual/retrieval-heavy question such as `multi-session` or `knowledge-update`
  - one non-factual judge-sensitive question such as `single-session-preference` or one abstention case if available

Why not more:

- each context is extremely large
- each context carries 60 questions
- judge integration adds external dependency and cost

Candidate output root:

- `outputs/mab/longmemeval_smoke/<timestamp>-longmemeval-sstar-p7-smoke/`

Exact runnable MemGen command:

- not currently inferable from this repo, because no LongMemEval MemGen runner exists

Nearest official-harness command that is inferable today:

```bash
cd /mnt/18T/baishilong/benchmarks/MemoryAgentBench
python main.py \
  --agent_config configs/agent_conf/Long_Context_Agents/Long_context_agent_gpt-4o-mini.yaml \
  --dataset_config configs/data_conf/Accurate_Retrieval/LongMemEval/Longmemeval_s_star.yaml \
  --max_test_queries_ablation 2 \
  --force
```

This is only an official benchmark harness sanity command, not a MemGen P7 command.

Required smoke outputs:

- raw per-question outputs
- query metadata with `question_type`, `question_id`, `qa_pair_id`
- construction and query memory diagnostics
- latency / peak memory
- explicit `query_write_count == 0` during evaluation questions
- judge-ready JSON output

Expected runtime and risks:

- likely much slower than current EventQA runs because one context is on the order of `400k` configured tokens and about `1.6M` characters locally
- one-context smoke with 2 queries is still likely a tens-of-minutes to hour-scale job, depending on chunk count and memorization-turn cost
- largest evaluator risk is GPT-4o judge dependency
- largest protocol risk is accidental query-time write leakage across the two evaluation questions

Stopping conditions:

- if local loader for `longmemeval_s*` does not produce the expected 1 context / 2 query subset
- if query-time writes cannot be blocked cleanly
- if bank reset after context is not guaranteed
- if result packaging is not compatible with later GPT-4o judging
- if the implementation route requires modifying model code rather than only adding a runner/adapter

## Decision

LongMemEval is available and useful, but only through the MAB-converted benchmark path. It is a valid next benchmark candidate for the paper target, but a GPU run should wait until a dedicated LongMemEval MemGen adapter/runner and judge-compatible output contract exist.
