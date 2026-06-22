# MemoryAgentBench Configuration Plan for MemGen

> Historical setup plan. Environment and task-selection provenance remains
> useful, but current operations and mechanism routing are defined by
> `memoryagentbench_runbook.md` and `memoryagentbench_next_steps.md`.

Date: 2026-06-19  
Branch context: `rlm-memory-bank`  
Scope: configuration and execution planning only; no adapter implementation or experiment execution

Primary sources:

- Paper: <https://arxiv.org/abs/2507.05257>
- Repository: <https://github.com/HUST-AI-HYZ/MemoryAgentBench>
- Dataset: <https://huggingface.co/datasets/ai-hyz/MemoryAgentBench>
- Inspected upstream snapshot: commit `455306dcabc3842526eb83cd4e225e5d486c5c5d` (2026-05-21)

This plan treats the upstream repository and dataset as versioned external inputs. Exact paths, schemas, and commands below were checked against the inspected snapshot. They must be rechecked if the upstream commit changes.

## 1. Executive Summary

MemoryAgentBench should be used as MemGen's **main memory-specific benchmark**, but not as its only benchmark. It directly tests incremental memory construction and later query-time use across four competencies: Accurate Retrieval, Test-Time Learning, Long-Range Understanding, and Conflict Resolution. Its interaction model is substantially closer to MemGen's session-local latent memory bank than static question-answering alone.

GSM8K and TriviaQA remain auxiliary benchmarks for original-ability preservation. They answer a different but necessary question: whether enabling the memory bank preserves the accepted disabled behavior and does not damage MemGen's existing reasoning and QA capabilities.

The immediate next phase after this plan is **MAB-1**, an official-harness smoke test without MemGen using `Conflict_Resolution/Factconsolidation_sh_6k.yaml`, one context, and one query. Adapter implementation must not begin until the official loader, chunker, prompt formatting, metric calculation, and JSON output contract are verified end to end.

Initial comparison scope:

- Required: MemGen disabled.
- Required: MemGen enabled Version A, with retrieved memory injected into the Reasoner only.
- First benchmark tasks: `Factconsolidation_sh_6k` and `Factconsolidation_mh_6k`.
- Next task after those pass: `EventQA_64k`.
- No fork, full benchmark, LLM judge, or million-token run in the immediate phase.

## 2. Benchmark Role in the Project

### Fit to MemGen

MemoryAgentBench builds memory through sequential text chunks and then issues one or more questions against the accumulated context. This maps naturally to MemGen as follows:

- One benchmark item/context is one MemGen session.
- One context chunk is one memorization turn.
- All questions attached to that context are query turns in the same session.
- The session-local latent bank is reset after the final query and on every failure path.
- Sequential processing with `batch_size=1` matches the current enabled-mode constraint.
- The benchmark's chunk boundary supplies a natural prompt boundary. MemGen's Trigger normally firing once at prompt end can therefore produce at most one controlled memory-construction opportunity per chunk without changing Trigger training or placement.

The benchmark does not require memory to be textual or externally inspectable. It requires a final answer and accounting fields, so reasoner-space `latent_inputs_embeds` can remain internal.

### Competency priority

**Primary competencies**

- **Accurate Retrieval:** tests whether a later query retrieves useful information from the session-local latent bank and whether that information improves the Reasoner's answer.
- **Conflict Resolution:** tests whether replacement, recency, capacity, and stale-memory handling preserve current facts while suppressing superseded facts. This is the closest benchmark-level test of MemGen's current `thread_update`, last-retrieved decay, and replacement behavior.

**Secondary competencies**

- **Test-Time Learning:** useful for testing whether demonstrations or label mappings can be retained and applied, but it is less direct than factual memory and may confound memory with in-context task induction.
- **Long-Range Understanding:** useful later, but DetectiveQA and summarization require broad integration and long generation. Poor results may reflect reasoning or summarization limits rather than memory-bank quality.

### Claims the benchmark can support

With paired disabled/enabled runs, fixed prompts, fixed checkpoint, deterministic decoding where possible, and no leakage, selected automatic-metric tasks can support claims that:

- Version A session-local latent memory improves retrieval and use of previously ingested context.
- MemGen can process chunked, incremental memory construction rather than only a single static prompt.
- The current replacement/recency policy helps or hurts conflict resolution under controlled ablations.
- Improvements persist as context depth or conflict density increases within the tested task/config range.
- Disabled mode preserves the accepted non-memory execution path when combined with GSM8K/TriviaQA preservation evidence.

### Claims the benchmark cannot support

The initial subset cannot support claims that:

- MemGen is state of the art on the full MemoryAgentBench.
- MemGen solves general lifelong, persistent, or cross-session memory; all planned memory is session-local and reset per context.
- MemGen performs online parameter learning; stored state is latent activation memory, not weight updates.
- MemGen handles unrestricted million-token contexts or all four competencies equally well.
- MemGen outperforms commercial long-context systems unless those systems are reproduced under matched prompts, costs, models, and scoring.
- Latent memory is interpretable, factually grounded, or safe merely because answer accuracy improves.
- Memory caused an improvement unless diagnostics show effective writes, useful retrievals, correct injection boundaries, and no disabled-path divergence.

## 3. Repository and Environment Plan

### Checkout location and versioning

Use a persistent checkout outside both codebases:

```text
/mnt/18T/baishilong/benchmarks/MemoryAgentBench
```

Proposed read-only setup commands for MAB-1:

```bash
mkdir -p /mnt/18T/baishilong/benchmarks
git clone https://github.com/HUST-AI-HYZ/MemoryAgentBench.git \
  /mnt/18T/baishilong/benchmarks/MemoryAgentBench
cd /mnt/18T/baishilong/benchmarks/MemoryAgentBench
git checkout 455306dcabc3842526eb83cd4e225e5d486c5c5d
```

Record the commit SHA in every run manifest. Do not clone under the MemGen repository, vendor the benchmark, or add it as a submodule during MAB-1.

### Fork decision

Do **not** fork now. The first three phases can use a pinned upstream checkout plus an external adapter. A fork is justified only if all of the following become true:

- MAB-1 verifies the official pipeline.
- MAB-2 verifies disabled equivalence.
- The external adapter cannot maintain the required output/metric contract through public utilities.
- A small, reviewable upstream-harness change is clearly more stable than continued external orchestration.

### Conda environment

Use a separate environment for the official harness:

- Recommended name: `MABench`
- Recommended Python: `3.10.16`, matching the upstream README.

README installation commands:

```bash
conda create --name MABench python=3.10.16
conda activate MABench
pip install torch
pip install -r requirements.txt
pip install "numpy<2"
```

The upstream requirements are broad and unpinned. They include PyTorch/Transformers, explicit CUDA 12 wheels, `deepspeed`, `bitsandbytes`, `faiss-gpu`, LangChain packages, multiple provider SDKs, `mem0ai`, `minference`, and `flash_attn`. The README separately warns that HippoRAG requires a conflicting OpenAI version and that Cognee/Letta may require package workarounds. Additional risks include:

- PyTorch/CUDA wheel incompatibility with the host driver.
- `flash_attn` build failures and ABI coupling to the installed PyTorch/CUDA versions.
- `faiss-gpu` package availability and CUDA compatibility.
- OpenAI/LangChain API drift because versions are not pinned.
- NLTK `punkt` being downloaded lazily during chunking, which can fail offline.
- `numpy>=2` incompatibilities, hence the README's explicit `numpy<2` step.
- Provider/API packages that are irrelevant to the selected smoke but imported transitively.

Before installing, capture an environment lock after a successful setup (`conda env export` and `pip freeze`) and record GPU/PyTorch compatibility. Do not install the full MemoryAgentBench requirements into the current MemGen environment. That would create unnecessary risk to the accepted MemGen checkpoint/runtime and make disabled-equivalence failures difficult to interpret.

### Integration process boundary

The preferred later architecture is an **external adapter controller with a long-lived subprocess or local service boundary**:

- The controller uses a minimal MemoryAgentBench-compatible environment to load/filter data, chunk contexts, format prompts, and score outputs.
- A long-lived worker runs in the existing MemGen environment, loads the checkpoint once, and accepts structured session/chunk/query/reset requests.
- Communication is JSONL over stdin/stdout or a localhost-only HTTP/Unix-socket protocol.
- The worker returns generated text, token lengths, timing, and debug counters; latent tensors never cross the process boundary.

This is preferable to calling the official CLI for MemGen integration because `main.py` constructs the built-in `AgentWrapper`, which has no MemGen backend. It is also preferable to importing all MemoryAgentBench dependencies into MemGen. For MAB-1 only, use the official CLI unchanged.

## 4. Dataset Preparation Plan

### Dataset identity and splits

- HuggingFace dataset: `ai-hyz/MemoryAgentBench`
- Loader call in the inspected code: `load_dataset("ai-hyz/MemoryAgentBench", split=<split>, revision="main")`
- Splits:
  - `Accurate_Retrieval`
  - `Test_Time_Learning`
  - `Long_Range_Understanding`
  - `Conflict_Resolution`

For reproducibility, record the HuggingFace dataset revision or resolved cache commit in run manifests. The upstream loader currently requests mutable revision `main`; the external controller should pin the resolved dataset revision once MAB-1 establishes a working snapshot.

### Expected sample structure

Each loaded item is expected to contain:

- `context`: one long text string.
- `questions`: a string or list, normalized to a list.
- `answers`: a string/list or nested answer list, normalized by benchmark utilities.
- `metadata`: includes `source`, used to filter the requested `sub_dataset`; it may also contain `question_dates`, `question_types`, `question_ids`, `previous_events`, `qa_pair_ids`, and `demo`.

The external adapter must preserve `qa_pair_id` when present. It must not reorder questions, answers, chunks, or contexts.

### Session and turn mapping

```text
one dataset item / context
    -> one fresh MemGen session and one fresh latent memory bank
context string
    -> official sentence-aware chunks, in original order
each chunk
    -> one memorization prompt/turn (`memorizing=True`)
each question
    -> one query prompt/turn (`memorizing=False`) in the same session
answer(s)
    -> metric target only; never included in model input or memory
end of item or exception
    -> unconditional session reset and tensor cleanup
```

Chunking should initially reuse `ConversationCreator` and `chunk_text_into_sentences()` exactly: NLTK sentence splitting, `tiktoken` token counting, default `chunk_size: 4096`, no overlap, and original order. Chunk-size changes belong in an explicit ablation, not the first comparison.

All questions for a context are answered after the context chunks have been ingested. To preserve the benchmark's "inject once, query multiple times" contract and prevent query-to-query contamination, MAB-2/MAB-3 should define query turns as retrieval-and-answer turns with memory writes suppressed at the adapter boundary. If the current MemGen inference API cannot suppress query-time writes without core changes, stop and document the mismatch rather than silently changing the protocol. A later ablation may evaluate online query-time updates separately.

### Cache verification without a full experiment

Use metadata/filesystem inspection only:

```bash
hf cache ls --filter "repo_id=ai-hyz/MemoryAgentBench"
test -d "${HF_HOME:-$HOME/.cache/huggingface}/hub/datasets--ai-hyz--MemoryAgentBench"
find "${HF_HOME:-$HOME/.cache/huggingface}/datasets" \
  -maxdepth 4 -iname '*MemoryAgentBench*' -print
```

Then perform a loader-only validation in MAB-1, not a model run: load one configured split, filter by `metadata.source`, inspect `column_names`, and print only counts/types and the resolved cache revision. Do not print full contexts or answers into logs. Cache presence alone is not sufficient; loader validation must confirm the four split names and at least one item for the selected `sub_dataset`.

## 5. Task Subset Selection Plan

Names below distinguish repository YAML names from `sub_dataset` values where they differ.

| Task group | Candidate config/task name | Metric | Requires API or LLM judge? | Estimated cost | Relevance to MemGen | Priority | Use in phase |
|---|---|---|---|---|---|---|---|
| Conflict Resolution | `Factconsolidation_sh_6k.yaml` / `factconsolidation_sh_6k` | `substring_exact_match` (also generic EM/F1/ROUGE fields) | No judge; model backend only | Very low; 1 context, 6K, 10-token generation | Direct test of stale fact replacement and recency | P0 | MAB-1 smoke; MAB-4 first comparison |
| Conflict Resolution | `Factconsolidation_mh_6k.yaml` / `factconsolidation_mh_6k` | `substring_exact_match` | No judge; model backend only | Very low; 1 context, 6K | Direct multi-hop/conflict consolidation test | P0 | MAB-4 first comparison |
| Accurate Retrieval | `Eventqa_64k.yaml` / `eventqa_65536` | Paper/README primary: `substring_exact_match`; code also emits `eventqa_recall` | No judge; model backend only | Medium; up to 5 contexts, 64K | Direct retrieval from longer incremental event history | P1 | MAB-4 second wave after 6K tasks |
| Test-Time Learning | `ICL_trec_coarse.yaml` / `icl_trec_coarse_6600shot_balance` | `exact_match` | No judge; model backend only | Medium/high; 131K, up to 100 samples | Tests retained label mapping/task induction, with reasoning confound | P2 | MAB-6, one selected ICL task |
| Accurate Retrieval | `Ruler_qa1_197k.yaml` / `ruler_qa1_197K` | `substring_exact_match` | No judge; model backend only | High; 197K, up to 100 samples | Useful exact retrieval stress test | P2 | MAB-6 only if context/runtime feasible |
| Long-Range Understanding | `detectiveQA` (repository config `Detective_QA.yaml`, source `detective_qa`) | `exact_match` | No judge; model backend only | High; 200K, 2K-token output, up to 10 samples | Tests distributed evidence integration, less memory-specific | P2 | MAB-6 selected feasibility run |
| Accurate Retrieval | `Longmemeval_s.yaml` or `Longmemeval_s_star.yaml` | Paper-faithful GPT-4o LLM judge; generic code metrics are not equivalent | Yes, GPT-4o judge | High; 150K-400K and up to 500 samples | Relevant retrieval, but judge and protocol add confounds | P3 | Avoid initially; later validation only |
| Long-Range Understanding | `InfBench_sum.yaml` / `infbench_sum_eng_shots2` | HELMET-style F1 via LLM judge | Yes, GPT-4o-style judge | Very high; 1.2M context, 1.2K-token output, up to 100 samples | Weak initial fit to small-slot latent retrieval | P3 | Avoid initially |
| Test-Time Learning | `Recsys_redial_full.yaml` / `recsys_redial_full` | `Recall@5` primary; code emits Recall@1/5/10 | No judge, but requires `entity2id.json` and specialized parsing | Very high; 1.48M context | Interesting preference learning but specialized and costly | P3 | Avoid initially |

Selection order:

1. First: `Factconsolidation_sh_6k`, then `Factconsolidation_mh_6k`.
2. Second: `EventQA_64k` (`sub_dataset: eventqa_65536`).
3. Later: one ICL task, preferably TREC coarse.
4. Later if feasible: `ruler_qa1_197K` and/or `detective_qa`.
5. Avoid initially: LongMemEval, InfBench summarization, Recsys, 262K FactConsolidation, 421K RULER, and all 1M+ settings.

## 6. Official Smoke Test Plan

### Phase MAB-1 goal

Run the official MemoryAgentBench pipeline without MemGen to validate environment, dataset loading, chunking, prompting, one backend query, automatic scoring, resume behavior, and output serialization.

Selected task:

- Data config: `configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml`
- `dataset: Conflict_Resolution`
- `sub_dataset: factconsolidation_sh_6k`
- `max_test_samples: 1` already in the config
- `chunk_size: 4096`
- Primary metric: `substring_exact_match`

Use an officially supported long-context agent for this smoke only. The smallest practical upstream path is `configs/agent_conf/Long_Context_Agents/Long_context_agent_gpt-4o-mini.yaml`, which requires `OPENAI_API_KEY` (or configured Azure OpenAI variables). This API use validates the harness; it is not a benchmark baseline and must not be included in the MemGen comparison table.

Exact planned command:

```bash
conda run -n MABench python main.py \
  --agent_config configs/agent_conf/Long_Context_Agents/Long_context_agent_gpt-4o-mini.yaml \
  --dataset_config configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml \
  --max_test_queries_ablation 1 \
  --chunk_size_ablation 4096 \
  --force
```

Relevant controls:

- `max_test_samples`: YAML field; `1` in the selected config.
- `--max_test_queries_ablation 1`: stop after the first global query.
- `--chunk_size_ablation 4096`: explicit but equal to the config default; useful for manifest clarity.
- `--force`: rerun rather than resume an existing partial result. Use only after preserving/removing the prior smoke artifact deliberately.

The official CLI has no `--output_dir` flag. With the unmodified upstream agent config, the expected path is:

```text
./outputs/gpt-4o-mini/Conflict_Resolution/
factconsolidation_sh_6k_unknown_in6000_size10_shots0_max_samples1_results.json
```

After validation, copy the immutable result plus a manifest into:

```text
outputs/mab/official_smoke/
```

Alternatively, use an external copied agent YAML whose only change is `output_dir: /mnt/18T/baishilong/MemGen/outputs/mab/official_smoke`; do not edit the upstream YAML. In that case the JSON remains under an additional `Conflict_Resolution/` subdirectory because `_create_output_path()` appends the dataset name.

### Expected JSON schema

Top-level object:

```json
{
  "agent_config": {},
  "dataset_config": {},
  "data": [],
  "metrics": {},
  "time_cost": [],
  "averaged_metrics": {}
}
```

Each `data` record should contain at least:

```json
{
  "output": "...",
  "input_len": 0,
  "output_len": 0,
  "memory_construction_time": 0,
  "query_time_len": 0,
  "parsed_output": "...",
  "exact_match": false,
  "f1": 0.0,
  "substring_exact_match": false,
  "rougeL_f1": 0.0,
  "rougeL_recall": 0.0,
  "rougeLsum_f1": 0.0,
  "rougeLsum_recall": 0.0,
  "answer": ["..."],
  "query": "...",
  "query_id": 0,
  "qa_pair_id": "..."
}
```

`qa_pair_id` is optional when absent from the source. `metrics` contains per-query arrays; `averaged_metrics` contains means, with non-length/non-time values multiplied by 100 by the official serializer.

### Success criteria

- Pinned repository commit and environment manifest are recorded.
- Dataset split loads and filters to `factconsolidation_sh_6k` with exactly one selected context.
- Context chunking produces non-empty, ordered chunks and no empty first chunk.
- Exactly one query is sent to the backend.
- Exactly one result record is written.
- JSON parses and contains all required top-level keys.
- `substring_exact_match` exists in the record, metric arrays, and averaged metrics.
- Input/output lengths and query timing are finite and non-negative.
- The saved prediction and answer can be rescored by `metrics_summarization()` without schema conversion.
- No MemGen package, checkpoint, process, or code path is involved.

### Failure handling

- Dataset/cache error: stop MAB-1; record split, source filter, HuggingFace revision, cache path, and traceback. Do not start adapter work.
- Dependency/import error: repair only the isolated `MABench` environment, record the final lock, and rerun the same command.
- API/auth/rate-limit error: preserve logs, verify provider variables without printing secrets, and rerun only the one-query smoke.
- NLTK download error: pre-cache `punkt` in the isolated environment; do not alter chunking code.
- Empty dataset after filtering: verify exact case-sensitive `metadata.source`; do not substitute another task silently.
- Malformed/missing JSON metric: stop and compare the checked-out commit to the inspected snapshot before any adapter design is treated as executable.
- Any partial output must be archived with a `.failed` manifest rather than reused as a successful resume point.

## 7. MemGen Adapter Design Plan

This section specifies MAB-2/MAB-3 architecture only. It does not authorize implementation in this phase.

### Preferred architecture

Create a standalone integration package outside MemGen and MemoryAgentBench, for example:

```text
/mnt/18T/baishilong/integrations/memgen-memoryagentbench/
```

Logical components:

- **MAB controller:** imports pinned `ConversationCreator`, templates, and metric utilities from the benchmark checkout.
- **Protocol layer:** sends `start_session`, `memorize`, `query`, `debug_snapshot`, and `reset_session` requests.
- **MemGen worker:** runs in the existing MemGen environment, loads one checkpoint once, owns exactly one active session bank, and invokes existing inference APIs.
- **Result writer:** emits official-compatible JSON plus a sidecar diagnostic JSONL and immutable run manifest.

Prefer importing MemoryAgentBench's data/chunk/template/metric utilities over calling its CLI for MemGen runs. The CLI hardcodes built-in `AgentWrapper` construction and agent persistence. Reimplementing metric logic is also undesirable because it risks score drift. The external controller should pin the upstream commit and add only a thin normalization layer.

### Interaction mapping

| MemoryAgentBench object | Adapter action |
|---|---|
| One sample/context | `start_session(context_id)` with a fresh bank; assert no active prior session |
| One context chunk | Format with the official memorize template and call `memorize(chunk)` sequentially |
| One question/query | Format with the official query template and call `query(prompt, query_id)` |
| One answer/answer list | Keep outside model input; pass only to official post-processing/metrics |
| End/failure | `reset_session()` in `finally`, synchronize CUDA if needed, release session tensors |

### `memorizing=True`

- Accept exactly one chunk and preserve official chunk order.
- Format it as one complete prompt ending at the chunk boundary.
- Invoke the existing inference path with `batch_size=1`; do not manually call or relocate Trigger.
- Let the existing prompt-final Trigger decision determine whether Weaver runs and a reasoner-space memory is produced.
- If Trigger is positive, store the post-projection reasoner-space `latent_inputs_embeds` using the existing Version A bank/write-back contract.
- Return an internal acknowledgement such as `"Memorized"`; do not score it and do not append generated acknowledgement text to memory.
- Record whether Trigger fired, whether Weaver ran, whether a write occurred, and the resulting slot count. Trigger-positive-without-write must be explicit, not treated as success.

### `memorizing=False`

- Use the same checkpoint, tokenizer, decoding configuration, prompt template, and session as memorization.
- Disabled mode calls the accepted original generation path with no bank construction, retrieval, update, reset side effect, extra wrapping, padding, or prompt change beyond the benchmark query prompt itself.
- Enabled Version A retrieves from the session bank and injects retrieved reasoner-space latents only into the Reasoner-side candidate path.
- Do not concatenate retrieved text or latent representations into the benchmark prompt.
- Suppress memory writes during query turns for the initial inject-once/query-many protocol. Retrieval and answer generation remain enabled. Log the prompt-final Trigger decision even when its write is suppressed.
- Return the generated answer and accounting dictionary expected by MemoryAgentBench.

### Required output dictionary

For every query, return at least:

```python
{
    "output": str,
    "input_len": int,
    "output_len": int,
    "memory_construction_time": float,
    "query_time_len": float,
}
```

`input_len` and `output_len` should use the MemGen tokenizer and be documented as model-token counts. `memory_construction_time` should be cumulative context-ingestion time for that session and remain constant across its queries; `query_time_len` should cover retrieval plus answer generation for the current query. The official metric utility will add parsed output and metric fields.

### Non-negotiable invariants

- Hard assert `batch_size == 1` before creating an enabled session.
- One active `context_id` at a time; reject a new context until reset completes.
- Construct the bank inside session scope, never module/global scope.
- Reset in `finally` after every context, including exceptions and interrupted query loops.
- Assert slot count is zero before the first chunk of every context.
- Never serialize or restore MemGen session memory through MemoryAgentBench `agent_save_folder`.
- Retrieved memory may enter only the Reasoner. Weaver input length/content hashes must match the no-retrieval current-context input for the same turn.
- Stored values must be detached/cloned reasoner-space `latent_inputs_embeds`, not Weaver hidden states and not text embeddings.
- No changes to Weaver training, Trigger training, or their checkpoints.
- Disabled mode must not instantiate, retrieve, write, or reset a bank and must match the accepted MemGen disabled path.
- Use unique run/context/query identifiers in protocol messages and reject stale/out-of-order responses.
- After reset, assert zero slots, zero active session handles, and no retained context-specific cache references.

### Trigger-boundary preservation

Each chunk is one complete memorization prompt. The adapter must not split a chunk into token windows after official chunking, append synthetic query text after the chunk, force a Trigger-positive label, or invoke Trigger per generated token. The expected construction diagnostic is one prompt-final Trigger decision opportunity per memorization chunk. Query prompts may also reach the normal prompt-final Trigger site, but initial query-time writes are suppressed as stated above.

## 8. Baseline Plan

### Required paired baselines

1. **MemGen disabled:** `latent_memory_bank.enabled=false`; no bank construction or memory operation. This is both a benchmark baseline and a regression control.
2. **MemGen enabled Version A:** session-local reasoner-space bank; retrieval injected into the Reasoner only; retrieved memory never enters Weaver; `batch_size=1`.

Hold constant across the pair:

- MemGen checkpoint, tokenizer, prompt templates, official chunks, question order, decoding parameters, seed, maximum generation length, dataset revision, and sample/query IDs.
- Hardware assignment where practical.
- Query-time write policy.
- Result scorer and output normalization.

### Recommended ablations

Run only after the required pair is stable:

- **No recency:** set decay influence to neutral while preserving retrieval and update semantics.
- **No replacement:** retain writes until capacity, then do not replace; define overflow behavior explicitly before running.
- **Top-k only:** remove threshold filtering and retrieve fixed top-k slots.
- **Threshold variations:** pre-register a small grid around the default threshold; do not tune on test answers.
- **Capacity variations:** use a small pre-registered set such as low/default/high capacity and report slot utilization.

Every ablation must change one mechanism at a time and preserve the same task/sample/query set. The exact values should be finalized from the accepted default config in MAB-5, not chosen post hoc from test accuracy.

### Optional external baselines

- BM25 over the same official chunks.
- Embedding RAG over the same chunks with a fixed local embedding model and matched retrieval `k`.

These are useful later to distinguish latent-memory gains from ordinary text retrieval. They should use the same answer model if the comparison claim concerns memory rather than model capability.

### Avoid initially

- Mem0, Zep, Cognee, and Letta: substantial service/dependency/configuration surface and different persistence semantics.
- GPT-4o, Claude, Gemini, or other commercial long-context baselines: API cost, model drift, unmatched backbone capability, and limited diagnostic comparability.

The project does not need to reproduce all paper baselines initially. The first scientific question is narrower: does enabled Version A improve over the exact same MemGen model with memory disabled while preserving original ability and isolation? Broad baseline reproduction is justified only after the adapter is validated and the paired effect is measurable. Otherwise, engineering/API variance would dominate before MemGen's own causal comparison is trustworthy.

## 9. Logging and Diagnostics Plan

Write one structured JSONL record per chunk and per query, plus a context-end summary. Required fields:

- `context_id`
- `query_id` (null for chunk events)
- `task_name`
- `split`
- `sub_dataset`
- `chunk_count`
- `prompt_final_trigger_count`
- `trigger_positive_count`
- `weaver_call_count`
- `write_count`
- `memory_slot_count`
- `replacement_count`
- `retrieval_count`
- `retrieved_latents` (count/shape/dtype summary only, not raw tensors)
- `top_retrieval_scores`
- `input_len`
- `output_len`
- `prediction`
- `ground_truth_answers`
- `metric_fields`
- `latency` (separate load, memory construction, retrieval, generation, scoring, and total where available)
- `peak_cuda_memory_bytes` if CUDA memory statistics are available

Also record:

- Run ID, git SHA for both repositories, dataset revision, checkpoint identifier/hash, config snapshot, seed, device, CUDA/PyTorch/Transformers versions, and decoding parameters.
- Per-event slot count before/after, retrieved indices, replacement/eviction reason, threshold, top-k, capacity, and whether query-time write suppression was active.
- Weaver input token count or stable shape/hash evidence sufficient to prove retrieved memory did not enter Weaver.
- Reset outcome and post-reset slot count.
- Disabled-path counters, which must all remain zero for retrieval/write/bank creation.

Raw latent tensors must not be logged. If `retrieved_latents` is required for diagnosis, log only count, shape, dtype, norm summaries, and opaque slot IDs.

Accuracy alone is insufficient because the same score can arise from no Trigger activation, Trigger activation without a write, writes that are never retrieved, retrievals that are injected into the wrong component, prompt leakage, or cross-context contamination. Diagnostics are required to establish mechanism validity, identify silent no-op runs, separate memory construction from retrieval quality, and prove the disabled/Reasoner-only constraints.

## 10. Result Organization Plan

Required output roots:

```text
outputs/mab/official_smoke/
outputs/mab/memgen_disabled/
outputs/mab/memgen_enabled_vA/
outputs/mab/ablations/
```

Recommended per-run layout:

```text
outputs/mab/<phase_or_mode>/<task>/<run_id>/
  manifest.json
  results.json
  diagnostics.jsonl
  stdout.log
  stderr.log
  environment.txt
  config/
```

`manifest.json` is the source of truth for commit SHAs, dataset revision, checkpoint, mode, task, sample/query limits, and completion status. Partial runs must have `status: failed` or `status: interrupted`; they must not be merged into complete aggregates.

Research notes:

- `research_notes/benchmarks/memoryagentbench_configuration_plan.md`: this configuration and phase-gate plan.
- `research_notes/benchmarks/memoryagentbench_smoke_test.md`: MAB-1 command, environment, schema validation, failures, and decision.
- `research_notes/benchmarks/memoryagentbench_adapter_plan.md`: post-smoke interface contract, test matrix, exact external file layout, and implementation approval gate.
- `research_notes/benchmarks/memoryagentbench_results.md`: paired results, diagnostics, ablations, limitations, and supported claims.

Do not overwrite prior runs. Use deterministic run IDs containing date, mode, task, dataset revision prefix, and checkpoint/config identifier.

## 11. Risk and Stop Criteria

### Risks

- **Dependency conflicts:** broad unpinned requirements can destabilize PyTorch/CUDA and provider clients.
- **API dependence:** official supported agents and judge tasks may need paid, mutable APIs.
- **LLM judge tasks:** evaluator variance, cost, version drift, and difficult reproducibility.
- **Long-context cost:** many chunk-level MemGen calls plus long generations can dominate runtime and memory.
- **Adapter invasiveness:** modifying either core repository would enlarge the regression surface and obscure responsibility.
- **Prompt/format mismatch:** MemGen may respond poorly to memorize acknowledgements or benchmark answer formats.
- **Metric strictness:** ICL exact match rejects outputs such as `label: 43` when the target is `43`; output formatting can dominate score.
- **Trigger fires but no effective write:** Trigger-positive does not prove a valid latent was stored.
- **Memory write but no useful retrieval:** slot growth can coexist with zero/irrelevant retrieval.
- **Cross-sample leakage:** persistent process state, agent folders, caches, or global banks may contaminate later contexts.
- **Disabled path divergence:** adapter wrapping may alter prompts, tensor shapes, timing, or generation even with memory disabled.
- **Query-to-query leakage:** allowing query turns to write may expose later questions to earlier queries/answers.
- **Mutable upstream data/code:** `main` revisions can silently change schema or metrics.

### Mandatory stop criteria

- If the official MAB-1 smoke cannot run, stop and report before adapter work.
- If the selected dataset split/source cannot load, stop and report.
- If the disabled adapter changes accepted original MemGen behavior, stop MAB-2.
- If enabled memory leaks across contexts, stop MAB-3 and invalidate affected results.
- If retrieved memory enters Weaver, stop; the run is not Version A.
- If the output JSON cannot be consumed and scored by official utilities, stop.
- If enabled mode uses `batch_size != 1`, stop before inference.
- If Trigger/write/retrieval counters cannot be observed reliably, do not interpret accuracy as memory evidence.
- If query-time write suppression requires MemGen-core changes, stop and revise the adapter protocol before implementation.
- If a partial/resumed run cannot prove context memory reconstruction consistency, discard it and rerun from a clean session.

Each stop report must include the phase, exact command/config, source revisions, minimal traceback or invariant failure, affected artifacts, and the decision needed to continue.

## 12. Proposed Phase Schedule

| Phase | Objective | Scope | Exit gate |
|---|---|---|---|
| MAB-0 | Configuration plan | Produce this document; no execution | Plan covers environment, data, tasks, adapter contract, baselines, diagnostics, risks, and phases |
| MAB-1 | Official smoke test | Official checkout, `Factconsolidation_sh_6k`, 1 context, 1 query, no MemGen | Valid official-compatible JSON and automatic metric; environment/dataset revisions recorded |
| MAB-2 | MemGen disabled adapter | External controller/worker, same smoke item, memory fully disabled | Output scores; disabled path matches accepted behavior and all memory counters remain zero |
| MAB-3 | MemGen enabled Version A adapter | Same item, fresh per-context bank, Reasoner-only retrieval, batch size 1 | Writes/retrieval/reset observed; no Weaver injection or leakage; official scorer accepts output |
| MAB-4 | Small-scale comparison | Disabled vs enabled on SH 6K, MH 6K, then EventQA 64K | Paired complete runs with mechanism diagnostics and preservation checks |
| MAB-5 | Ablations | No recency, no replacement, top-k only, threshold and capacity variations | One-factor-at-a-time results with fixed sample/query set and pre-registered values |
| MAB-6 | Expanded subset | One ICL task, selected RULER/DetectiveQA, longer CR only if feasible | Costs and failure rate acceptable; claims remain scoped; no initial-avoid task added without a new gate |

No phase advances solely because a command completes. Every exit gate requires artifact/schema validation and an explicit go/stop decision in the corresponding research note.

## 13. Final Recommendation

Proceed with MemoryAgentBench as MemGen's main memory-specific benchmark, complemented by GSM8K and TriviaQA for original-ability preservation.

Do first:

1. Pin the inspected upstream commit and dataset revision.
2. Create the isolated `MABench` Python 3.10.16 environment outside the current MemGen environment.
3. Run MAB-1 only: official `Factconsolidation_sh_6k`, one context, one query, automatic scoring.
4. Record the exact JSON schema and environment in `memoryagentbench_smoke_test.md`.
5. Only after MAB-1 passes, write the detailed adapter implementation/test plan for MAB-2.

Do not yet:

- Implement or modify the MemGen adapter.
- Modify MemGen, Weaver training, Trigger training, or MemoryAgentBench.
- Run the full benchmark or long 262K/421K/1M+ settings.
- Use LongMemEval or InfBench LLM judges.
- Reproduce Mem0/Zep/Cognee/Letta or commercial long-context baselines.
- Tune thresholds/capacity against benchmark test answers.

Do not fork MemoryAgentBench now. Use a pinned external checkout and an external controller. Do not install MemoryAgentBench requirements into the current MemGen environment. Keep the official harness isolated, and later keep MemGen inference in its existing environment behind a controlled long-lived subprocess/service boundary.
