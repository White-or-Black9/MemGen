# MemoryAgentBench Feasibility Assessment for MemGen

Date: 2026-06-18
Branch context: `rlm-memory-bank`
Assessment mode: read-only feasibility and compatibility assessment

Sources inspected:
- Paper: https://arxiv.org/abs/2507.05257 and PDF https://arxiv.org/pdf/2507.05257
- Code: https://github.com/HUST-AI-HYZ/MemoryAgentBench
- Local clone for inspection only: `/tmp/MemoryAgentBench`
- Local clone commit: `455306dcabc3842526eb83cd4e225e5d486c5c5d`

No MemGen code, MemoryAgentBench code, training code, evaluation code, model files, commits, or experiments were modified or run.

## Executive Summary

Recommendation: use with caution.

MemoryAgentBench is scientifically relevant because it evaluates incremental memory construction, query-time retrieval/use, long-range consolidation, and conflict/update behavior. This is closer to MemGen's session-level latent memory bank than ordinary static QA benchmarks. The benchmark is not a direct fit out of the box because it assumes an agent interface that ingests many textual chunks through explicit "memorize this" calls before later query calls. MemGen can fit this if each benchmark context is treated as one MemGen session, chunks are processed sequentially with `batch_size=1`, and the adapter writes reasoner-space `latent_inputs_embeds` without exposing latent memory as text.

The best first subset is small, automatic-metric, single-context tasks:
- `Conflict_Resolution/Factconsolidation_sh_6k.yaml`
- `Conflict_Resolution/Factconsolidation_mh_6k.yaml`
- `Accurate_Retrieval/EventQA/Eventqa_64k.yaml`
- optionally one ICL task such as `Test_Time_Learning/ICL/ICL_trec_coarse.yaml`

Avoid initially:
- `infbench_sum_eng_shots2`: very long context, long generation, GPT-4o judge.
- `longmemeval_s*` and `longmemeval_s_-1_500`: GPT-4o judge path.
- `recsys_redial_full`: needs `processed_data/Recsys_Redial/entity2id.json` and has 1.48M context length.
- Full 262K conflict-resolution and 421K RULER tasks until a short smoke confirms adapter correctness.

The preferred integration shape is an external adapter script or thin external adapter package that imports MemoryAgentBench data/metric utilities and MemGen inference, rather than forking either repository immediately. A fork is only needed after MAB-1 official smoke and MAB-2 disabled MemGen adapter prove the flow.

## Paper Summary

MemoryAgentBench is designed to evaluate memory agents through incremental multi-turn interactions rather than static long-context input. The paper argues that memory agents should process information piece by piece, abstract or store useful information, update memory, and later answer questions from that memory.

The paper identifies four core memory competencies:
- Accurate Retrieval (AR): find the relevant stored information for a query.
- Test-Time Learning (TTL): learn new mappings, behaviors, labels, or recommendations during deployment without training.
- Long-Range Understanding (LRU): integrate information distributed across very long histories, often 100K tokens or more.
- Selective Forgetting (SF): update, overwrite, or ignore stale/conflicting information. The code and README often label this group as `Conflict_Resolution`.

Most relevant to MemGen:
- Accurate Retrieval is directly relevant to whether latent memory stores useful per-session evidence.
- Conflict Resolution / Selective Forgetting is directly relevant to decay, replacement, recency, and stale memory handling.
- Test-Time Learning is relevant if MemGen can store repeated examples or task rules in latent memory and apply them later.

Weakly aligned or lower-priority:
- Long-Range Understanding may test global summarization and broad narrative integration more than the current small-slot top-k latent memory mechanism.
- Recsys TTL is behaviorally interesting but requires specialized output parsing and local entity mapping.
- LLM-as-judge tasks add API cost and evaluator variance; they are not ideal for first MemGen evidence.

Baselines compared in the paper/code include:
- Long-context agents: GPT, Claude, Gemini, and related long-window models.
- Simple RAG: BM25.
- Embedding RAG: Contriever, OpenAI text embeddings, Qwen3 embedding.
- Structure-augmented RAG: GraphRAG, RAPTOR, HippoRAG, MemoRAG, Cognee, Zep.
- Agentic memory systems: Letta, Mem0, Self-RAG-style agentic retrieval.

Metrics:
- Accuracy via exact match or substring exact match for most QA/classification tasks.
- Recall@5 for movie recommendation.
- F1 / LLM-as-judge for InfBench summarization.
- GPT-4o LLM-as-judge for LongMemEval and InfBench summarization, per README and `llm_based_eval/`.

Benchmark assumptions:
- The agent accepts sequential textual memory-construction chunks.
- The agent can distinguish memory ingestion from query answering.
- A context/session can have multiple questions after one long ingestion pass.
- Memory may be text history, vector memory, external database, graph memory, or agentic memory. The paper explicitly discusses parameters, vectors, textual histories, and external databases, but the released code is mostly text/vector/API oriented.

## Code Map

Local inspected clone: `/tmp/MemoryAgentBench`.

Main files:
- `/tmp/MemoryAgentBench/main.py`
  - `parse_command_line_arguments()`: CLI args.
  - `process_context()`: one context/session loop.
  - `process_single_query()`: query call and metric update.
  - `save_results_to_file()`: writes JSON output after every query.
  - `main()`: top-level benchmark loop.
- `/tmp/MemoryAgentBench/initialization.py`
  - `setup_configs_and_directories()`: loads YAML configs and creates output path.
  - `create_agent_and_fetch_data()`: creates `ConversationCreator`.
  - `generate_agent_save_folder()`: per-context agent storage path.
  - `initialize_and_memorize_agent()`: creates agent, feeds context chunks, saves/loads agent.
  - `_memorize_context_chunks()`: sequentially calls `agent.send_message(chunk, memorizing=True)`.
- `/tmp/MemoryAgentBench/conversation_creator.py`
  - `ConversationCreator`: dataset loading, sample processing, chunking, query formatting.
  - `get_chunks()`: returns list of chunk lists, one list per benchmark context.
  - `get_query_and_answers()`: returns list of query/answer/qa_pair_id lists.
- `/tmp/MemoryAgentBench/agent.py`
  - `AgentWrapper`: central agent abstraction.
  - `send_message(message, memorizing=False, query_id=None, context_id=None)`: custom agent interface.
  - `_handle_long_context_agent()`, `_handle_memory_agent()`, `_handle_rag_agent()`: route to implementations.
- `/tmp/MemoryAgentBench/utils/eval_data_utils.py`
  - `load_data_huggingface()`: loads `ai-hyz/MemoryAgentBench`.
  - `_load_and_filter_dataset()`: loads split and filters by `metadata.source`.
  - `_process_single_sample_qa_lists()`: standardizes `questions`, `answers`, and metadata fields.
- `/tmp/MemoryAgentBench/utils/eval_other_utils.py`
  - `chunk_text_into_sentences()`: benchmark chunker.
  - `calculate_metrics()`: exact match, F1, substring exact match, ROUGE.
  - `post_process()`: dataset-specific metric routing.
  - `metrics_summarization()`: appends per-query results and running metrics.
- `/tmp/MemoryAgentBench/utils/templates.py`
  - `BASE_TEMPLATES`: memorize/query/system prompts for task families.
  - `get_template()`: maps sub-dataset and agent type to prompt templates.
- `/tmp/MemoryAgentBench/llm_based_eval/longmem_qa_evaluate.py`
  - GPT-4o judge for LongMemEval.
- `/tmp/MemoryAgentBench/llm_based_eval/summarization_evaluate.py`
  - GPT-4o-style judge prompts for InfBench summarization.

Important configs:
- Data configs under `/tmp/MemoryAgentBench/configs/data_conf/`.
- Agent configs under `/tmp/MemoryAgentBench/configs/agent_conf/`.
- Bash wrappers under `/tmp/MemoryAgentBench/bash_files/sh/`.
- README says the project recommends a dedicated Python 3.10.16 conda environment and warns of dependency conflicts around HippoRAG, OpenAI versions, Cognee, and Letta.

## Dataset and Sample Format

Primary dataset loader:
- HuggingFace dataset: `ai-hyz/MemoryAgentBench`.
- Splits: `Accurate_Retrieval`, `Test_Time_Learning`, `Long_Range_Understanding`, `Conflict_Resolution`.
- Filtering: `_load_and_filter_dataset()` filters examples by `sample["metadata"]["source"] == sub_dataset`.

Processed sample fields expected by the harness:
- `context`: long text for one memory-construction session.
- `questions`: list or scalar, normalized to list.
- `answers`: list or scalar, normalized to list.
- `metadata.source`: used to select sub-dataset.
- metadata fields normalized when present: `question_dates`, `question_types`, `question_ids`, `previous_events`, `qa_pair_ids`, `demo`.

One dataset item maps to:
- one context string,
- one list of context chunks,
- one list of query-answer pairs.

Each context may have multiple questions. This matches the paper's "inject once, query multiple times" design.

Dataset/sample limiting:
- Context/sample count is config-driven through `max_test_samples`.
- Query count can be limited by CLI `--max_test_queries_ablation`.
- `--chunk_size_ablation` overrides chunk size.

I did not download or run the dataset. Dataset sizes below are from the paper/code/configs, not from a local cache check.

## Chunking and Interaction Flow

Context is stored as whole text in the dataset, then chunked by the benchmark.

Chunking path:
- `ConversationCreator.get_chunks()` calls `chunk_text_into_sentences(context, chunk_size=self.chunk_size)`.
- `chunk_text_into_sentences()` uses NLTK sentence splitting and `tiktoken` token counting.
- No overlap is used in the main benchmark chunker.
- Default chunk size in data configs is usually `4096`.
- Memory agents may use `agent_chunk_size`; `ConversationCreator._determine_chunk_size()` asserts this is only for `mem0`, `letta`, `cognee`, or `zep`.
- CLI `--chunk_size_ablation` can override `dataset_config["chunk_size"]` and, for certain memory agents, `agent_config["agent_chunk_size"]`.

Interaction flow:
1. `main.py` loads agent and dataset configs.
2. `ConversationCreator` loads dataset examples and chunks each context.
3. For each context index, `process_context()` creates or loads a per-context agent.
4. `initialize_and_memorize_agent()` calls `_memorize_context_chunks()` unless the agent folder exists.
5. `_memorize_context_chunks()` calls `agent.send_message(chunk, memorizing=True)` for every chunk.
6. Then each query is sent as `agent.send_message(query, memorizing=False, query_id=..., context_id=...)`.
7. `metrics_summarization()` computes metrics and appends result records.
8. `save_results_to_file()` writes JSON after every query.

Answers are saved to:
- `os.path.join(agent_config["output_dir"], dataset_config["dataset"], f"{name_tag}_results.json")`.

RAG retrieved contexts may additionally be saved under:
- `/tmp/MemoryAgentBench/outputs/rag_retrieved/...`

## Agent Interface Requirements

The minimal custom agent contract is:

```python
class AgentWrapper:
    def send_message(
        self,
        message,
        memorizing=False,
        query_id=None,
        context_id=None,
    ):
        ...
```

For memorization:
- input: one text chunk.
- call: `send_message(chunk, memorizing=True)`.
- current built-ins usually return `"Memorized"` or `""`.

For query answering:
- input: one formatted question prompt.
- call: `send_message(query, memorizing=False, query_id=query_index, context_id=context_index)`.
- return value must be a dict with at least:
  - `output`
  - `input_len`
  - `output_len`
  - `memory_construction_time`
  - `query_time_len`

Best insertion options:
- Least invasive: an external adapter script that imports `ConversationCreator`, `metrics_summarization`, and MemGen inference code, bypassing `AgentWrapper`.
- Moderate: add a new `MemGenAgentWrapper` in MemoryAgentBench and a new YAML `agent_name`. This touches benchmark code.
- Avoid initially: modifying MemGen model/training/eval internals or adding MemoryAgentBench-specific behavior inside MemGen core code.

## Task Suitability Table

| Group | Task names/configs found | Metric | External API / judge | Expected difficulty for MemGen | Relevance | Priority | First-use decision |
|---|---|---|---|---|---|---|---|
| Accurate Retrieval | `ruler_qa1_197K`, `ruler_qa2_421K`, `longmemeval_s_-1_500`, `longmemeval_s*`, `eventqa_65536`, `eventqa_131072`, `eventqa_full` | `substring_exact_match`, `eventqa_recall`; LongMemEval also GPT-4o judge | EventQA/RULER automatic; LongMemEval requires judge script for paper-faithful scoring | Medium to high; depends on whether latent slots can retain exact snippets/events | High | High for EventQA/RULER small; low for LongMemEval initially | Include EventQA 64K first; later RULER; skip LongMemEval initially |
| Test-Time Learning | `icl_banking77_5900shot_balance`, `icl_clinic150_7050shot_balance`, `icl_nlu_8296shot_balance`, `icl_trec_coarse_6600shot_balance`, `icl_trec_fine_6400shot_balance`, `recsys_redial_full` | ICL `exact_match`; Recsys `Recall@5` plus `Recall@1/10` in code | ICL automatic; Recsys automatic but needs `entity2id.json` | Medium; ICL examples may fit latent memory if repeated labels are retained | Medium to high | Medium | Try one ICL task after AR/CR smoke; avoid Recsys first |
| Long-Range Understanding | `infbench_sum_eng_shots2`, `detective_qa` | DetectiveQA `exact_match`; InfBench summarization LLM judge / F1-style | InfBench requires judge; DetectiveQA automatic | High; summarization/global narrative may exceed small latent bank utility | Medium | Low to medium | Try DetectiveQA later; skip InfBench initially |
| Conflict Resolution | `factconsolidation_sh_6k/32k/64k/262k`, `factconsolidation_mh_6k/32k/64k/262k` | `substring_exact_match` | No judge needed | High but highly informative for decay/replacement | Very high | High | Include 6K SH/MH first, then 32K/64K |

## Adapter Feasibility Analysis

Feasibility: feasible with a non-trivial adapter.

Clean mapping:
- One MemoryAgentBench context should map to one MemGen session.
- Each context chunk should become one memory-construction turn/event.
- All questions for the context should be query turns within that same MemGen session.
- Memory must be reset after the context finishes, including after failures.
- Enabled mode must force `batch_size=1`.
- Disabled mode must bypass memory entirely and preserve original MemGen behavior.

MemGen-specific adapter requirements:
- Load a local MemGen checkpoint once, but create/reset a session-local memory bank per benchmark context.
- For `memorizing=True`, run a controlled inference path that produces reasoner-space `latent_inputs_embeds` suitable for memory writes, without training and without touching Weaver training or Trigger training.
- For `memorizing=False`, run normal answering with optional Reasoner-only retrieved-memory injection.
- Return the benchmark output dict expected by `metrics_summarization()`.
- Track input/output lengths in a comparable but not necessarily identical way. Token accounting may be approximate if MemGen tokenizer differs from `tiktoken`.

Compatibility with MemGen constraints:
- Reasoner-only injection is compatible if the adapter calls the existing Version A-aligned inference path.
- Retrieved memory must not enter Weaver. The adapter must not concatenate retrieved latent/text memory into benchmark context or Weaver prompts.
- Stored memory should remain latent reasoner-space `latent_inputs_embeds`; no text memory export is required by the benchmark.
- The benchmark is flexible enough to evaluate non-text latent memory because it only requires final answers and timing/length metadata. Its built-in agents are text/vector-oriented, but the external interface does not require memory to be inspectable as text.
- Batch size 1 is natural because the benchmark processes chunks and queries sequentially.
- Cross-sample leakage is avoidable if each context creates a fresh memory bank and no persistent agent folder is reused for MemGen enabled mode.

Potential mismatch:
- MemoryAgentBench assumes a "memorize chunk" interaction where an agent is instructed to remember content. MemGen currently may be optimized around reasoning/inference tasks rather than pure memory-ingestion acknowledgements.
- MemGen's current multi-turn manager may need an inference-only path for sequential chunk ingestion and later queries. If it cannot ingest chunks without generating normal task answers, an adapter will need careful prompt/response control.
- Chat-style history support may be required by the benchmark prompts. If MemGen's inference path cannot maintain or replay chat-style system/user/assistant turns cleanly, adapter complexity rises.

Leakage prevention:
- Do not store memory bank globally.
- Do not reuse `agent_save_folder` state for MemGen memory.
- Reset memory at the end of every context/session and on exception.
- Do not process multiple contexts in a batch.
- Do not write retrieved latent memory into benchmark result files except aggregate debug counters if needed.

## Compute, API, and Environment Feasibility

Expected compute:
- Short 6K/32K tasks are feasible on one RTX A6000 with a local Qwen/MemGen checkpoint if the adapter is efficient.
- 64K/128K tasks may be feasible but expensive because every 4096-token chunk triggers a MemGen inference/memory-write event.
- 262K to 1.44M contexts are likely expensive on one A6000 and should not be attempted until short tasks pass.
- Full benchmark runs are likely slow because some configs contain up to 100 examples and contexts in the 100K to 1M+ range.

Dataset size:
- Paper reports 2071 total questions and context depth from roughly 103K to 1.44M tokens.
- Code configs include `max_test_samples` values from 1 to 500 depending on task; many tasks use only 1 long context with many questions.
- The dataset is expected from HuggingFace as `ai-hyz/MemoryAgentBench`; I did not verify local cache or download it.

API dependence:
- Official long-context and many RAG baselines assume OpenAI, Azure OpenAI, Anthropic, Gemini, Zep, Letta, Mem0, Cognee, or other external services/packages.
- LLM judge scripts assume OpenAI GPT-4o.
- Automatic-metric tasks can be run offline if the dataset, model checkpoint, tokenizer/model dependencies, and any local entity files are present.

Can run with local Qwen/MemGen:
- Likely yes for a custom external adapter using local MemGen checkpoints.
- The official `AgentWrapper` does not provide a local HuggingFace generation agent for MemGen; adding one would be adapter work.

Environment:
- A separate conda environment is recommended for MemoryAgentBench official smoke tests because `requirements.txt` includes many packages that can conflict with MemGen, including `deepspeed`, CUDA package wheels, `faiss-gpu`, `mem0ai`, `cognee`, `letta`, `minference`, and `flash_attn`.
- README explicitly recommends Python 3.10.16 and notes dependency friction around HippoRAG/OpenAI versions, Cognee, and Letta.
- For MemGen adapter work, prefer using the existing MemGen conda environment for MemGen inference and import only minimal MemoryAgentBench utilities, or run adapter in a separate environment that calls MemGen via a controlled subprocess/API. Do not install MemoryAgentBench requirements into the current MemGen environment until a dependency plan exists.

Offline feasibility:
- Fully offline possible only for automatic-metric subsets after dataset cache is available and local model checkpoints are available.
- Not fully offline for GPT-4o judge subsets, official API baselines, OpenAI embeddings, Zep cloud, Letta API, or hosted model baselines.

## Scientific Fit for MemGen

Does it test the kind of memory improvement claimed?
- Yes, for session-local memory quality, especially AR and Conflict Resolution.
- Partially, for TTL if the claim includes learning mappings or examples at inference time.
- Weakly, for broad LRU summarization unless MemGen is extended to handle global consolidation beyond top-k latent retrieval.

Can it demonstrate benefits of latent session memory?
- Yes, if enabled Version A improves over disabled MemGen on identical local model/checkpoint, with one context per session and strict reset.
- The strongest evidence would come from tasks where the answer depends on information seen in earlier chunks and not available in the current query.

Can it distinguish MemGen from simple RAG?
- Potentially, but only if comparison includes BM25/embedding RAG baselines run under comparable local conditions.
- The benchmark's official baselines include strong API models and external memory systems, which may not be directly comparable to local MemGen unless normalized by base model, compute, and context limits.
- Conflict Resolution and TTL are more likely than simple AR to show differences beyond retrieval quality.

Can it support ablations?
- Yes, at the MemGen adapter/config level:
  - disabled memory
  - enabled Version A
  - no recency/last-retrieved decay
  - no replacement policy
  - top-k only
  - threshold variations
  - slot capacity variations
- The benchmark itself has `--chunk_size_ablation` and `--max_test_queries_ablation`, but MemGen-specific ablations should be controlled externally.

Informative failure cases:
- Correct retrieval but wrong answer format: parser/format alignment issue.
- Strong AR but weak Conflict Resolution: latent memory stores facts but update/replacement is insufficient.
- Weak TTL ICL: latent memory does not encode label mappings or repeated examples robustly.
- Short-context gains but long-context collapse: memory capacity/replacement limits.
- Enabled worse than disabled: retrieved latents interfere with reasoner hidden state.
- Cross-context improvement: likely leakage bug, not scientific signal.

Safe claims if MemGen improves:
- MemGen's session-local latent memory improves inference-time use of prior chunked context on selected MemoryAgentBench subsets.
- Reasoner-only latent memory can provide utility on memory-agent-style incremental benchmarks without storing text memory.
- Specific decay/replacement policies improve selected conflict/update tasks if ablations support it.

Claims to avoid:
- Do not claim full MemoryAgentBench state-of-the-art unless the full official benchmark and official baselines are reproduced.
- Do not claim general long-term memory across users/sessions; MemGen memory is session-local.
- Do not claim superiority to commercial memory systems unless run under comparable models, APIs, and costs.
- Do not claim human-like memory or persistent personalization.
- Do not claim improvements on LRU or LLM-judge tasks before those subsets are actually run and validated.

## Risks and Caveats

- API dependence: official baselines and judge scripts can require OpenAI, Anthropic, Gemini, Zep, Letta, and other services.
- Adapter invasiveness: a direct `AgentWrapper` patch touches benchmark code; a MemGen-core patch would be worse. Prefer external adapter first.
- Metrics strictness: ICL exact match and QA substring matching may penalize formatting differences. Output templates must be controlled.
- Task alignment: LRU summarization and Recsys may be less aligned with current top-k latent memory.
- Chunking assumptions: benchmark uses 4096-token sentence chunks with no overlap. This may favor text/vector memories and may not be optimal for MemGen latent writes.
- Long-context model assumptions: official long-context baselines use large context windows; local Qwen/MemGen may have different max context and generation behavior.
- Leakage risk: benchmark can load existing agent folders; MemGen adapter must not reuse session memory across contexts.
- State carryover risk: built-in memory systems persist to `./agents` or service-side stores. MemGen must explicitly reset.
- Reproducibility: README says more details are still coming, and dependency notes are rough.
- Code quality: many vendored libraries, broad dependencies, API branches, and environment-specific bash paths increase maintenance risk.
- Dataset availability: `ai-hyz/MemoryAgentBench` may auto-download; local cache status was not checked in this assessment.
- Paper/code naming mismatch: paper says Selective Forgetting; README/code use Conflict Resolution for that task group.
- Official comparison difficulty: paper baselines use closed/API models and multiple memory systems; local MemGen comparisons need carefully scoped claims.

## Final Recommendation

Use with caution.

Use MemoryAgentBench as a secondary benchmark for MemGen memory quality after GSM8K/TriviaQA infrastructure is stable, not as the immediate next formal run. It is suitable for testing whether latent session memory helps with incremental chunked context, especially Accurate Retrieval and Conflict Resolution. It is not suitable as a first full benchmark because official runs are API-heavy, dependency-heavy, and include very long contexts.

First subset:
- `Factconsolidation_sh_6k`
- `Factconsolidation_mh_6k`
- `EventQA_64k`
- one ICL task such as `ICL_trec_coarse`

Avoid initially:
- LongMemEval judge tasks.
- InfBench summarization.
- Recsys.
- Full 262K/421K/1.44M contexts.
- Agentic API baselines.

Repository strategy:
- Do not fork now.
- Keep the local clone in `/tmp` or reclone for future audits.
- Start with an external adapter script/package, outside MemGen core and outside MemoryAgentBench core.
- Fork MemoryAgentBench only if official smoke tests pass and a small MemGen adapter needs stable patching against their harness.

Integration location:
- Preferred: external adapter repo/script that imports MemoryAgentBench data utilities and MemGen inference.
- Acceptable later: lightweight MemoryAgentBench fork with a `MemGenAgentWrapper`.
- Avoid: embedding MemoryAgentBench-specific control flow in MemGen model/training/eval code.

## Proposed Next Steps

Phase MAB-0: audit
- Status: this document.
- Do not run benchmark experiments.
- Verify local dataset cache status later with read-only HuggingFace cache inspection.

Phase MAB-1: official smoke test
- Use a fresh dedicated `MABench` environment, not the current MemGen environment.
- Run only one official automatic-metric smoke with `--max_test_queries_ablation 1`.
- Prefer a simple long-context or BM25 baseline on a small config.
- Goal: confirm dataset loading, output JSON schema, and metric computation.

Phase MAB-2: MemGen disabled adapter
- External adapter only.
- One context, one query, disabled memory.
- Confirm output dict compatibility and exact disabled behavior relative to original MemGen inference constraints.
- Confirm memory is not initialized or used.

Phase MAB-3: MemGen enabled Version A adapter
- Enable session-local Reasoner-only latent memory.
- Force `batch_size=1`.
- Reset memory after each context/session and on exception.
- Confirm retrieved memory does not enter Weaver and is not exposed as text.

Phase MAB-4: small-scale comparison
- Run disabled vs enabled Version A on:
  - `factconsolidation_sh_6k`
  - `factconsolidation_mh_6k`
  - `eventqa_65536`
- Add one ablation only after disabled/enabled smoke is stable.

Phase MAB-5: full benchmark subset
- Expand to 32K/64K conflict resolution, selected EventQA/RULER, and one ICL task.
- Only then consider LLM-as-judge tasks or larger contexts.
- Keep claims subset-scoped unless the official full benchmark is reproduced.
