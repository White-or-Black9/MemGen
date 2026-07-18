# Research Plan（研究计划）

## Current Planning State（当前计划状态）

> Updated 2026-07-12. Historical R4 and pre-MAB planning below is retained for
> provenance; its claims that Version B is unimplemented or the next active
> step are superseded by this section.

- `paper/outline.md` is authoritative for the current paper organization.
- Current working title: **Inference-Time Latent Memory Management for
  Long-Horizon LLM Agents**.
- The paper studies a session-local bank that stores, retrieves, updates,
  replaces, and reuses latent memories during inference on long-context
  reasoning tasks.
- Current EventQA checkpoint: P7 non-strict is the paper-level main candidate
  among tested configurations, supported by five repeats. This is not yet a
  benchmark-general proof. EventQA is the current positive evidence source,
  not the definition of the full paper goal. Evidence:
  `outputs/mab/eventqa_five_repeat_stability_summary.md` and `.json`.
- Context 4 remains the main limitation. A frozen-bank q0-99 oracle
  counterfactual now supports an ordered tuple-level harmful interaction for
  P7 source-run tuple `[1,0]`. Evidence:
  `outputs/mab/eventqa_harmful_memory_attribution_context4_full/20260704T001824Z-p7-context4-q0-99/`.
- Further attribution expansion and utility-gate implementation are paused.
- The EventQA comparison, explicit-memory controls, no-query-retrieval
  ablation, method-separable cost package, tables, method figures, LoCoMo
  limitation appendix, and bibliography are complete for the current draft.
- Current state is `EventQA_supplementary_planning`: do not open new benchmark
  routes. First perform a read-only EventQA evidence-gap inventory and
  pre-register one focused follow-up; no EventQA GPU run is authorized until a
  separate execution decision.
- LongBench v2 follow-up is closed under the current frozen-P7 method: its
  protocol-clean 60-item paired comparison has no query-retrieval effect
  (`5/5/50` P7 wins/losses/ties versus no-query). Do not expand it, tune P7 on
  it, or use it as positive paper evidence.
- By DEC-0089, BABILong, MemBench, InfiniteBench, LongBench v1, and other new
  benchmark candidates are paused while the project focuses on EventQA
  supplementary evidence.
- Reopen sequence:
  1. independent skeptical review;
  2. resolve review-blocking manuscript or evidence issues;
  3. convert to the selected venue template only after the review gate.

### Additive memory benchmark campaign closure (2026-07-09)

- Scope remains additive to the frozen EventQA paper package and does not
  change the current paper fallback rule.
- FactConsolidation route is closed under the present frozen-P7 protocol:
  weak and inconsistent 6k/32k/64k results do not justify main-table
  promotion or further scaling.
- DetectiveQA route is also closed for the current paper cycle:
  - single-query full, extractor-aware rerun, and multi-query full are
    complete;
  - the pre-alignment runs had clean bank-lifecycle invariants and appeared to
    show `p7 > disabled`, but that effectiveness comparison is historical only;
  - the historical `disabled` vs `p7_no_query_retrieval` comparison was later
    found to be confounded by a query-generation mismatch
    (`disabled=10` vs bank-on `40` response-length contract);
  - the aligned multi-query full rerun removes that confound and yields
    `disabled=10/71`, `p7=9/71`, `p7_no_query_retrieval=10/71`;
  - use DetectiveQA only as appendix-only negative diagnostic /
    failure-analysis evidence, not as positive supplementary evidence.
- Consequence:
  the additive benchmark campaign is complete for this paper cycle.
  It does not reopen EventQA runs, does not broaden the paper claim, and does
  not replace the EventQA-centered fallback route.
- Historical note on the brief benchmark-policy reopen:
  - a bounded `RULER-QA2` exploratory trial was executed on 2026-07-10 under
    an adapted frozen-bank protocol;
  - the route is now re-closed because the full adapted `p7` run is
    mechanism-negative (`retrieved_latent_count=0` on all `100` queries) even
    though the runner itself is executable;
  - future benchmark expansion, if reopened again, should begin with a fresh
    benchmark-choice audit rather than assuming `RULER-QA2` remains the next
    candidate.
> Historical route, superseded by DEC-0088 and DEC-0089: the LongBench v2
> recovery below completed as a v3 paired comparison and no longer has an
> active scheduler, merge gate, or expansion path.

- Current route after benchmark closure:
  1. benchmark planning was explicitly reopened on 2026-07-12;
  2. use `research_notes/plans/p7_longbench_v2_benchmark_plan.md` as the active
     benchmark-planning authority;
  3. LongBench v2 Phase 0 dataset audit completed with frozen 18-row smoke and
     60-row bounded manifests;
  4. Phase 1 adapter/scorer and no-model lifecycle contracts completed with
     `15/15` tests passing;
  5. the frozen 18-item smoke completed with mechanically valid retrieval but
     weak effectiveness;
  6. the user explicitly approved the frozen 60-item bounded evaluation; its
     original single-worker run stopped after two items on a full-context
     Disabled OOM;
  7. recovery subsequently completed and merged under the v3 constrained-choice
     protocol; P7 and no-query tie `17/60`, so the candidate is closed;
  8. do not add explicit-memory controls or further benchmark expansion;
  9. focus only on read-only EventQA supplementary-evidence inventory before
     any new paper-facing or GPU work.

### Historical planning snapshot below

- Accepted formal results: Phase 0-7.
- Historical / exploratory records: Phase 8A GSM8K pilot, Phase 8C-alt
  controlled mechanism study, Phase 8D-0 / R4-1A infrastructure discovery, R4
  Search-R1 / TriviaQA full pipeline (infrastructure validation + threshold
  calibration + held-out exploratory comparison + case studies), and Phase R2 /
  R2-fix mechanism revisions.
- Current mechanism definition:
  - Reasoner-only retrieved-memory injection.
  - Retrieved memory does not enter Weaver.
  - Stored memory is reasoner-space `latent_inputs_embeds`.
  - Memory is session-local.
  - Enabled memory requires `batch_size=1`.
  - Retrieval uses last-retrieved decay.
  - No fallback top-1.
- Current R4 state:
  - Infrastructure validation: complete with caveats.
  - Threshold calibration: complete.
    - default `threshold=0.7` is inappropriate for TriviaQA scale
      (mean decayed-score 0.036, max 0.054)
    - `threshold=0.04` is a first calibrated candidate (score-based, not
      reward)
  - Held-out exploratory comparison (samples 20..79, threshold fixed at 0.04):
    - disabled 35/60, Version A t=0.04 35/60 — net gain 0
    - 1 rescue (sample 53), 1 regression (sample 21)
    - effect fragile and sample-dependent
  - Key mechanism caveat: pre-evidence memory write timing
    - first insert occurs before Search-R1 evidence in context
    - retrieved latent may amplify query entity salience rather than
      evidence-grounded answer
  - Threshold comment caveat: config comment says "cosine similarity
    threshold" but implementation uses decayed retrieval score
- Remaining caveats:
  - all R4 reward means are exploratory only, not benchmark evidence
  - duplicate system prompt appears in conversation artifacts
  - `answer.json` uses JSONL format
  - no direct artifact-level Reasoner-only injection assertion
  - threshold=0.04 overrides were in-memory diagnostics only
  - default threshold 0.7 incompatible with TriviaQA decayed-score scale
  - memory timing issue (pre-evidence write) may confound threshold-only fixes
- Version B status at the time of this historical snapshot: not started. This
  statement is superseded by the completed MAB-6A/MAB-6B and EventQA work
  summarized above and in
  `research_notes/benchmarks/memoryagentbench_mab6b_fr_eventqa_65536_n5.md`.

## Current Next Step（当前下一步）

> Superseded on 2026-07-04 by the current planning state above. The numbered
> R4 actions below are retained as historical provenance, not as the active
> execution plan.

1. R4 infrastructure validation and threshold calibration are complete with
   caveats.
2. Do NOT immediately run larger benchmarks, tune thresholds, or implement
   pipeline timing constraints.
3. Primary next step: read-only case study comparing rescue sample 53
   (Seymour Hersh / My Lai massacre) against harmful sample 21 (Gangsta's
   Paradise / Dangerous Minds).
   - Goal: understand when memory helps vs hurts.
   - Determine whether memory is acting as: useful latent prior,
     evidence-grounded clue, query-salience amplifier, or noisy perturbation.
4. After mechanism analysis, possible variants include:
   evidence-grounded memory only, suppress pre-evidence memory write,
   retrieve only evidence-grounded slots, answer-stage verification/gating,
   or threshold ablation informed by timing/content understanding.
5. Version B remains deferred until a separate explicit decision.

---

## Objective（研究目标）

将 MemGen 作为一个长期研究项目持续推进，并研究一种仅在推理阶段（inference-only）启用的、可选的（optional）、会话级（session-level）的 Retrieval-Augmented Recurrent Latent Memory Bank（检索增强循环潜在记忆库）。

---

# Non-Negotiable Constraints（不可违反的约束）

1. 不得修改 Weaver 的训练流程。
2. 不得修改 Trigger 的训练流程。
3. 当 latent_memory_bank.enabled=false 时，系统行为必须与原始 MemGen 完全一致。
4. 在后续阶段未获得明确批准之前，memory 必须保持 session-local（仅限当前会话），不得跨 sample 共享。
5. 在后续阶段未获得明确批准之前，memory-bank 实验默认使用 batch_size=1。
6. 每完成一个 Phase，都必须更新 PROGRESS.md。
7. 每完成一个实验，都必须更新 EXPERIMENTS.md。
8. 每做出一个重要设计决策，都必须更新 DECISIONS.md。
9. 一次只能执行一个 Phase，完成后必须停止并等待用户确认。

---

# Phase Template（阶段模板）

每个阶段应包含：

- 状态（proposed | approved | in_progress | completed | blocked）
- 目标（Goal）
- 范围（Scope）
- 明确不包含的内容（Out of scope）
- 前置条件（Preconditions）
- 预计修改的文件
- 实现步骤
- 验证方法
- 必须更新的文档
- 退出条件（Exit criteria）
- 回滚方案（Rollback plan）
- 用户批准（User approval）

---

# Initial Roadmap（初始路线图）

## Phase 0：Research Memory System and Repository Snapshot（建立研究记忆系统与仓库快照）

- 建立长期保存的 research_notes/ 项目记忆系统。
- 记录研究目标、约束条件、工作流程和仓库初始状态。
- 记录当前分支、commit、工作区状态、环境和可用资源。
- 不修改核心代码，不运行正式实验。

---

## Phase 1：Code Map and Inference Pipeline Audit（代码结构与推理流程审计）

- 梳理 inference 入口、配置流、session/sample 边界、latent 表示、generation 输出和 evaluation hooks。
- 明确 Weaver 与 Trigger 的训练边界不能修改。
- 找到 inference-only 的最佳集成位置及风险。
- 更新 CODE_MAP.md。

---

## Phase 2：Original Project Smoke Test（原项目 Smoke Test）

- 验证环境、依赖、模型加载、数据加载以及最小推理流程。
- 使用最小 sample 数量和 batch_size=1。
- 记录 warning、失败信息和输出文件。
- Smoke Test 不得作为正式 baseline。

---

## Phase 3：Original MemGen Baseline（建立原始 MemGen 基线）

- 建立可信、可复现的 MemGen 对照组。
- 如有必要，在批准范围内修复 baseline blocker。
- 固定 golden case。
- 固定随机种子、sample ID、checkpoint、解码参数和 evaluation script。
- 记录 task metric、latency 和 memory usage。
- 冻结 disabled-feature compatibility oracle。

---

## Phase 4：LatentMemoryBank Module Skeleton（LatentMemoryBank 模块骨架）

- 增加独立的 Memory Bank 数据结构和配置。
- 默认 latent_memory_bank.enabled=false。
- 实现生命周期、reset、capacity、validation 和 isolation。
- 本阶段不接入正式 inference。
- 增加针对 Memory Bank 本身的单元测试。

---

## Phase 5：Version A —— 仅注入 Reasoner

- Memory Bank 仅集成到 inference。
- 检索 latent memory 并注入 Reasoner。
- 禁止将检索结果输入 Weaver。
- memory 保持 session-local。
- 默认 batch_size=1。
- 所有写入 memory 的 latent 必须 detach，并显式处理 device/dtype。
- 不修改 Weaver 和 Trigger 的训练流程。

---

## Phase 6：Disabled Feature Equivalence Test（关闭功能等价性测试）

比较：

- latent_memory_bank.enabled=false

与

- Phase 3 冻结的 baseline。

要求完全一致：

- token ids
- augmentation masks
- metrics
- 输出格式
- tensor/control-flow

任何差异都视为阻塞性 regression。

---

## Phase 7：Version A Stability and Debug Experiment（Version A 稳定性与调试）

验证：

- session reset
- 无跨 sample 泄漏
- 空 memory
- capacity 上限
- dtype/device 一致性
- deterministic replay
- 长 session 行为

记录：

- latency
- memory overhead

仅修复 Version A 缺陷。

---

## Phase 8A：GSM8K Version A-simple Pilot

状态：已完成。

结果：

- G0（disabled）：0.60（12/20）
- G1/G4/G6/G7（enabled）：0.50（10/20）

解释：

- 不能作为论文主要证据。
- GSM8K 为短单轮任务，不适合验证长期 memory 假设。
- 当前比较的是 write-age decay，而不是 last-retrieved-turn decay。

---

## Phase 8B：Method / Implementation Alignment（方法与实现对齐）

状态：已完成。

已完成：

- 区分 Version A-simple、Version A-aligned、Version B。
- 先记录并确认历史当前 decay 是 write-age decay。
- 记录 threshold_topk 没有 fallback top-1。
- 实现 structured retrieval context。
- 实现并验证 thread_update。
- Phase R2 已将当前 Version A-aligned 修订为 last-retrieved decay。
- Phase R2 已将 full-bank `new_thread` eviction 修订为
  largest `last_retrieved_age`。

仍未实现：

- threshold_topk_with_fallback_top1

Version B 仍然不在范围内。

---

## Phase 8C：Target Task Transition（迁移到目标任务）

状态：已部分完成；notes review / commit preparation 已完成，target-task
infrastructure preparation 仍未完成。

执行顺序：

1. 清理并统一 notes。
2. 规划 TriviaQA baseline / infrastructure。
3. 准备 disabled single-sample structured harness。
4. 运行 Original MemGen / disabled-memory smoke。

将主要评测从 GSM8K 转移到 TriviaQA。

原因：

TriviaQA 是动态多轮搜索环境，具有：

- max_turns=5
- 持续增长的 interaction history
- observation truncation

更适合作为长期 memory 的验证任务。

---

## Phase 8D / R4：Further Validation and Test Environment Preparation

状态：当前下一阶段。

在开始之前必须：

- 准备 TriviaQA checkpoint。
- 准备 TriviaQA dataset cache。
- 准备 AgentBank cache。
- 验证 127.0.0.1:8001/retrieve。
- 验证 Search-R1 / Wikipedia index。
- 防止 silent fallback。
- 定义 sample_count=1 的 dynamic harness 和结构化 answer.json。

执行顺序：

- 先验证 disabled baseline path。
- 只有在 disabled path 稳定后，才运行 Version A-aligned enabled smoke。

之后的 enabled smoke 目标：

- 运行 Version A-aligned thread_update。
- 验证 session 内 memory 是否跨 turn 保留。
- 验证 episode 间是否 reset。
- 检查早期写入是否在后续 turn 被检索。
- 记录 latency、memory 和 retrieval/write event。

在 disabled 和 enabled 都稳定之前，不做性能结论。

---

## Phase 8E：Method-Aligned Version A Variants

状态：部分 superseded。

已完成：

- Version A-aligned last-retrieved-turn decay（Phase R2）

仅在 TriviaQA baseline 稳定后考虑：

- fallback top-1

Version A-simple 保留作为对照组。

每次修改都必须重新验证 disabled 等价性。

### Immediate Track A：TriviaQA Infra and Evaluation

- prepare / verify TriviaQA dataset
- prepare / verify checkpoint
- prepare / verify retrieval service / index
- build or adapt a dynamic single-sample structured harness
- run disabled baseline first
- run Version A-aligned enabled smoke second
- then decide whether to scale

定位：

- TriviaQA remains the immediate repository-aligned target-task path
- the goal is to observe the current Version A-aligned mechanism on a real
  target task
- no target-task claim is allowed until these runs are actually completed

### Immediate Track B：Controlled Diagnostic Subset Design

- small scale
- mechanism diagnostic only
- focus on revisit behavior and capacity pressure
- explicitly test last-retrieved decay
- explicitly test `last_retrieved_step` refresh semantics
- explicitly test `last_retrieved_age`-based scoring
- explicitly test selected / returned slots only updating
  `last_retrieved_step`
- explicitly test full-bank eviction by largest `last_retrieved_age`
- explicitly preserve no fallback top-1
- not a formal benchmark
- not a performance claim

定位：

- this subset exists to expose Version A-aligned mechanism behavior that
  TriviaQA may not clearly show
- it does not replace TriviaQA as the main short-term evaluation path

### Future Note

- MemoryAgentBench and LongMemEval are future memory-oriented benchmark
  candidates only
- no current integration
- no current implementation
- no current detailed investigation or adoption design
- user will investigate them later

### R4-1B：Dataset and Checkpoint Acquisition / Cache

状态：planned。

目标：

- 在任何 TriviaQA evaluation 之前，准备好或获取所需的资源。

输入条件：

- base model `Qwen/Qwen2.5-1.5B-Instruct` 已存在于本地 HF cache 中
- 所需的 TriviaQA checkpoint（检查点）缺失
- 所需的 TriviaQA dataset cache（数据集缓存）缺失
- 所需的 AgentBank triviaqa dataset cache 缺失

范围：

- dataset cache 验证 / 获取规划
- checkpoint cache 验证 / 部署规划
- 资产就绪后的离线加载验证
- 不运行 evaluation

任务：

1. 确认 TriviaQA checkpoint 是公开可下载的，还是需要手动部署 / 权限。
2. 决定数据集和 checkpoint 的缓存目录策略。
3. 缓存或部署 `mandarjoshi/trivia_qa`，config
   `rc.wikipedia.nocontext`，split `validation`。
4. 缓存或部署 `Solaris99/AgentBank`，config `triviaqa`，split `train`。
5. 缓存或部署 TriviaQA MemGen checkpoint：
   `MemGen/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8`。
6. 验证离线数据集加载。
7. 验证 checkpoint 路径可读性。
8. 本阶段不运行 evaluation。

成功标准：

- base model 已验证
- TriviaQA dataset 可离线加载
- AgentBank triviaqa 可离线加载
- checkpoint 路径存在且可读
- 未运行任何 evaluation

退出条件：

- 资产就绪，或已记录明确的 blocker

### R4-1C：Retrieval Service / Index Configuration

状态：completed as configuration check; formal retrieval setup remains blocked.

目标：

- 使 TriviaQA dynamic retrieval endpoint（动态检索端点）可靠可用。

范围：

- Search-R1-compatible retrieval service（检索服务）和 index（索引）的搭建 / 验证
- endpoint schema 兼容性检查
- 不运行完整 evaluation

任务：

1. 检查 Search-R1 仓库和 `search_r1/search/retrieval_server.py` 是否存在。
2. 检查 retriever 运行环境，或创建搭建计划。
3. 定位或部署 `wiki-18.jsonl`。
4. 定位或部署 `e5_Flat.index`。
5. 定位或部署 `intfloat/e5-base-v2`。
6. 启动或规划 Search-R1 `retrieval_server.py` 的启动流程。
7. 用 curl 验证 `http://127.0.0.1:8001/retrieve` 可达。
8. 确认响应 schema 与 `data/utils/retrieval_utils.py` 兼容。
9. 添加或规划 retrieval failure 的 fail-fast（快速失败）检查。
10. 本阶段不运行完整 evaluation。

成功标准：

- endpoint 可达
- curl 返回有效的检索结果
- 目标运行不会出现静默 retrieval degradation（静默检索降级）
- index/corpus 路径已文档化

退出条件：

- Search-R1-compatible retrieval service 就绪，或已记录 blocker

R4-1C check result:

- MemGen expects `http://127.0.0.1:8001/retrieve`
- request payload is
  `{"queries": [...], "topk": 3, "return_scores": true}`
- expected response is
  `{"result": [[{"document": {"contents": "Title\nBody"}, "score": ...}]]}`
- `data/triviaqa/env.py` still silently converts retrieval exceptions into
  `Cannot find corresponding pages.`
- endpoint and top-k are hard-coded in `data/utils/retrieval_utils.py`
- Search-R1 repo / server, `retrieval_server.py`, and `retrieval_launch.sh`
  were not found locally
- no `searchr1` / retriever conda env was found
- `faiss`, `pyserini`, `e5_Flat.index`, `wiki-18.jsonl`,
  `intfloat/e5-base-v2`, and the `8001` endpoint are missing / unavailable
- upstream Search-R1 schema appears compatible, except the default upstream
  port is `8000` while MemGen expects `8001`
- formal TriviaQA retrieval remains blocked until Search-R1-compatible service,
  corpus, index, and retriever model are available
- toy retrieval server remains smoke-only and cannot support formal TriviaQA
  results

Decision after R4-1C:

- continue formal Search-R1 setup later
- proceed to R4-1D harness design because structured output and retrieval
  failure accounting are required regardless of retrieval-service readiness

### R4-1D：Dynamic Single-Sample Structured Harness

状态：implemented.

目标：

- 在运行 baseline 之前，构建一个可信的最小化 dynamic evaluation harness
  （动态评估脚手架）。

范围：

- 仅限 dynamic TriviaQA sample selection（样本选择）和 structured output
  （结构化输出）
- 不改变 memory-bank 方法
- 不进入 Version B
- 不启用 fallback top-1（不启用 top-1 兜底）
- 不让 retrieved memory 进入 Weaver

任务：

1. 为 dynamic TriviaQA 添加 `sample_count=1` 或固定样本选择。
2. 为 dynamic evaluation 生成结构化 `answer.json`。
3. 保留 `conversations.txt`。
4. 记录 sample id。
5. 记录 question。
6. 记录 gold answer（标准答案）。
7. 记录 parsed answer（解析后的答案）。
8. 如适用，记录 exact match / relaxed match（精确匹配 / 宽松匹配）。
9. 记录 retrieval calls（检索调用次数）。
10. 记录 retrieval failures（检索失败）。
11. 记录是否出现了 `Cannot find corresponding pages.`。
12. 记录 `valid_run: bool`。
13. 记录 `invalid_reason: str | null`。
14. 记录 memory enabled flag。
15. 记录 batch size。
16. 记录 checkpoint path。
17. 记录 config overrides（配置覆盖项）。
18. 对于启用 memory 的运行，如有 memory trace 则保留之。
19. 保持 memory-bank 方法不变。
20. 如合适，为 harness 行为添加测试或 smoke checks（冒烟检查）。

成功标准：

- 单样本 dynamic 运行可确定性执行
- 结构化输出存在
- retrieval failure 可见
- disabled 和 enabled 配置可以复用同一个 harness

退出条件：

- harness 已准备好供 disabled baseline smoke 使用

R4-1D implementation result:

- added `scripts/eval/r4_triviaqa_dynamic_harness.py`
- added `tests/test_r4_triviaqa_dynamic_harness.py`
- CLI supports `--cfg-path`, `--checkpoint-path`, `--output-dir`,
  `--sample-index`, `--sample-count`, `--batch-size`, `--memory-mode`,
  `--require-retrieval-ok`, `--retrieval-endpoint`, `--retrieval-topk`,
  `--max-response-length`, `--temperature`, `--seed`, `--dry-run`, and
  `--preflight-only`
- `batch_size != 1` and `sample_count != 1` fail fast
- `memory_mode=disabled` keeps memory disabled
- `memory_mode=version_a_aligned` enables the current Version A-aligned
  `thread_update` memory-bank configuration without changing memory-bank logic
- structured `answer.json` records sample identity, question, gold answers,
  conversation, final response, strict parsed answer, retrieval accounting,
  run metadata, `memory_bank_debug`, `valid_run`, and `invalid_reason`
- `summary.json` records sample count, valid / invalid run counts, and
  retrieval-blocked count
- no formal experiment, disabled baseline, or enabled smoke was run during
  implementation

### R4-1E：Disabled Baseline Smoke

状态：planned。

目标：

- 在测试 memory-bank 启用行为之前，建立一个可信的 disabled baseline
  （禁用 memory 的基线）。

前置要求：

- R4-1B 资产就绪
- R4-1C retrieval service 就绪
- R4-1D harness 就绪

任务：

1. 对 1 个固定的 TriviaQA 样本，以 memory disabled 状态运行。
2. 使用 `batch_size=1`，以便与后续 enabled memory 路径直接对比。
3. 记录结构化 `answer.json`。
4. 记录 `conversations.txt`。
5. 记录 retrieval 成功 / 失败。
6. 确认没有 memory bank 被构造或使用。
7. 确认没有静默 retrieval failure。
8. 不声称任何性能结果。

成功标准：

- 运行完成
- retrieval 正常工作
- 结构化输出存在
- disabled 路径干净
- 没有意外的 memory-bank 工件出现

退出条件：

- disabled baseline smoke 通过验收

### R4-1F：Version A-Aligned Enabled Smoke

状态：planned。

目标：

- 对同一个固定的 TriviaQA 样本，以当前 Version A-aligned
  （当前对齐版 Version A）memory bank 启用状态运行。

前置要求：

- R4-1E disabled baseline 通过验收
- `batch_size=1`
- 尽可能使用与 disabled baseline 相同的样本
- 使用相同的 checkpoint / retrieval service / harness

任务：

1. 启用当前 Version A-aligned memory bank。
2. 保持 retrieved memory 仅进入 Reasoner（Reasoner-only）。
3. 确保 retrieved memory 不进入 Weaver。
4. 确保不启用 fallback top-1（不启用 top-1 兜底）。
5. 确保 memory bank 为 session-local（会话本地）。
6. 运行 1 个固定样本。
7. 记录结构化 `answer.json`。
8. 记录 `conversations.txt`。
9. 记录 memory trace 中的 retrieve count。
10. 记录 memory trace 中的 write count。
11. 记录 memory trace 中的 `last_retrieved_step`。
12. 记录 memory trace 中的 `last_retrieved_age`。
13. 记录 selected slots（被选中的 slot）。
14. 记录 eviction reason（淘汰原因），如有。
15. 与 disabled smoke 做定性对比。
16. 不声称任何性能提升。

成功标准：

- enabled 运行完成
- memory path 激活且无边界违规
- `batch_size=1`
- 结构化输出存在
- memory trace 确认 Version A-aligned 行为

退出条件：

- enabled smoke 通过验收，或已记录 blocker

### R4 Boundary and Claim Control（共享边界与声明控制）

- R4-1B 至 R4-1F 阶段属于 environment / harness / smoke phases
  （环境搭建 / 评测脚手架 / 冒烟测试阶段），不是最终 main result phases
  （主结果阶段）。
- 目前尚无任何 TriviaQA result。
- 目前尚无任何 target-task performance gain claim（目标任务性能声明）。
- Controlled diagnostic subset（受控诊断子集）与 TriviaQA infrastructure
  计划相互独立。
- Version B remains deferred（继续推迟）。
- R4 不包含 fallback top-1（不启用 top-1 兜底）。
- Retrieved memory 不进入 Weaver。
- Toy retrieval server（玩具检索服务）的输出，即使被使用，也仅限 smoke-only
  （仅用于冒烟测试），不能支撑正式的 TriviaQA result 或 performance claim。

---

## Phase 8F：TriviaQA Targeted Ablations（TriviaQA 消融实验）

比较：

- disabled Original MemGen
- Version A-simple
- Version A-aligned current（已包含 last-retrieved decay）
- Version A + fallback top-1
- Version A + matched-slot update
- threshold sweep
- top-k sweep

重点分析：

- 多轮行为
- 长轨迹行为
- context truncation

不再使用 GSM8K 作为主要证据。

---

## Phase 9：Version B

状态：historical plan, superseded. Exploratory Version B / Weaver-space bank
work was later implemented and evaluated in MAB-6A, MAB-6B, and EventQA. See
`research_notes/benchmarks/memoryagentbench_mab6b_weaver_space_bank.md` and
`research_notes/benchmarks/memoryagentbench_mab6b_fr_eventqa_65536_n5.md`.

The original gate text below is retained for provenance and is not the current
project state.

实现完整流程：

retrieve     ↓ Weaver revise / generate     ↓ matched write-back

特点：

- retrieval 输入 Weaver。
- 包含 fallback top-1。
- 可能包含 turn-aware / retrieval-aware decay 的更完整设计。
- 包含 matched-slot/thread update。

同时保留 Version A-simple 和 Version A 作为对照组。

---

## Phase 10：Paper-Level Consolidation（论文级整合）

整合：

- 主实验结果
- 消融实验
- 效率分析
- memory 行为
- failure analysis
- limitations

保证每个 claim 都能追溯到实验 ID 和原始 artifact。

冻结：

- 方法定义
- 可复现说明
- 论文证据

不得把 GSM8K pilot 或未经验证的观察提升为最终论文结论。

---

# Experiment Logging Standard（实验记录规范）

所有实验（包括失败、debug、smoke、ablation）必须记录：

- 日期与实验 ID
- git branch
- commit hash
- command
- config file
- model path
- checkpoint path
- dataset path
- sample count
- random seed
- decoding parameters
- output directory
- prediction file
- metric file
- latency
- memory usage（若可获得）
- notes 和 failures

缺失值必须明确写成 not available 并说明原因。

---

# Output Directory Standard（输出目录规范）

outputs/ ├── baseline/ ├── latent_bank_vA/ ├── latent_bank_vB/ └── ablations/

说明：

- outputs/baseline/：原始 MemGen 和 baseline。
- outputs/latent_bank_vA/：Version A 结果。
- outputs/latent_bank_vB/：Version B 结果。
- outputs/ablations/：消融实验结果。

每个实验必须使用唯一 Experiment ID，不得覆盖历史结果。

---

# Phase Gate（阶段门控）

## 开始执行前

必须满足：

- [ ] 用户明确批准该 Phase。
- [ ] Scope 和 Exit Criteria 已明确。
- [ ] Baseline 和验证命令已确定。

## 执行结束后

必须满足：

- [ ] 验证通过（或失败已记录）。
- [ ] 更新 PROGRESS.md。
- [ ] 更新 EXPERIMENTS.md。
- [ ] 更新 DECISIONS.md。
- [ ] 停止执行，等待用户确认后才能进入下一 Phase。
