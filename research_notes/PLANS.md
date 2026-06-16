# Research Plan（研究计划）

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
- 记录当前 decay 是 write-age decay。
- 记录 threshold_topk 没有 fallback top-1。
- 实现 structured retrieval context。
- 实现并验证 thread_update。

尚未实现：

- last_retrieved_decay
- threshold_topk_with_fallback_top1

Version B 仍然不在范围内。

---

## Phase 8C：Target Task Transition（迁移到目标任务）

执行顺序：

1. 清理 notes。
2. review 并提交 Version A-aligned 工作。
3. 规划 TriviaQA baseline。
4. 运行 Original MemGen / disabled-memory smoke。

将主要评测从 GSM8K 转移到 TriviaQA。

原因：

TriviaQA 是动态多轮搜索环境，具有：

- max_turns=5
- 持续增长的 interaction history
- observation truncation

更适合作为长期 memory 的验证任务。

---

## Phase 8D：TriviaQA Version A-Aligned Smoke

在开始之前必须：

- 准备 TriviaQA checkpoint。
- 准备 TriviaQA dataset cache。
- 准备 AgentBank cache。
- 验证 127.0.0.1:8001/retrieve。
- 验证 Search-R1 / Wikipedia index。
- 防止 silent fallback。
- 定义 sample_count=1 的 dynamic harness 和结构化 answer.json。

之后：

- 运行 Version A-aligned thread_update。
- 验证 session 内 memory 是否跨 turn 保留。
- 验证 episode 间是否 reset。
- 检查早期写入是否在后续 turn 被检索。
- 记录 latency、memory 和 retrieval/write event。

在 disabled 和 enabled 都稳定之前，不做性能结论。

---

## Phase 8E：Method-Aligned Version A Variants

仅在 TriviaQA baseline 稳定后考虑：

- last-retrieved-turn decay
- fallback top-1

Version A-simple 保留作为对照组。

每次修改都必须重新验证 disabled 等价性。

---

## Phase 8F：TriviaQA Targeted Ablations（TriviaQA 消融实验）

比较：

- disabled Original MemGen
- Version A-simple
- Version A + last-retrieved decay
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

只有在 TriviaQA 上证明 Version A 有足够价值之后才开始。

实现完整流程：

retrieve     ↓ Weaver revise / generate     ↓ matched write-back

特点：

- retrieval 输入 Weaver。
- 包含 fallback top-1。
- 包含 last-retrieved-turn decay。
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