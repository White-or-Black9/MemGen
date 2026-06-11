# 研究计划

## 目标

将 MemGen 作为长期科研项目持续维护，并研究一种仅作用于推理阶段、可选开启的
session-level Retrieval-Augmented Recurrent Latent Memory Bank。

## 不可妥协约束

1. 不修改 Weaver 训练流程。
2. 不修改 Trigger 训练流程。
3. 当 `latent_memory_bank.enabled=false` 时，行为必须与原始实现完全一致。
4. 在后续某个阶段被明确批准之前，memory 必须保持 session-local，且不得跨样本共享。
5. 在后续某个阶段被明确批准之前，memory-bank 相关实验默认使用 `batch_size=1`。
6. 每个 Phase 完成后都必须更新 `PROGRESS.md`。
7. 每个实验都必须更新 `EXPERIMENTS.md`。
8. 每个重要设计决策都必须更新 `DECISIONS.md`。
9. 每次只执行一个 Phase，完成后暂停并等待用户确认。

## Phase 模板

### Phase N: <标题>

- 状态：`proposed | approved | in_progress | completed | blocked`
- 目标：
- 范围：
- 明确不包含：
- 前置条件：
- 预计会修改的文件：
- 实施步骤：
- 验证方式：
- 必需更新的研究笔记：
- 退出条件：
- 回滚方案：
- 用户批准：

## 初始路线图

### Phase 0: Research Memory System and Repository Snapshot

- 建立可持续使用的 `research_notes/` 项目记忆系统。
- 记录研究目标、约束、工作流以及仓库初始状态。
- 记录分支、提交、工作树状态、环境与现有资产。
- 不修改核心代码，不进行实质性实验。

### Phase 1: Code Map and Inference Pipeline Audit

- 梳理推理入口、配置流、session/sample 边界、latent 表示、生成输出与评估钩子。
- 标记必须保持不变的 Weaver 与 Trigger 训练边界。
- 识别仅作用于推理阶段的候选集成点及其风险。
- 用经过验证的路径和符号更新 `CODE_MAP.md`。

### Phase 2: Original Project Smoke Test

- 验证文档声明的环境、依赖、模型加载、数据集加载，以及一条最小原始项目推理路径。
- 使用最小代表性样本数，并采用 `batch_size=1`。
- 记录所有 warning、失败、环境偏差与输出产物。
- 不得将 smoke test 视为已接受的科学基线。

### Phase 3: Original MemGen Baseline

- 建立可信、可复现的原始 MemGen comparator。
- 仅在明确批准的范围内修复或绕过 baseline blocker。
- 记录确定性的 golden cases、任务指标、延迟和显存使用。
- Golden cases 必须固定 random seed、decoding parameters、sample IDs、
  model checkpoint 和 evaluation script。
- 冻结后续阶段使用的 disabled-feature compatibility oracle。

### Phase 4: LatentMemoryBank Module Skeleton

- 添加独立的 session-level memory-bank 数据模型与配置 schema。
- 保持 `latent_memory_bank.enabled=false` 为默认值。
- 实现生命周期、校验、reset、capacity 和 isolation 脚手架，但不将其接入原始推理行为。
- 在本阶段，任何 production inference code path 都不应调用 memory bank。
- 为模块骨架添加聚焦的单元测试。

### Phase 5: Version A Integration — Reasoner Injection Only

- 仅在推理阶段集成可选 memory bank。
- 从已存储 latent memories 中检索，并仅注入到 Reasoner 路径。
- Version A 不得将检索结果输入到 Weaver。
- 保持 memory 只属于单个 session/sample，默认使用 `batch_size=1`。
- 所有存储的 latent memories 都必须从 computation graph 中 detach，且
  device/dtype 转换必须显式处理。
- 保持 Weaver 和 Trigger 训练流程不变。

### Phase 6: Disabled-Feature Equivalence Test

- 将实现后的 `latent_memory_bank.enabled=false` 与 Phase 3 冻结的 golden cases 对比。
- 要求 generated token IDs、augmentation masks、metrics、输出 schema 以及
  相关 tensor/control-flow invariants 完全一致。
- 任何差异都应视为阻断性回归。

### Phase 7: Version A Stability and Debug Experiment

- 在性能结论之前，先运行有边界的 Version A 稳定性实验。
- 测试 session reset、无跨样本泄漏、空 memory、capacity 限制、
  dtype/device 一致性、确定性重放与长 session 行为。
- 测量延迟和显存开销。
- 本阶段只修复 Version A 缺陷。

### Phase 8: Core Ablation Experiments

- 评估 retrieval on/off、write/update on/off、capacity、top-k、eviction、
  aggregation 和 recurrent update 等选择。
- 将每个 variant 与冻结的原始 baseline 以及 Version A 对比。
- 控制数据集、checkpoint、seed、decoding 和 sample count。
- 最低必做消融包括：
  `original MemGen`、`latest-k retrieval`、`random retrieval`、
  `cosine retrieval`、`cosine retrieval without recency decay`、
  `cosine retrieval with recency decay`、`append-only update`、
  `replace update`。
- 记录负结果和失败案例。

### Phase 9: Version B Integration — Weaver Input Retrieval

- 扩展 retrieval，使选中的 memory 可以参与 Weaver 输入条件化。
- 保留 Version A，作为单独可选 comparator。
- 保持所有改动仅作用于推理阶段，且 memory 仍然 session-local。
- 重新运行 disabled-feature equivalence 与有针对性的稳定性检查。

### Phase 10: Paper-Level Evidence Consolidation

- 整理 baseline、Version A、Version B、ablations、效率、鲁棒性和失败分析证据。
- 为每条 claim、每个表格和图像建立到 experiment IDs 与原始产物的可追踪关系。
- 冻结方法定义、局限性、可复现说明以及论文级证据。
- 不得把未被证据支持的假设写成结论。

## 实验记录标准

所有实验，包括 failed、aborted、smoke、debug 和 ablation runs，都必须追加记录到
`research_notes/EXPERIMENTS.md`，至少包含：

- date and experiment ID
- git branch and commit hash
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
- memory usage if available
- notes and failures

命令和路径必须记录到足以复现实验的精度。缺失项不得省略，必须明确写为
`not available`，并说明原因。

## 输出目录标准

实验产物应按方法家族归档：

```text
outputs/
├── baseline/
├── latent_bank_vA/
├── latent_bank_vB/
└── ablations/
```

- `outputs/baseline/`：原始项目 smoke 与已接受 MemGen baseline 的产物。
- `outputs/latent_bank_vA/`：Version A 的稳定性、debug 与主实验产物。
- `outputs/latent_bank_vB/`：Version B 的稳定性与主实验产物。
- `outputs/ablations/`：受控消融实验产物。

每次运行都应使用唯一的 experiment-ID 子目录，不得覆盖之前的结果。

## Phase 门禁

执行前：

- [ ] 该 Phase 已被用户明确批准。
- [ ] 范围与退出条件已经写清。
- [ ] baseline 与验证命令已经确定。

执行后：

- [ ] 验证通过，或失败已被完整记录。
- [ ] `PROGRESS.md` 已更新。
- [ ] 实验已记录到 `EXPERIMENTS.md`。
- [ ] 重要决策已记录到 `DECISIONS.md`。
- [ ] 工作已暂停，等待用户确认。
