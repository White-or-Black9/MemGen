# 论文笔记（Paper Notes）

## 初步论文主线（Provisional Story）

一种仅在推理阶段（inference-only）使用、会话级（session-level）的检索增强循环潜在记忆（retrieval-augmented recurrent latent memory），有望在无需重新训练 Weaver 或 Trigger 的情况下扩展 MemGen 可利用的上下文状态（usable contextual state），并且在关闭该功能时能够完全保持原始 MemGen 的行为不变。

> 注意：
> 这目前只是一个研究假设（hypothesis），而不是已经证实的论文结论（claim）。只有在有充分实验记录支持后，才能将其作为正式结论提出。

---

# 候选贡献（Candidate Contributions）

1. 为 MemGen 提出一种可选的、仅作用于推理阶段（inference-only）的潜在记忆机制（latent memory mechanism）。
2. 提出一种会话级（session-local）的检索与循环更新机制（retrieval and recurrent update），并提供明确的隔离保证（isolation guarantees）。
3. 提出一种兼容性设计，使得关闭 Memory Bank 时能够完全保持原始 MemGen 的行为不变（disabled-path compatibility）。
4. 提供关于模型质量（quality）、推理延迟（latency）、内存开销（memory cost）以及长会话行为（long-session behavior）的受控实验分析。

---

# 论文证据记录（Evidence Ledger）

| Claim ID | 候选结论（Candidate Claim） | 所需证据（Required Evidence） | 对应实验 ID（Experiment IDs） | 当前状态（Status） |
|----------|-----------------------------|-------------------------------|------------------------------|--------------------|
| CLM-01 | Disabled 模式能够保持与原始 baseline 完全一致 | Golden-case 等价性测试和回归测试 | TBD | ❌ 尚未支持（unsupported） |
| CLM-02 | Memory 能提升目标任务性能 | 在多个数据集和随机种子上的主实验比较 | TBD | ❌ 尚未支持 |
| CLM-03 | 性能提升来源于 retrieval 与 recurrence | 受控消融实验（controlled ablations） | TBD | ❌ 尚未支持 |
| CLM-04 | Memory 能保持 session 隔离，不发生泄漏 | 泄漏测试（leakage）和 reset 测试 | TBD | ❌ 尚未支持 |
| CLM-05 | Memory 带来的额外开销在实践中可接受 | latency 与 memory profiling | TBD | ❌ 尚未支持 |

---

# 论文结构（Paper Outline）

## 摘要（Abstract）

需要回答以下内容：

- Problem（问题）：
  - 要解决什么问题？
- Method（方法）：
  - 提出了什么方法？
- Main Result（主要结果）：
  - 实验得到什么结论？
- Cost（代价）：
  - 带来了多少时间或内存开销？
- Scope / Limitation（适用范围与局限性）：
  - 方法适用于什么情况？有哪些限制？

---

## 引言（Introduction）

建议包括：

- MemGen 在推理阶段存在什么限制？
- 为什么 inference-only adaptation（仅推理阶段适配）值得研究？
- 本文提出的核心思想是什么？
- 本文的主要贡献有哪些？

---

## 相关工作（Related Work）

需要介绍以下方向：

- Latent Memory（潜在记忆）
- Retrieval-Augmented Generation（检索增强生成，RAG）
- Recurrent Memory / State（循环记忆 / 状态机制）
- Inference-Time Adaptation（推理阶段适配方法）

---

## 方法（Method）

需要描述：

- Baseline MemGen 的推理流程
- Session-level Memory（会话级记忆）设计
- Retrieval（检索机制）
- Recurrent Update（循环更新机制）
- Memory 的生命周期与隔离机制（Lifecycle and Isolation）
- 算法复杂度（Complexity）

---

## 实验（Experiments）

需要回答：

- Research Questions（研究问题）
- Datasets and Metrics（数据集与评价指标）
- Baselines（对照方法）
- Main Results（主要实验结果）
- Ablations（消融实验）
- Efficiency（效率分析）
- Robustness and Failure Analysis（鲁棒性与失败案例分析）

---

## 讨论（Discussion）

包括：

- Interpretation（结果解释）
- Limitations（局限性）
- Broader Applicability（更广泛的适用性）

---

# 写作规则（Writing Rules）

1. 不要在没有实验 ID 支持的情况下，把研究假设（hypothesis）写成正式结论（claim）。

2. 如果负面结果（negative results）或失败实验（failed results）会影响论文结论，必须如实报告。

3. 保持训练流程不变（training unchanged）应作为一种设计约束（scoped design constraint）进行描述，而不能在没有验证的情况下宣称它是方法优势。

4. 论文中的每一张表格（table）和每一幅图（figure），都必须能够追溯到可复现的实验结果（reproducible artifacts）。