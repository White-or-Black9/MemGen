# 研究待办事项（Research TODO）

## 工作流程规则（Workflow Rules）

- 所有任务必须按 Phase（阶段） 进行分组。
- 未经对应 Phase 批准，不得将任务移动到实际开发（active work）。
- 对于被阻塞（blocked）的任务，必须注明阻塞原因（blocker）以及需要如何解除阻塞。
- 每完成一个 Phase 后，需要将本文件与 PROGRESS.md 保持一致并进行同步更新。

---

# 待办任务（Backlog）

## Phase 0：审计与基线（Audit and Baseline）

- [x] 确认推理（inference）入口及调用图（call graph）。
- [x] 确认 Weaver 与 Trigger 的训练边界，保证后续工作不会修改它们。
- [x] 定位运行时配置（runtime configuration）及默认处理逻辑。
- [x] 定义 session / sample 的生命周期。
- [x] 定位 latent representation（潜在表示）的创建与使用位置。
- [x] 定位 generation 输出、日志记录和评估（evaluation）相关 hook。
- [x] 选择官方 baseline checkpoint，并通过哈希校验（hash verification）验证其完整性。
- [x] 记录标准 baseline 命令及其评测指标（metric contract）。
- [x] 定义精确的 disabled-feature 兼容性测试。
- [x] 更新 CODE_MAP.md 与 BASELINE.md。
- [ ] 在一个单独批准的 Phase 中修复 BUG-0001。
- [ ] 在修复 loader 后，重新运行并正式接受完整 baseline。

---

## Phase 1：Session-Local Prototype（会话级原型）

- [ ] 定义 Memory Bank 的接口（interface）和生命周期（lifecycle）。
- [ ] 定义 retrieval（检索）、recurrent update（循环更新）、capacity（容量）和 eviction（淘汰）机制的默认策略。
- [ ] 添加可选（opt-in）配置，并默认保持关闭（disabled）。
- [ ] 实现仅限 inference 的集成（inference-only integration）。
- [ ] 强制保证每个 session 独立（per-session isolation），默认使用 batch_size=1。
- [ ] 测试 disabled-path 等价性（确保关闭 Memory Bank 时行为完全一致）。
- [ ] 测试 reset 逻辑以及不存在跨 sample 的 memory 泄漏。

---

## 后续阶段（Later Phases）

- [ ] 设计 retrieval 与 update 的消融实验（ablation）。
- [ ] 测量模型质量（quality）、延迟（latency）和内存开销（memory overhead）。
- [ ] 开展鲁棒性分析（robustness）和失败案例分析（failure analysis）。
- [ ] 整理形成可直接用于论文的实验证据（paper-ready evidence）。

---

# 当前进行中（Active）

- 无（None）。

当前 Phase 0 已在 baseline gate（基线验证关口）暂停。

---

# 阻塞项（Blocked）

- 可信（trusted）的 baseline 运行目前被 BUG-0001 所阻塞。

---

# 已完成（Done）

- [x] 初始化长期研究笔记（long-term research notes）和提示词模板（prompt templates）。