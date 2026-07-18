# 研究待办事项（Research TODO）

## 工作流程规则（Workflow Rules）

- 所有任务必须按 Phase（阶段） 进行分组。
- 未经对应 Phase 批准，不得将任务移动到实际开发（active work）。
- 对于被阻塞（blocked）的任务，必须注明阻塞原因（blocker）以及需要如何解除阻塞。
- 每完成一个 Phase 后，需要将本文件与 PROGRESS.md 保持一致并进行同步更新。

---

# 待办任务（Backlog）

## Current Next Step（当前下一步）

- [x] Complete LongBench v2 Phase 0 official-dataset audit and freeze the
  18-row smoke / 60-row bounded sample manifests.
- [x] Complete LongBench v2 Phase 1 adapter, deterministic scorer, lifecycle
  contracts, unit tests, and official-data no-model fixture.
- [x] Complete the LongBench v2 frozen 18-item three-method GPU smoke.
- [x] Complete and audit the approved LongBench v2 60-item bounded evaluation.
  The valid v3 merged artifact records P7/no-query `17/60` each, with paired
  wins/losses/ties `5/5/50` and exact sign-test `p=1.0`; the route is closed.
- [ ] Perform a read-only EventQA supplementary-evidence inventory, then
  pre-register exactly one focused follow-up experiment. GPU execution needs
  separate approval and must not overwrite the frozen P7 main result.
- [ ] (Paused by DEC-0089) Do not start BABILong, MemBench, InfiniteBench, or
  LongBench v1 through the former fallback gates.
- [ ] If the paper line is reopened, start with an independent skeptical
  review of the current EventQA-centered manuscript package.
- [ ] Keep DetectiveQA as appendix / supplementary / stress-test evidence
  only; do not promote it into the manuscript main table without a new
  explicit decision.
- [ ] If DetectiveQA is reopened later, use only the aligned runner; do not
  reuse the historical `disabled` vs `p7_no_query_retrieval` interpretation
  without an aligned rerun.
- [ ] If DetectiveQA is mentioned in the paper, describe it only as aligned
  negative diagnostic / failure-analysis evidence unless a new positive rerun
  explicitly changes that status.
- [ ] Keep FactConsolidation dropped under the current frozen-P7 setup unless
  a new benchmark question explicitly reopens it.
- [ ] Keep `RULER-QA2` dropped after the adapted frozen-bank negative trial;
  do not reopen it unless a new benchmark-policy decision explicitly requests
  a fresh audit.
- [ ] Do not launch new benchmark runs while DEC-0089 is active. EventQA
  supplementary execution, if selected after the evidence inventory, also
  requires an explicit execution-phase approval.

---

## Accepted Formal Results（已接受的正式结果）

- [x] Phase 0-7 records are the accepted formal result set.
- [x] Original MemGen fixed 20-sample GSM8K comparator accepted:
  `EXP-20260611-006`.
- [x] Disabled-path equivalence accepted:
  `EXP-20260612-013`.
- [x] Version A enabled stability/debug evidence accepted through Phase 7,
  without performance claims.

---

## Historical / Exploratory Records（历史或探索性记录）

- [x] Phase 8A GSM8K pilot exists, but is historical / exploratory only.
- [x] Phase 8C-alt controlled mechanism study exists, but is mechanism evidence
  only and does not replace TriviaQA.
- [x] Phase 8D-0 / R4-1A TriviaQA discovery / preflight exists, but is not a
  formal evaluation.
- [x] Phase R2 / R2-fix defines the current Version A-aligned mechanism, but no
  formal target-task experiment has been run after R2.

---

## Superseded / Resolved Historical Items（已由后续记录取代）

The items below are retained for provenance. They are no longer active blockers.

### Phase 0：审计与基线（Audit and Baseline）

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
- [x] 修复 BUG-0001：已由 Temporary Repair Phase 的 checkpoint adapter
  restore 逻辑修复，并通过 112/112 Weaver / Trigger adapter tensor 验证。
- [x] 修复 loader 后重新运行并正式接受 baseline：已由 Phase 3
  `EXP-20260611-006` 完成。

### Phase 1：Session-Local Prototype（会话级原型）

- [x] 定义 Memory Bank 的接口（interface）和生命周期（lifecycle）。
- [x] 定义 retrieval、capacity、update 和 eviction 的当前策略。
- [x] 添加可选（opt-in）配置，并默认保持关闭（disabled）。
- [x] 实现仅限 inference 的集成（inference-only integration）。
- [x] 强制保证每个 session 独立（per-session isolation），enabled mode
  默认 / 当前要求 `batch_size=1`。
- [x] 测试 disabled-path 等价性（确保关闭 Memory Bank 时行为完全一致）。
- [x] 测试 reset 逻辑以及不存在跨 sample 的 memory 泄漏。

---

## 后续阶段（Later Phases）

- [ ] 设计 retrieval 与 update 的消融实验（ablation）。
- [ ] 测量模型质量（quality）、延迟（latency）和内存开销（memory overhead）。
- [ ] 开展鲁棒性分析（robustness）和失败案例分析（failure analysis）。
- [ ] 整理形成可直接用于论文的实验证据（paper-ready evidence）。
- [ ] 这些后续任务当前不是自动执行项；只有在论文线或 benchmark
  policy 被明确重开后，才进入新的 formal experiment 设计。

---

# 当前进行中（Active）

- No benchmark scheduler or GPU experiment is active under this note update.
- The active planning task is EventQA supplementary-evidence inventory only;
  it is read-only until one pre-registered follow-up receives approval.

---

# 阻塞项（Blocked）

- New benchmark expansion is paused by DEC-0089.
- EventQA supplementary execution is blocked until the evidence inventory
  identifies one focused question and the execution phase is explicitly
  approved.
- Version B is blocked until the disabled-memory target-task path is stable and
  explicitly approved.

---

# 已完成（Done）

- [x] Synchronized the additive-benchmark closure across
  `research_notes/EXPERIMENTS.md`, `research_notes/DECISIONS.md`,
  `research_notes/PROGRESS.md`, `research_notes/PLANS.md`, and
  `research_notes/TODO.md`.
- [x] 初始化长期研究笔记（long-term research notes）和提示词模板（prompt templates）。
