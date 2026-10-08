# 分阶段修改计划：提交版为基准

用户授权先计划、执行并验收当前阶段，再依据证据制定下一阶段。
Weaver/Trigger 训练不修改；bank disabled 原行为保持；单 context 不跨样本共享。
新 benchmark 扩展在 M1–M4 核心机制证据完成前保持暂停。

当前进度（2026-10-08）：M0/M1/M2/M3已完成；M3全量15,000条及评分验收见
`m3/M3_REPORT.md`。M4-A来源/证据定位离线审计已获批准并完成，两次全量结果一致，
见`m4/M4_A_REPORT.md`。400个slot来源可重建，但500题证据span/latent相关性未知。
M4-B1已有trace诊断也已获批准完成，见`m4/M4_B1_REPORT.md`；full固定选槽且与
访问recency一致，不是last-written。M4-B2输出行为分解也已完成并复跑验收，
见`m4/M4_B2_REPORT.md`：净正确变化主要伴随无效候选↔正确转移，不证明格式因果。
M4核心分析结束；M5主张与证据缺口整合也已批准完成，见`m5/M5_REPORT.md`。
下一步需用户选择A收紧论文或B先验证机制；不自动新增GPU实验或编辑论文。

以下为2026-10-02路线记录（M3状态已被上述验收更新）：M0/M1 完成；M2 CPU + 10题 GPU 协议验证已完成，
见 `m2/M2_REPORT.md`。原 M2 的完整实验部分现在独立为下一阶段 M3，
执行规格见 `M3_PLAN.md`；六策略/25 construction jobs/15,000 个唯一回答，已批准实施。
正式实现与恢复入口见 `m3/M3_REPORT.md`；当前 campaign 为 `20261002-m3-six-policy-v3`。
完成 CPU 验收不等于正式效果验收，必须检查全部15,000条记录后才能汇总。
原路线中的 relevance/provenance 与 representation 检查顺延，仍以正式结果决定是否进入。

## M0：基线冻结与来源核验

- 以提交 PDF SHA256 为唯一身份；固定正式 campaign，不使用 latest/best 自动选择。
- 从逐题记录复算 6 方法和 3 消融的 EM/Recall/heuristic flags；45 个效果运行。
- 检查每次恰好 5×100 个唯一 context/query identity；正式 latent 路径核验 query writes=0、snapshot 未变化。
- 检查 18 个效率记录的样本数、计时边界、物理 GPU 与三次重复。
- 记录 raw artifact hashes、实际配置、来源缺口；生成基线、问题矩阵、结果表及机器报告。
- 原件、checkpoint、PDF 保留；不自动 Git 提交或 GPU 复跑。

验收：数值在提交版精度内一致、报告可重复生成、缺失来源明确披露。
`PASS_WITH_LIMITATIONS` 允许继续离线分析，但不代表历史代码可精确重建。

## M1：先修正评估定义，再拆解 EM（已完成，2026-10-02）

实际验收与结果见 `m1/M1_REPORT.md`、`m1/RESULTS.md` 和 `m1/summary.json`。
官方评分无差异；P7 strict candidate accuracy 为 11.00%，候选有效率 37.64%。
下一阶段执行规格已转入 `M2_PLAN.md`；下列 M1 规格保留作为已执行的契约。

输入使用 M0 冻结的逐题记录与其 hashes，不重新生成答案。
实现入口固定为 `scripts/eval/review_eventqa_decomposition.py`，输出放在
`research_notes/next_version/m1/`；不改 M0 的历史结果表。

1. 从实际 MemoryAgentBench scorer 源码读取 parser、normalize、substring EM 和 raw recall；保存源码 hash 与 benchmark revision。
   保留官方算法，对已存 P7 parsed_output 进行逐题一致性检查。
2. 报告四个独立概念：parser 返回 None；parser 返回空字符串；解析结果唯一对应 candidate；原有 heuristic flags。
   “parser 返回非空”与“candidate-valid”分别作为分母，不用 flags 代替。
   Candidate-valid 定义为 parsed text 经官方 normalize 后与唯一候选全文相等；
   包含多个候选、只有部分 event 文本或 normalization 冲突都单独标记，不算有效选择。
3. 同时报告原 substring EM、strict candidate accuracy、raw recall、各分母下的 conditional accuracy。
   明确 strict accuracy 是新增诊断，不能替换历史主指标。
4. 从官方问题候选提取 candidate set，按题核查 gold coverage、候选重复与多 gold。
   Chance 为均匀随机选择唯一 candidate 时命中 gold 的概率，按题计算后平均；无法解析时列为缺失，不默认为固定 1/6。
5. 使用互斥分类：正确有效选择；gold 在输出但有效选择错误；gold 在输出但无有效选择；gold 不在输出。
   同时保留多 gold、歧义输出的独立字段，校验分类总数。
6. 逐 seed/context 对齐比较 P7、Recent-text、Matched16、BM25、Dense、summary；报告每次结果及总体统计。
   不将格式稳定性直接解释成 evidence retrieval 改善，也不把种子重复当独立样本。

产物：独立离线 analyzer、机器结果、逐题诊断、Markdown 表、失败示例、claim/gate 判断。
验证用例覆盖 Answer 前缀、多行、空输出、花括号、超过24词但可正确解析、有效错误候选、gold 出现在后续行、重复候选与缺失 gold。

M1 gate：先验证官方分数逐题一致，再讨论正确率与格式贡献；若一致性失败，停在 scorer/data 对齐修复。
若格式变化为主，缩小 claim；是否重设计 retrieval 必须由 M2–M4 决定。

## 后续方向（未展开为执行规格）

M2 公平对照协议验证（已完成） → M3 六策略正式比较（已批准、实施中）
→ relevance/provenance → representation diagnostic（后两项依正式结果规划）。
此后依据结果决定保留或重设计 retrieval，再规划 evidence age、seed/statistics、dynamic bank、external reference 与论文重写。
第二任务/模型不是当前执行范围。
