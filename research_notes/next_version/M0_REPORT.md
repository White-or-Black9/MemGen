# M0 验收报告（2026-10-02）

## 结果

**PASS_WITH_LIMITATIONS**。已完成提交版基线冻结、逐题核验、证据来源修正和 M1 计划。
这是原始输出的离线核验，不是模型重跑或新的效果实验。

- 9 个方法/消融，45 个完整效果运行，每次 5 context × 100 题，共检查 22,500 条记录。
- 完整性检查覆盖 context/query 唯一性、每次样本数、正式 latent 的只读查询记录、实际 bank 参数。
- P7/no-decay 的 question text/gold 哈希在种子内及两方法间一致。
- 原始逐题 EM、raw recall、heuristic flags 与各自历史 aggregate 一致，且与提交 PDF 的均值精度一致。
- P7、no-decay、direct injection 的论文所报总体标准差也一致。
- 检查六方法各三次效率记录（18 个）、源文件存在性、计时边界和提交版 peak incremental allocation。
- 四项 focused unit tests 通过；覆盖缺失/重复题、冻结协议破坏、非法指标与总体标准差。
- `git diff --check` 与实验索引 JSON 校验通过。
- 完整审计连续执行两次；`baseline_evidence.json` 和 `RESULT_TABLE.md` 的
  SHA256 分别为 `92255eb611099e1830c5a256e3a5b0c0d148e5d5719a6176ff8813dc66664aa6`
  和 `59cea69ef22d6db084780cb70cfe1db493efbc96ae16ebdc5a51b04d69a4c96f`，两次逐字节一致。

## 实质发现

1. 主结果可靠：P7 EM 0.1880±0.0473，Recall 0.2308±0.0419。
   老 P7 索引的 0.1968，以及脚本默认参数 campaign 的 0.1656，都不是提交版结果。
2. No-decay 的提交来源是 `no_decay_effect_repeats/20260725T_no_decay_alpha0_full`。
   后续 corrected sweep 0.1476 不可替代提交版 0.1628。
3. 论文的“parser-format failures”是 heuristic flags 计数。
   官方 parser 可接受带 Answer 前缀或多行的输出，flags 与 parse success 不是同一概念。
   Reviewer 按 flags 推算的 parsed-only accuracy 不能作为确认结果。
4. 效率数字可复算，但主 campaign GPU5、Dense campaign GPU4；提交版跨方法“same GPU”说明需修正或补严格复测。
5. 正式 P7 历史运行是 dirty worktree，config 路径当前也存在缺失。
   输出与配置的可追溯性不等于完整历史源码可复现性。
6. 提交版已注明 log axes 和效率适用范围，不必将审稿人的相应建议当作全新缺项。

## 证据与变更

`baseline_evidence.json` 保存正式 artifact paths、校验和、运行元数据及数值；
`RESULT_TABLE.md` 为离线复算表；`BASELINE.md` 定义起点；`REVIEW_ISSUES.md` 为审稿问题矩阵。
`eval/exp/manifest.json` 修正正式 P7、controls 与已完成消融的来源，旧原始输出链接不变。
`research_notes/PROGRESS.md` 与 `TODO.md` 已同步当前路线。

## 下一阶段

按 `EXPERIMENT_PLAN.md` 的 M1 执行离线重评分/候选有效性/格式分解。
先保证官方 scorer 与保存的逐题评分一致，再分析质量与格式贡献；不要先启动 retrieval redesign。
M1 是下一阶段计划，尚未实施。没有启动 GPU、新 benchmark，或自动提交/推送 Git。
