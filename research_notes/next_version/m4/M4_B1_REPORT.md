# M4-B1：选槽时间机制诊断

日期：2026-10-08。状态：COMPLETE；两次全量核验一致，M4-B1完成。
父campaign：`20261002-m3-six-policy-v3`，来源父证据`run-20261008-a1/`。
review映射R03/R04/R07，exp_id/item_id=M4-B1，section_id=retrieval_diagnostics，
paper_role=main_text，evidence_role=claim-carrying limitation analysis。

## 结论

当前五个context中，full检索不是“最后写入的两个slot”，而是与按最近访问时间、
并以slot index处理并列的recency-only策略返回相同集合。
所有25个冻结bank中，100题均取固定的两个slot，尽管每题查询向量和cosine确实变化。
这限制了当前实验对query-adaptive语义选槽的支持；不证明latent不含证据，
也不证明相似度或衰减在所有设置中都无效。

## 全量结果

|检查/诊断|结果|
|---|---:|
|父回答/策略记录验证|15,000/15,000|
|seed/question组|2,500（500唯一问题，5独立context）|
|n=0 / n=1 / n=2|0 / 0 / 2,500|
|每bank选槽集合数为1|25/25|
|每bank不同查询向量数 / cosine向量数|均为100 / 100|
|full与recency-only集合相同|2,500/2,500|
|full与last-written集合相同|0/2,500|
|full与last-written槽交集|0/5,000个入选槽位置|
|full与cosine-only集合相同|2,000/2,500；仅context0不同|
|full与cosine-only槽交集|4,500/5,000|
|保存分数与复算的最大绝对误差|0|
|full入选最小分数−最佳未入选分数的全局最小margin|0.0073888958201274585|

每个job阈值以上槽数都大于2，不存在“阈值刚好只留两槽”导致的机械等价。
阈值以上槽数范围：context0为10–13，context1为6–7，context2为9–12，
context3为8–9，context4为3–6（五个seed均纳入）。
记录cosine与score完整排序也会随题变化；变化没有越过最终top-2边界。

## 最近访问不等于最近写入

以下chunk索引从0开始；每行模式在五个seed均一致。age以当前query retrieval step为基准，
本campaign每个construction turn恰好一次检索/写入，故write step与该尺度对齐。
source chunk仅指当前版本的直接构建文本，不等于gold evidence位置。

|context|full选槽|当前写入chunk|write-age|last-access-age|构建期累计access count|
|---|---|---|---|---|---|
|0|0,12|1,13|16,4|1,1|15,3|
|1|0,1|0,1|17,16|1,1|16,15|
|2|0,1|0,1|17,16|1,1|16,15|
|3|0,1|0,1|17,16|1,1|16,15|
|4|0,1|4,1|13,16|1,1|12,15|

所有2,500组的recency截止age都有3个并列槽，取2个时按较小index选取。
full也在这批记录中选中同样两个，但full在并列访问年龄的槽之间仍按实际cosine×decay
得分排名，不能将相同集合解释成“cosine数学上完全不起作用”。
源代码为未访问槽按age衰减、被访问槽更新last_retrieved_step；上表与构建期历史访问
保留早期槽的机制一致。未开展移除访问刷新/移除衰减的因果干预，不能仅凭这些关联
断言“有害反馈”或证明decay造成效果收益/损失。

重要协议边界：query之间恢复同一bank/RNG，age/last-write数组跨所有题保持冻结。
观察到的历史访问计数来自construction，而非按测试题顺序累积的强化。
此外，native score-order与canonical slot-order不同；context4 native顺序出现两种，
其他context固定一种。排序参考策略仍单列，不能混入纯内容选择指标。

## 实现与验证边界

- 实现`audit_temporal.py`，测试`test_audit_temporal.py`；新增10项纯CPU测试通过，
  加M4-A的12项共22项；另25项M3调度/M1评分回归通过。
- 检查六策略共享cosine/age/last-write/native indices/n一致；复算
  score=cosine×exp(-0.05×age)，threshold>=0.05，再稳定降序top-k2。
  threshold判断不加容差；浮点值核验容差abs/rel=1e-12，实际误差为0。
- 六种selector均重建，包括私有RNG random；逐条核验native/decomposition排名、
  阈值标记、age、decay及final score。排序与canonical输出顺序分别检查。
- JSON不能重新生成query/key tensor，所以没有把本次审计称为“tensor-level cosine重算”。
  已核验保存的native raw_cosine与独立adapter记录一致，只在这些已存实数上复算分数。
- M3只读validator再次检查源码/输入/模型/快照hash、15,000条身份/配对/预算/零写入；
  30,162份父JSON内容hash与M4-A清单一致，M4-A七份核心产物与验收hash一致。
- 父目录及M4-A来源目录前后文件集合/size/mtime一致。没有Torch导入、模型/GPU
  调用、网络下载、训练或推理配置变更，也没有修改论文。

产物：`run-20261008-b1/diagnostics.jsonl`、`by_job.json`、`summary.json`、
`exceptions.json`、`manifest.json`和`parent_integrity.json`。
独立复跑目录`run-20261008-b1-repeat/`，不覆盖首轮。
两次diagnostics、by_job、summary及exceptions四份核心产物hash完全一致；
分析器hash、M4-A来源hash一致，两轮均通过30,162份父JSON与M4-A清单对比。
`acceptance_b1.json`验收PASS；两轮产物各约4MB，未超过资源预算。
每份报告均保留context/seed/question三个层级，不把2,500条记录当IID样本做显著性检验。

## 下一路线

当前证据支持“在这五个冻结bank上表现为固定的、访问历史相关的slot选择”，
尚不足以支持query-adaptive semantic retrieval主张。
不自动修改decay、改变key表示或启动新推理。下一阶段计划见`../M4_B2_PLAN.md`，尚未执行：B2输出行为配对分解，
区分有效候选/选错/无效输出，解释已有质量收益与选槽/输出顺序的关系；
依然不把关联分析作为格式中介或语义证据利用的因果证明。
