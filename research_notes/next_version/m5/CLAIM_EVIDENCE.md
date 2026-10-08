# M5：提交版主张—证据账本

2026-10-08；这是内部修订材料，不是已修改稿或完成的rebuttal。
论文唯一锚点：[提交PDF](../../../review/nference_Time_Latent_Me.pdf)。
SHA256：`6b04a3c49b9bb68cccf282059be34f7dadc516e4069e42b204fcae99f68ffed3`。
位置为PDF页码；下面均为主张概括，不是逐字引文。

证据索引：E0=[M0](../M0_REPORT.md)，E1=[M1](../m1/RESULTS.md)，
E2=[M2](../m2/M2_REPORT.md)，E3=[M3](../m3/M3_REPORT.md)，
EA=[M4-A](../m4/M4_A_REPORT.md)，EB=[M4-B1](../m4/M4_B1_REPORT.md)，
EC=[M4-B2](../m4/M4_B2_REPORT.md)。审稿原件：[review](../../../review/review)。

## 不可混合的两个实验层

历史P7：提交版官方substring EM=18.80%，raw recall=23.08%；E1新增strict候选
准确率=11.00%，候选有效率=37.64%。历史dirty源码无法精确重建；输出可复算
不等于历史代码完全可复现。Matched-16是**文本16-token控制**，历史strict为0.80%。

当前M3：新冻结源码、同bank/RNG、六策略比较，15,000回答，2,500/策略。
full strict=20.20%、官方EM=28.40%、有效率=69.40%；random strict=3.92%，
last-written=1.80%，cosine-only=16.80%，recency-only=20.20%，native-order=11.80%。
这是新campaign，不能把20.20%当作历史P7重评分，也不能将这些策略与历史文本
控制拼成同一公平主表。五seed重复500题，独立context仅五个，不是2,500 IID样本。

|ID|提交位置/主张|裁定及可保留的边界|证据/审稿项|拟改动，尚未实施|
|---|---|---|---|---|
|C01|p2–4 冻结参数、session-local、有界bank生命周期|支持操作层描述；不构成算法原创性或语义正确性证明|E0/E3/EA；R10/R16|保留工程贡献；明确构建→冻结→独立查询→reset，刷新替换不保证语义合并|
|C02|p1/3 relevance-based、query-conditioned检索历史支持|需收紧：查询向量变化但每bank选同两槽；full与访问recency完全一致|EA/EB/EC；R03/R04|区别“按query计算分数/生成latent”与“已验证query-adaptive相关选槽”；后者不能声称|
|C03|p5–6 full优于文本控制|历史官方指标支持有限比较；不能排除任务适配、答案产出等混杂|E0/E1；R01/R02/R08|保留历史表，补strict/valid及chance参考；删除由结果推出已检索相关原文的排他性解释|
|C04|p1/5–7 更少parser-format failures|原命名错误：heuristic flags不是官方parser失败；有效候选也不是解析成功|E1/EC；R01|改指标定义及图注；真实None/空、非候选、多候选分别报告，不把旧flag重命名为valid率|
|C05|p6/7 消融证明Weaver consolidation必要且解释收益|支持接口选择影响行为；不证明融合机制或保留历史语义|E1/EC；R05|将机制必然性收紧为该配置下经验差异；补顺序敏感性及接口审计待办|
|C06|p6 temporal weighting improves reliability|无一致跨指标/因果支持：历史no-decay官方16.28%而strict11.72%高于full11.00%；现有B1不是no-decay干预|E0/E1/EB；R06/R07|保留数值与不确定性，删除普遍decay收益/显著性暗示；若保留强主张另行批准配对实验|
|C07|p1–2/7 long-horizon agents、tool use适用性|未验证：仅一个checkpoint、五个EventQA context；原文span/证据年龄仍未知|EA；R08/R09/R14|标题/摘要/结论收紧到EventQA和MemGen，agent/tool-use仅动机，不写实测能力|
|C08|p3–4 在线update/evict支持动态agent|实现生命周期不等于动态任务收益；主要评估是冻结查询，动态下降原因未确立|EA/E3；R09|分别陈述construction-time更新与query-time无写入；动态结果先补来源，未知原因显式保留|
|C09|p6–7 效率与matched no-bank、同GPU比较|仅历史协议下增量GPU分配及含准备摊销时间；same GPU与GPU4/5来源冲突|E0；R11/R12/R13|去掉同物理GPU保证及普遍效率推论；指明baseline、CPU bank、总量未测、每context100题摊销|
|C10|p2 lifecycle/latent方法贡献及竞争力|可描述MemGen接入点；相对latent竞争方法的效果/新颖性未验证|E0；R10|不写超过MEMORYLLM/M+；EM-LLM/Memoria按审稿线索待原文核查，不把审稿描述当已验证文献事实|
|C11|p7 泛化、可追踪与纠错|部分限制原文已有；slot输入来源可追踪不等于latent内容可解释/可纠错|EA；R14|补不支持的外推、刷新覆盖风险与内容审计缺口，不把provenance当语义标签|
|C12|全文 可复现及seed稳定性|新M3身份链支持；不修复旧dirty源码、greedy方差来源或统计效力|E0/E3；R06/R16|单列两campaign身份及恢复尝试；报告五context描述性稳定，CI探索性，不宣称显著|

## 必须保留的反向/限定结果

- 当前full−random净正确+407 = 无效↔正确净项+399 + 有效错误↔正确净项+8；
  full−last-written +460 = +470 −10。仅记账恒等式，不能当作格式/语义因果贡献率。
- 共同有效子集由输出选择：full vs last-written为25/55 vs35/55；
  full vs cosine为410/1420 vs415/1420。不能据此宣称对照总体更好，也不能删掉反向结果。
- full/recency 2,500对的原始输出、解析、候选、有效性、正确性全部相同。
  full/last-written槽集合交集0/5,000；访问recency不是末写入。
- full/cosine增益仅context0；full/native原始输出940/2,500不同、净正确差210。
  选槽顺序敏感不等于证据保留证明，也不直接解释历史greedy方差。
- 500题无可信source-span，gold摘要没有逐字匹配不是“原文没有证据”；
  50条blind bundle未人工标注。禁止输出伪semantic Recall/MRR。

当前最强可辩护结论：在指定MemGen/EventQA冻结bank配置下，latent conditioning
及其选槽/顺序改变候选答案产出；full优于指定随机/末写入策略。
尚不能证明查询相关历史证据检索、长程语义保留或真实动态agent能力。
