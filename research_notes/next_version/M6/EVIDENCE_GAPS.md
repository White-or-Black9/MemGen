# M6 — Evidence inventory and gaps

最新E03/E04离线证据见[E04_FAILURE_REPORT](E04_FAILURE_REPORT.md)：
Observed：E04六题四个raw变化仅前缀不同，候选正文6/6一致；c2-02回退为解析前缀，
未发现记录级错槽/写入/版本漂移。E03 raw recall3/10 vs2/10，不是全部格式效应。
Unknown：来源事实保留、语义融合、不同slot norm/访问age的因果作用；正式E02门未过。
应收窄“内容敏感”到输出接口层，不扩写成语义记忆复用；下一仅建议E05同槽接口规划。

最新E02探索性更新见[E02_EXPLORATORY_REPORT](E02_EXPLORATORY_REPORT.md)。
Observed：AI辅助标注的坐标/身份及6题source-to-current-slot关联可复算；
Unknown：独立人工一致性、latent事实保留、语义negative、下游内容贡献。
用户确认复核不改变原文件AI单标来源。本次不是正式E02门通过，也不支持调算法。

状态标签：**Observed**=已存记录/代码核验；**Inferred**=基于事实的有限推论；
**Hypothesized**=待干预检验；**Unknown**=没有足够数据。实现存在不等于效果已证明。
以下1–5节保留最初审计时的材料状态；E01/E01V新增证据及当前缺口见第6节。

## 1. 资产与信任分层

|资产|信任/用途|本次检查与限制|
|---|---|---|
|[提交PDF](../../../review/nference_Time_Latent_Me.pdf)、review原件|历史论文/评审锚点，不是新方法结果|PDF SHA已由M5 validator验证；历史dirty代码不可精确重建|
|[M0](../M0_REPORT.md)、[M1](../m1/RESULTS.md)|历史输出复算、比较/负向证据|官方指标与新增strict分开，不作为当前M3同协议baseline|
|[M3](../m3/M3_REPORT.md)、campaign RESULTS/results/scored_records|trusted主实验范围内证据；主文候选|本次确认25done/15,000路径与scored条数/25快照/92源码hash，不重跑效果|
|[M4-A](../m4/M4_A_REPORT.md)、provenance/blind_sample/chunk_offsets|可信输入来源追踪；语义标签仍unknown|32份A/B1/B2两轮核心产物hash通过M5验证；50blind样本未标注|
|[M4-B1](../m4/M4_B1_REPORT.md)|可信选槽诊断/限制，主文候选|固定选槽、recency一致、不同查询hash/cosine；仅五context|
|[M4-B2](../m4/M4_B2_REPORT.md)|可信输出记账，支持/负向证据|转移不是因果贡献率，共同有效集合有输出选择偏差|
|[M5账本](../m5/CLAIM_EVIDENCE.md)、[分流](../m5/REVISION_LEDGER.md)、decision/acceptance|可信修订材料；internal/reproducibility|decision null是历史状态；用户已选B在M6记录，不篡改M5|
|paper_experiment_matrix/新选定outline|未发现对应文件/未选定|M6用稳定实验ID映射C/R；不是已覆盖新论文outline|

## 2. 不混合历史P7与当前M3

|层|已有结果|能说明什么|
|---|---|---|
|历史P7/M1|full official EM18.80%、strict11.00%；direct_top1 official4.72%、strict3.40%；no-conditioning official.80%、strict0%|历史输出行为差异；不能与当前M3逐题比较Weaver机制|
|历史no-decay/M1|official16.28%、strict11.72%，full strict11.00%|跨指标方向不一致，不能宣称decay普遍改善正确率|
|当前M3|full/random/last-written/cosine/recency/native strict20.20/3.92/1.80/16.80/20.20/11.80%|同bank/RNG/count匹配下策略差异；没有当前off或direct组|
|当前M4-B2|full−random净+407，其中I↔C +399、W↔C +8；full−last +460=+470−10|大差异伴随有效候选产出变化，非格式或语义因果中介证明|

## 3. 分命题证据账本

|ID|命题与状态|已知范围/缺口|后续设计|
|---|---|---|---|
|GAP01|A：query变化会变score — Observed|25bank各100种query hash/cosine数组；hash不同不量化表示区分度|E01用已有norm/score；完整QQ几何需补向量|
|GAP02|B：query变化会变ranking — 部分Observed|M4已记录排序随题变化、native顺序ctx4变化；完整排名/Top-2边界应分开量化；full集合25/25固定|E01分raw/final全排名与集合，不把“分数不同”当选槽不同|
|GAP03|C：ranking选择相关历史事实 — Unknown|无可信span/latent标签；full与访问recency相同|E02标签后E01扩展来源关联；E04内容干预，不直接报semantic Recall|
|GAP04|Query表示可能受模板/候选尾部影响 — Hypothesized|已知last64输入embedding池化；比例、含义尚未知|E01窗口组成与可选E01V几何；不默认改pooling|
|GAP05|当前full与recency等价 — Observed，有限设置|2,500对选槽和原始输出全相同；不是最后写入（槽交集0/5,000）|沿用，不重复模型比较；不能推广所有任务/alpha|
|GAP06|来源可定位意味着latent事实保留 — 不成立的推论|EA可追踪400slot版本，但相关语义Unknown；历史依赖闭包仅可能输入|E02→E04；若无候选槽也不能证明压缩丢失|
|GAP07|Weaver收益源自格式 — Hypothesized H1|旧heuristic不是parser失败，当前无效多为noncandidate|E03/E05真parser与valid严格分开；记账不能证明中介|
|GAP08|Weaver改善候选选择 — Hypothesized H2|现有strict差支持行为变化，不排除大部分无效转移；条件子集有反向|E03全题3×3转移，E05控制接口；不得删反向或只算有效子集|
|GAP09|Weaver对历史内容有用且敏感 — Hypothesized H3|native顺序改变有输出变化，但顺序不是事实内容；source相关不等于内容正确|E04相关/不相关条件差，保留matched source/confounds|
|GAP10|收益依赖query/memory — Hypothesized H4|无当前配对on/off；random/last反向条件子集不是on/off伤害证据|E03 query-level配对，E04内容×query交互；5seed可靠性另批|
|GAP11|Gate可预测哪些query应使用memory — Unknown|已有阈值全部通过；未校准、无收益标签、无独立验证|E06只在E03可靠正负异质性后计划预测；不实现gate|
|GAP12|Regeneration比direct好就是融合机制更好 — Unknown|历史direct1 vs full2不等R；当前M3没有direct|E05同slot一槽桥接，避免budget混杂|
|GAP13|greedy就完全确定 — 不成立|M3保存Python/NumPy/Torch/CUDA RNG；deterministic_algorithms=false|E03先固定小规模重复，记录非确定性，不能择优答案|

## 4. 数据是否已经足够

- 已有：每query×slot raw cosine/final score/rank/age/decay、q hash、q norm；
  bank memory/key shape/norm/hash、已存tensor快照；rendered prompt；全部六策略答案。
- 本次逐条确认full 2,500个query_norm有值；query_cosine_to_first/previous全部null，
  query完整向量未序列化。每题单独调用runner导致这些跨query参考未累积。
  因此norm分布可离线做；QQ cosine、逐维方差不能从hash/norm完整恢复。
  16个key方向的cosine也不能唯一确定1536维query。
- bank tensors在可信本地产生的25份frozen_bank.pt中；M6仅核对文件hash，未反序列化。
  需要QQ几何时先选择embedding+projection-only CPU提取；无需必然跑Weaver/Reasoner生成，
  但需单独批准加载权重/导出诊断数据并核对dtype/tokenization。
- 无：可信source-span标注、latent语义真值、当前同snapshot off/direct的答案。
  旧P7这些组不能填补当前配对缺失。
- 50blind样本及五context原文/offset存在，可以提出标注方案，**可行性尚未由人验证**。
  gold概述0/500逐字匹配只是词面定位失败，不证明原文没有证据。

## 5. 负向结果必须保留

full/cosine严格净差+85仅ctx0；full/recency不仅准确率相等，所有原始输出相等。
共同有效full/last为25/55 vs35/55，full/cosine为410/1420 vs415/1420；这些是
输出选择后的条件描述，不能宣称某方法总体更好/更差或排除格式混杂。
全题strict始终是主指标。历史no-decay strict反向、旧来源限制、顺序敏感性一并保留。

## 6. Current board

最新E02-PREP更新：[准备报告](E02_PREPARATION_REPORT.md)。10题、完整原文、A/B
独立空表与协调员来源索引已导出并复跑一致；没有人工证据标签、没有语义mapping。
人员/门槛未确认，NOT_RELEASED。仅资产准备状态变化，不更新检索/latent正面主张。

此前E01V更新：[E01V_REPORT](E01V_REPORT.md)。Observed：500个CPU q在clue-only
全部不变、candidate-tail-only全部改变；Full集合均不变。CPU QQ均值0.996762，
有效秩21–36；标量近似复现及Full选择门通过，原GPU hash匹配仅670/2500记录。
Inferred：该初始prompt检索表示不直接利用前部clue。Hypothesized：窗口局限造成
EM损失或改Query能改善效果，未验证。Unknown：语义相关性、latent事实保留和
当前on/off收益。完整原GPU q仍未恢复；CPU几何不再缺失。下一请求E02双标pilot。

此前E01执行更新（其“QQ未知/下一步E01V”已被上面替代）：[E01_REPORT](E01_REPORT.md)。
Observed：500题重建last64均无question-clue；固定指令/chat46token；25bank的raw
与Full各有唯一Top-2集合。同bank机械置换raw/final完整排名分别改变1956/923组，
Top-2均0组改变。Inferred：初始embedding→线性投影池化路径不直接利用前部clue。
Hypothesized：该窗口局限造成检索/EM损失；尚未干预检验。Unknown：QQ几何、
source相关性、latent事实保留、当前on/off收益。metadata.source是子集名而非作品，
不能证明context独立。下一请求改为E01V CPU表示诊断/E02标注pilot，非算法修改。

以下保留原规划快照：

current_mainline=M6机制审计/实验设计；incumbent=现有冻结MemGen bank+retrieval+Weaver；
latest_decisive_result=固定选槽/recency等价及输出转移；active_blocker=可信source/latent
标签和当前on/off配对缺失；stale_routes=把旧P7消融当M3/把query regeneration当新创新；
next_decision_scope=批准E01离线诊断，之后独立批准标注或小GPU配对；budget_class=本轮
CPU只读/文档。下一版方法论文的事实复用/机制结论与可发表性不可混为一谈。
