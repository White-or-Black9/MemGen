# M6 — 机制门与方法论文路线

最新离线失败分析见[E04_FAILURE_REPORT](E04_FAILURE_REPORT.md)：E04替换只变前缀，
六题候选正文相同，未支持Gate2语义内容贡献，也不证明construction丢失。
E03有效性/格式转移不能单独通过Gate3机制归因；raw recall仍有1题净变化待解释。
继续保留算法冻结，建议下一阶段先规划E05同槽W1/D1，未授权新GPU或Gate实现。

最新E02探索性裁决及来源映射见[E02_EXPLORATORY_REPORT](E02_EXPLORATORY_REPORT.md)。
6/10题可列直接来源候选槽；技术可复算，不等于人工独立标注或latent内容贡献。
Gate2仍未通过：缺可信语义negative、独立人工验证与内容干预。
下一建议为另批E03 on/off全部10题pilot，不能自动进入GPU或算法修改。

最初阶段只完成审计/规划；后续E01与E01V分别获批执行，详见
[E01_REPORT](E01_REPORT.md)、[E01V_REPORT](E01V_REPORT.md)。
CPU诊断验收不等于semantic/content/regeneration门通过。下面授权说明为原规划快照；
最新许可包括E01/E01V CPU诊断及E02标注包准备；实际人工双标未开始，人员与门槛
未确认，模型生成/算法修改仍未获授权。
M5路线B=先验证机制；本文件Decision A/B/C/D=**验证之后**的方法选择，概念不同。
当前保留现有bank/retrieval/regeneration作为incumbent，training-free仅指本方法推理
扩展不训练新参数，不能声称底层MemGen/投影从未训练。

## Gate 0：身份、资源与授权

本次进入时git干净，HEAD 6784a26；M5验收、M3源码/快照hash和记录覆盖核验通过。
M6只有规划批准，无生成/标注/权重加载授权。新阶段开始前先冻结批准的协议/样本/
primary contrast/MEOI/预算，确认父材料无漂移、当前GPU资源适用，再实现独立诊断入口。
本次M6完成后出现的未跟踪M6文档是本阶段产物，不是可忽略的隐式提交授权。

## Gate 1：Retrieval Query Sensitivity — E01/E01V

E01V最新更新：CPU近似标量与Full选择复现门通过；clue-only的q/排序/选择全部
不变，candidate-tail使q/部分排序变化但Full集合仍不变。几何高度相似，不直接
等同语义失败或零信息。A/B必须注明响应来源，C未过门；下一建议另批E02。
不因定位窗口局限自动进入Decision B改算法，尚缺相关内容及下游收益证据。

此前E01更新：A有记录支持；B完整排名有变化但Full及raw Top-2各bank均不变；
C保持unknown。500题重建last64均不含问题线索，应先另批E01V定位Query表示，
不按“完整排名变了”判定semantic gate通过；E02标签入口仍保留。

通过定义分层：A分数变化；B排名变化（full ranking、raw/final、Top-K集合分别）；
C选择相关历史事实。当前仅有A及部分B观察，**full集合不变化**；C unknown。
通过后的下一步：A/B可量化且输入复现后，进入E02标签；标签可用后验证source关联
和相对recency/置换的独立差异，不能直接跨到“语义检索成立”。
未通过后的下一步：若表示/排序没有足够区分度，先定位pooling窗口、量化精度、
score与decay边界问题；只提出诊断建议，不自动改Query/threshold。
不确定：完整q缺失则另批E01V；无source标签则C保持unknown；same-selection不等于
latent无信息。只有五context固定选择不能推广所有任务。

## Gate 2：Evidence / Memory Content Contribution — E02/E04

E02-PREP最新状态：冻结10题和A/B初始包已通过程序验收，见
[准备报告](E02_PREPARATION_REPORT.md)。human_annotations=0，释放门NOT_RELEASED。
这是材料门，不是本Gate语义/内容贡献通过；人员、培训预算、覆盖/一致性阈值待确认。

通过：双标覆盖/一致性/匹配门达到事前批准值；source-related候选与matched source-
unrelated候选的等长内容替换在全纳入题目strict上有方向一致的贡献，并记录残余混杂。
该通过最多支持source-associated latent候选的内容敏感效应，不自动证明每项历史事实
都被保存；强retention主张需额外事实probe/反事实construction验证。
通过后的下一步：进入E05等预算integration桥接；必要时为强事实保留另立独立计划。
未通过：E02定位/一致性失败→停止依赖mapping的C检验/E04；E04无效→保留未知或
反证、考虑定位construction/representation，而非宣布latent绝无信息。
不确定：覆盖很少、negative无法可信定义、间接来源闭包过宽、匹配位置/age/norm失衡
或多段充分性不明→补标注/协议，不能只挑有效候选或更换成输出好的样本。

## Gate 3：Regeneration Benefit Decomposition — E03/E05

通过：当前同snapshot on/off可复现，W1/D1同slot/最终8tokens可比；全题strict、
parser、candidate-valid、raw recall和完整C/W/I转移可分开解释；delta达到批准门。
H1 parser/valid差、H2正确选择差、H3内容敏感、H4query/memory异质性分别登记，
不因某一项通过宣布其余成立。C/W/I算术分解不等于因果中介分解。
通过后的下一步：与Gate2联合决定是否保持regeneration，再查看Gate4收益预测。
未通过：如果W1/D1无稳健差异，收紧“再生成必要”的贡献；如果only valid改善，
贡献定位为该设置下答案产出，不能包装历史事实融合。不给立即改算法授权。
不确定：历史full2/direct1不等R、当前复现漂移、五context效力不足、长度位置未控制
→只补最小可比桥接或重复；不得用历史P7填补当前off/direct缺口。

## Gate 4：Query-level Benefit Predictability — E03/E06

通过：跨重复/seed有稳定正负收益，推理前特征在未见context的锁定评估中优于
always-on/off和仅校准阈值；效果超过delta_policy及事前伤害界，无答案/后验泄漏。
通过后的下一步：提出Selective Conditioning的独立算法实施计划，另取授权；
已有query-conditioned regeneration不作为新创新，只研究额外收益gate。
未通过：没有可复现正负异质性、不超过简单阈值、仅拟合context或样本太少→
暂缓Selective Regeneration，保持原阈值回退及诚实主张，不直接调gate。
不确定：50题/五context不足，训练折类别缺失或没有独立validation→不拟合强模型，
请求更独立数据或收紧结论。新数据/推理必须另批。
检查阈值重新参数化：若gate与原decayed-score threshold同分数同候选，功能只是
换阈值；raw cosine门虽绕开decay，但没有独立校准/增益证据仍不足以作新方法贡献。

## Gate 5：是否修改算法 — 条件式Decision A/B/C/D

|Decision|需要的证据|方法贡献/后续动作|不能据什么决定|
|---|---|---|---|
|A 保留现有算法|Gate2内容贡献与Gate3接口价值具有可信支持，或明确接受更窄的conditioning实证贡献|组织为冻结MemGen上的session-local生命周期与Weaver-mediated reuse；明确训练/协议范围；补公平integration/来源验证及外部baseline，再独立审稿|不能仅用full>random宣布相关检索/agent能力已证；现有regeneration不能重新称为新提出|
|B 改进Retrieval|相关信息在可用slot中有内容贡献，但原selector未选到，matched替代selector能提高检索关联与下游收益|再定位Query representation、alignment、score或key/value瓶颈，提出只改一个因素的算法计划|不能仅因Query有projection认定已对齐；也不能仅因full固定或score低认定retrieval是唯一瓶颈|
|C 改进Regeneration|有可利用memory内容，但同slot接口对照或稳定query伤害表明conditioning瓶颈；gate还须Gate4|Selective conditioning/query-conditioned fusion/latent adaptation选一个可识别方向；另批实现|不能把原本已有regeneration当新创新；不能从共同valid反向结果直接推出需要gate|
|D 检查Construction|可信来源事实在编码前可用，但slot内容验证/源事实干预表明写入/压缩/覆盖阶段没有可靠保留|先独立分析写入、压缩、refresh/evict版本，必要时设计construction可证伪实验|source无法定位、最终slot没有direct来源或probe失败均不足以单独宣称construction丢失事实|

若多门矛盾或不确定：保留incumbent，不默认挑一个重设计；报告最小缺口与停止条件。
当前判断：没有足够证据批准B/C算法改动或断言D为根因；Decision A也没有“已可投稿”
保证。只能保留方法作为对照，先验证。

## 下一版NAACL/COLING方法论文的最低证据边界

题目、摘要、贡献必须以已过机制门为准：检索、内容保留、再生成、选择性各自绑定
证据，不能用总体EM一条链替代。方法型定位还需要清楚的新颖性差异、可复现tensor
接口、可信baseline、合理任务范围。M6只诊断这条方法线，不做新文献事实声明，
不承诺会议接收，也不虚构第二任务/模型、竞争latent方法或动态读写结果。
R08/R09/R10/R11–13外部泛化、动态、竞争对照和效率缺口仍在M5账本，机制通过不
自动解决它们；这些任务须后续独立计划与批准，不能越过本阶段边界。

## 现在可以请求的明确批准

优先只请求 **E01 CPU离线诊断实现与运行**：复用原始分数/norm/排序及tokenizer-only
窗口检查，不加载LM/不生成答案、不改算法；结果决定是否需要E01V。
E02的10题双标pilot须确认标注人员、预算和一致性/覆盖门；E03的10题on/off
重复pilot须独立批准GPU、40次生成上限、诊断接口实现与资源/时间预算。
批准其中一项不等于批准全部后续项目；M6到此停止。
