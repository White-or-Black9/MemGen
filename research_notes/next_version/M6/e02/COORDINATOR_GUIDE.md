# E02协调员操作说明（标注尚未开始）

本次只完成准备；release_gate各人员/数值门仍未批准，不自动释放任务。
标注者身份、既往暴露、训练/长文阅读预算、预注册验收门确认后方可分发。
每个初始包只分发`rater_A/`或`rater_B/`，不要分发M6、coordinator_only、manifest
或其他父目录。当前目录隔离**不是访问权限隔离**；正式交付需独立副本/访问控制。
不联系或发送给任何人员，除非用户另行要求。

## 1. 两阶段独立标注

1. 为A/B创建各自工作副本，保留已hash的原空表。记录真实身份、角色、培训、
   先前接触本项目/模型输出情况。A/B不得互看或讨论pilot具体答案。
2. 两人各自阅读题面及完整原文，选择候选、定位span、说明必要事实与最小充分集合。
   用RATER_GUIDE状态；保留全部十题，包括unknown/ambiguous/conflict，不补换题。
3. 两份phase1都提交后分别冻结SHA、时间、原文/问题身份；不允许看gold后覆盖。
4. 才单独提供对应dataset gold（roster_and_gold），每人填写phase2_gold_review副本，
   记录phase1 hash、同意/疑似标签问题/理由。gold不是必须服从的事实真值。
5. 第三人以原文、A/B原始标注及gold审核材料裁决；先看原文证据再看gold，不提供
   方法预测、retrieval score或模型效果。不因“符合模型答案”选证据。

人员未知时不能称双标完成；同一人两遍、两个AI输出或协调员代填不算独立人工双标。
如果规则需要修订，保留v1与原phase1，对pilot重标并版本化；不能将修订后的pilot
冒充从未见过的确认集。扩至原冻结50题须另批，不自动启动。

## 2. 数据完整性检查

每题sample_id必须一一对应、无遗漏或额外题；rater_id非空；status为合法枚举，
候选index从0起且存在于题面；实际工作时间和completed_at记录完整。
context原文SHA与question SHA匹配，禁止清理文本后沿用旧offset。
每段span用Unicode码点[start,end)，逐字文本必须相同；`prepare_e02.check_span`
只验证坐标，不判断语义。supported至少一个sufficient集合与唯一候选，其他状态
可为空证据但须有理由。多段充分集合的每个必要成员及替代集合不得丢失。

phase2和裁决表都引用冻结phase1哈希，不能把空白表hash冒充人工提交。
程序校验通过不证明span足够或gold正确。semantic审核必须独立人工完成。

## 3. 一致性指标协议（本轮不计算）

主可行性指标：裁决后可定位充分证据题数/全部10题；另报A/B各自覆盖、未知和
歧义比例、每context分母2。原始10题全部保留，不只统计双方supported的有利子集。
门槛`delta_coverage`等仍待批准，不能看到结果后设标准。

- status：6×6混淆表、原始一致率、Cohen kappa；若期望一致率=1导致分母0，
  kappa报null并解释，不强行填1或0。10题类别稀疏，kappa不能单独决定可行性。
- 候选：全题选择集合一致率；同题双方未选候选与真正答案一致分开记录。
- span：冻结M3实际base tokenizer.json身份，仅tokenizers编码原context，无LM；
  token与字符span有正交集即归入span集合。每个证据集合对span取token union，
  pair IoU=交/并，F1=2交/(A+B)。空/空IoU与F1报null，不奖励“双方没找到”。
  A/B有多个替代充分集合时，对集合F1矩阵取最大权重一对一匹配（补空集合权重0）；
  以max(|setsA|,|setsB|)为分母平均。未匹配非空集合记0。span级也给相同匹配表，
  避免只挑最佳一对而忽略遗漏。必须同时报告无span题数和全题分母，不能只报双方有效。
- necessary facts：裁决者在不知道方法输出的情况下冻结原子事实对齐/同义规则；
  根据独立A/B原文事实条目计算集合precision/recall/F1。缺少可信对齐时标unknown，
  不用词面F1冒充语义事实一致性。empty/empty报null。
- 时间：每题阅读/定位时间与长context预读、培训、裁决时间分开记录；实测后才
  估算剩余40题。约6.7–13.3人时只是两人10题各20–40min情景，不含上述额外时间。

统计只作pilot描述：样本非IID、同context共用原文、独立context最多五，不做
10题/25seed或QQ pair的大样本显著性宣传。覆盖/一致性门待批准，不对缺失标签计算0%。

## 4. 裁决后才做来源映射（本轮不执行）

使用coordinator_only/chunk_offsets中的sentence_spans映射原文坐标，chunk来自
句子用空格拼接，chunk外包围[start,end)不等于逐字输入；注意句间空白不能当证据。
跨chunk证据保留全部多对多关系；一个充分集合的所有必要段应联合检查。

slot_sources只给各seed各bank**当前版本**的direct_source_chunk、indirect possible
来源与前版本依赖。直接来源相交→direct-source候选；只在间接链→indirect-possible；
不能确定内容关系→unknown。前版本已覆盖/淘汰时，不把其来源冒充当前slot直接来源。
既没有已标来源相交，也不自动等于semantic irrelevant：先审查间接链和证据集合
覆盖情况。无法可信定义negative则停止依赖它的E04，不制造不相关槽。

三层严格区分：Source Relevance / Latent Content Contribution / Answer Improvement。
本pilot最多建立来源层；后两层需要另行批准的等预算内容干预和配对答案实验。

## 5. 下一门

只有事前批准的覆盖/一致性门通过并有可用候选槽，才讨论50题确认标注或内容干预。
失败→保留未知/分歧并停止依赖mapping的语义结论；不调threshold掩盖缺口。
人员、预算或门槛未定→保持NOT_RELEASED，不开始标注，不进入E03或算法修改。
