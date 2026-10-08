# E03/E04离线失败分析与机制边界

2026-10-08；状态OFFLINE_FAILURE_AUDIT_COMPLETE，GPU新调用0。
父对象E04-X六题等预算替换与E03十题on/off，使用analysis-campaign固定证据/可比性/停止门。
exp_id=M6-E04-FAILURE；section_id=memory_content；item_id=E04-FAILURE；
claim_links=C02/C05/C11/R04；paper_role=reference_only。原始实验、算法、评分器均未改。
本次只核查已有记录，不生成答案，不修改数据集gold，不开展人工标注。

## 1. 最重要的修正：输出变化不是候选答案改变

E04 Original→Source有4/6题raw输出变化，但**六题最后一个非空行完全一致，
且raw全文唯一出现的候选编号也6/6一致**。四题变化均是候选正文前的额外前缀。
这收窄了此前“对内容敏感”的解释：只观察到Latent数值/槽身份替换影响输出前缀，
没有观察到这次替换改变所表达的候选选择，更没有证明语义事实敏感性。

|题目|Original→Source前缀变化|strict/valid转移|候选正文|
|---|---|---|---|
|c0-01|无 → `人名`|W→I|相同错误候选（邻居狗叫）|
|c0-02|无变化|W→W|相同错误候选（书籍遗忘）|
|c1-02|无 → `日志`|W→I|相同错误候选（调整眼镜）|
|c2-02|无 → `行`|C→I|相同正确候选（Elba可疑）|
|c3-01|无变化|W→W|相同错误候选（sneaky maneuver）|
|c4-02|`by /usr/bin/dash` → `多选题`|I→I|相同错误候选（visiting scientist）|

C=合法正确候选，W=合法错误候选，I=非法候选。此表不替代strict评分：
Original仍1/6、Source仍0/6，valid仍5/6→2/6；parser非空两组均6/6，
raw answer recall两组均1/6。无效答案不是parser没输出，而是解析内容不属于候选。

## 2. 唯一Strict回退题：c2-02 / context2 query45

Original `[0,1]`：

```text
Blaire mentioned that Pascal's stop at Elba was suspicious.
```

Source `[0,5]`：

```text
行
Blaire mentioned that Pascal's stop at Elba was suspicious.
```

官方`parse_output`首先尝试`Answer:`，否则返回第一行；本题没有`Answer:`，
因此Source parsed=`行`。strict与官方EM由1变0，但全文仍含完整gold，raw recall=1。
生成长度16→17 tokens，低于40-token上限，不支持“截断丢失答案”的解释。
这是这条记录中可直接观察的输出格式/解析原因，不是事实内容消失的证据。
**不删除前缀后重新发布更好EM，不更换parser，不将后验提取当新主指标。**

另外两条W→I也是同一错误候选前加前缀；c4-02只换无关前缀，仍I。
中文前缀的内部成因未知：没有保存逐层激活/再生成latent或做因果定位，
不能推断是压缩、Weaver训练、跨语言污染、某token维度或Norm直接导致。

## 3. 实现、来源版本与Tensor审计

完整E04 validator再次PASS；六题A重放与M3逐字一致，B两次逐字一致。
六题每次的pre-query完整16槽memory/key哈希、dtype、shape与旧M3快照记录一致。
固定位置0的memory哈希A/B一致，位置1的memory哈希六题均不同；实际返回索引与
固定计划一致，两槽各[8,1536]、conditioning16tokens、Weaver一次，无持久写入。
返回tensor的记录SHA与该指定槽的expected SHA一致；这些记录由运行时生成，
本次未再反序列化tensor。完整Bank文件SHA也受父协议检查，不能把日志自检等同
独立latent语义验证。未发现检查范围内的错槽/版本/顺序/预算/快照漂移错误。

M4 provenance与E02逐槽索引的当前direct来源、版本逐项一致：

|题目|位置1：A→B|B当前版本 / direct chunk|检索访问age A→B|Memory norm A→B|Key norm A→B|
|---|---|---|---|---|---|
|c0-01|12→1|slot1-v1 / 2|1→12|326.35→318.88|98.29→90.52|
|c0-02|12→6|slot6-v1 / 7|1→10|326.35→333.97|98.29→99.51|
|c1-02|1→13|slot13-v1 / 13|1→4|329.48→362.14|99.76→81.86|
|c2-02|1→5|slot5-v1 / 5|1→12|328.41→358.06|101.30→78.62|
|c3-01|1→11|slot11-v1 / 11|1→6|319.62→363.88|96.71→78.63|
|c4-02|1→3|slot3-v1 / 3|1→14|328.11→365.40|97.73→78.16|

age取原始native selection记录的pre-retrieval ages，**不是出生/写入时间年龄**。
返回slot的last_retrieved_step可能已被native选择更新，不用它倒算原始age。
Norm/age变化未匹配，不做六点相关回归，也不把变化宣布为失败原因。
c0-01/c2-02/c4-02的B分数分别约.04663/.03966/.03783，低于原threshold .05；
这是预先披露的forced-source干预，不是原检索意外漏过过滤，更不是新算法结果。

来源相关≠slot含事实；M4间接链不能当确认保留。Source候选是AI辅助来源标注，
正式E02独立人工门未通过，也没有可信semantic-negative。Random改变filler与来源，
旧Random没有在当前GPU重放，仍只作辅助；不能用它证明纯语义相关性差异。

## 4. E03与E04的共同解释：不能都归为事实利用，也不能都归为格式

E03全10题：on strict3/10、off0/10，valid7/10 vs0/10，parser非空都10/10。
全部strict转移为I→C 3、I→W 4、I→I 3；没有有效候选之间的正确性转移。
raw recall却是on3/10、off2/10，**并非完全不变**；不能说所有E03收益均来自前缀。

- c2-02：off=`简短`加完整gold；on只保留同一gold。支持格式/解析层差异。
- c3-02：off含完整gold但前加`[list of events]`及解释；on为合法gold。
  同样支持格式差异，但该题E02仍有人物别名/指代歧义，gold正确性未独立确认。
- c4-01：on为gold“观察孩子玩耍”，off抄已有事件“无法原谅丈夫”，
  raw gold从无到有，说明不止格式变化；但这题E02仍not_locatable，
  不能证明新增gold来自历史事实而非候选先验/表示变化。

E03十题有4题raw全文唯一候选编号相同（后验诊断），其余不等于“选错→选对”：
off可能没有候选，不能把无候选算成一个明确选择。完整十题原输出保留在分析产物。
E04六题是E02可映射子集，不能拿其1/6与E03全10题3/10直接比较或合并分母。
E03和E04共享题/Bank，不是两项独立证据；重复与seed不能增加独立context数。

## 5. 证据账本与下一路线

Observed：当前checkpoint/protocol下conditioning改变格式/候选有效性；
等预算槽替换在六题只改变前缀，没有改变候选正文；来源版本与tensor交付可复核。
Inferred：接口/表示适配值得优先检查，但还不是Weaver本身的唯一原因。
Hypothesized：Weaver改善Reasoner可用表示或答案格式，检索/构建也可能有瓶颈。
Unknown：历史事实保留、语义融合、Query可预测正负收益、construction根因。

建议**先规划E05同槽W1/D1接口对照**，不立即扩大E04或调整Retrieval/Gate。
E05应使用原冻结全部10题（不按本次成败筛题），同snapshot原生score最高槽，
不是canonical排序的slot0；W1该槽→Weaver→8新tokens，D1该槽→投影→8tokens。
先审计W1/D1同槽/位置/预算可实现性，再冻结预注册、重复策略、数值门与成本，
另获用户批准才运行。旧full2/direct1混杂对照不能替代它。E05只能解释接口差，
仍不能单独证明语义事实融合；不将已有Query-conditioned regeneration包装为新方法。
E03已经是旧消融的严格复核，不建议再扩同类开/关pilot来充当新贡献。

本次到离线审计完成为止，未制定已批准的新GPU campaign，不自动进入E05。
Retrieval/Regeneration均保留为冻结对照，尚无可靠依据批准算法改动。

## 6. 复现、资源与验收

分析脚本[analyze_failures.py](e04/analyze_failures.py)只用现有MABench Python、
官方parser及JSON，不加载Torch、tokenizer、权重、Bank tensor或GPU。
7项新增合成测试通过；完整E04 validator再次通过。两次分析输出核心SHA一致，
每轮76个审计输入前后SHA不变，约4.6秒/轮（含父文件hash核查）。
MEOI/一致性正式数值门没有按结果补设；无显著性/等价宣称。

[逐题E04记录](e04/failure-20261008-01/e04_cases.json)、
[逐题E03记录](e04/failure-20261008-01/e03_cases.json)、
[汇总](e04/failure-20261008-01/summary.json)、
[输入保全与验收](e04/failure-20261008-01/acceptance.json)。
最后一行/全文候选识别是明确后验的诊断口径，不进入M1 strict或官方指标，不用于
挑选更有利答案，也不证明真实推理选择。AI对前缀的解释不是独立人工语义标注。
