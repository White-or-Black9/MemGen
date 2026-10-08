# M6-E01 — Retrieval离线诊断结果

2026-10-08；状态：执行完成、复跑一致。仅E01得到批准；E01V/E02/E03未启动。
遵循[结果前冻结契约](E01_EXECUTION.md)，不修改算法、不加载模型、不生成答案。
本阶段使用analysis-campaign的配对控制、证据边界与停止门；没有自动进入下一阶段。

## 1. 范围与可信性

- 分支review，HEAD `6784a26e65290e91be2ad8a965cf108cb19c7b22`；无提交或Push。
- 复用M3五seed×五context×100题×六策略，核验15,000条原始内容checksum、身份、
  snapshot、配对输入和selector；随机策略按原seed复算，不重新生成或重新评分。
- 每轮核验30,162份父JSON前后hash、25份bank快照及92份冻结源码hash；M3/M4
  inventory前后相同，M5旧validator再次PASS。未反序列化bank或载入任何模型权重。
- 两轮四份核心文件字节hash一致；[验收](e01/acceptance.json)、
  [首轮manifest](e01/run-20261008-01/manifest.json)、
  [复跑manifest](e01/run-20261008-02/manifest.json)保存身份、资源与完整hash索引。
- 新增16项测试、M4回归31项、M3/M1回归25项，共72项通过。
- 每轮约41秒，峰值RSS约296MiB，每轮产物约10.8MiB；GPU=0，模型运行=0，标注=0。

这些是离线分析验收，不是semantic retrieval或regeneration有效性验收。

## 2. 核心结果：A、B、C必须分开

### A：分数确实随Query变化 — Observed

25个bank各100种保存query hash和raw cosine数组；norm分布：均值0.596636、
总体标准差0.002777、范围0.589367–0.612806。norm集中不能证明方向坍塌或语义相同。
40,000个query–slot raw cosine范围0.048096–0.100098，均值0.073141，
中间90%区间0.057617–0.091309。低绝对cosine不能单独证明跨空间失效。

### B：完整排名会变，但Top-2集合不变 — Observed（此批设置）

固定同bank无自配Query置换，2,500个比较中：

|比较|完整排名改变|Top-2集合改变|平均Spearman / Kendall tau-b|
|---|---:|---:|---:|
|原Query vs donor，raw cosine|1,956（78.24%）|0|0.994980 / 0.980085|
|原Query vs donor，decayed score|923（36.92%）|0|0.997895 / 0.990317|

排名变化与高相关并存，不能把“发生变化”解释为明显改变检索结果。此处donor只是
机械置换，未验证问题证据是否独立，不能叫semantic-negative对照。

|Context|raw唯一完整排名/100题|final唯一完整排名/100题|raw Top-2集合|Full集合|Last-written集合|
|---|---:|---:|---|---|---|
|0|24|8|{4,12}|{0,12}|{14,15}|
|1|22|4|{0,1}|{0,1}|{2,15}|
|2|27|1|{0,1}|{0,1}|{2,15}|
|3|6|7|{0,1}|{0,1}|{2,15}|
|4|17|6|{0,1}|{0,1}|{14,15}|

上表模式在五seed均相同；slot ID只在各自bank有意义，不代表跨context相同内容。
不仅Full，**semantic-only也在每个bank固定一个Top-2集合**。因此不能把固定选槽
全部归因于decay；raw score本身的Top-2边界也未随这100题变化。

### C：是否选择相关历史事实 — Unknown

没有可信evidence-span/source-related标签，更没有latent事实保留真值。E01不能给出
semantic Recall/MRR，也不能证明历史事实被保留、Weaver利用了它们或算法完全无效。

## 3. Semantic与Recency贡献

与Full比较2,500组Top-2集合：Recency 2,500组一致，Cosine-only 2,000组一致，
Last-written 0组一致，Random 21组一致；交集槽位分别5,000/4,500/0/658。
Native-order集合也全部一致，但顺序并非总一致，不是内容选择对照。

同题raw→final平均Spearman0.355412、tau-b0.257563；16槽120对中平均43.182对
发生严格顺序反转。这描述decay对排名的影响，不是它改善语义相关性的证据。
raw top1–2 / top2–3 margin均值0.005274 / 0.008245，final为0.005605 / 0.015884；
final top2–3最小仍0.007389。边界稳定与选槽不变相符，不能据此设新成功阈值。

M3全题strict仍复用旧结果：Full=Recency20.20%，Cosine-only16.80%；净差仅ctx0。
这不足以分离语义、访问年龄及具体内容的独立收益。不重跑已存在的对照。
M5共同有效子集的反向结果和历史no-decay strict反向依然保留，不用E01推翻它们。

## 4. 新发现：Retrieval池化窗口没有问题线索

冻结checkpoint tokenizer仅作CPU编码，不使用transformers/Torch。500个不同rendered
prompt的编码长度均与保存input_len一致；结构offset归类为：

|最后64token组成|每题token数|占比|
|---|---:|---:|
|候选列表尾部|17|26.5625%|
|答案指令|41|64.0625%|
|Chat尾模板|5|7.8125%|
|候选列表/指令边界mixed|1|1.5625%|
|问题线索、候选标题、prefix、未映射|0|0%|

41个指令token与5个chat token在500题的各自token ID序列均完全相同。
mixed有两种ID，不能全部算固定模板。编码中pad token出现次数为0。
保存记录未包含完整实际input token IDs，因此长度一致不是逐ID输入一致的独立证明；
本结论基于冻结tokenizer、保存rendered prompt、长度一致和当前代码路径的可追溯重建。

代码核对：`modeling_memgen.py:505`输入embedding lookup；`:598`选candidate输入；
`:677`逐位置线性投影；`latent_memory_bank.py:368`最后64位置mean。该检索表示
不是经过LM attention的上下文化hidden state。因此在这条初始prompt路径中，前面的
问题线索没有通过上下文化“隐含进入”尾部embedding。**局部推论：本批retrieval q
直接依赖候选尾部与模板，而不是前面的问题线索**。候选本身也有语义信息，不能因此
声称q完全无语义。Weaver与Reasoner仍接收完整query，不能说整个模型没有看到问题。

这把原“模板可能影响Query”的假设收紧为可定位的输入窗口局限，但没有证明其
造成EM损失。只改变前部clue且保持最后64token IDs不变时，当前公式的q应不变；
数值复核及完整QQ几何需要另批E01V权重提取，不在本轮做。

## 5. 独立性与未回答的问题

2,500组包含seed重复，不是2,500独立题；500个不同prompt共享五context。
metadata.source五行均为`eventqa_65536`，是数据子集标签，不是作品身份；
qa_pair_ids清单hash也相同，不能用它证明五context来自不同书。独立context最多五，
其实际作品级独立性仍未知。本轮只描述，不作IID显著性或语义泛化推断。

完整q缺失，QQ cosine/逐维方差依旧blocked；同bank on/off/direct答案缺失，
内容敏感贡献及可预测conditioning benefit依旧unknown。没有新增EM结果。

## 6. 决策与停止

保留现有Retrieval/Regeneration作为冻结对照；**当前不修改算法、不实现Gate**。
不能把既有query-conditioned regeneration重新包装成创新。

建议优先另批E01V的CPU embedding+projection-only诊断：固定尾部换clue、固定clue
换候选尾部、QQ几何及数值精度检查；禁止LM生成。价值是定位Query表示，不是证明
记忆有用；资源需先只读核验checkpoint分片布局、CPU加载范围再定预算，禁止为
取embedding意外载入完整LM。E02十题双标pilot仍是内容机制检验的必要入口。
E03 on/off pilot仍有价值，但需独立GPU审批，不随本轮启动。

复核入口：

```bash
/home/baishilong/miniconda3/envs/MABench/bin/python research_notes/next_version/M6/e01/validate_e01.py
/home/baishilong/miniconda3/envs/memgen/bin/python -m unittest discover -s research_notes/next_version/M6/e01 -p 'test_*.py'
```

完成后停止；用户确认下一项范围前不加载权重、不标注、不跑inference、不提交Git。
