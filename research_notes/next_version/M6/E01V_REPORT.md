# M6-E01V — Query表示机制验证结果

2026-10-08；状态：CPU诊断完成，两个验收运行核心hash一致。
用户批准范围仅E01V，无GPU、模型生成、标注、算法修改或Git提交。
父对象M3/M4与[E01](E01_REPORT.md)；[结果前契约](E01V_EXECUTION.md)固定对照与容差。
analysis-campaign技能用于固定变量、数值复现门、失败保留及claim边界，不改变模型。

## 1. 最重要的结论

在本批500个初始prompt及当前embedding→projection→last64 mean路径中：

- **只替换前部问题线索且保持尾64 IDs不变，CPU重建q全部完全不变。**
  25bank×100题的分数、完整排名、Full Top-2也全部不变。
- **只替换候选尾部17个token，500个q全部变化。** 原始/最终完整排名分别在
  1,905/925组变化，但Full Top-2仍在全部2,500组保持不变。
- 同context不同题的q方向高度相似，平均QQ cosine约0.996762；不是相同向量，
  中心化有效秩约21–36。不能把“高度相似”写成“所有表示完全坍塌”。

这支持更窄而明确的机制结论：**本设置的retrieval q对前部问题线索不敏感，对
候选尾部敏感；这些变化尚未转化为不同的Full选槽。** 不代表Weaver/Reasoner
看不到完整问题，也没有证明这是EM损失原因、Memory没有信息或整个方法无效。

## 2. 实现与可比性

只读mmap base safetensors的BF16 embedding tensor，复制6,413个需要的token行；
读取projs.bin，只使用已有reasoner_to_weaver weight/bias。未读取Transformer tensor、
未实例化MemGen/Reasoner/Weaver/Trigger。可信bank hash通过后，weights_only白名单
载入数据，用原16个slot keys；不恢复RNG或调用bank写入。

从M3 environment_contract选择实际base tokenizer；500题编码与checkpoint tokenizer
逐ID相同，长度与原input_len一致，尾64 hashes与E01一致。batch1完整序列进行
BF16 linear（含bias），随后取最后64位置BF16 mean，不使用pool-first替代原路径。

对照固定：权重、题目索引、bank snapshot、slot顺序、alpha=.05、threshold=.05、
Top-K=2。没有gold/正确性输入，没有生成答案，不更新已有EM。

|Slice|改变的主要变量|保持不变|可比性限制|
|---|---|---|---|
|V1 原始几何|只观察CPU重建原q|原prompt和权重|部分q不是原GPU bitwise值，见复现门|
|V2 clue-only|换成同context固定donor的问题线索字符段|候选、模板、尾64 IDs|只检验检索表示，不检验答案对clue敏感性|
|V3 candidate-tail-only|17个候选尾位置的token IDs|前部clue、候选更早部分、mixed边界、模板、长度|token机械反事实，不保证自然文本或语义负例|
|Precision FP32|BF16权重转FP32的线性/池化及cosine计算|原prompt、参数数值、bank、公式|数值对照，不是训练新参数或原GPU结果|
|Precision pool-first|BF16先mean再linear|原prompt、参数、bank|精确算术等价不代表BF16逐位等价；未部署|

500题均实际改变clue前缀；候选对照每题改变7–17个位置（有些donor token相同），
没有挑选能改变结果的donor。规则在结果前固定，源于同context SHA循环置换。

## 3. 原M3数值复现门

|核对项|结果|
|---|---:|
|预设norm / raw cosine绝对容差|0.001 / 0.001|
|norm最大绝对误差|0.0000323653|
|raw cosine最大绝对误差|0.00048828125|
|q hash逐位一致|670 / 2,500记录|
|Full Top-2集合完全一致|2,500 / 2,500|
|raw完整排名完全一致|2,480 / 2,500|
|final完整排名完全一致|2,490 / 2,500|
|slot级threshold通过状态差异|5 / 40,000|

预设标量/Full集合门均通过；**不能称全部原GPU向量逐位复现**。五次threshold差异
来自同一ctx0/q16在五seed的重复，没有改变Top-2；不以容差掩盖临界差异。
V2的完全相同是在CPU原q与CPU干预q之间确认；原GPU路径的机制解释同时依赖
已审计的逐位置embedding/linear/mean实现，不冒充重跑GPU反事实。

## 4. 干预结果（2,500组含seed重复）

|相对原CPU路径|q完全相同|raw排名改变|final排名改变|Full集合改变|slot级阈值状态改变|
|---|---:|---:|---:|---:|---:|
|换clue|2,500|0|0|0|0|
|换候选尾17token|0|1,905|925|0|1,355|
|FP32数值对照|0|2,010|430|0|125|
|BF16 pool-first对照|0|585|165|0|30|

候选干预平均q cosine0.996813（最小0.988036），平均L2差0.047449；不是完全无响应。
完整排名对数值路径也敏感，不能仅凭“排名随Query变化”宣称语义检索成功。
阈值状态变化是slot×query计数，不是改变答案/Query数，且不是新Gate实验。
所有对照Full集合未变，仅适用于当前500题、五context及该匹配/预算，不推广任意Query。

## 5. 表示几何

每context100个唯一CPU q；双精度计算geometry，取QQ上三角去对角4,950对。
中心化协方差有效秩使用participation ratio `(sum lambda)^2 / sum(lambda^2)`，
逐维方差用总体分母N，完整矩阵/方差/特征值保存于queries.npz和geometry.json。

|Context|平均QQ cosine|最小QQ cosine|方差能量/平均平方norm|中心化有效秩|
|---|---:|---:|---:|---:|
|0|0.996805|0.994137|0.3176%|33.77|
|1|0.996801|0.986456|0.3205%|21.16|
|2|0.996755|0.992756|0.3231%|31.73|
|3|0.996924|0.993963|0.3061%|35.57|
|4|0.996527|0.992522|0.3456%|24.71|

共同分量占绝大部分能量，但这个比例**不是模板因果贡献率**：共同分量也可能包含
projection bias或共享候选词汇。没有模板/bias的独立消融，不能定量归因于某一来源。
几何相似不等于语义相关性差的直接证明；候选本身也包含事件语义。

500题与24,750个QQ配对不彼此独立；五seed不制造新q样本。context最多五、作品级
独立性未知。因此本轮不做IID置信区间、显著性或跨任务泛化成功声明。

## 6. 验收、成本与失败保留

首轮01在JSON序列化时失败，保留status=FAILED及未验收产物；修复及路径调整见
[执行补记](E01V_EXECUTION_ADDENDUM.md)。没有放宽原容差或改题目/干预设计。

验收轮02/03的五个核心文件byte hashes完全相同。每轮52.5–54.4秒，峰值RSS
约3.2–3.5GiB，产物约35.2MiB；都低于预设4线程/16GiB/30min/100MiB上限。
含保留的首轮失败文件，整个e01v目录当前约102MiB；100MiB门是每轮输出上限，
并非整个目录总上限。失败尝试的额外计算耗时没有完整manifest，不伪造总耗时。
Torch CUDA未初始化、模型forward=0、GPU=0、人工标注=0。新增13项测试、E01回归
16项、M4回归31项、M3/M1回归25项，共85项通过；M5 validator仍PASS。
输入/源码hash前后相同，父M3和E01 inventory未变；最终只读validator另核对M4核心
与30,162份父JSON。历史M0–M5、论文、模型/训练/推理算法未改动，无commit/push。

[验收收据](e01v/acceptance.json)；[首个验收manifest](e01v/run-20261008-02/manifest.json)；
[复跑manifest](e01v/run-20261008-03/manifest.json)。只读复核：

```bash
/home/baishilong/miniconda3/envs/MABench/bin/python research_notes/next_version/M6/e01v/validate_e01v.py
```

## 7. Claim更新与下一步（本轮停止）

- Observed：CPU clue-only不变；candidate-tail变；高QQ相似度；Full集合稳定；
  数值实现影响部分完整排名/阈值，但本批未改变Top-2。
- Inferred：该初始prompt检索路径未直接利用前部问题线索，候选尾部是响应来源之一。
- Hypothesized：这导致选错记忆或EM损失，改进表示会提高效果；均未验证。
- Unknown：slot语义相关性、latent事实保留、当前Weaver内容利用/on-off收益。

现有Query表示已暴露值得研究的局限，但还不能选择“Retrieval是主要效果瓶颈”或
断言应改Construction/Regeneration。保留冻结算法，不调阈值、不实现Selective Gate。

推荐下一项单独批准E02十题双人证据标注pilot，先确认source→slot映射能否可信建立。
价值：为相关/不相关记忆对照提供标签；GPU0，但需两位标注者和裁决，人工成本须
先确认，失败则停止依赖映射的后续语义检验。不得由模型自动补标签冒充人工双标。
E03当前on/off GPU pilot仍可作为另行批准的收益验证，不随本轮启动。
改Query的算法试验必须另立公平对照与资源计划，不能把这次CPU几何诊断当EM证据。
