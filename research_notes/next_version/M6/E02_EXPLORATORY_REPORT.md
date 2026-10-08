# E02探索性裁决与来源映射报告

2026-10-08；状态EXPLORATORY_SOURCE_MAPPING_ONLY；技术通过，正式人工双标门未通过。
计划见[E02_EXPLORATORY_PLAN](E02_EXPLORATORY_PLAN.md)，结果见
[summary](e02/exploratory-20261008-01/summary.json)及
[逐题逐槽映射](e02/exploratory-20261008-01/source_slot_mapping.jsonl)。
exp_id=M6-E02-SOURCE-EXPLORATORY；paper_role=reference_only；section_id=source_evidence；
item_id=E02；claim_links=C02/R03/R04/R07。

## 1. 身份与协议偏离

用户确认已复核并授权自行裁决、进入下一步。现有原始文件明确披露为AI单标，
故接受为“用户已口头确认复核的AI辅助输入”，不能改称人工独立标注。
未收到逐题人工复核记录、两位独立人类提交或第三人人工裁决。本次是AI追加裁决，
已接触项目背景，不宣称独立盲法，不算第二位人类。不计算一致率/kappa或人工耗时；
原elapsed_minutes保存为未核验字段，不把各题同填54分钟当实测。

原始标注、准备包及正式release_gate未改。本次探索性旁路不追认覆盖/一致性门通过，
不事后定阈值。6/10是本次AI裁决状态的描述，不是60%人工准确率或正式验收门。

## 2. 裁决与完整分母

|题目|本次状态|处理与来源chunk / 当前直接来源slot（索引从0起）|
|---|---|---|
|c0-01|supported（暂定）|chunk2 → slot1|
|c0-02|supported（暂定）|chunk7 → slot6|
|c1-01|conflict|亲属谈话早于奶酪作坊说明；仅支持时序冲突，不认定gold错误|
|c1-02|supported（暂定）|chunk13 → slot13|
|c2-01|ambiguous|去掉跨章节despair作为充分证据；补近邻失望/承认对话，候选4/5暂不可唯一判定|
|c2-02|supported（暂定）|chunk5 → slot5|
|c3-01|supported（暂定）|chunk11 → slot11|
|c3-02|ambiguous|昏倒者为母亲，但Mrs. Kayla/bare Kayla命名与题面有歧义，不确认人物错误|
|c4-01|not_locatable|保留未定位，不自动当不存在证据|
|c4-02|supported（暂定）|chunk3 → slot3|

分母固定全部10题：6 supported、1 conflict、2 ambiguous、1 not_locatable。
supported按context为2/2、1/2、1/2、1/2、1/2。冲突与歧义集合sufficiency改partial、
answer_support=false；未知保持unknown。其余6题沿用原充分性判断，技术验证不是
独立语义复核。原始18段、裁决后19段Unicode坐标均有效。
裁决副本先冻结SHA，再读取协调员索引；代码不以gold字段决定裁决。但同一助手
此前已见部分协调员信息，不能声称全程盲法。

## 3. 来源映射证明什么

85个chunk按sentence_spans和空格拼接重建SHA通过；25bank共400个当前slot的
直接来源SHA及potential=direct∪indirect通过。当前版本身份存入逐槽结果，
不把前版本来源冒充当前直接来源；句间空白不冒充输入证据。

6题的全部已标必要段各落在一个chunk，五seed均有一个当前版本的直接来源候选槽，
索引如上表。每题每seed有15种两槽组合包含它。第二槽只是预算补齐，不能保证
语义无关；15组合不是15次独立支持。这也不是原Full实际选择了这些组合：
本次**没有计算Full选槽的语义命中率**。

c0-01除一个直接来源槽还有14个indirect-possible槽；其余5个supported题除一个
直接来源槽，其余15槽均no-recorded-source-link。五seed关系一致。
**无来源链关联不等于语义负例**，间接链也不等于信息保留。目前没有已验证的
irrelevant槽，E04的Related-vs-Irrelevant控制仍未就绪。

Observed：坐标/来源身份与关联可复算，原文关联候选槽确实存在。
Inferred：可设计以这些槽为source-related候选的探索性干预。
Unknown：latent是否保留事实、Full是否利用事实、换槽是否提高答案正确率。
不能推广到全部500题；独立context至多5，五seed不是五次独立原文证据。
不计算置信区间或显著性，不凭来源层宣布算法有效、无效或construction丢失。

## 4. 验收与资源

- 13项新合成测试+13项准备测试=26项通过。
- 两次exploratory-20261008-01/02的三份核心SHA完全一致；新validator逐项重放裁决与映射。
- 每轮28个输入前后SHA相同；冻结准备包22文件SHA通过；M5 validator仍PASS。
- 单CPU标准库，无模型、Torch、tokenizer或bank反序列化，GPU=0。
  每轮约0.14秒，峰值RSS约22.5MiB，输出236,543字节；不是人工阅读耗时。
- 原准备validator仍只验收准备包，新validator仅授予探索性技术通过。

```bash
/home/baishilong/miniconda3/envs/MABench/bin/python -m unittest discover -s research_notes/next_version/M6/e02 -p 'test_*.py'
/home/baishilong/miniconda3/envs/MABench/bin/python research_notes/next_version/M6/e02/validate_exploratory.py
```

核心SHA：裁决`e0a2a349ff6addbbfd07bffdc57a5a48ca0ffcb3e857566ae277f62a4b6ec6db`；
映射`ff7729f7d55f6c0deea6f6d0000e34bb93477eba7e2fd076bfef0045fe50ef53`；
summary `f42aa6cd256b9d178c9b6acdd3e2849bab0b70e730e06926e986ddf09cd60722`。

## 5. 下一路线与停止

本次离线切片完成并停止。建议下一项另批E03原设计：固定原10题，同bank/checkpoint
conditioning on/off，各重复一次，共40次GPU生成；保留全部题strict评价，不只取6题。
它不依赖来源标注真值，直接检验conditioning正确率/格式变化及正负收益是否并存。
GPU时间需资源检查和pilot实测，本次未启动、不自动授权。

若要正式语义检索/论文人工证据主张，仍需真实独立人工标注和适当验收，AI裁决不能补足。
E04暂缓，先解决可信negative与来源≠内容的对照。Retrieval/Regeneration保持冻结，
尚无依据修改算法或实现新Gate。
