# M4-B2：输出有效性与正确性的配对分解

日期：2026-10-08。状态COMPLETE；两次全量分析一致，M4-B2完成。
父campaign为M3 `20261002-m3-six-policy-v3`，提交论文身份不变。
exp_id/item_id=M4-B2，section_id=evaluation_decomposition，paper_role=appendix supporting，
claim_links=R01/R02/R03/R05，display=配对状态转移表。

## 主要结果

相对随机与末写入基线，full的净正确增益主要与“无效候选输出↔正确候选”的
净转移同时出现；在两侧都产生有效候选的子集，优势很小或反向。
这支持“候选答案产出行为变化”的描述，不证明改善语义证据利用，
也不证明格式修复是收益的因果中介。

状态定义：C=有效且严格正确，W=有效但错误，I=无效候选。
矩阵方向固定为对照→full。正确净变化严格满足
ΔC=(I→C−C→I)+(W→C−C→W)。

|对照|净正确变化|I↔C净项|W↔C净项|严格正确率差|
|---|---:|---:|---:|---:|
|random|+407|+399|+8|+16.28个百分点|
|last_written|+460|+470|−10|+18.40个百分点|
|cosine_only|+85|+90|−5|+3.40个百分点|
|recency_only|0|0|0|0|
|native_order|+210|+205|+5|+8.40个百分点|

这是算术分解，不是因果贡献率；分解项可以超过净值或为负，不能强制归一成
0–100%的“格式/语义贡献”。所有转移表的行列边际、wins/losses/ties及准确率净值
与原M3正式结果完全一致。

## 两项主要对照的完整转移表

行是对照状态、列是full状态，每张表共2,500个seed/question配对。

|random→full|C|W|I|
|---|---:|---:|---:|
|C|59|8|31|
|W|16|213|41|
|I|430|1009|693|

|last_written→full|C|W|I|
|---|---:|---:|---:|
|C|25|10|10|
|W|0|20|15|
|I|480|1200|740|

大量I→W也伴随候选有效率增加，不能把“变成有效候选”本身视为答对。
full总计C505、W1230、I765；有效候选为1735/2500，但其中只有505/1735正确。

## 共同有效子集：保留反向结果与选择偏差

|对照|共同有效分母|full正确|对照正确|
|---|---:|---:|---:|
|random|296|75/296=25.34%|67/296=22.64%|
|last_written|55|25/55=45.45%|35/55=63.64%|
|cosine_only|1420|410/1420=28.87%|415/1420=29.23%|
|recency_only|1735|505/1735=29.11%|505/1735=29.11%|
|native_order|1060|295/1060=27.83%|290/1060=27.36%|

这些子集由两个策略的输出共同决定，且不同对照对应不同题目群体。
不能作为随机子样本、不能跨行直接比较方法强弱，也不能由last-written这55条
的反向结果宣称其总体更好。全题主指标仍沿用M3，不替换正式分母。

## “无效候选”不等于“parser失败”

重新调用冻结M1 scorer与官方parser，对15,000条逐字段一致性复核通过。
无效细分是附加诊断，不改变原candidate-valid定义：parser None/空优先，
其后是归一化歧义、解析文本包含多个候选、其余非候选。

|策略|无效总数|非候选|多候选文本|parser空|parser None|
|---|---:|---:|---:|---:|---:|
|full|765|760|5|0|0|
|random|2132|2129|2|1|0|
|last_written|2420|2415|5|0|0|
|cosine_only|1070|1065|5|0|0|
|recency_only|765|760|5|0|0|
|native_order|1430|1345|5|80|0|

多候选文本诊断仅在官方已解析文本中检查多个归一化候选的完整词面包含；
它不是新的语义判定，也没有将这类文本重新算作有效候选。
full有265条输出提及gold却无有效候选；random为510、last-written为505。
提及gold、官方substring EM、有效候选、严格正确四者保持分开，heuristic flag
不代替parser或正确率。不能把本表主要差异笼统写作“消除了语法/格式错误”。

## 与B1选槽诊断的联系

- full/recency的2,500组不仅正确率一致：原始字符串、parsed text、解析后的候选、
  候选有效性和strict正确性五个层级均全部一致。
- full/native的原始输出相同1560/2500，即有940条文本变化；在相同选槽集合下，
  顺序参考仍有明确的描述性输出差异，但本阶段不新增因果中介或显著性检验。
- full−cosine_only的+85全部集中在context0，其中I↔C净项+90、W↔C净项−5。
  full−native的+210集中在context0（+95）和context4（+115）。
- 主比较分context、seed及25个seed/context的完整转移均存入comparisons.json；
  500个唯一问题、五个独立context与2,500个seed/question重复测量不混为IID样本。

## 验证与产物

实现`audit_output.py`、测试`test_audit_output.py`。9项新增测试通过，M4总31项，
另25项M3调度/M1评分回归通过；编译及diff空白检查通过。
两侧有效性导致的选择偏差、配对错位、重复、净值守恒、空分母及gold提及但无效
均有测试。缺失记录由15,000唯一identity完整清单门拒绝。

父30,162份JSON与M4-A冻结hash清单一致，源码/输入/模型身份再次核验；逐条比较
scored_records与原query record，复算全部M1/官方字段及六策略summary，均一致。
父目录文件集合/size/mtime前后不变。不加载Torch/模型、不使用GPU、不下载依赖。

产物：`run-20261008-b2/`内states.jsonl、policy_summary.json、comparisons.json、
summary.json、examples.jsonl及manifest。examples仅在完整状态表冻结后，
按固定identity SHA每个非空转移桶取两条，保留桶分母，不人工挑成功案例。
复跑保存在`run-20261008-b2-repeat/`，不覆盖首轮。
两次states、comparisons、policy_summary、summary和examples五份核心产物的
hash完全一致；父主要输入hash、分析器hash一致。验收凭据`acceptance_b2.json`
为PASS；每轮产物约5.8MB，未超过资源预算。

## 下一路线

M4核心分析到此停止，不再叠加同类slice。下一阶段独立计划见`../M5_PLAN.md`，未执行。
建议下一阶段先整理提交论文的主张边界和review剩余缺口：当前支持更可靠的候选
答案产出及对输入/顺序的行为敏感性，不支持已证明的query-adaptive相关检索或
长程证据保留。然后由用户选择收紧论文定位，或另行批准可信span/latent验证及
必要的机制改进。现阶段不自动修改论文、decay/key表示、重训练或扩大benchmark。
