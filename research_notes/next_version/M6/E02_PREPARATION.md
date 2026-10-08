# M6-E02：10题人工双标准备契约

2026-10-08；批准范围为标注准备：冻结题目、去泄漏导出、规则、空白表与验收检查。
不实际人工标注、不由AI生成证据标签、不启动模型/GPU、不修改M0–M5或自动提交。
两位标注者/裁决者尚未确认；覆盖/一致性数值门未批准，不宣称E02机制验证通过。

## 目的与预先固定的设计

父对象M4-A冻结50条blind_sample及M6 E01/E01V；RQ=能否可信建立Question→原文
必要证据→来源chunk→当前slot版本的关联？来源关联不等于latent事实保留。
exp_id=M6-E02-PREP；item_id=E02；section_id=source_evidence；claim_links=C02/R03/R04/R07；
paper_role=reference_only/reproducibility；selected_outline_ref=未选定。

每context在原冻结10题中，继续按`sha256(M4-A-20261008|context|query)`排序取前2，
不依据gold、模型预测、得分、选槽或标注难度筛题。10题分别为：
c0 q16/q79；c1 q44/q87；c2 q19/q45；c3 q75/q3；c4 q80/q29。
初始rater包用中性sample ID，不给query_id、slot、策略、model输出、score或gold字段。
原始候选列表完整保留；候选中出现gold文本是正常题面，不是泄漏gold身份。

## 资源与实现边界

一个CPU进程、RAM<=4GiB、每次导出<=100MiB、wall cap10min；现有MABench Python+
pyarrow，不导入Torch/transformers，不联网安装，不反序列化bank。读取已冻结parquet、
M4-A核心与M3身份JSON；SHA核验后复制原文/问题及协调员来源索引，不重新chunk。
输出两轮新目录`e02/prepare-20261008-01`和`-02`，不覆盖已有文件。
本轮不制作浏览器UI，不计算基于真实标签的agreement或运行语义干预。

## 标注释放门

导出可完成；正式分发/开始双标仍需确认：

- 人员A、B、裁决者身份及独立性；裁决可由第三人承担，优先与A/B不同。
- 标注者既往接触model输出/机制结论的情况；已知暴露必须记录，不能声称完全盲法。
- 培训、每题时间和长文阅读预算；两人10题若各20–40分钟，仅任务情景约6.7–13.3
  人时，另加阅读五长context及裁决；不是已测成本。
- 可定位覆盖、status一致性、span/fact一致性、足够候选槽数量门，均需事前批准；
  没有数值时只能准备，不能按结果选择“通过”标准或自动扩到50题。

人员/预算/门槛在coordinator_only/release_gate.json中保持null；initial annotations
保持未填写。不能把生成空表、SHA通过或可打开原文当作标注可行性已通过。

## 验收与停止

两轮相对路径/核心文件hash一致；50→10子集、每context2题、原文/题面/候选SHA及
空标签检查通过；A/B初始包同内容且无gold/slot/score等字段；协调员材料分目录。
源材料前后hash/inventory不变，M5旧validator仍PASS。添加合成数据单元测试，不把
合成标签用于真实pilot。报告状态只能PREPARATION_COMPLETE/ANNOTATION_NOT_STARTED。
文件目录隔离不是权限隔离：只交付rater_X/子目录，不把M6/父输出整个目录给标注者。
完成后停止并等待人员/门槛确认，不自动进入标注、E03或算法改进。
