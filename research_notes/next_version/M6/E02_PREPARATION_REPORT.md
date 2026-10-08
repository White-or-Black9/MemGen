# E02标注准备交付（不是人工标注结果）

2026-10-08；状态：**PREPARATION_COMPLETE / ANNOTATION_NOT_STARTED / NOT_RELEASED**。
分支review；HEAD `6784a26e65290e91be2ad8a965cf108cb19c7b22`；未提交/Push。
用户批准后只准备10题pilot，不填写人工证据/相关性标签，不启动GPU/模型生成。

## 已完成

1. 从M4-A冻结50题中，按原SHA排序每context取前2题：c0 q16/q79、c1 q44/q87、
   c2 q19/q45、c3 q75/q3、c4 q80/q29；没有按模型表现或可标注程度挑题。
2. 两个独立包，各含10题、完整五context、标注说明和10份空白phase1记录。
   使用中性sample_id；不提供gold身份、模型输出、score、query_id或slot信息。
3. 协调员目录单独保存dataset gold/原题索引、句子级chunk offsets、400个当前slot
   版本的来源索引，以及空白gold复核/裁决表；这些不是已建立的Question→Slot标签。
4. [标注规则](e02/RATER_GUIDE.md)明确Unicode偏移、多段充分集合、歧义/未定位状态；
   [协调员说明](e02/COORDINATOR_GUIDE.md)固定两阶段盲法、独立冻结、裁决、
   status混淆/kappa、span匹配与事实对齐评价协议，以及人工标注后才进行的来源映射。
5. [释放门](e02/prepare-20261008-01/coordinator_only/release_gate.json)人员、门槛和
   审批记录均未填写；没有把空表或程序验收当作可行性成功。

## 可使用的材料

- [标注者A说明](e02/prepare-20261008-01/rater_A/README.md)
- [标注者A空表](e02/prepare-20261008-01/rater_A/annotations_blank.jsonl)
- [标注者B说明](e02/prepare-20261008-01/rater_B/README.md)
- [标注者B空表](e02/prepare-20261008-01/rater_B/annotations_blank.jsonl)
- [协调员操作说明](e02/COORDINATOR_GUIDE.md)
- [结果前准备契约](E02_PREPARATION.md)、[准备验收收据](e02/acceptance.json)

只能将rater_X子目录复制给相应人员，不要分发M6/父输出/协调员目录。现有目录分隔
不是文件访问权限保护，尚未真正分发给任何人。保留空表原件，用工作副本填写。

## 核验与成本

13项新增测试通过；两次导出的22份包文件逐字hash一致，A/B初始内容对称；样本
均在原冻结50题中，原文/题目/候选身份一致，全部20个phase1表项保持未标注。
来源chunk按原句子offset与空格拼接重建hash核验，未重新chunk。M3/M4/E01/E01V
inventory前后未变；M4-A核心hash与数据集SHA核验通过，M5旧validator仍PASS。
没有重复计算M3指标、没有反序列化bank或加载权重，Torch/transformers未导入。

每次导出约4.1–4.2秒、RSS约307–312MiB、输出约3.18MiB。GPU/推理/人工标注均0。
actual annotation cost未知：两人10题各20–40分钟仅约6.7–13.3人时情景，另加五篇
长context阅读、培训和裁决；须先确认人员和预算，不能当作测得耗时。

复核命令：

```bash
/home/baishilong/miniconda3/envs/MABench/bin/python research_notes/next_version/M6/e02/validate_e02.py
/home/baishilong/miniconda3/envs/MABench/bin/python -m unittest discover -s research_notes/next_version/M6/e02 -p 'test_*.py'
```

## 证明边界与下一步

本轮仅证明**标注材料可追溯、导出可复现、空标签与隔离设计经过程序检查**。
没有证明人能够定位证据、标注者一致、Memory包含事实、检索相关或regeneration有用。
此前E01/E01V发现保留不变，不因为本次导出增加正面机制结论。

继续需要用户确认A/B及裁决人员、既往暴露、培训/时间预算、覆盖与一致性等事前门。
确认后才开展独立人工标注；没有人时不能由AI代填冒充双标。本轮使用
analysis-campaign的固定样本、证据隔离与停止门，避免准备工作被误写成实验结果。
完成后停止，不自动扩50题、不进行E03/E04或算法修改。
