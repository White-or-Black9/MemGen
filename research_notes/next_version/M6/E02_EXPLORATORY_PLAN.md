# E02探索性裁决与来源映射：执行边界

2026-10-08，用户确认复核完成，并授权自行处理疑点、进行下一步。
父对象为E02冻结10题pilot、单份`annotations_codex_single_rater.jsonl`及M4来源链。
使用analysis-campaign的最小证据切片和停止门；本次为reference_only，
exp_id=M6-E02-SOURCE-EXPLORATORY；section_id=source_evidence；claim_links=C02/R03/R04/R07。

## 协议变更与真实性

用户希望将现有材料作为人工独立标注。现有文件明确披露为AI单标，无法更改其实际
生成来源。本次接受该文件为**用户已确认复核的AI辅助输入**；用户复核的具体逐题
记录/身份/耗时未提供。AI裁决不是第二位人类，也不是独立盲标，不计算一致率/kappa，
不声称完成人工双标。原始标注、准备包和release_gate全部保留原样。

原E02正式双标要求不因此追认通过：本次另立探索性旁路，不用于确认性语义检索结论。
未预先批准的覆盖/一致性数值门保持未批准，不按本次结果反设门槛。

## 本次唯一执行切片

1. 校验10题身份、候选范围、原文SHA及每个Unicode `[start,end)` span。
2. 单独保存AI裁决副本：c2-01不再凭跨章节“despair”认定下一事件充分；
   c1-01的先后冲突保留；c3-02保留人物别名/指代歧义。冲突证据不计答案充分覆盖。
3. 冻结裁决文件后再读取协调员索引；不使用gold改变裁决。
4. 用sentence_spans而非chunk外包围区间映射每段；多段联合充分集合不拆为独立证据。
5. 对supported且具有充分集合的题，列当前slot版本的direct-source关联及
   indirect-possible链；列固定两槽预算下能覆盖全部必要来源的组合。
   不相交只记no-recorded-source-link，不能命名为irrelevant/negative。

输出限定于M6/e02下的新脚本、测试、独立run目录与报告；同步M6板面。
一个CPU进程，标准库，GPU=0，无模型/推理/latent读取，RAM目标<=1GiB，
wall cap10min、输出<=20MiB。不修改模型代码、论文、M0–M5；不提交/Push。

## 验证与停止

合成测试覆盖Unicode坐标、身份/原文错配、句间空白、跨chunk、多段联合、间接链
与无来源关联。真实数据跑两次，核心hash必须一致；所有输入前后SHA一致。
技术通过仅表示坐标与来源关系可复算，语义裁决为AI判断，不是客观真值。
主observable为10题全分母的状态/候选槽可用性，按context分列，不做显著性推断；
相同context的5seed是来源复核，不是5次独立证据。

完成即停止。E02正式双标Gate仍未通过；E03 on/off与E04内容替换需要独立批准。
不能由来源关联推出latent事实保留、语义检索有效或construction丢失。
