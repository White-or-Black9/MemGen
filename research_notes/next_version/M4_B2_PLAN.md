# M4-B2：已有输出的配对行为分解（已完成）

状态更新（2026-10-08）：用户批准并完成本阶段，两轮15,000条评分字段复算一致，
五份核心产物hash一致；9项新增、31项M4总测试及25项评分/调度回归通过。
验收报告`m4/M4_B2_REPORT.md`、凭据`m4/acceptance_b2.json`；下一`M5_PLAN.md`
主张与剩余缺口整合待批准。以下保留原预设协议，不自动开启论文修改或新实验。

父结果：M3正式评分、M4-A来源边界、`m4/M4_B1_REPORT.md`。
M4-B1显示查询表示变化但每bank固定选槽；full/recency集合相同，full/last-written
完全不重叠。下一问题是：相对对照的正确率收益，与候选有效性变化如何同时发生？

## 范围与执行顺序

仅CPU单进程，使用原15,000份回答、scored_records及M1评分定义，无模型/Torch/GPU。
首次预算30分钟、目标新增产物<1GiB；输入漂移即停止，不更改已接受评分或重生成答案。
首要比较full−random、full−last_written；辅助full−cosine_only、full−recency_only、
full−native_order。保持同seed/context/query配对，不将2,500条当独立样本。

1. 冻结输入hash，验证15,000唯一身份及已有评分字段；必要时调用已冻结M1纯离线
   scorer复算，逐条一致性失败则停止，不悄悄修订历史summary。
2. 每条输出按候选有效性与strict正确性建立互斥状态：有效且正确、有效但错误、
   无效候选。无效状态再区分parser None/空、非候选、歧义/多候选，保留原定义。
   gold在raw输出中出现与strict正确分开记录，不能用heuristic format flag替代parser。
3. 对五组比较给出完整3×3配对转移表、胜/负/平及各状态分母；区分invalid→correct、
   valid-wrong→correct及相反转移，并复核其净值与M3准确率差严格相等。
4. 分context、seed及seed/context呈现转移；共同有效子集单列两侧strict accuracy，
   明示这一子集由策略输出共同决定，存在选择偏差，不用于因果归因。
5. full/recency必须核验原始字符串、解析候选、有效性和正确性四个层级；不能仅因
   EM相同就断言文本相同。native-order单列，以相同选槽但不同顺序作为已有参考。

完成门：计数互斥且完备；转移表行列边际与各策略状态总数一致；净正确变化与
原M3主比较一致；复跑相同；全部输入未改；未知和失败保留。
测试覆盖空/None、多候选、错误但有效、gold提及却无效、双侧共同有效选择偏差、
配对错位/缺失/重复和转移净值守恒。若字段不足，先报告缺口，不凭预测文本猜评分。

输出：逐题状态、逐比较转移、context/seed摘要、manifest、B2报告及无模型测试。
定性例子仅在状态表冻结后按identity hash固定取样，给出抽样规则与分母，
不按最显著案例挑选，不新建模型/人工因果标签。

## 主张边界与停止

本阶段只描述策略改变时输出有效性与正确性的联合变化，不证明格式修复是
准确率收益的因果中介，也不证明latent保留了语义证据。
完成后制定下一路线：收紧论文机制措辞或提出独立可信source-span/latent验证方案；
不自动改论文、重设计检索、复跑no-decay、扩benchmark或提交Git。

论文映射：exp_id/item_id=M4-B2，section_id=evaluation_decomposition，claim_links=R01/R02/R03/R05，
paper_role=appendix supporting（主文可引用总结），display=配对状态转移表；
selected_outline_ref=EXPERIMENT_PLAN.md，RQ=output-validity-versus-correctness，
design=offline paired decomposition，method/comparators为M3六策略。
