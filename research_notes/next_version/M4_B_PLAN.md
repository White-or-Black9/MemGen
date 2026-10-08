# M4-B：已有trace与输出行为诊断（B1/B2已完成）

最新状态（2026-10-08）：B2也已完成，见`m4/M4_B2_REPORT.md`。
M4-A/B1/B2均经独立复跑验收；下一阶段`M5_PLAN.md`待批准，不继续叠加同类分析。

状态更新（2026-10-08）：用户批准并完成B1；两轮全量审计核心产物hash一致，
10项新增测试、22项M4总测试及25项评分/调度回归测试通过。报告
`m4/M4_B1_REPORT.md`，验收`m4/acceptance_b1.json`。
25个bank均固定选两槽，cosine随题变化；full/recency集合2500/2500相同，
full/last-written槽交集为0。下一独立计划`M4_B2_PLAN.md`，尚未执行。
以下保留B1预设规格，不意味着B2获准自动执行。

父结果：`m4/M4_A_REPORT.md`、M3正式15,000条回答；review R01/R02/R03/R04/R07。
M4-A已确定直接输入来源可重建，但500题缺经核验原文span，latent保留未知。
因此本阶段不自动计算语义Recall@K/MRR，不补建bank，不生成新答案。

## 先执行的唯一子阶段：B1 选槽时间机制诊断

问题：full/recency为何得到相同集合？“recency”实际来自最近写入，还是历史被反复访问？
只读M3全量trace、M4-A版本来源、冻结公式/配置与construction日志。
使用CPU单进程、无Torch/模型、30分钟预算、目标新增<1GiB；父身份/hash漂移即停止。

1. 校验2,500个seed/question中策略共享的cosine、age、last-write等基础量一致。
   分别定义write-age与last-access-age，不能用后者替代证据年龄。
2. 每seed/context报告full查询间选择集合数、单槽入选频率、与recency/last-written/
   cosine-only的重合率；n=0/1/2分别给数量及分母，不能把空集合相等当成功检索。
3. 复算raw cosine、exp(-alpha*age)、final score、阈值门与top-k，逐条与保存结果
   比对；处理稳定index tie-break，区分阈值筛选和排名作用。
4. 按25个job报告score范围、排名一致性、选择槽的当前写入chunk和历史访问计数。
   来源chunk仅是构建输入来源，不命名为gold evidence年龄。
5. 仅在保存的实数精度允许时比较跨query cosine/排名变化；浮点重算误差要设
   明确容差并测试。不把观察到的等价推断为所有问题/超参数下公式严格等价。

输出：逐job摘要、逐题可复算diagnostic、错误/未知清单、版本化manifest、B1报告。
测试覆盖age定义、n=0、ties、负cosine、阈值边界、排名与canonical输出顺序分离、
缺失/重复记录和父数据不变。保留500唯一问题、五context、五seed三个层级，
不对2,500条重复问答做IID显著性测试。

验收后停止并制定B2；无论结果如何，不直接修改decay或key表示。
如full/recency只在有限样本上一致，报告样本边界；若观察到长期反复访问的早期槽
支配选择，报告机制一致性证据，但不单凭关联声称存在有害反馈的因果关系。

## B2 后续框架，B1验收后才细化

基于既有scored_records做full与各对照的候选有效性×strict正确性配对转移；
区分格式/候选无效与有效但选错，保留共同有效条件子集的选择偏差。
不声称格式是因果中介。只有另行获得可信source span/语义标签后才另规划相关性指标。

论文映射：exp_id/item_id=M4-B1，section_id=retrieval_diagnostics，claim_links=R03/R04/R07，
paper_role=main_text，evidence_role=claim-carrying（限制机制主张），display=机制边界表。
selected_outline_ref=EXPERIMENT_PLAN.md（修订路线）；RQ-M4；design=offline trace reconstruction。
当前method/comparators沿用M3六策略，评价合同不改；不以新分析替换历史P7结果。
不可执行自动分支：GPU复跑、新benchmark、oracle生成、模型/训练/生产源码或论文修改。
