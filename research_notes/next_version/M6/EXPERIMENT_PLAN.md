# M6 — 最小可证伪机制验证设计

**DESIGNED / NOT EXECUTED**。本轮只有代码/旧数据检查与文档；本文件不是执行授权。
父对象=M3六策略及M4审计；协议锚点见[实现审计](IMPLEMENTATION_AUDIT.md)。
selected_outline_ref=none（尚未选新论文outline）；当前为pre-outline method decision。
所有数值成功阈值、执行预算、标注者、数据split须事先批准，不由实验结果反推。

## 0. 资源与统计共同契约

本轮无GPU调用/新campaign/标注/训练/算法改动。只读主机资源采样：755GiB RAM，
约631GiB available，磁盘约3.2TiB available；这是瞬时资源，不是后续预留。
MABench环境可做JSON/数据核查但无Torch；未来权重提取使用已有memgen环境，
不在本阶段安装依赖。未查询或承诺GPU空闲；未来启动前重新检查，沿用同型号
RTX A6000且>=12GiB空闲/每卡一个worker/独立输出和tmux的资源门，资源不达标不启动。

数据契约：固定bank、checkpoint、tokenizer、实际rendered prompt与40-token解码设置；
查询无实际写入、无跨题访问状态积累；完整恢复Python/NumPy/Torch/CUDA RNG。
当前CUDA topology会校验，未来不能直接跨不同可见卡数恢复而假称等价。
每个新arm只发布一次正式答案；中断/重复须记attempt，禁止选最佳。原始文件只读，
新结果必须独立目录并记录父hash/实验协议。将来诊断adapter/hook须另批，实现不能
改M3冻结源码或生产算法；需要不能由已有接口完成的hook时，先提交实现计划。

### 指标字典（所有生成实验E03/E04/E05/E06共用，不能换分母）

- Primary：Strict EM = M1严格候选正确率，归一化后唯一匹配合法候选且正确；
  **全预注册题目**为分母。与官方substring EM分开并同时保留官方分数。
- Secondary：官方raw answer recall；parser nonempty rate；candidate-valid rate。
  “Format Success”必须拆成parser非None/非空和candidate有效率，旧heuristic flags另列，
  不得用`500−flags`构造parsed分母。
- Parsed-only accuracy：每arm官方EM/parsed-nonempty与strict/parsed-nonempty分别报，
  同报分子分母；两arm共同parsed/共同valid子集仅描述，注明输出选择偏差。
- correctness 2×2：on对/off错、on错/off对、都对、都错；reported improved/regressed/
  unchanged指strict变化，unchanged再分都对/都错，不等于文本未变。
- parser成功2×2、candidate有效2×2、raw-recall 2×2；另C/W/I完整3×3：
  C=valid正确、W=valid错误、I=invalid。显式报I→有效、有效→I、两侧有效的正确性
  变化、两侧无效；每一类保留分母、context和seed，不只挑共同有效中的正结果。
- source标签相关/不相关条件差与正确率差分开；不把候选来源相关称为latent语义真值。

### 独立性、置信区间与预注册

先按context等权计算配对效果；seed在context内平均，是重复测量而非新增context。
五个context是当前最多五个独立cluster的工作假设；先核查原始source是否共享小说/
重叠段落，若共享则按更高层source合并或做source-block敏感性，不能机械认定五个
context完全独立。以下五cluster区间/检验仅在该独立性假设成立时适用。
50题试验固定每context10题/seed42，仅用于机制可行性，不估计500题总体主结果。
确认阶段若另批5seed，则复用同50题，报告每题跨seed方向稳定/不稳定；不能扩大IID分母。
报告5个context全部差值、均值、范围、leave-one-context-out敏感性。
可用10,000次context-block配对bootstrap给探索性95%CI（保留block内所有seed/question），
但只有5个cluster，区间/覆盖率不稳，不以其不跨零自动宣称显著。
若设计允许独立context符号交换，5cluster双侧精确检验最小p=2/32=.0625；
相关题级McNemar不作为主要显著性检验。不得用25个seed/context块当25独立context。
多探索指标不挑最优p；指定一项primary contrast，其余exploratory。

每项的Minimum Effect of Interest记为delta_*，**待执行前审批**。
正向门需效果超过已批准delta、方向跨context有支撑、不是单context驱动、没有协议违约；
低于门不支持主张；CI宽或标签覆盖差时为inconclusive，不能称无效或等价。
等价结论需事前等价界与足够效力，当前五context不默认具备。
不因观察结果更换primary、校准阈值、重抽样或增加seed直到“通过”。

## E01 — 已有分数/排序/Query置换离线诊断（第一优先）

执行更新：2026-10-08用户独立批准E01，已实现并双跑验收；结果见
[E01_REPORT](E01_REPORT.md)、[执行契约](E01_EXECUTION.md)。以下设计保留为结果前
协议；原“尚未实现/运行”描述已失效。每轮实际约41秒、峰值296MiB、10.8MiB产物，
GPU0。E01V/E02–E06仍未执行，不因本次授权自动批准。

Experiment ID=M6-E01；RQ=Query如何影响分数、排名和最终选择？
Hypothesis：A分数敏感、B全排名敏感可以成立而full Top-2保持固定；C语义相关不能
仅凭A/B成立。Existing Evidence：M4-B1固定集合及变化的cosine/score；norm已保存。

Controls：Original Full、Semantic-only（cosine_only）、Recency-only、Random-2、
Last-written-2，直接复用M3/M4；native_order只作顺序参考，不混成内容selector。
Variables：原始q分数行与同bank donor-q分数行；保持bank、slot版本、ages、
阈值/count、recipient identity不变。本实验不生成答案，不改真实bank访问状态。

设计：

1. 按bank/seed/context统计query norm分布；hash唯一数只表明非同字节，不当区分度。
   统计cosine min/max/分位数/每query范围，top1−top2和top2−top3 margin，raw/final
   排名Kendall tau-b（显式ties）/Spearman（midrank）、ranking变化频率及full Top-2频率。
   bf16 ties保留；不能给存储精度以下差异赋予语义解释。
2. 按策略slot选择频率/组合数、Jaccard/交集数、原生顺序和canonical集合分别报告。
   semantic与recency排序贡献用固定ages下raw→final的rank翻转及Top-2边界变化，
   不对带负值cosine使用log分解，不强行算“语义贡献百分比”。
3. Query permutation：固定同一个seed/context bank，把该bank另一题已存cosine行
   用于recipient的retrieval计算；不能拿别的context的cosine行（key不同）。
   每题选一个固定SHA规则的无自配donor，先报告全题机械敏感性；若同bank每题full
   集合相同，则行置换不会改变full集合，这是由旧记录导出的预期，不是新模型结果。
4. E02之后增加两种donor：证据段确认不共享且长度/候选数量/实体等尽量匹配的
   hard-negative；共享证据的positive control。排除近重复query；没有可靠不共享标签
   就不把普通置换称为语义负例。若匹配失败，保留mechanical部分，语义部分N/A。
5. 事后标签可计算source-associated candidate slot的hit@2、coverage/MRR，必须命名
   为来源关联指标，unknown不填false。full/semantic/recency/置换使用同标注分母。
   只有source关联、内容敏感干预与下游收益共同支持，才讨论semantic独立贡献。
6. 独立tokenizer-only检查实际rendered prompt最后64tokens组成：问题线索、候选、
   公共指令/角色模板、padding/裁剪比例；不运行LM，不自动改变窗口或重排候选。

Required Data：已有B1 diagnostics、query_score_decomposition、runtime rendered prompt、
bank config；第4/5项需要E02标签。QQ cosine与逐维方差不在已有数据可回答范围。
Metrics：上述分布/排名/组合/来源指标，不能用EM替代检索相关性。
Statistical Protocol：context内分布及五context分列；置换固定规则/次数，统计关联的
delta_source需批准；不是在本阶段搜索阈值。无人工标签时不做C命题检验。
Cost：GPU0，全部2,500 full trace+既有B1文件；建议1 CPU进程、RAM<=4GiB、输出<=100MiB，
wall cap30min为拟批准预算，不是已测时长。E01程序尚未实现或运行。
tokenizer-only窗口检查可用本地tokenizers/已有环境；依赖不可用时只完成JSON部分并
记录窗口检查blocked，不安装依赖、不偷偷加载完整模型。
Success Criteria：A/B与selection层级清楚、匹配置换可复核、完整覆盖/unknown清楚；
只有source contrast达批准门才能支持来源相关，仍不等于latent内容有效。
Failure Criteria：排序仅年龄驱动或标签结果不支持来源关联；必须保留，不调参数补正。
Stop Conditions：hash/配对不一致、缺关键分数、无法构造有效donor即停相应子分析。
Expected Value：零新推理判断瓶颈在表示/排序/选择的哪层，避免重复六策略实验。
paper_role=main limitation/diagnostics；section_id=retrieval；item_id=E01；
claim_links=C02/R03/R04/R07；display=score/rank/selection分层表。

## E01V — 完整Query几何（可选，不能冒充已有数据）

执行更新：用户独立批准后已完成，见[E01V_REPORT](E01V_REPORT.md)与
[冻结契约](E01V_EXECUTION.md)。原始几何、clue-only、candidate-tail-only及精度
对照均运行；预设近似数值/Full选槽复现门通过，但并非全部原GPU q bitwise一致。
两个验收运行52.5–54.4秒、峰值3.2–3.5GiB、每轮35.2MiB，GPU0；技术失败保留。
下面为结果前设计，实际500个唯一prompt并明确保存2500记录索引/seed重复。
E02及后续未启动，仍须单独批准。

RQ/Hypothesis：不同query向量是否近乎共线，方差是否由模板尾部主导？未知。
Existing Evidence：norm/hash有值，跨query cosine全null，无完整q。
Controls/Variables：同一已冻结prompt的原始表示，按context比较；不更换query算法。
Required Data：实际token序列、投影参数、Reasoner input embedding；只读可信本地权重。
优先embedding+linear-only CPU提取原q，不需要Weaver/Reasoner transformer forward；
必须复现batch1、最后64窗口、dtype、bias与裁剪；用已存norm/cosine核对数值误差。
CPU bf16与原GPU舍入不能保证bitwise；先指定容差，不能用近似结果替代原临界排序。
Metrics：QQ cosine矩阵、每维/总方差、中心化协方差有效秩、norm、同/跨问题区分度；
hash不同不充当QQ cosine。完整q存[2500,1536]，fp32约14.65MiB。
Stats：context分列，delta_geometry审批；几何差异不证明语义相关。
Cost：GPU默认0；CPU/RAM建议<=16GiB、数据<=100MiB；embedding权重加载I/O与
tokenizer版本兼容需预先确认，无法可靠估wall time。不能直接沿用会强制CUDA的模型loader。
Success：提取与旧标量核对达容差、几何可解释；Failure：未复现输入/误差跨排序边界。
Stop：需完整生成/修改核心路径时另报批准；Expected Value：补表示信息缺口，非主要效果实验。
paper_role=appendix；claim_links=C02/R04；不作为E03的强制前提。

## E02 — Source-to-Slot标注可行性（第二优先；本轮不标注）

准备更新：用户批准后已完成E02-PREP去泄漏标注包导出、规则、空表与验收，见
[准备报告](E02_PREPARATION_REPORT.md)和[准备契约](E02_PREPARATION.md)。10题按
原冻结SHA规则选出，未实际标注；A/B/裁决人员、培训预算、数值门未确认，保持
NOT_RELEASED。现有来源索引不是Question→Slot语义标签，不能宣布E02机制门通过。

RQ：能否可信建立Question→必要证据→原文span→候选slot版本？
Hypothesis：部分EventQA可定位必要事件，但多段依赖/摘要改写可能使映射不可用。
Existing Evidence：EA有400slot来源和50blind bundle；0/500可信span，不能假设已可用。
Controls：用冻结50题，每context10题，不根据预测/选槽成功替换题。
Variables：独立标注者判断与裁决，不做memory/model干预。

标注协议：

- 先10题（每context在冻结十题中按既定SHA取2题）作为规则可行性试标。
  两位独立熟悉英文叙事阅读的标注者，随后独立扩至全部50；裁决者不替代独立标注。
  10题为pilot不作机制结论；规则若变，10题重标/版本记录，确认集另审批。
- 初始界面只给原始question（含所有候选）和完整原context，不给model答案、策略、
  选中槽、score、正确率或当前算法建议。先自行选择事件并定位证据；第二步再核查
  dataset gold，单列label_mismatch，不强迫原文支持gold。不使用LM代替人工真值。
- 保存context SHA、Unicode字符[start,end) offset、逐字span、必要事实/时序/共指理由，
  distinguish anchor-event / necessary supporting / sufficient combined evidence；
  最小**充分证据集合**可由多段组成；替代充分集合单独编号，不能把任何单段命中当满足全部。
- 无法定位、只靠常识/候选先验、证据矛盾、多个候选均可、gold疑似错误、需未提供前文
  分别编码unknown/ambiguous/label_mismatch等；不填不相关、不自动删除或换题。
- 独立阶段结束后评价：可定位覆盖率、status原始一致率与Cohen kappa（稀疏类别时
  同报混淆表）、span token-IoU/F1（固定tokenizer、一对多最佳匹配规则）、必要事实
  集合F1；给逐context分母，不以kappa一个数掩盖范围分歧。
- 第三人依据两套证据及原文裁决，记录理由和未决项，不能以模型预测决定裁决。
  在标注者独立阶段隐藏slot信息；裁决后才按EA chunk_offsets映射到slot当前版本。
- 映射分direct-source候选、indirect-possible候选与无来源候选；refresh旧内容与evict
  的版本分开。跨度跨chunk时保留多对多关系。关联不是latent内容真值；indirect closure
  不能扩为“这些事实必定被保存”。无法确定negative slot时不要假造irrelevant组。

Required Data：50题bundle、五context全文、官方chunk offsets、EA provenance版本；
原文/候选/正确gold审核材料，不新增数据集。未来标注UI/去泄漏导出单独批准。
Metrics：覆盖率、一致性、span/事实集合指标、关联槽可用性；不评价方法正确率。
Stats：以题和context报告，delta_coverage/delta_agreement执行前审批；至少覆盖五context，
pilot仅判断可行性，不能当对整个EventQA机制结论。
Cost：GPU0。两位标注者+裁决者；若每题每人20–40min，50题双标约33–67人时，
**仅排期情景**，不含阅读五长context、培训和裁决，尚无实测；pilot先测人时再审批50题预算。
Success：事前批准覆盖/一致性门达到，并能给足够source-related/unrelated候选；
数值未批准不标“passed”。Failure：定位或一致性不足、负对照几乎不存在。
Stop：pilot不可行即停依赖映射的E04及E01语义部分；不靠语义搜索自动补成真值。
Expected Value：打通独立来源关联层，区分没有证据/无法定位/latent未验证。
paper_role=supporting methods；section_id=evidence_mapping；item_id=E02；claim_links=C02/C11/R04/R08。

## E03 — 当前bank的Conditioning On/Off配对（第三优先/首个GPU候选）

RQ：有无历史conditioning，哪些query收益/受损？H1/H2/H4。
Hypothesis：当前on可能主要改变有效候选产出，且query-level可能存在正负收益；均待检验。
Existing Evidence：M3只有on；旧P7 off不是同snapshot配对，不能复用作off。
Controls：on=当前M3 full canonical两槽；off=同bank同检索但配置
`retrieve_but_do_not_condition`，仍生成8新latent。只在查询边界关闭conditioning，
不能重新构建一个off bank。每题先记录selected indices，on/off相同。
Variable：R是否进入Weaver。**off无conditioning tokens是干预本身**，不能同时满足
两臂“实际conditioning token数相同”；检索预算均2槽16tokens、Reasoner新增均8tokens。
不能用zero/random latent当作off来伪装预算匹配；长度/position变化属于该接口总效应，
若要隔离内容效应使用E04等长换memory。
Required Data：冻结50题、seed42对应五snapshot、原始RNG、checkpoint与当前runtime。
Metrics：共同指标全套，primary全题strict on−off；H1用parser/valid变化描述，
H2用全题C/W/I表，不能以共同valid子集直接推格式控制后的因果。
Pilot/reuse：先10题（每context2题、固定SHA）各on/off各重复一次：40次生成，含
20次额外重复用于检查复现，不选择较好答案。若on重放与旧记录一致且运行identity
匹配，剩余40题直接复用旧on，只新跑40个off，E03基础上限80次新生成。
如果旧on无法复现，保留漂移/非确定性证据，停复用，不自动补跑直到匹配。
Stats：50题seed42仅探索；如需“稳定正负样本”，单次greedy一对不够，另批5seed
同题确认，报每query的正/负/零分布。不能先筛成功/失败再充当总体结果。
delta_onoff与最小正负样本门待批；多seed扩展不能自动进行。
Cost：GPU需要。上限80次生成+五snapshot读取、无新bank构建；total GPU hours=
加载/核验时间+sum各arm实际latency/3600。旧M3存在争用且不同调用耗时变动，
不能用论文P7平均时间外推；10题pilot后提供wall-time/显存/吞吐再批准扩展。
拟pilot单卡12GiB资源门、一CPU协调、输出<=100MiB，时间上限待pilot执行批准。
Success：复用门通过，净效果/伤害超过预注册门并非单context/数值漂移造成。
Failure：只有相同结果/漂移占主要差异/无稳定异质性；不能因此给gate造标签。
Stop：snapshot/source漂移、重复不稳定无法解释、预算达到即停；不给Selective gate自动授权。
Expected Value：最少推理建立当前真实on/off总效应与正负样本，为后续gate研究设门。
paper_role=main_required；section_id=conditioning；item_id=E03；claim_links=C05/R01/R05/R06。

## E04 — 等预算Latent内容干预（E02通过后，条件式执行）

RQ：历史memory的来源内容与query的关系是否影响答案？H3/H4。
Hypothesis：source-related候选比matched source-unrelated候选更有帮助；不是默认latent真值。
Existing Evidence：仅来源映射/顺序敏感，未有相关性标签或内容干预。

|arm|改变的唯一主要变量|控制/来源|
|---|---|---|
|A Original|无（参考）|复用M3 full，两槽canonical|
|B Related candidate|选入memory内容/slot身份|标注后预注册source-related槽，不按模型答对挑选|
|C Unrelated candidate|选入memory内容/slot身份|标注后matched source-unrelated槽，不是latent已证无关|
|D Random|选择memory内容/slot身份|复用同题同seed M3 Random-2，identity固定|
|E Memory permutation|同两个slot的先后次序|reverse Original canonical；槽内8tokens不打乱|
|F No conditioning|是否提供R|复用E03；这是零conditioning预算的特定例外，单列不冒充等长度内容对照|

B/C主要contrast采用两个槽中固定一个filler，仅替换另一个source-related vs source-
unrelated槽；filler固定、不能自己完整携带目标证据。选槽不依模型输出；匹配age、
memory/key norm、chunk长度/位置可行范围，在见结果前批准matching caliper。
两槽均8tokens，slot拼接统一index canonical；替换导致relative position变化时，
应预注册位置匹配（例如相同替换位置，filler位置不变）并在导出后检查，不能让排序
变化伪装内容效应。不可实现则改为明确“内容+位置”非隔离对照，不通过内容因果门。
其他多段证据需保留组合充分性；无单槽可覆盖不改标签强行找related。每题选槽
table包含direct/indirect关联、缺失、norm/age与位置匹配质量。
E既不是跨query donor置换，也不是latent token shuffle；如果native顺序恰为reverse
可复用native_order，否则单独生成；记得canonical可能已经score顺序一致。

Required Data：E02通过的标注、同bank存活slot版本、相同snapshot与M3 RNG/模型。
Fixed：同题、同bank、checkpoint、decoding、RNG；A–E均2槽16conditioning tokens，
Weaver/Reasoner新latent均8tokens。F例外如上；所有arm为同一组预注册题目，不能
输出后删invalid；E02无法定位题目在50题coverage表保留N/A，不当作答错或不相关。
若B/C不可构建，则停止语义contrast；不偷偷替换冻结样本或缩小为成功题。
Metrics：primary B−C全纳入题目strict差；secondary共同指标/转移、相关性×query交互，
A/D/E/F作为预注册解释性对照。所有分母、未纳入原因和五context覆盖公开。
Stats：按共同契约；delta_content、最小覆盖、匹配容差先审批。内容可造成输出变化
不够，须相关性与正确目标方向一致；仍不能唯一排除源位置/表示分布等残余混杂。
Cost：最多50题：A/D旧记录复用，F复用E03，新B/C/E最多150次生成；若E可复用减少，
不为凑满而跑。GPU小时依pilot t_B/t_C/t_E，不能可靠预估；先批准适格10题上限30次，
再报价扩展。标注可能导致eligible题数很少是主要风险。
Success：可信来源+matched内容干预达到delta_content且跨context方向支撑；只支持
“来源相关latent候选有内容敏感下游贡献”，不宣称每个所需事实已逐位存于latent。
Failure：相关/不相关无差、只有valid变化、匹配混杂未控或coverage不足；不能倒推
latent无信息。Stop：E02失败/无可靠negative槽/匹配不可实现即停dependent contrast。
Expected Value：区分Source Relevance、Latent Content Contribution、Downstream Improvement。
若还需历史事实保留强主张，另设计有监督事实probe/最小源事实反事实construction实验；
probe失败不证明无信息、probe成功可能是模型先验；本计划不授权此扩展或重构bank。
paper_role=main_required mechanism；section_id=memory_content；item_id=E04；claim_links=C02/C05/C11/R04。

## E05 — Weaver vs Direct的一槽桥接（不是全组合）

RQ：排除检索数量和Reasoner新增长度后，Weaver接口差异还存在吗？H1–H4。
Hypothesis：相同slot信息经Weaver重生成比直接投影使用有不同收益，原因待拆解。
Existing Evidence：历史full2/direct1总体差异非纯融合对照，当前M3无direct组。
Controls：W1=同snapshot原生最高分槽→Weaver→8新tokens；D1=同一个槽→P_W→R→8tokens；
W0=E03 off；W2=M3 full。W1/D1同一个检索槽、相同顺序、同8conditioning候选预算及
最终新增8tokens；D1没有Weaver conditioning输入，本质干预就是使用路径。
Variable：W1/D1之间只变latent再生成 vs直接使用接口，不同时换query表示/检索来源。
W2−W1只作为槽数敏感性，不能当fusion差异。不要做Direct2（最终16 vs8会混淆）。
Required Data：E03同50题、top1索引、RNG、模型；确认W1可通过query-only adapter
截取同一个原生top1，需另批诊断实现，不在M6改核心配置。
Metrics：primary W1−D1全题strict；H1 parser/valid、H2 C/W/I、H4按query配对；
H3还需要E04内容证据，不能由W1−D1单独证明semantic consolidation。
最小稀疏扩展：若E04表明内容贡献且E05路径差仍不清楚，只批准同一个related与
unrelated单槽分别走W1/D1的2×2（最多10题40次新调用）测路径×内容交互，不做全组合。
Stats：共同契约，delta_regeneration先审批，报告位置/长度是否真正一致与剩余LM先验混杂。
Cost：W0/W2复用；W1/D1最多100次新生成（50题），可先10题20次；optional40次另批。
总GPU时间未知：D1跳过Weaver，需分别测t_D1/t_W1，不用full latency硬套。
Success：匹配门通过且delta_regeneration达门；Failure：预算不等/无稳健差异/只有
输出合法性变化却声称历史事实融合。Stop：E03复现门失败或top1不一致即停。
Expected Value：为“为什么Weaver有帮助”提供最小接口证据，不重新包装已有regeneration。
paper_role=main_required；section_id=regeneration；item_id=E05；claim_links=C05/R05。

## E06 — Query收益可预测性审查（只在E03稳定异质性后）

RQ：推理前特征能否在未见context预测conditioning收益，而非拟合答案？
Hypothesis：可能有可预测收益；存在正负例只是必要条件，非充分条件。
Existing Evidence：当前无可靠on/off标签，阈值全部通过；**现在不能进入gate实现**。
Controls：always-on、always-off、仅校准raw max-cosine阈值、现行decayed-score阈值，
以及极小预注册特征模型。后者只是离线反事实policy评估，不改运行算法。
Variables/Features：raw max cosine、top1/2及top2/3 margin、decay、access/write age、
q norm/可得几何摘要、bank容量等推理前字段；不得输入gold、correctness、raw输出、
后验转移、标注source-span标签或query ID/答案信息泄漏。训练目标可用独立训练折的
on/off正确性差，但这些绝不是推理输入；test gold只用于锁定后的最终评估。
Required Data：E03稳定paired结果及需要时追加批准的多seed；已保存预决策特征。
Split：按context而非question/seed划分；同题所有seed同折。五context outer LOCO，
inner仅训练context选阈值/模型，outer test一次评估；不把outer调参结果当独立验证。
五context和50题通常过少，若正负类别在训练context不足不拟合、不硬报AUC；
真正确认需另批独立contexts，不能用同context另一seed冒充validation。
Metrics：held-out模拟policy的全题strict相对always-on/off与两种阈值；coverage/use rate、
harmful rejection/beneficial acceptance、校准/类别分母；不是只报分类AUC。
Stats：context配对探索性CI、delta_policy与伤害界待批；记录训练调参全部尝试。
Cost：复用paired表时GPU0，CPU<=4GiB、输出<=100MiB；确认所需新context推理费用
不在本预算内。Success：真实held-out收益超过最强简单阈值且不靠泄漏/context记忆。
Failure：不优于阈值/训练不稳定/无正负支持。Stop：E03异质性未确认、缺独立验证
或阈值重新参数化解释未排除，建议暂缓Selective Regeneration而非发布新算法。
Expected Value：判断gate有没有独立研究价值，不预先实现。
paper_role=conditional future method；section_id=selectivity；item_id=E06；claim_links=C02/C05/R04。

## 成本总账与执行顺序

所有项目状态均DESIGNED；当前启动次数0、人工标注0。

|次序|项目|最小工作单元/新增生成上限|资源/授权|
|---|---|---|---|
|1|E01|复用全部旧分数/排序/norm及tokenizer-only窗口检查；不生成答案|CPU；首先单独批准实现/运行|
|2|E02|10题双标pilot→50题；不生成答案|两标注者+裁决，工时先测再批准|
|3|E03|10题on/off双重复40次→50题基础最多80次|GPU；不依赖E02语义标签，但依赖复现门|
|条件|E01V|embedding+projection-only，2,500向量，不做LM生成|CPU权重提取另批|
|条件|E04|相关/不相关/逆序最多150次，先适格10题最多30次|E02过门后GPU另批|
|条件|E05|50题W1/D1最多100次，先10题20次|GPU；2×2扩展最多40次另批|
|最后|E06|paired特征表离线交叉拟合，无新gate|CPU；类别与独立验证不足则停止|

基础小规模GPU上限80+150+100=330次，不含另批重复/跨seed确认/optional40，
不是已批准预算；每个项目pilot后均有停止/报价门，不自动串行全部运行。
不重跑M3六策略15,000条，不全量组合memory×integration×seed。
优先依赖顺序为E01→决策；E02与E03分别解决标签/收益缺口；E04/E05按早期结果择需。
