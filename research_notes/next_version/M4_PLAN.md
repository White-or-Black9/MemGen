# M4：来源可追溯性与检索相关性诊断（M4-A已完成）

制定日期：2026-10-08。当前交付仅为计划；不启动模型、GPU或新回答生成。
执行顺序：先批准并完成M4-A，验收后按可追溯性结果细化M4-B；不一次执行全部方向。

状态更新（2026-10-08）：用户已批准并完成M4-A，两次全量审计及12项新测试、
25项评分/调度回归测试通过；见`m4/M4_A_REPORT.md`及`m4/acceptance.json`。
实际机器产物分别保存在`m4/run-20261008-a1/`和`m4/run-20261008-a2/`，不覆盖复跑。
400个slot来源可重建，500题证据span与latent相关性仍未知；下一计划
`M4_B_PLAN.md`只优先诊断已有trace。以下正文保留原预设规格，不代表M4-B已获执行授权。

## 1. 父证据、问题与边界

- 唯一提交论文：`review/nference_Time_Latent_Me.pdf`，SHA256
  `6b04a3c49b9bb68cccf282059be34f7dadc516e4069e42b204fcae99f68ffed3`。
- 父campaign：`outputs/mab/review_m3_formal/20261002-m3-six-policy-v3`；
  contract SHA256 `a548306d4131e4df0cba1340d3272a4882feae1d9c993c881e281f4551e433c9`。
  25 jobs、15,000回答已完整验收；父报告`m3/M3_REPORT.md`。
- M3 full相对random/last_written严格正确率+16.28/+18.40个百分点，
  但full与recency_only在2,500/2,500组中选择集合一致、严格正确性一致。
- 对应review矩阵R03（公平选槽对照）、R04（cosine相关性）、R07（decay解释），
  辅助关联R01/R02（输出有效性）与R08（证据年龄）。
- 研究问题RQ-M4：现有记录能否可靠建立question→证据来源→存活slot的链条？
  若能，这条链如何限制“相关检索”主张；若不能，明确最小证据缺口。
- 不再用下游EM间接给slot贴relevant标签；不把gold文本出现当作充分证据，
  不把来源关联等同于latent语义保留，不把当前源码结果替换历史dirty P7。

## 2. 已检查的执行条件与资产

当前CPU可用内存约736GiB，工作盘余量约3.2TiB；本阶段选择单进程流式JSON分析，
不依赖空闲GPU，不加载checkpoint或Torch pickle，不下载/安装依赖。
实施后首次运行预算为30分钟、目标新增产物<1GiB；达到预算先留下partial状态，
不静默改变协议。实际用时在报告中记录，不当作模型效率实验。

已有资产：25份construction.json、context_identity.json及snapshot_identity.json，
15,000份逐题原子记录与score decomposition、正式scored_records/results，以及
manifest绑定的原始parquet、tokenizer、benchmark模板/分块代码。
本次抽查construction turn含写入/替换/逐槽状态与retrieved_indices_before_write，
context identity含chunk hashes；尚未证明每个turn与chunk的一一对应，也未证明
原始数据有足以唯一定位证据的source/event标签。这正是M4-A的验收目标。

可执行性：M4-A日志/身份审计runnable-now；精确chunk重建取决于本地既有依赖，
有缺失则记录blocked，不用另一种tokenizer/分块方式近似替代。
M4-B相关性指标受标签质量门控；oracle生成、key-space重设计、新benchmark、
完整人工语义标注或外部LLM标注均不在当前计划执行范围。

## 3. 第一子阶段 M4-A：先审计来源链，不生成相关性结论

### A1 固定输入与完整性门

新分析器/测试放`research_notes/next_version/m4/`，避免修改M3冻结源码清单。
输出独立目录，不覆盖M0–M3。manifest记录输入路径/hash、分析器hash、
父contract、rubric版本和实际环境；核对父评分COMPLETE、25 jobs与15,000记录覆盖。
复用M3身份/内容校验逻辑；父输入漂移、重复或缺失时停止，不修改父manifest绕过。

### A2 重建slot生命周期与来源的可识别范围

逐job按construction turn回放insert/refresh/replace/evict，区分slot index与slot
incarnation（索引复用后不能当作同一memory）；校验最终槽数、last-write和冻结摘要。
精确重建context chunks，要求每块SHA与context_identity一致，再检查实际构建循环
如何把turn映射到chunk。不能仅因两个列表等长便认定一一对应。

每个存活slot输出：job、slot incarnation、当前写入turn、chunk/hash、refresh次数、
可追踪的历史写入/检索依赖、unknown原因及来源定位依据。
直接写入来源、可能的累计输入前缀、经历史slot输入的间接依赖分别保存，
不把依赖闭包中的全部事件算作被latent保留，也不把refresh视为必然保留旧内容。
若函数实际输入含累积context而非单chunk，要按实际输入记录，禁止简化成单chunk来源。

### A3 审计question→事件/证据定位，冻结标注规则

先检查原始五个context共500道唯一问题的数据schema：question/gold/candidate、
event/source ID、时间/实体/关系字段与来源span是否实际存在。
对每题输出唯一精确定位、多处匹配、边界跨chunk、仅gold词面命中、证据缺失或无法判定。
只把明确事件ID或同时满足题目约束的完整事件span记为可核验的source evidence；
gold字符串搜索单独列为lexical proxy。多gold按全部合法答案处理，歧义不强行取首个。

冻结两层标签并严格区分：

- `source_evidence_overlap`：可核验证据与slot实际构建输入来源的关联，非latent relevance。
- `latent_semantic_relevance`：现有日志通常不能直接识别，默认unknown；除非另行批准
  可检验的语义验证协议，不产出伪造的二值gold-slot标签。

人工检查若必要，先冻结rubric和样本：每context从唯一question ID中按
SHA256("M4-A-20261008|context|query")排序取前10题，共50题，不按回答成功与否挑样本。
定位时隐藏策略、预测及正确性，保留原始span/offset和判断理由；只有一名分析者时
明确单人定性校验，不声称独立双人一致性。该样本不能替代全500题覆盖/unknown统计。

### A4 验证、交付和停止门

无模型测试至少覆盖：索引被evict后复用、refresh不等同累积保留、缺失turn、
chunk边界与hash漂移、同名事件/歧义匹配、多gold、unknown与负标签区分、
五seed重复问题不扩增独立分母，以及分析前后父输入不变。

产物：`m4/manifest.json`、`m4/provenance.jsonl`、`m4/evidence_mapping.jsonl`、
`m4/coverage.json`、`m4/M4_A_REPORT.md`及实现/测试。
报告分别给25个job的slot生命周期覆盖率、500个唯一问题的定位状态、
2,500个seed/question关联的可识别范围，保留全部失败/unknown分母。
不要求unknown低于人为门槛来判“成功”；如实发现来源不可识别也是有效审计结果。

验收必须满足：可重跑、身份/hash正确、无父产物改写、无模型调用，
每个来源/标签可回溯或明确unknown，测试通过，报告给出下一路线。
M4-A完成后停止，再细化M4-B；不自动启动补充构建或GPU复跑。

## 4. M4-B 候选方向（仅框架，依M4-A结果重新制定）

1. 优先用已有traces量化temporal dominance：查询间选择集合变化、full/recency/
   last-written overlap、cosine/age/decay/final score与排名；不混淆last-write与last-access。
2. 仅在来源标签可识别时计算source-evidence hit@实际n、bank可用证据覆盖与选择覆盖。
   n=0记为选择未命中，未知标签不当作负例；覆盖不足时给上下界/条件分母。
   Recall@1/@2/MRR只在完整候选排序与相应标签可恢复时定义；canonical slot顺序
   不是相关性排名，不能拿它计算MRR。所有指标明确为来源proxy，而非latent保留证明。
3. 对已有六策略输出按候选有效性与strict正确性做配对转移分解，保留共同有效子集
   的选择偏差；最多证明关联，不作格式或证据利用的因果中介结论。

先保留5个context的独立分组、再汇总seed；不把五次同题当独立样本。
若M4-A发现来源根本不可定位，M4-B收缩为纯trace/输出诊断，并提出单独的
instrumented construction计划，不能在旧bank上追补虚构来源标签。

## 5. 论文映射与路线

无可用artifact paper-contract接口；使用本地review矩阵和本文件作为证据账本，
不新造论文已验证结论。selected_outline_ref=`EXPERIMENT_PLAN.md`（修订路线，非成稿大纲）。
experimental_designs=`offline provenance audit`；todo_items=`M4-A1`至`M4-A4`。

|slice_id / exp_id / item_id|section_id|claim_links|paper_role / display|comparability / next action|
|---|---|---|---|---|
|M4-A|retrieval_diagnostics|R03,R04,R07|supporting；appendix来源覆盖表|同M3输入，仅离线审计；完成后制定M4-B|
|M4-B-temporal（待细化）|retrieval_diagnostics|R03,R04,R07|claim-carrying；main_text机制边界表|仅原trace；决定是否需重设计计划|
|M4-B-output（待细化）|evaluation_decomposition|R01,R02|supporting；appendix配对分解|同M3评分；不宣称因果|

当前method/comparator为M3六策略；manuscript_takeaway仍是：full胜过random/last-written，
但未与recency选择区分，semantic retrieval尚未建立。
若来源可识别，下一步制定M4-B的指标/分母/分析门；若不可识别，记录缺口并
制定最小观测补强计划。无论哪条路线，都不自动编辑论文、重训练、改推理配置、
开启新数据集或提交Git。
