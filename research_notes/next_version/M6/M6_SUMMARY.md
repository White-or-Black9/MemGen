# M6 — 机制审计与最小实验计划总结

最新离线分析已完成，见[E04_FAILURE_REPORT](E04_FAILURE_REPORT.md)。E04-X 18次
生成/评分及复算通过：Original1/6、Source0/6、Random0/6。四题输出变化均为前缀，
六题候选正文完全一致；唯一C→I题仍含完整gold，官方parser读到前缀“行”。
这收窄了“内容敏感”主张：当前仅输出前缀敏感，不证明历史事实利用或丢失。
E03 raw recall3/10 vs2/10，不能把全部收益都归于格式；两道收益题的来源证据仍不确定。
7项新分析测试通过，双跑/输入保全通过，无新GPU。下一建议另批E05同槽接口规划，
不自动扩实验或修改算法。下文E04启动描述是历史快照。

最新E04-X更新：用户批准六题三臂探索性来源候选替换，已GPU5/tmux后台启动，
PID2542702，详见[E04_EXECUTION](E04_EXECUTION.md)、[E04_STARTUP](E04_STARTUP.md)。
16项新测试通过；A/D复用，最多18次新增生成，主对照固定filler0和替换位置1。
尚无已验收E04结果，不等同原B−C语义确认实验或E02人工门通过，不自动扩样。
E03已完成40/40并评分：10题on strict3/10、off0/10，重复一致、旧on答案与选槽
10/10一致；这是当前配对pilot，不是新颖方法或事实利用证明。
下文启动时状态为历史快照，不作为当前完成状态依据。

最新E03更新：用户已批准固定原10题的on/off双重复40次pilot；
已在GPU0独立tmux后台启动，PID2160354，详见[E03_STARTUP](E03_STARTUP.md)。
11项新测试通过；启动核验只确认worker/显存/日志/输出根，不代表实验完成。
当前无已验收E03结论，不自动扩至50题、实现Gate或进入E04。

最新更新（2026-10-08）：已按用户授权完成[E02探索性AI裁决与来源映射](E02_EXPLORATORY_REPORT.md)。
10题为6 supported/1 conflict/2 ambiguous/1 not_locatable；6题各seed均有直接来源候选槽。
这是用户口头确认复核的AI辅助材料，不是人工独立双标，正式E02 Gate未通过。
26项E02测试通过，两次三核心SHA一致；未运行GPU或修改算法。下一建议另批E03
全部10题on/off配对pilot，不依赖本次来源真值。下面准备阶段文字保留为历史快照。

日期2026-10-08。当前状态：**审计/设计完成；E01与E01V的CPU诊断执行并验收**。
用户分别批准了E01/E01V，详见[E01_REPORT](E01_REPORT.md)、
[E01V_REPORT](E01V_REPORT.md)与[E01V验收](e01v/acceptance.json)。
E02标注准备已完成，人工标注未开始；E03–E06未执行。

最新E02-PREP更新：[准备报告](E02_PREPARATION_REPORT.md)。冻结10题、两个不含gold
身份/模型结果/slot的初始包、协调员来源索引及空表齐备。13项新测试与两次22文件
hash一致性通过；human_annotations=0，release=NOT_RELEASED。等待A/B/裁决人员、
暴露声明、培训/预算和覆盖/一致性门确认，不能把准备验收写成source mapping已验证。
下文“仅五份规划/实验未执行”等描述保留为最初规划阶段的历史快照，不是当前状态。
新增材料仅在M6目录；没有语言模型forward/答案生成、GPU、人工标注或Git提交。
E01V仅载入必要embedding行、projection及bank数据，不载入完整模型。

此前E01V板面：换clue的500个CPU q完全不变，换候选尾17token全部变化，但Full
Top-2仍2,500/2,500不变；平均QQ cosine0.996762。CPU标量容差与Full选择复现门
通过，但只有670/2,500记录的q与原GPU hash一致，不称全量bitwise复现。数值实现
也会影响完整排名。85项测试通过，两轮core一致；一个序列化失败尝试保留。
其建议的E02已推进到准备完成，尚未开展双标；不自动改Query或运行GPU。

此前E01板面：72项测试通过，两轮核心hash一致，历史证据保持原样。500题池化
窗口问题线索均0token，固定指令/chat共46/64token；raw/final完整排名可变，但
Full与semantic-only Top-2各bank均固定。语义相关性/latent保留/conditioning收益
仍unknown。该时建议的E01V现已完成，剩余未知按最新报告更新。
本阶段继承用户选定的M5路线B，继续方法论文方向，不预先认定算法有效或无效。
仅在本M6目录新增/更新材料；未回写M0–M5、论文、模型/训练/推理代码，无提交或Push。

## 1. 执行前检查与实际核验

- 分支review；HEAD `6784a26e65290e91be2ad8a965cf108cb19c7b22`；进入时干净，
  本地领先origin/review一提交；没有为本阶段拉取远端。
- M5_PLAN、CLAIM_EVIDENCE、REVISION_LEDGER、decision、acceptance存在并读取。
  M5旧decision的未选路线是历史状态，本阶段已选择B，原文件不改。
- M5只读validator再次PASS：12claim/16review ID、文档链接、PDF身份、M4两次
  共32份核心产物hash及当前汇总一致性。不是重新运行M4测试。
- M3：25任务done；六策略各2,500原始记录路径存在，scored记录15,000条；
  92源码hash与冻结清单相同，25bank文件hash与snapshot_identity相同。
- 2,500条full保存trace均返回2槽/16conditioning tokens、一次Weaver调用、
  Trigger inactive；query norm均有值，跨query cosine全null，完整q未序列化。
  这是既有记录范围检查，不是新实验，不等于再次验证所有15,000条内容hash。
- M4-A/B1/B2报告与验收材料完整；原始25bank、rendered prompt、分数及50blind
  样本索引可用。关键缺失是标签/完整q/当前off与direct答案，不是缺失已宣称完成的文件。

身份摘要：

|对象|SHA256|
|---|---|
|提交PDF|`6b04a3c49b9bb68cccf282059be34f7dadc516e4069e42b204fcae99f68ffed3`|
|M3 manifest.json|`3de9d00983fb3fef6b99cecb5bf0911f3a0e3ede575482b253c8ce02cd2c7ade`|
|M3 results.json|`8c1cd57bd2d7ccd681419253a85c7a88c16f2e0f2e9f11042bb354b976865071`|
|M3 protocol contract|`a548306d4131e4df0cba1340d3272a4882feae1d9c993c881e281f4551e433c9`|

## 2. 真实机制与M5之后新增认识

1. Query为Reasoner输入embedding→已有P_R→W→最后64位置mean；不是Reasoner
   last-layer hidden state。Key是Weaver输出8tokens池化；兼容空间不等于已语义对齐。
2. 历史两槽共16tokens拼在投影后的query之后；Weaver再拼8个learned query tokens，
   取最后8个输出hidden states投影回Reasoner，后者保留原始query。不是16历史tokens
   直接给full Reasoner，也不是只有latent输入Reasoner。
3. 当前M3 full按slot index canonical排序，native原路径按score排序；排序差必须单列。
4. 当前full和off的接口可以用已有配置表达，off保留retrieval与Weaver，仅不提供R。
   原生阈值/空检索fallback已有；本批2,500条全返回2槽，所以没有实际弃用历史memory。
5. checkpoint inactive Trigger使prompt调用一次，max_inference_aug_num=0禁止后续增强；
   通用active Trigger仍可SKIP，不能推广本设置为所有query永远调用Weaver。
6. 当前direct_top1只有一个槽8tokens；full用两槽16tokens后再生成8tokens。
   历史总体差异有retrieval预算混杂，应先同一槽Weaver-1/direct-1桥接。
7. q norm已经保存，可离线分析；QQ cosine/逐维方差需要完整q。优先只提取embedding
   和projection，不必跑完整LM。last64模板/候选尾部是否影响query仍是未验证假设。
8. full日志的reasoner_injected_latent_count初始化0未在正常full填充，不能当“未注入”；
   真实8tokens由代码与配置确立。未来诊断应核验tensor shape，不信单个debug默认值。

## 3. 核心证据仍缺什么

full>random/last-written已观察，full=recency与固定选槽/条件子集反向/旧no-decay
strict反向也必须保留。当前不能证明query-adaptive semantic retrieval、所需历史事实
保留、Weaver内容利用或可预测的query收益。当前M3没有同bank on/off/direct，旧P7
不能补缺。source chunk含证据也不能自动说明latent保留证据。详细分Observed /
Inferred / Hypothesized / Unknown见[EVIDENCE_GAPS](EVIDENCE_GAPS.md)。

## 4. 最值得先做的三个项目（尚未执行）

|优先|设计|价值|成本与主要风险|
|---|---|---|---|
|1 E01|复用M3/M4分数、norm、排序、固定bank query置换和tokenizer-only尾部检查|分清分数敏感/排名敏感/选槽敏感，先排除无需推理的数据缺口|GPU0，建议单CPU/RAM4GiB/30min cap；无标签不证明语义相关|
|2 E02|冻结50题，先10题双人独立标注pilot，裁决后映射source spans到slot版本|建立来源关联层，明确是否能做可信相关/不相关对照|GPU0，需两人+裁决；50题双标情景33–67人时外加长文阅读，实际需pilot报价；可能无法定位/无可靠negative|
|3 E03|同bank conditioning on/off，先10题各重复一次|取得当前真正的正/负收益与输出转移，决定gate是否值得研究|pilot40次GPU生成，复用门通过后50题总新增上限80次；时间需pilot测，非确定性可能阻止旧on复用|

E01/E02/E03优先级不表示自动执行许可。E03不必等待E02标签，但同样须独立批准。
E04等预算内容替换最多150次、E05同槽桥接最多100次是后续**条件式预算**；
不是新实验结果或已启动任务。完整细节、指标/MEOI/统计/停止门见
[EXPERIMENT_PLAN](EXPERIMENT_PLAN.md)。

## 5. 复用 / CPU / GPU边界

直接复用：M3五种选择策略及native-order答案；M4来源链、50样本索引、排序和
状态转移表。不得重复跑这些大实验。
可CPU离线：norm/score/rank/组合、保存分数的query置换、指标转移及已有阈值覆盖；
有标签后来源关联；完整q可另批CPU embedding+projection提取。
新生成需要GPU：当前on/off配对、相关/不相关内容替换、不能复用的逆序、W1/D1。
E06收益预测主要CPU，但现在没有可信paired标签，不能开始gate研究/实现。

## 6. 是否保留算法与下一版贡献

暂时**保留当前Retrieval和Regeneration作为冻结对照**，不表示已验证semantic机制。
已有Query-conditioned regeneration不是下一版新提出的算法。当前没有充分依据直接
改retrieval、实现Selective gate或归因construction丢失。
通过条件见[DECISION_GATES](DECISION_GATES.md)：内容可利用但选错才优先Retrieval；
内容有用但integration/可预测伤害成瓶颈才考虑Regeneration；构建丢失需独立证据。
无可信mapping时停止依赖它的实验；未知不能充当对算法无效的证据。
方法论文仍需新颖性、外部竞争对照/泛化、动态/效率边界等，M6不承诺已足够投稿。

## 7. 最初规划阶段的批准请求（已由E01结果更新）

**只先批准M6-E01离线诊断的实现与执行**：用已有JSON/tokenizer，不载入LM，
不生成答案，不改变算法；输出独立分析目录，先检查当前身份再按计划验收。
之后依据结果决定是否另批E01V、10题标注或40次GPU on/off pilot。
覆盖/一致性/MEOI数值及GPU时间预算在对应执行前明确批准，不事后定门。

本轮仅交付以下五份规划文档并停止：

- [IMPLEMENTATION_AUDIT](IMPLEMENTATION_AUDIT.md)：真实tensor流与源码行号。
- [EVIDENCE_GAPS](EVIDENCE_GAPS.md)：信任/证据范围与未验证命题。
- [EXPERIMENT_PLAN](EXPERIMENT_PLAN.md)：E01/E01V/E02–E06最小设计与成本。
- [DECISION_GATES](DECISION_GATES.md)：五机制门及条件式方法路线。
- 本SUMMARY：当前板面、实际完成范围与批准请求。

本阶段使用intake-audit和analysis-campaign的信任分层、可比性与停止门。
专用artifact/bash_exec/memory接口不可用，复用现有本地只读工具与Markdown；
没有伪造artifact登记或更新持久memory。用户限制优先，M0–M5文件保持原样。

文档验收：五份文件齐全，七项设计（含可选E01V）、五个机制门与18个文档链接
检查通过；M5 validator再次PASS，`git diff HEAD`为空，新增仅M6目录。
未重跑历史单元测试或实验，不将文档检查写成机制验证通过。
