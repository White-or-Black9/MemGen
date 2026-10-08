# E04-X：六题等预算来源候选替换pilot

2026-10-08，用户批准上一条六题Original / Source-related candidate / Random提议。
父对象：E02探索性AI来源映射与E03已完成40次配对pilot；M3原始Bank与结果只读。
exp_id=M6-E04-X；section_id=memory_content；item_id=E04-X；claim_links=C02/C05/C11/R04；
paper_role=reference_only（探索性，非原E04确认实验）。使用analysis-campaign固定切片，
detached-launch执行后台启动核验；不改模型/训练/算法/论文，不提交或Push。

## 问题与证据边界

RQ：同题同Bank固定两槽16tokens时，更换槽内Latent是否改变答案？
主描述对照B−A的全纳入六题strict差，另报输出变化；B−D为辅助对照。
H：来源候选可能改变答案，可能有正/负效果；不是假定来源相关等于latent包含事实。
本次不是原E04的B−C：没有可靠semantic-negative、人类独立双标或事前MEOI，
所以不授予Gate2/内容语义贡献门通过，不确认检索瓶颈/事实保留/算法有效性。
这是用户另批的探索性旁路，保留原E04计划和正式E02门未通过的记录。

## 冻结选题与选择规则（不按答案表现选槽）

只用E02 source_mapping_eligible且seed42的6题，保留全部原10题纳入/排除表：
c0-01/c0-02/c1-02/c2-02/c3-01/c4-02。四题因来源裁决非supported排除，
不是依据模型输出。AI裁决依赖与coverage=6/10必须披露；不能推广500题。

A=原M3 full canonical槽位；D=原M3 random，同M3确定性policy_seed，不重新抽样。
B=固定filler slot0，第二槽为覆盖该充分集合全部来源chunk的当前direct-source槽；
唯一时直接选，多个时最小slot index，禁止按正确率选。六题实际direct槽分别
1/6/13/5/11/3，因此B分别[0,1]/[0,6]/[0,13]/[0,5]/[0,11]/[0,3]。
A均[0,12]或[0,1]，filler0位置恒0、被替换位置恒1；不会将顺序变化混成内容效应。
filler0在六题均无已记录来源链关联，但**不保证语义无关**；A第二槽可能间接关联。
D两槽均canonical且预算相同，但filler/来源/age不匹配，只作辅助，不声称单槽因果。

只用实验目录内临时per-instance返回值适配器强制候选；基础full检索照常记录，
返回Memory及其真实索引/分数；不改变key、Bank持久状态或生产Retrieval实现。
B不是实际retrieval策略得出的语义选择，可能覆盖原threshold，必须标forced-source。
同16tokens、同filler及位置，替换仍同时改变槽的来源年龄、tensor数值/范数等属性；
打印年龄/范数用于诊断，没有预注册caliper，不宣称隔离了纯语义相关性。

## 复用与预算

三臂×六题=18个主分析答案。A/D旧答案只读复用；A先在当前GPU用同一适配器
重放六次，必须与M3/E03答案及完整原检索trace一致，否则停止复用并停止pilot。
B各生成两次（主分析repeat0，repeat1仅稳定性），共12次；新增GPU生成最多18次。
不重跑旧Random，不扩至其余44题/其他seed；旧Random未在当前GPU重放是复用限制。
同seed42原snapshot、Python/NumPy/Torch/CUDA RNG完整恢复、模型与tokenizer、
渲染prompt、40-token解码、determinism设置、一次Weaver生成8新latent、batch1。
单张A6000固定执行全部新调用；按资源检查选择GPU5（目前空闲显存约43.5GiB、
利用率5%，有其他用户任务）。启动前重查>=12GiB，不杀/迁移他人进程。
单worker tmux，自动串行；wall cap60min，输出<=100MiB。E03实际约63秒仅作
参考，不承诺共享GPU时延；本次不增加GPU来拆同题对照。

## 指标、检查与停止

全六题strict EM、官方EM/raw recall、parser非空、candidate有效、parsed-only分子分母，
逐题improved/regressed/unchanged，C/W/I转移、文本变化及raw/format转移。
另列五context效果、context等权均值/leave-one-context-out；c0两题，其余各一题，
context至多5独立单元且来源独立性未充分确认。不做题级显著性或等价结论。
MEOI未批，只有技术可重复性门；不能事后按六题结果选成功标准。

先做合成单元测试，prepare核验M3源码/模型输入/五snapshot/18相关旧记录与E02/E03。
每次记录候选tensor形状/哈希、来源版本、完整native选择、实际返回选择、预算与prompt；
检查当前B返回就是指定Bank tensor，bank写入0及指纹保持不变；重复不可选最佳。
来源映射漂移、source slot缺失、filler覆盖已标来源、预算/位置/prompt不符、
A复用门失败、B重复不稳、资源不足或达到上限均保存失败证据并停止，不自动重跑。
完整生成后自动用原官方/M1评分，写逐题报告；启动验证后不持续轮询。

解释规则：输出变化只能证明该接口对latent数值/槽身份敏感，不证明利用历史事实；
B更好仍是来源候选探索证据，B相同/更差不能证明latent无信息。暂不改变算法。
完成后停止，E04-C语义负例、事实反事实构建、扩样和Gate研究均须另批。
