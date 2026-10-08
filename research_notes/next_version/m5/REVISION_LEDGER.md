# M5：R01–R16 修订分流

父矩阵：[REVIEW_ISSUES](../REVIEW_ISSUES.md)；证据缩写见[账本](CLAIM_EVIDENCE.md)。
Emergency/Official/AI均为[原审稿](../../../review/review)的来源标签。
行内分号拆开同一review ID的子问题；不表示所有子问题已经解决。
所有“修改”均为拟议，提交PDF和论文源码尚未修改。没有最终author response。

|ID|来源/原问题概括|类别；回应立场|已有证据与剩余缺口|主路由 / 优先级 / 完成门|
|---|---|---|---|---|
|R01|Emergency/AI：EM收益是否主要来自解析/格式|evidence_gap；部分接受，纠正parser前提|E1/EC有真parser、strict/valid及转移；没有因果中介证据|evidence_repackaging，P0；分开展示五种指标，禁止写格式因果|
|R02|Emergency/AI：parsed-only、chance、参考上限|evidence_gap+experiment_gap；部分完成|E1有parsed/chance16.67%理论参考；EC有带分母条件子集；同协议模型chance、oracle/长上下文参考未跑|evidence_repackaging，P0；公开分母/偏差；supplementary_experiment，P1，另批参考实验|
|R03|Emergency：random/last-two与近期槽假说|evidence_gap+claim_scope；控制已补，归因仍不足|E3比较完成；EB/EC支持访问recency一致，不支持末写入假说或语义相关|claim_downgrade，P0；新增独立M3诊断表和固定选槽负向发现，不混历史主表|
|R04|Emergency：query/key跨空间、低阈值相关性|experiment_gap；接受未验证|EA无可信span；EB分数复算不是相关性验证，projection不等于alignment|claim_downgrade，P0；若选B走可信span与latent保留验证，禁止仅调阈值替代|
|R05|Emergency/AI：Weaver接口、token数、顺序、mask/position、Trigger、分布偏移|text_only+experiment_gap；接受|EC有顺序行为差异；张量接口逐项源码说明仍待写；无训练分布稳健性证明|text_revision，P0；下一写作前只读实际tensor-flow审计，区分retrieved与new/injected tokens；偏移只能列限制|
|R06|Emergency：greedy为何有seed方差|evidence_gap；未解释|E0复算历史方差但dirty源码缺失；E3配对RNG保证当前可比不回答历史来源|explicit_limitation，P0；记录已知/未知RNG点，不猜初始化因果；必要smoke另批|
|R07|Emergency/AI：decay收益与显著性|claim_scope+experiment_gap；收紧|历史strict方向反转；EB是访问关联不是干预；五context效力有限|claim_downgrade，P0；去除强decay因果；如坚持则批准同bank/RNG no-decay配对，并预注册主指标/统计单位|
|R08|Emergency/Official：一模型任务、无agent/tool-use外部验证|claim_scope+experiment_gap；接受|E3仍同一模型五context；EA证据年龄未知；vanilla模型/外任务未补|claim_downgrade，P0；改外推；外部同协议baseline与真实agent实验为单独批准门，A也不能称已解决|
|R09|Official/AI：动态读写下降与冻结读只协议|evidence_gap+experiment_gap；接受边界|主文已有冻结还原；EA仅构建生命周期追踪；动态记录来源及下降机制待审计|explicit_limitation，P0；先baseline_recovery，P1，再决定受控dynamic干预，不声称污染已确诊|
|R10|Official/AI：新颖性、遗漏先例、latent竞争对照|evidence_gap+experiment_gap；接受待核查|文中已讨论MEMORYLLM/M+；评审提出EM-LLM/Memoria；无新增原文核查或公平实验|literature_positioning，P1；原始文献逐项功能/训练/接口核验；baseline_recovery及可比性门后才考虑GPU|
|R11|Official/AI：Fig4 log轴、CPU overhead/效率边界|editorial+experiment_gap；部分已说明|PDF p7图注已有log，正文有限制；CPU复制/检索细分与问题数摊销未测|evidence_repackaging，P0；答复指现有位置；成本细分走独立supplementary_experiment，P1|
|R12|审计扩展：p6 same GPU来源冲突|evidence_gap；接受纠错|E0：P7 GPU5、Dense GPU4；不能保证同物理GPU|text_revision，P0；删除不实同GPU措辞/披露来源，若保留严格比较则同设备复测另批|
|R13|Official/AI：总VRAM、CPU bank、摊销break-even|experiment_gap+claim_scope；接受|E0只支持peak incremental GPU及现行100题/context摊销，非总系统内存|explicit_limitation，P0；保留边界；强效率结论需total GPU+CPU与构建/query计时及问题数成本曲线另批|
|R14|Official：泛化、依赖Weaver、黑盒纠错风险|claim_scope+text_only；接受|原p7已有单任务/模型限制；EA来源元数据不是内容解释，迁移和纠错未验证|explicit_limitation，P0；补适用范围和覆盖/错误追踪风险，不虚构可迁移模块|
|R15|AI：typo、符号、tuple/set、last-access、matched路径、拒绝率|editorial+evidence_gap；接受|PDF确认拼写rretrieves；E0确认Matched-16文本16tokens；拒绝率未知|text_revision，P0；统一符号/路径/token说明；拒绝率先审计日志，缺日志明确unknown，不能由完整运行推零|
|R16|审计扩展：来源/复现链，避免latest替换|evidence_gap；部分解决|E0输出冻结但旧dirty源码缺；E3新campaign身份/恢复/25快照链完整|evidence_repackaging，P0；附新旧身份、尝试与限制；不宣传历史源码完全复现|

## 执行顺序与批准门

P0（两路线共同）：评估定义、campaign分离、固定选槽与recency一致性、顺序敏感、
负向结果、same-GPU纠错、收紧decay与agent外推。写作授权后才实际改稿。
接口tensor-flow只读审计应先于方法段写作；未审计不能补出猜测的mask/position/token数。

P1（按论文保留的主张选择，不是自动全跑）：文献原文定位；动态历史来源；
Matched-16拒绝日志；参考上限/同协议plain模型；成本计时。
每项先有可比性/输入可用性审计，再给单独执行计划。未满足不能标记review fully addressed。

P2（路线B机制门）：可信span标注→latent语义保留验证→按证据决定是否改方法。
不直接扩benchmark，不默认表示学习、改decay或再训Weaver。

作者回复结构仅作为后续模板：承认/澄清问题→准确证据位置→实际稿件改动位置→残余限制。
本阶段只有前两项及拟改动，不能使用“we have revised/added”宣称尚未发生的改稿。
