# 提交版审稿问题与证据矩阵

来源：`review/review` 的 Emergency、Official、AI 三份审稿；页码以提交 PDF 为准。
候选建议中的因果解释仅作为待检验假设，不提前写成论文结论。

| ID | 提交版位置与问题 | 当前证据 / 状态 | 下一动作 |
|---|---|---|---|
| R01 | p5–6 Fig2：EM 是否主要来自格式变化 | M1历史重评分完成；M4-B2当前M3净正确差主要伴随无效候选↔正确净转移。无效绝大多数为非候选，不是parser空/None | 修订评估措辞；不把候选有效性差异写成格式修复因果或语义证据提升 |
| R02 | p5–6：parsed-only accuracy、chance 与 upper reference 缺失 | M1原评分/chance已拆分；B2共同有效子集明确分母及选择偏差，last-written子集有反向结果。oracle未执行 | M5整合已验证分母/限制；不以条件子集替代全题指标，oracle仍需独立批准 |
| R03 | p3–4：retrieval 是否优于 random/last-written | M3 full严格20.20%，random3.92%、last-written1.80%；B1验证full/访问recency集合2500/2500相同，与last-written槽交集为0；每bank固定选两槽 | B2已完成输出分解，M5已整合；不作语义归因或将访问recency混同末写入 |
| R04 | p3 query/key cosine；低 threshold 语义含义 | 投影不证明relevance；M4-A来源可追溯但500题无可信span。B1显示查询向量/cosine随题变化而top2固定，分数复算误差0 | 收紧query-adaptive语义选槽主张；语义验证仍需可信标签，不计算伪Recall/MRR |
| R05 | p4 Weaver(H,R) 接口、排序、mask、position、Trigger、token 数 | 论文描述不足；见 modeling_memgen.py Weaver-space 路径及 config | 用实际运行配置和源码说明；区分 retrieved tokens 与 regenerated/injected tokens |
| R06 | p4–6：greedy 但 P7 seed variance | 实际方差已复算；历史 dirty code 影响可追溯性 | 静态 RNG 与初始化审计，必要时独立 deterministic smoke |
| R07 | p6：decay 因果结论与显著性 | B1选槽access-age1、write-age4–17，构建期访问模式关联；query间状态恢复。旧no-decay差异仍无新增因果/显著性证据 | 不将访问关联当有害反馈或decay收益证明；不自动改decay/复跑 |
| R08 | p1–2：long-horizon agent/tool-use 外推 | 主实验证据仍是 EventQA，一模型、五 context | 先 evidence-age 分析；检索机制通过后再讨论扩任务 |
| R09 | p3–5：dynamic read-write 与冻结协议 | 提交版主文是 frozen；dynamic 证据需单独定位补充来源 | 后续写入/refresh/evict 诊断，不把污染猜测写成已知原因 |
| R10 | p2：novelty 与 latent competitors | 已讨论 MEMORYLLM/M+；EM-LLM/Memoria 未纳入；survey 已有 | 后续查原始文献、精确界定 regeneration 贡献，评估对照兼容性 |
| R11 | p7 Fig4：对数轴与效率边界 | 图注明确说明 log scale；正文已有硬件/协议限制 | 标记已说明；CPU copying/retrieval、amortization 仍待补 |
| R12 | p7：“same GPU” | 主 campaign GPU5，Dense campaign GPU4，证据冲突 | 修订同设备措辞或设计严格同设备复测；M0 不复跑 |
| R13 | p7：绝对 VRAM、构建摊销、CPU bank 开销 | 增量分配指标已有边界；不能解释为总系统内存 | 后续区分 total/incremental GPU 与 CPU，以及问题数摊销 |
| R14 | p7 Limitations：泛化与黑盒风险 | 单任务/模型与硬件范围已有；可追踪、纠错风险不足 | 后续写作补充适用范围与可追溯元数据讨论 |
| R15 | p1 typo，p3–4 notation，p6 Matched16 reject rate | typo rretrieves 存在；Matched16 是文本16-token控制；拒绝率未量化 | 写作阶段修正；从日志核查失败/拒绝数量，不从完成运行假定零拒绝 |
| R16 | 全文：实验来源与复现链 | 旧索引过期、正式运行 dirty、记录 config 路径缺失 | M0 新证据清单；新实验强制源码/数据/环境身份 |

M0审计时不把“Full≈Last2”或“提升只有格式效果”当作已知；更新后M3只确认当前
campaign的full与recency-only选择集合一致，不等同last-written，也不替换历史P7。
未来 parsed accuracy 或 recall 变好仍不足以单独证明检索相关性；需独立 retrieval diagnostics。
M4-A/B1/B2均已完成；M5修订分流与剩余门见`m5/REVISION_LEDGER.md`，
路线选择见`m5/M5_REPORT.md`。拟议论文改动尚未实施，review并非全部解决。
