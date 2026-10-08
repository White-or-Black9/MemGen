# M3：六策略 EventQA 正式比较（已批准，实施中）

## 依据与目标

父证据：`m2/M2_REPORT.md`；审稿项 R03、R07。
M2 的 70 个输出和协议检查已通过，但 smoke 只有 seed42/context0 前 10 题。
full 与 recency 的选槽/输出全部相同，native 顺序改变 q4/q8 的原始答案。
下一阶段应验证完整策略的优势是否跨 context/seed 稳定，不能把 smoke 当成效果证明。
用户已批准实施本阶段，并明确使用所有符合资源门槛的 GPU；不授权后续阶段。

## 1. 正式协议与实现

- 新增正式 runner，复用 M2 策略/快照接口；保留 M2 runner 的 10 题硬限制，不把它改成通用长跑入口。
- 六组：full（canonical slot-index order）、random、last_written、cosine_only、recency_only、native_order。
  前五组隔离内容选择，第六组保留原 P7 score-order 作为参考，不混入纯内容选择结论。
- 固定 5 个 EventQA contexts、每个 100 题；bank seeds 为 42/142/242/342/442，
  construction seed 为 base seed + context index。六组全部运行：25 个 seed/context jobs、
  25 次 bank construction、15,000 个唯一策略回答（前五组 12,500，加 native 2,500）。
- 所有 jobs 重新构建 bank；不把 M2 旧来源指纹的快照直接当作新源码快照使用。
  每个 seed/context 只构建一次，保存 CPU keys/values、全部元数据/计数器、最后写入日志、RNG 和 hashes。
- 保留 M0 的默认 unconstrained prompt、checkpoint/backbone、chunk4096、slot8、capacity16、
  top-k2、retrieve0.05/update0.10、alpha0.05、generation40、batch1、Weaver-space/CPU storage。
- 每题恢复冻结 bank 和生成 RNG；full 原始阈值/top-k 决定 n，其他策略取同样 n。
  random 用独立 seed/context/query RNG。所有策略 query writes=0，不跨 context 分享 bank。
- master manifest 固定源码、checkpoint、配置、数据和 scorer 身份；运行前核验，源码/输入漂移时停止，
  不把不同版本的输出合并成同一 campaign。不按历史 EM 接近程度挑选运行结果。

## 2. 测试和执行门

- CPU 验证 25-job/15,000-record 清单、唯一任务标识、seed 派生、n=0/1/2、六组预算对齐、
  私有 RNG、来源漂移拒绝、失败/恢复和完整性聚合；复跑 M2 的 14 项测试及 M1 的 11 项评分测试。
- 保留并披露旧 bank 测试的两项 debug 字段断言失败，不在本阶段改 production 或训练。
- 加入正式 runner 的无模型 mocked 集成测试，覆盖 construction 一次、多题恢复、六策略结果保存、
  异常中断和恢复。验证未完成清单不能产出正式汇总，不额外增加 GPU smoke 回答预算。
- 用户批准正式阶段且 CPU 验收通过后，才重新检查 GPU/缓存/CPU 负载。每个 GPU 至少有
  12 GiB free VRAM，使用所有符合条件的 RTX A6000、每卡一个串行 worker；只有一张则单卡运行，
  无可用卡则报告并停止，不干预其他任务。物理 GPU 不沿用旧快照假定。
- detached tmux；每 worker 只加载一次模型，按明确 seed/context job 清单调度；同一 job 的
  六策略在同一 GPU/model 上执行。记录 job 到物理 GPU 的映射，不把耗时当论文效率数据。
- 每份逐题输出写入唯一标识并原子完成；已完成且身份/源码/hash 一致的结果才允许跳过。
  失败保留日志，仅瞬时 I/O 错误最多重试一次；OOM、身份漂移或契约失败停止 campaign。
  记录真实 generation attempts，不从多次尝试中选最好答案。
- 文件锁领取完整 seed/context job；恢复仅回收本机已死亡进程的 lease，不抢占存活 worker。
  完整但来源不一致的输出直接拒绝；损坏输出保留原字节和错误记录，不静默删除。
- 固定 master timestamp、构建输入和 RNG；避免不同 worker 或恢复时的时间变化进入 prompt。

## 3. 指标与验收

- 必须得到完整 15,000 条唯一记录，六策略每组 2,500；缺失、重复、候选提取错误或不同身份均阻止正式汇总。
- 主指标：全题 strict candidate accuracy。同步报告 official EM、raw recall、candidate-valid、
  parsed/valid 条件指标及 heuristic flags，分开呈现，禁止从格式标记推算 parser accuracy。
- 主比较预设为 full−random、full−last_written；报告每题胜/负、25 个 seed/context 差值、
  5 个 context 均值和 5 个 seed 均值。native−full、full−cosine/recency 是预设辅助比较。
- 对 5 个 context 进行 10,000 次整块 bootstrap（分析 RNG seed20261002），在块内保留
  同一 context 的全部题目和五个 seed，报告差值的 95% exploratory percentile interval。
  明示只有五个独立 context、区间脆弱；不把 2,500 条重复问答当作独立样本，不声称稳健显著性。
- “方向稳定”仅作为规划标准：两个主比较均全局均值为正，且分别至少 4/5 contexts 和
  4/5 seeds 的平均差值为正；它不是统计显著性或 semantic relevance 的证明。
- 分别报告全题/n>0 子集；按 seed/context 汇总 full 与 recency/last/random 的选槽重合、
  查询间选槽变化、cosine/age/final-score 范围；共同有效候选子集保留分母和选择偏差说明。
- 汇总前独立校验 source/input/snapshot hashes、配对 question/gold/query 身份、数量/latent budget、
  零实际写入、冻结状态不变、官方 scorer 身份及 raw 到 summary 的计算一致性。

## 4. 交付和停止条件

- 独立新 campaign 保存 master/worker manifests、bank snapshots、逐题 traces、失败日志、
  official/strict 评分、逐 seed/context 表和正式阶段报告。M0/M1/M2 原件只读。
- 按结果收紧或保留主张：方向稳定时规划直接 relevance 验证；优势不稳定时优先诊断
  temporal dominance、key/query representation 或输出行为，不把不显著解释为等价。
- 即使全局 full 优于 random/last，也必须检查 recency 重合，不能自动归因于语义检索。
- 完整结果验收或明确失败后停止，再制定下一阶段；不自动扩大 benchmark、改 key space、
  删除 decay、编辑论文正文或提交 Git。
