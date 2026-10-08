# M2：固定同一 bank，比较检索内容选择

## 本阶段批准范围（2026-10-02）

用户选择 CPU 实现/测试 + seed42/context0/q0–9 GPU 验证，已执行并验收。
只构建一次 bank；五策略 50 个主要输出，另有 10 个 native-order 参考和
10 个 full 重复输出，共 70 个输出。运行入口 `scripts/eval/review_m2_smoke.py`，
策略及快照接口 `scripts/eval/review_retrieval_policies.py`。
下文正式 12,500/15,000 个 query 仅是未来规格，**不在本阶段执行**。
完成或阻塞后记录 `m2/M2_REPORT.md`，然后停止；不修改论文、训练或提交 Git。
实际验收：14 项 M2 测试通过，70 个输出及重复/预算/状态/hash 检查通过，退出0。
原生排序改变 q4/q8 原始输出，未来正式比较必须包含第六个 native-order 参考组。
下一阶段尚未启动，详细解释和限制见 `m2/M2_REPORT.md`。

## 目标

在 M1 的严格候选指标下，检验完整检索是否优于 random、last-written、semantic-only 和 recency-only。
M1 证明输出行为改善，尚未证明检索 relevance；这决定 M2 必须先隔离“选了哪些 slot”的作用。

## 1. CPU 协议与策略实现

新增实验专用策略适配层，不改 production bank 默认策略或 disabled 路径。
五策略：原始 semantic×decay、均匀无放回 random、最近写入、cosine-only、last-access recency-only。
最近写入依据 construction insert/refresh 日志派生 last_write_step，不能把 last_retrieved_step 当作 last-write。
recency-only 依据原公式 retrieval age；所有 ties 按冻结 slot index 稳定排序。
为隔离排列顺序，选中集合统一按 slot index 拼接；完整策略也按同一规则执行。

每题先在冻结 bank 的副本上计算原始 P7 threshold/top-k 的选中数量 n（0–2）。
其他四策略各选择 n 个 slot，pool 为同一 bank 的全部存活 slot；不额外应用阈值。
这个 count-matched 设计明确只比较内容选择，不将 random/last 的 score 与 cosine threshold 混用。
n=0 时全部走同一空检索路径；n>0 子集与全量结果分别报告。
记录这属于实验控制，不直接替代论文原方法的排序接口。

验证空 bank、n=0/1/2、slot 不足、score ties、refresh 与 write-order、重复选择和随机独立性。
在已有 P7 轨迹上先计算 score range 和 full/last overlap；只报告可从记录恢复的量，不伪造 latent relevance 标签。

## 2. 保存 bank 与限定 smoke

历史正式 P7 五个 context 的 frozen_bank_path 均为空，不能声称可直接复用历史 latent tensors。
使用当前源码重新构建 bank，固定提交版的配置；每个 seed/context 只构建一次，保存所有 slot
values/keys、retrieval counter、元数据、来源日志、RNG 状态、hash、源码与 checkpoint 身份。
这是新 campaign：所有比较均对照此 campaign 的完整策略，不要求精确复现 dirty-code 的历史数值。

先做 seed42/context0 的 10 题五策略 GPU smoke；每题/策略恢复同一 bank、metadata 和生成 RNG 状态。
random policy 使用按 seed/context/query 派生的独立 Python RNG，不消耗模型 RNG。
CPU provenance/选择标签只用于分析，不给 Weaver 或 Reasoner。
smoke 需通过选中数量一致、latent budget 一致、query writes=0、snapshot 不变和可重复输出的契约。
同一完整策略按新拼接规则与原始顺序做 smoke 对照；只要任一题输出改变，
正式 campaign 必须额外运行第六个 native-order P7 策略，单独报告顺序效应。
五策略内容比较仍统一 slot-index 顺序；native-order 仅作原路径参考，不混入该内容比较。

## 3. 正式比较

smoke 验收后才能规划和启动完整 campaign：5 contexts × 100 questions × 5 bank seeds × 5 policies，
总计 12,500 个 query 输出和 25 次 bank construction；若触发顺序对照则为 15,000 个 query。
批次1，同一 checkpoint/prompt/generation/scorer。
GPU 启动前查实际占用和可用环境；按空闲设备安全并行、detached tmux，日志与输出留在仓库。
控制同 seed/context 内共享 snapshot，每个 query 恢复状态，禁止 policy/question 之间污染。
部分失败保留原日志，按明确 seed/context/policy/query 恢复；禁止只挑完成或最好结果汇总。

## 4. 指标与验收

沿用 M1 官方 EM、raw recall、candidate-valid rate、strict candidate accuracy、条件指标和 heuristic flags。
主决策指标为 paired strict candidate accuracy；另外报告 official EM，不能只看格式。
记录选中 slot IDs、排序、cosine/age/final scores、n、full与last集合重合率、每题 policy RNG 身份。
提供逐 context/seed 差值、胜负数；按 context 聚类汇总，五 context 的不确定性须明确。

## 决策门

- 若完整策略相对 random/last 有稳定的严格正确率优势，下一阶段 M3 检验直接 retrieval relevance。
- 若未建立优势，不将不显著自动写成等价；记录效果方向和不确定性，优先诊断 retrieval/representation，而非扩 benchmark。
- semantic-only、recency-only 的结果用于规划下一步；不会仅凭单次均值就删除 decay 或更改 key space。
- 本文件是下一阶段计划，不授权当前 M1 自动启动上述 GPU 实验。
