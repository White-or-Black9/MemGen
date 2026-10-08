# M1：官方评分、候选选择与格式分解

Gate: **PASS_WITH_LIMITATIONS**。九种方法，五次重复；500 个唯一问题，不是 2,500 个独立样本。

条件指标按每次运行计算再取均值；完整分子/分母、逐 context 和配对统计见 summary.json。

| Method | Official EM | Raw recall | Parsed nonempty | Candidate valid | Strict candidate accuracy | Parsed conditional EM | Valid conditional strict accuracy | Heuristic flags |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p7 | 0.1880 | 0.2308 | 1.0000 | 0.3764 | 0.1100 | 0.1880 | 0.2867 | 0.2372 |
| recent_text | 0.0340 | 0.0960 | 0.9980 | 0.0060 | 0.0020 | 0.0341 | 0.3333 | 0.4220 |
| rolling_summary | 0.0120 | 0.0780 | 1.0000 | 0.0020 | 0.0000 | 0.0120 | 0.0000 | 0.5340 |
| bm25_top2 | 0.0300 | 0.2260 | 1.0000 | 0.0040 | 0.0020 | 0.0300 | 0.5000 | 0.5300 |
| dense_top2 | 0.0300 | 0.2400 | 1.0000 | 0.0020 | 0.0000 | 0.0300 | 0.0000 | 0.5700 |
| matched16 | 0.0680 | 0.1800 | 1.0000 | 0.0180 | 0.0080 | 0.0680 | 0.4444 | 0.6940 |
| no_retrieved_memory_conditioning | 0.0080 | 0.1780 | 1.0000 | 0.0020 | 0.0000 | 0.0080 | 0.0000 | 0.7540 |
| direct_top1 | 0.0472 | 0.1864 | 0.9984 | 0.0796 | 0.0340 | 0.0473 | 0.4283 | 0.7092 |
| no_decay | 0.1628 | 0.2508 | 1.0000 | 0.3608 | 0.1172 | 0.1628 | 0.3160 | 0.3780 |

Candidate chance：0.1667，有效题数 500/500。异常题及重复/多 gold 见 summary.json。这是始终输出合法候选的均匀选择理论参考，不是实测同协议模型 baseline。

## 错误分解（每次 500 题的平均数量）

| Method | Correct valid | Gold mentioned, wrong valid | Gold mentioned, no valid | Gold absent |
|---|---:|---:|---:|---:|
| p7 | 55.0 | 0.0 | 60.4 | 384.6 |
| recent_text | 1.0 | 0.0 | 47.0 | 452.0 |
| rolling_summary | 0.0 | 0.0 | 39.0 | 461.0 |
| bm25_top2 | 1.0 | 0.0 | 112.0 | 387.0 |
| dense_top2 | 0.0 | 0.0 | 120.0 | 380.0 |
| matched16 | 4.0 | 0.0 | 86.0 | 410.0 |
| no_retrieved_memory_conditioning | 0.0 | 0.0 | 89.0 | 411.0 |
| direct_top1 | 17.0 | 0.0 | 76.2 | 406.8 |
| no_decay | 58.6 | 0.0 | 66.8 | 374.6 |

## P7 与对照的共同有效题目

每个 seed 独立配对。下面是五次配对的平均分母及准确率，不作显著性检验。

| Control | Both candidate-valid n | P7 strict acc | Control strict acc | P7 wins / losses |
|---|---:|---:|---:|---:|
| recent_text | 1.8 | 0.0000 | 0.4000 | 0.0 / 0.8 |
| rolling_summary | 0 | null | null | 0 / 0 |
| bm25_top2 | 0 | null | null | 0 / 0 |
| dense_top2 | 0 | null | null | 0 / 0 |
| matched16 | 3.4 | 0.2067 | 0.2467 | 0.0 / 0.2 |
| no_retrieved_memory_conditioning | 0.8 | 0.0000 | 0.0000 | 0.0 / 0.0 |
| direct_top1 | 16.8 | 0.3633 | 0.3956 | 0.6 / 0.6 |
| no_decay | 88.6 | 0.2994 | 0.2977 | 2.8 / 2.2 |

## P7−Matched16 的 EM 记账分解

平均 ΔEM = 0.1200 = Δ严格候选正确率 0.1020 + Δ仅 substring 命中率 0.0180。
这是互斥正确输出类别的算术分解，不是检索机制或格式变化的因果归因。

## 解释边界

- 官方 EM 是 substring EM；strict candidate accuracy 是新增诊断。
- parser 能提取非空文本不代表答案是合法候选。heuristic flags 不是 parser failure。
- raw recall 按官方多 gold 规则；分类中的 gold mentioned 指任一 gold，二者分别保留。
- 共同有效集合仍是条件子集，不自动证明历史证据检索相关或公平控制了全部难度。
- 种子目录不保证 controls 的实际随机性；沿用 M0 的历史来源限制。
- 格式和有效选择的关联不能解释为因果。retrieval utility 待 M2–M4 验证。
