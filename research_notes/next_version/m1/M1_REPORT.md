# M1 验收与结论（2026-10-02）

## 验收

**PASS_WITH_LIMITATIONS**。冻结的九种方法、五次重复，共 22,500 条记录全部重新调用
官方 MemoryAgentBench scorer；原 substring EM、raw recall 和 P7/no-decay 已存 parsed output 无差异。
输入的 435 个文件 hash 已核验。500 题候选提取无异常：均为六个唯一候选，理论 chance 为 1/6。
新增 analyzer 使用现有 MABench 环境，不加载 MemGen，不生成新答案。

11 项测试覆盖 parser/格式分离、空分母、严格候选、substring 命中、共同有效集合、
问题身份冲突、安全候选提取、重复/normalization 冲突及多 gold。
M0 的历史 dirty-code 和 control seed provenance 限制仍然保留。
Recent-text 与两项注入消融只有 context ID/query index 可供身份核对；其他文本对照还核对 official query hash。

## 核心发现

| 指标 | P7 | Matched-16 | 含义 |
|---|---:|---:|---|
| 官方 substring EM | 18.80% | 6.80% | 原论文主指标，逐题一致 |
| Raw recall | 23.08% | 18.00% | 原始文本提及 gold，不等于检索到了证据 |
| Parser 非空率 | 100% | 100% | EM 差异不是来自是否提取到文本 |
| 唯一候选有效率 | 37.64% | 1.80% | P7 大幅改善合法候选形式；未作统计显著性检验 |
| 严格候选正确率 | 11.00% | 0.80% | 完整匹配正确候选的全体题目比例 |
| 有效候选条件正确率 | 28.67% | 44.44% | 不同条件子集，不可直接归因于语义质量 |
| 旧 heuristic flags | 23.72% | 69.40% | 不是 parser failure，也不等于候选无效率 |

P7 的 11% strict candidate accuracy 低于始终合法输出的均匀猜测参考 16.67%。
该 chance 是数学参考，不是与模型同解码协议的实测对照。
其有效候选条件正确率高于这个参考，但只覆盖约 37.6% 的输出；不能据此宣称整体超过 chance。

P7 与 Matched-16 的 Δ官方 EM = 0.120 = Δ严格候选正确率 0.102 + Δ仅 substring 命中率 0.018。
这个互斥类别的记账分解说明增益中存在完整正确候选输出的增加，但不证明由相关历史检索造成。
两者都能 parse 出文本，reviewer 按 `500 - heuristic_flags` 推算的 parsed-only accuracy 分母不成立。

共同 candidate-valid 的题目平均只有 3.4/500，P7 与 Matched-16 的平均条件正确率为
20.67% 与 24.67%；集合太小且由两种系统选择，不能作为条件选择质量优劣的可靠结论。
完整的逐 seed、逐 context、共同 parsed 集合和配对胜负统计在 `summary.json`。

## 其他边界

- No-decay strict candidate accuracy 为 11.72%，高于 P7 的 11.00%；raw recall 25.08% 也更高，
  但官方 EM 16.28% 较低。当前不能把 alpha 的 EM 优势写成准确答案选择改善。
- Dense raw recall 24.00% 高于 P7，但 strict candidate accuracy 为 0%；提及事件和选择事件必须分开。
- P7 每次 500 题平均 55 个正确有效选择、60.4 个提到 gold 但没有有效选择、384.6 个未提到 gold。
- 本阶段只做描述性分析，没有把五个重复 seed 当成 2,500 个独立问题进行显著性检验。

## 允许与不允许的当前表述

允许：在固定 EventQA 协议下，P7 提高官方 substring EM、有效候选输出率和全题严格候选正确率。
允许：Weaver-mediated 路径比所检查的直接注入和 no-conditioning 路径产生更多正确有效候选。
不支持：这些提升已经证明 semantic retrieval 找到了相关历史证据。
不支持：整体严格答案选择超过随机候选参考，或 temporal decay 改善了条件选择准确率。
需要修订：将论文“parser-format failures”明确改称实际 heuristic format flags，并补充真实解析及候选有效性指标。

## 下一阶段

M2 优先检验检索是否优于 random/last-written，不能先扩大 benchmark 或直接进入 key-space redesign。
执行规格见 `../M2_PLAN.md`。M2 尚未实施，本阶段没有 GPU 推理或自动 Git 提交。
