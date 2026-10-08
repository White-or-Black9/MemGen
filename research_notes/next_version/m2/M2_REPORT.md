# M2：CPU 协议实现与限定 GPU 验证

## 证据问题与阶段边界

关联 `REVIEW_ISSUES.md` 的 R03（检索相对 random/last-written）和 R07（decay）。
父证据为 M1 已核验结果。本阶段为协议验证，不能用 10 题均值证明检索有效。
唯一论文锚点仍为 `review/nference_Time_Latent_Me.pdf`。
不运行完整五种子 campaign，不修改训练和 production bank 默认行为，不提交 Git。

## 实现

- 五策略 full/random/last_written/cosine_only/recency_only；原生 P7 阈值/top-k 决定 n，
  其他策略从全部存活 slot 取同样 n；n=0 均为空。统一 slot-index 顺序，native-order 独立参考。
- last-write 来源是实际 construction insert/replace/eviction 日志，不用 last-access 替代。
- 保存所有 bank 私有状态、keys/values、计数器、元数据及 Python/NumPy/Torch/CUDA RNG，
  校验 bank hash 与来源，支持本地可信快照重放；不接受第三方 pickle。
- 每题/策略使用独立 bank 副本并恢复相同生成 RNG；random 使用独立派生 RNG。
- 只读 query、冻结快照、同数量/latent budget、单次 query retrieval、重复输出和排序对照均为硬门。
- 沿用提交版参数：top-k2、capacity16、retrieve0.05/update0.10、alpha0.05、slot8、
  chunk4096、batch1、generation40、默认 unconstrained prompt。
- 输出 source/input hashes、源码 diff、runtime、construction 日志、bank 快照、每题选择明细，
  在 MABench 环境调用 M1 的真实官方 scorer 和严格候选指标。

## 验收状态

CPU gate 已通过：14 项 M2 测试、11 项 M1 scorer 测试通过；bash 语法、Python 编译和 git diff --check 均通过。
现有 `test_latent_memory_bank.py` 共 51 项，49 项通过、2 项失败：
`test_thread_update_debug_fields_complete` 和 `_with_split_thresholds` 的严格字段集合
未包含 HEAD 已存在的 `dropped_write`。生产 bank 和该测试文件均无本次修改，
属于既有断言陈旧问题；保留失败，不在本阶段修改。与 M2 新增策略/隔离契约无关。
缓存/数据/config/CUDA preflight PASS：无缺失文件；MemGen 环境 torch2.12.0+cu126。
限定 smoke 已完成：2026-10-02T04:33:51Z（UTC），进程退出码 0，运行 gate PASS。
阶段结论为 **PASS_WITH_LIMITATIONS**：协议成立，但只有一个 context/seed 的前 10 题，
不支持检索策略优劣的统计结论；两项既有 bank 断言失败仍保留。
后台启动：GPU6，tmux `memgen-m2-smoke-20261002`，初始 PID3801639，
run id `20261002-m2-seed42-ctx0`。输出根目录
`outputs/mab/review_m2_smoke/20261002-m2-seed42-ctx0`，日志
`runtime_logs/review_m2_smoke/20261002-m2-seed42-ctx0/run.log`。
已独立复核 70 份逐题 artifact、7 策略各 10 题、50 个主要回答、成对问题/答案/查询身份、
选取数量和 latent budget、实际写入为零、快照不变及 10 次 full 重复。
全部题目 n=2，检索预算为 16 latent tokens；冻结库 16 slots，construction 共 16 次 insert、
1 次 matched replacement。source/input/snapshot/scorer hashes 全部与 manifest 一致，
无候选提取异常。M0 两份权威证据 hash 保持不变。
隔离环境内 NVIDIA 驱动不可见；隔离环境外只读检查确认 RTX A6000/驱动560.35.03 正常。
各卡已有其他用户任务，运行前必须确认显存余量；不终止或改变其他进程。
历史同路径 cost artifact 的已分配 peak 约 9.43 GB（不作为新运行峰值承诺）。

## 结果（仅 smoke，不是正式效果实验）

| 策略 | 官方 EM | 严格候选正确数 | 有效候选数 |
|---|---:|---:|---:|
| full，统一 slot-index 排列 | 4/10 | 3/10 | 6/10 |
| random | 1/10 | 1/10 | 2/10 |
| last_written | 0/10 | 0/10 | 0/10 |
| cosine_only | 1/10 | 1/10 | 4/10 |
| recency_only | 4/10 | 3/10 | 6/10 |
| native_order，原路径顺序参考 | 4/10 | 3/10 | 7/10 |

full_repeat 与 full 的选择明细和原始输出逐题完全一致。
full 相对 random/cosine_only 的严格配对胜/负为 2/0，相对 last_written 为 3/0；
这些数量只描述本次 10 题，未做显著性或总体优势推断。

full 与 recency_only 在全部 10 题均选 `[0,1]`，原始输出也完全一致。
random 与 last_written 在全部 10 题改变了选中集合，cosine_only 改变了 8 题。
因此本 smoke 的对照确实改变了内容，但 full 与 recency 在此样本无法区分。
不得据此声称 full 在其他 contexts/seeds 必然等价于 recency，也不能把 full 的相对增益
归因于 semantic relevance。last_written 候选有效率为零，也不能把其差异全当成语义选择质量。

native-order 改变了 query indices 4、8 的原始答案，但这两题未改变严格正确与否，
整体 strict 和 EM 数量不变，candidate-valid 数量由 6 增至 7。
按预设条件，正式 campaign 必须增加 native-order 第六组；不能因主指标不变而省略。

主结果与完整配对/条件指标：
`outputs/mab/review_m2_smoke/20261002-m2-seed42-ctx0/RESULTS.md`、`results.json`。
这只是当前源码的新 campaign，不能替代历史 dirty-code P7 的精确复现。
共享 GPU 和慢初始化期间的耗时不是论文效率证据。

## 预设下一步

验收通过；停止本阶段。下一阶段应制定六组、五种子、五 contexts 的正式比较计划，
总计 15,000 个 query 输出，按 context/seed 报告 paired strict accuracy 与不确定性。
正式比较需要检查 full/recency 的集合重合和 score/age 范围，不能仅比较一个全局均值。
该正式 campaign 尚未启动，须获得下一阶段批准。
即使 smoke 分数出现差异，也不据此删除 decay、修改 key space 或扩展 benchmark。
