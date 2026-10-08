# M3 实施记录

父论文：`review/nference_Time_Latent_Me.pdf`。父证据：M2_REPORT；审稿项 R03/R07。
本阶段已获批准，使用所有满足门槛的 RTX A6000，不限两张。

## 实现与验收

- 正式入口 `scripts/eval/review_m3_formal.py`：prepare/launch/worker/resume/status/score。
- 重模型 worker 单独实现；不修改 M2 的 seed42/context0/10题硬限制。
- 25 个 seed/context jobs，六策略、每 context100题，共15,000个唯一回答。
- 带文件锁领取整 job；每 worker 加载一次模型；逐题完整 JSON 原子发布。
- master 固定源码、模型所有文件、输入、官方 scorer 和时间戳；身份漂移拒绝混合。
- 每策略恢复同一 bank/RNG；零实际写入和冻结状态检查；只允许瞬时 I/O 重试一次。
- 损坏记录保留原始文件；恢复不抢占活进程；已失败契约需诊断，不能自动重试。
- 严格候选准确率为主，配对胜负/25块差值/5 context/5 seed及探索性 block CI 分开报告。
- 共同有效候选条件指标保留分母及输出选择偏差警告；recency 重合及查询间变化单独记录。

CPU 验收：M3 14项集成测试、M1 11项评分测试、M2 14项策略测试全部通过（39项）。
shell 语法及两个新 Python 模块编译检查通过。未安装新依赖。
旧 bank 套件的两个 `dropped_write` 精确字段断言失败仍保留，不改 production/training。
状态与最新计数以后续启动记录为准；未得到15,000条完整记录前，不报告正式效果。

## 操作入口

所有命令在仓库根目录执行；控制器无 torch 依赖，用 MABench Python。
正式 campaign 路径：`outputs/mab/review_m3_formal/20261002-m3-six-policy-v3`。
首次准备的 `20261002-m3-six-policy` 在启动前显存 CSV 解析失败（值带 MiB），
没有启动 GPU worker、构建 bank 或生成回答；原清单保留。修复解析并覆盖单位后缀
测试后，用 v2 身份重新冻结源码，不在旧清单中混入修复后的代码。
v2 的两个 worker 在模块导入期间被主动中断，尚未构建 bank 或生成回答；
补齐模型调用前的持久 attempt ledger 后用 v3 冻结。中断前启动但未保存的调用
不会在恢复时消失；记录的是已启动执行次数，未完成 generation 的中断也保留。
恢复不择优，使用同一冻结 bank/RNG 重做未保存的调用，正式结果披露总尝试数。

```bash
/home/baishilong/miniconda3/envs/MABench/bin/python scripts/eval/review_m3_formal.py status --campaign outputs/mab/review_m3_formal/20261002-m3-six-policy-v3
```

恢复仅在明确中断后使用 `resume` 回收已死亡 worker 的 job，随后在合格 GPU 的 tmux
中启动 shell worker。已成功保存的快照不重构建，已验证完整的逐题回答不重生成。
当25个 job全部生成成功时，最后完成的 worker 调用 MABench 官方评分环境进行汇总。
若手动评分，使用同一入口的 `score`；未完成清单会被拒绝。

本阶段仍不编辑论文，不提交 Git，不自动启动下一阶段。
方向稳定仅决定后续计划，不是显著性或语义检索的证明。

## v3 启动验收快照（非最终结果）

- tmux 启动时间：2026-10-02 17:09:25 Asia/Shanghai。
- GPU0：PID538104；session `memgen-m3-20261002-m3-six-policy-v3-gpu0`。
- GPU6：PID538067；session `memgen-m3-20261002-m3-six-policy-v3-gpu6`。
- 启动前空闲显存分别14,019 MiB、17,613 MiB；其余卡未达到12GiB门槛。
- 启动检查时两进程存活，日志各67字节，仍在模块导入/CUDA初始化；
  GPU6进程已出现在GPU列表（10MiB初始化占用），不是模型已加载的证明。
- 当时25个job均pending，已完成正式回答0/15,000，尚无 `WORKER_READY`。
- 启动资源与身份证据：campaign内 `launch_audit.json`、`launch.json`、`manifest.json`。
- 日志根：`runtime_logs/review_m3_formal/20261002-m3-six-policy-v3/`。
- 这是时间点快照，不是持续状态保证；后续以queue、逐题记录和最终results为准。

正式源码指纹校验通过；M0两份冻结证据的SHA256与原记录相同。
GPU生成和正式效果验收仍待完成，不能将本阶段标记为completed。

随后日志检查：两卡的缓存/CUDA/显存 preflight 均 PASS，已进入模型加载。
仍无 WORKER_READY、正式回答仍0/15,000；qwen2→memgen兼容提示与M2相同，
不将该提示本身认定为运行失败。队列尚未领取job，因为模型加载完成后才领取。

## 继续执行：部分记录完整性核验（2026-10-02 21:08 Asia/Shanghai）

用户要求继续计划；仍限当前M3，未进入下一阶段。
遵循 analysis-campaign 的可比性与部分证据边界，新增无模型、只读核验器
`research_notes/next_version/m3/audit_partial.py`。它位于冻结推理源码清单之外，
不改变worker、队列、模型、快照或回答；核验器自身SHA保存在报告中。

- 审计状态：`PASS_PARTIAL`；固定发布文件列表的202条回答全部通过。
- `s42-c0`：135条回答，22题完成六策略；`s42-c1`：67条，11题完成六策略。
- 202条已启动执行记录对应202条回答，没有已保存回答的重复尝试。
- 源码、模型/配置/数据/scorer全部输入哈希通过；两份冻结bank文件哈希通过。
- 逐条校验完整标识、内容校验和、问题/gold/prompt身份、候选槽数量/latent预算、
  canonical顺序、零实际写入/冻结状态标记，以及持久attempt计数。
- 仍为2个job running、23个pending。没有正式评分，不据此前缀推断效果或显著性。
- GPU6进程的单次状态采样出现`os_acquire_rwlock_read`等待；这不足以单独
  诊断慢速原因。进度从20:51的181条增加到本次202条，确有推进，但吞吐很低。

证据：`outputs/mab/review_m3_formal/20261002-m3-six-policy-v3/audits/continuation-01.json`。
前缀审计不要求未完成六策略的在途题目已经齐全；正式汇总仍必须满足15,000条。
无损坏修复、无重启、无新增GPU smoke。当前路线：继续原队列，结束后执行完整
官方/strict评分与配对分析，再按结果制定下一阶段，不自动扩大或更改协议。

### 同阶段资源调度补充

继续计划期间复查显示GPU7空闲15,374MiB，达到原批准的RTX A6000/12GiB门槛；
拟再次实时核查后为同一v3队列增加一个GPU7 worker。每卡仍一个worker，
按完整seed/context job领取；已运行的GPU0/6不迁移、不重启。
正式清单、模型、源码、指标和15,000唯一回答预算不变，不是新campaign或下一阶段。
额外启动资源证据存为campaign中的`launch_additions/gpu7.json`；
该卡也必须通过worker的独立环境/显存/身份门，进程启动不等于任务完成。

GPU7实时复查仍为空闲15,374MiB，已于21:14:07在
`memgen-m3-20261002-m3-six-policy-v3-gpu7`启动；启动检查PID1348372存活，
日志67字节，尚在导入/初始化，未记录WORKER_READY，不能算作已执行第三个job。
GPU0/6原PID仍存活，未重启、未迁移。随后文件计数209/15,000（约1.39%），
这是新时间点计数；上述完整哈希审计仅涵盖此前固定前缀202条。
进程级GPU单次采样中，本实验两进程SM利用率显示`-`，同卡其他进程有SM活动；
结合GPU6锁等待，资源竞争是可能原因，未进行充分剖析，不能认定为已确诊。
保持原batch1/生成长度/模型协议；运行耗时显著偏长，目前不给可靠结束时间。

## 再次扩容资源核查（2026-10-03 08:02 Asia/Shanghai）

用户明确要求增加GPU并行；本次读取实时GPU状态和原worker信息后，
未使用卡1/2/3/4/5的空闲显存分别为3,018/9,354/8,796/7,808/5,359MiB，
均低于原批准的12GiB启动门槛。GPU4/5利用率较低不意味着模型显存足够。
既有GPU0/6/7的三个PID和tmux会话仍存活，日志继续产出，无异常堆栈/OOM匹配。
GPU7本次余量仅776MiB，现有worker占用12,842MiB，不能再叠加worker。

因此本次新增worker数量为0，仅“扩容”暂受资源限制，M3原队列继续执行。
不改冻结源码、精度、batch、长度或预算，不清理其他任务、不重启现有worker。
后续扩容条件仍为同型号空闲卡>=12GiB，且每卡仅一个本campaign worker；
达到条件后可使用原队列追加，不需要重复已完成的实验。

## 断点恢复（2026-10-08，用户已批准）

进程检查发现原worker/tmux均已消失，队列留下三个running lease。
最后回答写入时间为10月3日18:35；当前主机10月8日10:30启动，不能仅凭
这两个时间断言此前中断原因。已保存14,663条、22个job完成，尚缺337条。

恢复前独立审计`audits/resume-preflight-20261008.json`为PASS_PARTIAL：
全部14,663条记录的身份/内容哈希/配对预算/零写入/attempt计数，以及全部25份
快照和源码/模型/配置/数据/scorer输入哈希通过。八卡均有48,569MiB空闲显存。

恢复调度保持原卡：GPU6→s442-c2（缺64），GPU7→s442-c3（缺47），
GPU0→s442-c4（缺226）。为避免先完成者领取其他卡的旧job，新增
`resume_pinned_worker.py`仅替换带锁领取任务的调度操作，不改变冻结的推理函数。
该辅助调度器在冻结源码清单之外，自身SHA写入recovery记录；两项无模型测试通过，
覆盖乱序领取时的原GPU归属、不可重复领取、STOPPED不领取及最终GENERATED转换。

恢复前保存旧队列、旧job runtime和已完成回答的文件元数据；回收已死亡lease后
仅启动三个worker。加载原可信快照和RNG，跳过完整且校验一致的回答，不再构建bank。
原日志追加，不覆盖；中断但未保存的attempt保留，最终评分披露总尝试数。
只有25个job/15,000条完整记录全部通过后才评分。本节仅记恢复实施，不宣称完成。

## 最终验收（2026-10-08 11:19 Asia/Shanghai，M3完成）

上述恢复已执行完毕：补齐337条，25个job全部done，六策略各2,500条，
共15,000条唯一回答。自动正式评分于11:17:24完成，`results.json`状态COMPLETE；
三个恢复worker均exit0。原14,663份回答的size/mtime均未改变；25个job均仅有
一次construction attempt，冻结快照哈希保持不变，没有重建bank。

独立最终审计`audits/resume-final-20261008.json`验证15,000/15,000条记录、全部
源码/输入/快照哈希、身份配对、预算/顺序及零实际写入约束通过。
注意该通用前缀验证器固定输出PASS_PARTIAL及旧的next_route提示；不将该标签
单独作为正式验收依据，完整性以实际15,000覆盖、25个done和正式评分COMPLETE共同确认。
`audits/recovery-preservation-20261008.json`另保存旧回答未改和construction次数检查。
总计15,003次started attempt：3条中断未保存回答恢复后再次执行，其余没有重试；
未筛选多次答案，旧attempt留存并在结果中披露。

|策略|严格候选正确率|官方EM|候选有效率|
|---|---:|---:|---:|
|full|20.20%|28.40%|69.40%|
|random|3.92%|5.52%|14.72%|
|last_written|1.80%|4.40%|3.20%|
|cosine_only|16.80%|25.40%|57.20%|
|recency_only|20.20%|28.40%|69.40%|
|native_order|11.80%|20.00%|42.80%|

full−random为+16.28个百分点（446胜/39负/2015平），full−last_written为
+18.40个百分点（480胜/20负/2000平）。两项均在5/5个context与5/5个seed为正，
满足预注册的描述性方向稳定门。五context探索性block bootstrap 95%区间分别为
[13.96,19.00]与[15.40,21.20]个百分点，不作统计显著性结论。

重要边界：full与recency_only的2,500组选择集合全部一致，严格正确性全部相同。
full−cosine_only的+3.40个百分点仅集中在1/5个context；因此不能将相对随机/末写
基线的收益归因于语义相关检索，也不能把种子重复当作独立问题扩增。
这仍是当前源码的新campaign，不是历史dirty P7的精确复现。

按原M3规则，下一阶段应先制定relevance-label分析计划，重点解释full/recency等价、
所选槽与问题证据的关系，以及输出有效率差异；尚未启动下一阶段或修改论文。
正式数值与逐项配对分析见campaign内`RESULTS.md`、`results.json`与`scored_records.json`。
