# E04-X后台启动快照（不是实验完成报告）

2026-10-08约19:12北京时间。执行契约见[E04_EXECUTION](E04_EXECUTION.md)。
用户批准六题三组探索性内容替换；仅本pilot，不自动扩样或启动语义negative实验。

- tmux：`memgen-m6-e04-pilot-20261008-01`，worker PID `2542702`。
- 固定GPU5（RTX A6000），CUDA仅可见一张卡，新调用自动串行。
- 启动前约44,568MiB空闲，但利用率瞬时升到93%；非独占卡，不干涉其他进程。
- worker preflight PASS：输入路径齐全，CUDA可用，空闲显存46,453,882,880字节。
- 启动核验时tmux/pid存在，该PID GPU显存260MiB，正在加载模型，日志已增长至
  1,278字节，状态PREFLIGHT、completed_generations=0。不是已生成18个答案。
- 11项选择/配对规则测试与5项CPU tensor适配器测试通过（隐藏CUDA，无GPU测试调用）。
  prepare核验E03重复/旧on一致性、M3源码/模型输入与snapshot；M5 validator仍PASS。
- A Original和D Random各六个主答案复用旧M3；新增最多18次（A重放6，B双重复12）。
  主分析仍为六题×三组18个答案，B重复不是额外独立样本。

输出：[manifest](e04/pilot-20261008-01/manifest.json)、
[status](e04/pilot-20261008-01/status.json)、[日志](e04/pilot-20261008-01/run.log)。
完成后自动评分写results.json，失败写STOPPED及traceback，不自动重试。
独立只读复算命令（仅完成后运行，不启动模型）：

```bash
/home/baishilong/miniconda3/envs/MABench/bin/python research_notes/next_version/M6/e04/validate_content.py --output research_notes/next_version/M6/e04/pilot-20261008-01
```

60分钟/100MiB是安全停止界，不是预计耗时。使用analysis-campaign固定可比性、
detached-launch核查tmux/PID/显存/日志/输出后停止持续监控；状态需后续读取确认。
没有修改算法/训练/模型参数/论文/M0–M5，无commit/push。
E02独立人工门仍未通过，无semantic-negative，不将来源候选替换宣布为事实利用证明。
