# E03 pilot后台启动记录（不是完成报告）

2026-10-08约18:06北京时间。用户批准的40次生成已交给独立后台worker。
使用analysis-campaign固定配对与停止门，使用detached-launch后台及启动核验流程。
仅本E03切片，不自动进入50题或E04。执行契约见[E03_EXECUTION](E03_EXECUTION.md)。

- tmux：`memgen-m6-e03-pilot-20261008-01`，pane/worker PID `2160354`。
- GPU0，RTX A6000，单卡可见、自动串行on0/off0/on1/off1。
- 启动前GPU0约36,277MiB空闲，其他用户任务仍在；不是独占GPU。
- worker preflight PASS，CUDA可用、路径齐全、空闲显存36,998,742,016字节。
- 首次启动快照：该PID已在GPU0占用约9,662MiB，日志从空增长到1,278字节，
  状态PREFLIGHT/model loading，尚无完成答案。本记录不推断最终分数或完成时间。
- 11项实验驱动测试通过；M3源码/输入、五snapshot和10题父记录在prepare阶段核验。
  M5 validator仍PASS；无模型/生产源码/论文修改，无commit/push。

启动确认后的最近一次交接快照：状态RUNNING，已保存27/40次有效生成，正在
c3-01/off/repeat1；已完成部分尚未触发检索预算、写入、重复稳定性检查错误。
日志已包含QUERY_COMPLETE，确认不仅是模型加载。该计数不是最终验收或效果结论。

输出根：[pilot-20261008-01](e03/pilot-20261008-01/manifest.json)；
[状态](e03/pilot-20261008-01/status.json)、[日志](e03/pilot-20261008-01/run.log)。
全部40次完成且检查通过后自动调用MABench评分，写results.json并停止。
失败会写STOPPED/error/traceback，不自动重启或选择较好重复。
120分钟alarm和100MiB输出限制为停止界，不是ETA；当前共享设备无法可靠估时。
启动核验后不持续轮询；用户要求进度时再读取状态、日志和产物。
