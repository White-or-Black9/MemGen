# E01V执行补记：输出序列化修复

首轮`e01v/run-20261008-01`完成500向量提取和25bank计算后，在写query_metrics时
遇到`TypeError: Object of type int64 is not JSON serializable`。status=FAILED，
保留该目录及未验收的queries.npz/不完整JSON；不得引用为验收结果。

原因：诊断脚本用numpy.exp生成分数，threshold布尔及sum转为numpy.int64。
修复：改用与原检索公式相同的math.exp，确保JSON字段为原生Python类型，并新增
输出序列化测试。未修改模型或父数据；未放宽容差、未改变题目/donor/干预设计。

验收运行改为新目录`run-20261008-02`与`run-20261008-03`；首轮失败不覆盖、不删除。
本补记是实现修复与路径变更，不是根据机制结果修改假设、成功标准或资源预算。
