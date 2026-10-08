# 事件证据标注说明（独立阶段）

你收到的包只有这份说明、questions.jsonl、五份contexts全文、annotations_blank.jsonl。
不要读取协调员材料或另一位标注者结果；不使用语言模型生成标签，不与其他标注者
讨论本批题目。包内没有数据集gold身份或模型输出。

## 工作流程

1. 用独立工作副本填写annotations_blank.jsonl，保留原始空表。填写rater_id、
   prior_exposure（接触过本研究结果则说明）和elapsed_minutes。候选编号从0开始。
2. 阅读完整原文及题目。先自行判断哪项事件紧随题目线索发生，再在原文定位支持
   该判断的证据。不能只凭候选合理性、常识或记忆中的同名作品作判断。
3. 保存最小充分证据集合。需要多段文本共同说明人物、事件和时序时，放在同一
   evidence_set；不同可替代充分集合分别编号，不能把其中任意单段当作全部证据。
4. 完成所有10题后冻结本阶段文件（协调员记录SHA与时间），此后才由协调员提供
   数据集答案审核材料。第二阶段另填审核表，不回写第一阶段以迎合gold。

## 状态

- supported：能给出至少一个充分证据集合，并明确唯一候选。
- ambiguous：原文支持多个候选或时序存在不可消解歧义；列出可能候选及理由。
- not_locatable：已经搜索但未能定位；不能等同原文不存在证据。
- conflict：原文与题目线索或候选叙述冲突；引用冲突证据。
- insufficient_context：判断依赖未提供前文/信息。
- prior_only：只能依据常识或候选先验判断，未找到原文支持。

不要删除难题、换样本、把未知写成“不相关”。未能定位时仍填写status和说明。
正式提交不能保留status=null；null仅表示空表尚未标注。

## Span与必要事实

start/end使用**原context的Unicode码点索引**，左闭右开 `[start,end)`；text必须与
`context[start:end]`逐字一致。空格、换行也占位置，不能清理或改写原文。
不是UTF-8字节索引，也不是浏览器JavaScript的UTF-16 code-unit索引；遇emoji等
非BMP字符必须转换。坐标辅助检查只能验证字符串匹配，不能代替语义判断。

每个evidence_set包含：

- set_id：本题内唯一字符串。
- sufficiency：sufficient / partial / unknown。
- spans：每段填写start、end、text、role（anchor_event / supporting / target_event）、
  reason；明确为何能支持人物对应、下一事件、时序或共指。
- required_facts：用简短原子事实说明这个集合需要联合满足什么，不复制候选作证据。
- rationale：为何这些段落联合足够；证据缺口也必须说明。

若只发现目标事件，但不能把它与题目已有事件建立顺序联系，不应直接标sufficient。
相邻叙述是否提供顺序依据需结合原文解释。全文不是默认“最小充分span”。

## 提交与边界

填写chosen_candidate_indices、evidence_sets、notes、completed_at等。任何label mismatch
只能在独立阶段冻结之后登记于第二阶段审核表。协调员之后才映射来源chunk和slot，
你不需要判断latent内容，也不需要猜当前检索会选择什么。

本pilot判断标注是否可行，不评价模型正确率。不能从“source包含证据”推出“latent
保存了证据”。人员、培训、时间预算和验收门槛须由协调员确认后再开始正式工作。
