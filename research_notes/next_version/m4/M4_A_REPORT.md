# M4-A：来源追溯与证据定位审计

日期：2026-10-08。状态：COMPLETE_WITH_LIMITATIONS；两次全量审计通过，M4-A完成。
父证据：M3 `20261002-m3-six-policy-v3`；提交论文仍为review目录下的原PDF。
审稿映射：R03/R04/R07；paper_role=supporting，section_id=retrieval_diagnostics，
exp_id/item_id=M4-A，display=appendix来源覆盖表。

## 结果与主张边界

|检查对象|结果|可以说明什么|
|---|---:|---|
|父记录完整性|15,000/15,000通过|M3回答及身份/配对/预算/零写入约束有效|
|构建任务|25/25通过|构建日志可按顺序回放|
|存活slot当前版本|400/400重建来源|当前版本直接输入chunk和可能的历史依赖可追溯|
|构建写入事件|425|400次insert、10次replace_matched、15次evict_oldest_insert|
|唯一问题|500|独立问题分母，不是2,500个独立样本|
|已核验证据原文位置|0/500|当前输入没有直接提供经核验的source span|
|gold事件概述逐字匹配原文|0/500|词面定位方法不可直接使用，不代表证据不存在|
|seed/question关联|2,500全部unknown|没有把未定位题目当作负例|
|latent语义相关性|未知|未解码latent、未验证其证据保留能力|

主要发现不是“检索不相关”，而是两条不同证据链的可识别性不同：
slot←构建输入来源可以从冻结代码/配置及日志重建；question→原文证据→latent内容
不能仅由已有日志直接确立。当前收益仍不能归因于语义检索。

## 来源重建依据

实际执行路径为原始MAB chunker→顺序memorization prompts→CompressedManager→
prompt augmentation→Weaver生成→bank write_back。

- 官方chunker使用NLTK句子边界与tiktoken，而不是Reasoner tokenizer硬切。
  五个context共85块重建后，逐块hash与25份context identity全部一致。
- CompressedManager每轮只呈现system和当前user observation；没有累计文本前缀。
- 冻结配置max_inference_aug_num=0、trigger.active=false；prompt阶段固定调用Weaver，
  后续生成不再augmentation。结合顺序environment与逐轮write count，构建write turn
  对应当前chunk，不仅凭“两个列表长度相等”做映射。
- 这条映射是冻结源码/配置推导并由日志计数验证，旧日志没有独立保存每次write的
  prompt hash或原文span；不宣称存在比实际更强的运行时来源证据。
- refresh是用新latent替换匹配槽，不是自动合并旧语义；eviction也在原索引替换。
  分开保存slot_version、thread_id、previous_content_version和被检索版本。
- 历史检索依赖闭包仅表示可能参与构建的来源。即使旧slot被Weaver读取，也不能
  认定其全部事件被新latent保留，更不能据闭包直接计算semantic Recall/MRR。

回放核对最终slot数、created_step、last-retrieved/access计数、age，及全部15,000条
query trace的last-write数组。400个最终槽均有一致的来源版本信息。

## 证据定位为何仍未知

原始parquet每行仅有context/questions/answers/metadata。
EventQA非空metadata主要是source、qa_pair_ids和previous_events；question_dates、
question_ids、question_types及其他来源字段为空。qa_pair_id标识问答，previous_events
是已发生事件概述，都不是小说原文的offset或经核验source-event链接。

分析遍历全部500题并校验原始question/gold与M3身份；按所有合法gold进行大小写
不敏感的完整短语匹配，原文中均未发现逐字匹配。gold是改写事件，不应据此报告
“无证据”或“检索失败”。没有实施模糊相似度自动贴标签或根据答案正确性倒推相关性。

两层标签均采用nullable字段：source_evidence_overlap与latent_semantic_relevance。
所有未知保留原因和分母。来源没有定位时，既不填false，也不把未知slot当作不相关。
冻结SHA排序每context十题的50题blind bundle仅作为可复现的后续检查入口；
本轮没有人工语义标注，不声称双人一致性或人工核验过50题。

## 实现、验证与运行

实现：`audit_provenance.py`；测试：`test_audit_provenance.py`，均在M3冻结源码根之外。
不加载Torch、checkpoint或pickle，不启动GPU，无网络下载/依赖安装。

- 12项新CPU测试通过：索引复用、refresh依赖、缺失turn、摘要漂移、chunk hash、
  原文offset、跨chunk、多gold/多匹配、未知非负例、固定样本和父文件变更检测。
- 25项M3 controller/M1 scorer回归测试通过。
- 首次尝试顺带运行旧M2策略测试时，当前MABench环境无Torch，模块导入失败。
  本轮未安装Torch、未宣称这14项重新通过；原M2/M3验收的通过记录保持历史身份。
- module compilation及git diff --check通过。全量首轮完成状态已写入status.json；
  coverage中的elapsed_seconds记录核心审计时长，不含最后全量JSON哈希清单的耗时，
  不用作模型/系统效率数字。
- 父目录30,192份文件的集合、大小、mtime在分析前后保持不变；父输入/源码/模型
  hash和25份bank文件hash由只读M3 validator验证。本轮未重建bank或改写回答。

产物保留在`run-20261008-a1/`；复跑在独立`run-20261008-a2/`，不覆盖第一轮。
两次七份核心产物逐字节hash一致：provenance、evidence_mapping、seed/question
associations、chunk_offsets、blind_sample、lifecycle及schema_audit。
coverage除运行时长外一致；30,162份父JSON的内容hash清单一致，分析器hash一致。
验收凭据`acceptance.json`为PASS；两轮产物各约6MB，总量远低于1GiB预算。
manifest记录父JSON文件hash、分析器hash、外部chunkerhash、静态源码依据、rubric、
环境及论文映射。parent_integrity沿用旧validator的PASS_PARTIAL字面标签，但其
实际验证数为15,000、queue为25 done，并与正式COMPLETE结果联合核验。

## 下一路线与停止条件

M4-A的结果支持来源可追溯性，不支持semantic retrieval主张。
下一阶段计划已写入`../M4_B_PLAN.md`，尚未执行。优先M4-B1已有trace诊断，重点是full/recency等价、last-access与
last-write的差别以及输出有效性转移；暂不计算没有可信标签的相关性Recall/MRR。
若需要直接语义证据，另设原文span标注与latent保留验证协议，不能仅补构建日志
就声称相关性标签可得。M4-A结束后停止，不自动新建GPU实验、重设计或修改论文。
