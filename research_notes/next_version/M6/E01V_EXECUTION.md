# M6-E01V 执行契约（结果前冻结，2026-10-08）

用户批准：CPU Query表示诊断；没有模型生成、GPU、E02标注、算法修改或Git提交授权。
父对象M3 `20261002-m3-six-policy-v3`、E01验收；claim C02 / reviewer R04。
analysis_role=claim-boundary/representation diagnostic；paper_role=appendix；
section_id=retrieval；item_id=E01V；selected_outline_ref=未选定；
target_display=表示几何与机械干预表；method=冻结M3；main_or_appendix=appendix。
专用artifact接口及paper_experiment_matrix未发现；回写仅M6证据/总结，不改M0–M5。

## 执行包络与资源门

现有CPU Xeon Gold6330，约632GiB可用RAM、3.1TiB磁盘；使用一个进程，Torch/BLAS
最多4线程，禁用CUDA可见设备且不调用CUDA推理；每轮wall cap30min，RSS<=16GiB，
输出<=100MiB。依赖为现有memgen环境torch/numpy/tokenizers，不联网或安装。

只读检查确认base safetensors中`model.embed_tokens.weight`为BF16 [151936,1536]，
466,747,392字节；使用只读mmap仅复制本轮需要的行，不读取其他Transformer tensor。
`projs.bin`约13.2MiB；只使用reasoner_to_weaver BF16 weight/bias，不运行反向投影。
可信bank文件先核验SHA，weights_only安全白名单反序列化约0.47MiB/份，只使用keys，
不实例化或修改bank，不恢复RNG，不加载Weaver/Reasoner/Trigger模型或调用generate。
两轮新目录`e01v/run-20261008-01`及`-02`，禁止覆盖；E01全部产物保持原样。

## 复现与判定（不得按结果放宽）

- 从M3 environment_contract确定真实base tokenizer，逐题比较base/checkpoint编码及
  E01 tail hashes；长度/尾ID不一致即停止相应重建，不将旧窗口当真实输入。
- 500个原始prompt按batch1完整序列embedding→BF16 linear→最后64位置BF16 mean；
  投影包含bias，与原始算法同序。不将mean-before-projection当原始路径。
- 同题五seed复用提取q，但分别对25个冻结bank核验全部2,500条saved norm、40,000
  saved raw cosines；q精确hash使用原dtype/shape/bytes算法。所有q均保存，标识CPU来源。
- 预设近似容差：norm绝对误差<=0.001、raw cosine绝对误差<=0.001；全部通过才称
  标量一致的CPU近似。只有原hash相同才能称该q与原GPU bitwise一致。
- Full Top-2集合需2,500/2,500相同，才能把本轮选槽反事实与M3选择机制直接对应。
  raw/final完整排序、阈值跨越及ties变化均报告，不以容差抹平临界差异。若gate失败，
  输出PARTIAL，几何与反事实仅描述CPU实现，原始M3数值复现保持unknown；不调整门。

## 三个最小slice

V1 原始表示与几何：500原q，按context每100题QQ cosine矩阵（去对角配对）、
norm、逐维方差、总方差、中心化协方差participation effective rank，以及方差能量/
平均平方norm。SVD在100×1536中心化矩阵计算，不做1536维大矩阵特征分解。
2500数组按job/query明确索引，seed重复不是独立表示样本。无语义“坍塌”成功阈值。

V2 clue-only：同context100题用固定SHA循环donor（调用E01 donors，job_id=
`E01V-c{context}`），替换原prompt的question_clue字符区段，候选/答案指令/chat不变。
完整编码后必须last64 IDs不变；比较q exact equality、cosine/最大坐标差、raw/final
ranking及Full集合。预期q恒等是结构预测，不是答案不受clue影响的预测。

V3 candidate-tail-only：以原完整token序列为底，仅将E01归类的17个candidate_list
尾位置换为同donor相应17个IDs；原clue、候选更早部分、mixed边界、指令、chat及长度
全部保持。明确这是token-level机械反事实，不保证可读/语义相关或自然问题分布。
比较q、score、排名及Full集合敏感性；不作为相关memory或答案正确性检验。

Precision辅助slice：同500题比较BF16原顺序、同BF16权重转FP32计算、以及仅诊断用
BF16 mean-before-linear；比较q/score/选槽与边界，绝不替换算法或选择最有利实现。
FP32与pool-first均是数值对照，不是原始GPU向量。

每slice固定权重、bank、slot次序、alpha=.05、threshold=.05、Top-K=2；无答案或gold
作输入，不使用后验结果挑选题目/donor。自然语言干预和token干预可比性边界分开。

## 验收、失败与停止

先跑新单元测试；两轮core byte hashes应一致（耗时/资源元数据除外），父JSON/
源码/权重/snapshot hash前后相同，E01只读验收/M5旧验收继续通过。
资源、输入身份、token重建失败即停止，不修旧产物，不自动切换GPU。CPU精度门
失败留PARTIAL，不将近似几何伪称原q。描述性分析无EM/MEOI成功宣布或IID显著性。
最终记录slice证据、claim更新、可比性与next_route=停止并请求下一项批准。
不证明semantic relevance、latent retention、EM损失或Gate收益。
