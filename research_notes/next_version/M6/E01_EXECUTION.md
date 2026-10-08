# M6-E01执行契约（结果计算前冻结）

用户2026-10-08批准E01实现与执行；不包含E01V、E02标注、E03推理或算法改动。
父campaign固定`20261002-m3-six-policy-v3`，M4-B1为来源；独立输出`M6/e01/run-20261008-01`
及`run-20261008-02`复跑，不覆盖父结果。代码/测试均在M6目录，不改冻结源码根。

主诊断：同bank的Query变化是否改变raw/final排名与full Top-2集合；这是描述性机制
审计，无语义标签，不设事后效应门、不作显著性/semantic retrieval成功判定。

- 每bank100query；2500 seed/query记录并非IID；按25bank及五context分列。
- donor规则：对同bank query ID按SHA256(`M6-E01-v1:{job}:{query}`)排序，循环后移
  一位，产生无自配一一置换；仅一次固定置换，不搜索显著或最大的变化。
- score对比保留bf16原值/ties；raw/final排名的Spearman使用midrank，Kendall tau-b
  分别处理单侧ties，全并列导致分母0时返回null而非伪造1。
- 比较同题raw vs final；原题 vs donor分别比较raw/final。ranking_changed指稳定index
  排序不同，不等于统计上大差异；另报Top-2集合变化、top1/2与top2/3 margin。
- 复用五策略及native-order全部15,000条selection trace，核验配对、内容checksum及
  M4-A parent JSON hash索引；不重新评分/生成答案。permutation不生成新答案。
- cosine-only取整个bank的相同n，无full阈值；recency index tie-break；random用原
  identity私有seed，不搜索随机对照。slot频率、组合及full overlap单独报告。
- tokenizer使用冻结checkpoint的本地tokenizer.json，tokenizers-only，无transformers/
  Torch/model。add_special_tokens=False因为prompt已render；encoded length必须与
  保存input_len一致，否则标记该题无法确认真实窗口，不将估算当真实输入。
- token窗口按字符offset划分：问题线索、候选标题、候选列表、答案指令、chat尾模板、
  prefix模板、跨边界mixed。字符区段划分为结构标签，不是人工语义相关性标签。
  pool最后64token；保留offset/边界混合与每题类别计数；不改变token顺序或query。
- q norm有值即可分析分布；不从hash/norm恢复完整q，不计算缺数据的QQ cosine。
- 原始source metadata只用来检查context是否来自不同作品；来源名不同不自动证明
  无文本重叠或统计完全独立，结果只做描述。

资源：单CPU进程，建议RAM<=4GiB、结果<=100MiB、每轮wall cap30min；两次运行核心
产物要求字节相同，输入hash与inventory前后不变。依赖不可用时窗口部分明确blocked，
不安装包、不加载LM。hash/identity/预算/配对漂移即失败，不修父文件或自动重试。
验收新增单元测试、M3/M1与M4纯CPU回归、复跑core hashes及M5旧验收仍通过。
next_route=报告E01结果并请求下一项授权，不自动进入E01V/E02/E03。
