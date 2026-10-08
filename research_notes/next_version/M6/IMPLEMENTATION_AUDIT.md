# M6 — Retrieval / Regeneration implementation audit

2026-10-08；只读审计。当前分支 `review`，HEAD
`6784a26e65290e91be2ad8a965cf108cb19c7b22`。进入时工作区干净。
用户已选M5路线B；M5历史decision中的null保留，不回写旧材料。
代码、报告位置均相对仓库根；行号绑定此HEAD，不代表历史dirty P7代码完全相同。

## E01后续更新

E01V进一步验证：500个CPU重建q在保持尾64 IDs、替换前部clue时全部完全不变，
仅换候选尾17tokens时全部变化；Full Top-2仍不变。实际base/checkpoint tokenizer
500题逐ID一致，数值近似/Full选择复现门通过，但原GPU hash只匹配670/2500记录。
这支持当前初始prompt路径的clue盲点，不证明EM损失。见[E01V_REPORT](E01V_REPORT.md)。

CPU tokenizer-only重建进一步定位last64窗口：500题question-clue均0token，
固定指令41、chat5、候选尾部17、mixed1。逐位置输入embedding/projection不通过
attention带入更早clue；这是当前prompt路径的局限，不代表Weaver/Reasoner没看到
完整query。完整范围与输入ID重建限制见[E01_REPORT](E01_REPORT.md)。未修改模型。

## 1. 信任锚点与核验范围

- 父campaign：`outputs/mab/review_m3_formal/20261002-m3-six-policy-v3/`。
  manifest contract SHA：`a548306d4131e4df0cba1340d3272a4882feae1d9c993c881e281f4551e433c9`。
- M5已有validator本次再次PASS：提交PDF身份、M5 claim/review覆盖、M4三份
  PASS凭据与两次运行共32份核心文件hash、M3/B2数值一致。
- 额外只读检查：M3清单92份源码hash一致；25份snapshot文件hash与各自
  snapshot_identity一致；queue为25 done；六策略各2,500原始记录路径存在；
  scored_records.json包含15,000条。存在性不替代逐条内容重新验收。
- 读取全部2,500条full记录的已存trace，核验每题一次检索、一次Weaver调用、
  返回2槽/16tokens、Trigger inactive、query norm有值、跨query cosine字段为null。
  未重新评分或推理，未加载Torch/pickle/checkpoint参数；未重做全部父记录内容hash。
- 当前代码与M3冻结源码对应；提交PDF及历史P7仅是历史比较层，不能把M3当作
  历史代码的精确复现。当前HEAD改变不意味着M3来源清单失效：比较的是文件hash。

## 2. Retrieval真实数据流

代码锚点：

|位置|事实|
|---|---|
|[modeling_memgen.py:148](/mnt/18T/baishilong/MemGen/memgen/model/modeling_memgen.py:148)|`reasoner_to_weaver`与反向投影均为带bias的nn.Linear|
|`modeling_memgen.py:505,567,598`|初始`X`为Reasoner token embedding；candidate取自current_inputs_embeds，而非最后一层contextual hidden states|
|`modeling_memgen.py:673–688`|Weaver-space bank模式先对candidate逐位置投影，再送bank检索|
|`latent_memory_bank.py:361–378`|query池化最后L位置；key池化每个memory的全部位置|
|`latent_memory_bank.py:480–525`|cosine乘访问年龄衰减，score>=threshold后top-k；并列较小index；无强制top-1 fallback|
|`modeling_memgen.py:903–915`|Weaver-space bank保存原始Weaver输出；不是先投影到Reasoner再投影回来|
|`modeling_memgen.py:1177–1191`|投影、latent queries、LayerNorm/scale来自checkpoint，不是本阶段新训练|

本M3配置：L=64、max_slots=16、top_k=2、retrieve_threshold=.05、
update_threshold=.10、alpha=.05、thread_update、threshold_topk、CPU storage。
runtime启用`memory_bank_storage_space=weaver`和`retrieved_memory_to_weaver=True`；
通用代码默认仍可能走reasoner-space路径，不能把M3配置推广为全部路径。
加载设置见`scripts/eval/mab6b_weaver_space_bank_detectiveqa_n10.py:110–115`，
M3调用见`scripts/eval/review_m3_worker.py:105–115`。

令N为当前有效输入长度、d_R/d_W为两个模型维度，当前二者均1536：

    X_q [1,N,d_R]
      → P_R→W(X_q) [1,N,d_W]
      → mean(last min(64,N) positions) → q [d_W]
    slot_i.memory [8,d_W] → mean(8 positions) → k_i [d_W]
    a_i = cosine(q,k_i) * exp(-.05 * access_age_i)
      → inclusive threshold .05 → score top-2

q与k处于兼容维度/表示接口，不是未经投影直接跨空间cosine；但q是**投影后的输入
embedding池化**，k是**Weaver输出hidden states池化**，没有独立语义alignment验证。
bank池化函数不接收attention mask；本设置batch1，并不证明所有padding/clipping情况
均不会污染池化。后续必须用实际rendered/tokenized prompt检查，不能仅看question字符串。

静态推论（非本阶段实验结果）：线性投影加mean pooling没有上下文编码器，忽略有限
精度舍入时，对同一最后64-token多重集合的内部排列不敏感；窗口边界/词项改变则可变。
共同指令后缀与候选列表可能影响该窗口，**其占比及是否造成固定选槽尚未测定**。
应先检查窗口组成，不直接认定“模板主导”或“查询无效”。

## 3. Ordering：native实现与正式full不是同一个顺序

bank原生返回score降序；M3 adapter把full选中集合按slot index升序送入Weaver，
其余count-matched策略同样canonical排序。`native_order`保留score顺序。
见`scripts/eval/review_retrieval_policies.py:32–55,182–223`。
cosine-only在整个bank选相同数量，不复用full阈值过滤；random为identity固定私有RNG。
这是明确的实验比较契约，不可称为所有策略使用同一阈值或相同排序语义。
未来实验须注明复用的是M3 canonical full还是生产native路径。

## 4. Regeneration tensor-level路径

两槽分别为[8,1536]。`torch.cat(slot.memory,dim=0).unsqueeze(0)`得到
R=[1,16,1536]（`modeling_memgen.py:709–723`）。当前查询路径：

|阶段|conditioning on|conditioning off / empty|
|---|---|---|
|Weaver外层输入|`cat([P(X_q),R],dim=1)` → [1,N+16,1536]|P(X_q) → [1,N,1536]|
|attention mask|query mask后拼16个1|原query mask|
|position IDs|根据拼接mask的cumsum重新生成|原query positions|
|Weaver内部learned queries|LN+scale后的8个prompt query tokens拼接到末尾|相同8个learned queries|
|Weaver LM实际输入|[1,N+16+8,1536]|[1,N+8,1536]|
|Weaver输出|最后8位置的last-layer hidden states [1,8,1536]|同形状、内容可不同|
|投影与Reasoner输入|P_W→R(M')，拼回原X → [1,N+8,1536]|相同shape的无历史conditioning新latent|

外层concat/mask/position见`modeling_memgen.py:732–745`；position生成见
`modeling_utils.py:141–158`；内部learned query、mask、positions与last-layer提取见
`weaver.py:119–148,163–169`；Reasoner拼接见`modeling_memgen.py:785,861–868,893–899`。
Weaver使用底层causal LM的标准mask行为，没有独立memory cross-attention模块或
另一个显式semantic fusion模块；不能把abstract Weaver(H,R)解释成不存在的接口。
原始16历史tokens不直接进入full的Reasoner；Reasoner并非只接收新latent，而是仍有X_q。
Weaver内部的8个learned query tokens也不能与retrieval query向量q混称。

代码与配置可推出full新生成/注入8tokens。一个日志陷阱：本次2,500个full记录的
`reasoner_injected_latent_count=0`，此字段只在direct路径被填充，正常full沿用初始化0；
不能据此判断full未注入latent。16是conditioning预算，8是Reasoner新增预算，
Matched-16文本控制也不是Reasoner消耗16个新latent的证明。

## 5. Trigger、只读与no-conditioning

- `_should_augment`见`modeling_utils.py:349–420`。prompt先进入Trigger调用，
  **一般active Trigger仍可能SKIP**，不能仅依注释称prompt无条件INVOKE。
- 当前checkpoint trigger_active=false；`trigger.py:69–75`固定INVOKE logits，
  max_inference_aug_num=0使后续生成不再增强。本M3 full记录均一次prompt Weaver调用。
  配置prompt_latents_len=8、inference_latents_len=8；后者本次未用于生成期增强。
- `query_retrieved_memory_conditioning=False`/`retrieve_but_do_not_condition`
  保留检索，不拼R，仍运行Weaver；不是关闭全部latent生成。见model:622–642。
- no-retrieval是read-only proxy返回空检索；不同于“检索但不conditioning”。
  见EventQA runner:839–858,1287–1305。
- native检索会暂时修改访问元数据；proxy在finally恢复，query writes被拦截；
  M3每策略另恢复同bank和RNG。见runner:804–878，worker:108–117。
  因此read-only是外部持久状态不变，不是函数内部从不修改任何计数。
- 现有门控是`cosine×decay>=.05`导致R为空时的回退，不是独立校准的query收益gate。
  本M3全部返回2槽，empty fallback没有实际触发，不代表其他配置不触发。

## 6. Direct injection的可比性缺口

当前`direct_top1`设置top_k_override=1，绕过Weaver，将一个槽的8tokens经
P_W→R直接拼给Reasoner。见model:665–688,761–767,805–837。
full是两个槽16tokens→Weaver→8新tokens。因此历史full/direct_top1对比同时涉及
检索槽数、conditioning token预算及Weaver执行，**不是严格同R下只变融合操作**。
历史P7 EM18.80% vs direct4.72%属于历史观察，不应当作新M3的配对因果结论。
后续最小公平桥接用相同一个slot的Weaver-1 vs direct-1（均最终注入8tokens），
再单独比较Weaver-2，不能直接笛卡尔扩展所有组合。

## 7. 与Reviewer表述的差异

需要澄清：有P_R→W；Query不是last-layer hidden state；16历史tokens与8新tokens
不同；存在空检索回退；当前Trigger不是学得的query选择gate；M3 full顺序由adapter
固定；真实parser成功不等于合法候选。仍需承认：cosine相关性、latent事实保留、
再生成内容利用、query收益可预测性和训练分布偏移稳健性均未由这些实现事实证明。
