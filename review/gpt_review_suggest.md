可以。根据这三份 review，我建议下一版不要采取“reviewer 提到什么就补什么”的方式，而是围绕一个核心目标重构：

> **证明提升来自“检索到相关历史 latent memory + query-conditioned regeneration”，而不是 recency、格式修复或额外输入。**

目前 reviewer 最集中的质疑就是两条因果链没有被验证：  
**retrieval → relevant memory**，以及 **relevant memory → better evidence use → EM**。Emergency reviewer直接要求 Random/Last-two retrieval、parsed-output accuracy、chance rate，并质疑 Reasoner query 与 Weaver key 是否处于可比较空间。:chatgpt-content-reference{index="0"} :chatgpt-content-reference{index="1"}  
AI reviewer也提出 retrieval/selection/format error decomposition、long-horizon evidence-age analysis，以及 frozen-Weaver tensor-level specification。:chatgpt-content-reference{index="2"}

下面这个计划可以直接作为之后交给 Codex 的总 roadmap。

---

# 一、下一版论文的目标定位

我建议把论文中心从：

> **“我们给 MemGen 加了一个 latent memory bank。”**

调整成：

> **“Persistent latent memories cannot simply be reused; effective reuse requires relevance-aware retrieval followed by query-conditioned latent regeneration.”**

因为 reviewer 已经认为 similarity retrieval、temporal decay、LRU-style eviction 本身 novelty 不够。:chatgpt-content-reference{index="3"}

真正被 reviewer 认可、也最有区分度的是：

\[
\text{Retrieve}
\rightarrow
\text{Weaver conditioning}
\rightarrow
\text{Regenerated latent}
\rightarrow
\text{Reasoner}
\]

AI reviewer甚至明确指出：

> Weaver-mediated regeneration 是比 latent storage 更 distinctive 的贡献。:chatgpt-content-reference{index="4"}

因此下一版的核心贡献最好变成三个：

| Contribution | 下一版需要证明什么 |
|---|---|
| Persistent latent memory | latent memory 可以跨 long-horizon context 被保存和复用 |
| Relevant retrieval | 检索真正选择了与当前 query 有关的 memory，而不是最近的 slot |
| Query-conditioned regeneration | historical latent 不能简单注入，而必须经过 Weaver 重新融合 |

---

# 二、Phase 0：先冻结当前结果，不立即改算法

Codex 第一件事不是跑新实验，而是建立一个 **review-response baseline snapshot**。

让它先读取当前：

`research_notes/PROGRESS.md`

以及当前论文、实验 config、已有 EventQA 结果，然后生成：

```text
research_notes/
└── next_version/
    ├── BASELINE.md
    ├── REVIEW_ISSUES.md
    ├── EXPERIMENT_PLAN.md
    └── RESULT_TABLE.md
```

其中 `BASELINE.md` 必须记录：

```text
model checkpoint
EventQA subset
number of contexts
number of questions
generation config
bank config
retrieve_th
update_th
capacity
top_k
decay_alpha
number of runs
EM
Recall
Format failures
runtime
VRAM
git commit
```

之后所有实验只能相对这个 snapshot 比较。

**Definition of done：**

任何下一版实验都能通过一个 config + commit hash 完全复现。

---

# 三、Phase 1：第一优先级——拆解 EM 提升到底来自哪里

这是目前最大的 reviewer objection。

当前 reviewer 看到的是类似：

\[
EM:\ 0.008\rightarrow0.188
\]

但：

\[
Recall:\ 0.178\rightarrow0.231
\]

同时 format failure 大幅下降，因此怀疑大部分 EM gain 来自 parsing / formatting，而不是 memory。:chatgpt-content-reference{index="5"}

### Codex 需要增加统一 evaluation script

至少输出：

| Metric | 含义 |
|---|---|
| Raw EM | 当前指标 |
| Raw answer recall | gold event 是否出现在 raw output |
| Format success rate | parser 是否成功 |
| Parsed-only EM | 只在成功 parse 的样本中计算 EM |
| Parsed-only accuracy | 成功解析以后答案是否正确 |
| Gold mentioned but wrong selection | 找到了证据，但最终选错 |
| Gold mentioned but parse failed | 有正确 evidence，但格式失败 |
| Gold absent | memory/reasoning 根本没找到 gold |
| Candidate chance level | 随机猜 candidate 的期望准确率 |

尤其要明确：

\[
Acc_{\text{parsed}}
=
\frac{\#correct\ parsed}
{\#successfully\ parsed}
\]

Reviewer 已经粗略算出 full method parsed-only 大约 0.25，而 Matched-16 大约 0.22，所以这个实验结果可能没有原始 EM 那么漂亮。:chatgpt-content-reference{index="6"}

但必须正面报告。

### 推荐最终表格

```text
Method          Recall   Format Succ.  Parsed Acc.  EM
-------------------------------------------------------
Recent-text
Matched-16
BM25
Dense E5
Latent Bank
```

### Gate A

如果出现：

\[
ParsedAcc_{bank} > ParsedAcc_{text}
\]

且 recall 也提升：

很好，说明 memory 本身确实贡献了答案质量。

如果：

\[
ParsedAcc_{bank}\approx ParsedAcc_{text}
\]

而主要提升来自 Format Success：

那么当前“memory improves evidence use”的 claim 必须削弱，后面的 retrieval redesign 就变成必须做。

---

# 四、Phase 2：最重要的新实验——证明 retrieval 不是“最近两个”

这是我认为**整个下一版最优先的新实验**。

Reviewer明确提出：

> Random slots?

> Last-written slots?

> full retrieval 是否其实就是最近两个？:chatgpt-content-reference{index="7"}

Codex 应该统一加入以下 retrieval policies：

| Policy | Score |
|---|---|
| `random_k` | 随机 K 个 |
| `last_k` | 最近写入的 K 个 |
| `semantic_only` | cosine similarity |
| `recency_only` | temporal score |
| `semantic_recency` | 当前 full |
| `oracle` | gold-relevant slot，仅分析用 |

保持所有其他变量完全相同：

```text
same context
same bank
same K
same Weaver
same Reasoner
same decoding
```

只替换 retrieval policy。

最终至少比较：

\[
Random\text{-}2
\]

\[
Last\text{-}2
\]

\[
Semantic\text{-}2
\]

\[
Recency\text{-}2
\]

\[
Semantic+Recency
\]

---

# 五、Phase 3：增加 retrieval relevance diagnostics

不能再只看 downstream EM。

必须直接回答：

> 取出来的 memory 到底是不是相关的？

建议构建 slot-level metadata，仅用于 evaluation：

```text
slot_id
source_context
source_event
source_position
write_step
last_access_step
refresh_count
```

**注意：metadata 只用于分析，不能给模型。**

对于每个 EventQA question，利用 benchmark 已知 gold event/source，把 slot 标成：

\[
relevant(m_i,q)\in\{0,1\}
\]

然后计算：

\[
Recall@1
\]

\[
Recall@2
\]

\[
MRR
\]

以及：

\[
P(\text{retrieved slots are last-2})
\]

还要统计：

```text
cosine score distribution
recency score distribution
final retrieval score distribution
```

最好画：

\[
x=\text{memory age}
\]

\[
y=\text{selection probability}
\]

这样能直接回答 reviewer：

> temporal decay 是否压过 semantic similarity？

---

# 六、Phase 4：解决“Reasoner query vs Weaver key 不同空间”

这个问题我认为很关键，而且有机会变成下一版真正的方法改进。

目前：

\[
q=\text{pooled Reasoner hidden state}
\]

但：

\[
k_i=\text{pooled Weaver latent}
\]

然后直接：

\[
\cos(q,k_i)
\]

Reviewer合理地问：

> 为什么两个模型/模块产生的 representation 可以直接 cosine？:chatgpt-content-reference{index="8"}

### 先做 diagnostic

构造：

```text
positive pair:
question query ↔ gold-event memory

negative pair:
question query ↔ unrelated memory
```

比较：

\[
S_{positive}
\]

和

\[
S_{negative}
\]

输出：

- mean similarity
- histogram
- ROC-AUC
- Recall@K

如果两类明显分离，可以证明 current cosine 有意义。

---

## 如果不能分离：我建议直接升级成真正的 Key–Value Latent Bank

这是下一版可能最重要的方法修改。

不要：

\[
key = Weaver\ latent
\]

而是：

\[
key_i = \text{Reasoner-space representation at write time}
\]

\[
value_i = Weaver\ latent\ memory
\]

也就是：

\[
B=
\{(k_i,v_i)\}
\]

其中：

\[
k_i\in Reasoner\ space
\]

\[
v_i\in latent\ memory\ space
\]

query 同样是：

\[
q\in Reasoner\ space
\]

于是 retrieval：

\[
s_i=\cos(q,k_i)
\]

变成**同空间比较**。

然后：

\[
v_{top-k}
\rightarrow Weaver
\rightarrow regenerated\ latent
\rightarrow Reasoner
\]

逻辑就非常干净：

```text
Reasoner-space key → retrieval
Latent value → memory content
Weaver → regeneration
```

这比当前直接把 latent 本身同时充当 key 和 value 的设计更合理。

而且仍然满足：

> no additional training

这是非常值得尝试的 **Version B**。

---

# 七、Phase 5：证明“long-horizon”而不是普通 QA improvement

现在 reviewer 对标题里的 **Long-Horizon Agents** 很不满意，因为实际只做 EventQA。:chatgpt-content-reference{index="9"}

在增加新 benchmark 之前，可以先把 EventQA 挖透。

对于每个问题，记录：

\[
distance =
query\ position-gold\ event\ position
\]

按 evidence age 分桶：

```text
0–20%
20–40%
40–60%
60–80%
80–100%
```

或者：

```text
recent
medium
old
very old
```

比较：

```text
Recent-text
Matched-16
Latent Bank
```

分别计算：

\[
EM(age)
\]

\[
Recall(age)
\]

\[
RetrievalRecall@K(age)
\]

如果 bank 的优势随 evidence age 增长而增大，例如：

```text
Recent evidence       ≈ baseline
Medium evidence       + small
Old evidence          + large
Very old evidence     + largest
```

这张图会非常强。

因为它直接证明：

> improvement specifically appears when information becomes long-range.

AI reviewer也明确建议做 performance as a function of evidence age/source position。:chatgpt-content-reference{index="10"}

---

# 八、Phase 6：彻底检查“seed variance”问题

这个问题千万不要忽视。

Reviewer发现：

> greedy decoding

但：

> bank-on 有 std

而：

> bank-off zero variance。:chatgpt-content-reference{index="11"}

Codex 需要审计所有 stochastic source：

```text
Python random
NumPy random
torch CPU RNG
torch CUDA RNG
model.generate()
do_sample
temperature
top_p
tie breaking
slot initialization
iteration order
context ordering
GPU nondeterminism
```

必须回答：

> **所谓不同 seed 到底改变了什么？**

如果实际没有理论上的随机来源，但结果仍变化，这是实验 pipeline bug 或 CUDA nondeterminism，需要查清楚。

下一版不要只写：

```text
mean ± std over seeds
```

还应该使用：

### paired question-level comparison

例如：

- McNemar test：Bank vs baseline EM
- bootstrap 95% CI

最好按 context 做 cluster bootstrap，避免把同一个 context 内的 100 questions 当成完全独立样本。

---

# 九、Phase 7：重新研究 temporal decay，而不是继续 sweep 参数

Review 已经指出：

\[
EM_{decay}=0.188
\]

和 no-decay 的差距小于一个标准差，而 no-decay recall 反而更高。:chatgpt-content-reference{index="12"}

所以不要再做很多：

```text
alpha=0.01
0.03
0.05
0.07
...
```

意义不大。

应该回答更基本的问题：

\[
\boxed{\text{Do we actually need temporal decay?}}
\]

实验：

```text
Semantic only
Recency only
Semantic + Recency
```

如果 semantic-only 最好：

**直接删掉 decay。**

方法反而更简单、更有说服力。

不要为了保持原论文设计而强行保留 reviewer 已经质疑且实验支持不足的组件。

---

# 十、Phase 8：Frozen Weaver 的 tensor flow 必须完全写清楚

Reviewer已经两次指出这一点。:chatgpt-content-reference{index="13"} :chatgpt-content-reference{index="14"}

Codex 应该从真实代码反向生成一个 architecture specification。

必须明确：

```text
H_t shape
retrieved memory shape
K
latent tokens per memory
concatenation dimension
ordering
attention mask
position encoding
Weaver input
Weaver output
Reasoner input
Trigger state
```

论文里不能再只写：

\[
Weaver(H_t,R_t)
\]

而要写成类似：

\[
R_t=
[m_{i_1};m_{i_2}]
\in\mathbb{R}^{16\times1536}
\]

然后明确它到底是：

\[
[H_t;R_t]
\]

还是其他 conditioning 方式。

尤其必须解释：

**Retrieved 2 × 8 latent tokens ≠ Reasoner 最终收到 16 latent tokens**

如果实际 Weaver 最终重新生成 8 个 latent：

就必须明确：

\[
16\ retrieved
\rightarrow Weaver
\rightarrow8\ regenerated
\rightarrow Reasoner
\]

这是 reviewer 当前明显误解的地方之一。

---

# 十一、Phase 9：Dynamic read-write 不要只报告“效果下降”

Official reviewer把这个当成 real-agent deployment 的主要弱点。:chatgpt-content-reference{index="15"}

Codex 需要先诊断：

```text
每个 question 后新增多少 slot
多少 slot 被 refresh
多少 slot 被 eviction
bank similarity 如何变化
retrieved gold-slot rate 如何变化
query-generated memory 是否污染 context memory
```

我怀疑一个值得重点检查的原因是：

> query/answer derived memory 被重新写入 bank，逐渐挤掉原始 context-derived memory。

如果成立，可以尝试三个模式：

```text
A. Frozen
context write → query read

B. Fully dynamic
context write → query read/write

C. Gated dynamic
context write → query read
only high-confidence memory can write
```

如果 C 明显恢复性能，就可以形成一个新的结果：

> unconstrained online writing causes latent-memory contamination; gated updates mitigate it.

这个甚至可以成为很不错的 supplementary finding。

---

# 十二、Phase 10：补一个 external reference point

Reviewer不是单纯要求“再多跑一个模型”，而是说现在**完全没有坐标系**：

- 没有 published EventQA reference
- 没有 plain Qwen
- 没有 latent-memory competitor
- 没有 upper bound :chatgpt-content-reference{index="16"}

最容易先补：

### Lower / upper references

```text
Chance
Recent text
Matched text
Full long context
Gold evidence oracle
```

如果可以，再增加：

```text
Plain Qwen2.5-1.5B-Instruct + same retrieval
```

这样至少可以回答：

> MemGen + bank 相比 vanilla backbone 到底处于什么位置？

---

# 十三、Phase 11：第二 benchmark / agent task 放在最后

我不建议现在第一件事就开始做一个复杂 tool-use benchmark。

先把前面的 mechanism 验证完。

否则如果后面发现：

\[
Full\approx Last2
\]

那之前所有新 benchmark 都是在验证一个机制有问题的版本。

等 retrieval 和 evaluation 都稳定以后，再选一个第二任务。

优先选择：

**同一个 MemoryAgentBench 中、能够复用现有 pipeline 的 long-context / multi-turn task。**

其次才是完整 tool agent benchmark。

---

# 十四、如果资源有限，我会这样定义“最低可投稿版本”

下一版至少必须完成下面这一组：

| 必做 | 实验 |
|---|---|
| ✅ | Parsed-only accuracy |
| ✅ | Error decomposition |
| ✅ | Chance / oracle |
| ✅ | Random-2 |
| ✅ | Last-2 |
| ✅ | Semantic-only |
| ✅ | Recency-only |
| ✅ | Retrieval Recall@1/@2 |
| ✅ | Last-2 overlap rate |
| ✅ | query-key representation diagnostic |
| ✅ | evidence-age analysis |
| ✅ | randomness/seed audit |
| ✅ | tensor-level Weaver description |
| ✅ | EM-LLM / Memoria related work |
| ✅ | significance / confidence intervals |

这一套的价值远高于再做：

```text
capacity=12/16/20/24
threshold=0.03/0.04/0.05
alpha=0.03/0.05/0.07
```

reviewer 已经不缺这种 hyperparameter sweep 了。

---

# 十五、结果出来以后设置三个“决策门”

这是后续 Codex 特别需要遵守的，避免机械地执行到底。

### Gate 1 — Semantic retrieval 是否成立

如果：

\[
Full > Last2 > Random
\]

且：

\[
Recall@2_{semantic}
\gg Recall@2_{random}
\]

保留当前 retrieval。

如果：

\[
Full\approx Last2
\]

停止扩 benchmark。

直接进入 **aligned key-space redesign**。

---

### Gate 2 — Memory improvement 是否不仅是 formatting

如果：

\[
Recall\uparrow
\]

且：

\[
ParsedAccuracy\uparrow
\]

那么可以继续 claim：

> latent memory improves historical evidence utilization.

如果只有：

\[
FormatSuccess\uparrow
\]

那必须重新定位论文，不能继续把全部 EM gain 写成 memory retrieval improvement。

---

### Gate 3 — Temporal decay 是否有价值

如果：

\[
SemanticOnly\ge Semantic+Decay
\]

删掉 decay。

如果 decay 在 older evidence 明显有帮助，再保留。

---

# 十六、论文结构也建议一起改

新的结构我建议变成：

```text
1 Introduction

2 Background
  2.1 Generated Latent Memory
  2.2 Persistent Memory Management
  2.3 Why Direct Latent Reuse Fails

3 Method
  3.1 Session-local Key–Value Latent Bank
  3.2 Retrieval
  3.3 Query-conditioned Latent Regeneration
  3.4 Memory Update / Eviction
  3.5 Complexity

4 Experimental Setup

5 Does Retrieval Find Relevant Memory?
  Random / Last / Semantic
  Retrieval Recall@K

6 Does Latent Memory Improve Long-Horizon Evidence Use?
  EM
  Recall
  Parsed accuracy
  Evidence-age analysis

7 Why Regeneration?
  Drop retrieval
  Direct injection
  Weaver-mediated regeneration

8 Online Memory Dynamics
  Frozen
  Dynamic
  Gated dynamic

9 Efficiency

10 Limitations
```

也就是说，**retrieval correctness 本身变成一个正式实验 section**，而不是只通过最终 EM 间接说明。

---

# 十七、Related Work 重构

reviewer已经给了很好的分类方式：:chatgpt-content-reference{index="17"}

```text
1. Cached hidden-state memory
2. Trainable latent memory
3. Generative latent memory
4. Explicit agent memory
```

至少补：

```text
EM-LLM
Memoria
MEMORYLLM
M+
MemGen
```

重点不要说：

> 我们第一个做 persistent memory。

而应该说得更精确：

> prior methods manage cached, textual, or trained latent states; we study inference-time lifecycle management and query-conditioned regeneration of dynamically generated MemGen memories.

这样 novelty 会安全很多。

---

# 十八、最后给 Codex 的整体执行顺序

以后正式让 Codex 做的时候，我建议严格按照：

```text
M0  Freeze baseline
 ↓
M1  Evaluation decomposition
 ↓
M2  Random / Last / Semantic / Recency retrieval
 ↓
M3  Retrieval relevance diagnostics
 ↓
M4  Representation-space diagnostic
 ↓
        ├─ retrieval成立 → keep current design
        └─ retrieval失败 → aligned Key–Value redesign
 ↓
M5  Evidence-age analysis
 ↓
M6  Seed/randomness audit + statistics
 ↓
M7  Dynamic-bank diagnosis
 ↓
M8  Oracle / long-context / plain-Qwen references
 ↓
M9  Second task/model
 ↓
M10 Rewrite paper
```

其中 **M0–M4 是绝对不能跳过的核心阶段**。

我甚至建议在 Codex 的总任务说明里明确写：

> **Do not start new benchmark scaling until M1–M4 are complete and the retrieval mechanism passes the decision gates.**

这样可以避免 Codex 一股脑把 reviewer 提出的所有实验全跑了，却没有先判断论文最核心的方法是否成立。

等你下一步准备让 Codex 正式开始时，我建议我们就从 **M0** 开始，我可以直接给你一份适合 Codex CLI 的完整 master prompt，把这些 milestone、禁止事项、结果记录格式、GPU 使用规则和 decision gate 全写进去。