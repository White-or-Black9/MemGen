# MemGen 代码阅读指南

## 概述

本文档是 MemGen (Memory Generation) 项目的代码阅读导航。MemGen 是一种为 LLM Agent 提供动态生成式隐记忆（Generative Latent Memory）的框架。

### 核心思想

- **Reasoner（推理器）**：冻结的基座 LLM，执行实际的 token 生成（永不更新）
- **Weaver（记忆编织器）**：小型 LM + LoRA，根据 reasoner 的状态动态生成隐记忆 token
- **Trigger（记忆触发器）**：小型 LM + 二分类器，决策何时调用记忆编织

### 术语速查

| 术语 | 含义 |
|------|------|
| Latent Query | 可学习的记忆"钩子"，作为 Weaver 的输入 |
| Prompt Latents | 任务开始时生成的规划记忆 |
| Inference Latents | 推理过程中生成的工作记忆 |
| Augmentation | 在分隔符位置注入隐记忆的过程 |
| Delimiter | 触发点：`,`, `.`, `\n` |
| GRPO | Group Relative Policy Optimization（强化学习算法） |
| info_mask | 标识训练时哪些位置参与 loss 计算 |

---

## 目录结构

```
MemGen/
├── memgen/
│   ├── model/              # 核心模型定义
│   │   ├── modeling_memgen.py  ★ MemGenModel（核心）
│   │   ├── modeling_utils.py   ★ Mixin 类（LoRA、生成）
│   │   ├── weaver.py           ★ 记忆编织器
│   │   └── trigger.py          ★ 记忆触发器
│   ├── trainer/
│   │   ├── weaver_grpo_trainer.py  ★ Weaver GRPO 训练器
│   │   ├── trigger_grpo_trainer.py ★ Trigger GRPO 训练器
│   │   └── utils.py               ★ NaN 安全的统计函数
│   ├── runner.py            ★ 训练/评估调度器
│   └── utils.py             ★ 评估记录器、工具函数
├── interactions/
│   ├── base_interaction.py       ★ 交互管理基类和配置
│   ├── singleturn_interaction.py ★ 单轮交互（STATIC 环境）
│   ├── multiturn_interaction.py  ★ 多轮交互（DYNAMIC 环境）
│   └── tensor_utils.py           ★ Tensor 操作工具
├── data/
│   ├── base_env.py          ★ 环境基类（STATIC / DYNAMIC）
│   ├── base_builder.py      ★ 数据集构造器基类
│   └── gsm8k/               ★ GSM8K 数据集实现（示例）
├── common/
│   ├── config.py            ★ 配置管理（OmegaConf）
│   └── logger.py            ★ 日志设置
├── configs/                 ★ YAML 配置文件
├── scripts/                 ★ 训练/评估脚本
└── main.py                  ★ 程序入口
```

---

## 阅读计划（按层递进）

### Layer 0: 背景与高维理解

**目标**：建立整体认知，理解 MemGen 解决的问题和方法。

1. **论文**：阅读 `userfiles/` 下的 MemGen 论文 PDF，重点关注：
   - Section 3: Generative Latent Memory 的定义
   - Section 4: Weaver + Trigger 的架构设计
   - Section 5: 训练流程（SFT → GRPO）

2. **README.md**: 快速了解项目功能和用法

3. **configs/ 和 scripts/**: 查看配置文件（GSM8K 的 latent_memory/gsm8k.yaml）和启动脚本（train/qwen2_5_gsm8k_sft.sh），了解训练配置的全貌

4. **main.py**: 程序入口。理解 MemGen 的启动流程：
   - 解析参数 → 加载配置 → 构建模型 → 创建 Runner → 训练/评估

### Layer 1: 核心模型架构

**目标**：透彻理解 MemGenModel、Weaver、Trigger 的实现。

**阅读顺序**：`weaver.py` → `trigger.py` → `modeling_utils.py` → `modeling_memgen.py`

#### 1.1 MemGenWeaver (`memgen/model/weaver.py`)

- **类文档**：理解 Weaver 的类比（任务规划记忆 vs 工作记忆）
- **__init__**: 两个可学习参数 `prompt_query_latents` 和 `inference_query_latents`
- **_augment**: 核心方法（公式 5），理解：
  - 如何将 latent query 拼接到 inputs_embeds
  - 如何使用 Weaver 的 LM 生成 latent hidden states
  - 为什么需要对 latent 做 LayerNorm + scale
- **augment_prompt / augment_inference**: 两个公开接口

#### 1.2 MemGenTrigger (`memgen/model/trigger.py`)

- **两种模式**：active（主动决策） vs 非 active（固定 INVOKE）
- **forward**: 在每个位置输出 INVOKE/SKIP 的 logits
- **classifier head**: 简单的 Linear(2) 分类层

#### 1.3 Mixin 类 (`memgen/model/modeling_utils.py`)

- **MemGenLoraSwitchMixin**:
  - `_insert_lora_adapters`: 为 weaver/trigger 注入 LoRA
  - `fix_component`: 冻结指定组件
- **MemGenGenerationMixin**:
  - `_select_augment_points_after_delimiter`: 训练时选择 augmentation 位置
  - `_should_augment`: 推理时 trigger 的决策逻辑
  - `_append_one_step`: 自回归生成的一步
  - `_get_next_token`: 采样或贪心选择

#### 1.4 MemGenModel (`memgen/model/modeling_memgen.py`)

这是**最重要的文件**，建议重点阅读：

- **__init__**: 理解完整架构（reasoner → weaver → trigger → 投影层）
- **_forward**: 训练流程（选择 augment 点 → 分段处理 → 注入记忆 → reasoner 处理）
- **forward**: 统一前向入口（instruction 模式 / conversation 模式）
- **generate**: 推理时的增强生成流程（trigger 决策 + weaver 记忆注入）
- **from_config / save_pretrained / from_pretrained**: 模型序列化

**关键数据流**（训练时）：
```
input_ids → _select_augment_points_after_delimiter() → 分段
  └─ 每段：reasoner_to_weaver() → weaver.augment_*() → weaver_to_reasoner()
      → reasoner 处理增强后的序列 → 收集 logits
```

**关键数据流**（推理时）：
```
generate() → 自回归循环：
  每生成一个 token → _check_ends_with_delimiter() → 是分隔符？
    ├─ Yes → trigger._should_augment() → INVOKE?
    │   ├─ Yes → weaver 生成记忆 → 注入 → 继续生成
    │   └─ No → 跳过，继续生成
    └─ No → 继续生成
```

### Layer 2: 训练与交互

**目标**：理解训练流程、GRPO 算法实现、Agent-Environment 交互。

#### 2.1 WeaverGRPOTrainer (`memgen/trainer/weaver_grpo_trainer.py`)

- **__init__**: 接收 env_class 和 generation_manager
- **_get_per_token_logps**: 使用 `outputs.supervised_labels` 获取监督掩码
- **_generate_and_score_completions**: 核心 rollout 流程：
  - 通过 `generation_manager.run_agent_loop()` 生成完整轨迹
  - 提取 prompts、completions、info_mask
  - 计算参考模型的 logps（用于 KL 惩罚）
  - 计算奖励和优势值（组内归一化）
- **_compute_loss**: GRPO/BNPO/DR_GRPO loss 计算
  - `supervised_mask = completion_mask * supervise_mask * ...`
  - 确保只在有效位置计算 loss
- **training_step**: OOM 保护

#### 2.2 TriggerGRPOTrainer (`memgen/trainer/trigger_grpo_trainer.py`)

- 关键区别：使用 `augmentation_mask` 而不是文本 completion
- `_get_per_token_logps`: 计算 INVOKE/SKIP 的二分类 logps
- `_generate_and_score_completions`: 通过 `model.generate(return_augmentation_mask=True)` 获取 trigger 决策
- 参考模型使用 `model.trigger` 的子模块（而不是整个 MemGenModel）

#### 2.3 MemGenRunner (`memgen/runner.py`)

- **train()**: 分阶段训练（Weaver SFT → Weaver GRPO → Trigger GRPO）
- **evaluate()**: 根据环境类型选择评估方法
- **_create_weaver_trainer**: 支持 SFT 和 GRPO
- **_create_trigger_trainer**: 只支持 GRPO
- **OOM 处理**: 显存不足时保存紧急 checkpoint

### Layer 3: 交互管理与数据

**目标**：理解 Agent 与环境的交互机制。

#### 3.1 Interaction 基类 (`interactions/base_interaction.py`)

- **InteractionConfig**: 交互配置
- **InteractionDataProto**: 数据协议（batch + no_tensor_batch）
- **InteractionManager**: 抽象基类

#### 3.2 SingleTurnInteractionManager (`interactions/singleturn_interaction.py`)

- 单轮生成流程
- **info_mask** 的作用

#### 3.3 MultiTurnInteractionManager (`interactions/multiturn_interaction.py`)

- 多轮交互循环
- **active_mask**: 跟踪哪些样本还在交互中
- **inter_histories**: 完整对话历史

#### 3.4 TensorHelper (`interactions/tensor_utils.py`)

- padding 操作、position IDs、attention mask 创建

### Layer 4: 数据与环境

**目标**：理解数据集构造和环境抽象。

- **BaseEnv**: STATIC（单轮）/ DYNAMIC（多轮）两种环境
- **BaseBuilder**: SFT / GRPO 模式的 dataset 构造
- **GSM8K 实现**: 具体的数据集示例
- **compute_reward**: 奖励函数（作为 GRPO 训练的 reward_func）

---

## 快速查找指南

| 你想了解什么 | 看哪个文件 |
|------------|-----------|
| 整体启动流程 | `main.py` |
| 配置结构 | `configs/latent_memory/gsm8k.yaml`, `common/config.py` |
| 模型架构 | `modeling_memgen.py` |
| Weaver 实现 | `weaver.py` |
| Trigger 实现 | `trigger.py` |
| 训练辅助逻辑 | `modeling_utils.py` |
| 训练调度 | `runner.py` |
| Weaver GRPO 训练 | `weaver_grpo_trainer.py` |
| Trigger GRPO 训练 | `trigger_grpo_trainer.py` |
| 单轮交互 | `singleturn_interaction.py` |
| 多轮交互 | `multiturn_interaction.py` |
| 数据集构造 | `data/base_builder.py`, `data/gsm8k/builder.py` |
| 环境抽象 | `data/base_env.py`, `data/gsm8k/env.py` |
| 评估记录 | `memgen/utils.py` |
| 训练脚本 | `scripts/` |

---

## 建议的阅读路径

### 快速理解（适合初次接触）

```
main.py → gsm8k.yaml → modeling_memgen.py (类文档 + __init__)
  → weaver.py → trigger.py → modeling_utils.py (关键方法)
  → runner.py (train + evaluate)
```

### 深入理解训练（适合理解 GRPO）

```
weaver_grpo_trainer.py → trigger_grpo_trainer.py
  → singleturn_interaction.py / multiturn_interaction.py
  → base_interaction.py → tensor_utils.py
```

### 全面理解（适合做修改/扩展）

```
main.py → config.py → modeling_memgen.py (完整阅读)
  → modeling_utils.py (完整阅读)
  → weaver.py → trigger.py
  → weaver_grpo_trainer.py → trigger_grpo_trainer.py
  → runner.py
  → interactions/*
  → data/*
  → utils.py
```

---

## 三阶段训练流程速览

```
Stage 1: Weaver SFT
  ├─ 固定 Trigger（不更新）
  ├─ 训练数据只包含 assistant 回复的文本
  ├─ 使用标准 Cross-Entropy Loss
  └─ 目的：让 Weaver 学会生成有意义的隐记忆

Stage 2: Weaver GRPO
  ├─ 固定 Trigger（固定输出 INVOKE）
  ├─ 生成多个候选轨迹
  ├─ 奖励由任务完成情况驱动
  └─ 目的：优化 Weaver 的记忆生成策略

Stage 3: Trigger GRPO
  ├─ 固定 Weaver
  ├─ Trigger 在分隔符位置决策 INVOKE/SKIP
  ├─ 奖励同样由任务完成驱动
  └─ 目的：学会在恰当的时机调用记忆
```

---

## 注意事项

1. **左填充（Left Padding）**: MemGen 批量生成时使用左填充，这与一般习惯不同，是为了配合自回归生成中的 KV cache 管理
2. **KV Cache 失效**: 注入隐记忆后，KV cache 需要重建
3. **supervised_labels**: MemGenModel 自定义的输出，用于标识哪些位置是 agent 生成的内容（排除 latent token 位置）
4. **info_mask**: 用于 GRPO 训练，区分 prompt 部分和 completion 部分
5. **NumPy 随机种子**: `set_seed` 中 import NumPy 的方式在 Python 3.12+ 可能有问题，但这在代码中不影响功能
