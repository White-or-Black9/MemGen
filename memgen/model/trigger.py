from peft import PeftModel
import torch
import torch.nn as nn


class MemGenTrigger(nn.Module):
    """
    MemGen 的 Memory Trigger（记忆触发器）模块。

    核心功能：
    - 作为"元认知监控器"，持续观察 reasoner 的推理状态
    - 在句子边界（delimiter 位置）决定是否调用记忆编织（weaver）
    - 输出二分类决策：INVOKE (1) 或 SKIP (0)

    训练方式：
    - 通过强化学习（GRPO）训练
    - 目标是平衡"在关键时机调用记忆"和"避免不必要的记忆调用"
    - 奖励由下游任务结果驱动（稀疏奖励）

    两种模式：
    1. active=True: 用 LoRA adapter + 分类器做出真正的决策
    2. active=False: 固定输出 INVOKE（用于消融或调试）
    """

    adapter_name = "trigger"

    def __init__(
        self,
        model: PeftModel,
        active: bool,
    ):
        super().__init__()

        self.active = active                             # 是否启用 trigger 的主动决策
        self.model = model                               # 带 LoRA 的小型 LM（通常与 reasoner 同架构）
        self.output_layer = nn.Linear(model.base_model.config.hidden_size, 2)  # 二分类输出头

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        position_ids: torch.Tensor
    ) -> torch.FloatTensor:
        """
        前向传播：对每个位置输出 INVOKE/SKIP 的 logits。

        Args:
            input_ids: shape [batch_size, seq_len]
            attention_mask: shape [batch_size, seq_len]
            position_ids: shape [batch_size, seq_len]

        Returns:
            logits: shape [batch_size, seq_len, 2]
                - logits[..., 0]: SKIP 的分数
                - logits[..., 1]: INVOKE 的分数
        """

        if self.active:
            # 主动决策模式：用 LoRA 模型提取 hidden states，通过线性层分类
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]    # 取最后一层 hidden states
            logits = self.output_layer(hidden_states)    # [B, L, 2]

        else:
            # 非主动模式：固定让所有位置都输出 INVOKE (logits[1] = 1.0)
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros(batch_size, seq_len, 2, device=input_ids.device)
            logits[..., 1] = 1.0                          # 所有位置倾向 INVOKE

        return logits
