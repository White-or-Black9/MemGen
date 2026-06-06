from peft import PeftModel
import torch
import torch.nn as nn


class MemGenWeaver(nn.Module):
    """
    MemGen 的 Memory Weaver（记忆编织器）模块。

    核心功能：
    - 接收 agent 当前的推理状态作为 "hook"（刺激信号）
    - 通过一个小型 LM 处理当前上下文 + 可学习的 latent query 嵌入
    - 生成固定长度的 latent token 序列作为机器原生记忆
    - 将这些 latent 记忆注入到 reasoner 的 hidden states 中

    类比人类记忆：
    - prompt_query_latents: 类似"任务规划记忆"——在任务开始时设定整体策略
    - inference_query_latents: 类似"工作记忆/程序记忆"——在推理中提供步骤级提示

    与 paper 的对应关系：
    - W_weaver(H_{t,<j}) -> M_t = [m_{t,1}, ..., m_{t,K}] (公式 5)
    - 输出 latent hidden states 会被投影回 reasoner 空间并 prepend 到 reasoner 的 hidden states (公式 6)

    可训练参数：
    1. prompt_query_latents: prompt 阶段的 latent 查询嵌入
    2. inference_query_latents: 推理阶段的 latent 查询嵌入
    3. LoRA adapter 参数
    4. LayerNorm 和 scale 参数
    """

    adapter_name = "weaver"

    def __init__(
        self,
        model: PeftModel,
        prompt_latents_len: int,
        inference_latents_len: int,
    ):
        super().__init__()

        self.model = model                                              # 带 LoRA 的小型 LM
        hidden_size = model.base_model.config.hidden_size

        # --- 可学习的 latent query 嵌入 ---
        # prompt_query_latents: 在 prompt 结束时插入的 latent token 序列
        self.prompt_query_latents = nn.Parameter(
            torch.randn(prompt_latents_len, hidden_size),
            requires_grad=True
        )
        # inference_query_latents: 在推理过程中插入的 latent token 序列
        self.inference_query_latents = nn.Parameter(
            torch.randn(inference_latents_len, hidden_size),
            requires_grad=True
        )

        # --- 归一化和缩放 ---
        # 对生成的 latent 做 layer norm（稳定训练）
        self.prompt_latent_ln = nn.LayerNorm(hidden_size)
        self.inference_latent_ln = nn.LayerNorm(hidden_size)
        # 可学习的缩放因子
        self.prompt_latent_scale = nn.Parameter(torch.ones(1))
        self.inference_latent_scale = nn.Parameter(torch.ones(1))

    @property
    def prompt_latents_num(self) -> int:
        """prompt 阶段的 latent token 数量（即 K_prompt）"""
        return self.prompt_query_latents.size(0)

    @property
    def inference_latents_num(self) -> int:
        """推理阶段的 latent token 数量（即 K_inference）"""
        return self.inference_query_latents.size(0)

    @property
    def device(self):
        assert self.prompt_query_latents.device == self.inference_query_latents.device
        return self.prompt_query_latents.device

    def _augment(
        self,
        latents: torch.Tensor,
        latent_ln: nn.LayerNorm,
        latent_scale: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        核心 augment 函数。

        处理流程（对应 paper Section 4.3）：
        1. 对 latent query 做 LayerNorm + scale 归一化
        2. 将归一化后的 latent 嵌入拼接到 inputs_embeds 末尾
        3. 扩展 attention_mask 和 position_ids 覆盖 latent 位置
        4. 运行 Weaver 的小型 LM 做一次完整前向传播
        5. 提取 latent 位置的 hidden states 作为"增强后的记忆"

        注意：这里的 latent query 相当于"记忆钩子"——Weaver 的 LM
        会根据当前上下文（inputs_embeds）和这个钩子，生成上下文相关的
        latent 记忆表示。

        Args:
            latents: latent query 嵌入矩阵 [latent_len, hidden_size]
            latent_ln: latent 的 LayerNorm
            latent_scale: latent 的缩放因子
            inputs_embeds: 当前上下文的嵌入 [B, L, H]
            attention_mask: 当前 attention mask [B, L]
            position_ids: 当前 position ids [B, L]

        Returns:
            latents_hidden_states: 生成的 latent 记忆 [B, latent_len, H]
            latents_mask: latent 位置的 mask [B, latent_len]
            latents_position_ids: latent 位置的 position ids [B, latent_len]
        """

        batch_size = attention_mask.shape[0]
        latents_num = latents.size(0)

        # Step 1: 归一化和缩放 latent query
        latents = latent_ln(latents) * latent_scale
        latents = latents.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, K, H]

        # Step 2: 将 latent 嵌入拼接到 inputs_embeds 末尾
        inputs_embeds = torch.cat([inputs_embeds, latents], dim=1)

        # Step 3: 扩展 attention mask（让 latent 位置可见）
        latents_mask = torch.ones(latents.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, latents_mask], dim=1)

        # Step 4: 扩展 position IDs（latent 位置紧随当前序列之后）
        last_position_ids = position_ids.max(dim=1)[0]
        latents_relative_positions = torch.arange(latents_num, device=attention_mask.device)
        latents_position_ids = last_position_ids.unsqueeze(1) + latents_relative_positions + 1
        position_ids = torch.cat([position_ids.long(), latents_position_ids.long()], dim=1)

        assert inputs_embeds.shape[:2] == attention_mask.shape == position_ids.shape

        # Step 5: 运行 Weaver 的 LM 前向传播，提取 hidden states
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        latents_hidden_states = hidden_states[:, -latents_num:, :]  # 只取 latent 位置的 hidden states

        return latents_hidden_states, latents_mask, latents_position_ids

    def augment_prompt(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        在 prompt 阶段进行记忆增强。
        用于在任务开始时生成"任务规划记忆"——类似于人类的"前额叶皮层"
        在开始一项任务时激活的规划能力。

        对应 paper 中的：prompt augmentation（公式 5 中的 M_t，用于 prompt 阶段）
        """
        return self._augment(
            latents=self.prompt_query_latents,
            latent_ln=self.prompt_latent_ln,
            latent_scale=self.prompt_latent_scale,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

    def augment_inference(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        在推理过程中进行记忆增强。
        用于生成"程序记忆/工作记忆"——类似于人类在执行任务时
        从海马体中提取相关经验的认知过程。

        对应 paper 中的：inference augmentation
        """
        return self._augment(
            latents=self.inference_query_latents,
            latent_ln=self.inference_latent_ln,
            latent_scale=self.inference_latent_scale,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
