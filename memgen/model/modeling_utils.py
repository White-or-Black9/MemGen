from dataclasses import dataclass
import logging
import os
from typing import Optional, Literal, Set

from peft import PeftModel, LoraConfig
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from memgen.model.trigger import MemGenTrigger
from memgen.model.weaver import MemGenWeaver
from memgen.utils import (
    CONVERSATION_TEMPLATE,
    fix_model_parameters,
    open_model_parameters
)


@dataclass
class MemGenOutputWithPast(CausalLMOutputWithPast):
    """
    自定义的输出类，在标准 CausalLMOutputWithPast 基础上增加了 supervised_labels。
    supervised_labels 用于记录哪些位置是受监督的（即需要计算 loss 的位置），
    在 GRPO 训练中用于区分 agent 回复和其他文本。
    """
    supervised_labels: Optional[torch.LongTensor] = None


class MemGenLoraSwitchMixin:
    """
    LoRA adapter 的插入和开关管理 Mixin。

    管理三个组件的 LoRA：
    - reasoner: 始终冻结（保持通用能力）
    - weaver: 可训练（编码经验知识）
    - trigger: 可训练（学习何时调用记忆）

    通过 fix_component / open_component 控制训练/冻结切换。
    """

    def _insert_lora_adapters(
        self,
        weaver_model: PreTrainedModel,
        weaver_lora_config: dict,
        trigger_model: PreTrainedModel,
        trigger_lora_config: dict
    ) -> tuple[PeftModel, PeftModel]:
        """
        向 weaver 和 trigger 的 base model 插入 LoRA adapter。

        LoRA 配置示例 (gsm8k.yaml):
            r: 16, lora_alpha: 32, target_modules: ["q_proj", "k_proj", ...]
        """
        weaver_lora_config = LoraConfig(**weaver_lora_config)
        trigger_lora_config = LoraConfig(**trigger_lora_config)

        weaver_model_with_lora = PeftModel(
            weaver_model, weaver_lora_config, adapter_name=MemGenWeaver.adapter_name
        )
        trigger_model_with_lora = PeftModel(
            trigger_model, trigger_lora_config, adapter_name=MemGenTrigger.adapter_name
        )

        return weaver_model_with_lora, trigger_model_with_lora

    def fix_component(self, name: Literal["weaver", "trigger"]):
        """冻结指定组件的所有参数（停止训练）。"""
        component = getattr(self, name)
        fix_model_parameters(component)
        if name == "weaver":
            # 同时冻结投影层
            fix_model_parameters(self.weaver_to_reasoner)
            fix_model_parameters(self.reasoner_to_weaver)

    def open_component(self, name: Literal["weaver", "trigger"]):
        """
        打开指定组件的参数（允许训练），但只训练 LoRA 参数。

        具体行为：
        - 先解冻所有参数
        - 重新冻结 base model（非 LoRA 参数）
        - 只保留 lora_A 和 lora_B 为可训练
        """
        component = getattr(self, name)
        open_model_parameters(component)
        if name == "weaver":
            open_model_parameters(self.weaver_to_reasoner)
            open_model_parameters(self.reasoner_to_weaver)

        # 只微调指定组件的 LoRA adapter，base model 保持冻结
        fix_model_parameters(component.model.base_model)

        # 验证：只有 lora_A/lora_B 是可训练的
        for n, p in component.model.named_parameters():
            if "lora_A" in n or "lora_B" in n:
                if name in n:
                    assert p.requires_grad, f"{n} should be trainable"
                else:
                    assert not p.requires_grad, f"{n} should be frozen"


class MemGenGenerationMixin(GenerationMixin):
    """
    MemGen 的自定义生成 Mixin。

    扩展了 HuggingFace 的 GenerationMixin，提供了：
    1. 增强点选择（_select_augment_points_after_delimiter）
    2. Trigger 决策（_should_augment）
    3. 单步生成和 KV-cache 管理（_append_one_step）
    4. 左填充对齐（_left_pad, _left_clip_pad_tokens）
    5. Delta 位置编码生成
    6. 对话/指令模式检测
    """

    def _get_next_token(
        self,
        next_token_logits: torch.Tensor,
        do_sample: bool,
        temperature: Optional[float] = 0.0
    ) -> torch.Tensor:
        """
        从 logits 中选择下一个 token。

        Args:
            do_sample=True: 用 temperature 采样的方式随机选择
            do_sample=False: 贪心解码，取概率最大的 token
        """
        if len(next_token_logits.shape) != 2:
            raise ValueError("Input logits must be a 2D tensor [batch_size, vocab_size]")

        if do_sample and temperature != 0:
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        else:
            return torch.argmax(next_token_logits, dim=-1, keepdim=True)

    def _generate_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        根据 attention_mask 生成 position IDs。
        使用累计和的方式，确保 padding 位置 position_id = 0。
        """
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        position_ids.masked_fill_(attention_mask == 0, 0)
        return position_ids

    def _is_conversation(self, input_ids: torch.Tensor, tokenizer) -> bool:
        """
        判断输入是否为多轮对话格式。

        检测方法：统计 <|im_start|>assistant 的出现次数。
        如果出现超过 1 次，说明是多轮对话。
        """
        if len(input_ids.shape) != 2:
            raise ValueError("input_ids must be a 2D tensor of shape (batch_size, seq_len)")

        seq = input_ids[0].tolist()

        im_start_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
        assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)

        target_seq = im_start_ids + assistant_ids

        count = 0
        for i in range(len(seq) - len(target_seq) + 1):
            if seq[i:i + len(target_seq)] == target_seq:
                count += 1

        return count > 1

    def _postprocess_assistant_labels(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer
    ) -> torch.Tensor:
        """
        后处理 labels：将 <|im_start|>assistant\\n 标记位置的 label 设为 -100。

        这是因为这些是格式 token，不应该参与 loss 计算。
        """
        if tokenizer.chat_template != CONVERSATION_TEMPLATE:
            raise ValueError(
                "Invalid tokenizer.chat_template detected.\n"
                f"Expected:\n{CONVERSATION_TEMPLATE}\n\n"
                f"Got:\n{tokenizer.chat_template}\n\n"
                "Please ensure that you are using the correct conversation template."
            )

        pattern_ids: list[int] = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

        batch_size, seq_len = input_ids.shape
        new_labels = labels.clone()

        for b in range(batch_size):
            seq = input_ids[b].tolist()
            for i in range(len(seq) - len(pattern_ids) + 1):
                if seq[i: i + len(pattern_ids)] == pattern_ids:
                    new_labels[b, i: i + len(pattern_ids)] = -100

        return new_labels

    def _get_delimiter_token_ids(self, tokenizer, delimiters: list[str]) -> Set[int]:
        """预计算 delimiter 对应的 token ids（避免重复 decode 的开销）。"""
        delimiter_token_ids = set()
        for d in delimiters:
            ids = tokenizer.encode(d, add_special_tokens=False)
            delimiter_token_ids.update(ids)
        return delimiter_token_ids

    def _check_ends_with_delimiter(
        self, input_ids: torch.Tensor, tokenizer, delimiters: list[str]
    ) -> torch.Tensor:
        """
        检查每个序列的最后一个有效 token（跳过 padding）是否是 delimiter token。

        高效实现：O(1) 每序列，无需 decode。
        用于在推理时判断当前是否到达句子边界，从而决定是否触发 trigger 决策。
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # 获取每个序列的最后一个非 padding token
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        mask = input_ids != pad_token_id
        last_positions = mask.sum(dim=1).clamp(min=1) - 1
        last_tokens = input_ids[torch.arange(batch_size, device=device), last_positions]

        # 缓存 delimiter token ids（只需计算一次）
        cache_key = '_delimiter_token_tensor'
        if not hasattr(self, cache_key):
            token_ids = self._get_delimiter_token_ids(tokenizer, delimiters)
            setattr(self, cache_key, torch.tensor(list(token_ids), device=device))

        delimiter_tensor = getattr(self, cache_key)
        is_delimiter = (last_tokens.unsqueeze(1) == delimiter_tensor).any(dim=1)

        return is_delimiter.unsqueeze(1)

    def _select_augment_points_after_delimiter(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        delimiters: list[str],
        tokenizer: PreTrainedTokenizerBase,
        max_num: int = 10,
    ) -> list[int]:
        """
        选择训练时在 labels 区域的增强点位置。

        实现 paper Section 4.2 中描述的 sentence-granularity 策略：
        - 只在 delimiter token（逗号、句号、换行符）之后的位置插入 latent 记忆
        - 在 prompt 结束时固定插入一次 prompt augmentation

        返回的位置列表在 _forward() 中被遍历，逐段处理。

        Args:
            input_ids: [B, L] 输入 token IDs
            labels: [B, L] 训练标签（-100 表示忽略）
            delimiters: 句子分隔符列表
            max_num: 最大增强点数量

        Returns:
            sorted list of augmentation positions (column indices)
        """
        assert input_ids.shape == labels.shape
        B, seq_len = input_ids.size(0), input_ids.size(1)

        prompt_augment_idx = []
        inference_augment_idx = []

        for i in range(1, seq_len):
            # 检测 prompt 结束位置：从 label=-100 变为 label≠-100 的位置
            if (labels[:, i] != -100).all() and (labels[:, i - 1] == -100).all():
                prompt_augment_idx.append(i)

            # 检测推理增强点：在 labels 区域内且前一个 token 是 delimiter
            elif (labels[:, i] != -100).all() and (labels[:, i - 1] != -100).all():
                batch_tokens_before_i = input_ids[:, :i]
                if self._check_ends_with_delimiter(batch_tokens_before_i, tokenizer, delimiters).any():
                    inference_augment_idx.append(i)

        # 验证：单轮数据必须恰好有一个 prompt augmentation 点
        if len(prompt_augment_idx) != 1:
            logging.error("Unexpected number of prompt augment indices: %s", prompt_augment_idx)
            logging.error("The inference_augment_idx: %s", inference_augment_idx)
            logging.error("Batch size = %d, seq_len = %d", B, seq_len)

            for b in range(B):
                ids = input_ids[b].tolist()
                labs = labels[b].tolist()
                toks = tokenizer.convert_ids_to_tokens(ids)

                logging.error("---- Sample %d ----", b)
                logging.error("Decoded text:\n%s", tokenizer.decode(ids, skip_special_tokens=False))

                vis = []
                for t, l in zip(toks, labs):
                    tag = "MASK" if l == -100 else "LAB"
                    vis.append(f"{t}<{tag}>")

                logging.error("Token-level view:\n%s", " ".join(vis))

                boundaries = []
                for i in range(1, seq_len):
                    if labs[i] != -100 and labs[i - 1] == -100:
                        boundaries.append(i)
                logging.error("Detected prompt->label boundaries at positions: %s", boundaries)
            raise ValueError("Single-turn forward must have exactly one prompt augment index")

        final_points = prompt_augment_idx[:1]

        # 限制推理增强点的数量
        if len(inference_augment_idx) > max_num:
            inference_augment_idx = inference_augment_idx[:max_num]

        final_points.extend(inference_augment_idx)

        if len(final_points) == 0:
            raise RuntimeError("No valid augmentation points found")

        final_points.sort()
        return final_points

    @torch.no_grad()
    def _should_augment(
        self,
        input_ids: torch.LongTensor,
        sentence_augment_count: torch.LongTensor,
        do_sample: bool,
        temperature: float,
        is_prompt: bool = False
    ) -> torch.LongTensor:
        """
        在生成（推理）时，决定当前步是否要调用 Weaver 进行记忆增强。

        这是 Trigger 在推理时的实际使用逻辑（对应 paper Section 4.2）：
        1. 只在 delimiter 位置触发 trigger 决策
        2. 超过最大增强次数后停止触发
        3. prompt 阶段（i==0）总是触发

        Args:
            input_ids: 当前已生成的 token IDs [B, L]
            sentence_augment_count: 每个序列已增强的次数 [B]
            do_sample: Trigger 是否使用采样
            temperature: 采样温度
            is_prompt: 是否为 prompt 阶段的第一次调用

        Returns:
            aug_vector: [B] 每个序列的决策
                - -100: 不触发 trigger（不在 delimiter 位置 或 超过限制）
                - 0: trigger 决策为 SKIP
                - 1: trigger 决策为 INVOKE
        """
        tokenizer = self.tokenizer
        delimiters = self.delimiters
        trigger = self.trigger
        max_augment_num = self.config.max_inference_aug_num

        batch_size = input_ids.size(0)

        if is_prompt:
            # prompt 阶段：所有序列都触发（位置 0）
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            position_ids = self._generate_position_ids(attention_mask)
            aug_vector = torch.zeros((batch_size,), dtype=torch.long, device=input_ids.device)
            trigger_indices = (aug_vector != -100).nonzero(as_tuple=True)[0]

        else:
            # 推理阶段：只在 delimiter 位置触发
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            position_ids = self._generate_position_ids(attention_mask)
            aug_vector = torch.full((batch_size,), -100, dtype=torch.long, device=input_ids.device)
            ends_with_delimiters = self._check_ends_with_delimiter(input_ids, tokenizer, delimiters).squeeze(1)
            aug_vector[ends_with_delimiters] = 0       # 标记为"待决策"
            over_limit = (sentence_augment_count >= max_augment_num)
            aug_vector[over_limit] = -100              # 超过限制的不触发
            trigger_indices = (aug_vector != -100).nonzero(as_tuple=True)[0]

        if trigger_indices.numel() > 0:
            # 用 trigger 模型做决策
            trigger_logits = trigger(
                input_ids=input_ids[trigger_indices],
                attention_mask=attention_mask[trigger_indices],
                position_ids=position_ids[trigger_indices]
            )
            last_token_logits = trigger_logits[:, -1]  # [n_active, 2]

            next_tokens = self._get_next_token(
                last_token_logits,
                do_sample=do_sample,
                temperature=temperature
            ).view(-1)

            aug_vector[trigger_indices] = next_tokens  # 0=SKIP, 1=INVOKE

        return aug_vector

    @torch.no_grad()
    def _append_one_step(
        self,
        reasoner_outputs,
        current_inputs_embeds: torch.Tensor,
        current_attention_mask: torch.Tensor,
        current_position_ids: torch.Tensor,
        current_input_ids: torch.Tensor,
        do_sample: bool,
        temperature: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将 reasoner 生成的下一个 token 追加到当前序列中。

        这是自回归生成的核心步骤：
        1. 从 reasoner 输出中取最后一个位置的 logits
        2. 采样/贪心选择下一个 token
        3. 更新 input_ids, inputs_embeds, attention_mask, position_ids
        """
        B = current_inputs_embeds.size(0)

        next_token_logits = reasoner_outputs.logits[:, -1]
        next_token_ids = self._get_next_token(next_token_logits, do_sample=do_sample, temperature=temperature)

        # 更新 input_ids
        current_input_ids = torch.cat([current_input_ids, next_token_ids], dim=1)

        # 更新 inputs_embeds
        next_token_embeds = self.reasoner.get_input_embeddings()(next_token_ids)
        current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embeds], dim=1)

        # 更新 attention mask
        attn_mask = torch.ones((B, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)
        current_attention_mask = torch.cat([current_attention_mask, attn_mask], dim=1)

        # 更新 position ids
        next_position_id = current_position_ids[:, -1:] + 1
        current_position_ids = torch.cat([current_position_ids, next_position_id], dim=1)

        return current_inputs_embeds, current_attention_mask, current_position_ids, current_input_ids

    @torch.no_grad()
    def _left_pad(
        self,
        input_embeds: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        position_ids: torch.LongTensor,
        pad_num: int
    ) -> tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        """
        在左侧填充指定数量的 0（用于对齐 batch 中不同长度的序列）。

        在 MemGen 中，当 batch 中某些序列被增强（插入了 latent）、
        而其他序列没有被增强时，需要做左填充来对齐。
        """
        if input_embeds is not None:
            B, L, D = input_embeds.shape
            pad_embeds = torch.zeros((B, pad_num, D), dtype=input_embeds.dtype, device=input_embeds.device)
            input_embeds = torch.cat([pad_embeds, input_embeds], dim=1)

        if attention_mask is not None:
            B = attention_mask.size(0)
            pad_mask = torch.zeros((B, pad_num), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([pad_mask, attention_mask], dim=1)

        if position_ids is not None:
            B = position_ids.size(0)
            pad_pos = torch.zeros((B, pad_num), dtype=position_ids.dtype, device=position_ids.device)
            position_ids = torch.cat([pad_pos, position_ids], dim=1)

        return input_embeds, attention_mask, position_ids

    @torch.no_grad()
    def _left_clip_pad_tokens(
        self, inputs_embeds: torch.FloatTensor, attention_mask: torch.LongTensor, position_ids: torch.LongTensor
    ) -> tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        """
        裁剪掉所有序列左侧公共的 padding tokens。

        在左填充后调用，用于恢复紧凑的内存布局。
        """
        B, L, D = inputs_embeds.shape

        # 找到每个序列的第一个非 padding token 的位置
        first_nonpad_idx = []
        for b in range(B):
            nonzero = (attention_mask[b] != 0).nonzero(as_tuple=True)[0]
            if len(nonzero) == 0:
                first_nonpad_idx.append(L)
            else:
                first_nonpad_idx.append(nonzero[0].item())

        # 取 batch 中最小的 padding 长度
        min_pad = min(first_nonpad_idx)

        if min_pad == 0:
            return inputs_embeds, attention_mask, position_ids

        # 裁剪
        inputs_embeds = inputs_embeds[:, min_pad:, :]
        attention_mask = attention_mask[:, min_pad:]
        position_ids = position_ids[:, min_pad:]

        return inputs_embeds, attention_mask, position_ids

    @torch.no_grad()
    def _check_generate(self, input_ids: torch.LongTensor, augmentation_pos: torch.LongTensor):
        """
        验证生成结果：检查 augmentation_pos=1 的位置之前是否确实是 delimiter。

        仅在 DEBUG_MODE 下启用，避免训练时的性能开销。
        """
        if os.environ.get('DEBUG_MODE', '').lower() != 'true':
            return

        delimiters = self.delimiters
        tokenizer = self.tokenizer

        B, L = input_ids.shape
        assert augmentation_pos.shape == input_ids.shape

        for b in range(B):
            for i in range(1, L):
                is_augment_point = augmentation_pos[b, i].item()

                if is_augment_point == -100:
                    continue

                if is_augment_point == 1 or is_augment_point == 0:
                    prefix_input_ids = input_ids[b, :i].unsqueeze(0)

                    ends_with_delimiter = self._check_ends_with_delimiter(
                        prefix_input_ids, tokenizer, delimiters
                    ).item()

                    if not ends_with_delimiter:
                        decoded_prefix = tokenizer.decode(prefix_input_ids.squeeze(0), skip_special_tokens=False)

                        raise ValueError(
                            f"Augmentation position error at batch {b}, index {i}. "
                            f"augmentation_pos is 1, but the prefix does NOT end with a delimiter.\n"
                            f"Prefix: '...{decoded_prefix[-50:]}'\n"
                            f"Delimiters: {delimiters}"
                        )
                else:
                    raise ValueError(
                        f"Invalid value in augmentation_pos at batch {b}, index {i}: {is_augment_point}. "
                        "Expected 1, 0, or -100."
                    )
