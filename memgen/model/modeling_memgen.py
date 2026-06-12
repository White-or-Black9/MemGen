import logging
import os
import random
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    DynamicCache
)
from transformers.modeling_utils import PreTrainedModel

from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.modeling_utils import (
    MemGenOutputWithPast,
    MemGenLoraSwitchMixin,
    MemGenGenerationMixin,
)
from memgen.model.trigger import MemGenTrigger
from memgen.model.weaver import MemGenWeaver
from memgen.utils import (
    CONVERSATION_TEMPLATE,
    fix_model_parameters,
    log_trainable_params
)


class MemGenModel(PreTrainedModel, MemGenLoraSwitchMixin, MemGenGenerationMixin):
    """
    ================================================================
    MemGenModel —— MemGen 的主模型类
    ================================================================

    这是整个 MemGen 框架的"主板"，将三个核心组件整合在一起：

    1. Reasoner（推理核心）
       - 冻结的预训练 LLM，保持通用能力
       - 负责实际的 token 生成和推理
       - 从不更新参数

    2. Weaver（记忆编织器）
       - 带 LoRA adapter 的小型 LM
       - 接收 reasoner 的 hidden states 作为输入
       - 生成 latent token 序列作为"记忆"
       - 通过 SFT 或 GRPO 训练

    3. Trigger（记忆触发器）
       - 带 LoRA adapter 的小型 LM + 二分类头
       - 监控 reasoner 的推理状态
       - 在句子边界决定是否调用 Weaver
       - 通过 GRPO 训练

    架构流程图：
                              Trigger 决定是否激活
        输入 ──> Reasoner ──> ────────────────>  生成输出
                          \\                      /
                           \\──> Weaver ────> Latent Memory
                                 生成记忆    注入回 Reasoner

    数据流（训练时 _forward）：
        1. 选择 augmentation 点（在 delimiter 后）
        2. 遍历每个 augmentation 点：
           a. 将当前段输入 reasoner 获取 embeddings
           b. 用 projection 层映射到 weaver 空间
           c. Weaver 生成 latent hidden states
           d. 反向映射回 reasoner 空间
           e. 将 latent 注入到序列中
        3. Reasoner 处理增强后的完整序列
        4. 计算损失（跳过 latent 位置的 logits）
    """

    config_class = MemGenConfig
    INSTRUCTION_STATE = 0       # 单轮指令状态
    CONVERSATION_STATE = 1     # 多轮对话状态

    def __init__(
        self,
        config: MemGenConfig,
        base_tokenizer,
        reasoner_base_model: PreTrainedModel,
        weaver_base_model: PreTrainedModel,
        trigger_base_model: PreTrainedModel,
    ):
        super().__init__(config)

        self.config = config

        # ═══ Step 1: 注入 LoRA adapter ═══
        weaver_model_w_lora, trigger_model_w_lora = self._insert_lora_adapters(
            weaver_base_model, config.weaver_lora_config,
            trigger_base_model, config.trigger_lora_config,
        )

        # ═══ Step 2: 初始化三个核心组件 ═══
        self.weaver = MemGenWeaver(weaver_model_w_lora, config.prompt_latents_len, config.inference_latents_len)
        self.trigger = MemGenTrigger(trigger_model_w_lora, config.trigger_active)

        # reasoner（始终冻结）
        self.reasoner = reasoner_base_model
        self.tokenizer = base_tokenizer

        # ═══ Step 3: 投影层 ═══
        # reasoner_hidden -> weaver_hidden：将 reasoner 的 embeddings 映射到 weaver 空间
        self.reasoner_to_weaver = nn.Linear(reasoner_base_model.config.hidden_size, weaver_base_model.config.hidden_size)
        # weaver_hidden -> reasoner_hidden：将 weaver 生成的 latent 映射回 reasoner 空间
        self.weaver_to_reasoner = nn.Linear(weaver_base_model.config.hidden_size, reasoner_base_model.config.hidden_size)

        # ═══ Step 4: 句子边界检测 ═══
        # 只在逗号、句号、换行符后触发记忆调用（paper Section 4.2 的 sentence-granularity 策略）
        self.delimiters: list[str] = [",", ".", "\n"]

        self.state = None

        # ═══ Step 5: 后处理 ═══
        self._postprocess_models()
        logging.info("##### MemGen Initialization #####")
        log_trainable_params(self)

    def _postprocess_models(self):
        """冻结 reasoner 参数，确保 tokenizer 配置正确。"""
        fix_model_parameters(self.reasoner)  # Reasoner 永远冻结！

        # 确保 tokenizer 有 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"
            logging.info(
                f"Tokenizer has no pad token. Using EOS token ({self.tokenizer.eos_token}) as pad token."
            )

        # 规范化 chat template
        self.tokenizer.chat_template = CONVERSATION_TEMPLATE

    @property
    def device(self):
        return self.reasoner.device

    # ====================================================================
    # 前向传播（训练）
    # ====================================================================

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        单轮增强前向传播——MemGen 的核心算法实现。

        对应 paper Section 4 的"interleaving memory and reasoning"：
        1. 选择增强点（在 delimiter 位置之后）
        2. 遍历增强点，分段处理：
           a. 当前段（从上一个增强点到当前增强点）交给 reasoner 获取 embeddings
           b. 映射到 weaver 空间，weaver 生成 latent memory
           c. 映射回 reasoner 空间，注入到序列中
        3. Reasoner 处理增强后的完整序列
        4. 从 logits 中剔除 latent 位置，得到有效 logits

        关键设计：
        - 增强相当于在原始 token 序列中插入额外的 "记忆 token"
        - 这些记忆 token 在 loss 计算中被跳过（避免让 reasoner 预测记忆）
        - Reasoner 只预测它自己生成的 token
        """
        assert input_ids.shape == attention_mask.shape == labels.shape

        tokenizer = self.tokenizer
        reasoner = self.reasoner
        weaver = self.weaver
        delimiters = self.delimiters
        max_augment_num = self.config.max_inference_aug_num
        device = self.device
        embeds_dtype = reasoner.get_input_embeddings().weight.dtype
        B, _ = input_ids.shape
        hidden_size = self.config.hidden_size

        # Step 1: 选择增强点（只在 delimiter 后）
        augmentation_indices = self._select_augment_points_after_delimiter(
            input_ids, labels, delimiters, tokenizer, max_augment_num
        )

        # Step 2: 获取 reasoner 的输入 embeddings
        inputs_embeds = reasoner.get_input_embeddings()(input_ids)

        # Step 3: 遍历增强点，逐段注入 latent memory
        current_start_idx = 0
        current_inputs_embeds = torch.empty((B, 0, hidden_size), device=device, dtype=embeds_dtype)
        current_attention_mask = torch.empty((B, 0), device=device, dtype=attention_mask.dtype)
        current_latents_mask = torch.empty((B, 0), device=device, dtype=torch.bool)

        for aug_point_idx in augmentation_indices:
            # 3a: 取当前段（从上一个增强点到此增强点）
            segment_inputs_embeds = inputs_embeds[:, current_start_idx:aug_point_idx]
            segment_attention_mask = attention_mask[:, current_start_idx:aug_point_idx]
            segment_latents_mask = torch.zeros((B, segment_inputs_embeds.size(1)), device=device, dtype=torch.bool)

            current_inputs_embeds = torch.cat([current_inputs_embeds, segment_inputs_embeds], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, segment_attention_mask], dim=1)
            current_position_ids = self._generate_position_ids(current_attention_mask)
            current_latents_mask = torch.cat([current_latents_mask, segment_latents_mask], dim=1)

            # 3b: 映射到 weaver 空间
            weaver_inputs_embeds = self.reasoner_to_weaver(current_inputs_embeds)

            # 3c: 判断当前是 prompt 增强还是 inference 增强
            is_prompt_end_aug = (labels[:, aug_point_idx] != -100).all() and (labels[:, aug_point_idx - 1] == -100).all().item()

            if is_prompt_end_aug:
                weaver_hidden_states, attn_mask, pos_ids = weaver.augment_prompt(
                    weaver_inputs_embeds, current_attention_mask, current_position_ids
                )
            else:
                weaver_hidden_states, attn_mask, pos_ids = weaver.augment_inference(
                    weaver_inputs_embeds, current_attention_mask, current_position_ids
                )

            # 3d: 映射回 reasoner 空间并注入
            latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states)

            current_inputs_embeds = torch.cat([current_inputs_embeds, latent_inputs_embeds], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, attn_mask], dim=1)
            current_start_idx = aug_point_idx

            latent_mask = torch.ones((B, latent_inputs_embeds.size(1)), device=device, dtype=torch.bool)
            current_latents_mask = torch.cat([current_latents_mask, latent_mask], dim=1)

        # 3e: 处理最后一段（最后一个增强点之后）
        remaining_inputs_embeds = inputs_embeds[:, current_start_idx:]
        remaining_attention_mask = attention_mask[:, current_start_idx:]
        latent_mask = torch.zeros((B, remaining_attention_mask.size(1)), device=device, dtype=torch.bool)

        current_inputs_embeds = torch.cat([current_inputs_embeds, remaining_inputs_embeds], dim=1)
        current_attention_mask = torch.cat([current_attention_mask, remaining_attention_mask], dim=1)
        current_position_ids = self._generate_position_ids(current_attention_mask)
        current_latents_mask = torch.cat([current_latents_mask, latent_mask], dim=1)

        # Step 4: Reasoner 处理增强后的完整序列
        reasoner_outputs = reasoner(
            inputs_embeds=current_inputs_embeds,
            attention_mask=current_attention_mask,
            position_ids=current_position_ids
        )
        logits = reasoner_outputs.logits

        # Step 5: 剔除 latent 位置的 logits（skip latents for loss calc）
        shifted = torch.zeros_like(current_latents_mask)
        shifted[:, :-1] = current_latents_mask[:, 1:]
        valid_mask = ~shifted

        valid_logits = logits[valid_mask].view(logits.size(0), -1, logits.size(2))
        return valid_logits

    def _instructional_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        单轮指令数据的 forward pass。

        适用于 SFT 数据，直接委托给 _forward() 处理。
        """
        logits = self._forward(input_ids, attention_mask, labels, **kwargs)
        return logits, labels

    def _conversational_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        """
        多轮对话数据的 forward pass。

        多轮对话的处理策略：
        1. 用 labels 识别每轮对话的起止位置
        2. 每轮独立调用 _forward()
        3. 第 i 轮的 latents 在第 i+1 轮不可见（保持轮次隔离）

        Args:
            input_ids: [1, L] 必须是 batch_size=1
        """
        assert input_ids.shape[0] == 1, "Conversational SFT currently only supports batch_size = 1"
        seq_len = input_ids.shape[1]
        vocab_size = self.config.vocab_size
        device = input_ids.device

        # 识别对话中每轮"assistant 回复"的起止位置
        label_row = labels[0]
        should_supervise = label_row != -100
        if not should_supervise.any():
            raise ValueError("At least one completion segment is required")

        valid_mask = should_supervise.int()
        diff = torch.diff(torch.cat([torch.tensor([0], device=device), valid_mask]))
        valid_starts = (diff == 1).nonzero(as_tuple=True)[0].tolist()
        ends = (diff == -1).nonzero(as_tuple=True)[0].tolist()
        if len(ends) < len(valid_starts):
            ends.append(seq_len)
        assert len(valid_starts) == len(ends)

        # 构建 (前一段起始, 监督段起始, 监督段结束) 的三元组
        triplets = []
        start = 0
        for s, e in zip(valid_starts, ends):
            triplets.append((start, s, e))
            start = e

        # 如果超过 max_prompt_aug_num 轮，随机选择
        if len(triplets) <= self.config.max_prompt_aug_num:
            select_turns = [1] * len(triplets)
        else:
            triplets_num = len(triplets)
            selected_indices = set(random.sample(range(triplets_num), self.config.max_prompt_aug_num))
            select_turns = [1 if i in selected_indices else 0 for i in range(triplets_num)]

        # 逐轮处理
        all_logits = torch.zeros(1, seq_len, vocab_size, device=device)
        all_labels = torch.full((1, seq_len), -100, device=device)

        for triplet, should_supervise in zip(triplets, select_turns):
            start, valid_start, end = triplet
            if should_supervise:
                cur_input_ids = input_ids[0, :end].unsqueeze(0)
                cur_attention = attention_mask[0, :end].unsqueeze(0)
                cur_labels = torch.full((1, end), -100, device=device)
                cur_labels[0, valid_start:end] = labels[0, valid_start:end]

                logits = self._forward(cur_input_ids, cur_attention, cur_labels, **kwargs)

                all_logits[0, start:end, :] = logits[0, start:end, :]
                all_labels[0, start:end] = labels[0, start:end]

        return all_logits, all_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> MemGenOutputWithPast:
        """
        统一的前向传播入口。

        流程：
        1. 后处理 labels（mask 掉格式 token）
        2. 检测是单轮指令还是多轮对话
        3. 分 batch 调用对应的 forward 函数
        4. 计算交叉熵损失
        """
        assert labels is not None, "Reasoning Processor requires input labels for training"

        labels = self._postprocess_assistant_labels(input_ids, labels, tokenizer=self.tokenizer)

        # 用第一个样本检测模式
        if self.state is None:
            self.state = MemGenModel.CONVERSATION_STATE if self._is_conversation(input_ids, self.tokenizer) else MemGenModel.INSTRUCTION_STATE

        forward_func = self._instructional_forward if self.state == MemGenModel.INSTRUCTION_STATE else self._conversational_forward

        batch_size = 1
        iter_num = input_ids.size(0) // batch_size

        logits, supervised_labels = [], []
        for i in range(iter_num):
            batch_input_ids = input_ids[i * batch_size: (i + 1) * batch_size]
            batch_attention_mask = attention_mask[i * batch_size: (i + 1) * batch_size]
            batch_labels = labels[i * batch_size: (i + 1) * batch_size]

            batch_logits, batch_supervised_labels = forward_func(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels,
                **kwargs
            )
            logits.append(batch_logits)
            supervised_labels.append(batch_supervised_labels)

        all_logits = torch.concat(logits, dim=0)
        all_labels = torch.concat(supervised_labels, dim=0)

        # 计算因果语言模型损失
        shift_logits = all_logits[..., :-1, :].contiguous()
        shift_labels = all_labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        outputs = MemGenOutputWithPast(loss=loss, logits=all_logits)
        outputs.supervised_labels = all_labels
        return outputs

    # ====================================================================
    # 生成（推理/评估）
    # ====================================================================

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: GenerationConfig = None,
        return_augmentation_mask: bool = False,
        latent_memory_bank: Optional[object] = None,
        **kwargs
    ) -> Union[torch.LongTensor, tuple[torch.LongTensor, torch.LongTensor]]:
        """
        带记忆增强的自回归生成。

        这是 MemGen 在推理时的核心循环，对应 paper Figure 2：
        1. 初始化：将 prompt 转换为 embeddings
        2. 对每个生成步骤：
           a. Trigger 检查当前是否在 delimiter 位置
           b. 如果是，Trigger 决定是否 INVOKE 或 SKIP
           c. 如果 INVOKE：
              - Weaver 根据当前 hidden states 生成 latent memory
              - 将 latent memory 注入回 reasoner 的 hidden states
              - 更新 embedding 序列（重新计算 KV cache）
           d. Reasoner 生成下一个 token
        3. 直到生成结束或达到最大增强次数

        augmentation_pos 记录：
            -100: 该位置没有被触发 trigger 检查
            0: trigger 检查了但决定 SKIP
            1: trigger 检查了且决定 INVOKE（实际插入记忆）
        """
        tokenizer = self.tokenizer
        reasoner = self.reasoner
        weaver = self.weaver
        max_augment_num = self.config.max_inference_aug_num
        invalid_token_id = -100

        # 预处理
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        max_new_tokens = generation_config.max_new_tokens
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        prompt_len = input_ids.size(1)

        inputs_embeds = reasoner.get_input_embeddings()(input_ids)
        B, _, hidden_size = inputs_embeds.shape
        device = inputs_embeds.device
        if latent_memory_bank is not None and B != 1:
            raise ValueError(
                "latent_memory_bank-enabled generate currently supports batch_size=1 only"
            )

        # 初始化生成循环
        current_inputs_embeds = inputs_embeds
        current_attention_mask = attention_mask
        current_position_ids = self._generate_position_ids(current_attention_mask)
        current_input_ids = input_ids
        current_cache: DynamicCache = None

        sentence_augment_count = torch.zeros(B, dtype=torch.int, device=device)

        # augmentation_pos 记录每个 token 的增强状态
        augmentation_pos = torch.full((B, max_new_tokens), fill_value=invalid_token_id, device=device)

        # ═══ 生成主循环 ═══
        for i in range(max_new_tokens):
            assert current_inputs_embeds.shape[:2] == current_attention_mask.shape == current_position_ids.shape

            # Step 1: Trigger 决定是否增强
            augment_decision = self._should_augment(
                current_input_ids,
                sentence_augment_count=sentence_augment_count,
                do_sample=generation_config.trigger_do_sample,
                temperature=generation_config.temperature,
                is_prompt=(i == 0)
            )
            augmentation_pos[:, i] = augment_decision
            augment_indices = torch.where(augment_decision == 1)[0]

            # Step 2: 如果需要增强，调用 Weaver
            if len(augment_indices) > 0:
                if i != 0:
                    sentence_augment_count[augment_indices] += 1

                candidate_inputs_embeds = current_inputs_embeds[augment_indices]
                candidate_attention_mask = current_attention_mask[augment_indices]
                candidate_position_ids = current_position_ids[augment_indices]

                # 映射到 weaver 空间，生成 latent memory
                weaver_inputs_embeds = self.reasoner_to_weaver(candidate_inputs_embeds)
                if i == 0:
                    weaver_hidden_states, attn_mask, _ = weaver.augment_prompt(
                        weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids
                    )
                else:
                    weaver_hidden_states, attn_mask, _ = weaver.augment_inference(
                        weaver_inputs_embeds, candidate_attention_mask, candidate_position_ids
                    )
                latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states)
                if latent_memory_bank is None:
                    # Preserve the original disabled path exactly.
                    candidate_inputs_embeds = torch.cat(
                        [candidate_inputs_embeds, latent_inputs_embeds], dim=1
                    )
                    candidate_attention_mask = torch.cat(
                        [candidate_attention_mask, attn_mask], dim=1
                    )
                    pad_len = (
                        weaver.prompt_latents_num
                        if i == 0
                        else weaver.inference_latents_num
                    )
                else:
                    retrieved_slots = latent_memory_bank.retrieve(
                        candidate_inputs_embeds.detach(),
                        device=device,
                        dtype=current_inputs_embeds.dtype,
                    )
                    if retrieved_slots:
                        retrieved_memory = torch.cat(
                            [slot.memory for slot in retrieved_slots], dim=0
                        ).unsqueeze(0)
                        retrieved_attention_mask = torch.ones(
                            (1, retrieved_memory.size(1)),
                            device=device,
                            dtype=current_attention_mask.dtype,
                        )
                        candidate_inputs_embeds = torch.cat(
                            [
                                candidate_inputs_embeds,
                                retrieved_memory,
                                latent_inputs_embeds,
                            ],
                            dim=1,
                        )
                        candidate_attention_mask = torch.cat(
                            [
                                candidate_attention_mask,
                                retrieved_attention_mask,
                                attn_mask,
                            ],
                            dim=1,
                        )
                    else:
                        candidate_inputs_embeds = torch.cat(
                            [candidate_inputs_embeds, latent_inputs_embeds], dim=1
                        )
                        candidate_attention_mask = torch.cat(
                            [candidate_attention_mask, attn_mask], dim=1
                        )
                    latent_memory_bank.write(latent_inputs_embeds.detach())
                    pad_len = candidate_inputs_embeds.size(1) - current_inputs_embeds.size(1)

                # 合并增强和未增强的序列
                new_len = candidate_inputs_embeds.size(1)
                merged_inputs_embeds = torch.zeros((B, new_len, hidden_size), device=device, dtype=current_inputs_embeds.dtype)
                merged_attention_mask = torch.zeros((B, new_len), device=device, dtype=current_attention_mask.dtype)

                merged_inputs_embeds[augment_indices] = candidate_inputs_embeds
                merged_attention_mask[augment_indices] = candidate_attention_mask

                # 未增强的序列做左填充对齐
                non_augment_indices = torch.where(augment_decision != 1)[0]
                if len(non_augment_indices) > 0:
                    non_aug_inputs_embeds = current_inputs_embeds[non_augment_indices]
                    non_aug_attention_mask = current_attention_mask[non_augment_indices]
                    non_aug_inputs_embeds, non_aug_attention_mask, _ = self._left_pad(
                        non_aug_inputs_embeds, non_aug_attention_mask, None, pad_len
                    )

                    merged_inputs_embeds[non_augment_indices] = non_aug_inputs_embeds
                    merged_attention_mask[non_augment_indices] = non_aug_attention_mask

                current_inputs_embeds = merged_inputs_embeds
                current_attention_mask = merged_attention_mask
                current_position_ids = self._generate_position_ids(current_attention_mask)
                current_cache = None  # 注入后 KV cache 失效，需重建

            # Step 3: 检查是否达到最大增强次数，如果是则一次生成剩余 token
            if (sentence_augment_count >= max_augment_num).all():
                generation_config_continue = GenerationConfig(
                    do_sample=generation_config.weaver_do_sample,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    use_cache=False,
                    max_new_tokens=max_new_tokens - i
                )
                generated = reasoner.generate(
                    inputs_embeds=current_inputs_embeds,
                    attention_mask=current_attention_mask,
                    generation_config=generation_config_continue
                )
                current_input_ids = torch.cat([current_input_ids, generated], dim=1)
                break

            # Step 4: Reasoner 生成下一个 token
            if current_cache is not None:
                reasoner_inputs_embeds = current_inputs_embeds[:, -1:]
                reasoner_position_ids = current_position_ids[:, -1:]
            else:
                reasoner_inputs_embeds = current_inputs_embeds
                reasoner_position_ids = current_position_ids

            outputs = reasoner(
                inputs_embeds=reasoner_inputs_embeds,
                attention_mask=current_attention_mask,
                position_ids=reasoner_position_ids,
                output_hidden_states=False,
                use_cache=True,
                past_key_values=current_cache
            )

            current_inputs_embeds, current_attention_mask, current_position_ids, current_input_ids = self._append_one_step(
                outputs,
                current_inputs_embeds,
                current_attention_mask,
                current_position_ids,
                current_input_ids,
                do_sample=generation_config.weaver_do_sample,
                temperature=generation_config.temperature
            )
            current_cache = outputs.past_key_values

            # Step 5: 检查是否全部生成了 EOS token
            if (current_input_ids[:, -1] == eos_token_id).all():
                break

            del outputs

        # 后处理
        new_generated_len = current_input_ids.size(1) - prompt_len
        augmentation_pos = augmentation_pos[:, :new_generated_len]

        self._check_generate(
            current_input_ids[:, prompt_len:],
            augmentation_pos
        )

        if return_augmentation_mask:
            return (current_input_ids, augmentation_pos)
        else:
            return current_input_ids

    # ====================================================================
    # 模型保存与加载
    # ====================================================================

    @classmethod
    def from_config(cls, config_dict: dict):
        """
        从配置字典构建 MemGenModel。

        加载流程：
        1. 解析配置参数
        2. 加载三个 LLM（reasoner, weaver, trigger）
        3. 如果指定了 load_model_path，从 checkpoint 加载权重
        4. 否则初始化新模型
        """
        model_name = config_dict.get("model_name")
        max_prompt_aug_num = config_dict.get("max_prompt_aug_num", 1)
        max_inference_aug_num = config_dict.get("max_inference_aug_num", 5)

        weaver_config = config_dict.get("weaver", {})
        prompt_latents_len = weaver_config.get("prompt_latents_len", 8)
        inference_latents_len = weaver_config.get("inference_latents_len", 8)
        weaver_lora_config_dict = weaver_config.get("lora_config", None)
        weaver_model_name = weaver_config.get("model_name", None)

        trigger_config = config_dict.get("trigger", {})
        trigger_active = trigger_config.get("active", False)
        trigger_lora_config_dict = trigger_config.get("lora_config", None)
        trigger_model_name = trigger_config.get("model_name", None)

        # 构建 MemGenConfig
        from transformers import AutoConfig
        memgen_config = AutoConfig.from_pretrained(model_name)
        memgen_config = MemGenConfig.from_pretrained(
            model_name,
            max_prompt_aug_num=max_prompt_aug_num,
            max_inference_aug_num=max_inference_aug_num,
            prompt_latents_len=prompt_latents_len,
            inference_latents_len=inference_latents_len,
            weaver_lora_config=weaver_lora_config_dict,
            trigger_active=trigger_active,
            trigger_lora_config=trigger_lora_config_dict
        )

        # 加载预训练模型
        base_tokenizer = AutoTokenizer.from_pretrained(model_name)
        reasoner_base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        weaver_base_model = AutoModelForCausalLM.from_pretrained(weaver_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        trigger_base_model = AutoModelForCausalLM.from_pretrained(trigger_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

        load_model_path = config_dict.get("load_model_path", None)

        if not load_model_path:
            model = cls(
                config=memgen_config,
                base_tokenizer=base_tokenizer,
                reasoner_base_model=reasoner_base_model,
                weaver_base_model=weaver_base_model,
                trigger_base_model=trigger_base_model
            )
        else:
            model = cls.from_pretrained(
                load_model_path,
                config=memgen_config,
                base_tokenizer=base_tokenizer,
                reasoner_base_model=reasoner_base_model,
                weaver_base_model=weaver_base_model,
                trigger_base_model=trigger_base_model
            )

        return model

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        保存模型权重到指定目录。

        保存的内容（分开存储）：
        - config.json: MemGenConfig 配置
        - projs.bin: 投影层权重（reasoner_to_weaver, weaver_to_reasoner）
        - weaver.bin: Weaver 的 query latents + LN + scale
        - trigger.bin: Trigger 的分类头
        - weaver/: Weaver 的 LoRA adapter 权重
        - trigger/: Trigger 的 LoRA adapter 权重
        """
        os.makedirs(save_directory, exist_ok=True)

        self.config.save_pretrained(save_directory)

        torch.save(
            {
                "reasoner_to_weaver": self.reasoner_to_weaver.state_dict(),
                "weaver_to_reasoner": self.weaver_to_reasoner.state_dict(),
            },
            os.path.join(save_directory, "projs.bin"),
        )

        torch.save(
            {
                "prompt_query_latents": self.weaver.prompt_query_latents.data,
                "inference_query_latents": self.weaver.inference_query_latents.data,
                "prompt_latent_ln": self.weaver.prompt_latent_ln.state_dict(),
                "inference_latent_ln": self.weaver.inference_latent_ln.state_dict(),
                "prompt_latent_scale": self.weaver.prompt_latent_scale.data,
                "inference_latent_scale": self.weaver.inference_latent_scale.data,
            },
            os.path.join(save_directory, "weaver.bin"),
        )

        torch.save(
            {
                "output_layer": self.trigger.output_layer.state_dict(),
            },
            os.path.join(save_directory, "trigger.bin"),
        )

        self.weaver.model.save_pretrained(os.path.join(save_directory, "weaver"))
        self.trigger.model.save_pretrained(os.path.join(save_directory, "trigger"))

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
        *,
        config,
        base_tokenizer,
        reasoner_base_model,
        weaver_base_model,
        trigger_base_model,
    ):
        """
        从已保存的 checkpoint 加载模型权重。

        与 save_pretrained 对应，分别加载投影层、weaver 和 trigger 的权重。
        """
        model = cls(
            config=config,
            base_tokenizer=base_tokenizer,
            reasoner_base_model=reasoner_base_model,
            weaver_base_model=weaver_base_model,
            trigger_base_model=trigger_base_model,
        )

        # 加载投影层
        proj_path = os.path.join(load_directory, "projs.bin")
        proj_state = torch.load(proj_path, map_location="cpu")
        model.reasoner_to_weaver.load_state_dict(proj_state["reasoner_to_weaver"])
        model.weaver_to_reasoner.load_state_dict(proj_state["weaver_to_reasoner"])

        # 加载 Weaver 参数
        weaver_path = os.path.join(load_directory, "weaver.bin")
        weaver_state = torch.load(weaver_path, map_location="cpu")
        model.weaver.prompt_query_latents.data.copy_(weaver_state["prompt_query_latents"])
        model.weaver.inference_query_latents.data.copy_(weaver_state["inference_query_latents"])
        model.weaver.prompt_latent_ln.load_state_dict(weaver_state["prompt_latent_ln"])
        model.weaver.inference_latent_ln.load_state_dict(weaver_state["inference_latent_ln"])
        model.weaver.prompt_latent_scale.data.copy_(weaver_state["prompt_latent_scale"])
        model.weaver.inference_latent_scale.data.copy_(weaver_state["inference_latent_scale"])

        # 加载 Trigger 参数
        trigger_path = os.path.join(load_directory, "trigger.bin")
        trigger_state = torch.load(trigger_path, map_location="cpu")
        model.trigger.output_layer.load_state_dict(trigger_state["output_layer"])

        # Replace the initialization adapters with the checkpoint adapters.
        # Loading onto base_model would wrap the existing LoraModel a second time.
        model.weaver.model.delete_adapter(MemGenWeaver.adapter_name)
        model.weaver.model.load_adapter(
            os.path.join(load_directory, "weaver", "weaver"),
            adapter_name=MemGenWeaver.adapter_name,
        )
        model.weaver.model.set_adapter(MemGenWeaver.adapter_name)

        model.trigger.model.delete_adapter(MemGenTrigger.adapter_name)
        model.trigger.model.load_adapter(
            os.path.join(load_directory, "trigger", "trigger"),
            adapter_name=MemGenTrigger.adapter_name,
        )
        model.trigger.model.set_adapter(MemGenTrigger.adapter_name)

        logging.info("##### MemGen from Pretrained #####")
        log_trainable_params(model)

        return model
