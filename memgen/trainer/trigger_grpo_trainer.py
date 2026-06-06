from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import maybe_apply_chat_template
from trl.models import unwrap_model_for_generation, create_reference_model
from trl.trainer.utils import selective_log_softmax
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback
)
from peft import PeftConfig

from typing import Union, Callable, Optional, Any
from contextlib import nullcontext
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset
from accelerate.utils import gather_object

from interactions.base_interaction import InteractionDataProto
from interactions.tensor_utils import TensorHelper, TensorConfig

from memgen.trainer.utils import (
    nanstd,
    nanmax,
    nanmin,
    generate_position_ids
)
from memgen.model.modeling_memgen import MemGenModel

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class TriggerGRPOTrainer(GRPOTrainer):
    """
    Trigger 的 GRPO 强化学习训练器。

    核心功能：
    - 使用 GRPO 算法训练 Trigger（记忆触发器）
    - Trigger 学习在句子边界（分隔符位置）决定是否调用记忆编织
    - 奖励由下游任务完成情况驱动（稀疏奖励）

    与标准 GRPOTrainer 和 WeaverGRPOTrainer 的关键区别：
    1. _get_per_token_logps 基于 augmentation_mask 计算（不是用文本 completion）
    2. 通过 model.generate(return_augmentation_mask=True) 生成的同时获取 trigger 决策
    3. loss 在 augmentation_valid_mask 位置计算（不是 completion_mask）
    4. 使用 model.trigger 的子模块作为参考模型（而不是整个 MemGenModel）
    """

    def __init__(
        self,
        model: MemGenModel,
        processing_class: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        args: Optional[GRPOConfig] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional[PeftConfig] = None,
    ):
        # NOTE - 梯度累积需要缩放 loss。父类中 loss 缩放取决于模型是否接受 loss 相关的 kwargs。
        # 由于我们自己计算 loss，这个检查不重要。设置 self.model_accepts_loss_kwargs = False 以启用缩放。
        self.model_accepts_loss_kwargs = False

        super().__init__(
            model=model,
            args=args,
            reward_funcs=reward_funcs,
            reward_processing_classes=reward_processing_classes,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config
        )

        # If PEFT configuration is not provided, create a reference model based on the initial model.
        # 创建 Trigger 的参考模型（用于 KL 散度惩罚）
        # 注意：这里只复制 model.trigger 而不是整个 MemGenModel
        ref_model = create_reference_model(model.trigger)
        self.ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=self.processing_class.pad_token_id,
            max_prompt_length=self.max_prompt_length,
            max_obs_length=None,
            max_start_length=None
        ))

    def _set_signature_columns_if_needed(self):
        """
        重写签名列设置：不做过滤。

        注：如果 `self.args.remove_unused_columns` 为 True，非签名列会被移除。
        默认此方法将签名列设置为模型的预期输入。
        由于 TriggerGRPOTrainer 预处理数据，使用模型签名列不起作用。
        因此覆盖此方法使其不做任何操作。
        """
        pass

    def _get_per_token_logps(
        self,
        model,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        augmentation_mask: torch.LongTensor
    ) -> torch.Tensor:
        """
        计算 Trigger 在每个位置的 INVOKE/SKIP 对数概率。

        与 Weaver 不同，Trigger 不是在文本 token 上计算 logps，
        而是在每个分隔符位置计算"是否调用记忆"的二分类决策的 logps。

        处理流程：
        1. 计算 prompt 长度
        2. 生成 position_ids
        3. 调用 model.trigger() 获取每个位置 INVOKE/SKIP 的 logits
        4. 只保留 completion 部分（去掉 prompt 部分）
        5. 用 selective_log_softmax 计算二分类 logps
        6. 对无效位置（augmentation_mask == -100）将 logp 设为 0

        Args:
            model: trigger 模型
            input_ids: [B, L] 输入 token IDs
            attention_mask: [B, L] 注意力掩码
            augmentation_mask: [B, C] augmentation 掩码
                1 = INVOKE, 0 = SKIP, -100 = 无效位置

        Returns:
            logps: [B, C] 每个 completion 位置的 log 概率
        """
        prompt_len = attention_mask.size(1) - augmentation_mask.size(1)

        assert input_ids.shape == attention_mask.shape
        position_ids = generate_position_ids(attention_mask)
        augmentation_logits = model.trigger(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        # 取 completion 部分的 logits（注意 offset：trigger 在 prompt 最后一个 token 上的输出对应第一个 completion token）
        clipped_logits = augmentation_logits[:, prompt_len - 1: -1]
        assert clipped_logits.shape[:-1] == augmentation_mask.shape

        temp_mask = augmentation_mask.clone()
        augmentation_valid_mask = (temp_mask == -100).clone()

        temp_mask[augmentation_valid_mask] = 0  # 将无效位置设为 0（SKIP），避免计算 logps 时出错
        logps = selective_log_softmax(clipped_logits, temp_mask)
        logps[augmentation_valid_mask] = 0  # 无效位置的 logp 设为 0（不贡献梯度）

        return logps

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        生成轨迹并计算触发器的奖励和优势值。

        与 WeaverGRPOTrainer 的关键区别：
        1. 使用 model.generate(return_augmentation_mask=True) 获取 trigger 决策
        2. augmentation_mask 是 trigger 在每个位置的 INVOKE/SKIP 决策
        3. 奖励计算同时考虑任务完成情况和 augmentation_mask

        返回值中 augmentation_mask 会被反馈到 inputs，供 reward function 使用。
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        invalid_augmentation_id = -100

        # modified: pop those keys for generation
        batch_gen_keys = []
        if "prompt" in inputs[0]:
            batch_gen_keys.append("prompt")
        if "tools_kwargs" in inputs[0]:
            batch_gen_keys.append("tools_kwargs")
        if "interaction_kwargs" in inputs[0]:
            batch_gen_keys.append("interaction_kwargs")
        if "agent_name" in inputs[0]:
            batch_gen_keys.append("agent_name")
        if "env" in inputs[0]:
            batch_gen_keys.append("env")

        # build generation batch
        gen_batch = InteractionDataProto()
        for key in batch_gen_keys:
            gen_batch.no_tensor_batch[key] = [x[key] for x in inputs]

        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        gen_batch.batch["input_ids"] = prompt_ids.to(device)
        gen_batch.batch["attention_mask"] = prompt_mask.to(device)

        # Regular generation path
        # 使用 model.generate(return_augmentation_mask=True) 同时得到文本生成和 trigger 决策
        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            with (
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext()
            ):
                prompt_ids = gen_batch.batch["input_ids"]
                prompt_mask = gen_batch.batch["attention_mask"]
                # 关键调用：generate 返回 augmentation_mask（每个位置的 INVOKE/SKIP 决策）
                prompt_completion_ids, augmentation_mask = unwrapped_model.generate(
                    prompt_ids, prompt_mask, generation_config=self.generation_config, return_augmentation_mask=True
                )
                # Compute prompt length and extract completion ids
                prompt_length = prompt_ids.size(1)
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]
                assert completion_ids.shape == augmentation_mask.shape

            # Mask everything after the first EOS token
            # 将 EOS 之后的部分全部 mask 掉
            is_eos = completion_ids == self.processing_class.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            completion_ids = torch.where(
                completion_mask.bool(),
                completion_ids,
                torch.full_like(completion_ids, self.processing_class.eos_token_id)
            )

            # 只保留有效性位置的 augmentation 决策
            augmentation_valid_mask = completion_mask * (augmentation_mask != invalid_augmentation_id)
            augmentation_mask = torch.where(
                augmentation_valid_mask.bool(),
                augmentation_mask,
                torch.full_like(augmentation_mask, invalid_augmentation_id)
            )

        # If a truncation-based output strategy is used,
        # 如果使用截断策略，则对于未生成 EOS 的序列，忽略其 loss
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P + C)

        with torch.no_grad():
            # 计算 old_per_token_logps
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps = self._get_per_token_logps(
                    self.model.trigger, prompt_completion_ids, attention_mask, augmentation_mask
                )
            else:
                old_per_token_logps = None

            # 计算参考模型的 logps（用于 KL 散度）
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask, augmentation_mask
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model.trigger, prompt_completion_ids, attention_mask, augmentation_mask
                        )
            else:
                ref_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions = completions_text

        # 将 augmentation_mask 注入 inputs，供 reward function 使用
        for i in range(len(inputs)):
            inputs[i]["augmentation_mask"] = augmentation_mask[i]

        # Convert tensor to a list of lists of token IDs
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards: GRPO 组内归一化
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        # Log metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths
        completion_lengths = completion_mask.sum(1)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Log augmentation lengths（记录 INVOKE 决策的数量）
        augmentation_lengths = (augmentation_mask == 1).sum(dim=1)
        agg_augmentation_lengths = self.accelerator.gather(augmentation_lengths)
        self._metrics[mode]["augmentations/mean_length"].append(agg_augmentation_lengths.float().mean().item())
        self._metrics[mode]["augmentations/min_length"].append(agg_augmentation_lengths.float().min().item())
        self._metrics[mode]["augmentations/max_length"].append(agg_augmentation_lengths.float().max().item())

        # Log terminated sequences
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Log rewards
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "augmentation_mask": augmentation_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }

    def _compute_loss(self, model, inputs):
        """
        计算 Trigger 的 GRPO / BNPO / DR_GRPO loss。

        与 Weaver 的关键区别：
        - 使用 augmentation_mask 作为监督目标（而不是文本 completion token）
        - loss 在 augmentation_valid_mask（augmentation_mask != -100）位置计算
        - model 参数是 model.trigger（不是整个 MemGenModel）

        augmentation_mask 含义：
        - 1: INVOKE（调用记忆编织）
        - 0: SKIP（跳过）
        - -100: 无效位置（prompt 部分、padding 等）
        """
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        augmentation_mask = inputs["augmentation_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, augmentation_mask)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # 只在 augmentation 有效的位置计算 loss
        augmentation_valid_mask = (augmentation_mask != -100)
        if self.loss_type == "grpo":
            loss = ((per_token_loss * augmentation_valid_mask).sum(-1) / augmentation_valid_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * augmentation_valid_mask).sum() / augmentation_valid_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * augmentation_valid_mask).sum() / (augmentation_valid_mask.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * augmentation_valid_mask).sum() / augmentation_valid_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * augmentation_valid_mask).sum() / augmentation_valid_mask.sum()
        high_clip = (is_high_clipped * augmentation_valid_mask).sum() / augmentation_valid_mask.sum()
        clip_ratio = (is_region_clipped * augmentation_valid_mask).sum() / augmentation_valid_mask.sum()

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss
