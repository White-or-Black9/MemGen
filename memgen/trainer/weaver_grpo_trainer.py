from contextlib import nullcontext
import logging
import os
from typing import Any, Callable, Optional, Union

import torch
from accelerate.utils import gather_object
from datasets import Dataset, IterableDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.utils import is_peft_available
from trl import GRPOTrainer, GRPOConfig
from trl.trainer.utils import selective_log_softmax
from trl.data_utils import maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
if is_peft_available():
    from peft import PeftConfig

from interactions.base_interaction import (
    InteractionManager, InteractionDataProto
)
from data.base_env import StaticEnv, DynamicEnv

from .utils import (
    nanstd, nanmax, nanmin
)
from ..model.modeling_memgen import MemGenModel

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class WeaverGRPOTrainer(GRPOTrainer):
    """
    Weaver 的 GRPO 强化学习训练器。

    核心功能：
    - 使用 GRPO (Group Relative Policy Optimization) 算法训练 Weaver
    - 通过 InteractionManager 与环境交互生成完整 rollout
    - 支持静态环境（单轮问答）和动态环境（多轮交互）
    - 自定义 loss 计算，支持 GRPO / BNPO / DR_GRPO 三种 loss 类型
    - OOM 保护：训练 step 中发生 OOM 时保存紧急 checkpoint

    与标准 GRPOTrainer 的关键区别：
    1. 使用 _get_per_token_logps 返回 supervised_mask（基于我们自己的 labels）
    2. generate 通过 generation_manager.run_agent_loop 完成（支持多轮交互）
    3. loss 计算中使用 supervised_mask 加权聚合
    """

    def __init__(
        self,
        model: MemGenModel,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        env_class=None,          # 环境类（用于创建环境实例）
        env_main_config=None,    # 环境配置
        generation_manager: InteractionManager = None  # 交互管理器
    ):
        super().__init__(
            model,
            reward_funcs,
            args,
            train_dataset,
            eval_dataset,
            processing_class,
            reward_processing_classes,
            callbacks,
            optimizers,
            peft_config
        )

        self.env_class = env_class
        self.env_main_config = env_main_config
        self.generation_manager = generation_manager

    def _build_multiturn_envs(self, inputs: list[dict[str, Union[torch.Tensor, Any]]]) -> tuple[list[list[dict]], list]:
        """
        为多轮交互构建初始消息和环境实例。

        对每个输入样本：
        1. 创建 DynamicEnv 实例
        2. 调用 env.set_env() 获取系统提示和初始用户消息
        3. 构建 [system_message, user_message] 列表
        """
        init_messages, envs = [], []

        for task_config in inputs:
            env: DynamicEnv = self.env_class(self.env_main_config)
            system_prompt, init_user_prompt = env.set_env(task_config)

            system_message = {"role": "system", "content": system_prompt}
            init_user_message = {"role": "user", "content": init_user_prompt}

            init_messages.append([system_message, init_user_message])
            envs.append(env)

        return init_messages, envs

    def _get_per_token_logps(
        self, model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        logits_to_keep: int,
        batch_size: int = None
    ) -> torch.Tensor:
        """
        计算每个 token 的对数概率，并返回监督掩码。

        与标准 GRPOTrainer 的区别：
        - 使用 outputs.supervised_labels（MemGenModel 自定义输出的标签）计算 supervised_mask
        - supervised_mask 标识哪些位置是 agent 实际生成的内容（排除 latent token 位置）
        - 通过 batch_size 参数分块计算以减少显存峰值

        Args:
            model: 模型
            input_ids: [B, L] 输入 token IDs
            attention_mask: [B, L] 注意力掩码
            labels: [B, L] 标签（-100 表示忽略）
            logits_to_keep: 需要保留的 logits 数量（即 completion 长度）
            batch_size: 分块大小，None 表示不分块

        Returns:
            logps: [B, logits_to_keep] 每个 token 的 log 概率
            supervise_masks: [B, logits_to_keep] 监督掩码（1 表示需要监督的位置）
        """
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        supervise_masks = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start: start + batch_size]
            attention_mask_batch = attention_mask[start: start + batch_size]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch, "labels": labels}
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            outputs = model(**model_inputs)
            logits = outputs.logits
            labels = outputs.supervised_labels

            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            all_logps.append(logps)

            labels = labels[:, -logits_to_keep:]
            mask = (labels != -100).long()
            supervise_masks.append(mask)

        logps = torch.cat(all_logps, dim=0)
        masks = torch.cat(supervise_masks, dim=0)
        return logps, masks

    # NOTE - currently we only deal with text input and leave multimodal as a feature work
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]  # batch_size * num_generations
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        生成完整的 rollout 并计算奖励和优势值。

        这是 GRPO 训练的核心步骤：
        1. 为每个输入构建 gen_batch（支持静态/动态环境）
        2. 通过 generation_manager.run_agent_loop() 运行完整 agent 交互
        3. 提取 prompt、completion、attention_mask、info_mask
        4. 构建 labels（只监督 agent 回复部分）
        5. 计算 old_per_token_logps（当前策略的对数概率）
        6. 计算 ref_per_token_logps（参考模型的对数概率，用于 KL 惩罚）
        7. 计算奖励和优势值
        8. 记录各种训练指标

        与标准 GRPOTrainer 的关键区别：
        - 使用 generation_manager 进行 rollout（支持多轮交互）
        - 使用 custom info_mask 来标识有效的监督位置
        - 使用 supervised_labels（来自 MemGenModel 的 forward 输出）
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # build no-tensor part
        batch_gen_keys = []
        if "prompt" in inputs[0]:  # text-based raw prompt
            batch_gen_keys.append("prompt")
        if "tools_kwargs" in inputs[0]:  # tool-integrated
            batch_gen_keys.append("tools_kwargs")
        if "interaction_kwargs" in inputs[0]:  # interaction args
            batch_gen_keys.append("interaction_kwargs")
        if "agent_name" in inputs[0]:  # agent name
            batch_gen_keys.append("agent_name")

        gen_batch = InteractionDataProto()
        for key in batch_gen_keys:
            gen_batch.no_tensor_batch[key] = [x[key] for x in inputs]

        # Single-turn env: 静态环境，直接处理 prompt
        if issubclass(self.env_class, StaticEnv):
            prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
            prompt_inputs = self.processing_class(
                text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
            )

            prompts, prompt_mask = prompt_inputs["input_ids"].to(device), prompt_inputs["attention_mask"].to(device)
            if self.max_prompt_length is not None:
                prompts = prompts[:, -self.max_prompt_length:]
                prompt_mask = prompt_mask[:, -self.max_prompt_length:]

            gen_batch.batch["input_ids"] = prompts
            gen_batch.batch["attention_mask"] = prompt_mask
        # Multi-turn env: 动态环境，需要构建初始消息和环境
        elif issubclass(self.env_class, DynamicEnv):
            init_prompts, envs = self._build_multiturn_envs(inputs)
            gen_batch.no_tensor_batch["init_prompts"] = init_prompts
            gen_batch.no_tensor_batch["envs"] = envs

            for example, env in zip(inputs, envs):
                example["envs"] = env
        else:
            raise ValueError("Unsupported environment type")

        # Regular generation path: 通过 InteractionManager 进行 rollout
        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            with (
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext()
            ):
                # Use GenerationManager to coordinate the interaction between the agent and the environment
                self.generation_manager.actor_rollout_wg = unwrapped_model
                final_gen_batch_output = self.generation_manager.run_agent_loop(gen_batch=gen_batch)

        # parse outputs: 从 rollout 结果中提取各种张量
        prompts = final_gen_batch_output.batch["prompts"].to(device)  # prompt ids
        completion_ids = final_gen_batch_output.batch["responses"].to(device)  # completion ids
        prompt_completion_ids = final_gen_batch_output.batch["input_ids"].to(device)  # prompt and completion ids
        attention_mask = final_gen_batch_output.batch["attention_mask"].to(device)  # attention_mask on prompt and response
        prompt_mask = attention_mask[:, :prompts.size(1)]
        completion_mask = final_gen_batch_output.batch["info_mask"][:, prompts.size(1):].to(device)
        is_eos = completion_ids == self.eos_token_id
        assert completion_ids.shape == completion_mask.shape

        # Construct labels: Supervise only the agent response portion.
        # 构建标签：只监督 agent 回复的有效位置，其余位置设为 -100
        prompt_labels = torch.full(prompt_mask.shape, -100, device=device)
        completion_labels = torch.where(completion_mask == 1, completion_ids, -100)
        labels = torch.cat([prompt_labels, completion_labels], dim=1)

        # Convert tensor to a list of lists of token IDs
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # completion lengths for logging
        completion_lengths = completion_mask.sum(1)
        logits_to_keep = completion_mask.size(1)

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        with torch.no_grad():
            # 当 num_iterations==1 时，old_per_token_logps 可以跳过计算（用 per_token_logps.detach() 替代）
            if self.num_iterations > 1 or self.args.steps_per_generation > self.args.gradient_accumulation_steps:
                old_per_token_logps, old_supervise_mask = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, labels, logits_to_keep
                )
            else:
                old_per_token_logps, old_supervise_mask = None, None

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, ref_supervise_mask = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask, labels, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, ref_supervise_mask = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, labels, logits_to_keep
                        )
            else:
                ref_per_token_logps, ref_supervise_mask = None, None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions = completions_text

        # compute rewards: 计算每个 reward function 的分数
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards: 组内归一化（GRPO 核心）
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

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
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompts,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "old_supervise_mask": old_supervise_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "ref_supervise_mask": ref_supervise_mask
        }

    def _compute_loss(self, model, inputs):
        """
        计算 GRPO / BNPO / DR_GRPO loss。

        GRPO 损失函数：
        L = -E[ min(π/π_old * A, clip(π/π_old, 1-ε, 1+ε) * A) ] + β * KL(π || π_ref)

        其中：
        - π/π_old: 新旧策略的概率比 (coef_1)
        - A: 优势值
        - clip: 双边裁剪，防止策略更新过大
        - KL: 与参考模型的 KL 散度惩罚

        三种 loss 类型：
        - grpo: 按序列平均后取 batch 平均
        - bnpo: 所有 token 平均
        - dr_grpo: 按 (batch_size * max_completion_length) 归一化

        关键点：
        - supervised_mask = completion_mask * supervise_mask * old_supervise_mask * ref_supervise_mask
          确保只在所有模型都一致认为"需要监督"的位置计算 loss
        """
        device = self.accelerator.device

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        old_supervise_mask, ref_supervise_mask = inputs["old_supervise_mask"], inputs["ref_supervise_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        prompt_labels = torch.full(prompt_mask.shape, -100, device=device)
        completion_labels = torch.where(completion_mask == 1, completion_ids, -100)
        labels = torch.cat([prompt_labels, completion_labels], dim=1)
        logits_to_keep = completion_labels.size(1)

        assert prompt_ids.shape == prompt_mask.shape
        assert completion_ids.shape == completion_mask.shape
        assert input_ids.shape == attention_mask.shape == labels.shape
        per_token_logps, supervise_mask = self._get_per_token_logps(model, input_ids, attention_mask, labels, logits_to_keep)

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

        # Two-sided clipping（可选的双边上限裁剪）
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if old_supervise_mask is None:
            old_supervise_mask = supervise_mask
        if ref_supervise_mask is None:
            ref_supervise_mask = supervise_mask
        # Consistency check: 所有监督掩码必须是 completion_mask 的子集
        assert (
            torch.all(supervise_mask <= completion_mask) and
            torch.all(old_supervise_mask <= completion_mask) and
            torch.all(ref_supervise_mask <= completion_mask)
        )
        # 最终的监督掩码取所有掩码的交集
        supervised_mask = completion_mask * supervise_mask * old_supervise_mask * ref_supervise_mask

        if self.loss_type == "grpo":
            loss = ((per_token_loss * supervised_mask).sum(-1) / supervised_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * supervised_mask).sum() / supervised_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * supervised_mask).sum() / (supervised_mask.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * supervised_mask).sum() / supervised_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        # Compute the clipped probability ratios（计算裁剪比例用于监控）
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * supervised_mask).sum() / supervised_mask.sum()
        high_clip = (is_high_clipped * supervised_mask).sum() / supervised_mask.sum()
        clip_ratio = (is_region_clipped * supervised_mask).sum() / supervised_mask.sum()

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        重写 training_step 以捕获 OOM 异常并保存 checkpoint。

        当 GPU 显存不足时：
        1. 记录错误日志
        2. 清理缓存
        3. 保存紧急 checkpoint
        4. 抛出 RuntimeError 停止训练
        """
        try:
            # 调用父类的 training_step
            loss = super().training_step(model, inputs, num_items_in_batch)
            return loss
        except torch.cuda.OutOfMemoryError as e:
            # OOM 发生时保存 checkpoint
            logging.error(f"[OOM] CUDA OutOfMemoryError occurred at step {self.state.global_step}")
            logging.error(f"[OOM] Error message: {str(e)}")

            # 清理缓存以释放内存
            torch.cuda.empty_cache()

            # 保存 emergency checkpoint
            oom_ckpt_dir = os.path.join(self.args.output_dir, f"oom_checkpoint_step_{self.state.global_step}")
            logging.info(f"[OOM] Saving emergency checkpoint to {oom_ckpt_dir}")

            try:
                self.save_model(oom_ckpt_dir)
                logging.info(f"[OOM] Emergency checkpoint saved successfully")
            except Exception as save_error:
                logging.error(f"[OOM] Failed to save checkpoint: {save_error}")

            # 重新抛出异常，让训练停止
            raise RuntimeError(
                f"Training stopped due to OOM at step {self.state.global_step}. "
                f"Emergency checkpoint saved to {oom_ckpt_dir}. "
                f"You can resume training from this checkpoint."
            ) from e
