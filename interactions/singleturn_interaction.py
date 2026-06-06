import torch
from typing import Dict, List

from interactions.base_interaction import (
    InteractionConfig,
    InteractionManager,
    InteractionDataProto
)


class SingleTurnInteractionManager(InteractionManager):
    """
    单轮交互管理器（用于 STATIC 环境）。

    适用于如 GSM8K 等单轮问答数据集：
    1. 接收 batch 的 prompt
    2. 调用 model.generate() 生成回复
    3. 组装最终输出（prompts + responses + attention_mask + info_mask）

    info_mask 的作用：
    - 标识哪些位置是"信息性内容"（即 agent 回复中需要计算 loss 的部分）
    - 用于 GRPO 训练时区分 prompt 和 completion
    """

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: InteractionConfig,
        is_validation: bool = False,
    ):
        super().__init__(
            tokenizer, actor_rollout_wg, config, is_validation
        )

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses,
            add_special_tokens=False,
            return_tensors='pt',
            padding="longest"
        )['input_ids']

    def _info_masked_concatenate_with_padding(self,
        prompt: torch.Tensor,
        prompt_with_mask: torch.Tensor,
        response: torch.Tensor,
        info: torch.Tensor = None,
        pad_to_left: bool = True
    ) -> torch.Tensor:
        """
        拼接张量并处理 padding。同时为 info 块创建掩码（info_mask）。

        当存在 info（如工具调用后的观察结果）时，info 位置会被 mask 掉
        （在训练时不参与 loss 计算）。
        """
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            # information mask: info 位置的 mask 设为 pad_id（即不参与 loss 计算）
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)
            tensors_with_mask.append(info_mask)

        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(
        self, right_side: Dict,
        cur_responses: torch.Tensor,
        next_obs_ids: torch.Tensor = None
    ) -> Dict:
        """更新 right side 的状态（用于累积多步回复）。"""
        if next_obs_ids is not None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_info_mask'],
                cur_responses,
                next_obs_ids,
                pad_to_left=False
            )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_info_mask'],
                cur_responses,
                pad_to_left=False
            )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def run_agent_loop(self, gen_batch: InteractionDataProto) -> InteractionDataProto:
        """
        运行单轮生成。

        流程：
        1. 提取 initial_input_ids
        2. 切分 left_side（prompt）和 right_side（空的 response 占位）
        3. 裁剪到有效长度
        4. 调用 model.generate() 生成回复
        5. 处理 EOS 之后的 token
        6. 组装最终输出
        """
        initial_input_ids = gen_batch.batch["input_ids"]
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}

        # postprocess model inputs: 裁剪到有效长度
        rollings = gen_batch
        rollings.batch = self.tensor_fn.cut_to_effective_len(
            rollings.batch,
            keys=['input_ids', 'attention_mask']
        )
        rollings_active = {
            k: v for k, v in rollings.batch.items()
        }

        # model generation: 调用 MemGenModel.generate()
        gen_output = self.actor_rollout_wg.generate(
            rollings_active["input_ids"],
            rollings_active["attention_mask"],
            generation_config=self.generation_config,
        )
        responses_ids = gen_output[:, rollings_active["input_ids"].size(1):]
        responses_ids = self.tensor_fn.erase_after_first_eos(responses_ids, self.tokenizer.eos_token_id)

        # update right side
        original_right_side = self._update_right_side(original_right_side, responses_ids, next_obs_ids=None)

        # construct final output
        return self._compose_final_output(original_left_side, original_right_side)

    def _compose_final_output(
        self, left_side: Dict,
        right_side: Dict,
    ) -> InteractionDataProto:
        """
        组装最终输出。

        构建：
        - prompts: left_side 的 input_ids
        - responses: 生成的回复
        - input_ids: prompt + response 的拼接
        - attention_mask: 完整序列的 attention mask
        - info_mask: 标识哪些位置是"信息性内容"

        info_mask 用于 WeaverGRPOTrainer 中区分 prompt 部分和 completion 部分。
        """
        final_output_batch = right_side.copy()
        final_output_batch['prompts'] = left_side['input_ids']
        final_output_batch["responses"] = right_side['responses']

        # Combine input IDs: input_ids + responses
        final_output_batch['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)

        # Create attention mask
        final_output_batch['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output_batch['responses'])
        ], dim=1)

        # info_mask: 在 GRPO 训练中用于标识哪些是 completion 位置
        final_output_batch['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output_batch['responses_with_info_mask'])
        ], dim=1)

        final_output = InteractionDataProto(batch=final_output_batch)

        return final_output
