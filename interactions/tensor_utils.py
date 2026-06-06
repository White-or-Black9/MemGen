import torch
from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class TensorConfig:
    """
    Tensor 工具配置。

    - pad_token_id: padding token ID
    - max_prompt_length: 最大 prompt 长度
    - max_obs_length: 最大观察长度（DYNAMIC 环境）
    - max_start_length: 初始 prompt 最大长度
    """
    pad_token_id: int
    max_prompt_length: int
    max_obs_length: int
    max_start_length: int


class TensorHelper:
    """
    Tensor 操作工具类。

    提供常用 tensor 操作的统一接口：
    - 裁剪到有效长度
    - padding 结构转换（left/right padding 互转）
    - attention mask / position ids 创建
    - 拼接和填充
    - EOS 后 token 清理
    """

    def __init__(self, config: TensorConfig):
        self.config = config

    def cut_to_effective_len(self, tensor_dict: Dict[str, torch.Tensor],
                            keys: List[str], cut_left: bool = True) -> Dict[str, torch.Tensor]:
        """
        根据 attention mask 裁剪 tensor 到有效长度（去除 padding）。
        """
        effective_len = tensor_dict['attention_mask'].sum(dim=1).max()
        result = tensor_dict.copy()

        for key in keys:
            if cut_left:  # 裁剪左侧（left padding）
                result[key] = tensor_dict[key][:, -effective_len:]
            else:  # 裁剪右侧
                result[key] = tensor_dict[key][:, :effective_len]
        return result

    def convert_pad_structure(self, tensor: torch.Tensor, pad_to_left: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        转换 padding 结构。

        Args:
            tensor: 输入的 tensor
            pad_to_left: True 表示转换为 left padding，False 表示转换为 right padding

        Returns:
            sorted_tensor: 排序后的 tensor
            sorted_indices: 排序索引
        """
        mask = tensor != self.config.pad_token_id if pad_to_left else tensor == self.config.pad_token_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        return tensor.gather(1, sorted_indices), sorted_indices

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """从 input_ids 创建 attention mask（1=有效 token, 0=padding token）。"""
        return torch.where(input_ids != self.config.pad_token_id, 1, 0)

    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """从 attention mask 创建 position ids。"""
        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    def concatenate_with_padding(
        self, tensors: List[torch.Tensor],
        pad_to_left: bool = True
    ) -> torch.Tensor:
        """拼接多个 tensor 并处理 padding。"""
        concatenated = torch.cat(tensors, dim=1)
        padded_tensor, _ = self.convert_pad_structure(concatenated, pad_to_left)
        return padded_tensor

    def example_level_pad(
        self, responses: torch.Tensor,
        responses_str: List[str],
        active_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        将 active 样本填充到完整 batch 维度。

        在多轮交互中，只有 active 样本参与生成，
        需要用 pad_token 填充 inactive 样本的位置。
        """
        assert active_mask.sum() == responses.shape[0]
        batch_size = active_mask.shape[0]
        seq_len = responses.shape[1]
        padded_responses = torch.full(
            (batch_size, seq_len), self.config.pad_token_id,
            dtype=responses.dtype, device=responses.device
        )
        padded_responses[active_mask] = responses

        # Create masked response strings
        padded_responses_str = [""] * batch_size

        s = 0
        for i, is_active in enumerate(active_mask):
            if is_active:
                padded_responses_str[i] = responses_str[s]
                s += 1

        return padded_responses, padded_responses_str

    def erase_after_first_eos(self, completion_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
        """
        将第一个 EOS 之后的所有 token 替换为 EOS。

        确保生成的序列在第一个 EOS 后终止。
        """
        is_eos_mask = (completion_ids == eos_token_id)
        first_eos_indices = torch.argmax(is_eos_mask.int(), dim=1)
        seq_len = completion_ids.size(1)
        col_indices = torch.arange(seq_len, device=completion_ids.device)
        mask_to_replace = (col_indices > first_eos_indices.unsqueeze(1)) & is_eos_mask.any(dim=1).unsqueeze(1)
        completion_ids[mask_to_replace] = eos_token_id
        return completion_ids
