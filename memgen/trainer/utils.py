import torch

# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    计算 tensor 的标准差，忽略 NaN 值。仅支持 1D tensor。

    Args:
        tensor (`torch.Tensor`): 输入张量，形状为 `(N,)`

    Returns:
        `torch.Tensor`: 忽略 NaN 后的标准差
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)
    count = torch.sum(~torch.isnan(tensor))
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)

def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    计算 tensor 的最大值，忽略 NaN 值。仅支持 1D tensor。

    Args:
        tensor (`torch.Tensor`): 输入张量，形状为 `(N,)`

    Returns:
        `torch.Tensor`: 忽略 NaN 后的最大值。如果所有值都是 NaN，返回 NaN。
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])

def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    计算 tensor 的最小值，忽略 NaN 值。仅支持 1D tensor。

    Args:
        tensor (`torch.Tensor`): 输入张量，形状为 `(N,)`

    Returns:
        `torch.Tensor`: 忽略 NaN 后的最小值。如果所有值都是 NaN，返回 NaN。
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])

def generate_position_ids(attention_mask):
    """
    从 attention mask 生成 position IDs。

    用于 Trigger 的前向传播，为每个 token 分配其在序列中的位置。
    """
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids
