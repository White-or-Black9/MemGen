from abc import ABC, abstractmethod
from typing import Type

from datasets import DatasetDict

from data.base_env import BaseEnv


class BaseBuilder(ABC):
    """
    数据集构造器的抽象基类。

    核心职责：
    - 根据训练模式（sft / grpo）加载对应的数据集
    - 提供环境类（env_cls）供 Runner 使用

    两种模式：
    - sft: 使用 _build_sft_datasets 构建 SFT 格式的数据集
    - grpo: 使用 _build_rl_datasets 构建 RL 格式的数据集
    """

    def __init__(self, cfg: dict = None):
        super().__init__()

        self.mode = cfg.get("mode", "sft")
        self.config = cfg.get(self.mode)

    def get_dataset_dict(self) -> DatasetDict:
        """
        根据 mode 选择对应的方法构建数据集。

        Returns:
            DatasetDict: 包含 train/valid/test 分割的数据集
        """
        method_builder_map = {
            "sft": self._build_sft_datasets,
            "grpo": self._build_rl_datasets,
        }

        if self.mode not in method_builder_map:
            raise ValueError("Unsupported datasets mode")

        return method_builder_map[self.mode]()

    @abstractmethod
    def get_env_cls(self) -> Type[BaseEnv]:
        """
        返回环境类（用于在 Runner 中初始化环境）。

        Returns:
            BaseEnv 的子类
        """
        ...

    @abstractmethod
    def _build_sft_datasets(self) -> DatasetDict:
        """构建 SFT 训练模式的数据集。"""
        ...

    @abstractmethod
    def _build_rl_datasets(self) -> DatasetDict:
        """构建 RL（GRPO）训练模式的数据集。"""
        ...
