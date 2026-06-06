from abc import ABC, abstractmethod
from typing import Literal, Dict, Tuple


class BaseEnv(ABC):
    """
    环境基类。

    定义了两种环境类型：
    - STATIC: 静态环境（单轮问答，如 GSM8K）
    - DYNAMIC: 动态环境（多轮交互，如网页导航）

    ENV_CARD 用于在 Runner 中自动选择对应的 InteractionManager。
    """
    ENV_CARD: Literal["STATIC", "DYNAMIC"] = None

    def __init__(self, config):
        self.config = config

    @classmethod
    @abstractmethod
    def compute_reward(cls, **kwargs):
        """
        计算奖励的类方法。

        在 WeaverGRPOTrainer 中作为 reward_func 使用。
        接收 prompts、completions 等参数，返回每个样本的奖励值。
        """
        ...


class StaticEnv(BaseEnv):
    """
    静态环境。

    适用于有标准答案的数据集（如 GSM8K）。
    Prompt → 模型生成 → 检查答案是否正确。
    """
    ENV_CARD = "STATIC"


class DynamicEnv(BaseEnv):
    """
    动态环境。

    适用于需要多步 agent 交互的场景。
    每一步：agent 生成行动 → 环境执行 → 返回观察 → agent 继续。
    """
    ENV_CARD = "DYNAMIC"

    @abstractmethod
    def set_env(self, task_config: Dict) -> Tuple[str, str]:
        """
        根据任务配置设置环境。

        Args:
            task_config: 任务配置字典

        Returns:
            (system_prompt, init_user_prompt): 系统提示和初始用户提示
        """
        ...

    @classmethod
    @abstractmethod
    def preprocess_action(self, action: str) -> str:
        """
        对 agent 生成的动作进行预处理。
        例如：提取结构化命令、清理格式等。
        """
        ...

    @abstractmethod
    def step(self, action: str) -> Tuple[str, bool]:
        """
        执行一步环境交互。

        Args:
            action: agent 生成的动作

        Returns:
            (observation, done): 观察结果和是否完成
        """
        ...

    @abstractmethod
    def feedback(self) -> Tuple[float, bool]:
        """
        获取环境的最终反馈（在多轮交互结束后调用）。

        Returns:
            (reward, done): 奖励值和是否完成
        """
        ...
