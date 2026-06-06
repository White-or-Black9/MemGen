from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Optional

from transformers import GenerationConfig

from interactions.tensor_utils import TensorHelper, TensorConfig


@dataclass
class InteractionConfig:
    """
    Agent 与环境交互的配置。

    控制 agent 生成和多轮交互的各项参数：
    - max_turns: 最大交互轮数（仅 DYNAMIC 环境使用）
    - max_start_length: 初始 prompt 的最大长度
    - max_prompt_length: 完整 prompt（含历史）的最大长度
    - max_response_length: 每轮最大回复长度
    - max_obs_length: 环境观察的最大长度（仅 DYNAMIC 环境使用）
    - temperature: 生成温度
    - batch_size: 批大小
    - weaver_do_sample: Weaver 生成时是否使用采样（True=采样, False=greedy）
    - trigger_do_sample: Trigger 决策时是否使用采样
    """
    max_turns: int = 1
    max_start_length: int = 1024
    max_prompt_length: int = 4096
    max_response_length: int = 512
    max_obs_length: int = 512
    temperature: float = 1.0
    batch_size: int = 8
    output_dir: Optional[str] = None
    weaver_do_sample: bool = False
    trigger_do_sample: bool = False


@dataclass
class InteractionDataProto:
    """
    Interaction 数据协议。

    包含两部分：
    - batch: 张量数据（input_ids, attention_mask, responses 等）
    - no_tensor_batch: 非张量数据（prompts, envs, 对话历史等）
    """
    batch: dict = field(default_factory=dict)
    no_tensor_batch: dict = field(default_factory=dict)


class InteractionManager(ABC):
    """
    Agent 与环境交互管理器的抽象基类。

    核心职责：
    - 管理与环境的交互生命周期
    - 维护 tokenizer、generation config、tensor helper
    - 提供 run_agent_loop 抽象方法供子类实现

    两种子类实现：
    - SingleTurnInteractionManager: 单轮交互（静态环境）
    - MultiTurnInteractionManager: 多轮交互（动态环境）
    """

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: InteractionConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"  # left padding for batch generation
        self.actor_rollout_wg = actor_rollout_wg  # 模型（unwrapped）
        self.config = config
        self.is_validation = is_validation

        assert tokenizer.pad_token_id is not None
        # 初始化 tensor 工具类（用于 padding、attention mask 等操作）
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

        # generation configs for agent
        # 默认生成配置
        self.generation_config = GenerationConfig(
            max_new_tokens=self.config.max_response_length,
            temperature=self.config.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        # 扩展 GenerationConfig 以支持 MemGen 特有的配置
        self.generation_config.weaver_do_sample = self.config.weaver_do_sample
        self.generation_config.trigger_do_sample = self.config.trigger_do_sample

        logging.info(f"Weaver do sample: {self.generation_config.weaver_do_sample}, Trigger do sample: {self.generation_config.trigger_do_sample}")

    @abstractmethod
    def run_agent_loop(self, gen_batch: InteractionDataProto) -> InteractionDataProto:
        """
        运行 agent 与环境的完整交互循环。

        Args:
            gen_batch: 包含初始输入（prompts、input_ids 等）

        Returns:
            包含生成结果（completions、对话历史等）的 InteractionDataProto
        """
        ...
