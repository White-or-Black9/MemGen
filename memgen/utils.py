from dataclasses import dataclass, field
import glob
import json
import logging
import os
import shutil
from typing import Optional, Callable, Dict, List

from safetensors import safe_open
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


# ===== chat template =====

# from https://huggingface.co/HuggingFaceTB/SmolLM3-3B/blob/main/chat_template.jinja
# MemGen 使用的对话模板（ChatML 格式）
CONVERSATION_TEMPLATE = r"""
{# ───── main loop ───── #}
{%- for message in messages -%}
    {%- set content = message.content if message.content is string else "" -%}
    {%- if (message.role == "user") or (message.role == "system") -%}
        {{ "<|im_start|>" + message.role + "\n"  + content + "<|im_end|>\n" }}
    {%- elif message.role == "assistant" -%}
        {%- generation -%}
        {{ "<|im_start|>assistant\n" + content + "<|im_end|>\n" }}
        {%- endgeneration -%}
    {%- elif message.role == "tool" -%}
    {{ "<|im_start|>" + "user\n"  + content + "<|im_end|>\n" }}
    {%- endif -%}
{%- endfor -%}
{# ───── generation prompt ───── #}
{%- if add_generation_prompt -%}
    {{ "<|im_start|>assistant\n" }}
{%- endif -%}
""".strip()

# ===== torch part =====
def load_state_dict_from_safetensor(model_path) -> Dict:
    """
    从 safetensor 文件加载状态字典。

    Args:
        model_path (str): safetensor 文件路径

    Returns:
        Dict[str, torch.Tensor]: 模型参数字典
    """
    model_state_dict = {}
    with safe_open(model_path, framework="pt") as f:
        for key in f.keys():
            model_state_dict[key] = f.get_tensor(key)
    return model_state_dict

def fix_model_parameters(model: nn.Module):
    """冻结模型的所有参数（设置为不可训练）。"""
    for parameter in model.parameters():
        parameter.requires_grad = False

def open_model_parameters(model: nn.Module):
    """解冻模型的所有参数（设置为可训练）。"""
    for parameter in model.parameters():
        parameter.requires_grad = True

def log_trainable_params(model: nn.Module):
    """记录模型中所有可训练参数的信息。"""
    logging.info("Trainable parameters in the model:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info(f"  {name}: {param.numel()} params, shape={param.shape}")


# ===== Eval Part =====
@dataclass
class StaticEvalRecorder:
    """
    静态评估记录器（用于单轮问答任务）。

    功能：
    - 计算并累积多种评估指标（通过 compute_metrics）
    - 将每个样本的 prompt、标准答案、模型输出和指标写入日志文件
    - 通过 TensorBoard 记录指标变化
    """
    compute_metrics: List[Callable[[str, str, str], float]] = field(default_factory=list)
    log_file: Optional[str] = None
    writer: Optional[object] = None

    # Internal storage
    metric_sums: Dict[str, float] = field(init=False)
    metric_counts: Dict[str, int] = field(init=False)

    def __post_init__(self):
        self.metric_sums = {metric.__name__: 0.0 for metric in self.compute_metrics}
        self.metric_counts = {metric.__name__: 0 for metric in self.compute_metrics}
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, 'w') as f:
                f.write('')  # Clear file

    def record_batch(self, completions: List[str], examples: List[Dict]):
        """
        记录一个 batch 的模型输出。

        Args:
            completions (List[str]): 模型的回答
            examples (List[Dict]): 每个回答对应的问题和属性（含 "prompt" 和 "solution"）
        """
        # Extract all keys from the first example
        keys = [key for key in examples[0]]
        # Build kwargs for metrics computation (one list per field)
        reward_kwargs = {key: [example[key] for example in examples] for key in keys}
        reward_kwargs['completions'] = completions

        # Compute all metrics in batch
        batched_results = {}
        for metric in self.compute_metrics:  # iterate over each metric function
            metric_name = metric.__name__   # use function name as metric name
            batched_scores = metric(**reward_kwargs)  # compute scores for the entire batch
            batched_results[metric_name] = batched_scores

        # Record experiment results for each example
        for i, (completion, example) in enumerate(zip(completions, examples)):
            # Collect the metric results for this specific example
            metrics_result = {
                metric_name: batched_results[metric_name][i]
                for metric_name in batched_results
            }

            # Update running totals for metrics
            for metric_name, score in metrics_result.items():
                self.metric_sums[metric_name] += score
                self.metric_counts[metric_name] += 1

            # Create a log record with prompt, solution, completion, and metrics
            prompt = example.get("prompt", "")
            solution = example.get("solution", "")
            record = {
                'prompt': prompt,
                'solution': solution,
                'completion': completion,
                'metrics': metrics_result
            }

            # Write the record into a log file (if available)
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

            # Update TensorBoard metrics (if writer is available)
            if self.writer:
                mean_metrics = self.get_mean_metrics()  # get average metrics across all data so far
                for name, value in mean_metrics.items():
                    self.writer.add_scalar(name, value, global_step=self.metric_counts[name])


    def get_mean_metrics(self) -> Dict[str, float]:
        """获取所有指标的平均值。"""
        return {
            name: (self.metric_sums[name] / self.metric_counts[name]) if self.metric_counts[name] > 0 else 0.0
            for name in self.metric_sums
        }

    def finalize(self):
        """结束评估，记录最终指标平均值。"""
        mean_metrics = self.get_mean_metrics()
        final_record = {
            'summary_metrics': mean_metrics
        }

        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(final_record, ensure_ascii=False) + '\n')

        if self.writer:
            mean_metrics = self.get_mean_metrics()
            for name, value in mean_metrics.items():
                self.writer.add_scalar(name + "_final", value, global_step=self.metric_counts[name])


@dataclass
class DynamicEvalRecorder:
    """
    动态评估记录器（用于多轮交互任务）。

    功能：
    - 记录完整的对话历史
    - 计算平均奖励
    - 通过 TensorBoard 记录奖励变化
    """
    log_file: Optional[str] = None  # path to the txt log file
    writer: object = field(default=None)  # TensorBoard SummaryWriter

    def __post_init__(self):
        if self.log_file is None:
            raise ValueError("log_file path must be provided")

        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.logger = logging.getLogger("DynamicEvalRecorder")

        # Internal counters
        self._total_reward = 0.0
        self._count = 0

        # Initialize the file (clear previous content if any)
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("DynamicEvalRecorder Log\n\n")

    def record_batch(self, conversations: List[str], rewards: List[float]):
        """
        记录一个 batch 的对话和奖励。

        Args:
            conversations (List[str]): 对话文本列表
            rewards (List[float]): 对应的奖励值
        """
        if len(conversations) != len(rewards):
            raise ValueError("conversations and rewards must have the same length")

        # Append batch results to the log file
        with open(self.log_file, "a", encoding="utf-8") as f:
            for conv, rew in zip(conversations, rewards):
                f.write(f"Conversation:\n{conv}\n")
                f.write(f"Reward: {rew:.4f}\n")
                f.write("-" * 40 + "\n")

                # Update statistics
                self._total_reward += rew
                self._count += 1

        # Compute running average reward
        avg_reward = self._total_reward / self._count if self._count > 0 else 0.0

        # Write running average to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar("reward/avg", avg_reward, self._count)

        # Log summary info
        self.logger.info(f"Recorded {len(conversations)} items, avg_reward={avg_reward:.4f}")

    def finalize(self):
        """结束评估，记录最终平均奖励。"""
        avg_reward = self._total_reward / self._count if self._count > 0 else 0.0

        # Append final result to log file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\nFinal Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Average Reward: {avg_reward:.4f}\n")

        # Write final result to TensorBoard
        if self.writer:
            self.writer.add_scalar("ave_reward_final", avg_reward, global_step=self._count)


# --- helper functions ---
def create_tensorboard(save_dir: str):
    """创建 TensorBoard SummaryWriter。"""
    log_dir = os.path.join(save_dir, "runs")
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def remove_trainer_checkpoints(trainer_output_dir):
    """清理训练过程中产生的 checkpoint 目录。"""
    ckpt_paths = glob.glob(os.path.join(trainer_output_dir, "checkpoint-*"))
    for ckpt in ckpt_paths:
        shutil.rmtree(ckpt, ignore_errors=True)

import torch.distributed as dist

def gather_objects(obj):
    """
    跨进程收集对象（all_gather）。

    用于分布式评估时，将各进程的生成结果收集到主进程。
    """
    if not dist.is_initialized():
        return obj
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered
