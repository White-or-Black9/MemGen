import os
import random

from accelerate import Accelerator
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig, GRPOConfig
from trl.models import unwrap_model_for_generation

from data import (
    BaseBuilder,
)
from interactions.base_interaction import (
    InteractionConfig,
    InteractionManager,
    InteractionDataProto
)
from interactions.singleturn_interaction import SingleTurnInteractionManager
from interactions.multiturn_interaction import MultiTurnInteractionManager

from memgen.model.modeling_memgen import MemGenModel
from memgen.trainer.weaver_grpo_trainer import WeaverGRPOTrainer
from memgen.trainer.trigger_grpo_trainer import TriggerGRPOTrainer
from memgen.utils import (
    StaticEvalRecorder,
    DynamicEvalRecorder,
    create_tensorboard,
    log_trainable_params,
    gather_objects
)


class MemGenRunner:
    """
    MemGen 训练/评估的总调度器（Orchestrator）。

    核心职责：
    1. 接收组装好的 MemGenModel、数据集构造器、配置
    2. 根据配置（train_weaver / train_trigger）创建对应的 Trainer
    3. 协调训练流程（SFT → GRPO / Trigger GRPO）
    4. 支持分布式评估（静态单轮 / 动态多轮）

    训练流程（分阶段）：
    - Stage 1 (Weaver SFT):  固定 trigger, 用 SFT 训练 weaver
    - Stage 2 (Weaver GRPO): 固定 trigger, 用 GRPO 强化学习训练 weaver
    - Stage 3 (Trigger GRPO): 固定 weaver, 用 GRPO 训练 trigger（学习何时调用记忆）
    """

    def __init__(
        self,
        model: MemGenModel,
        data_builder: BaseBuilder,
        config: dict,
        working_dir: str,
    ):
        # parse configs
        self.config = config
        self.working_dir = working_dir

        self._parse_configs(config.get("run"))

        # parse model
        self.processing_class = model.tokenizer
        self.model = model

        # initialize envs and generation managers
        self.dataset_dict = data_builder.get_dataset_dict()
        self.env_cls = data_builder.get_env_cls()
        self.env = self.env_cls(config.get("dataset"))

        # partition datasets
        # trigger 只用 1/3 的数据训练（减少计算量，因为 trigger 学习的是是否调用记忆的决策）
        self.weaver_train_dataset, self.trigger_train_dataset = self._parse_train_dataset(self.dataset_dict["train"])
        self.weaver_valid_dataset, self.trigger_valid_dataset = self._parse_valid_dataset(self.dataset_dict["valid"])
        self.test_dataset = self.dataset_dict["test"]

        # 根据最大长度过滤过长的样本
        self.weaver_train_dataset = self._filter_dataset(self.weaver_train_dataset)
        self.trigger_train_dataset = self._filter_dataset(self.trigger_train_dataset)
        self.weaver_valid_dataset = self._filter_dataset(self.weaver_valid_dataset)
        self.trigger_valid_dataset = self._filter_dataset(self.trigger_valid_dataset)

        # 根据环境类型（STATIC/DYNAMIC）选择对应的交互管理器
        if self.env_cls.ENV_CARD == "STATIC":
            self.inter_cls = SingleTurnInteractionManager
        elif self.env_cls.ENV_CARD == "DYNAMIC":
            self.inter_cls = MultiTurnInteractionManager
        else:
            raise ValueError("Unsupported environment type.")

        # 初始化交互管理器（负责 agent 与环境的交互循环）
        self.generation_manager: InteractionManager = self.inter_cls(
            self.processing_class, self.model, self.interaction_config
        )

    def _parse_train_dataset(self, train_dataset: Dataset) -> tuple[Dataset, Dataset]:
        """划分训练集：weaver 用全部数据，trigger 只用 1/3"""
        trigger_trainset_size = min(len(train_dataset) // 3, len(train_dataset))
        rand_indices = random.sample(range(len(train_dataset)), trigger_trainset_size)
        return train_dataset, train_dataset.select(rand_indices)

    def _parse_valid_dataset(self, valid_dataset: Dataset) -> tuple[Dataset, Dataset]:
        """划分验证集：weaver 用全部数据，trigger 只用 1/3"""
        trigger_validset_size = min(len(valid_dataset) // 3, len(valid_dataset))
        rand_indices = random.sample(range(len(valid_dataset)), trigger_validset_size)
        return valid_dataset, valid_dataset.select(rand_indices)

    def _filter_dataset(self, dataset: Dataset) -> Dataset:
        """过滤掉序列长度超过最大允许长度的样本"""
        tokenizer = self.processing_class

        # 根据训练模式确定最大长度
        max_len = 1024
        if self.train_weaver and self.train_weaver_method == "sft":
            max_len = self.weaver_sft_training_args.max_length
        elif self.train_weaver and self.train_weaver_method == "grpo":
            max_len = self.weaver_grpo_training_args.max_prompt_length
        elif self.train_trigger and self.train_trigger_method == "grpo":
            max_len = self.trigger_grpo_training_args.max_prompt_length
        else:
            raise ValueError("Wrong training mode.")

        # Function to filter out samples exceeding max length
        def filter_func(sample):
            if "prompt" in sample and sample["prompt"] is not None:
                prompt = tokenizer.apply_chat_template(sample["prompt"], tokenize=True)
                return len(prompt) < max_len
            elif "messages" in sample and sample["messages"] is not None:
                conversation = tokenizer.apply_chat_template(sample["messages"][:2], tokenize=True)
                return len(conversation) < max_len
            return True

        # Apply filtering
        dataset = dataset.filter(filter_func)

        return dataset

    # ===== train weaver =====
    def _create_weaver_trainer(self):
        """
        创建 Weaver 训练器。

        支持两种训练方法：
        1. SFT: 监督微调，用于初始训练
        2. GRPO: 强化学习，基于环境奖励优化记忆生成策略
        """
        # SFT Trainer: 监督微调
        if self.train_weaver_method == "sft":
            weaver_trainer = SFTTrainer(
                model=self.model,
                args=self.weaver_sft_training_args,
                train_dataset=self.weaver_train_dataset,
                eval_dataset=self.weaver_valid_dataset,
                processing_class=self.processing_class,
            )

        # GRPO Trainer: 强化学习
        elif self.train_weaver_method == 'grpo':
            # GRPO 训练时不做 eval（节省计算资源）
            self.weaver_grpo_training_args.do_eval = False
            self.weaver_grpo_training_args.eval_strategy = 'no'
            # 配置 generation manager：weaver 使用采样，trigger 固定（不训练 trigger）
            self.generation_manager.generation_config.weaver_do_sample = True
            self.generation_manager.generation_config.trigger_do_sample = False
            self.generation_manager.generation_config.temperature = self.weaver_grpo_training_args.temperature
            self.generation_manager.generation_config.max_new_tokens = self.weaver_grpo_training_args.max_completion_length

            weaver_trainer = WeaverGRPOTrainer(
                model=self.model,
                reward_funcs=[self.env_cls.compute_reward],
                args=self.weaver_grpo_training_args,
                train_dataset=self.weaver_train_dataset,
                eval_dataset=self.weaver_valid_dataset,
                processing_class=self.processing_class,
                # --- add env into trainer ---
                env_class=self.env_cls,
                env_main_config=self.config.get("dataset"),
                generation_manager=self.generation_manager,
            )
        else:
            raise ValueError("Unsupported weaver training method.")

        return weaver_trainer

    # ===== train trigger =====
    def _create_trigger_trainer(self):
        """
        创建 Trigger 训练器。

        目前只支持 GRPO 方法——trigger 通过强化学习学习何时调用记忆编织。
        奖励由下游任务完成情况驱动（稀疏奖励）。
        """
        if self.train_trigger_method == "grpo":
            self.trigger_grpo_training_args.do_eval = False
            self.trigger_grpo_training_args.eval_strategy = 'no'

            # 配置 generation manager：trigger 使用采样，weaver 固定（不训练 weaver）
            self.generation_manager.generation_config.trigger_do_sample = True
            self.generation_manager.generation_config.weaver_do_sample = False
            self.generation_manager.generation_config.temperature = self.weaver_grpo_training_args.temperature
            self.generation_manager.generation_config.max_new_tokens = self.weaver_grpo_training_args.max_completion_length

            trigger_trainer = TriggerGRPOTrainer(
                model=self.model,
                processing_class=self.processing_class,
                train_dataset=self.trigger_train_dataset,
                eval_dataset=self.trigger_valid_dataset,
                reward_funcs=[self.env_cls.compute_reward],
                args=self.trigger_grpo_training_args
            )
        else:
            raise ValueError("Unsupported trigger training method.")

        return trigger_trainer

    # ===== train weaver/trigger =====
    def train(self):
        """
        主训练入口。

        根据配置选择训练 Weaver 或 Trigger：
        - 训练 Weaver 时，固定 Trigger（不更新其参数）
        - 训练 Trigger 时，固定 Weaver（不更新其参数）

        包含 OOM（显存不足）保护机制：在 OOM 发生时尝试保存紧急 checkpoint。
        """
        if self.train_weaver:
            trainer = self._create_weaver_trainer()
            self.model.fix_component('trigger')  # 固定 trigger 的参数

        if self.train_trigger:
            trainer = self._create_trigger_trainer()
            self.model.fix_component('weaver')  # 固定 weaver 的参数

        log_trainable_params(self.model)

        try:
            trainer.train()
            trainer.save_model()
        except RuntimeError as e:
            # 检查是否是 OOM 相关的错误
            if "OOM" in str(e) or "out of memory" in str(e).lower():
                logging.error(f"[Runner] Training stopped due to OOM: {e}")
                # 尝试最后一次保存
                try:
                    oom_dir = os.path.join(self.working_dir, "model_oom_final")
                    logging.info(f"[Runner] Attempting to save final checkpoint to {oom_dir}")
                    trainer.save_model(oom_dir)
                    logging.info(f"[Runner] Final checkpoint saved successfully")
                except Exception as save_e:
                    logging.error(f"[Runner] Failed to save final checkpoint: {save_e}")
                raise
            else:
                # 非 OOM 错误，直接抛出
                raise


    # ===== evaluate =====
    def evaluate(self):
        """
        主评估入口。

        根据环境类型（STATIC/DYNAMIC）选择对应的评估方法：
        - STATIC:  单轮生成，使用 StaticEvalRecorder 记录指标
        - DYNAMIC: 多轮交互，使用 DynamicEvalRecorder 记录完整对话历史
        """
        self.model = self.model.to(torch.bfloat16)

        evaluate_func_mapping = {
            "STATIC": self._static_evaluate,
            "DYNAMIC": self._dynamic_evaluate
        }
        evaluate_func = evaluate_func_mapping.get(self.env.ENV_CARD)
        if evaluate_func is None:
            raise ValueError("The env has unrecogonized ENV_CARD attribute")

        return evaluate_func()

    def _static_evaluate(self):
        """
        静态环境评估（单轮问答）。

        适用于如 GSM8K 这类有标准答案的数据集。
        流程：
        1. 准备测试 dataloader
        2. 对每个 batch 调用 model.generate() 生成回答
        3. 收集所有进程的生成结果
        4. 使用 StaticEvalRecorder 计算指标并记录
        """
        accelerator = Accelerator()

        if accelerator.is_main_process:
            writer = create_tensorboard(save_dir=self.working_dir)
            save_file = os.path.join(self.interaction_config.output_dir, "answer.json")
            recorder = StaticEvalRecorder(
                compute_metrics=[self.env_cls.compute_reward],
                writer=writer,
                log_file=save_file
            )
        else:
            writer = None
            recorder = None

        batch_size = self.interaction_config.batch_size

        test_dataloader = accelerator.prepare(DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch
        ))

        model_wrapped = accelerator.prepare_model(model=self.model, evaluation_mode=True)
        model_wrapped.eval()

        for test_batch in tqdm(test_dataloader, disable=not accelerator.is_main_process):
            with unwrap_model_for_generation(model_wrapped, accelerator) as unwrapped_model:
                prompts = [x["prompt"] for x in test_batch]
                prompt_inputs = self.processing_class.apply_chat_template(
                    prompts,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=True,
                    return_dict=True
                )
                prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
                gen_batch = InteractionDataProto()
                gen_batch.batch["input_ids"] = prompt_ids.to(accelerator.device)
                gen_batch.batch["attention_mask"] = prompt_mask.to(accelerator.device)
                gen_batch.no_tensor_batch["initial_prompts"] = prompts

                self.generation_manager.actor_rollout_wg = unwrapped_model
                gen_output = self.generation_manager.run_agent_loop(gen_batch)

                completion_ids = gen_output.batch["responses"]
                completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

            # only main rank can write the json
            local_completions = completions
            local_batches = test_batch

            all_completions = gather_objects(local_completions)
            all_batches = gather_objects(local_batches)

            if accelerator.is_main_process:
                if all_batches and isinstance(all_batches[0], list):
                    all_completions = [
                        completion
                        for rank_completions in all_completions
                        for completion in rank_completions
                    ]
                    all_batches = [
                        example
                        for rank_batch in all_batches
                        for example in rank_batch
                    ]
                recorder.record_batch(all_completions, all_batches)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            recorder.finalize()
            writer.close()

    def _dynamic_evaluate(self):
        """
        动态环境评估（多轮交互）。

        适用于需要多步 agent 交互的环境（如网页导航、工具使用等）。
        流程：
        1. 为每个测试样本创建独立的环境实例
        2. 在每个 step 中：agent 生成行动 → 环境返回观察结果
        3. 记录完整的对话历史和最终奖励
        """
        def _set_batch_envs(batch: list) -> tuple[list[str], list[str], list]:
            """为 batch 中的每个任务初始化环境，返回系统提示和初始用户提示"""
            system_prompts, init_user_prompts, envs = [], [], []
            for task_config in batch:
                env = self.env_cls(self.config.get("dataset"))
                system_prompt, init_user_prompt = env.set_env(task_config)

                system_prompts.append(system_prompt)
                init_user_prompts.append(init_user_prompt)
                envs.append(env)

            return system_prompts, init_user_prompts, envs

        def _build_data_proto(
            system_prompts: list[str], init_user_prompts: list[str], envs: list
        ) -> InteractionDataProto:
            """构建包含初始消息和环境的 InteractionDataProto"""
            messages = []
            for system_prmopt, init_user_prompt in zip(system_prompts, init_user_prompts):
                system_message = {"role": "system", "content": system_prmopt}
                user_message = {"role": "user", "content": init_user_prompt}
                init_messages = [system_message, user_message]
                messages.append(init_messages)

            data_proto = InteractionDataProto()
            data_proto.no_tensor_batch["init_prompts"] = messages
            data_proto.no_tensor_batch["envs"] = envs

            return data_proto

        # ===== body =====
        accelerator = Accelerator()

        if accelerator.is_main_process:
            writer = create_tensorboard(save_dir=self.working_dir)
            save_file = os.path.join(self.interaction_config.output_dir, "conversations.txt")
            recorder = DynamicEvalRecorder(writer=writer, log_file=save_file)
        else:
            writer = None
            recorder = None

        batch_size = self.interaction_config.batch_size

        # prepare dataset and dataloader
        test_dataloader = accelerator.prepare(DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch  # use the identity function
        ))

        # prepare model
        model_wrapped = accelerator.prepare_model(model=self.model, evaluation_mode=True)
        model_wrapped.eval()

        # batch generate
        for step, test_batch in tqdm(enumerate(test_dataloader), desc="Evaluation"):
            with unwrap_model_for_generation(
                model_wrapped, accelerator
            ) as unwrapped_model:
                system_prompts, init_user_prompts, envs = _set_batch_envs(test_batch)
                input_data_proto = _build_data_proto(system_prompts, init_user_prompts, envs)

                self.generation_manager.actor_rollout_wg = unwrapped_model
                outputs: InteractionDataProto = self.generation_manager.run_agent_loop(input_data_proto)

                inter_histories = outputs.no_tensor_batch["inter_histories"]
                inter_context = self.processing_class.apply_chat_template(inter_histories, tokenize=False)

            # calculate batch rewards
            rewards = []
            for env in input_data_proto.no_tensor_batch["envs"]:
                reward = env.feedback()
                rewards.append(reward)

            all_contexts = gather_objects(inter_context)
            all_rewards = gather_objects(rewards)

            if accelerator.is_main_process:
                for conts, rs in zip(all_contexts, all_rewards):
                    recorder.record_batch(conts, rs)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            recorder.finalize()
            writer.close()

    def _parse_configs(self, configs):
        """
        解析配置文件，初始化所有训练参数。

        解析内容：
        - 训练模式（train_weaver / train_trigger）
        - Weaver 训练参数（SFT + GRPO）
        - Trigger 训练参数（GRPO）
        - 交互参数（max_turns, batch_size, temperature 等）
        """
        self.train_weaver = configs.get("train_weaver", True)
        self.train_trigger = configs.get("train_trigger", False)

        # --- Parse weaver training args ---
        self.train_weaver_method = configs.get("train_weaver_method", "sft")
        if self.train_weaver_method not in ["sft", "grpo"]:
            raise ValueError("Unsupported weaver training method.")

        # parse weaver sft training args
        weaver_config = configs.get("weaver", dict())
        weaver_sft_config = weaver_config.get("sft", dict())
        self.weaver_sft_training_args = SFTConfig(**weaver_sft_config)

        # parse weaver grpo training args
        weaver_grpo_config = weaver_config.get("grpo", dict())
        self.weaver_grpo_training_args = GRPOConfig(**weaver_grpo_config)

        # --- Parse trigger training args ---
        trigger_config = configs.get("trigger", dict())
        self.train_trigger_method = configs.get("train_trigger_method", "grpo")
        if self.train_trigger_method not in ["grpo"]:
            raise ValueError("Unsupported trigger training method.")

        trigger_grpo_config = trigger_config.get("grpo", dict())
        self.trigger_grpo_training_args = GRPOConfig(**trigger_grpo_config)

        # --- update training args ---
        updated_args = {
            "output_dir": os.path.join(self.working_dir, "model"),
            "logging_dir": os.path.join(self.working_dir, "run"),
            "save_strategy": "no"
        }
        for k, v in updated_args.items():
            setattr(self.weaver_sft_training_args, k, v)
            setattr(self.weaver_grpo_training_args, k, v)
            setattr(self.trigger_grpo_training_args, k, v)

        # --- parse interaction args ---
        interaction_configs = configs.get("interaction", {})
        self.interaction_config = InteractionConfig(
            max_turns=interaction_configs.get("max_turns", 30),
            max_start_length=interaction_configs.get("max_start_length", 1024),
            max_prompt_length=interaction_configs.get("max_prompt_length", 4096),
            max_response_length=interaction_configs.get("max_response_length", 512),
            max_obs_length=interaction_configs.get("max_obs_length", 512),
            temperature=interaction_configs.get("temperature", 0.0),
            batch_size=interaction_configs.get("batch_size", 32),
            output_dir=os.path.join(self.working_dir, "evaluate"),
            weaver_do_sample=interaction_configs.get("weaver_do_sample", False),
            trigger_do_sample=interaction_configs.get("trigger_do_sample", False),
        )
