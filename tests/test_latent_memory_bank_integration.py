"""
Phase 5 LatentMemoryBank 集成测试。

测试覆盖：
- disabled 模式 no-op + batch_size 限制不生效
- enabled 模式拒绝 batch_size > 1
- single-turn 每个 session 创建独立的 bank（无跨 session 泄漏）
- multi-turn 同一 episode 内跨 turns 共享 bank
- generate() disabled path 保持原始 Reasoner-only injection 行为
- 空 bank no-op（仅注入新 latent，不检索）
- 检索到的 memory 仅注入 Reasoner，存储的是 reasoner-space latent
- generate() 检索时使用 reasoner dtype
- thread_update 策略使用 retrieve_with_context + write_back
- enabled batch_size > 1 拒绝
"""

import math
import unittest
from types import SimpleNamespace

import torch
from transformers import GenerationConfig

from interactions.base_interaction import InteractionConfig, InteractionDataProto
from interactions.multiturn_interaction import MultiTurnInteractionManager
from interactions.singleturn_interaction import SingleTurnInteractionManager
from memgen.model.latent_memory_bank import LatentMemoryBank, LatentMemoryBankConfig
from memgen.model.latent_memory_bank import (
    LatentMemoryRetrievalResult,
    LatentMemorySlot,
)
from memgen.model.modeling_memgen import MemGenModel


class FakeTokenizer:
    """假 tokenizer：用于不需要真实文本 tokenization 的集成测试。"""
    pad_token_id = 0
    eos_token_id = 99

    def __init__(self):
        self.padding_side = "left"

    def __call__(
        self,
        responses,
        add_special_tokens=False,
        return_tensors="pt",
        padding="longest",
    ):
        rows = []
        for index, _ in enumerate(responses, start=1):
            rows.append(torch.tensor([index + 10], dtype=torch.long))
        return {"input_ids": torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)}

    def batch_decode(self, responses, skip_special_tokens=True):
        return ["action"] * responses.shape[0]

    def encode(self, text, add_special_tokens=False, return_tensors="pt"):
        return torch.tensor([[1, 2, 3]], dtype=torch.long)

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        padding=True,
        return_tensors="pt",
        return_dict=True,
        return_assistant_tokens_mask=False,
        add_special_tokens=True,
    ):
        if not tokenize:
            return ["chat"] * len(messages)

        rows = []
        assistant_masks = []
        for index, chat in enumerate(messages, start=1):
            length = max(1, len(chat))
            row = torch.arange(index, index + length, dtype=torch.long)
            rows.append(row)
            if return_assistant_tokens_mask:
                assistant_masks.append(torch.ones(length, dtype=torch.long))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            rows,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = (input_ids != self.pad_token_id).long()
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if return_assistant_tokens_mask:
            output["assistant_masks"] = torch.nn.utils.rnn.pad_sequence(
                assistant_masks,
                batch_first=True,
                padding_value=0,
            )
        return output


class RecordingSingleTurnActor:
    """记录 generate() 调用的假 actor，用于 single-turn 集成测试。"""
    def __init__(self):
        self.calls = []

    def generate(self, input_ids, attention_mask, generation_config=None, latent_memory_bank=None):
        bank_id = id(latent_memory_bank) if latent_memory_bank is not None else None
        initial_slots = len(latent_memory_bank) if latent_memory_bank is not None else None
        if latent_memory_bank is not None:
            latent_memory_bank.write(torch.ones(1, 2), {"origin": "single"})
        self.calls.append({
            "bank_id": bank_id,
            "initial_slots": initial_slots,
        })
        response = torch.full(
            (input_ids.size(0), 1),
            fill_value=5,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, response], dim=1)


class FakeEnv:
    """假环境：用于 multi-turn 集成测试，2 步后 done。"""
    def __init__(self):
        self.steps = 0

    def preprocess_action(self, response):
        return response

    def step(self, response):
        self.steps += 1
        done = self.steps >= 2
        return f"obs-{self.steps}", 0.0, done


class RecordingMultiTurnActor:
    """记录 generate() 调用的假 actor，用于 multi-turn 集成测试。"""
    def __init__(self):
        self.calls = []

    def generate(self, input_ids, attention_mask, generation_config=None, latent_memory_bank=None):
        bank_id = id(latent_memory_bank) if latent_memory_bank is not None else None
        initial_slots = len(latent_memory_bank) if latent_memory_bank is not None else None
        if latent_memory_bank is not None:
            latent_memory_bank.write(torch.ones(1, 2), {"origin": "multi"})
        self.calls.append({
            "bank_id": bank_id,
            "initial_slots": initial_slots,
        })
        response = torch.full(
            (input_ids.size(0), 1),
            fill_value=6,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, response], dim=1)


class FakeReasoner:
    """假 Reasoner：记录输入 embeds，用于验证 latent memory 是否正确注入到 Reasoner。"""
    def __init__(self, embedding_dtype=torch.float32):
        self.embedding = torch.nn.Embedding(128, 2).to(dtype=embedding_dtype)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[1] = torch.tensor([1.0, 2.0], dtype=embedding_dtype)
        self.recorded_inputs = []

    def get_input_embeddings(self):
        return self.embedding

    def __call__(
        self,
        inputs_embeds,
        attention_mask,
        position_ids,
        output_hidden_states=False,
        use_cache=True,
        past_key_values=None,
    ):
        self.recorded_inputs.append(inputs_embeds.detach().clone())
        return SimpleNamespace(past_key_values=None)


class FakeWeaver:
    """假 Weaver：返回固定的 latent hidden states，用于验证 Memory → Reasoner 注入链路。"""
    prompt_latents_num = 2
    inference_latents_num = 2

    def __init__(self):
        self.prompt_inputs = []
        self.inference_inputs = []

    def augment_prompt(self, inputs_embeds, attention_mask, position_ids):
        self.prompt_inputs.append(inputs_embeds.detach().clone())
        hidden = torch.tensor(
            [[[1.0, 1.0], [2.0, 2.0]]],
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        attn_mask = torch.ones((1, 2), dtype=attention_mask.dtype, device=attention_mask.device)
        return hidden, attn_mask, None

    def augment_inference(self, inputs_embeds, attention_mask, position_ids):
        self.inference_inputs.append(inputs_embeds.detach().clone())
        hidden = torch.tensor(
            [[[3.0, 3.0], [4.0, 4.0]]],
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        attn_mask = torch.ones((1, 2), dtype=attention_mask.dtype, device=attention_mask.device)
        return hidden, attn_mask, None


class FakeMemoryBank:
    """假 memory bank（legacy 策略用）：返回预设的检索结果，记录调用。"""
    def __init__(self, retrieved_memory):
        self.retrieved_memory = retrieved_memory
        self.retrieve_calls = []
        self.write_calls = []

    def retrieve(self, query_or_hidden_states, *, device=None, dtype=None):
        self.retrieve_calls.append({
            "query": query_or_hidden_states.detach().clone(),
            "device": device,
            "dtype": dtype,
        })
        return [
            LatentMemorySlot(
                memory=self.retrieved_memory.to(device=device, dtype=dtype),
                key=self.retrieved_memory.mean(dim=0).to(device=device, dtype=dtype),
            )
        ]

    def write(self, memory, metadata=None):
        self.write_calls.append(memory.detach().clone())
        return True


class FakeThreadUpdateMemoryBank:
    """假 memory bank（thread_update 策略用）：返回预设的 retrieval_result，记录 write_back 调用。"""
    def __init__(self, retrieval_result):
        self.config = SimpleNamespace(update_policy="thread_update")
        self.retrieval_result = retrieval_result
        self.retrieve_calls = []
        self.write_back_calls = []

    def retrieve_with_context(
        self,
        query_or_hidden_states,
        *,
        device=None,
        dtype=None,
    ):
        self.retrieve_calls.append({
            "query": query_or_hidden_states.detach().clone(),
            "device": device,
            "dtype": dtype,
        })
        return self.retrieval_result

    def write_back(self, memory, retrieval_result, metadata=None):
        self.write_back_calls.append({
            "memory": memory.detach().clone(),
            "retrieval_result": retrieval_result,
        })
        return True


def build_fake_memgen(embedding_dtype=torch.float32):
    """构造假 MemGenModel，用于 generate() 的集成测试。

    关键组件：
    - reasner_to_weaver = tensor + 100（reasoner-space → weaver-space）
    - weaver_to_reasoner = tensor + 10（weaver-space → reasoner-space）
    - Weaver generated latents: prompt=[1,1],[2,2]; inference=[3,3],[4,4]
    - 因此 reasoner-space latents: prompt=[[11,11],[12,12]]; inference=[[13,13],[14,14]]
    """
    model = SimpleNamespace()
    model.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=99)
    model.reasoner = FakeReasoner(embedding_dtype=embedding_dtype)
    model.weaver = FakeWeaver()
    model.config = SimpleNamespace(max_inference_aug_num=3)
    model.device = torch.device("cpu")
    model.reasoner_to_weaver = lambda tensor: tensor + 100
    model.weaver_to_reasoner = lambda tensor: tensor + 10
    model._should_augment = lambda *args, **kwargs: torch.tensor([1], dtype=torch.long)
    model._generate_position_ids = lambda attention_mask: torch.cumsum(attention_mask, dim=1) - 1
    model._check_generate = lambda *args, **kwargs: None

    def append_one_step(
        outputs,
        current_inputs_embeds,
        current_attention_mask,
        current_position_ids,
        current_input_ids,
        do_sample=False,
        temperature=0.0,
    ):
        """假单步推理：追加一个 EOS token embedding 表示生成结束。"""
        next_embed = torch.zeros(
            (current_inputs_embeds.size(0), 1, current_inputs_embeds.size(2)),
            dtype=current_inputs_embeds.dtype,
            device=current_inputs_embeds.device,
        )
        next_mask = torch.ones(
            (current_attention_mask.size(0), 1),
            dtype=current_attention_mask.dtype,
            device=current_attention_mask.device,
        )
        next_pos = current_position_ids[:, -1:] + 1
        next_ids = torch.full(
            (current_input_ids.size(0), 1),
            fill_value=99,
            dtype=current_input_ids.dtype,
            device=current_input_ids.device,
        )
        return (
            torch.cat([current_inputs_embeds, next_embed], dim=1),
            torch.cat([current_attention_mask, next_mask], dim=1),
            torch.cat([current_position_ids, next_pos], dim=1),
            torch.cat([current_input_ids, next_ids], dim=1),
        )

    model._append_one_step = append_one_step
    return model


class LatentMemoryBankIntegrationTest(unittest.TestCase):
    """Phase 5 集成测试套件。

    用假组件的目的是：不依赖真实模型加载即可验证 interaction manager
    和 generate() 中 Phase 5 的全部行为。
    """

    def test_disabled_mode_noop_and_batch_size_limit_not_applied(self):
        """disabled 模式：bank=None，batch_size=2 不报错。"""
        tokenizer = FakeTokenizer()
        actor = RecordingSingleTurnActor()
        config = InteractionConfig(
            batch_size=2,
            max_response_length=1,
            latent_memory_bank={"enabled": False, "batch_size": 1},
        )
        manager = SingleTurnInteractionManager(tokenizer, actor, config)
        gen_batch = InteractionDataProto(batch={
            "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            "attention_mask": torch.ones((2, 2), dtype=torch.long),
        })

        manager.run_agent_loop(gen_batch)

        self.assertEqual([call["bank_id"] for call in actor.calls], [None])

    def test_enabled_mode_rejects_batch_size_greater_than_one(self):
        """enabled 模式：batch_size=2 必须报错。"""
        tokenizer = FakeTokenizer()
        actor = RecordingSingleTurnActor()
        config = InteractionConfig(
            batch_size=2,
            max_response_length=1,
            latent_memory_bank={"enabled": True, "batch_size": 1},
        )
        manager = SingleTurnInteractionManager(tokenizer, actor, config)
        gen_batch = InteractionDataProto(batch={
            "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            "attention_mask": torch.ones((2, 2), dtype=torch.long),
        })

        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            manager.run_agent_loop(gen_batch)

    def test_single_turn_bank_is_session_local(self):
        """single-turn：每次 run_agent_loop() 创建新 bank，且 initial_slots=0。
        Phase R2 变更：移除了 Python id() 断言，改用 debug-state 检查（slot_count、memory_write_count）。
        因为 id() 可能被内存分配器复用，debug-state 断言更可靠。
        """
        tokenizer = FakeTokenizer()
        actor = RecordingSingleTurnActor()
        config = InteractionConfig(
            batch_size=1,
            max_response_length=1,
            latent_memory_bank={"enabled": True, "batch_size": 1},
        )
        manager = SingleTurnInteractionManager(tokenizer, actor, config)
        gen_batch = InteractionDataProto(batch={
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        })

        manager.run_agent_loop(gen_batch)
        first_debug = manager.latest_memory_bank_debug
        manager.run_agent_loop(gen_batch)

        self.assertEqual(actor.calls[0]["initial_slots"], 0)
        self.assertEqual(actor.calls[1]["initial_slots"], 0)
        self.assertEqual(first_debug["memory_write_count"], 1)
        self.assertEqual(first_debug["slot_count"], 1)

    def test_multi_turn_shares_bank_within_episode_and_resets_next_episode(self):
        """multi-turn：同一 episode 内 bank_id 相同，下一 episode 创建新 bank。
        Phase R2 变更：移除了 Python id() 断言，改用 initial_slots 验证 session-local 语义。
        同一 episode 的第二 turn 应有 initial_slots >= 1，下一 episode 的第一 turn 为 0。
        """
        tokenizer = FakeTokenizer()
        actor = RecordingMultiTurnActor()
        config = InteractionConfig(
            batch_size=1,
            max_turns=2,
            max_response_length=1,
            latent_memory_bank={"enabled": True, "batch_size": 1},
        )
        manager = MultiTurnInteractionManager(tokenizer, actor, config)
        batch = InteractionDataProto(no_tensor_batch={
            "init_prompts": [[{"role": "user", "content": "q"}]],
            "envs": [FakeEnv()],
        })

        manager.run_agent_loop(batch)
        first_episode_calls = list(actor.calls)
        batch.no_tensor_batch["envs"] = [FakeEnv()]
        manager.run_agent_loop(batch)

        self.assertEqual(first_episode_calls[0]["bank_id"], first_episode_calls[1]["bank_id"])
        self.assertEqual(first_episode_calls[0]["initial_slots"], 0)
        self.assertGreaterEqual(first_episode_calls[1]["initial_slots"], 1)
        self.assertEqual(actor.calls[2]["initial_slots"], 0)

    def test_generate_disabled_path_stays_on_original_reasoner_injection(self):
        """disabled 路径：bank=None，行为与 Phase 3 一致 —— Weaver prompt 收到原始 embeds，
        Reasoner 收到 prompt + latents 共 3 个 token 的 embeds（1 prompt + 2 latents）。"""
        model = build_fake_memgen()
        generation_config = GenerationConfig(
            max_new_tokens=1,
            temperature=0.0,
            pad_token_id=0,
            eos_token_id=99,
        )
        generation_config.weaver_do_sample = False
        generation_config.trigger_do_sample = False

        MemGenModel.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            generation_config=generation_config,
            latent_memory_bank=None,
        )

        self.assertEqual(model.weaver.prompt_inputs[0].shape[1], 1)
        self.assertEqual(model.reasoner.recorded_inputs[0].shape[1], 3)

    def test_empty_bank_noop_injects_only_new_latents(self):
        """空 bank：write 1 次，retrieve 0 次，Reasoner 输入仅为 prompt + 新 latents。"""
        model = build_fake_memgen()
        bank = LatentMemoryBank(LatentMemoryBankConfig(enabled=True, batch_size=1))
        generation_config = GenerationConfig(
            max_new_tokens=1,
            temperature=0.0,
            pad_token_id=0,
            eos_token_id=99,
        )
        generation_config.weaver_do_sample = False
        generation_config.trigger_do_sample = False

        MemGenModel.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            generation_config=generation_config,
            latent_memory_bank=bank,
        )

        self.assertEqual(model.weaver.prompt_inputs[0].shape[1], 1)
        self.assertEqual(model.reasoner.recorded_inputs[0].shape[1], 3)
        self.assertEqual(bank.debug_summary()["memory_write_count"], 1)
        self.assertEqual(bank.debug_summary()["memory_retrieve_count"], 0)

    def test_retrieved_memory_only_injected_into_reasoner_and_stores_reasoner_space_latent(self):
        """关键验证：检索到的旧记忆仅注入 Reasoner（不入 Weaver），存储的是 reasoner-space latent。

        Weaver 收到的输入 = prompt_embed + 100 → [[1.0, 2.0]] + 100 = [[101.0, 102.0]]
        Reasoner 收到的输入 = [prompt_embed, retrieved_memory, new_latent]
            = [[1.0, 2.0]], [[7.0, 8.0]], [[11.0, 11.0], [12.0, 12.0]]
            = shape [1, 4, 2]
        写入 bank 的是 reasoner-space latent = [[11.0, 11.0], [12.0, 12.0]]
        """
        model = build_fake_memgen()
        fake_bank = FakeMemoryBank(torch.tensor([[7.0, 8.0]], dtype=torch.float32))
        generation_config = GenerationConfig(
            max_new_tokens=1,
            temperature=0.0,
            pad_token_id=0,
            eos_token_id=99,
        )
        generation_config.weaver_do_sample = False
        generation_config.trigger_do_sample = False

        MemGenModel.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            generation_config=generation_config,
            latent_memory_bank=fake_bank,
        )

        expected_prompt_embed = torch.tensor([[[1.0, 2.0]]])
        expected_new_latent = torch.tensor([[[11.0, 11.0], [12.0, 12.0]]])
        expected_reasoner_input = torch.cat(
            [expected_prompt_embed, torch.tensor([[[7.0, 8.0]]]), expected_new_latent],
            dim=1,
        )
        self.assertTrue(torch.equal(model.weaver.prompt_inputs[0], expected_prompt_embed + 100))
        self.assertTrue(torch.equal(model.reasoner.recorded_inputs[0], expected_reasoner_input))
        self.assertEqual(len(fake_bank.write_calls), 1)
        self.assertTrue(torch.equal(fake_bank.write_calls[0], expected_new_latent))

    def test_generate_retrieves_with_reasoner_dtype(self):
        """检索时使用 Reasoner 的 dtype（float64），非 bank 的存储 dtype。"""
        model = build_fake_memgen(embedding_dtype=torch.float64)
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                batch_size=1,
                retrieve_policy="topk",
                top_k=1,
                decay_alpha=0.0,
                storage_device="cpu",
            )
        )
        bank.write(torch.tensor([[7.0, 8.0]], dtype=torch.float32))
        generation_config = GenerationConfig(
            max_new_tokens=1,
            temperature=0.0,
            pad_token_id=0,
            eos_token_id=99,
        )
        generation_config.weaver_do_sample = False
        generation_config.trigger_do_sample = False

        MemGenModel.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            generation_config=generation_config,
            latent_memory_bank=bank,
        )

        self.assertEqual(model.reasoner.recorded_inputs[0].dtype, torch.float64)

    def test_generate_thread_update_uses_current_context_and_write_back(self):
        """thread_update 策略：使用 retrieve_with_context() + write_back() 而非 retrieve() + write()。"""
        model = build_fake_memgen()
        retrieved_memory = torch.tensor([[7.0, 8.0]], dtype=torch.float32)
        retrieved_slot = LatentMemorySlot(
            memory=retrieved_memory,
            key=retrieved_memory.mean(dim=0),
        )
        retrieval_result = LatentMemoryRetrievalResult(
            slots=[retrieved_slot],
            scores=(1.0,),
            max_score=1.0,
            argmax_index=0,
            threshold_passed=True,
            retrieved_indices=(0,),
            retrieved_scores=(1.0,),
            bank_step=1,
        )
        bank = FakeThreadUpdateMemoryBank(retrieval_result)
        generation_config = GenerationConfig(
            max_new_tokens=1,
            temperature=0.0,
            pad_token_id=0,
            eos_token_id=99,
        )
        generation_config.weaver_do_sample = False
        generation_config.trigger_do_sample = False

        MemGenModel.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            generation_config=generation_config,
            latent_memory_bank=bank,
        )

        expected_prompt_embed = torch.tensor([[[1.0, 2.0]]])
        expected_new_latent = torch.tensor(
            [[[11.0, 11.0], [12.0, 12.0]]]
        )
        expected_reasoner_input = torch.cat(
            [
                expected_prompt_embed,
                retrieved_memory.unsqueeze(0),
                expected_new_latent,
            ],
            dim=1,
        )
        self.assertEqual(len(bank.retrieve_calls), 1)
        self.assertEqual(len(bank.write_back_calls), 1)
        self.assertIs(
            bank.write_back_calls[0]["retrieval_result"],
            retrieval_result,
        )
        self.assertTrue(
            torch.equal(
                bank.write_back_calls[0]["memory"],
                expected_new_latent,
            )
        )
        self.assertTrue(
            torch.equal(
                model.weaver.prompt_inputs[0],
                expected_prompt_embed + 100,
            )
        )
        self.assertTrue(
            torch.equal(
                model.reasoner.recorded_inputs[0],
                expected_reasoner_input,
            )
        )

    def test_generate_thread_update_respects_split_thresholds(self):
        """split thresholds：retrieve_threshold 可见、update_threshold 控制写回替换。"""
        model = build_fake_memgen()
        with torch.no_grad():
            model.reasoner.embedding.weight[1] = torch.tensor([1.0, 0.0])

        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                batch_size=1,
                max_slots=2,
                threshold=0.03,
                retrieve_threshold=0.03,
                update_threshold=0.05,
                top_k=1,
                retrieve_policy="threshold_topk",
                update_policy="thread_update",
                decay_alpha=0.0,
                storage_device="cpu",
            )
        )
        slot_memory = torch.tensor([[0.04, math.sqrt(1.0 - 0.04**2)]], dtype=torch.float32)
        empty_result = bank.retrieve_with_context(torch.tensor([1.0, 0.0]))
        bank.write_back(slot_memory, empty_result, {"id": "seed"})

        generation_config = GenerationConfig(
            max_new_tokens=1,
            temperature=0.0,
            pad_token_id=0,
            eos_token_id=99,
        )
        generation_config.weaver_do_sample = False
        generation_config.trigger_do_sample = False

        MemGenModel.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            generation_config=generation_config,
            latent_memory_bank=bank,
        )

        final_debug = bank.debug_summary()
        last_write_back = final_debug["last_write_back"]
        self.assertEqual(final_debug["effective_retrieve_threshold"], 0.03)
        self.assertEqual(final_debug["effective_update_threshold"], 0.05)
        self.assertTrue(last_write_back["retrieve_threshold_passed"])
        self.assertFalse(last_write_back["update_threshold_passed"])
        self.assertEqual(last_write_back["write_action"], "insert")
        self.assertEqual(last_write_back["update_reason"], "new_thread")
        self.assertEqual(final_debug["write_action_counts"], {"insert": 2})
        self.assertEqual(final_debug["update_reason_counts"], {"empty_bank": 1, "new_thread": 1})
        self.assertEqual(final_debug["slot_count"], 2)
        self.assertEqual(model.weaver.prompt_inputs[0].shape[1], 1)
        self.assertEqual(model.reasoner.recorded_inputs[0].shape[1], 4)

    def test_generate_rejects_enabled_batch_size_greater_than_one(self):
        """enabled bank + batch_size=2：generate() 拒绝。disabled + batch_size=2：允许。"""
        model = build_fake_memgen()
        bank = LatentMemoryBank(LatentMemoryBankConfig(enabled=True, batch_size=1))
        generation_config = GenerationConfig(
            max_new_tokens=1,
            temperature=0.0,
            pad_token_id=0,
            eos_token_id=99,
        )
        generation_config.weaver_do_sample = False
        generation_config.trigger_do_sample = False

        with self.assertRaisesRegex(ValueError, "batch_size=1 only"):
            MemGenModel.generate(
                model,
                input_ids=torch.tensor([[1], [1]], dtype=torch.long),
                attention_mask=torch.ones((2, 1), dtype=torch.long),
                generation_config=generation_config,
                latent_memory_bank=bank,
            )

        MemGenModel.generate(
            model,
            input_ids=torch.tensor([[1], [1]], dtype=torch.long),
            attention_mask=torch.ones((2, 1), dtype=torch.long),
            generation_config=generation_config,
            latent_memory_bank=None,
        )


if __name__ == "__main__":
    unittest.main()
