import unittest
from types import SimpleNamespace

import torch
from transformers import GenerationConfig

from interactions.base_interaction import InteractionConfig, InteractionDataProto
from interactions.multiturn_interaction import MultiTurnInteractionManager
from interactions.singleturn_interaction import SingleTurnInteractionManager
from memgen.model.latent_memory_bank import LatentMemoryBank, LatentMemoryBankConfig
from memgen.model.latent_memory_bank import LatentMemorySlot
from memgen.model.modeling_memgen import MemGenModel


class FakeTokenizer:
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
    def __init__(self):
        self.steps = 0

    def preprocess_action(self, response):
        return response

    def step(self, response):
        self.steps += 1
        done = self.steps >= 2
        return f"obs-{self.steps}", 0.0, done


class RecordingMultiTurnActor:
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


def build_fake_memgen(embedding_dtype=torch.float32):
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
    def test_disabled_mode_noop_and_batch_size_limit_not_applied(self):
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
        self.assertNotEqual(actor.calls[0]["bank_id"], actor.calls[1]["bank_id"])
        self.assertEqual(first_debug["memory_write_count"], 1)
        self.assertEqual(first_debug["slot_count"], 1)

    def test_multi_turn_shares_bank_within_episode_and_resets_next_episode(self):
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
        self.assertNotEqual(first_episode_calls[0]["bank_id"], actor.calls[2]["bank_id"])
        self.assertEqual(actor.calls[2]["initial_slots"], 0)

    def test_generate_disabled_path_stays_on_original_reasoner_injection(self):
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

    def test_generate_rejects_enabled_batch_size_greater_than_one(self):
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
