import unittest

import torch
from transformers import GenerationConfig

from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.latent_memory_bank import (
    LatentMemoryRetrievalResult,
    LatentMemorySlot,
)
from memgen.model.modeling_memgen import MemGenModel
from scripts.eval import mab6b_weaver_space_bank_detectiveqa_n10 as harness
from tests.test_latent_memory_bank_integration import (
    FakeMemoryBank,
    FakeThreadUpdateMemoryBank,
    build_fake_memgen,
)


def _generation_config():
    generation_config = GenerationConfig(
        max_new_tokens=1,
        temperature=0.0,
        pad_token_id=0,
        eos_token_id=99,
    )
    generation_config.weaver_do_sample = False
    generation_config.trigger_do_sample = False
    return generation_config


class MAB6BWeaverSpaceBankTest(unittest.TestCase):
    def test_memgen_config_defaults_reasoner_storage(self):
        config = MemGenConfig()

        self.assertEqual(config.memory_bank_storage_space, "reasoner")

    def test_default_config_keeps_reasoner_bank_and_version_a_routing(self):
        model = build_fake_memgen()
        fake_bank = FakeMemoryBank(torch.tensor([[7.0, 8.0]], dtype=torch.float32))

        MemGenModel.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            generation_config=_generation_config(),
            latent_memory_bank=fake_bank,
        )

        debug = model._last_generation_debug

        self.assertEqual(debug["memory_bank_storage_space"], "reasoner")
        self.assertEqual(debug["stored_latent_space"], "reasoner")
        self.assertEqual(debug["retrieval_query_space"], "reasoner")
        self.assertEqual(debug["retrieved_memory_space"], "reasoner")
        self.assertFalse(debug["stored_weaver_latents_in_bank"])
        self.assertFalse(debug["retrieved_weaver_latents_from_bank"])
        self.assertFalse(debug["retrieved_memory_projected_to_weaver"])

    def test_mab6a_reasoner_bank_still_projects_retrieval_to_weaver(self):
        model = build_fake_memgen()
        model.config.retrieved_memory_to_weaver = True
        model.config.memory_bank_storage_space = "reasoner"
        fake_bank = FakeMemoryBank(torch.tensor([[7.0, 8.0]], dtype=torch.float32))

        MemGenModel.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            generation_config=_generation_config(),
            latent_memory_bank=fake_bank,
        )

        debug = model._last_generation_debug

        self.assertEqual(fake_bank.retrieve_calls[0]["query"].tolist(), [[[1.0, 2.0]]])
        self.assertEqual(debug["stored_latent_space"], "reasoner")
        self.assertEqual(debug["retrieval_query_space"], "reasoner")
        self.assertEqual(debug["retrieved_memory_space"], "reasoner")
        self.assertTrue(debug["retrieved_memory_projected_to_weaver"])
        self.assertFalse(debug["retrieved_weaver_latents_from_bank"])

    def test_mab6b_stores_weaver_space_memory(self):
        model = build_fake_memgen()
        model.config.retrieved_memory_to_weaver = True
        model.config.memory_bank_storage_space = "weaver"
        retrieved_memory = torch.tensor([[107.0, 108.0]], dtype=torch.float32)
        bank = FakeMemoryBank(retrieved_memory)

        MemGenModel.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            generation_config=_generation_config(),
            latent_memory_bank=bank,
        )

        debug = model._last_generation_debug
        expected_weaver_hidden = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])

        self.assertEqual(len(bank.write_calls), 1)
        self.assertTrue(torch.equal(bank.write_calls[0], expected_weaver_hidden))
        self.assertEqual(debug["memory_bank_storage_space"], "weaver")
        self.assertEqual(debug["stored_latent_space"], "weaver")
        self.assertTrue(debug["stored_weaver_latents_in_bank"])

    def test_mab6b_retrieves_with_weaver_space_query(self):
        model = build_fake_memgen()
        model.config.retrieved_memory_to_weaver = True
        model.config.memory_bank_storage_space = "weaver"
        retrieved_memory = torch.tensor([[107.0, 108.0]], dtype=torch.float32)
        bank = FakeMemoryBank(retrieved_memory)

        MemGenModel.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            generation_config=_generation_config(),
            latent_memory_bank=bank,
        )

        debug = model._last_generation_debug

        self.assertEqual(bank.retrieve_calls[0]["query"].tolist(), [[[101.0, 102.0]]])
        self.assertEqual(debug["retrieval_query_space"], "weaver")
        self.assertEqual(debug["retrieved_memory_space"], "weaver")
        self.assertTrue(debug["retrieved_weaver_latents_from_bank"])
        self.assertFalse(debug["retrieved_memory_projected_to_weaver"])

    def test_mab6b_concatenates_retrieved_weaver_memory_directly(self):
        model = build_fake_memgen()
        model.config.retrieved_memory_to_weaver = True
        model.config.memory_bank_storage_space = "weaver"
        retrieved_memory = torch.tensor([[107.0, 108.0]], dtype=torch.float32)
        bank = FakeMemoryBank(retrieved_memory)

        MemGenModel.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            generation_config=_generation_config(),
            latent_memory_bank=bank,
        )

        expected_weaver_input = torch.tensor([[[101.0, 102.0], [107.0, 108.0]]])
        debug = model._last_generation_debug

        self.assertTrue(torch.equal(model.weaver.prompt_inputs[0], expected_weaver_input))
        self.assertTrue(debug["retrieved_latents_enter_weaver"])
        self.assertTrue(debug["weaver_conditioned_on_retrieved_memory"])
        self.assertEqual(debug["weaver_conditioning_token_count"], 1)

    def test_mab6b_injects_only_fused_reasoner_space_latent(self):
        model = build_fake_memgen()
        model.config.retrieved_memory_to_weaver = True
        model.config.memory_bank_storage_space = "weaver"
        retrieved_memory = torch.tensor([[107.0, 108.0]], dtype=torch.float32)
        bank = FakeMemoryBank(retrieved_memory)

        MemGenModel.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            generation_config=_generation_config(),
            latent_memory_bank=bank,
        )

        expected_reasoner_input = torch.tensor(
            [[[1.0, 2.0], [11.0, 11.0], [12.0, 12.0]]]
        )
        debug = model._last_generation_debug

        self.assertTrue(torch.equal(model.reasoner.recorded_inputs[0], expected_reasoner_input))
        self.assertTrue(debug["fused_latent_generated"])
        self.assertFalse(debug["raw_retrieved_latents_enter_reasoner"])
        self.assertFalse(debug["retrieved_latents_enter_reasoner"])

    def test_mab6b_thread_update_uses_weaver_query_and_stores_weaver_hidden(self):
        model = build_fake_memgen()
        model.config.retrieved_memory_to_weaver = True
        model.config.memory_bank_storage_space = "weaver"
        retrieved_memory = torch.tensor([[107.0, 108.0]], dtype=torch.float32)
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

        MemGenModel.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            generation_config=_generation_config(),
            latent_memory_bank=bank,
        )

        write_back = bank.write_back_calls[0]
        debug = model._last_generation_debug

        self.assertEqual(bank.retrieve_calls[0]["query"].tolist(), [[[101.0, 102.0]]])
        self.assertTrue(torch.equal(write_back["memory"], torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])))
        self.assertEqual(debug["retrieval_query_space"], "weaver")
        self.assertEqual(debug["stored_latent_space"], "weaver")
        self.assertEqual(debug["retrieved_memory_space"], "weaver")
        self.assertTrue(debug["stored_weaver_latents_in_bank"])
        self.assertTrue(debug["retrieved_weaver_latents_from_bank"])
        self.assertFalse(debug["retrieved_memory_projected_to_weaver"])

    def test_disabled_bank_remains_noop_even_if_weaver_bank_flags_are_enabled(self):
        model = build_fake_memgen()
        model.config.retrieved_memory_to_weaver = True
        model.config.memory_bank_storage_space = "weaver"

        MemGenModel.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            generation_config=_generation_config(),
            latent_memory_bank=None,
        )

        debug = model._last_generation_debug

        self.assertEqual(model.weaver.prompt_inputs[0].shape[1], 1)
        self.assertEqual(model.reasoner.recorded_inputs[0].shape[1], 3)
        self.assertEqual(debug["memory_bank_storage_space"], "weaver")
        self.assertIsNone(debug["stored_latent_space"])
        self.assertIsNone(debug["retrieval_query_space"])
        self.assertIsNone(debug["retrieved_memory_space"])
        self.assertFalse(debug["stored_weaver_latents_in_bank"])
        self.assertFalse(debug["retrieved_weaver_latents_from_bank"])
        self.assertFalse(debug["retrieved_memory_projected_to_weaver"])

    def test_runner_contract_records_weaver_bank_configuration(self):
        class Args:
            dataset_root = "/data"
            mab_repo = "/repo"
            checkpoint_path = "/tmp/checkpoint"
            model_checkpoint_id = "ckpt"

        manifest = harness._build_manifest(
            "run",
            Args(),
            "now",
            git_status_before="before",
            git_status_after="after",
        )
        config = harness._bank_config()

        self.assertEqual(manifest["experiment_name"], harness.EXPERIMENT_NAME)
        self.assertTrue(manifest["retrieved_memory_to_weaver"])
        self.assertEqual(manifest["memory_bank_storage_space"], "weaver")
        self.assertEqual(config["threshold"], 0.03)
        self.assertEqual(config["retrieve_threshold"], 0.03)
        self.assertEqual(config["update_threshold"], 0.05)
        self.assertEqual(config["top_k"], 1)
        self.assertEqual(config["max_slots"], 8)
        self.assertEqual(config["retrieve_policy"], "threshold_topk")
        self.assertEqual(config["update_policy"], "thread_update")


if __name__ == "__main__":
    unittest.main()
