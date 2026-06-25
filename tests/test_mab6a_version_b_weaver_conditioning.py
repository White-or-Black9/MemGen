import unittest

import torch
from transformers import GenerationConfig

from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.latent_memory_bank import (
    LatentMemoryRetrievalResult,
    LatentMemorySlot,
)
from memgen.model.modeling_memgen import MemGenModel
from scripts.eval import mab6a_version_b_weaver_conditioned_detectiveqa_n10 as harness
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


class MAB6AVersionBWeaverConditioningTest(unittest.TestCase):
    def test_memgen_config_defaults_retrieved_memory_to_weaver_false(self):
        config = MemGenConfig()

        self.assertFalse(config.retrieved_memory_to_weaver)

    def test_version_a_default_keeps_retrieved_memory_reasoner_only(self):
        model = build_fake_memgen()
        fake_bank = FakeMemoryBank(torch.tensor([[7.0, 8.0]], dtype=torch.float32))

        MemGenModel.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            generation_config=_generation_config(),
            latent_memory_bank=fake_bank,
        )

        expected_prompt_embed = torch.tensor([[[1.0, 2.0]]])
        expected_new_latent = torch.tensor([[[11.0, 11.0], [12.0, 12.0]]])
        expected_reasoner_input = torch.cat(
            [expected_prompt_embed, torch.tensor([[[7.0, 8.0]]]), expected_new_latent],
            dim=1,
        )
        debug = model._last_generation_debug

        self.assertTrue(torch.equal(model.weaver.prompt_inputs[0], expected_prompt_embed + 100))
        self.assertTrue(torch.equal(model.reasoner.recorded_inputs[0], expected_reasoner_input))
        self.assertFalse(debug["retrieved_memory_to_weaver"])
        self.assertFalse(debug["retrieved_latents_enter_weaver"])
        self.assertFalse(debug["weaver_conditioned_on_retrieved_memory"])
        self.assertEqual(debug["weaver_conditioning_token_count"], 0)
        self.assertTrue(debug["raw_retrieved_latents_enter_reasoner"])
        self.assertTrue(debug["retrieved_latents_enter_reasoner"])
        self.assertFalse(debug["fused_latent_generated"])

    def test_version_b_routes_retrieved_memory_through_weaver_only(self):
        model = build_fake_memgen()
        model.config.retrieved_memory_to_weaver = True
        fake_bank = FakeMemoryBank(torch.tensor([[7.0, 8.0]], dtype=torch.float32))

        MemGenModel.generate(
            model,
            input_ids=torch.tensor([[1]], dtype=torch.long),
            attention_mask=torch.ones((1, 1), dtype=torch.long),
            generation_config=_generation_config(),
            latent_memory_bank=fake_bank,
        )

        expected_conditioned_reasoner_input = torch.tensor([[[1.0, 2.0], [7.0, 8.0]]])
        expected_fused_latent = torch.tensor([[[11.0, 11.0], [12.0, 12.0]]])
        expected_reasoner_input = torch.cat(
            [torch.tensor([[[1.0, 2.0]]]), expected_fused_latent],
            dim=1,
        )
        debug = model._last_generation_debug

        self.assertTrue(
            torch.equal(
                model.weaver.prompt_inputs[0],
                expected_conditioned_reasoner_input + 100,
            )
        )
        self.assertTrue(torch.equal(model.reasoner.recorded_inputs[0], expected_reasoner_input))
        self.assertTrue(debug["retrieved_memory_to_weaver"])
        self.assertTrue(debug["retrieved_latents_enter_weaver"])
        self.assertTrue(debug["weaver_conditioned_on_retrieved_memory"])
        self.assertEqual(debug["weaver_conditioning_token_count"], 1)
        self.assertTrue(debug["fused_latent_generated"])
        self.assertFalse(debug["raw_retrieved_latents_enter_reasoner"])
        self.assertFalse(debug["retrieved_latents_enter_reasoner"])

    def test_version_b_thread_update_does_not_double_inject_raw_retrieval(self):
        model = build_fake_memgen()
        model.config.retrieved_memory_to_weaver = True
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
        self.assertEqual(len(bank.write_back_calls), 1)
        self.assertFalse(debug["raw_retrieved_latents_enter_reasoner"])
        self.assertFalse(debug["retrieved_latents_enter_reasoner"])
        self.assertTrue(debug["retrieved_latents_enter_weaver"])
        self.assertEqual(debug["weaver_conditioning_token_count"], 1)

    def test_disabled_bank_remains_noop_even_if_version_b_flag_is_enabled(self):
        model = build_fake_memgen()
        model.config.retrieved_memory_to_weaver = True

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
        self.assertFalse(debug["retrieved_latents_enter_weaver"])
        self.assertFalse(debug["raw_retrieved_latents_enter_reasoner"])
        self.assertEqual(debug["weaver_conditioning_token_count"], 0)

    def test_runner_contract_records_version_b_configuration_and_diagnostics(self):
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
        self.assertEqual(manifest["query_mode"], "first-query-only")
        self.assertEqual(manifest["full_history_policy"], "over_capacity_invalid")
        self.assertTrue(manifest["retrieved_memory_to_weaver"])
        self.assertEqual(config["threshold"], 0.03)
        self.assertEqual(config["retrieve_threshold"], 0.03)
        self.assertEqual(config["update_threshold"], 0.05)
        self.assertEqual(config["top_k"], 1)
        self.assertEqual(config["max_slots"], 8)
        self.assertEqual(config["retrieve_policy"], "threshold_topk")
        self.assertEqual(config["update_policy"], "thread_update")


if __name__ == "__main__":
    unittest.main()
