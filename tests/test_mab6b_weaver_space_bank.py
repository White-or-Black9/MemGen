import unittest
import tempfile
from pathlib import Path
import json
import importlib
import inspect
from types import SimpleNamespace

import torch
from transformers import GenerationConfig

from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.latent_memory_bank import (
    LatentMemoryBank,
    LatentMemoryBankConfig,
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
    def test_eventqa_runner_uses_isolated_note_and_metric(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )

        self.assertEqual(eventqa.SUB_DATASET, "eventqa_65536")
        self.assertEqual(eventqa.METRIC_KEY, "substring_exact_match")
        self.assertEqual(
            eventqa.RESEARCH_NOTE_PATH,
            Path(
                "research_notes/benchmarks/"
                "memoryagentbench_mab6b_fr_eventqa_65536_n5.md"
            ),
        )
        self.assertNotEqual(eventqa.RESEARCH_NOTE_PATH, harness.RESEARCH_NOTE_PATH)

    def test_eventqa_question_payload_tracks_question_and_gold_answers(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )

        context_payload = {
            "context_id": "eventqa-ctx-0",
            "context_index": 0,
            "chunks": ["chunk-1", "chunk-2"],
            "chunk_token_lengths": [4, 5],
            "memorization_prompts": ["m1", "m2"],
            "questions": ["q0", "q1"],
            "answers": [["a0"], ["a1", "a1-alt"]],
            "question_ids": ["qid0", "qid1"],
            "question_types": ["type0", "type1"],
            "qa_pair_ids": ["pair0", "pair1"],
            "previous_events": [["p0"], ["p1"]],
            "dataset_config": {"sub_dataset": "eventqa_65536"},
        }

        payload = eventqa.build_question_payload(context_payload, 1)

        self.assertEqual(payload["context_id"], "eventqa-ctx-0")
        self.assertEqual(payload["query_id"], 1)
        self.assertEqual(payload["question_id"], "qid1")
        self.assertEqual(payload["qa_pair_id"], "pair1")
        self.assertEqual(payload["gold_answers"], ["a1", "a1-alt"])
        self.assertIn("The event that happens next is:", payload["query_prompt"])
        self.assertNotIn(
            "Output exactly one event from the candidate list.",
            payload["query_prompt"],
        )

    def test_eventqa_strict_prompt_is_opt_in_and_preserves_official_tail(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )

        context_payload = {
            "context_id": "eventqa-ctx-0",
            "context_index": 0,
            "chunks": ["chunk-1"],
            "chunk_token_lengths": [4],
            "memorization_prompts": ["m1"],
            "questions": ["q0"],
            "answers": [["a0"]],
            "question_ids": ["qid0"],
            "question_types": ["type0"],
            "qa_pair_ids": ["pair0"],
            "previous_events": [["p0"]],
            "dataset_config": {"sub_dataset": "eventqa_65536"},
            "strict_official_eventqa_prompt": True,
        }

        payload = eventqa.build_question_payload(context_payload, 0)

        self.assertIn("The event that happens next is:", payload["query_prompt"])
        self.assertIn(
            "Output exactly one event from the candidate list.",
            payload["query_prompt"],
        )
        self.assertIn("Do not output Chinese", payload["query_prompt"])
        self.assertNotIn("Answer:", payload["query_prompt"])
        self.assertTrue(
            payload["query_prompt"].rstrip().endswith(
                "The event that happens next is:"
            )
        )

    def test_eventqa_first_line_prompt_is_opt_in_and_lightweight(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )

        context_payload = {
            "context_id": "eventqa-ctx-0",
            "context_index": 0,
            "chunks": ["chunk-1"],
            "chunk_token_lengths": [4],
            "memorization_prompts": ["m1"],
            "questions": ["q0"],
            "answers": [["a0"]],
            "question_ids": ["qid0"],
            "question_types": ["type0"],
            "qa_pair_ids": ["pair0"],
            "previous_events": [["p0"]],
            "dataset_config": {"sub_dataset": "eventqa_65536"},
            "first_line_official_eventqa_prompt": True,
        }

        payload = eventqa.build_question_payload(context_payload, 0)

        self.assertIn(
            "In your response, only include the event answer on the first line.",
            payload["query_prompt"],
        )
        self.assertIn("The event that happens next is:", payload["query_prompt"])
        self.assertNotIn("Answer:", payload["query_prompt"])
        self.assertNotIn(
            "Output exactly one event from the candidate list.",
            payload["query_prompt"],
        )
        self.assertNotIn("Do not output Chinese", payload["query_prompt"])

    def test_eventqa_first_line_prompt_and_bank_modes_share_same_visible_query(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )

        context_payload = {
            "context_id": "eventqa-ctx-0",
            "context_index": 0,
            "chunks": ["chunk-1"],
            "chunk_token_lengths": [4],
            "memorization_prompts": ["m1"],
            "questions": ["q0"],
            "answers": [["a0"]],
            "question_ids": ["qid0"],
            "question_types": ["type0"],
            "qa_pair_ids": ["pair0"],
            "previous_events": [["p0"]],
            "dataset_config": {"sub_dataset": "eventqa_65536"},
            "first_line_official_eventqa_prompt": True,
        }

        bank_off_payload = eventqa.build_question_payload(context_payload, 0)
        bank_on_payload = eventqa.build_question_payload(context_payload, 0)

        self.assertEqual(
            bank_off_payload["query_prompt"],
            bank_on_payload["query_prompt"],
        )

    def test_eventqa_manifest_records_generation_length_and_episode_protocol(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        args = SimpleNamespace(
            dataset_root="/data",
            mab_repo="/repo",
            checkpoint_path="/tmp/checkpoint",
            model_checkpoint_id="checkpoint",
            requested_contexts=1,
            context_index=None,
            question_limit=1,
            eventqa_protocol="frozen_context_bank",
        )

        manifest = eventqa._build_manifest(
            "run", args, "now", git_status_before="clean"
        )

        self.assertEqual(manifest["generation_max_length"], 40)
        self.assertEqual(manifest["effective_generation_max_length"], 40)
        self.assertEqual(manifest["eventqa_protocol"], "frozen_context_bank")
        self.assertFalse(manifest["context_bank_rebuilt_per_question"])
        self.assertTrue(manifest["context_bank_reused_across_questions"])
        self.assertEqual(
            manifest["bank_off_mode"], "compressed_bridge_no_persistent_bank"
        )
        self.assertFalse(manifest["bank_off_is_official_long_context_baseline"])
        self.assertIsNone(manifest["context_index"])
        self.assertEqual(manifest["selected_context_indices"], [])

    def test_eventqa_protocol_cli_defaults_to_frozen_context_bank(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )

        args = eventqa.build_parser().parse_args([])

        self.assertEqual(args.eventqa_protocol, "frozen_context_bank")
        self.assertIsNone(args.context_index)
        self.assertFalse(args.construction_only)
        self.assertFalse(args.reseed_per_context)
        self.assertFalse(args.trace_score_decomposition)
        self.assertFalse(args.save_frozen_bank)
        self.assertFalse(args.bank_transition_diagnostics)
        self.assertFalse(args.strict_official_eventqa_prompt)
        self.assertFalse(args.first_line_official_eventqa_prompt)

    def test_eventqa_diagnostic_cli_flags_are_opt_in(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )

        args = eventqa.build_parser().parse_args(
            [
                "--reseed-per-context",
                "--trace-score-decomposition",
                "--save-frozen-bank",
                "--bank-transition-diagnostics",
            ]
        )

        self.assertTrue(args.reseed_per_context)
        self.assertTrue(args.trace_score_decomposition)
        self.assertTrue(args.save_frozen_bank)
        self.assertTrue(args.bank_transition_diagnostics)

    def test_eventqa_tensor_hash_is_stable_and_includes_shape_and_dtype(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        first = eventqa._tensor_hash(tensor)
        second = eventqa._tensor_hash(tensor.clone())
        changed = eventqa._tensor_hash(torch.tensor([[1.0, 3.0]]))
        reshaped = eventqa._tensor_hash(tensor.reshape(2, 1))
        changed_dtype = eventqa._tensor_hash(tensor.to(torch.float64))

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertNotEqual(first, reshaped)
        self.assertNotEqual(first, changed_dtype)

    def test_eventqa_runtime_metadata_has_reproducibility_fields(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        args = eventqa.build_parser().parse_args(
            ["--requested-contexts", "1", "--context-index", "3"]
        )

        metadata = eventqa._runtime_reproducibility_metadata(
            args,
            selected_context_indices=[3],
            argv=["runner.py", "--context-index", "3"],
        )

        for key in (
            "argv",
            "git_commit",
            "git_dirty",
            "python_version",
            "torch_version",
            "transformers_version",
            "cuda_version",
            "gpu",
            "dtype",
            "checkpoint_path",
            "dataset_path",
            "seed",
            "deterministic_state",
            "process_layout",
        ):
            self.assertIn(key, metadata)
        self.assertEqual(metadata["process_layout"]["mode"], "single_context")
        self.assertEqual(metadata["process_layout"]["selected_context_indices"], [3])

    def test_eventqa_manifest_only_enables_diagnostics_when_requested(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        normal_args = eventqa.build_parser().parse_args([])
        diagnostic_args = eventqa.build_parser().parse_args(
            ["--reseed-per-context", "--trace-score-decomposition"]
        )

        normal = eventqa._build_manifest(
            "normal",
            normal_args,
            "now",
            git_status_before="clean",
            selected_context_indices=[0, 1, 2, 3, 4],
        )
        diagnostic = eventqa._build_manifest(
            "diagnostic",
            diagnostic_args,
            "now",
            git_status_before="clean",
            selected_context_indices=[0, 1, 2, 3, 4],
        )

        self.assertFalse(normal["diagnostics_enabled"])
        self.assertNotIn("reproducibility_diagnostics", normal)
        self.assertTrue(diagnostic["diagnostics_enabled"])
        self.assertEqual(
            diagnostic["diagnostic_options"],
            {
                "reseed_per_context": True,
                "trace_score_decomposition": True,
                "save_frozen_bank": False,
                "bank_transition_diagnostics": False,
            },
        )

    def test_eventqa_transition_artifacts_are_strictly_opt_in(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        row = {
            "context_index": 0,
            "context_id": "ctx-0",
            "query_id": 0,
            "question_id": None,
            "qa_pair_id": "pair-0",
            "gold_answers": ["gold"],
            "bank_off_prediction": "wrong",
            "bank_on_prediction": "gold",
            "bank_off_parsed_prediction": "wrong",
            "bank_on_parsed_prediction": "gold",
            "bank_off_substring_exact_match": 0,
            "bank_on_substring_exact_match": 1,
            "bank_off_eventqa_recall": 0.0,
            "bank_on_eventqa_recall": 1.0,
            "bank_off_format_flags": {},
            "bank_on_format_flags": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            eventqa._write_bank_transition_artifacts(root, [row], enabled=False)
            self.assertFalse((root / "bank_transition_diagnostics.jsonl").exists())
            self.assertFalse((root / "bank_transition_aggregate.json").exists())

            eventqa._write_bank_transition_artifacts(root, [row], enabled=True)
            diagnostics = root / "bank_transition_diagnostics.jsonl"
            aggregate = root / "bank_transition_aggregate.json"
            self.assertTrue(diagnostics.is_file())
            self.assertTrue(aggregate.is_file())
            self.assertTrue(json.loads(diagnostics.read_text())["helpful_memory"])

    def test_eventqa_reseed_per_context_records_effective_seed(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )

        metadata = eventqa._prepare_context_rng(
            base_seed=42,
            context_index=3,
            reseed_per_context=True,
        )

        self.assertTrue(metadata["reseed_applied"])
        self.assertEqual(metadata["effective_seed"], 45)
        for key in (
            "python_random_state_hash",
            "numpy_random_state_hash",
            "torch_cpu_rng_state_hash",
            "torch_cuda_rng_state_hash",
            "deterministic_state",
        ):
            self.assertIn(key, metadata)

    def test_eventqa_bank_tensor_snapshot_records_slot_hashes_and_metadata(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(enabled=True, max_slots=2, top_k=1)
        )
        bank.write(torch.tensor([[3.0, 4.0]], dtype=torch.float32))
        slot = bank._slots[0]
        slot.created_step = 7
        slot.last_retrieved_step = 5
        slot.access_count = 2

        snapshot = eventqa._bank_tensor_snapshot(
            bank, context_index=1, context_id="ctx-1"
        )

        self.assertEqual(snapshot["slot_count"], 1)
        self.assertEqual(snapshot["context_index"], 1)
        self.assertEqual(len(snapshot["combined_frozen_bank_hash"]), 64)
        recorded = snapshot["slots"][0]
        self.assertEqual(recorded["slot_index"], 0)
        self.assertEqual(recorded["created_step"], 7)
        self.assertEqual(recorded["last_retrieved_step"], 5)
        self.assertEqual(recorded["access_count"], 2)
        self.assertEqual(recorded["memory_token_count"], 1)
        self.assertEqual(len(recorded["memory_tensor_hash"]), 64)
        self.assertEqual(len(recorded["key_tensor_hash"]), 64)
        self.assertAlmostEqual(recorded["memory_norm"], 5.0)

    def test_eventqa_score_decomposition_matches_bank_formula(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                max_slots=2,
                top_k=1,
                retrieve_threshold=0.1,
                decay_alpha=0.5,
            )
        )
        bank.write(torch.tensor([[1.0, 0.0]], dtype=torch.float32))
        bank.write(torch.tensor([[0.0, 1.0]], dtype=torch.float32))
        bank._slots[0].last_retrieved_step = 1
        bank._slots[1].last_retrieved_step = 2
        query = torch.tensor([1.0, 0.0], dtype=torch.float32)

        decomposition = eventqa._compute_score_decomposition(
            bank,
            query,
            retrieval_step=3,
            selected_indices=[0],
            query_id="q0",
        )

        slot0 = decomposition["slots"][0]
        self.assertAlmostEqual(slot0["raw_cosine"], 1.0)
        self.assertEqual(slot0["last_retrieved_age"], 2)
        self.assertAlmostEqual(slot0["decay_factor"], 0.36787944117)
        self.assertAlmostEqual(slot0["final_score"], 0.36787944117)
        self.assertTrue(slot0["threshold_passed"])
        self.assertTrue(slot0["selected_by_topk"])
        self.assertEqual(slot0["raw_cosine_rank"], 1)
        self.assertEqual(slot0["final_score_rank"], 1)

    def test_eventqa_construction_provenance_includes_slot_targets(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        turns = [
            {
                "construction_turn_index": 0,
                "write_action": "insert",
                "target_slot_index": 0,
                "replaced_slot_index": None,
            },
            {
                "construction_turn_index": 1,
                "write_action": "replace_matched",
                "target_slot_index": 0,
                "replaced_slot_index": 0,
            },
        ]
        context_payload = {
            "context_index": 2,
            "chunks": ["chunk zero", "chunk one"],
            "memorization_prompts": ["prompt zero", "prompt one"],
        }

        enriched = eventqa._enrich_construction_provenance(
            turns, context_payload
        )

        self.assertEqual(enriched[0]["context_index"], 2)
        self.assertEqual(enriched[0]["chunk_index"], 0)
        self.assertEqual(enriched[0]["target_slot_index"], 0)
        self.assertIsNone(enriched[0]["replaced_slot_index"])
        self.assertEqual(enriched[1]["replaced_slot_index"], 0)
        self.assertEqual(len(enriched[0]["chunk_text_hash"]), 64)
        self.assertEqual(len(enriched[0]["memorization_prompt_hash"]), 64)

    def test_eventqa_question_identity_falls_back_to_index_and_text_hash(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        context_payload = {
            "question_ids": [None],
            "qa_pair_ids": [None],
            "questions": ["What happened next?"],
        }

        identities = eventqa._question_identity_records(context_payload)

        self.assertEqual(identities[0]["question_index"], 0)
        self.assertIsNone(identities[0]["question_id"])
        self.assertIsNone(identities[0]["qa_pair_id"])
        self.assertEqual(len(identities[0]["question_text_hash"]), 64)
        self.assertEqual(identities[0]["stable_fallback_id"], "question_index:0")

    def test_eventqa_save_frozen_bank_serializes_scoring_state(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(enabled=True, max_slots=2, top_k=1)
        )
        bank.write(torch.tensor([[1.0, 2.0]], dtype=torch.float32))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bank.pt"
            eventqa._save_frozen_bank(
                path,
                bank,
                context_index=0,
                context_id="ctx-0",
                runtime_metadata={"run_id": "run"},
            )
            payload = torch.load(path, weights_only=False)

        self.assertEqual(payload["context_id"], "ctx-0")
        self.assertEqual(payload["context_index"], 0)
        self.assertEqual(payload["retrieval_step"], bank._retrieval_step)
        self.assertEqual(len(payload["slots"]), 1)
        self.assertIn("memory", payload["slots"][0])
        self.assertIn("key", payload["slots"][0])
        self.assertIn("bank_config", payload)

    def test_eventqa_parser_accepts_construction_only(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )

        args = eventqa.build_parser().parse_args(["--construction-only"])

        self.assertTrue(args.construction_only)

    def test_eventqa_default_cli_values_are_the_runtime_bank_config(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        args = eventqa.build_parser().parse_args([])

        config = eventqa._eventqa_bank_config(args)

        self.assertEqual(config["retrieve_threshold"], 0.005)
        self.assertEqual(config["update_threshold"], 0.08)
        self.assertEqual(config["max_slots"], 16)
        self.assertEqual(config["top_k"], 1)
        self.assertEqual(config["decay_alpha"], 0.05)
        self.assertEqual(args.generation_max_length, 40)
        self.assertNotEqual(config, harness._bank_config())

    def test_eventqa_cli_overrides_match_runtime_and_manifest(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        args = eventqa.build_parser().parse_args(
            [
                "--retrieve-threshold", "0.02",
                "--update-threshold", "0.09",
                "--max-slots", "12",
                "--top-k", "3",
                "--decay-alpha", "0.01",
                "--generation-max-length", "55",
                "--eventqa-protocol", "independent_episode",
                "--context-index", "3",
                "--requested-contexts", "4",
            ]
        )
        runtime_config = eventqa._eventqa_bank_config(args)
        manifest = eventqa._build_manifest(
            "run",
            args,
            "now",
            git_status_before="clean",
            selected_context_indices=[3],
        )
        bank = LatentMemoryBank(LatentMemoryBankConfig(**runtime_config))

        eventqa._assert_runtime_bank_config_matches(bank.config, manifest)
        self.assertEqual(manifest["retrieve_threshold"], 0.02)
        self.assertEqual(manifest["update_threshold"], 0.09)
        self.assertEqual(manifest["max_slots"], 12)
        self.assertEqual(manifest["top_k"], 3)
        self.assertEqual(manifest["decay_alpha"], 0.01)
        self.assertEqual(manifest["generation_max_length"], 55)
        self.assertEqual(manifest["eventqa_protocol"], "independent_episode")
        self.assertEqual(manifest["context_index"], 3)
        self.assertEqual(manifest["requested_contexts"], 4)

    def test_eventqa_runtime_config_mismatch_fails_with_values(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        args = eventqa.build_parser().parse_args([])
        runtime_config = eventqa._eventqa_bank_config(args)
        manifest = eventqa._build_manifest(
            "run", args, "now", git_status_before="clean"
        )
        manifest["update_threshold"] = 0.05
        bank = LatentMemoryBank(LatentMemoryBankConfig(**runtime_config))

        with self.assertRaisesRegex(
            RuntimeError,
            r"update_threshold.*actual.*0\.08.*recorded.*0\.05",
        ):
            eventqa._assert_runtime_bank_config_matches(bank.config, manifest)

    def test_eventqa_main_does_not_use_detectiveqa_bank_config(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )

        self.assertNotIn("weaver_bank._bank_config", inspect.getsource(eventqa.main))

    def test_eventqa_parser_accepts_context_index(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )

        args = eventqa.build_parser().parse_args(["--context-index", "3"])

        self.assertEqual(args.context_index, 3)

    def test_eventqa_select_context_indices_defaults_to_requested_prefix(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )

        selected = eventqa.select_context_indices(5, 2)

        self.assertEqual(selected, [0, 1])

    def test_eventqa_select_context_indices_honors_explicit_context_index(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )

        selected = eventqa.select_context_indices(5, 2, context_index=3)

        self.assertEqual(selected, [3])

    def test_eventqa_select_context_indices_rejects_invalid_context_index(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )

        with self.assertRaisesRegex(
            ValueError, "context-index 5 is out of range for 5 matched contexts"
        ):
            eventqa.select_context_indices(5, 1, context_index=5)

    def test_eventqa_manifest_records_selected_context_index_metadata(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        args = SimpleNamespace(
            dataset_root="/data",
            mab_repo="/repo",
            checkpoint_path="/tmp/checkpoint",
            model_checkpoint_id="checkpoint",
            requested_contexts=5,
            context_index=3,
            question_limit=None,
            eventqa_protocol="frozen_context_bank",
            strict_official_eventqa_prompt=True,
        )

        manifest = eventqa._build_manifest(
            "run",
            args,
            "now",
            git_status_before="clean",
            selected_context_indices=[3],
        )

        self.assertEqual(manifest["requested_contexts"], 5)
        self.assertEqual(manifest["context_index"], 3)
        self.assertEqual(manifest["selected_context_indices"], [3])
        self.assertTrue(manifest["strict_official_eventqa_prompt"])
        self.assertIn(
            "Output exactly one event from the candidate list.",
            manifest["eventqa_query_template"],
        )
        self.assertTrue(
            manifest["eventqa_query_template"].rstrip().endswith(
                "The event that happens next is:"
            )
        )

    def test_eventqa_manifest_records_first_line_prompt_template(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        args = SimpleNamespace(
            dataset_root="/data",
            mab_repo="/repo",
            checkpoint_path="/tmp/checkpoint",
            model_checkpoint_id="checkpoint",
            requested_contexts=5,
            context_index=3,
            question_limit=None,
            eventqa_protocol="frozen_context_bank",
            strict_official_eventqa_prompt=False,
            first_line_official_eventqa_prompt=True,
        )

        manifest = eventqa._build_manifest(
            "run",
            args,
            "now",
            git_status_before="clean",
            selected_context_indices=[3],
        )

        self.assertTrue(manifest["first_line_official_eventqa_prompt"])
        self.assertIn(
            "In your response, only include the event answer on the first line.",
            manifest["eventqa_query_template"],
        )
        self.assertNotIn("Answer:", manifest["eventqa_query_template"])

    def test_eventqa_query_only_payload_does_not_replay_construction(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        payload = {
            "chunks": ["chunk-1", "chunk-2"],
            "chunk_token_lengths": [4, 5],
            "memorization_prompts": ["m1", "m2"],
            "query_prompt": "query",
        }

        query_payload = eventqa._query_only_payload(payload)

        self.assertEqual(query_payload["chunks"], [])
        self.assertEqual(query_payload["chunk_token_lengths"], [])
        self.assertEqual(query_payload["memorization_prompts"], ["query"])
        self.assertEqual(payload["chunks"], ["chunk-1", "chunk-2"])

    def test_eventqa_frozen_query_retrieval_restores_bank_snapshot(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                threshold=0.005,
                retrieve_threshold=0.005,
                update_threshold=0.08,
                top_k=1,
                max_slots=16,
                retrieve_policy="threshold_topk",
                update_policy="thread_update",
            )
        )
        bank.write_back(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            LatentMemoryRetrievalResult(
                slots=[],
                scores=(),
                max_score=None,
                argmax_index=None,
                threshold_passed=False,
                retrieved_indices=(),
                retrieved_scores=(),
                bank_step=0,
            ),
        )
        lifecycle = {}
        proxy = eventqa._QueryReadOnlyBank(bank, lifecycle, freeze_retrieval_state=True)
        before = eventqa._bank_state_fingerprint(bank)

        proxy.begin_query()
        result = proxy.retrieve_with_context(
            torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
        )
        proxy.write_back(torch.tensor([[[0.0, 1.0]]]), result)
        proxy.capture_post_query()

        self.assertEqual(eventqa._bank_state_fingerprint(bank), before)
        self.assertFalse(lifecycle["bank_snapshot_changed_after_query"])
        self.assertEqual(lifecycle["query_write_count_delta"], 0)
        self.assertEqual(lifecycle["query_write_attempt_count_delta"], 1)

    def test_eventqa_query_proxy_can_disable_query_retrieval_without_state_change(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                threshold=0.005,
                retrieve_threshold=0.005,
                update_threshold=0.08,
                top_k=1,
                max_slots=16,
                retrieve_policy="threshold_topk",
                update_policy="thread_update",
            )
        )
        bank.write_back(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            LatentMemoryRetrievalResult(
                slots=[],
                scores=(),
                max_score=None,
                argmax_index=None,
                threshold_passed=False,
                retrieved_indices=(),
                retrieved_scores=(),
                bank_step=0,
                retrieval_step=0,
            ),
        )
        lifecycle = {}
        proxy = eventqa._QueryReadOnlyBank(
            bank,
            lifecycle,
            freeze_retrieval_state=True,
            disable_query_retrieval=True,
        )
        before = eventqa._bank_state_fingerprint(bank)
        retrieval_step_before = bank._retrieval_step

        proxy.begin_query()
        result = proxy.retrieve_with_context(
            torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
        )
        proxy.capture_post_query()

        self.assertEqual(result.retrieved_indices, ())
        self.assertEqual(len(result.slots), 0)
        self.assertEqual(result.retrieval_step, retrieval_step_before)
        self.assertEqual(bank._retrieval_step, retrieval_step_before)
        self.assertEqual(eventqa._bank_state_fingerprint(bank), before)
        self.assertFalse(lifecycle["bank_snapshot_changed_after_query"])

    def test_eventqa_bank_fingerprint_supports_bfloat16_slots(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )

        class FakeBank:
            def debug_summary(self):
                return {
                    "memory_retrieve_count": 1,
                    "retrieved_latent_count": 8,
                }

            def state_dict(self):
                return {
                    "step": 2,
                    "retrieval_step": 1,
                    "slots": [
                        {
                            "memory": torch.tensor(
                                [[1.0, 2.0]], dtype=torch.bfloat16
                            ),
                            "key": torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16),
                            "metadata": {"slot_id": 0},
                            "created_step": 0,
                            "last_access_step": 1,
                            "last_retrieved_step": 1,
                            "access_count": 1,
                            "last_score": 0.5,
                            "original_device": "cpu",
                            "original_dtype": "torch.bfloat16",
                        }
                    ],
                }

        fingerprint = eventqa._bank_state_fingerprint(FakeBank())

        self.assertIsInstance(fingerprint, str)
        self.assertEqual(len(fingerprint), 64)

    def test_eventqa_construction_trace_records_true_write_actions(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                threshold=0.005,
                retrieve_threshold=0.005,
                update_threshold=0.08,
                top_k=1,
                max_slots=16,
                retrieve_policy="threshold_topk",
                update_policy="thread_update",
            )
        )
        lifecycle = {"construction_turn_diagnostics": []}
        trace = {}
        restore = eventqa._install_eventqa_bank_trace(
            bank,
            trace,
            lifecycle,
            trace_construction_provenance=True,
        )

        retrieval = bank.retrieve_with_context(
            torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
        )
        bank.write_back(torch.tensor([[[1.0, 0.0]]]), retrieval)
        restore()

        self.assertEqual(len(lifecycle["construction_turn_diagnostics"]), 1)
        turn = lifecycle["construction_turn_diagnostics"][0]
        self.assertEqual(turn["write_action"], "insert")
        self.assertIsNone(turn["best_matched_score"])
        self.assertEqual(turn["slot_count_after_write"], 1)
        self.assertEqual(turn["slot_count_before_write"], 0)
        self.assertEqual(turn["target_slot_index"], 0)
        self.assertIsNone(turn["replaced_slot_index"])
        self.assertEqual(turn["created_step_after_write"], 1)
        self.assertEqual(turn["access_count_after_write"], 0)
        self.assertEqual(trace["last_retrieval"]["scores"], [])

    def test_eventqa_default_construction_trace_schema_is_unchanged(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(enabled=True, update_policy="thread_update")
        )
        lifecycle = {"construction_turn_diagnostics": []}
        restore = eventqa._install_eventqa_bank_trace(bank, {}, lifecycle)

        retrieval = bank.retrieve_with_context(torch.tensor([1.0, 0.0]))
        bank.write_back(torch.tensor([[1.0, 0.0]]), retrieval)
        restore()

        turn = lifecycle["construction_turn_diagnostics"][0]
        self.assertNotIn("target_slot_index", turn)
        self.assertNotIn("retrieval_score_decomposition_before_write", turn)

    def test_eventqa_query_trace_records_opt_in_score_decomposition(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        bank = LatentMemoryBank(
            LatentMemoryBankConfig(
                enabled=True,
                max_slots=2,
                top_k=1,
                retrieve_threshold=0.01,
                decay_alpha=0.05,
            )
        )
        bank.write(torch.tensor([[1.0, 0.0]], dtype=torch.float32))
        lifecycle = {
            "construction_turn_diagnostics": [],
            "query_phase_active": True,
        }
        trace = {}
        score_state = {}
        restore = eventqa._install_eventqa_bank_trace(
            bank,
            trace,
            lifecycle,
            trace_score_decomposition=True,
            score_trace_state=score_state,
            query_id="query-0",
        )

        result = bank.retrieve_with_context(
            torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
        )
        restore()

        decomposition = lifecycle["query_score_decomposition"]
        self.assertEqual(decomposition["query_id"], "query-0")
        self.assertEqual(decomposition["retrieved_indices"], [0])
        self.assertEqual(decomposition["retrieved_latent_count"], 1)
        self.assertTrue(decomposition["slots"][0]["selected_by_topk"])
        self.assertEqual(list(result.retrieved_indices), [0])
        self.assertIn("first_query", score_state)
        self.assertIn("previous_query", score_state)

    def test_eventqa_query_proxy_blocks_real_writes_and_records_attempts(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )

        class FakeBank:
            config = SimpleNamespace(retrieve_threshold=0.005)

            def __init__(self):
                self.write_count = 0
                self.retrieve_count = 0

            def __len__(self):
                return 1

            def write_back(self, *args, **kwargs):
                self.write_count += 1
                return True

            def debug_summary(self):
                return {
                    "slot_count": 1,
                    "memory_write_count": self.write_count,
                    "memory_retrieve_count": self.retrieve_count,
                    "thread_insert_count": 1,
                    "matched_replace_count": max(0, self.write_count - 1),
                    "capacity_evict_count": 0,
                    "write_action_counts": {
                        "insert": 1,
                        "replace_matched": max(0, self.write_count - 1),
                    },
                    "slots": [{"created_step": self.write_count}],
                }

        lifecycle = {}
        bank = FakeBank()
        proxy = eventqa._QueryReadOnlyBank(bank, lifecycle)

        proxy.write_back("construction")
        proxy.begin_query()
        proxy.write_back("query")
        proxy.capture_post_query()
        bank.write_count = 0
        proxy.capture_post_query()

        self.assertEqual(lifecycle["pre_query_bank_summary"]["write_count"], 1)
        self.assertEqual(lifecycle["post_query_bank_summary"]["write_count"], 1)
        self.assertEqual(lifecycle["query_write_count_delta"], 0)
        self.assertEqual(lifecycle["query_write_attempt_count_delta"], 1)
        self.assertTrue(lifecycle["query_read_only_enforced"])

    def test_eventqa_query_diagnostics_preserve_true_actions_and_raw_scores(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        lifecycle = {
            "pre_query_bank_summary": {
                "slot_count": 2,
                "write_count": 17,
                "retrieval_count": 16,
                "slot_indices": [0, 1],
                "slots": [],
                "true_insert_count": 2,
                "true_matched_replace_count": 15,
                "true_capacity_evict_count": 0,
                "true_replace_old_slot_count": 0,
                "write_action_counts": {"insert": 2, "replace_matched": 15},
            },
            "post_query_bank_summary": {
                "slot_count": 2,
                "write_count": 17,
                "retrieval_count": 17,
                "slot_indices": [0, 1],
                "slots": [],
                "true_insert_count": 2,
                "true_matched_replace_count": 15,
                "true_capacity_evict_count": 0,
                "true_replace_old_slot_count": 0,
                "write_action_counts": {"insert": 2, "replace_matched": 15},
            },
            "query_write_count_delta": 0,
            "query_write_attempt_count_delta": 1,
            "query_read_only_enforced": True,
        }
        query_turn = {
            "scores": [0.05, 0.04],
            "retrieved_indices": [0],
            "retrieved_scores": [0.05],
        }

        diagnostics = eventqa._query_memory_diagnostics(
            lifecycle, query_turn, retrieve_threshold=0.005
        )

        self.assertEqual(diagnostics["query_candidate_scores"], [0.05, 0.04])
        self.assertEqual(diagnostics["query_candidate_slot_count"], 2)
        self.assertTrue(diagnostics["query_slot_1_existed"])
        self.assertTrue(diagnostics["query_slot_1_lost_top_k1_ranking"])
        self.assertEqual(diagnostics["true_insert_count"], 2)
        self.assertEqual(diagnostics["true_matched_replace_count"], 15)
        self.assertEqual(diagnostics["true_capacity_evict_count"], 0)
        self.assertEqual(diagnostics["query_write_count_delta"], 0)

    def test_eventqa_context_summary_distinguishes_available_and_attempted_questions(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        context_payload = {
            "context_index": 0,
            "context_id": "eventqa-ctx-0",
            "question_count": 100,
            "chunks": ["chunk"],
            "chunk_token_lengths": [1],
        }

        summary = eventqa._build_context_summary(context_payload, [])

        self.assertEqual(summary["question_count_available"], 100)
        self.assertEqual(summary["question_count"], 0)

    def test_eventqa_frozen_context_summary_records_single_memorization_and_reuse(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        context_payload = {
            "context_index": 0,
            "context_id": "eventqa-ctx-0",
            "question_count": 100,
            "chunks": ["chunk-1", "chunk-2"],
            "chunk_token_lengths": [4, 5],
        }
        rows = [
            {
                "error_or_stop_reason": None,
                "bank_off_substring_exact_match": 0,
                "bank_on_substring_exact_match": 0,
                "improved": 0,
                "regressed": 0,
                "output_changed": False,
                "bank_on_final_slot_count": 2,
                "bank_on_query_turn_retrieved_indices": [0],
                "bank_on_query_turn_retrieved_latent_count": 8,
                "bank_on_query_turn_score_range": {"min": 0.1, "max": 0.1},
                "true_insert_count": 2,
                "true_matched_replace_count": 0,
                "true_capacity_evict_count": 0,
                "true_replace_old_slot_count": 0,
                "bank_on_query_turn_candidate_scores": [0.1, 0.05],
                "bank_on_query_turn_candidate_slot_count": 2,
                "bank_on_query_slot_1_lost_top_k1_ranking": True,
                "query_write_count": 0,
                "query_write_attempt_count": 1,
                "query_read_only_enforced": True,
                "cross_context_leakage_detected": False,
                "retrieved_latents_enter_weaver": True,
                "raw_retrieved_latents_enter_reasoner": False,
                "bank_off_eventqa_recall": 0.0,
                "bank_on_eventqa_recall": 0.0,
                "bank_off_empty_output": False,
                "bank_on_empty_output": False,
                "bank_off_format_flags": {},
                "bank_on_format_flags": {},
                "peak_cuda_memory": 100,
                "bank_instance_id": 123,
                "context_memorization_performed": index == 0,
                "query_write_count_delta": 0,
                "bank_snapshot_changed_after_query": False,
                "pre_query_bank_summary": {"slot_count": 2},
                "construction_turn_diagnostics": (
                    [
                        {
                            "construction_turn_index": 0,
                            "write_action": "insert",
                            "best_matched_score": None,
                            "slot_count_after_write": 1,
                        },
                        {
                            "construction_turn_index": 1,
                            "write_action": "insert",
                            "best_matched_score": 0.01,
                            "slot_count_after_write": 2,
                        },
                    ]
                    if index == 0
                    else []
                ),
            }
            for index in range(3)
        ]

        summary = eventqa._build_context_summary(
            context_payload,
            rows,
            eventqa_protocol="frozen_context_bank",
            cleanup_slot_count=0,
        )

        self.assertEqual(summary["context_memorization_count"], 1)
        self.assertTrue(summary["same_frozen_bank_reused_across_queries"])
        self.assertTrue(summary["all_query_write_deltas_zero"])
        self.assertTrue(summary["bank_snapshot_unchanged_across_queries"])
        self.assertEqual(summary["total_construction_tokens"], 9)
        self.assertEqual(summary["final_construction_slot_count"], 2)
        self.assertEqual(summary["construction_write_action_sequence"], ["insert", "insert"])
        self.assertTrue(summary["bank_reset_after_context"])

    def test_eventqa_construction_only_context_summary_records_bank_without_questions(self):
        eventqa = importlib.import_module(
            "scripts.eval.mab6b_weaver_space_bank_eventqa_65536_n5"
        )
        context_payload = {
            "context_index": 0,
            "context_id": "eventqa-ctx-0",
            "question_count": 100,
            "chunks": ["chunk-1", "chunk-2"],
            "chunk_token_lengths": [4, 5],
        }
        construction_result = {
            "context_memorization_performed": True,
            "pre_query_bank_summary": {"slot_count": 1},
            "true_insert_count": 1,
            "true_matched_replace_count": 1,
            "true_capacity_evict_count": 0,
            "true_replace_old_slot_count": 0,
            "construction_turn_diagnostics": [
                {
                    "construction_turn_index": 0,
                    "write_action": "insert",
                    "best_matched_score": None,
                    "slot_count_after_write": 1,
                },
                {
                    "construction_turn_index": 1,
                    "write_action": "replace_matched",
                    "best_matched_score": 0.06,
                    "slot_count_after_write": 1,
                },
            ],
        }

        summary = eventqa._build_context_summary(
            context_payload,
            [],
            eventqa_protocol="frozen_context_bank",
            cleanup_slot_count=0,
            construction_result=construction_result,
        )

        self.assertEqual(summary["context_memorization_count"], 1)
        self.assertEqual(summary["question_count"], 0)
        self.assertEqual(summary["final_construction_slot_count"], 1)
        self.assertEqual(summary["true_insert_count"], 1)
        self.assertEqual(summary["true_matched_replace_count"], 1)
        self.assertEqual(
            summary["construction_write_action_sequence"],
            ["insert", "replace_matched"],
        )

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

    def test_relaxation_setting_label_and_note_suppression(self):
        from scripts.eval import (
            mab6b_weaver_space_bank_detectiveqa_n10_retrieve_threshold_relaxation as relax,
        )

        self.assertEqual(relax._setting_label(0.03, 1), "rt003_topk1")
        self.assertEqual(relax._setting_label(0.005, 4), "rt0005_topk4")

        note_path = relax._suppress_research_note(
            output_dir=Path("/tmp/relaxation") / "rt003_topk1"
        )
        self.assertEqual(
            note_path,
            Path("/tmp/relaxation") / "rt003_topk1" / "suppressed_research_note.md",
        )

    def test_relaxation_aggregate_reads_worker_roots(self):
        from scripts.eval import (
            mab6b_weaver_space_bank_detectiveqa_n10_retrieve_threshold_relaxation as relax,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "rt003_topk1"
            second = root / "rt003_topk4"
            first.mkdir()
            second.mkdir()

            (first / "worker_result.json").write_text(
                json.dumps(
                    {
                        "setting_label": "rt003_topk1",
                        "retrieve_threshold": 0.03,
                        "top_k": 1,
                        "bank_off_EM": 0.0,
                        "bank_on_EM": 0.1,
                        "final_slot_counts": [16],
                        "query_turn_retrieved_indices_by_context": [[0]],
                        "query_turn_retrieved_latent_count_by_context": [8],
                        "bank_on_format_failure_count": 1,
                        "bank_off_format_failure_count": 0,
                        "files_written": ["a.json"],
                        "failed": False,
                    }
                ),
                encoding="utf-8",
            )
            (second / "worker_result.json").write_text(
                json.dumps(
                    {
                        "setting_label": "rt003_topk4",
                        "retrieve_threshold": 0.03,
                        "top_k": 4,
                        "bank_off_EM": 0.0,
                        "bank_on_EM": 0.0,
                        "final_slot_counts": [16],
                        "query_turn_retrieved_indices_by_context": [[0, 1, 2, 3]],
                        "query_turn_retrieved_latent_count_by_context": [32],
                        "bank_on_format_failure_count": 2,
                        "bank_off_format_failure_count": 0,
                        "files_written": ["b.json"],
                        "failed": False,
                    }
                ),
                encoding="utf-8",
            )

            aggregate = relax.aggregate_existing_artifacts(root)

        self.assertEqual(len(aggregate["rows"]), 2)
        self.assertEqual(
            [row["setting_label"] for row in aggregate["rows"]],
            ["rt003_topk1", "rt003_topk4"],
        )
        self.assertTrue(aggregate["rows"][1]["top_k_4_reached_32_latents"])


if __name__ == "__main__":
    unittest.main()
