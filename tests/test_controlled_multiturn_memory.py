import inspect
import unittest
from types import SimpleNamespace

from scripts.eval.phase8c_controlled_memory import (
    build_verification,
    build_summary,
    build_turn_prompts,
    compute_metrics,
    exact_match,
    generate_episodes,
    get_group_config,
    leakage_flags,
    parse_relaxed_answer,
    parse_strict_answer,
    required_episode_fields_present,
)


class ControlledMultiTurnMemoryTest(unittest.TestCase):
    def test_episode_generation_is_deterministic_and_has_two_types(self):
        first = generate_episodes(4, seed=42)
        second = generate_episodes(4, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(
            {episode.episode_type for episode in first},
            {"exact_code", "semantic_relation"},
        )

    def test_turn3_prompt_excludes_early_fact_gold_and_distractor(self):
        episode = generate_episodes(1, seed=42)[0]
        prompts = build_turn_prompts(episode)
        turn3_text = " ".join(message["content"] for message in prompts[2])
        flags = leakage_flags(turn3_text, episode)
        self.assertFalse(flags["prompt_contains_early_fact"])
        self.assertFalse(flags["prompt_contains_gold_answer"])
        self.assertFalse(flags["prompt_contains_distractor"])
        self.assertTrue(flags["leakage_passed"])

    def test_g3_oracle_prompt_includes_early_fact_and_gold(self):
        episode = generate_episodes(1, seed=42)[0]
        prompts = build_turn_prompts(episode, oracle_visible=True)
        turn3_text = " ".join(message["content"] for message in prompts[2])
        flags = leakage_flags(turn3_text, episode, oracle_visible=True)
        self.assertIn(episode.early_fact_text, turn3_text)
        self.assertIn(episode.gold_answer, turn3_text)
        self.assertTrue(flags["prompt_contains_early_fact"])
        self.assertTrue(flags["prompt_contains_gold_answer"])
        self.assertTrue(flags["leakage_passed"])

    def test_leakage_failure_marks_episode_invalid(self):
        episode = generate_episodes(1, seed=42)[0]
        leaked_prompt = (
            f"{episode.early_fact_text}\n"
            f"Answer with {episode.gold_answer}."
        )
        flags = leakage_flags(leaked_prompt, episode)
        self.assertTrue(flags["prompt_contains_early_fact"])
        self.assertTrue(flags["prompt_contains_gold_answer"])
        self.assertFalse(flags["leakage_passed"])

    def test_strict_parser_extracts_last_answer(self):
        response = "<answer>wrong</answer>\n<answer>482731</answer>"
        self.assertEqual(parse_strict_answer(response), "482731")

    def test_strict_parser_rejects_untagged_answer(self):
        self.assertIsNone(
            parse_strict_answer("The access code is 770487.")
        )

    def test_relaxed_exact_code_extracts_unique_standalone_code(self):
        parsed = parse_relaxed_answer(
            "The access code is 770487.",
            "exact_code",
        )
        self.assertEqual(parsed.answer, "770487")
        self.assertEqual(parsed.parser_mode, "exact_code_single_candidate")
        self.assertTrue(parsed.success)

    def test_relaxed_exact_code_marks_multiple_codes_ambiguous(self):
        parsed = parse_relaxed_answer(
            "The old code is 111111 and the new code is 770487.",
            "exact_code",
        )
        self.assertIsNone(parsed.answer)
        self.assertEqual(parsed.parser_mode, "ambiguous")
        self.assertFalse(parsed.success)

    def test_relaxed_exact_code_returns_none_without_code(self):
        parsed = parse_relaxed_answer(
            "I do not know the access code.",
            "exact_code",
        )
        self.assertIsNone(parsed.answer)
        self.assertEqual(parsed.parser_mode, "none")
        self.assertFalse(parsed.success)

    def test_relaxed_parser_does_not_accept_gold_answer(self):
        parameters = inspect.signature(parse_relaxed_answer).parameters
        self.assertNotIn("gold_answer", parameters)
        with self.assertRaises(TypeError):
            parse_relaxed_answer(
                "The code is 111111.",
                "exact_code",
                gold_answer="770487",
            )

    def test_relaxed_semantic_relation_normalizes_short_direct_answer(self):
        episode = generate_episodes(2, seed=42)[1]
        metrics = compute_metrics('"Room Juniper."', episode)
        self.assertEqual(metrics["parsed_answer_relaxed"], "Room Juniper")
        self.assertTrue(metrics["relaxed_exact_match"])

    def test_verbose_semantic_response_is_not_fuzzy_matched(self):
        episode = generate_episodes(2, seed=42)[1]
        metrics = compute_metrics(
            "The assigned archive is Room Juniper.",
            episode,
        )
        self.assertFalse(metrics["relaxed_exact_match"])

    def test_strict_success_is_reused_by_relaxed_parser(self):
        parsed = parse_relaxed_answer(
            "Ignore 111111. <answer>770487</answer>",
            "exact_code",
        )
        self.assertEqual(parsed.answer, "770487")
        self.assertEqual(parsed.parser_mode, "strict_tag")
        self.assertTrue(parsed.success)

    def test_exact_match_numeric_code(self):
        self.assertTrue(exact_match(" 482731 ", "482731"))
        self.assertFalse(exact_match("482732", "482731"))

    def test_compute_metrics_records_strict_and_relaxed_fields(self):
        episode = generate_episodes(1, seed=42)[0]
        metrics = compute_metrics(
            "The access code for Project Lumen is 770487.",
            episode,
        )
        self.assertEqual(
            set(metrics),
            {
                "parsed_answer_strict",
                "parsed_answer_relaxed",
                "strict_parser_success",
                "relaxed_parser_success",
                "parser_mode",
                "strict_exact_match",
                "relaxed_exact_match",
            },
        )

    def test_g3_raw_response_is_relaxed_match_only(self):
        episode = generate_episodes(1, seed=42)[0]
        metrics = compute_metrics(
            "The access code for Project Lumen is 770487.",
            episode,
        )
        self.assertFalse(metrics["strict_exact_match"])
        self.assertTrue(metrics["relaxed_exact_match"])

    def test_g0_and_g2_raw_responses_fail_both_metrics(self):
        episode = generate_episodes(1, seed=42)[0]
        responses = [
            (
                "I'm sorry, but I don't have any specific information about "
                "a Project Lumen access code."
            ),
            (
                "I'm sorry,I don't have any specific information about a "
                "Project Lumen access code."
            ),
        ]
        for response in responses:
            with self.subTest(response=response):
                metrics = compute_metrics(response, episode)
                self.assertFalse(metrics["strict_exact_match"])
                self.assertFalse(metrics["relaxed_exact_match"])

    def test_turn3_prompt_uses_calibrated_instruction_for_all_groups(self):
        episode = generate_episodes(1, seed=42)[0]
        for oracle_visible in (False, True):
            prompts = build_turn_prompts(
                episode,
                oracle_visible=oracle_visible,
            )
            turn3 = prompts[2][1]["content"]
            self.assertIn("Return exactly one line:", turn3)
            self.assertIn("<answer>VALUE</answer>", turn3)
            self.assertIn("Do not include any other text.", turn3)

    def test_group_configs_map_to_expected_memory_modes(self):
        disabled = get_group_config("G0_disabled")
        simple = get_group_config("G1_vA_simple")
        thread_update = get_group_config("G2_vA_thread_update")
        self.assertFalse(disabled["memory_enabled"])
        self.assertEqual(disabled["memory_mode"], "disabled")
        self.assertEqual(simple["update_policy"], "replace_oldest")
        self.assertEqual(thread_update["update_policy"], "thread_update")

    def test_disabled_artifact_fields_use_no_bank(self):
        episode = generate_episodes(1, seed=42)[0]
        record = {
            "episode_id": episode.episode_id,
            "group": "G0_disabled",
            "entity": episode.entity,
            "attribute": episode.attribute,
            "gold_answer": episode.gold_answer,
            "turns": [],
            "final_answer": None,
            "parsed_answer_strict": None,
            "parsed_answer_relaxed": None,
            "strict_parser_success": False,
            "relaxed_parser_success": False,
            "parser_mode": "none",
            "strict_exact_match": False,
            "relaxed_exact_match": False,
            "exact_match": False,
            "exact_match_metric": "strict_exact_match_deprecated_alias",
            "reward": 0.0,
            "valid_episode": False,
            "invalid_reason": "dry_run",
            "memory_bank_debug": None,
            "bank_lifecycle": {
                "bank_created": False,
                "bank_id": None,
                "initial_slots": None,
                "final_slots": None,
            },
            "trigger_calls": 0,
            "weaver_prompt_calls": 0,
            "weaver_inference_calls": 0,
            "latency": 0.0,
            "errors": [],
        }
        self.assertTrue(required_episode_fields_present(record))
        self.assertFalse(record["bank_lifecycle"]["bank_created"])
        self.assertIsNone(record["memory_bank_debug"])

    def test_summary_schema_contains_required_fields(self):
        records = [
            {
                "valid_episode": True,
                "strict_exact_match": True,
                "relaxed_exact_match": True,
                "strict_parser_success": True,
                "relaxed_parser_success": True,
                "exact_match": True,
                "reward": 1.0,
                "latency": 0.5,
                "errors": [],
                "turns": [
                    {},
                    {},
                    {"leakage_flags": {"leakage_passed": True}},
                ],
                "bank_lifecycle": {"bank_created": False},
                "memory_boundary_checks": {
                    "retrieved_memory_reasoner_only": {"passed": True}
                },
            }
        ]
        summary = build_summary("G0_disabled", records, total_latency=0.5)
        self.assertEqual(summary["summary"]["sample_count"], 1)
        self.assertEqual(summary["summary"]["valid_episode_count"], 1)
        self.assertEqual(summary["summary"]["strict_exact_match_count"], 1)
        self.assertEqual(summary["summary"]["relaxed_exact_match_count"], 1)
        self.assertEqual(summary["summary"]["strict_parser_success_count"], 1)
        self.assertEqual(summary["summary"]["relaxed_parser_success_count"], 1)
        self.assertEqual(summary["summary"]["exact_match_count"], 1)
        self.assertEqual(
            summary["summary"]["exact_match_metric"],
            "strict_exact_match_deprecated_alias",
        )
        self.assertEqual(summary["summary"]["leakage_pass_count"], 1)

    def test_verification_schema_contains_both_metric_sets(self):
        records = [
            {
                "valid_episode": True,
                "strict_exact_match": False,
                "relaxed_exact_match": True,
                "strict_parser_success": False,
                "relaxed_parser_success": True,
                "exact_match": False,
                "reward": 0.0,
                "latency": 0.5,
                "errors": [],
                "turns": [
                    {},
                    {},
                    {"leakage_flags": {"leakage_passed": True}},
                ],
                "bank_lifecycle": {"bank_created": False},
                "memory_boundary_checks": {
                    "retrieved_memory_reasoner_only": {"passed": True}
                },
            }
        ]
        summary = build_summary("G3_oracle_visible", records, 0.5)
        verification = build_verification(
            records,
            summary,
            get_group_config("G3_oracle_visible"),
            peak_cuda_memory_bytes=None,
        )
        self.assertEqual(verification["strict_exact_match_count"], 0)
        self.assertEqual(verification["relaxed_exact_match_count"], 1)
        self.assertEqual(verification["strict_parser_success_count"], 0)
        self.assertEqual(verification["relaxed_parser_success_count"], 1)
        self.assertEqual(
            verification["exact_match_metric"],
            "strict_exact_match_deprecated_alias",
        )

    def test_validate_args_rejects_group_memory_mismatch(self):
        from scripts.eval.phase8c_controlled_memory import validate_args

        args = SimpleNamespace(
            group="G2_vA_thread_update",
            memory_mode="disabled",
            batch_size=1,
            sample_count=1,
            seed=42,
            max_response_length=32,
        )
        with self.assertRaisesRegex(ValueError, "requires memory_mode"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
