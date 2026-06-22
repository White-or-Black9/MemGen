import unittest

from scripts.eval import mab5b_raised_shared_threshold_detectiveqa_n10 as harness


class MAB5BRaisedSharedThresholdDetectiveQAN10Test(unittest.TestCase):
    def test_build_manifest_uses_raised_shared_threshold_contract(self):
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

        self.assertEqual(manifest["experiment_name"], harness.EXPERIMENT_NAME)
        self.assertEqual(manifest["query_mode"], "first-query-only")
        self.assertEqual(manifest["full_history_policy"], "over_capacity_invalid")
        self.assertEqual(manifest["threshold"], 0.05)

    def test_parser_defaults_keep_distinct_output_root(self):
        args = harness.build_parser().parse_args([])

        self.assertEqual(args.output_root, "outputs/mab/raised_shared_threshold_detectiveqa_n10")

    def test_threshold_contract_remains_single_threshold(self):
        self.assertFalse(hasattr(harness, "retrieve_threshold"))
        self.assertFalse(hasattr(harness, "update_threshold"))
        self.assertEqual(harness.DEFAULT_THRESHOLD, 0.05)
        self.assertEqual(harness.DEFAULT_TOP_K, 1)
        self.assertEqual(harness.DEFAULT_MAX_SLOTS, 8)
        self.assertEqual(harness.DEFAULT_RETRIEVE_POLICY, "threshold_topk")


if __name__ == "__main__":
    unittest.main()
