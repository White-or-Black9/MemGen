import tempfile
import unittest
from pathlib import Path

from scripts.eval.memory_benchmark_campaign_baseline import build_manifest, validate_manifest


class MemoryBenchmarkCampaignBaselineTest(unittest.TestCase):
    def test_build_manifest_hashes_required_eventqa_and_paper_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            paper = repo_root / "draft.md"
            aggregate = repo_root / "eventqa.json"
            paper.write_text("EventQA baseline", encoding="utf-8")
            aggregate.write_text(
                '{"schema_version":"eventqa-final-table-package/v1"}',
                encoding="utf-8",
            )

            manifest = build_manifest(
                repo_root=repo_root,
                required_paths=[paper, aggregate],
                accepted_commit="14767eb",
            )

            self.assertEqual(manifest["accepted_commit"], "14767eb")
            self.assertEqual(
                manifest["schema_version"], "memory-benchmark-campaign-baseline/v1"
            )
            self.assertTrue(manifest["files"]["draft.md"]["sha256"])
            self.assertTrue(manifest["files"]["eventqa.json"]["sha256"])

    def test_validate_manifest_fails_on_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            paper = repo_root / "draft.md"
            paper.write_text("EventQA baseline", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "baseline hash mismatch"):
                validate_manifest(
                    {
                        "files": {
                            "draft.md": {
                                "size_bytes": paper.stat().st_size,
                                "sha256": "deadbeef",
                            }
                        }
                    },
                    repo_root=repo_root,
                )


if __name__ == "__main__":
    unittest.main()
