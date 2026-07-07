import tempfile
import unittest
from pathlib import Path


class MethodFiguresTest(unittest.TestCase):
    def test_render_all_exports_both_figures_in_all_formats(self):
        from scripts.figures.make_method_figures import render_all

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = render_all(Path(tmpdir))

            expected = {
                Path(tmpdir) / f"{stem}.{suffix}"
                for stem in ("fig1_method_architecture", "fig2_frozen_bank_protocol")
                for suffix in ("svg", "pdf", "tiff", "png")
            }
            self.assertEqual(set(outputs), expected)
            for path in expected:
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 1_000, path)

    def test_svg_exports_keep_required_semantics_as_editable_text(self):
        from scripts.figures.make_method_figures import render_all

        with tempfile.TemporaryDirectory() as tmpdir:
            render_all(Path(tmpdir))
            architecture = (
                Path(tmpdir) / "fig1_method_architecture.svg"
            ).read_text(encoding="utf-8")
            protocol = (
                Path(tmpdir) / "fig2_frozen_bank_protocol.svg"
            ).read_text(encoding="utf-8")

            self.assertIn("<text", architecture)
            self.assertIn("Trigger", architecture)
            self.assertIn("Weaver", architecture)
            self.assertIn("Reasoner", architecture)
            self.assertIn("Session-local latent bank", architecture)
            self.assertIn("max 16 slots", architecture)
            self.assertIn("query-time writes blocked", architecture)

            self.assertIn("<text", protocol)
            self.assertIn("Snapshot and freeze", protocol)
            self.assertIn("Question 1", protocol)
            self.assertIn("Question N", protocol)
            self.assertIn("query_write_count = 0", protocol)
            self.assertIn("bank_after_query = frozen_snapshot", protocol)

    def test_check_outputs_rejects_missing_or_noneditable_exports(self):
        from scripts.figures.make_method_figures import check_outputs, render_all

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            render_all(output_dir)
            self.assertEqual(check_outputs(output_dir), [])

            svg = output_dir / "fig1_method_architecture.svg"
            svg.write_text(svg.read_text(encoding="utf-8").replace("<text", "<path"))
            errors = check_outputs(output_dir)
            self.assertTrue(any("editable text" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
