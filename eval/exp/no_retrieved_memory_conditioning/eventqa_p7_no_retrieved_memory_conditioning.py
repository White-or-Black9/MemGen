"""Stable eval/exp entrypoint for the P7 conditioning ablation."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(
            Path(__file__).resolve().parents[3]
            / "scripts/eval/eventqa_p7_no_retrieved_memory_conditioning.py"
        ),
        run_name="__main__",
    )
