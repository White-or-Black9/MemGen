#!/usr/bin/env python3
"""Formal frozen P7 EventQA update-threshold sensitivity at 0.05."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.exp.p7 import mab6b_weaver_space_bank_eventqa_65536_n5 as p7


def main() -> int:
    if "--update-threshold" in sys.argv:
        raise SystemExit("eventqa_p7_update_threshold_005.py fixes --update-threshold 0.05")
    sys.argv.extend(["--update-threshold", "0.05"])
    return p7.main()


if __name__ == "__main__":
    raise SystemExit(main())
