#!/usr/bin/env python3
"""Formal P7 construction ablation: drop unmatched writes at full capacity."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.exp.p7 import mab6b_weaver_space_bank_eventqa_65536_n5 as p7


def main() -> int:
    if "--construction-write-policy" in sys.argv:
        raise SystemExit(
            "eventqa_p7_drop_capacity_eviction.py fixes "
            "--construction-write-policy drop_capacity_eviction"
        )
    sys.argv.extend(["--construction-write-policy", "drop_capacity_eviction"])
    return p7.main()


if __name__ == "__main__":
    raise SystemExit(main())
