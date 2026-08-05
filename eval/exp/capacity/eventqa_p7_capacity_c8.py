#!/usr/bin/env python3
"""Formal frozen P7 EventQA capacity ablation with C=8 slots.

Only ``max_slots`` changes from the formal P7 configuration (C=16).  All
other evaluation and bank settings remain owned by the canonical P7 runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.exp.p7 import mab6b_weaver_space_bank_eventqa_65536_n5 as p7


def main() -> int:
    if "--max-slots" in sys.argv:
        raise SystemExit(
            "eventqa_p7_capacity_c8.py fixes --max-slots 8; "
            "use the canonical P7 runner for a different capacity."
        )
    sys.argv.extend(["--max-slots", "8"])
    return p7.main()


if __name__ == "__main__":
    raise SystemExit(main())
