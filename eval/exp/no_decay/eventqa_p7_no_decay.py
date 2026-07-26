"""Locked no-decay (alpha=0) ablation for the formal EventQA P7 protocol.

This wrapper delegates all construction, retrieval, generation, and scoring to
the formal P7 runner.  It accepts no user override of ``--decay-alpha`` so the
only method difference is removal of last-retrieved temporal decay.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from eval.exp.p7 import mab6b_weaver_space_bank_eventqa_65536_n5 as p7


def main() -> int:
    if "--decay-alpha" in sys.argv:
        raise SystemExit("no-decay ablation fixes --decay-alpha to 0.0; do not override it")
    sys.argv.extend(["--decay-alpha", "0.0"])
    return p7.main()


if __name__ == "__main__":
    raise SystemExit(main())
