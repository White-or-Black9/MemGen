#!/usr/bin/env python3
"""Formal P7 EventQA runner with query-time bank mutation enabled.

Diagnostic only: it uses the canonical P7 runner and changes no dataset,
prompt, scorer, checkpoint, generation budget, or bank hyperparameter.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.exp.p7 import mab6b_weaver_space_bank_eventqa_65536_n5 as p7


def main() -> int:
    if "--allow-query-bank-mutation" not in sys.argv:
        sys.argv.append("--allow-query-bank-mutation")
    return p7.main()


if __name__ == "__main__":
    raise SystemExit(main())
