"""Continuous-cost entry point for the unchanged BM25 top-2 evaluator."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.exp.common.continuous_explicit_cost import main

if __name__ == "__main__":
    raise SystemExit(main(["--method", "bm25_top2", *sys.argv[1:]]))
