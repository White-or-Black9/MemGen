"""Compare EventQA Bank-off stability and Bank-on divergence offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.eventqa_transition_diagnostics import (
    compare_eventqa_records,
    load_eventqa_records,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--left",
        nargs="+",
        required=True,
        help="One or more EventQA run roots or parents containing run roots.",
    )
    parser.add_argument(
        "--right",
        nargs="+",
        required=True,
        help="One or more EventQA run roots or parents containing run roots.",
    )
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    left, resolved_left = load_eventqa_records(args.left)
    right, resolved_right = load_eventqa_records(args.right)
    payload = {
        "left_roots": resolved_left,
        "right_roots": resolved_right,
        "comparison": compare_eventqa_records(left, right),
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
