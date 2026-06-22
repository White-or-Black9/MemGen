"""MAB-5B: detective_qa raised shared-threshold Bank-off vs Bank-on on 10 contexts."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval import mab5a_detectiveqa_compressed_n10 as base


EXPERIMENT_NAME = "MAB-5B: detective_qa Raised Shared-threshold Bank-off vs Bank-on n10"
RUN_PREFIX = "detectiveqa-raised-shared-threshold-n10"
DEFAULT_OUTPUT_ROOT = "outputs/mab/raised_shared_threshold_detectiveqa_n10"
DEFAULT_THRESHOLD = 0.05
DEFAULT_TOP_K = 1
DEFAULT_MAX_SLOTS = 8
DEFAULT_RETRIEVE_POLICY = "threshold_topk"


base.EXPERIMENT_NAME = EXPERIMENT_NAME
base.RUN_PREFIX = RUN_PREFIX
base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
base.DEFAULT_THRESHOLD = DEFAULT_THRESHOLD
base.DEFAULT_TOP_K = DEFAULT_TOP_K
base.DEFAULT_MAX_SLOTS = DEFAULT_MAX_SLOTS
base.DEFAULT_RETRIEVE_POLICY = DEFAULT_RETRIEVE_POLICY


select_match_indices = base.select_match_indices
count_context_matches = base.count_context_matches
render_compressed_query_messages = base.render_compressed_query_messages
prompt_contains_chunk_leak = base.prompt_contains_chunk_leak
build_parser = base.build_parser
_build_manifest = base._build_manifest
main = base.main


if __name__ == "__main__":
    raise SystemExit(main())
