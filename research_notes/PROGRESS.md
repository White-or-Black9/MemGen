# Project Progress

## Current State

- Current status: Version A-aligned `thread_update` completed; target-task
  baseline planning is next.
- Status: `completed`
- Last updated: 2026-06-12
- Decay/fallback implementation audit: `completed`
- Method/plan alignment update: `completed`
- Step 2 structured retrieval context: `completed`
- Step 3 thread-aware write-back: `completed`
- Step 4 thread-update mechanism smoke: `completed`
  (`EXP-20260612-024`)
- Stop condition: Reached; do not enter a new implementation or experiment
  phase without explicit approval.
- Next recommended step: complete notes review and commit preparation, then plan
  and establish an Original MemGen / disabled-memory TriviaQA baseline. Version
  B remains not started.

## Research Goal

Add an optional session-level Retrieval-Augmented Recurrent Latent Memory Bank at
inference time without changing MemGen's training workflows.

## Phase 0 Outcome

Phase 0 remains complete under the current roadmap.

## Phase 1 Outcome

Phase 1 is complete.

- [x] Audited inference entry file and main dispatch.
- [x] Audited config loading and runtime config handoff.
- [x] Audited static and dynamic session/sample/episode boundaries.
- [x] Located Trigger call sites.
- [x] Located Weaver call sites.
- [x] Located latent memory generation and Reasoner injection sites.
- [x] Located generation outputs and evaluation hooks.
- [x] Marked protected Weaver / Trigger training boundaries.
- [x] Assessed candidate LatentMemoryBank integration points and risks.
- [x] Updated research notes only. No core code or training workflow changed.

## Phase 2 Outcome

Phase 2 is complete with blocking caveats.

- [x] Checked environment, repository state, and runnable project environment.
- [x] Located an official minimal evaluation route based on GSM8K + Qwen2.5-1.5B-Instruct + Weaver-SFT.
- [x] Verified that the current `base` environment is not suitable for MemGen execution.
- [x] Verified that the `memgen` environment can initialize the project correctly.
- [x] Verified that local base-model and dataset caches are available offline.
- [x] Ran the original evaluation path on 1 GSM8K sample through `Config -> MemGenModel.from_config -> MemGenRunner.evaluate()`.
- [x] Confirmed that model loading, dataset loading, interaction setup, and generation start all work in the recommended environment.
- [x] Confirmed that the official static-eval path currently fails at result recording, leaving `answer.json` empty.
- [x] Confirmed separately that the original generation path can produce a completion when recorder logic is bypassed in a script-only harness.
- [x] Reconfirmed that official LoRA loading is still not trustworthy; no smoke result from this Phase is a valid scientific baseline.
- [x] Updated research notes only. No core code, Weaver training, or Trigger training logic changed.

## Temporary Environment Alignment Outcome

The temporary environment alignment phase is complete.

- [x] Reviewed `README.md`, `requirements.txt`, `memgen.yml`, the official
  Qwen2.5 GSM8K evaluation script, and the GSM8K configuration.
- [x] Confirmed the active shell is the `base` environment with Python `3.13.9`
  and must not be used to run MemGen.
- [x] Confirmed the existing `memgen` environment is present at
  `/home/baishilong/miniconda3/envs/memgen`.
- [x] Confirmed the existing `memgen` Python is `3.10.20`, matching the README
  setup instructions.
- [x] Confirmed all required Python imports succeed and `pip check` reports no
  broken requirements.
- [x] Confirmed CUDA and BF16 are available from the `memgen` environment when
  run outside the filesystem sandbox on one NVIDIA RTX A6000.
- [x] Confirmed the GSM8K YAML, local Qwen snapshot, official MemGen checkpoint,
  and local GSM8K dataset cache are readable.
- [x] Determined that no environment recreation or package installation is
  justified before the Repair Phase.
- [x] Recorded environment specification drift and the broken PATH-level
  `conda` shim.
- [x] Modified research notes only. No project code or environment package was
  changed.

## Temporary Repair Phase Outcome

The temporary Repair Phase is complete.

- [x] Reproduced `BUG-0001` with the original nested PEFT loading path.
- [x] Confirmed that the original path expected 392 incorrectly nested adapter
  keys while each official checkpoint contains 112 trained q/v LoRA tensors.
- [x] Replaced only the checkpoint adapter restoration logic; initial Weaver and
  Trigger training adapter construction remains unchanged.
- [x] Verified all 112 Weaver and all 112 Trigger tensors against the official
  safetensors with no missing, unexpected, shape-mismatched, or value-mismatched
  entries.
- [x] Reproduced `BUG-0002` as a caller/recorder contract mismatch: the runner
  passed one string and one dictionary where the recorder requires two lists.
- [x] Normalized gathered static-eval batches and preserved sample order and
  metric semantics.
- [x] Ran the official `Config -> MemGenModel.from_config ->
  MemGenRunner.evaluate()` path on one GSM8K sample with seed 42 and batch size 1.
- [x] Confirmed a non-empty `answer.json` with one prediction and one summary
  record.
- [x] Confirmed the generation path called the Trigger decision entry 85 times,
  Weaver prompt augmentation once, and Weaver inference augmentation three
  times.
- [x] Did not modify Weaver training, Trigger training, training scripts,
  dependencies, or environment packages.
- [x] Did not enter Phase 3.

## Temporary Repair Review Outcome

The temporary Repair Review and Sanity Check is complete.

- [x] Reviewed the complete core diff in `memgen/model/modeling_memgen.py` and
  `memgen/runner.py`.
- [x] Confirmed no diff under `memgen/trainer/`, `scripts/train/`, Weaver/Trigger
  training scripts, or `memgen/model/modeling_utils.py`.
- [x] Confirmed fresh Weaver SFT uses `load_model_path=null` and does not enter
  the repaired checkpoint restoration branch.
- [x] Confirmed checkpoint-driven runs keep the same control flow while now
  restoring the trained adapter tensors correctly.
- [x] Parameterized the Repair Phase smoke harness only; no additional core-code
  change was required.
- [x] Ran three GSM8K test samples with seed 42 and batch size 1 through
  `Config -> MemGenModel.from_config -> MemGenRunner.evaluate()`.
- [x] Confirmed `answer.json` contains exactly three non-empty prediction records
  and one summary record.
- [x] Reconfirmed exact 112/112 tensor matches for both Weaver and Trigger
  adapters with no missing, unexpected, shape, or value mismatches.
- [x] Confirmed aggregate generation tracing: 193 Trigger decision calls, three
  Weaver prompt calls, and eight Weaver inference calls.
- [x] Did not treat the three-sample reward as a baseline metric.
- [x] Did not enter Phase 3.

## Phase 3 Outcome

Phase 3 is complete.

- [x] Accepted `memgen-gsm8k-sft-official-v1` as the Original MemGen comparator.
- [x] Fixed the comparison set to GSM8K `main/test` indices 0 through 19.
- [x] Used seed 42, batch size 1, greedy decoding, and maximum response length
  1024.
- [x] Ran the official `Config -> MemGenModel.from_config ->
  MemGenRunner.evaluate()` path.
- [x] Produced 20 non-empty prediction records and one summary record.
- [x] Recorded mean `compute_reward=0.60` on the fixed 20-sample subset.
- [x] Reconfirmed exact 112/112 Weaver and 112/112 Trigger adapter loading.
- [x] Recorded 1,722 Trigger decision calls, 20 Weaver prompt calls, and 43
  Weaver inference calls.
- [x] Recorded total latency 115.728 seconds, mean latency 5.786 seconds/sample,
  and peak allocated CUDA memory 9,415,716,352 bytes.
- [x] Replayed fixed samples 0, 1, and 2 and obtained identical response-token
  and augmentation-mask SHA-256 hashes.
- [x] Archived prediction, verification, TensorBoard, and metric-contract
  artifacts under `outputs/baseline/`.
- [x] Did not modify core method or training code in Phase 3.
- [x] Did not enter Phase 4.

## Phase 4 Outcome

Phase 4 is complete.

- [x] Added standalone `LatentMemoryBankConfig`, `LatentMemorySlot`, and
  `LatentMemoryBank` in `memgen/model/latent_memory_bank.py`.
- [x] Added disabled-by-default
  `configs/latent_memory_bank/default.yaml`.
- [x] Added 16 standard-library unit tests in
  `tests/test_latent_memory_bank.py`.
- [x] Implemented disabled and empty-bank no-op behavior.
- [x] Implemented recent-token mean query pooling and memory mean key pooling.
- [x] Implemented cosine similarity with exponential recency decay.
- [x] Implemented threshold, top-k, and threshold-plus-top-k retrieval.
- [x] Implemented append, replace-lowest-score, and replace-oldest capacity
  behavior.
- [x] Enforced Phase 4 batch size 1 tensor shapes.
- [x] Enforced detach and clone on write and detached clone on retrieval.
- [x] Confirmed caller mutation of retrieved tensors or nested metadata cannot
  modify bank-owned slot state.
- [x] Implemented explicit storage and retrieval device/dtype movement.
- [x] Defined `_step` as successful memory-write count, not generation-token
  count.
- [x] Defined `replace` as lowest-`last_score` replacement with an oldest-slot
  fallback when all slots are unscored.
- [x] Added debug summary and detached state-dict-like snapshots.
- [x] Passed compilation, YAML parsing, and all 16 unit tests.
- [x] Confirmed production inference, `generate()`, runner, trainer, and training
  scripts do not import or call the module.
- [x] Confirmed importing `MemGenModel` does not load the memory-bank module.
- [x] Did not modify existing GSM8K configuration or original inference behavior.
- [x] Did not enter Phase 5.

## Phase 5 Outcome

Phase 5 is complete.

- [x] Integrated the optional LatentMemoryBank into inference only.
- [x] Kept the bank session-local and owned by each interaction-manager
  `run_agent_loop()` call.
- [x] Used one bank per single-turn session and one bank shared across all turns
  in one multi-turn episode.
- [x] Passed the bank explicitly into `MemGenModel.generate()` and did not store
  any bank object on `MemGenModel`.
- [x] Preserved the original disabled path by keeping
  `latent_memory_bank=None` / `enabled=false` on the original code branch with no
  new retrieval, write, mask, or tensor-packaging work.
- [x] Implemented Version A retrieval so retrieved memory is injected only into
  the Reasoner path and is never passed into `reasoner_to_weaver()`,
  `augment_prompt()`, or `augment_inference()`.
- [x] Wrote only reasoner-space `latent_inputs_embeds` into the bank after
  `weaver_to_reasoner(...)`.
- [x] Added explicit retrieved-memory attention-mask handling and separate debug
  bookkeeping for `memory_write_count`, `memory_retrieve_count`,
  `retrieved_latent_count`, `new_latent_count`, and `slot_count`.
- [x] Rejected `enabled=true` evaluation with `batch_size > 1` and kept
  disabled mode unrestricted.
- [x] Added lightweight integration tests for disabled no-op, empty-bank no-op,
  session reset, no cross-sample leakage, Reasoner-only injection,
  reasoner-space writes, dtype/device compatibility, and enabled batch-size
  rejection.
- [x] Passed `py_compile`, full `unittest`, and `git diff --check`.
- [x] Ran disabled-path golden replay `EXP-20260612-010` on GSM8K samples
  `0..2`; response-token hashes, augmentation-mask hashes, Trigger call count,
  Weaver prompt count, and Weaver inference count matched
  `EXP-20260611-007` exactly.
- [x] Ran enabled debug `EXP-20260612-011` on GSM8K sample `0`; the run did not
  crash and recorded 4 writes, 3 retrievals, 24 retrieved latent tokens,
  32 newly written latent tokens, and 4 resident slots.
- [x] Did not modify `memgen/trainer/**`, `scripts/train/**`, Weaver training
  logic, Trigger training logic, or implement Version B.
- [x] Did not enter Phase 6.

## Phase 6 Outcome

Phase 6 is complete.

- [x] Ran a 20-sample disabled-path equivalence test on GSM8K test IDs `0..19`
  against the frozen Phase 3 baseline `EXP-20260611-006`.
- [x] Used seed `42`, batch size `1`, greedy decoding, and maximum response
  length `1024`.
- [x] Kept `latent_memory_bank` disabled and verified `memory_bank_debug=null`.
- [x] Confirmed `answer.json` remained non-empty and contained exactly 20
  prediction records plus one summary record.
- [x] Confirmed summary `compute_reward=0.60`, matching
  `EXP-20260611-006` exactly.
- [x] Confirmed every response-token SHA-256 hash matched the frozen baseline.
- [x] Confirmed every augmentation-mask SHA-256 hash matched the frozen
  baseline.
- [x] Confirmed Trigger decision calls matched exactly: `1722`.
- [x] Confirmed Weaver prompt augmentation calls matched exactly: `20`.
- [x] Confirmed Weaver inference augmentation calls matched exactly: `43`.
- [x] Reconfirmed adapter loading integrity:
  Weaver `112/112`, Trigger `112/112`, with zero missing, unexpected, shape, or
  value mismatches.
- [x] Re-ran `git diff --check`, `py_compile`, and full `unittest`; all passed.
- [x] Confirmed no diff under `memgen/trainer/**`, `scripts/train/**`,
  `memgen/model/weaver.py`, or `memgen/model/trigger.py`.
- [x] Found no disabled-path regression and no new blocking bug.
- [x] Did not modify core method code in this phase.
- [x] Did not enter Phase 7.

## Phase 7 Outcome

Phase 7 is complete.

- [x] Ran only enabled-path bounded debug and stability checks; no performance
  claim was made.
- [x] Kept seed `42`, batch size `1`, greedy decoding, and maximum response
  length `1024`.
- [x] Confirmed `git status` was clean before the run and protected training
  paths had no diff.
- [x] Re-ran `git diff --check`, `py_compile`, and full `unittest`; all passed.
- [x] Ran Tier 1 smoke on one GSM8K test sample in enabled mode.
- [x] Ran Tier 2 small stability on GSM8K test samples `0..2` in enabled mode.
- [x] Ran Tier 3 bounded capacity on GSM8K test samples `0..4` in enabled mode.
- [x] Confirmed all enabled runs wrote non-empty `answer.json` files with the
  expected prediction count plus one summary record.
- [x] Confirmed no crash, NaN, OOM, CUDA error, shape mismatch, device mismatch,
  or dtype mismatch was observed in any Tier.
- [x] Confirmed each single-turn session started with `initial_slots=0`.
- [x] Confirmed no cross-sample leakage across Tier 2 or Tier 3 sessions.
- [x] Confirmed retrieved memory remained Reasoner-only; Weaver input token
  counts always matched `reasoner_to_weaver` input token counts.
- [x] Confirmed stored latent memories remained reasoner-space tensors with
  hidden size `1536`.
- [x] Confirmed slot storage remained explicit: CPU storage, original device
  `cuda:0`, original dtype `torch.bfloat16`, stored dtype `torch.bfloat16`.
- [x] Confirmed `slot_count` never exceeded `max_slots=8`.
- [x] Observed no replacement-policy activation in this bounded run because the
  largest per-session slot count was `4`.
- [x] Ran a post-Phase-7 capacity-trigger supplement with `max_slots=2` on one
  real enabled session and confirmed replacement activation in the real
  inference path.
- [x] Confirmed the supplement recorded `append_count=2`, `replace_count=2`,
  `rejected_write_count=0`, and
  `update_action_trace=["append", "append", "replace", "replace"]`.
- [x] Confirmed the supplement kept `final slot_count=2 <= max_slots=2` while
  `memory_write_count=4 > max_slots`.
- [x] Resolved the only outstanding Phase 7 warning: replacement policy is now
  observed in the real enabled debug path.
- [x] Recorded Tier 1 stats: writes `4`, retrieves `3`, retrieved latents `24`,
  new latents `32`, slot count `4`, latency `8.658 s`, peak CUDA memory
  `9,385,351,168` bytes.
- [x] Recorded Tier 2 per-session stats:
  sample 0 -> writes `4`, retrieves `3`, retrieved latents `24`, new latents
  `32`, slot count `4`;
  sample 1 -> writes `2`, retrieves `1`, retrieved latents `8`, new latents
  `16`, slot count `2`;
  sample 2 -> writes `4`, retrieves `3`, retrieved latents `24`, new latents
  `32`, slot count `4`;
  total latency `14.066 s`, mean latency `4.689 s/sample`, peak CUDA memory
  `9,385,351,168` bytes.
- [x] Recorded Tier 3 per-session stats:
  sample 0 -> writes `4`, retrieves `3`, retrieved latents `24`, new latents
  `32`, slot count `4`;
  sample 1 -> writes `2`, retrieves `1`, retrieved latents `8`, new latents
  `16`, slot count `2`;
  sample 2 -> writes `4`, retrieves `3`, retrieved latents `24`, new latents
  `32`, slot count `4`;
  sample 3 -> writes `2`, retrieves `1`, retrieved latents `8`, new latents
  `16`, slot count `2`;
  sample 4 -> writes `4`, retrieves `3`, retrieved latents `24`, new latents
  `32`, slot count `4`;
  total latency `21.562 s`, mean latency `4.312 s/sample`, peak CUDA memory
  `9,395,434,496` bytes.
- [x] Found no new blocking bug in Phase 7.
- [x] Modified only the debug harness and research notes in this Phase.
- [x] Added debug-only bank summary fields and debug-harness CLI overrides for
  capacity-trigger validation; no disabled-path or training-path semantics were
  changed.
- [x] Did not modify `memgen/trainer/**`, `scripts/train/**`, Weaver training
  logic, Trigger training logic, or implement Version B.
- [x] Did not enter Phase 8.

## End-of-Day Validation Outcome

The 2026-06-11 end-of-day validation is complete.

- [x] Confirmed branch `rlm-memory-bank` at
  `506bd21ffd53531a0cac442093ccce403e8b3891`.
- [x] Confirmed the working tree contains uncommitted Phase 4 module, config,
  tests, and research-note updates.
- [x] Confirmed no diff in protected training paths, `MemGenModel.generate()`,
  runner, interaction managers, or the existing GSM8K configuration.
- [x] Passed `py_compile` for the repaired model/runner, Phase 4 module, smoke
  harness, and unit tests.
- [x] Passed all 16 LatentMemoryBank unit tests.
- [x] Re-read the accepted Phase 3 JSONL artifact: 20 predictions, one summary,
  and `compute_reward=0.60`.
- [x] Re-read the EXP-20260611-007 golden replay artifacts: three predictions,
  one summary, and sample IDs 0, 1, and 2.
- [x] Reconfirmed stored adapter evidence: Weaver 112/112, Trigger 112/112, with
  zero missing, unexpected, shape, or value mismatches.
- [x] Reconfirmed `BUG-0001` and `BUG-0002` remain fixed.
- [x] Reconfirmed the Phase 4 module is standalone and disabled by default.
- [x] Did not run inference, implement integration, or enter Phase 5.

## Files Audited in Phase 1

- `main.py`
- `common/config.py`
- `memgen/runner.py`
- `interactions/base_interaction.py`
- `interactions/singleturn_interaction.py`
- `interactions/multiturn_interaction.py`
- `memgen/model/modeling_memgen.py`
- `memgen/model/modeling_utils.py`
- `memgen/model/weaver.py`
- `memgen/model/trigger.py`
- `memgen/utils.py`
- `data/__init__.py`
- `data/base_builder.py`
- `data/base_env.py`

## Repository Snapshot at Phase 1 Closeout

- Branch: `rlm-memory-bank`
- Commit: `7b8b9a44eb30325a676a6c9576c35b3a10b52c32`
- Working tree had uncommitted changes before Phase 2 note updates: `yes`
- Phase 2 modified research notes only: `yes`

## Key Phase 1 Conclusions

- Inference evaluation enters through `main.py -> MemGenRunner.evaluate()`.
- Static and dynamic evaluations use different interaction managers, but both
  funnel generation through `MemGenModel.generate()`.
- Trigger gating, Weaver latent generation, and latent-to-Reasoner injection all
  happen inside `MemGenModel.generate()` on the inference path.
- The safest future memory reset boundary is the interaction-manager session, not
  a global model lifetime.
- A future memory-bank design should use explicit inference-only state passing,
  not persistent global memory on `MemGenModel`.
- At Phase 1 closeout, baseline trust was blocked by `BUG-0001`; the later
  Repair Phase resolved it before Phase 3.

## Key Phase 2 Conclusions

- Recommended runtime environment for this repository is
  `/home/baishilong/miniconda3/envs/memgen` with Python `3.10.20`.
- The current `base` environment is unsuitable for MemGen because it uses Python
  `3.13.9`, the project's `conda` wrapper is broken by CRLF, and sandboxed runs
  cannot expose CUDA to PyTorch.
- Offline execution requires bypassing the inherited `HF_ENDPOINT=https://hf-mirror.com`
  and local proxy variables, and using the local cached Qwen snapshot path.
- The original MemGen evaluation stack reached real generation on GPU for a
  1-sample GSM8K smoke run.
- The official static-eval route did not complete because
  `StaticEvalRecorder.record_batch()` crashed with `KeyError: 0`; the created
  `evaluate/answer.json` file remained empty.
- A script-only harness that kept the original model and interaction logic but
  bypassed the broken recorder produced a completion and wrote
  `manual_answer.json`.
- At Phase 2 closeout, `BUG-0001` remained open and no Phase 2 output was valid
  as a baseline; the later Repair Phase resolved it before Phase 3.

## Environment Alignment Conclusions

- Canonical Python executable:
  `/home/baishilong/miniconda3/envs/memgen/bin/python`
- Recommended activation:
  `source /home/baishilong/miniconda3/bin/activate memgen`
- Direct absolute Python invocation is preferred for automated runs because the
  current PATH-level `/home/baishilong/bin/conda` wrapper has a CRLF shebang.
- README recommends Python `3.10`; `memgen.yml` specifies Python `3.11.13`.
  The existing environment uses Python `3.10.20` and has already reached real
  GPU generation, so it is the lower-risk repair environment.
- Installed key versions:
  - PyTorch `2.12.0+cu126`
  - Transformers `4.55.4`
  - PEFT `0.17.1`
  - Accelerate `1.10.1`
  - Datasets `4.0.0`
  - FlashAttention `2.8.3`
- The installed PyTorch is newer than both checked-in manifests, but imports,
  dependency validation, CUDA/BF16 checks, and Phase 2 generation succeeded.
  Do not downgrade it before repairing the known code defects.
- Current proxy variables point to `127.0.0.1:7898`. For reproducible offline
  runs, unset proxy and custom HF endpoint variables, set Hugging Face offline
  flags, and use the verified local snapshot/cache paths.
- No project file needs modification for environment alignment.

## Phase History

| Date | Phase | Outcome | Evidence |
|---|---|---|---|
| 2026-06-11 | Phase 0 - Research Memory System and Repository Snapshot | Completed | `research_notes/` structure confirmed; repository and environment snapshot recorded |
| 2026-06-11 | Phase 1 - Code Map and Inference Pipeline Audit | Completed | `research_notes/CODE_MAP.md` updated with verified inference path, boundaries, tensor notes, and integration risks |
| 2026-06-11 | Phase 2 - Original Project Smoke Test | Completed with caveats | Original evaluation path reached generation but failed in static recorder; script-only harness produced one completion; baseline remains invalid due `BUG-0001` |
| 2026-06-11 | Temporary Environment Alignment Phase | Completed | Existing `memgen` environment validated; CUDA/BF16 and local assets confirmed; no install or environment rebuild required |
| 2026-06-11 | Temporary Repair Phase | Completed | `BUG-0001` and `BUG-0002` fixed; one-sample official static smoke wrote a non-empty `answer.json` with exact adapter tensor verification |
| 2026-06-11 | Temporary Repair Review and Sanity Check | Completed | Core repair diff reviewed; training files unchanged; three-sample official static eval produced three predictions plus one summary |
| 2026-06-11 | Phase 3 - Original MemGen Baseline | Completed | Fixed 20-sample baseline accepted at `compute_reward=0.60`; three golden cases replayed with exact token/mask hashes |
| 2026-06-11 | Phase 4 - LatentMemoryBank Module Skeleton | Completed | Standalone disabled-by-default bank added; 16 unit tests passed after cleanup; production inference and training paths remain disconnected |
| 2026-06-11 | Phase 4 cleanup | Completed | Clarified replace fallback and write-step semantics; retrieval-copy isolation covered; 16 unit tests passed |
| 2026-06-11 | End-of-Day Validation | Completed | Compilation and 16 unit tests passed; Repair and baseline artifacts revalidated; Phase 4 remains isolated and ready to commit |

## Historical Phase 4 Session Handoff

This section preserves the Phase 4 closeout state and is not the current
project status.

- Phase 4 is complete.
- The memory bank exists only as a standalone, unit-tested skeleton.
- `latent_memory_bank.enabled` defaults to `false`.
- No production inference or training path imports or calls the module.
- Phase 3 baseline and golden artifacts remain the compatibility oracle.
- Current Phase 4 work is ready to commit; the working tree is not yet clean.
- Next suggested phase is Phase 5: Version A Integration — Reasoner Injection
  Only, after explicit approval.
- Do not enter Phase 5 without explicit approval.

## 2026-06-12 - Phase 8A Core Ablation Pilot

- Status: `completed`
- Overall result: `PASS`
- Scope:
  - Pilot only
  - GSM8K test sample IDs `0..19`
  - `sample_count=20`
  - `seed=42`
  - `batch_size=1`
  - greedy decoding
  - `max_response_length=1024`
  - no latest-k retrieval
  - no random retrieval
  - no Version B
- Compared groups:
  - `G0` disabled anchor: reused `EXP-20260612-013` with frozen baseline
    reference `EXP-20260611-006`
  - `G1` Version A anchor: `EXP-20260612-019`
  - `G4` cosine retrieval without recency decay: `EXP-20260612-020`
  - `G6` append-only update: `EXP-20260612-021`
  - `G7` replace update: `EXP-20260612-022`
- Outcomes:
  - `G0`: `compute_reward=0.60` (`12/20`)
  - `G1`: `compute_reward=0.50` (`10/20`)
  - `G4`: `compute_reward=0.50` (`10/20`)
  - `G6`: `compute_reward=0.50` (`10/20`)
  - `G7`: `compute_reward=0.50` (`10/20`)
- Stability/debug observations:
  - all enabled groups produced non-empty `answer.json`
  - all enabled groups wrote `20` predictions and `1` summary
  - no crash, NaN, OOM, CUDA error, or shape/device/dtype mismatch
  - all enabled groups kept `initial_slots=0` for every session
  - no cross-sample leakage observed
  - retrieved memory remained Reasoner-only in all enabled groups because
    `weaver_input_token_counts` matched
    `reasoner_to_weaver_input_token_counts`
  - stored latent memories remained reasoner-space `[8, 1536]` tensors
  - `slot_count` never exceeded `4`, so `max_slots=8` was not saturated in
    this pilot
- Interpretation:
  - this pilot does not support any performance claim
  - on this 20-sample slice, all enabled variants underperformed the disabled
    anchor by the same observed margin
  - within this pilot, removing current write-age decay or switching among the
    currently implemented update-policy settings did not change
    `compute_reward`
  - update-policy behavior was not effectively separated because
    `max_slots=8` was not saturated and `replace_count=0`
  - current `threshold_topk` has no fallback top-1
  - current decay is write-age decay, not last-retrieved-turn decay
- Next-step recommendation:
  - do not expand GSM8K directly as the primary main experiment
  - plan a trusted TriviaQA disabled baseline before enabled-memory runs
  - after target-task stability, evaluate method-aligned Version A variants
    before considering Version B
- Gate:
  - do not treat Phase 8A as a paper-level result
  - Phase 8B has not started
  - Phase 9 has not started

## 2026-06-12 - Step 2 Structured Retrieval Context

- Status: `completed`
- Added immutable `LatentMemoryRetrievalResult`.
- Added `retrieve_with_context(...)` with:
  - full-bank scores in original slot-index order
  - pre-filter maximum score and argmax index
  - threshold-pass status
  - filtered retrieved indices and scores
  - current memory-write bank step
- Kept `retrieve(...)` as the legacy slot-list API.
- Preserved current write-age scoring, threshold-without-fallback behavior,
  detached retrieval copies, and all existing write/update policies.
- Did not modify `MemGenModel.generate()`.
- Did not implement matched-thread write-back, fallback top-1, or
  last-retrieved decay.
- Step 3 remains gated on explicit approval.

## 2026-06-12 - Step 3 Thread-Aware Write-Back

- Status: `completed`
- Added `update_policy=thread_update`.
- Added `write_back(memory, retrieval_result, metadata=None)`.
- Implemented:
  - empty bank -> insert
  - high current-query score -> replace current argmax slot
  - low score with capacity -> insert new thread
  - low score at capacity -> evict oldest and insert new thread
- Added stale retrieval-step and matched-index validation.
- Added separate debug counts and event traces for thread insertion, matched
  replacement, and capacity eviction.
- Integrated only the `thread_update` policy with
  `retrieve_with_context(...)` and `write_back(...)` in generation.
- Preserved Reasoner-only retrieved-memory injection and unchanged Weaver
  inputs.
- Preserved legacy update policies, no-fallback threshold retrieval,
  write-age decay, and the disabled path.
- Validation:
  - `py_compile` passed for the modified model and test files
  - full unit discovery passed `47/47`
  - `git diff --check` passed
  - disabled golden replay
    `EXP-20260612-023-step3-disabled-replay` matched
    `EXP-20260611-007` on all three response-token hashes, all three
    augmentation-mask hashes, Trigger calls (`193`), Weaver prompt calls (`3`),
    and Weaver inference calls (`8`)
  - disabled sessions created no memory bank and exposed no memory debug state
- Did not enter Version B.
- Step 4 smoke remains gated on explicit approval.

## 2026-06-12 - Step 4 Thread-Update Mechanism Validation

- Status: `completed`
- Experiment: `EXP-20260612-024`
- Output:
  `outputs/latent_bank_vA/EXP-20260612-024-thread-update-smoke/`
- Real enabled inference:
  - one GSM8K test sample completed
  - non-empty answer file, one prediction, one summary
  - no crash, NaN, OOM, CUDA, shape, device, or dtype error
  - `memory_write_count=4`
  - `memory_retrieve_count=3`
  - `thread_insert_count=1`
  - `matched_replace_count=3`
  - `capacity_evict_count=0`
  - observed update reasons: one `empty_bank`, three `matched_thread`
- Boundaries:
  - Weaver input counts matched reasoner-to-Weaver input counts exactly
  - retrieved memory remained Reasoner-only
  - stored latent shape remained `[8, 1536]`
  - session began with `initial_slots=0`
- Controlled branch evidence:
  - four targeted tests passed for empty insert, low-score new-thread insert,
    high-score matched replacement, and full-bank oldest eviction
  - full test discovery passed `47/47`
  - `git diff --check` passed
- Scope:
  - no Step 4 code modification was needed
  - no disabled-path rerun was required because no core or generate logic
    changed during Step 4
  - this is mechanism validation, not a performance result
  - no fallback top-1 or last-retrieved decay
  - Version B has not started
- Recommendation:
  - return to TriviaQA baseline planning before adding further method variants
