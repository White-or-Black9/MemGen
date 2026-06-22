# Experiment Log

Record every experiment, including failed, aborted, and exploratory runs. Never
overwrite prior records; append a new entry.

## Current MAB Evidence Boundary (2026-06-22)

- MAB-5A is the current compressed-memory reference baseline.
- Original full-history detective_qa is `over_capacity_invalid` and was not run.
- MAB-5A official exact match was `0.0` for both modes, while retrieval was
  active and all 10 outputs changed. Zero exact match does not imply an inactive
  mechanism.
- `output_changed` is not an improvement metric.
- Current Version A injects retrieved memory into Reasoner only, not Weaver.
- The next experiment is MAB-5C decoupled retrieve/update thresholds, not a
  shared-threshold-only sweep.
- Earlier dated recommendations remain historical records.

## Historical Post-R2 Mechanism Boundary Note

This snapshot predates the completed R4 full runs and MAB-5A. It is retained to
preserve the interpretation boundary at that time; the current MAB boundary is
the section above.

Phase R2 changed the current Version A-aligned mechanism from historical
write-age decay to last-retrieved decay. This was a code / test / documentation
revision, not a formal target-task experiment. R4 later ran TriviaQA
infrastructure smokes and a retrieval-positive diagnostic, but still no formal
target-task performance experiment has been run after R2.

Interpretation boundary:

- Phase 8A GSM8K pilot results remain historical write-age evidence.
- Phase 8C-alt controlled G0/G1/G2/G3 results remain historical mechanism
  evidence under the older write-age mechanism.
- These pre-R2 runs must not be reinterpreted as last-retrieved-decay
  experiments.
- There is still no formal TriviaQA performance result and no target-task
  performance claim.

## Historical Evidence Classification (through early R4)

Accepted formal result set:

- Phase 0-7 records are the accepted formal project results.
- `EXP-20260611-006` is the accepted fixed 20-sample GSM8K Original MemGen
  comparator.
- `EXP-20260612-013` is the accepted disabled-memory equivalence result against
  the frozen comparator.
- Phase 7 enabled-memory records are bounded stability / debug evidence only;
  they are not performance claims.

Historical / exploratory records:

- Phase 8A GSM8K pilot records are historical and exploratory. They used the
  pre-R2 write-age mechanism and must not be interpreted as current
  last-retrieved-decay evidence.
- Phase 8C-alt controlled records are mechanism / harness evidence only. They
  do not replace TriviaQA and do not establish target-task performance.
- Phase 8D-0 / R4-1A records are infrastructure discovery / preflight only.
  They are not evaluation results.
- R4 Search-R1 / TriviaQA records `EXP-20260618-001` through
  `EXP-20260618-004` are infrastructure smoke / path-coverage / diagnostic
  evidence only. They are not formal target-task performance results.
- Phase R2 / R2-fix define the current mechanism but did not run formal
  target-task experiments.

Current mechanism to use for future experiments:

- Reasoner-only retrieved-memory injection.
- Retrieved memory does not enter Weaver.
- Stored memory is reasoner-space `latent_inputs_embeds`.
- Memory is session-local.
- Enabled memory requires `batch_size=1`.
- Retrieval uses last-retrieved decay with no fallback top-1.

Current next experiment gate:

- R4 infrastructure validation is complete with caveats. Before any larger
  TriviaQA run, decide whether to keep default `threshold=0.7` and search for
  naturally matching samples, or design a threshold calibration / ablation plan.
- Version B remains deferred.

## Experiment Index

| ID | Date | Phase | Question | Status | Key Result |
|---|---|---|---|---|---|
| EXP-20260611-001 | 2026-06-11 | Phase 0 | Can the official GSM8K SFT checkpoint produce a trusted smoke baseline? | `failed` | LoRA keys were not loaded; direct smoke then failed on projection dtype |
| EXP-20260611-002 | 2026-06-11 | Phase 2 | Can the original MemGen inference stack run a one-sample GSM8K smoke test in the recommended environment? | `completed_with_caveats` | Original eval path reached generation but crashed in static recorder; script-only harness produced one completion; result is not a valid baseline because LoRA loading remains broken |
| EXP-20260611-003 | 2026-06-11 | Environment Alignment | Is the existing `memgen` environment suitable for the Repair Phase without package changes? | `completed` | Imports, dependency check, CUDA/BF16, config, model snapshot, checkpoint, and dataset cache validated; no install required |
| EXP-20260611-004 | 2026-06-11 | Temporary Repair | Do the minimal adapter-loader and static-recorder fixes unblock the official one-sample smoke path? | `completed` | Both adapters matched 112/112 tensors exactly; official static eval wrote a non-empty answer file |
| EXP-20260611-005 | 2026-06-11 | Repair Review | Do the repaired loader and recorder remain correct across three sequential batch-size-1 samples? | `completed` | Three predictions plus one summary were written; adapter and augmentation checks passed |
| EXP-20260611-006 | 2026-06-11 | Phase 3 | What is Original MemGen performance on the fixed 20-sample GSM8K comparison subset? | `completed` | 20/20 predictions completed; mean `compute_reward=0.60` |
| EXP-20260611-007 | 2026-06-11 | Phase 3 | Are the fixed golden outputs deterministic under exact replay? | `completed` | Samples 0-2 reproduced identical response-token and augmentation-mask hashes |
| EXP-20260611-008 | 2026-06-11 | Phase 4 | Does the standalone memory-bank skeleton satisfy its tensor, retrieval, capacity, and isolation contracts? | `completed` | 16/16 unit tests passed after cleanup; production inference and training references remained absent |
| EXP-20260611-009 | 2026-06-11 | End-of-Day Validation | Are the Repair fixes, Phase 3 baseline artifacts, and Phase 4 skeleton ready for commit and later continuation? | `completed` | Compilation and 16/16 tests passed; baseline/golden artifacts and adapter evidence remained complete; Phase 4 remained isolated |
| EXP-20260612-010 | 2026-06-12 | Phase 5 | Does `latent_memory_bank.enabled=false` preserve the exact Phase 3 golden behavior after Version A integration? | `completed` | Samples 0-2 matched Phase 3 response-token hashes, augmentation-mask hashes, and Trigger/Weaver call counts exactly |
| EXP-20260612-011 | 2026-06-12 | Phase 5 | Does enabled Version A run on one sample without crashing and produce separate memory write/retrieve bookkeeping? | `completed` | One-sample debug completed with 4 writes, 3 retrievals, 24 retrieved latent tokens, 32 new latent tokens, and 4 resident slots |
| EXP-20260612-013 | 2026-06-12 | Phase 6 | Does the full 20-sample disabled path remain exactly equivalent to the frozen Phase 3 baseline? | `completed` | All 20 response-token hashes, all 20 augmentation-mask hashes, summary metric, and Trigger/Weaver call counts matched `EXP-20260611-006` exactly |
| EXP-20260612-014 | 2026-06-12 | Phase 7 | Does enabled Tier 1 smoke run complete before adding per-session debug trace? | `completed_with_caveats` | One-sample enabled run succeeded, then was superseded by `EXP-20260612-015` to capture session-level initial-slot evidence |
| EXP-20260612-015 | 2026-06-12 | Phase 7 | Does enabled Tier 1 smoke run complete with correct Version A debug and session-local evidence? | `completed` | One-sample enabled run completed with `initial_slots=0`, 4 writes, 3 retrievals, 24 retrieved latent tokens, and Reasoner-only injection evidence |
| EXP-20260612-016 | 2026-06-12 | Phase 7 | Do three enabled single-turn sessions remain isolated and stable on GSM8K samples 0..2? | `completed` | All three sessions started with `initial_slots=0`; no cross-sample leakage, no tensor errors, and slot count stayed within bounds |
| EXP-20260612-017 | 2026-06-12 | Phase 7 | Does enabled Version A remain stable on a bounded five-sample run without exceeding slot limits? | `completed` | Five enabled sessions completed without crash or leakage; slot count never exceeded 4 and no replacement-policy activation was needed |
| EXP-20260612-018 | 2026-06-12 | Phase 7 Supplement | Can the real enabled inference path be forced to trigger replacement by lowering `max_slots` to 2? | `completed` | One enabled sample completed with `memory_write_count=4`, `slot_count=2`, and `update_action_trace=[append, append, replace, replace]` |
| EXP-20260612-019 | 2026-06-12 | Phase 8A G1 | Does the Version A-simple anchor run stably on the fixed GSM8K pilot slice? | `completed` | Stable 20-sample run; `compute_reward=0.50` (`10/20`) |
| EXP-20260612-020 | 2026-06-12 | Phase 8A G4 | What changes when current write-age decay is disabled? | `completed` | Stable 20-sample run; `compute_reward=0.50` (`10/20`); this is not a last-retrieved-decay comparison |
| EXP-20260612-021 | 2026-06-12 | Phase 8A G6 | Does append-only update run stably on the pilot slice? | `completed` | Stable 20-sample run; `compute_reward=0.50` (`10/20`); capacity did not saturate |
| EXP-20260612-022 | 2026-06-12 | Phase 8A G7 | Does the legacy replace policy run stably on the pilot slice? | `completed` | Stable 20-sample run; `compute_reward=0.50` (`10/20`); `replace_count=0` |
| EXP-20260612-023-step3-disabled-replay | 2026-06-12 | Step 3 | Does disabled behavior remain exact after `thread_update` integration? | `completed` | Samples 0-2 exactly matched frozen response-token hashes, augmentation-mask hashes, and Trigger/Weaver call counts |
| EXP-20260612-024-thread-update-smoke | 2026-06-12 | Step 4 | Does Version A-aligned `thread_update` operate correctly on the real enabled inference path? | `completed` | Mechanism smoke only: one empty-bank insert and three current-argmax matched replacements; Reasoner-only and reasoner-space boundaries held |
| EXP-20260612-025 | 2026-06-12 | Phase 8C-alt | Can the first controlled G0 harness revision run on the real checkpoint? | `failed` | Harness left the model on CPU, causing a FlashAttention CPU-backend error; no core defect |
| EXP-20260612-026 | 2026-06-12 | Phase 8C-alt | Does the controlled three-turn disabled path run without visible-history leakage? | `pre_parser_calibration_smoke` | Runtime/leakage smoke only; old strict-only exact match was `0/1` |
| EXP-20260612-027 | 2026-06-12 | Phase 8C-alt | Does Version A-aligned `thread_update` preserve one bank across controlled turns? | `pre_parser_calibration_smoke` | Lifecycle/boundary smoke only; slots `[1,2,3]`; old strict-only exact match was `0/1` |
| EXP-20260613-001 | 2026-06-13 | Phase 8C-alt G3 | Can the checkpoint answer when the early fact is visible in the final oracle prompt and satisfy the tagged-output protocol? | `pre_parser_calibration_smoke` | Correct gold content was generated without tags; this motivated the frozen dual-metric parser contract |
| EXP-20260613-002 | 2026-06-13 | Phase 8C-alt calibrated G0 | What does disabled memory produce under the frozen prompt/parser contract? | `completed` | Unique wrong code `123456`; strict `0/1`, relaxed `0/1`; no bank |
| EXP-20260613-003 | 2026-06-13 | Phase 8C-alt calibrated G2 | Does calibrated G2 preserve lifecycle and recover the hidden fact? | `completed` | Slots `[1,2,3]`, 12 writes, 11 retrievals; unique wrong code `123456`; strict `0/1`, relaxed `0/1` |
| EXP-20260613-004 | 2026-06-13 | Phase 8C-alt calibrated G3 | Does the calibrated oracle-visible control validate deterministic relaxed scoring? | `completed` | Correct untagged code `770487`; strict `0/1`, relaxed `1/1` |
| EXP-20260613-005 | 2026-06-13 | Phase 8C-alt calibrated G1 | Does the calibrated Version A-simple legacy path run correctly as a one-episode mechanism smoke? | `completed` | Legacy `replace_oldest` path ran with slot trace `[4,8,8]`; unique wrong code `123456`; strict `0/1`, relaxed `0/1` |
| EXP-20260618-001 | 2026-06-18 | R4 Search-R1 preflight | Can the local Search-R1 retrieval service serve a MemGen-compatible `/retrieve` schema? | `completed_with_caveats` | Search-R1 served port `8000` with compatible schema after multi-GPU FAISS load using `CUDA_VISIBLE_DEVICES=0,2,3,4,7` |
| EXP-20260618-002 | 2026-06-18 | R4 disabled TriviaQA smoke | Can the R4 dynamic harness complete one disabled-memory TriviaQA sample with live retrieval? | `completed` | One sample valid; retrieval calls `1`, failures `0`, `valid_run=True` |
| EXP-20260618-003 | 2026-06-18 | R4 Version A TriviaQA smoke | Can Version A-aligned memory run on one dynamic TriviaQA sample with live retrieval? | `completed` | Enabled memory wrote 2 slots and performed 1 retrieval turn, but default threshold `0.7` returned `retrieved_latent_count=0` |
| EXP-20260618-004 | 2026-06-18 | R4 retrieval-positive diagnostic | Can non-empty retrieved latent memory be exercised under a controlled low threshold? | `completed_diagnostic_only` | Diagnostic `threshold=0.01` produced `retrieved_latent_count=8` and `replace_matched`; not default behavior or performance evidence |
| EXP-20260618-005 | 2026-06-18 | R4 audit | Does the LatentMemoryBank active retrieval path match the intended last-retrieved-age design? | `completed` | Read-only audit confirmed score formula, exact age semantics, thread eviction, and debug exports all correct; threshold comment has terminology mismatch |
| EXP-20260618-006 | 2026-06-18 | R4 default-threshold natural trigger scan | Does default `threshold=0.7` trigger non-empty retrieval on TriviaQA samples 1..5? | `completed` | 0/5 triggers; max_score 0.02–0.045 |
| EXP-20260618-007 | 2026-06-18 | R4 threshold calibration score scan | What is the decayed retrieval score scale under default threshold on samples 0..19? | `completed` | Mean 0.036, median 0.037, range 0.010–0.054; threshold 0.04 estimated 40% trigger rate |
| EXP-20260618-008 | 2026-06-18 | R4 threshold=0.04 behavior scan | Does threshold=0.04 activate retrieved latent memory on samples 0..19? | `completed` | 8/20 triggered, exactly matched offline estimate; behavior validation only |
| EXP-20260618-009 | 2026-06-18 | R4 held-out comparison s20_39 | Does Version A t=0.04 affect TriviaQA reward on held-out samples 20..39? | `completed` | Disabled 0.60 vs Version A 0.55; one regression (sample 21), no rescue |
| EXP-20260618-010 | 2026-06-18 | R4 sample 21 regression case study | Why did sample 21 regress from 1.0 to 0.0 under Version A t=0.04? | `completed` | Memory-induced regression: query-entity salience amplification of "Gangsta's Paradise" |
| EXP-20260618-011 | 2026-06-18 | R4 triggered held-out audit s20_39 | What effect did memory triggering have on samples 20..39? | `completed` | 0 helpful, 1 harmful, 5 neutral (among 6 triggered) |
| EXP-20260618-012 | 2026-06-18 | R4 rescue/regression scan s40_79 | Does Version A t=0.04 rescue any disabled-wrong answers on fresh held-out samples 40..79? | `completed` | 1 rescue (sample 53 Seymour Hersh), 0 regression, mean diff +0.025 |
| EXP-20260618-013 | 2026-06-18 | R4 combined held-out analysis s20_79 | What is the net effect across 60 held-out TriviaQA samples? | `completed` | Net gain 0 (both 35/60); effect fragile and sample-dependent |
| EXP-20260620-019 | 2026-06-20 | MAB-1A | Can local MAB loading, chunking, templates, and metrics run without an API or model? | `completed_infrastructure_smoke` | Real local `factconsolidation_sh_6k` data path validated; not a benchmark score |
| EXP-20260620-020 | 2026-06-20 | MAB-2 | Does original MemGen complete a one-context full-history Bank-off run? | `completed_valid_one_context` | Harness, scoring, and absence of LatentMemoryBank validated |
| EXP-20260620-021 | 2026-06-20 | MAB-3 | Does Version A complete the paired full-history Bank-on run? | `completed_valid_one_context` | Session lifecycle and Reasoner-only boundary validated |
| EXP-20260620-022 | 2026-06-20 | MAB-3A | Do low shared thresholds activate retrieval? | `completed_valid_diagnostic` | Retrieval activated on one context; not performance evidence |
| EXP-20260620-023 | 2026-06-20 | MAB-4A | Can Bank-on answer from a compressed query prompt? | `completed_exploratory_one_context` | Chunk and acknowledgement history excluded; latent retrieval exercised |
| EXP-20260620-024 | 2026-06-20 | Paired MAB attempt | Can `factconsolidation_sh_6k` support a paired n10 run? | `completed_with_dataset_limitation` | Only one matching local context; not n10 evidence |
| EXP-20260620-025 | 2026-06-20 | MAB data audit | Which local task supports a 10-context compressed pilot? | `completed_read_only_audit` | detective_qa has 10 rows but full history is over capacity |
| EXP-20260620-026 | 2026-06-20 | Over-context diagnostic | Is original full-history behavior valid beyond 32,768 tokens? | `completed_diagnostic` | No explicit guard; real over-capacity prompts must be rejected before generation |
| EXP-20260621-001 | 2026-06-21 | MAB-5A | Does compressed Bank-on improve over compressed Bank-off on detective_qa n10? | `completed` | Both exact match 0.0; retrieval active and all outputs changed |

## Recorded Experiments

### EXP-20260611-001: Official GSM8K SFT Smoke Baseline

- Phase: 0
- Status: `failed`
- Research question: Can the official checkpoint be loaded faithfully and run on
  one deterministic GSM8K test sample?
- Hypothesis: Official assets plus the documented environment are sufficient for
  a trusted local comparator.
- Baseline/comparator: `memgen-gsm8k-sft-official-v1`
- Code revision: `5e59fee296092fa056f140b38a07b927651ffdb5`
- Working tree state: clean before note updates
- Environment: Python 3.10.20, PyTorch 2.12.0+cu126, Transformers 4.55.4,
  PEFT 0.17.1, RTX A6000
- Dataset and split: cached `gsm8k/main`, test sample index 0
- Configuration: prompt augmentation 1, inference augmentation 3, latent lengths
  8/8, inactive Trigger, greedy decoding, maximum 128 new tokens
- Random seed: 42
- Batch size: 1
- Checkpoint: `.cache/baselines/memgen-gsm8k-sft/model`
- Raw artifact: none; run terminated before generation output

#### Observations

- Official file SHA-256 values matched Hugging Face LFS metadata.
- PEFT warned that all expected named `weaver` and `trigger` adapter keys were
  missing while loading.
- Checkpoint tensors use keys without adapter-name suffixes, while the nested
  loaded model expected keys such as `lora_A.weaver.weight`.
- Direct `MemGenModel.generate()` then failed because reasoner embeddings were
  BF16 while projection weights remained FP32. The normal runner converts the
  whole model to BF16, so this dtype failure is a smoke harness issue, not the
  primary baseline blocker.
- No metric, completion, token hash, or latency result is valid.

#### Conclusion

- Hypothesis supported: No.
- Interpretation: The current checkpoint-loading path cannot establish a trusted
  comparator because trained LoRA tensors are silently skipped.
- Follow-up: Repair and test checkpoint loading in a separately approved Phase
  before running the baseline again.
- Related decisions: `DEC-0005`, `DEC-0006`
- Related bug: `BUG-0001`

### EXP-20260611-002: Phase 2 Original Project Smoke Test

- Phase: 2
- Status: `completed_with_caveats`
- Research question: Can the current repository run the original MemGen
  inference path on a minimal GSM8K sample in the recommended local environment?
- Hypothesis: With the correct local environment, local model snapshot, local
  dataset cache, and `batch_size=1`, the original inference stack should at
  least reach generation.
- Baseline/comparator: none; this is a smoke test only
- Code revision: `7b8b9a44eb30325a676a6c9576c35b3a10b52c32`
- Working tree state: research-note changes present; no core-code edits
- Environment: `/home/baishilong/miniconda3/envs/memgen`, Python 3.10.20,
  PyTorch 2.12.0+cu126, Transformers 4.55.4, PEFT 0.17.1, FlashAttention 2.8.3,
  single RTX A6000
- Dataset and split: cached `gsm8k/main`, test sample count `1`
- Inputs/session definition: one GSM8K test item, greedy decoding, static
  single-turn interaction
- Configuration: `configs/latent_memory/gsm8k.yaml`, prompt aug `1`, inference
  aug `3`, latents `8/8`, inactive Trigger, `batch_size=1`,
  `max_response_length=128`
- Random seeds: `42`
- Model path: local Qwen snapshot
  `/home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Checkpoint path: `.cache/baselines/memgen-gsm8k-sft/model`
- Dataset path:
  `/home/baishilong/.cache/huggingface/datasets/gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866`
- Command:
  1. Original-eval smoke harness using `Config -> MemGenModel.from_config -> MemGenRunner.evaluate()`, with `runner.test_dataset = runner.test_dataset.select(range(1))`
  2. Script-only manual harness using the same model/config/interaction path but bypassing the broken static recorder
- Output directory:
  - original eval run:
    `.cache/evaluate/gsm8k/home/pn=1_pl=8_in=3_il=8_20260611-103526_phase2_smoke`
  - manual completion run:
    `.cache/evaluate/gsm8k/home/pn=1_pl=8_in=3_il=8_20260611-104054_phase2_manual_smoke_cuda`
- Start/end time: 2026-06-11 Phase 2 execution window

#### Observations

- Quantitative:
  - original eval path processed one test sample and entered generation
  - original eval output file `evaluate/answer.json` was created but remained
    empty (`0` bytes)
- Qualitative:
  - original runner path failed after generation in `StaticEvalRecorder.record_batch`
  - manual harness produced a completion ending with `\\boxed{18}`
- Runtime/latency:
  - original runner reached the single-step progress bar and failed after about
    8 seconds on the only sample
- Peak memory:
  - not measured in this Phase
- Failures or anomalies:
  - inherited proxy and `HF_ENDPOINT` variables caused offline cache misses until
    they were unset
  - sandboxed execution hid CUDA from PyTorch, so GPU-backed smoke verification
    required unsandboxed execution
  - `BUG-0001` remained reproducible
  - `BUG-0002` was newly confirmed on the original static evaluation path

#### Conclusion

- Hypothesis supported: partially
- Interpretation: The recommended local environment can initialize the original
  MemGen project, load the cached base model and dataset, and reach real
  generation on one sample. However, the official static evaluation path is not
  currently end-to-end runnable because result recording crashes.
- Limitations:
  - not a scientific baseline
  - no trustworthy LoRA-loading guarantee
  - no aggregate metric should be used
- Follow-up:
  - repair `BUG-0001` and `BUG-0002` in a separately approved Phase
  - rerun the same one-sample smoke test before Phase 3
- Related decision IDs: `DEC-0005`, `DEC-0006`, `DEC-0009`
- Related bug IDs: `BUG-0001`, `BUG-0002`
- Artifacts:
  - empty original output:
    `.cache/evaluate/gsm8k/home/pn=1_pl=8_in=3_il=8_20260611-103526_phase2_smoke/evaluate/answer.json`
  - original run log:
    `.cache/evaluate/gsm8k/home/pn=1_pl=8_in=3_il=8_20260611-103526_phase2_smoke/logs/log.txt`
  - manual completion artifact:
    `.cache/evaluate/gsm8k/home/pn=1_pl=8_in=3_il=8_20260611-104054_phase2_manual_smoke_cuda/evaluate/manual_answer.json`

### EXP-20260611-003: Existing Environment Alignment Validation

- Phase: Temporary Environment Alignment Phase
- Status: `completed`
- Research question: Can the existing `memgen` environment be used as the
  controlled runtime for repairing `BUG-0001` and `BUG-0002` without changing
  installed packages?
- Hypothesis: The existing Python 3.10 environment is sufficient because it
  already reached real GPU generation in Phase 2.
- Baseline/comparator: checked-in `requirements.txt` and `memgen.yml`
- Code revision: `dd6eda02c3c06823670e217c8b0217199b24235c`
- Git branch: `rlm-memory-bank`
- Working tree state: clean before environment-alignment note updates
- Environment:
  `/home/baishilong/miniconda3/envs/memgen`, Python `3.10.20`
- Config file: `configs/latent_memory/gsm8k.yaml`
- Model path:
  `/home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Checkpoint path: `.cache/baselines/memgen-gsm8k-sft/model`
- Dataset path:
  `/home/baishilong/.cache/huggingface/datasets/gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866`
- Dataset/sample count: no samples evaluated
- Random seed: not applicable
- Decoding parameters: not applicable
- Output directory: none
- Prediction file: none
- Metric file: none
- Commands:
  - `/home/baishilong/miniconda3/bin/conda env list`
  - `python --version`
  - `/home/baishilong/miniconda3/envs/memgen/bin/python --version`
  - `/home/baishilong/miniconda3/envs/memgen/bin/python -c "import torch; ..."`
  - `/home/baishilong/miniconda3/envs/memgen/bin/python -c "import transformers, peft, accelerate, datasets; ..."`
  - `/home/baishilong/miniconda3/envs/memgen/bin/python -m pip check`
  - OmegaConf load of `configs/latent_memory/gsm8k.yaml`
  - filesystem readability checks for model, checkpoint, and dataset caches
- Latency: not measured
- Memory usage: not measured

#### Observations

- Base environment:
  - Python `3.13.9`
  - active prefix `/home/baishilong/miniconda3`
  - unsuitable for MemGen execution
- Existing `memgen` environment:
  - Python `3.10.20`
  - PyTorch `2.12.0+cu126`
  - Transformers `4.55.4`
  - PEFT `0.17.1`
  - Accelerate `1.10.1`
  - Datasets `4.0.0`
  - FlashAttention `2.8.3`
- `pip check`: no broken requirements
- CUDA outside sandbox:
  - available: `True`
  - device: NVIDIA RTX A6000
  - BF16 supported: `True`
- Sandbox-only CUDA result:
  - unavailable because the execution sandbox hides CUDA/NVML
  - this is not an environment-package failure
- Local assets:
  - Qwen snapshot readable, including single-file `model.safetensors`
  - MemGen projection, Weaver, Trigger, and adapter files readable
  - cached GSM8K loads successfully in offline mode with 7,473 train rows and
    1,319 test rows
  - a sandboxed dataset load was blocked only because Datasets attempted to
    create a lock file under the read-only home cache; the same command
    succeeded outside the sandbox without downloading data
- Environment variables:
- `HTTP_PROXY` and `HTTPS_PROXY` target `127.0.0.1:7898`
- `NO_PROXY` only covers localhost

### EXP-20260612-010: Phase 5 Disabled-Path Golden Replay

- Phase: 5
- Status: `completed`
- Research question: After Version A integration, does
  `latent_memory_bank.enabled=false` preserve the accepted Phase 3 golden
  behavior exactly?
- Hypothesis: The disabled path should remain byte-for-byte identical to
  `EXP-20260611-007` on samples `0..2`.
- Baseline/comparator: `EXP-20260611-007`
- Git branch: `rlm-memory-bank`
- Working tree state: uncommitted Phase 5 inference-only integration, tests,
  validation script, and research-note updates
- Environment:
  `/home/baishilong/miniconda3/envs/memgen`, Python `3.10.20`, PyTorch
  `2.12.0+cu126`, single NVIDIA RTX A6000 via `CUDA_VISIBLE_DEVICES=7`
- Config file: `configs/latent_memory/gsm8k.yaml`
- Optional config override:
  `run.latent_memory_bank.enabled=false`
- Model path:
  `/home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Checkpoint path:
  `/mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model`
- Dataset path:
  `/home/baishilong/.cache/huggingface/datasets/gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866`
- Dataset and split: cached `gsm8k/main`, test samples `0..2`
- Random seed: `42`
- Batch size: `1`
- Decoding: greedy, temperature `0.0`, max response length `1024`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/latent_bank_vA/EXP-20260612-010-disabled-replay --sample-start 0 --sample-count 3 --max-response-length 1024`
- Output directory:
  `outputs/latent_bank_vA/EXP-20260612-010-disabled-replay`
- Prediction file:
  `outputs/latent_bank_vA/EXP-20260612-010-disabled-replay/evaluate/answer.json`
- Verification file:
  `outputs/latent_bank_vA/EXP-20260612-010-disabled-replay/verification.json`

#### Observations

- Adapter verification remained exact: Weaver `112/112`, Trigger `112/112`.
- Response-token SHA-256 hashes matched `EXP-20260611-007` exactly for all
  three records.
- Augmentation-mask SHA-256 hashes matched `EXP-20260611-007` exactly for all
  three records.
- Trigger decision calls matched exactly: `193`.
- Weaver prompt calls matched exactly: `3`.
- Weaver inference calls matched exactly: `8`.
- `memory_bank_debug` remained `null`, confirming no bank was created on the
  disabled path.
- `answer.json` contained three prediction records and one summary record.
- Summary metric on this three-sample subset was `compute_reward=1.0`.
- Total latency was `18.026` seconds; peak allocated CUDA memory was
  `9,391,621,120` bytes.

#### Conclusion

- Hypothesis supported: yes
- Interpretation: The disabled Version A path preserved the accepted golden
  behavior exactly on samples `0..2`.
- Scope note: This is an equivalence check only; it does not replace a future
  broader Phase 6 disabled-path campaign.
- Related decisions: `DEC-0002`, `DEC-0017`, `DEC-0018`

### EXP-20260612-011: Phase 5 Enabled Version A Debug

- Phase: 5
- Status: `completed`
- Research question: Can enabled Version A run on a real GSM8K sample, write and
  retrieve session-local reasoner-space memories, and keep the mechanism within
  the intended scope?
- Hypothesis: One-sample enabled debug should complete without crashing and
  should record separate write/retrieve bookkeeping.
- Baseline/comparator: none; debug only
- Git branch: `rlm-memory-bank`
- Working tree state: uncommitted Phase 5 inference-only integration, tests,
  validation script, and research-note updates
- Environment:
  `/home/baishilong/miniconda3/envs/memgen`, Python `3.10.20`, PyTorch
  `2.12.0+cu126`, single NVIDIA RTX A6000 via `CUDA_VISIBLE_DEVICES=7`
- Config file: `configs/latent_memory/gsm8k.yaml`
- Optional config override:
  `run.latent_memory_bank.enabled=true`
- Model path:
  `/home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Checkpoint path:
  `/mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model`
- Dataset path:
  `/home/baishilong/.cache/huggingface/datasets/gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866`
- Dataset and split: cached `gsm8k/main`, test sample `0`
- Random seed: `42`
- Batch size: `1`
- Decoding: greedy, temperature `0.0`, max response length `1024`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/latent_bank_vA/EXP-20260612-011-enabled-debug --sample-start 0 --sample-count 1 --max-response-length 1024 --memory-enabled`
- Output directory:
  `outputs/latent_bank_vA/EXP-20260612-011-enabled-debug`
- Prediction file:
  `outputs/latent_bank_vA/EXP-20260612-011-enabled-debug/evaluate/answer.json`
- Verification file:
  `outputs/latent_bank_vA/EXP-20260612-011-enabled-debug/verification.json`

#### Observations

- Adapter verification remained exact: Weaver `112/112`, Trigger `112/112`.
- The run completed without crash on one sample.
- `memory_bank_debug` recorded:
  - `memory_write_count=4`
  - `memory_retrieve_count=3`
  - `retrieved_latent_count=24`
  - `new_latent_count=32`
  - `slot_count=4`
- Stored slots remained in Reasoner hidden size `1536` and were stored on CPU
  with original source device recorded as `cuda:0`.
- Weaver prompt and inference call counts were `1` and `3`.
- The trace recorded identical token counts for `reasoner_to_weaver` inputs and
  Weaver inputs on every augmentation call, consistent with retrieved memory not
  being passed into Weaver.
- `answer.json` contained one prediction record and one summary record.
- Summary metric on this one-sample debug run was `compute_reward=1.0`.
- Total latency was `9.255` seconds; peak allocated CUDA memory was
  `9,385,351,168` bytes.

#### Conclusion

- Hypothesis supported: yes
- Interpretation: Enabled Version A mechanism works on a real sample and records
  separate write/retrieve statistics without touching training code.
- Scope note: This is a mechanism debug only. It must not be treated as a
  performance or quality claim relative to the baseline.
- Related decisions: `DEC-0017`, `DEC-0018`

### EXP-20260612-013: Phase 6 Full Disabled-Path Equivalence

- Phase: 6
- Status: `completed`
- Research question: After Phase 5 integration, does the disabled path remain
  exactly equivalent to the frozen 20-sample Phase 3 baseline
  `EXP-20260611-006`?
- Hypothesis: With `latent_memory_bank` disabled, the official evaluation path
  should reproduce every frozen baseline artifact and control-flow statistic on
  GSM8K test IDs `0..19`.
- Baseline/comparator: `EXP-20260611-006`
- Git branch: `rlm-memory-bank`
- Working tree state: no Phase 6 core-code changes; only existing Phase 5 code
  and note updates present
- Environment:
  `/home/baishilong/miniconda3/envs/memgen`, Python `3.10.20`, PyTorch
  `2.12.0+cu126`, single NVIDIA RTX A6000 via `CUDA_VISIBLE_DEVICES=7`
- Config file: `configs/latent_memory/gsm8k.yaml`
- Optional config override:
  `run.latent_memory_bank.enabled=false`
- Model path:
  `/home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Checkpoint path:
  `/mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model`
- Dataset path:
  `/home/baishilong/.cache/huggingface/datasets/gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866`
- Dataset and split: cached `gsm8k/main`, test samples `0..19`
- Random seed: `42`
- Batch size: `1`
- Decoding: greedy, temperature `0.0`, max response length `1024`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/baseline/EXP-20260612-013-phase6-disabled-equivalence --sample-start 0 --sample-count 20 --max-response-length 1024 --reference-verification outputs/baseline/EXP-20260611-006/verification.json`
- Output directory:
  `outputs/baseline/EXP-20260612-013-phase6-disabled-equivalence`
- Prediction file:
  `outputs/baseline/EXP-20260612-013-phase6-disabled-equivalence/evaluate/answer.json`
- Verification file:
  `outputs/baseline/EXP-20260612-013-phase6-disabled-equivalence/verification.json`

#### Equivalence criteria

- `answer.json` exists and is non-empty
- prediction count is `20`
- one summary record exists
- summary `compute_reward` matches the baseline exactly
- every response-token SHA-256 hash matches the baseline record-by-record
- every augmentation-mask SHA-256 hash matches the baseline record-by-record
- Trigger decision call count matches exactly
- Weaver prompt augmentation call count matches exactly
- Weaver inference augmentation call count matches exactly
- adapter verification remains exact and has zero missing, unexpected, shape, or
  value mismatches
- `memory_bank_debug` remains `null`, proving no bank was constructed

#### Observations

- `answer.json` was non-empty and contained 20 prediction records plus one
  summary record.
- Summary `compute_reward=0.60`, matching `EXP-20260611-006`.
- All 20 response-token hashes matched `EXP-20260611-006` exactly.
- All 20 augmentation-mask hashes matched `EXP-20260611-006` exactly.
- Trigger decision calls matched exactly: `1722`.
- Weaver prompt calls matched exactly: `20`.
- Weaver inference calls matched exactly: `43`.
- Weaver adapter verification remained `112/112`.
- Trigger adapter verification remained `112/112`.
- Missing, unexpected, shape-mismatch, and value-mismatch lists remained empty.
- `memory_bank_debug` was `null`.
- Total latency was `96.615` seconds; peak allocated CUDA memory was
  `9,415,716,352` bytes.

#### Conclusion

- Hypothesis supported: yes
- Interpretation: The disabled path remains exactly equivalent to the frozen
  20-sample Phase 3 baseline under the accepted comparator protocol.
- Consequence: Phase 5 integration does not introduce a disabled-path
  regression on the accepted baseline.
- Related decisions: `DEC-0002`, `DEC-0017`, `DEC-0018`, `DEC-0019`
  - no `HF_ENDPOINT` was present in the final alignment shell
- Manifest differences:
  - README specifies Python 3.10
  - `memgen.yml` specifies Python 3.11.13
  - `requirements.txt` specifies PyTorch 2.7.1+cu128
  - `memgen.yml` specifies PyTorch 2.7.1+cu118
  - installed PyTorch is 2.12.0+cu126

#### Conclusion

- Hypothesis supported: yes
- Interpretation: The existing `memgen` environment is suitable for controlled
  Repair Phase work. Rebuilding or changing packages now would add risk without
  evidence of benefit.
- Limitations: The environment manifests are internally inconsistent and do not
  exactly reproduce the installed environment.
- Failures:
  - the PATH-level `/home/baishilong/bin/conda` shim has a CRLF shebang and
    cannot execute normally
- Follow-up:
  - use the direct Python path or activate with the real Miniconda activation
    script
  - preserve package versions through the Repair Phase
- Related decision IDs: `DEC-0009`, `DEC-0010`
- Related bug IDs: `BUG-0003`, `BUG-0004`

### EXP-20260611-004: Repaired Official Static Smoke Test

- Phase: Temporary Repair Phase
- Status: `completed`
- Date: 2026-06-11
- Research question: Do the minimal fixes for `BUG-0001` and `BUG-0002`
  restore a trustworthy one-sample original MemGen smoke path?
- Baseline/comparator: official Qwen2.5-1.5B GSM8K Weaver-SFT checkpoint; smoke
  verification only
- Git branch: `rlm-memory-bank`
- Base commit: `ed741d9be111b3f549740dce6db0f90c4ae11632`
- Working tree: uncommitted Repair Phase changes in the adapter loader, static
  evaluator, smoke harness, and research notes
- Environment:
  `/home/baishilong/miniconda3/envs/memgen/bin/python`
- Package versions: Python 3.10.20, PyTorch 2.12.0+cu126, Transformers 4.55.4,
  PEFT 0.17.1, Accelerate 1.10.1, Datasets 4.0.0
- Config file: `configs/latent_memory/gsm8k.yaml`
- Model path:
  `/home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Checkpoint path:
  `/mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model`
- Dataset path:
  `/home/baishilong/.cache/huggingface/datasets/gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866`
- Dataset/split/sample count: `gsm8k/main`, test index 0, one sample
- Random seed: 42
- Batch size: 1
- Decoding: greedy, temperature 0.0, maximum response length 128, Weaver and
  Trigger sampling disabled
- Output directory: `outputs/baseline/EXP-20260611-004`
- Prediction file:
  `outputs/baseline/EXP-20260611-004/evaluate/answer.json`
- Metric file: prediction file summary record plus
  `outputs/baseline/EXP-20260611-004/verification.json`
- Successful command:

```bash
env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  CUDA_VISIBLE_DEVICES=0 \
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  /home/baishilong/miniconda3/envs/memgen/bin/python \
  -m scripts.eval.repair_phase2_smoke \
  --cfg-path configs/latent_memory/gsm8k.yaml \
  --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model \
  --output-dir /mnt/18T/baishilong/MemGen/outputs/baseline/EXP-20260611-004
```

#### Results

- Weaver adapter: 112 runtime keys and 112 checkpoint keys; zero missing,
  unexpected, shape-mismatched, or value-mismatched tensors.
- Trigger adapter: 112 runtime keys and 112 checkpoint keys; zero missing,
  unexpected, shape-mismatched, or value-mismatched tensors.
- Adapter loading warnings: none related to missing or unexpected keys.
- Prediction: `\boxed{18}` for the selected GSM8K sample.
- Metric: `compute_reward=1.0` for this one sample; no aggregate performance
  conclusion is permitted.
- `answer.json`: 1,006 bytes, two JSONL records, non-empty.
- Generation trace: Trigger decision entry 85 calls, Weaver prompt augmentation
  1 call, Weaver inference augmentation 3 calls.
- Latency: 8.438 seconds for `runner.evaluate()`.
- Peak allocated CUDA memory: 9,391,613,952 bytes.
- Initial failed launch: direct script execution raised
  `ModuleNotFoundError: common` before model loading; module execution fixed the
  harness import path without project changes.

#### Conclusion

- Both Phase 2 smoke blockers are repaired.
- This run establishes readiness to execute Phase 3, not a formal baseline.
- Related decisions: `DEC-0011`, `DEC-0012`
- Related bugs: `BUG-0001`, `BUG-0002`

#### Implementation Summary

- Adapter fix:
  - removed the constructor-created placeholder adapter only during checkpoint
    restoration
  - loaded the saved adapter into the existing PEFT model under the original
    component name
  - avoided creating a second nested PEFT wrapper
- Static recorder fix:
  - preserved the recorder's `List[str]` and `List[Dict]` batch contract
  - flattened only the optional rank nesting introduced by distributed gather
  - did not bypass the official recorder or metric hook
- Scope:
  - no changes to Weaver or Trigger training initialization
  - no changes to trainer classes or training scripts
  - no LatentMemoryBank implementation
  - no dependency or environment changes

### EXP-20260611-005: Repair Review Three-Sample Sanity Check

- Phase: Temporary Repair Review and Sanity Check
- Status: `completed`
- Date: 2026-06-11
- Purpose: Review the Repair Phase diff and verify the repaired official static
  evaluation path across more than one sample.
- Scientific status: sanity check only; not a formal baseline
- Git branch: `rlm-memory-bank`
- Base commit: `ed741d9be111b3f549740dce6db0f90c4ae11632`
- Environment:
  `/home/baishilong/miniconda3/envs/memgen/bin/python`
- Package versions: PyTorch 2.12.0+cu126, Transformers 4.55.4, PEFT 0.17.1,
  Accelerate 1.10.1, Datasets 4.0.0
- Config file: `configs/latent_memory/gsm8k.yaml`
- Model path:
  `/home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Checkpoint path:
  `/mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model`
- Dataset path:
  `/home/baishilong/.cache/huggingface/datasets/gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866`
- Dataset/split/sample IDs: `gsm8k/main`, test indices 0, 1, and 2
- Sample count: 3
- Batch size: 1
- Random seed: 42
- Decoding: greedy, temperature 0.0, maximum response length 128, Weaver and
  Trigger sampling disabled
- Output directory: `outputs/baseline/EXP-20260611-005`
- Prediction file:
  `outputs/baseline/EXP-20260611-005/evaluate/answer.json`
- Verification file:
  `outputs/baseline/EXP-20260611-005/verification.json`
- Command:

```bash
env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  CUDA_VISIBLE_DEVICES=0 \
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  /home/baishilong/miniconda3/envs/memgen/bin/python \
  -m scripts.eval.repair_phase2_smoke \
  --cfg-path configs/latent_memory/gsm8k.yaml \
  --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model \
  --output-dir /mnt/18T/baishilong/MemGen/outputs/baseline/EXP-20260611-005 \
  --sample-count 3
```

#### Diff Review

- Core files reviewed:
  - `memgen/model/modeling_memgen.py`
  - `memgen/runner.py`
- Harness reviewed and parameterized:
  - `scripts/eval/repair_phase2_smoke.py`
- Protected training paths checked with `git diff --name-only`:
  - `memgen/trainer/`
  - `scripts/train/`
  - `scripts/weaver_sft.sh`
  - `scripts/weaver_grpo.sh`
  - `scripts/trigger_train.sh`
  - `memgen/model/modeling_utils.py`
- Protected training path diff result: empty
- Review verdict: the repair is narrowly scoped to checkpoint restoration and
  static evaluation result collation.

#### Results

- `answer.json`: 2,549 bytes and four JSONL records.
- Prediction records: 3.
- Summary records: 1.
- All three prediction records contain non-empty completions.
- One-sample rewards: 1.0, 1.0, and 0.0.
- Summary reward: 0.6666666666666666; not accepted as a baseline metric.
- Weaver adapter: 112/112 exact tensor match.
- Trigger adapter: 112/112 exact tensor match.
- Missing keys: 0.
- Unexpected keys: 0.
- Shape mismatches: 0.
- Value mismatches: 0.
- Adapter-related load warnings: 0.
- Trigger decision calls: 193.
- Weaver prompt augmentation calls: 3.
- Weaver inference augmentation calls: 8.
- Three augmentation masks were captured.
- Evaluation latency: 14.633 seconds.
- Peak allocated CUDA memory: 9,391,613,952 bytes.

#### Caveats

- Transformers warned that `temperature` may be ignored under greedy decoding;
  sampling was disabled, so this does not change the intended deterministic
  protocol.
- Accelerate warned that Linux kernel 5.4 is below its recommended 5.5 minimum;
  the run completed without a hang.
- This experiment does not establish aggregate GSM8K performance.

#### Conclusion

- No Repair Phase regression was found.
- The fixes remain suitable for proceeding to an explicitly approved Phase 3.
- At this Repair Review closeout, the baseline gate remained closed until the
  explicitly approved Phase 3 run.
- Related bugs: `BUG-0001`, `BUG-0002`

### EXP-20260611-006: Original MemGen Fixed-Subset Baseline

- Phase: Phase 3 - Original MemGen Baseline
- Status: `completed`
- Baseline ID: `memgen-gsm8k-sft-official-v1`
- Date: 2026-06-11
- Git branch: `rlm-memory-bank`
- Core code revision: `c0f1f2c3d79828c2d4e4f74eb9756bfb50890653`
- Working tree during run: evaluation-harness changes only
- Environment:
  `/home/baishilong/miniconda3/envs/memgen/bin/python`
- Config: `configs/latent_memory/gsm8k.yaml`
- Model path:
  `/home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Checkpoint:
  `/mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model`
- Dataset cache:
  `/home/baishilong/.cache/huggingface/datasets/gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866`
- Split and sample IDs: GSM8K `main/test`, indices 0 through 19
- Sample count: 20
- Seed: 42
- Batch size: 1
- Decoding: greedy, temperature 0.0, maximum response length 1024, Weaver and
  Trigger sampling disabled
- Output directory: `outputs/baseline/EXP-20260611-006`
- Prediction file:
  `outputs/baseline/EXP-20260611-006/evaluate/answer.json`
- Verification file:
  `outputs/baseline/EXP-20260611-006/verification.json`
- Metric contract:
  `outputs/baseline/EXP-20260611-006/json/metric_contract.json`
- Command:

```bash
env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  CUDA_VISIBLE_DEVICES=0 \
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  /home/baishilong/miniconda3/envs/memgen/bin/python \
  -m scripts.eval.repair_phase2_smoke \
  --cfg-path configs/latent_memory/gsm8k.yaml \
  --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model \
  --output-dir /mnt/18T/baishilong/MemGen/outputs/baseline/EXP-20260611-006 \
  --sample-start 0 --sample-count 20 --max-response-length 1024
```

#### Results

- Prediction records: 20/20, all non-empty.
- Summary records: 1.
- Mean `compute_reward`: 0.60.
- Correct samples: 12; incorrect samples: 8.
- Incorrect sample IDs: 7, 8, 11, 12, 13, 14, 15, 19.
- All 20 completions contained a boxed answer.
- Response length: minimum 53, maximum 235, mean 134.35 tokens.
- No sample reached the 1024-token limit.
- Weaver adapter: exact 112/112 tensor match.
- Trigger adapter: exact 112/112 tensor match.
- Missing/unexpected/shape/value mismatches: all zero.
- Adapter-related loading warnings: zero.
- Trigger decision calls: 1,722.
- Weaver prompt augmentation calls: 20.
- Weaver inference augmentation calls: 43.
- Total evaluation latency: 115.728 seconds.
- Mean evaluation latency: 5.786 seconds/sample.
- Peak allocated CUDA memory: 9,415,716,352 bytes.
- No NaN, OOM, CUDA error, empty completion, or incomplete sample.

#### Artifact Hashes

- `answer.json`:
  `b8e824b4c82c9fc0e6dcfd35b56bd96f26390756ceefef57ee2c35a36e21baea`
- `verification.json`:
  `da94bf8f27fbc67472c30dce35e001bdc054ee7fe59a357bbf1c84e65a6bd333`
- `metric_contract.json`:
  `facf67c5ff4d0742d6640583c41714a4ec767e70c976b767a0ae9e198e7e0026`

#### Interpretation

This is the accepted Original MemGen comparison point for later
LatentMemoryBank experiments on the same fixed subset. It is not an estimate of
full GSM8K test performance.

### EXP-20260611-007: Golden-Case Deterministic Replay

- Phase: Phase 3 - Original MemGen Baseline
- Status: `completed`
- Purpose: Replay fixed test indices 0, 1, and 2 under the exact baseline
  configuration.
- Core code revision: `c0f1f2c3d79828c2d4e4f74eb9756bfb50890653`
- Sample count: 3
- Seed: 42
- Batch size: 1
- Decoding: greedy, maximum response length 1024
- Output directory: `outputs/baseline/EXP-20260611-007`
- Result: all three response-token hashes and all three augmentation-mask hashes
  exactly matched `EXP-20260611-006`.
- Sample 0 response/mask:
  `b263835e26587cffe0d540125dc63a6acf27e924dfa9d5cb45885ce4081218f0` /
  `7dcc914e338423f3616d3d0139ac0df8a959cc0117c4343fb577c26bfd0b1cb4`
- Sample 1 response/mask:
  `560a6a6ffca3241005289a07d36b7c7820b6e13dea354e2beaf5b81e7f67849a` /
  `d042e76299bf72b3847744d4f3b0633de65ede4c6115c40f974188f495fc575b`
- Sample 2 response/mask:
  `dc2bbbddf83b56513d68a590277761b46e097eeaf71fec3429f1715ddd0f20fe` /
  `d6e708979b29292866684d928520c151868f06f891f456161ef56c9daca185e4`
- Conclusion: deterministic golden evidence established for later disabled-path
  equivalence tests.

### EXP-20260611-008: LatentMemoryBank Skeleton Unit Verification

- Phase: Phase 4 - LatentMemoryBank Module Skeleton
- Status: `completed`
- Date: 2026-06-11
- Code revision before Phase 4 changes:
  `506bd21ffd53531a0cac442093ccce403e8b3891`
- Environment:
  `/home/baishilong/miniconda3/envs/memgen/bin/python`
- Python: 3.10.20
- PyTorch: 2.12.0+cu126
- Dataset/model/checkpoint: not applicable; no inference experiment was run
- Sample count/seed/decoding: not applicable
- Output directory and prediction files: none
- Commands:

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python \
  -m py_compile \
  memgen/model/latent_memory_bank.py \
  tests/test_latent_memory_bank.py

/home/baishilong/miniconda3/envs/memgen/bin/python \
  -m unittest discover -s tests -v
```

- Test result after Phase 4 cleanup: 16 passed, 0 failed, 0 errors.
- Covered behavior:
  - disabled no-op and empty retrieval
  - explicit batch-size-1 configuration enforcement
  - detach, clone, and source tensor metadata
  - capacity, append refusal, replace-oldest, and replace-lowest-score
  - top-k, threshold, and recency decay
  - hidden-state and pre-pooled query input
  - reset and recent-token query pooling
  - explicit output dtype/device
  - retrieved tensor and nested-metadata mutation isolation
  - `replace` oldest-slot fallback when all slots are unscored
  - invalid shape and dtype errors
- Isolation checks:
  - no production references to the new module
  - no changes to model generation, runner, trainers, or training scripts
  - importing `MemGenModel` did not load
    `memgen.model.latent_memory_bank`
- Failures/anomalies: none.
- Conclusion: the Phase 4 skeleton is testable and isolated, with no performance
  or inference-integration claim.
- Related decisions: `DEC-0014`, `DEC-0015`, `DEC-0016`

### EXP-20260611-009: End-of-Day Validation

- Phase: End-of-Day Validation; no new roadmap phase
- Status: `completed`
- Date: 2026-06-11
- Purpose: Verify that the Repair fixes, accepted Phase 3 baseline, and
  standalone Phase 4 skeleton are recoverable and ready to commit.
- Code revision: `506bd21ffd53531a0cac442093ccce403e8b3891`
- Branch: `rlm-memory-bank`
- Working tree: dirty with uncommitted Phase 4 module, config, tests, and
  research-note updates
- Environment:
  `/home/baishilong/miniconda3/envs/memgen/bin/python`
- Model/dataset/checkpoint: no model or dataset was loaded; existing artifacts
  were inspected only
- Sample count/seed/decoding: not applicable; no inference run
- Commands:

```bash
/home/baishilong/miniconda3/envs/memgen/bin/python \
  -m py_compile \
  memgen/model/modeling_memgen.py \
  memgen/runner.py \
  memgen/model/latent_memory_bank.py \
  scripts/eval/repair_phase2_smoke.py \
  tests/test_latent_memory_bank.py

/home/baishilong/miniconda3/envs/memgen/bin/python \
  -m unittest discover -s tests -v
```

- Compilation: passed with exit code 0.
- Unit tests: 16 passed, 0 failed, 0 errors.
- Phase 3 artifact verification:
  - `EXP-20260611-006/evaluate/answer.json` is non-empty JSONL
  - 20 prediction records plus one summary record
  - summary `compute_reward=0.60`
  - `EXP-20260611-007` contains three prediction records, one summary, and a
    readable verification artifact for sample IDs 0, 1, and 2
- Repair verification:
  - Weaver adapter 112/112 and Trigger adapter 112/112
  - missing, unexpected, shape-mismatch, and value-mismatch lists are empty
  - the accepted baseline output confirms `StaticEvalRecorder` writes complete,
    non-empty JSONL
- Isolation verification:
  - no diff in protected training paths
  - no `LatentMemoryBank` reference in `MemGenModel.generate()`, runner,
    interaction managers, `main.py`, or `memgen.model` exports
  - `configs/latent_memory_bank/default.yaml` remains `enabled: false`
  - existing `configs/latent_memory/gsm8k.yaml` has no diff
- Failures/anomalies:
  - direct whole-file `json.loads()` is invalid because `answer.json` is JSONL;
    line-by-line parsing succeeded and confirmed the expected record counts
- Conclusion: current Phase 4 changes are ready to commit. Phase 5 has not
  started and still requires explicit approval.

## Experiment Template

### EXP-YYYYMMDD-NNN: <Short Name>

- Phase:
- Status: `planned | running | completed | failed | aborted`
- Research question:
- Hypothesis:
- Baseline/comparator:
- Code revision:
- Working tree state:
- Environment:
- Dataset and split:
- Inputs/session definition:
- Configuration:
- Random seeds:
- Command:
- Output directory:
- Start/end time:

#### Metrics

| Metric | Baseline | Variant | Delta |
|---|---:|---:|---:|
| TBD | TBD | TBD | TBD |

#### Observations

- Quantitative:
- Qualitative:
- Runtime/latency:
- Peak memory:
- Failures or anomalies:

#### Conclusion

- Hypothesis supported:
- Interpretation:
- Limitations:
- Follow-up:
- Related decision IDs:
- Artifacts:

### EXP-20260612-014: Phase 7 Tier 1 Pre-Trace Smoke

- Phase: 7
- Status: `completed_with_caveats`
- Research question: Does enabled Version A complete a bounded one-sample smoke
  run before adding per-session trace capture?
- Baseline/comparator: none; debug only
- Sample IDs: `0`
- Sample count: `1`
- Seed: `42`
- Batch size: `1`
- Output directory:
  `outputs/latent_bank_vA/EXP-20260612-014-phase7-tier1-smoke`
- Key result: the run succeeded with 4 writes, 3 retrievals, 24 retrieved
  latents, 32 new latents, and 4 resident slots, but it did not yet expose
  session-level `initial_slots`, so it was superseded for durable Phase 7
  evidence by `EXP-20260612-015`.

### EXP-20260612-015: Phase 7 Tier 1 Enabled Smoke

- Phase: 7
- Status: `completed`
- Research question: Can enabled Version A complete a one-sample bounded smoke
  run with session-local debug evidence?
- Baseline/comparator: none; debug only
- Sample IDs: `0`
- Sample count: `1`
- Seed: `42`
- Batch size: `1`
- Decoding: greedy, temperature `0.0`, max response length `1024`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/latent_bank_vA/EXP-20260612-015-phase7-tier1-smoke --sample-start 0 --sample-count 1 --max-response-length 1024 --memory-enabled`
- Output directory:
  `outputs/latent_bank_vA/EXP-20260612-015-phase7-tier1-smoke`
- Prediction file:
  `outputs/latent_bank_vA/EXP-20260612-015-phase7-tier1-smoke/evaluate/answer.json`
- Verification file:
  `outputs/latent_bank_vA/EXP-20260612-015-phase7-tier1-smoke/verification.json`

#### Observations

- `answer.json` contained one prediction and one summary record.
- No crash, NaN, OOM, CUDA error, or shape/device/dtype mismatch occurred.
- Session trace recorded `initial_slots=0`.
- Adapter verification remained exact: Weaver `112/112`, Trigger `112/112`.
- Final bank stats:
  - `memory_write_count=4`
  - `memory_retrieve_count=3`
  - `retrieved_latent_count=24`
  - `new_latent_count=32`
  - `slot_count=4`
- Stored latent tensors were reasoner-space `[8, 1536]` tensors.
- Stored slot metadata remained explicit:
  - `storage_device=cpu`
  - `storage_dtype=torch.bfloat16`
  - `original_device=cuda:0`
  - `original_dtype=torch.bfloat16`
- `weaver_input_token_counts` matched `reasoner_to_weaver_input_token_counts`
  exactly, which is consistent with retrieved memory not entering Weaver.
- Total latency: `8.658 s`
- Peak allocated CUDA memory: `9,385,351,168` bytes
- Auxiliary summary metric: `compute_reward=1.0`

#### Conclusion

- Hypothesis supported: yes
- Interpretation: Enabled Version A completed a bounded one-sample run with the
  expected write/retrieve behavior and session-local initialization evidence.
- Scope note: This is a mechanism/stability check only, not a performance
  result.

### EXP-20260612-016: Phase 7 Tier 2 Small Stability

- Phase: 7
- Status: `completed`
- Research question: Do three enabled single-turn sessions remain isolated and
  stable on GSM8K samples `0..2`?
- Baseline/comparator: none; debug only
- Sample IDs: `0..2`
- Sample count: `3`
- Seed: `42`
- Batch size: `1`
- Decoding: greedy, temperature `0.0`, max response length `1024`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/latent_bank_vA/EXP-20260612-016-phase7-tier2-stability --sample-start 0 --sample-count 3 --max-response-length 1024 --memory-enabled`
- Output directory:
  `outputs/latent_bank_vA/EXP-20260612-016-phase7-tier2-stability`
- Prediction file:
  `outputs/latent_bank_vA/EXP-20260612-016-phase7-tier2-stability/evaluate/answer.json`
- Verification file:
  `outputs/latent_bank_vA/EXP-20260612-016-phase7-tier2-stability/verification.json`

#### Observations

- `answer.json` contained three prediction records and one summary record.
- Each recorded session started with `initial_slots=0`.
- Session bank ids differed across all three samples.
- No cross-sample leakage was observed.
- Per-session bank summaries:
  - sample 0: writes `4`, retrieves `3`, retrieved latents `24`, new latents
    `32`, slot count `4`
  - sample 1: writes `2`, retrieves `1`, retrieved latents `8`, new latents
    `16`, slot count `2`
  - sample 2: writes `4`, retrieves `3`, retrieved latents `24`, new latents
    `32`, slot count `4`
- `slot_count` never exceeded `max_slots=8`.
- No crash, NaN, OOM, CUDA error, shape mismatch, device mismatch, or dtype
  mismatch occurred.
- `weaver_input_token_counts` matched `reasoner_to_weaver_input_token_counts`
  exactly.
- Total latency: `14.066 s`
- Mean latency: `4.689 s/sample`
- Peak allocated CUDA memory: `9,385,351,168` bytes
- Auxiliary summary metric: `compute_reward=0.6666666666666666`

#### Conclusion

- Hypothesis supported: yes
- Interpretation: Enabled Version A remained session-local and stable across
  three independent single-turn samples.
- Scope note: This is a bounded stability check only, not a comparative reward
  result.

### EXP-20260612-017: Phase 7 Tier 3 Bounded Capacity

- Phase: 7
- Status: `completed`
- Research question: Does enabled Version A remain stable on a bounded
  five-sample run without exceeding slot limits or showing leakage?
- Baseline/comparator: none; debug only
- Sample IDs: `0..4`
- Sample count: `5`
- Seed: `42`
- Batch size: `1`
- Decoding: greedy, temperature `0.0`, max response length `1024`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/latent_bank_vA/EXP-20260612-017-phase7-tier3-capacity --sample-start 0 --sample-count 5 --max-response-length 1024 --memory-enabled`
- Output directory:
  `outputs/latent_bank_vA/EXP-20260612-017-phase7-tier3-capacity`
- Prediction file:
  `outputs/latent_bank_vA/EXP-20260612-017-phase7-tier3-capacity/evaluate/answer.json`
- Verification file:
  `outputs/latent_bank_vA/EXP-20260612-017-phase7-tier3-capacity/verification.json`

#### Observations

- `answer.json` contained five prediction records and one summary record.
- All five recorded sessions started with `initial_slots=0`.
- No cross-sample leakage was observed.
- Per-session bank summaries:
  - sample 0: writes `4`, retrieves `3`, retrieved latents `24`, new latents
    `32`, slot count `4`
  - sample 1: writes `2`, retrieves `1`, retrieved latents `8`, new latents
    `16`, slot count `2`
  - sample 2: writes `4`, retrieves `3`, retrieved latents `24`, new latents
    `32`, slot count `4`
  - sample 3: writes `2`, retrieves `1`, retrieved latents `8`, new latents
    `16`, slot count `2`
  - sample 4: writes `4`, retrieves `3`, retrieved latents `24`, new latents
    `32`, slot count `4`
- `slot_count` never exceeded `4`; therefore the configured replacement policy
  was not triggered in this bounded run.
- No crash, NaN, OOM, CUDA error, shape mismatch, device mismatch, or dtype
  mismatch occurred.
- `weaver_input_token_counts` matched `reasoner_to_weaver_input_token_counts`
  exactly.
- Total latency: `21.562 s`
- Mean latency: `4.312 s/sample`
- Peak allocated CUDA memory: `9,395,434,496` bytes
- Auxiliary summary metric: `compute_reward=0.8`

#### Conclusion

- Hypothesis supported: yes
- Interpretation: Enabled Version A remained stable in a bounded five-sample
  run and did not expose leakage or capacity overruns.
- Scope note: No method-quality claim follows from this debug result.

### EXP-20260612-018: Phase 7 Capacity-Trigger Supplement

- Phase: 7 supplement
- Status: `completed`
- Research question: Can the real enabled Version A inference path be forced to
  trigger replacement by lowering `max_slots` to `2`?
- Baseline/comparator: none; debug only
- Sample IDs: `0`
- Sample count: `1`
- Seed: `42`
- Batch size: `1`
- Decoding: greedy, temperature `0.0`, max response length `1024`
- Memory overrides:
  - `max_slots=2`
  - `update_policy=replace_oldest`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/latent_bank_vA/EXP-20260612-018-phase7-capacity-trigger --sample-start 0 --sample-count 1 --max-response-length 1024 --memory-enabled --memory-max-slots 2 --memory-update-policy replace_oldest`
- Output directory:
  `outputs/latent_bank_vA/EXP-20260612-018-phase7-capacity-trigger`
- Prediction file:
  `outputs/latent_bank_vA/EXP-20260612-018-phase7-capacity-trigger/evaluate/answer.json`
- Verification file:
  `outputs/latent_bank_vA/EXP-20260612-018-phase7-capacity-trigger/verification.json`

#### Observations

- `answer.json` contained one prediction and one summary record.
- No crash, NaN, OOM, CUDA error, or shape/device/dtype mismatch occurred.
- Session trace recorded `initial_slots=0`.
- Adapter verification remained exact: Weaver `112/112`, Trigger `112/112`.
- Final bank stats:
  - `memory_write_count=4`
  - `memory_retrieve_count=3`
  - `retrieved_latent_count=24`
  - `new_latent_count=32`
  - `slot_count=2`
  - `append_count=2`
  - `replace_count=2`
  - `rejected_write_count=0`
  - `last_update_action=replace`
  - `update_action_trace=["append", "append", "replace", "replace"]`
- This run therefore satisfied both trigger conditions:
  - `memory_write_count > max_slots`
  - `replace_count > 0`
- Stored latent tensors remained reasoner-space `[8, 1536]` tensors.
- Stored slot metadata remained explicit:
  - `storage_device=cpu`
  - `storage_dtype=torch.bfloat16`
  - `original_device=cuda:0`
  - `original_dtype=torch.bfloat16`
- `weaver_input_token_counts` matched `reasoner_to_weaver_input_token_counts`
  exactly, which is consistent with retrieved memory not entering Weaver.
- Total latency: `8.563 s`
- Peak allocated CUDA memory: `9,385,351,168` bytes
- Auxiliary summary metric: `compute_reward=1.0`

#### Conclusion

- Hypothesis supported: yes
- Interpretation: The real enabled Version A inference path can trigger
  replacement cleanly under bounded debug conditions when `max_slots` is
  lowered to `2`.
- Scope note: This supplement verifies capacity/replacement behavior only. It
  is not a performance experiment and makes no baseline-improvement claim.

## Reproducibility Checklist

- [ ] Exact command recorded.
- [ ] Code revision and dirty state recorded.
- [ ] Configuration snapshot preserved.
- [ ] Seeds recorded.
- [ ] Dataset version/split recorded.
- [ ] Raw outputs retained.
- [ ] Metrics can be regenerated.
- [ ] Failures and exclusions documented.

## Phase 8A - Core Ablation Pilot

### Protocol

- Date: 2026-06-12
- Scope: pilot only; not a performance experiment
- Dataset: `gsm8k/main/test`
- Sample IDs: `0..19`
- Sample count: `20`
- Seed: `42`
- Batch size: `1`
- Decoding: greedy
- Max response length: `1024`
- Shared config path: `configs/latent_memory/gsm8k.yaml`
- Shared model path:
  `/home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Shared checkpoint path:
  `/mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model`
- Output root: `outputs/ablations/`

### Group Table

| Group | Experiment | Config difference | compute_reward | Correct / Total | Total latency (s) | Mean latency (s/sample) | Peak CUDA memory (bytes) |
|---|---|---|---:|---:|---:|---:|---:|
| G0 | `EXP-20260612-013` | disabled anchor | 0.60 | 12 / 20 | 96.615 | 4.831 | 9,415,716,352 |
| G1 | `EXP-20260612-019` | enabled, `decay_alpha=0.05`, `update_policy=replace_oldest` | 0.50 | 10 / 20 | 296.500 | 14.825 | 9,420,448,256 |
| G4 | `EXP-20260612-020` | enabled, `decay_alpha=0.0`, `update_policy=replace_oldest` | 0.50 | 10 / 20 | 239.576 | 11.979 | 9,420,448,256 |
| G6 | `EXP-20260612-021` | enabled, `decay_alpha=0.05`, `update_policy=append` | 0.50 | 10 / 20 | 295.256 | 14.763 | 9,420,448,256 |
| G7 | `EXP-20260612-022` | enabled, `decay_alpha=0.05`, `update_policy=replace` | 0.50 | 10 / 20 | 293.830 | 14.691 | 9,420,448,256 |

### G0: Disabled Anchor Reuse

- Status: `reused`, not rerun
- Comparator artifacts:
  - accepted baseline: `EXP-20260611-006`
  - current-harness disabled equivalence: `EXP-20260612-013`
- Rationale: Phase 6 already verified current disabled-path equivalence against
  the frozen Phase 3 baseline, so this pilot reused the validated disabled
  anchor instead of spending another full 20-sample run.
- Output directory:
  `outputs/baseline/EXP-20260612-013-phase6-disabled-equivalence`
- Key results:
  - `answer.json` non-empty
  - prediction count `20`
  - summary count `1`
  - `compute_reward=0.60`
  - correct / total `12 / 20`
  - Trigger decision calls `1722`
  - Weaver prompt calls `20`
  - Weaver inference calls `43`
  - no memory bank constructed
  - `memory_bank_debug=null`

### EXP-20260612-019: G1 Version A Anchor

- Status: `completed`
- Output directory:
  `outputs/ablations/EXP-20260612-019-phase8a-g1-anchor`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/ablations/EXP-20260612-019-phase8a-g1-anchor --sample-start 0 --sample-count 20 --max-response-length 1024 --memory-enabled --memory-max-slots 8 --memory-top-k 1 --memory-threshold 0.7 --memory-decay-alpha 0.05 --memory-update-policy replace_oldest --memory-retrieve-policy threshold_topk`
- Config:
  - `enabled=true`
  - `retrieve_policy=threshold_topk`
  - `top_k=1`
  - `threshold=0.7`
  - `decay_alpha=0.05`
  - `update_policy=replace_oldest`
  - `max_slots=8`
- Results:
  - `answer.json` non-empty
  - prediction count `20`
  - summary count `1`
  - `compute_reward=0.50`
  - correct / total `10 / 20`
  - total latency `296.500 s`
  - mean latency `14.825 s/sample`
  - peak CUDA memory `9,420,448,256` bytes
  - Trigger decision calls `1439`
  - Weaver prompt calls `20`
  - Weaver inference calls `50`
- Aggregated memory debug:
  - `memory_write_count=70`
  - `memory_retrieve_count=50`
  - `retrieved_latent_count=392`
  - `new_latent_count=560`
  - `max observed slot_count=4`
  - `append_count=70`
  - `replace_count=0`
  - `rejected_write_count=0`
  - every session started with `initial_slots=0`
- Boundary checks:
  - no cross-sample leakage observed
  - `weaver_input_token_counts` matched
    `reasoner_to_weaver_input_token_counts`
  - stored latents stayed in reasoner space with hidden size `1536`
  - no crash, NaN, OOM, CUDA error, or shape/device/dtype mismatch

### EXP-20260612-020: G4 Cosine Retrieval Without Recency Decay

- Status: `completed`
- Output directory:
  `outputs/ablations/EXP-20260612-020-phase8a-g4-cosine-no-decay`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/ablations/EXP-20260612-020-phase8a-g4-cosine-no-decay --sample-start 0 --sample-count 20 --max-response-length 1024 --memory-enabled --memory-max-slots 8 --memory-top-k 1 --memory-threshold 0.7 --memory-decay-alpha 0.0 --memory-update-policy replace_oldest --memory-retrieve-policy threshold_topk`
- Config difference from G1:
  - `decay_alpha=0.0`
- Results:
  - `answer.json` non-empty
  - prediction count `20`
  - summary count `1`
  - `compute_reward=0.50`
  - correct / total `10 / 20`
  - total latency `239.576 s`
  - mean latency `11.979 s/sample`
  - peak CUDA memory `9,420,448,256` bytes
  - Trigger decision calls `1434`
  - Weaver prompt calls `20`
  - Weaver inference calls `50`
- Aggregated memory debug:
  - `memory_write_count=70`
  - `memory_retrieve_count=50`
  - `retrieved_latent_count=392`
  - `new_latent_count=560`
  - `max observed slot_count=4`
  - `append_count=70`
  - `replace_count=0`
  - `rejected_write_count=0`
  - every session started with `initial_slots=0`
- Boundary checks:
  - no cross-sample leakage observed
  - retrieved memory remained Reasoner-only
  - stored latents stayed in reasoner space with hidden size `1536`
  - no crash, NaN, OOM, CUDA error, or shape/device/dtype mismatch

### EXP-20260612-021: G6 Append-Only Update

- Status: `completed`
- Output directory:
  `outputs/ablations/EXP-20260612-021-phase8a-g6-append`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/ablations/EXP-20260612-021-phase8a-g6-append --sample-start 0 --sample-count 20 --max-response-length 1024 --memory-enabled --memory-max-slots 8 --memory-top-k 1 --memory-threshold 0.7 --memory-decay-alpha 0.05 --memory-update-policy append --memory-retrieve-policy threshold_topk`
- Config difference from G1:
  - `update_policy=append`
- Results:
  - `answer.json` non-empty
  - prediction count `20`
  - summary count `1`
  - `compute_reward=0.50`
  - correct / total `10 / 20`
  - total latency `295.256 s`
  - mean latency `14.763 s/sample`
  - peak CUDA memory `9,420,448,256` bytes
  - Trigger decision calls `1439`
  - Weaver prompt calls `20`
  - Weaver inference calls `50`
- Aggregated memory debug:
  - `memory_write_count=70`
  - `memory_retrieve_count=50`
  - `retrieved_latent_count=392`
  - `new_latent_count=560`
  - `max observed slot_count=4`
  - `append_count=70`
  - `replace_count=0`
  - `rejected_write_count=0`
  - every session started with `initial_slots=0`
- Boundary checks:
  - no cross-sample leakage observed
  - retrieved memory remained Reasoner-only
  - stored latents stayed in reasoner space with hidden size `1536`
  - no crash, NaN, OOM, CUDA error, or shape/device/dtype mismatch

### EXP-20260612-022: G7 Replace Update

- Status: `completed`
- Output directory:
  `outputs/ablations/EXP-20260612-022-phase8a-g7-replace`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/ablations/EXP-20260612-022-phase8a-g7-replace --sample-start 0 --sample-count 20 --max-response-length 1024 --memory-enabled --memory-max-slots 8 --memory-top-k 1 --memory-threshold 0.7 --memory-decay-alpha 0.05 --memory-update-policy replace --memory-retrieve-policy threshold_topk`
- Config difference from G1:
  - `update_policy=replace`
- Results:
  - `answer.json` non-empty
  - prediction count `20`
  - summary count `1`
  - `compute_reward=0.50`
  - correct / total `10 / 20`
  - total latency `293.830 s`
  - mean latency `14.691 s/sample`
  - peak CUDA memory `9,420,448,256` bytes
  - Trigger decision calls `1439`
  - Weaver prompt calls `20`
  - Weaver inference calls `50`
- Aggregated memory debug:
  - `memory_write_count=70`
  - `memory_retrieve_count=50`
  - `retrieved_latent_count=392`
  - `new_latent_count=560`
  - `max observed slot_count=4`
  - `append_count=70`
  - `replace_count=0`
  - `rejected_write_count=0`
  - every session started with `initial_slots=0`
- Boundary checks:
  - no cross-sample leakage observed
  - retrieved memory remained Reasoner-only
  - stored latents stayed in reasoner space with hidden size `1536`
  - no crash, NaN, OOM, CUDA error, or shape/device/dtype mismatch

### Pilot Interpretation

- `G0` vs `G1`:
  - on this 20-sample pilot, enabled Version A anchor underperformed the
    disabled anchor (`0.50` vs `0.60`)
  - this is a pilot observation only, not a final claim
- `G1` vs `G4`:
  - removing current write-age decay did not change `compute_reward` on this
    slice
  - G4 reduced latency relative to G1 in this pilot
  - this comparison is not last-retrieved-turn decay versus no decay
- `G1` vs `G6`:
  - append-only update matched G1 on `compute_reward`
  - no capacity pressure appeared because no session exceeded `4` slots
- `G1` vs `G7`:
  - score-based `replace` matched G1 on `compute_reward`
  - `replace_count=0` because `max_slots=8` was not reached in this pilot

### Pilot Conclusion

- Phase 8A pilot status: `pass`
- All currently implemented groups ran stably on the 20-sample slice.
- No new blocker was observed.
- Quantitative observation:
  - disabled G0: `compute_reward=0.60`, `12/20`
  - enabled G1/G4/G6/G7: `compute_reward=0.50`, `10/20`
  - every enabled variant underperformed the disabled anchor on this pilot
- Stability observation:
  - all enabled variants completed without runtime or tensor-contract failure
  - no cross-sample leakage or retrieved-memory-to-Weaver leakage was observed
- Update-policy interpretation:
  - no session saturated `max_slots=8`
  - `replace_count=0` in G1, G4, G6, and G7
  - Phase 8A therefore did not produce an effective update-policy comparison
- Retrieval interpretation:
  - current decay is write-age decay measured in successful memory writes
  - current `threshold_topk` has no fallback top-1
  - G1/G4 compare write-age decay against no decay
- Scope:
  - Phase 8A is a short single-turn sanity and negative pilot
  - it is not aligned with the primary multi-turn, long-trajectory, or
    context-truncation hypothesis
  - it is not evidence that the full unimplemented Version B method fails
- Next-step motivation:
  - do not expand GSM8K directly into the primary main experiment
  - establish a dynamic multi-turn TriviaQA baseline
  - evaluate method-aligned Version A variants only after target-task stability
    is established

### EXP-20260612-023-step3-disabled-replay: Step 3 Disabled Replay

- Step: 3
- Status: `completed`
- Purpose: Compatibility verification after integrating
  `update_policy=thread_update`; this is not a performance experiment.
- Dataset: cached `gsm8k/main/test`, sample IDs `0..2`
- Runtime:
  - `sample_count=3`
  - `seed=42`
  - `batch_size=1`
  - greedy decoding
  - `max_response_length=1024`
- Output:
  `outputs/latent_bank_vA/EXP-20260612-023-step3-disabled-replay/`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/latent_bank_vA/EXP-20260612-023-step3-disabled-replay --sample-start 0 --sample-count 3 --max-response-length 1024`
- Frozen comparator: `EXP-20260611-007`
- Results:
  - three predictions plus one summary
  - all response-token hashes matched exactly
  - all augmentation-mask hashes matched exactly
  - Trigger decision calls matched at `193`
  - Weaver prompt calls matched at `3`
  - Weaver inference calls matched at `8`
  - no memory bank was constructed
  - `memory_bank_debug=null`
- Interpretation:
  - Step 3 did not change disabled-path behavior
  - this replay is compatibility evidence only

### EXP-20260612-024-thread-update-smoke: Thread-Update Mechanism Smoke

- Step: 4
- Status: `completed`
- Purpose: Mechanism validation only; this is not a performance experiment.
- Dataset: cached `gsm8k/main/test`, sample ID `0`
- Runtime:
  - `sample_count=1`
  - `seed=42`
  - `batch_size=1`
  - greedy decoding
  - `max_response_length=1024`
- Memory configuration:
  - `enabled=true`
  - `update_policy=thread_update`
  - `retrieve_policy=threshold_topk`
  - `threshold=0.7`
  - `top_k=1`
  - `max_slots=8`
  - `decay_alpha=0.05`
- Output:
  `outputs/latent_bank_vA/EXP-20260612-024-thread-update-smoke/`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=7 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase5_memory_bank_debug --cfg-path configs/latent_memory/gsm8k.yaml --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/latent_bank_vA/EXP-20260612-024-thread-update-smoke --sample-start 0 --sample-count 1 --max-response-length 1024 --memory-enabled --memory-max-slots 8 --memory-top-k 1 --memory-threshold 0.7 --memory-decay-alpha 0.05 --memory-update-policy thread_update --memory-retrieve-policy threshold_topk`
- Artifact checks:
  - non-empty `evaluate/answer.json`
  - prediction count `1`
  - summary count `1`
  - no crash, NaN, OOM, CUDA, shape, device, or dtype error
- Memory results:
  - `memory_write_count=4`
  - `memory_retrieve_count=3`
  - `thread_insert_count=1`
  - `matched_replace_count=3`
  - `capacity_evict_count=0`
  - final `slot_count=1`
  - observed reasons:
    `empty_bank`, `matched_thread`, `matched_thread`, `matched_thread`
  - `new_thread` and `new_thread_bank_full` were not observed in this real
    one-sample smoke
- Controlled mechanism evidence:
  - `empty_bank -> insert`: unit test passed
  - low score, available capacity -> `new_thread`: unit test passed
  - high score -> `replace_matched` / `matched_thread`: unit test passed and
    observed in real inference
  - low score, full bank -> `evict_oldest_insert` /
    `new_thread_bank_full`: unit test passed
- Boundary checks:
  - Weaver input token counts exactly matched reasoner-to-Weaver input token
    counts: `[96, 116, 140, 193]`
  - retrieved memory therefore remained Reasoner-only
  - stored latent shape was `[8, 1536]`, confirming reasoner-space storage
  - session started with `initial_slots=0`
- Interpretation:
  - Version A-aligned `thread_update` mechanism is operational
  - this run does not establish accuracy or performance benefit
  - current retrieval still has no fallback top-1
  - current decay remains write-age decay
  - Version B has not started

### EXP-20260612-025: Controlled G0 Initial Harness Failure

- Phase: 8C-alt
- Status: `failed`
- Purpose: First real one-episode disabled smoke for the controlled harness.
- Output:
  `outputs/controlled_memory/EXP-20260612-025-controlled-g0-disabled/`
- Result:
  - model loading succeeded
  - first Weaver prompt augmentation was reached
  - FlashAttention failed because the harness converted model dtype but did
    not move the model from CPU to CUDA
- Resolution:
  - fixed device placement in the harness only
  - no MemGen core logic changed
- Interpretation: Harness implementation failure, not a model or method result.

### EXP-20260612-026: Controlled G0 Disabled Smoke

- Phase: 8C-alt
- Status: `pre_parser_calibration_smoke`
- Evidence classification: runtime, leakage, and disabled-bank smoke only; not
  a calibrated comparison result.
- Purpose: Controlled multi-turn mechanism smoke, not a performance experiment.
- Output:
  `outputs/controlled_memory/EXP-20260612-026-controlled-g0-disabled/`
- Configuration:
  - group `G0_disabled`
  - one deterministic exact-code episode
  - three independent visible prompts
  - Turn 3 excludes early fact, value, distractor, and previous-turn text
  - `seed=42`, `batch_size=1`, greedy, `max_response_length=64`
  - GSM8K Weaver-SFT checkpoint reused with an explicit distribution-mismatch
    caveat
- Results:
  - three turns completed
  - leakage pass `1/1`
  - valid episodes `1/1`
  - exact match `0/1`
  - `bank_created=false`
  - `memory_bank_debug=null`
  - Trigger calls `135`
  - Weaver prompt calls `3`
  - Weaver inference calls `9`
  - total episode latency `16.343 s`
  - no crash, NaN, OOM, CUDA, shape, dtype, or device error
- Interpretation:
  - validates the controlled disabled protocol and leakage checks
  - does not establish a task-level baseline or performance conclusion

### EXP-20260612-027: Controlled G2 Thread-Update Smoke

- Phase: 8C-alt
- Status: `pre_parser_calibration_smoke`
- Evidence classification: bank lifecycle and Reasoner-only boundary smoke
  only; not a calibrated comparison result.
- Purpose: Verify cross-turn Version A-aligned memory lifecycle and boundaries.
- Output:
  `outputs/controlled_memory/EXP-20260612-027-controlled-g2-thread-update/`
- Configuration:
  - group `G2_vA_thread_update`
  - one deterministic exact-code episode
  - same session-local bank across three independent visible prompts
  - `seed=42`, `batch_size=1`, greedy, `max_response_length=64`
  - write-age decay, no fallback top-1, Reasoner-only injection
- Results:
  - three turns completed
  - leakage pass `1/1`
  - valid episodes `1/1`
  - exact match `0/1`
  - one bank persisted across all turns
  - slots after turns `[1, 2, 3]`
  - `memory_write_count=12`
  - `memory_retrieve_count=11`
  - `retrieved_latent_count=72`
  - `new_latent_count=96`
  - `thread_insert_count=3`
  - `matched_replace_count=9`
  - `capacity_evict_count=0`
  - stored latent hidden sizes were all `1536`
  - Weaver input counts equaled reasoner-to-Weaver input counts
  - Trigger calls `115`
  - Weaver prompt calls `3`
  - Weaver inference calls `9`
  - total episode latency `13.959 s`
  - no crash, NaN, OOM, CUDA, shape, dtype, or device error
- Interpretation:
  - confirms that the bank survives across controlled turns and that
    `thread_update` executes on the real model path
  - no tagged correct answer was produced
  - this one synthetic episode cannot establish benefit or failure
  - the GSM8K checkpoint is out of distribution for this task
  - this remains Version A and is not Version B

### EXP-20260613-001: Controlled G3 Oracle-Visible Smoke

- Phase: 8C-alt G3
- Status: `pre_parser_calibration_smoke`
- Evidence classification: oracle-content and parser-contract diagnostic only;
  not a calibrated comparison result.
- Purpose: Test the visible-context oracle upper bound and the controlled
  prompt/parser protocol, not memory performance.
- Output:
  `outputs/controlled_memory/EXP-20260613-001-controlled-g3-oracle-visible/`
- Configuration:
  - group `G3_oracle_visible`
  - one deterministic exact-code episode
  - Turn 3 visibly included the early fact and gold answer
  - `seed=42`, `batch_size=1`, greedy, `max_response_length=64`
  - memory disabled, `oracle_visible=true`
  - same model and checkpoint as `EXP-20260612-026` and
    `EXP-20260612-027`
- Command:
  `env -u HF_ENDPOINT -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/baishilong/miniconda3/envs/memgen/bin/python -m scripts.eval.phase8c_controlled_memory --model-path /home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --checkpoint-path /mnt/18T/baishilong/MemGen/.cache/baselines/memgen-gsm8k-sft/model --output-dir /mnt/18T/baishilong/MemGen/outputs/controlled_memory/EXP-20260613-001-controlled-g3-oracle-visible --group G3_oracle_visible --sample-count 1 --seed 42 --max-response-length 64 --batch-size 1 --memory-mode disabled`
- Results:
  - all five required artifacts were generated
  - `answer.json` was non-empty
  - valid episodes `1/1`
  - Turn 3 prompt contained the early fact and gold value `770487`
  - Turn 3 prompt length was `90` tokens
  - raw Turn 3 response:
    `The access code for Project Lumen is 770487.`
  - response contained no `<answer>...</answer>` span
  - parser returned `null`
  - strict exact match `0/1`
  - no bank was created
  - Trigger calls `116`
  - Weaver prompt calls `3`
  - Weaver inference calls `7`
  - total episode latency `5.477 s`
  - no crash, non-finite metric, OOM, CUDA, shape, dtype, or device error
- Execution note:
  - the first invocation used the script path directly and stopped before model
    loading with `ModuleNotFoundError: common`
  - rerunning the unchanged harness through
    `python -m scripts.eval.phase8c_controlled_memory` resolved the import-path
    issue
  - no partial artifact was created by the failed invocation
- Interpretation:
  - the checkpoint can extract and state the correct answer when it is visible
  - the strict tagged parser does not recognize the semantically correct raw
    answer because the checkpoint ignored the requested output tags
  - current controlled exact-match results are therefore confounded by
    instruction-format compliance
  - G3 is not memory evidence, is not a fair G0/G2 comparator, and does not
    replace TriviaQA
  - Version B, fallback top-1, and last-retrieved decay remain unimplemented
- Follow-up:
  - audit and pre-register the prompt/parser scoring contract before G1 or a
    larger controlled pilot

### 2026-06-13 Controlled Parser Calibration

- Status: `implemented_without_experiment_run`
- Purpose: Freeze one deterministic scoring contract before any controlled
  group comparison.
- Implementation:
  - strict parser accepts only the last complete `<answer>...</answer>` span
  - relaxed parser first reuses a strict candidate
  - exact-code fallback accepts exactly one standalone six-digit number
  - multiple six-digit candidates are `ambiguous`; zero candidates are `none`
  - semantic fallback evaluates only a normalized complete short response
  - legacy `exact_match` is a deprecated alias for `strict_exact_match`
- Prohibited scoring behavior:
  - no gold answer is passed to the relaxed extractor
  - no gold substring search or gold-guided candidate selection
  - no LLM judge
  - no fuzzy semantic matching
- Artifact changes:
  - episode and Turn 3 records include strict/relaxed parsed answers, parser
    success flags, parser mode, and both exact-match metrics
  - summaries and verification files include strict/relaxed counts and rates
    plus parser-success counts
- Prompt change:
  - all groups use the same exact one-line tagged-output instruction
  - only G3 includes the oracle-visible fact and value
- Validation:
  - no model experiment was run
  - targeted controlled-harness tests passed `22/22`
  - harness and controlled-test `py_compile` passed
  - full unit discovery passed `69/69`
  - `git diff --check` passed
- Next evidence rule:
  - G0/G2/G3 must be rerun under the frozen calibrated prompt and parser before
    their accuracy metrics can be compared
  - G1 and any small pilot remain gated
- Scope:
  - controlled evaluation remains mechanism evidence and does not replace
    TriviaQA
  - no fallback top-1, last-retrieved decay, or Version B was implemented

### EXP-20260613-002: Calibrated G0 Disabled Smoke

- Phase: 8C-alt calibrated G0
- Status: `completed`
- Output:
  `outputs/controlled_memory/EXP-20260613-002-calibrated-g0-disabled/`
- Configuration:
  - group `G0_disabled`, memory mode `disabled`
  - one deterministic exact-code episode
  - `seed=42`, `batch_size=1`, greedy, `max_response_length=64`
  - frozen calibrated prompt and dual strict/relaxed scoring
- Results:
  - valid episodes `1/1`; leakage checks passed
  - Turn 3 excluded the early fact and gold value
  - raw response:
    `The access code for Project Lumen is 123456.`
  - strict parser returned `null`
  - relaxed parser returned `123456` with
    `parser_mode=exact_code_single_candidate`
  - strict exact match `0/1`; relaxed exact match `0/1`
  - no bank was created and `memory_bank_debug=null`
  - Trigger calls `116`; Weaver prompt calls `3`; Weaver inference calls `7`
  - latency `5.668 s`
  - no crash, non-finite metric, OOM, CUDA, shape, dtype, or device error
- Interpretation:
  - disabled memory did not recover the hidden fact in this one-episode smoke
  - parser success does not imply answer correctness

### EXP-20260613-003: Calibrated G2 Thread-Update Smoke

- Phase: 8C-alt calibrated G2
- Status: `completed`
- Output:
  `outputs/controlled_memory/EXP-20260613-003-calibrated-g2-thread-update/`
- Configuration:
  - group `G2_vA_thread_update`, memory mode `vA_thread_update`
  - one deterministic exact-code episode
  - `seed=42`, `batch_size=1`, greedy, `max_response_length=64`
  - frozen calibrated prompt and dual strict/relaxed scoring
  - write-age decay, no fallback top-1, Reasoner-only retrieval
- Results:
  - valid episodes `1/1`; leakage checks passed
  - one bank persisted across all three turns
  - slots after turns `[1, 2, 3]`; final slots `3`
  - `memory_write_count=12`, `memory_retrieve_count=11`
  - `thread_insert_count=3`, `matched_replace_count=9`
  - `capacity_evict_count=0`
  - stored latent hidden sizes `[1536, 1536, 1536]`
  - Weaver input counts exactly matched reasoner-to-Weaver input counts
  - raw response began with the unique wrong code `123456`
  - strict parser returned `null`; relaxed parser returned `123456`
  - strict exact match `0/1`; relaxed exact match `0/1`
  - Trigger calls `115`; Weaver prompt calls `3`; Weaver inference calls `9`
  - latency `5.853 s`
  - no crash, non-finite metric, OOM, CUDA, shape, dtype, or device error
- Interpretation:
  - Version A-aligned lifecycle and boundaries remain operational
  - G2 did not recover the hidden fact under relaxed scoring in this episode
  - this is not evidence for or against unimplemented Version B

### EXP-20260613-004: Calibrated G3 Oracle-Visible Smoke

- Phase: 8C-alt calibrated G3
- Status: `completed`
- Output:
  `outputs/controlled_memory/EXP-20260613-004-calibrated-g3-oracle-visible/`
- Configuration:
  - group `G3_oracle_visible`, memory mode `disabled`
  - one deterministic exact-code episode
  - `seed=42`, `batch_size=1`, greedy, `max_response_length=64`
  - frozen calibrated prompt and dual strict/relaxed scoring
- Results:
  - valid episodes `1/1`
  - Turn 3 included the early fact and gold value `770487`
  - raw response:
    `The access code for Project Lumen is 770487.`
  - strict parser returned `null` because answer tags were absent
  - relaxed parser returned `770487` with
    `parser_mode=exact_code_single_candidate`
  - strict exact match `0/1`; relaxed exact match `1/1`
  - no bank was created
  - Trigger calls `116`; Weaver prompt calls `3`; Weaver inference calls `7`
  - latency `5.589 s`
  - no crash, non-finite metric, OOM, CUDA, shape, dtype, or device error
- Interpretation:
  - the oracle-visible prompt exposes enough information for a correct raw
    answer
  - relaxed exact-code extraction works as pre-registered
  - strict output-format compliance remains poor for this checkpoint
  - G3 is an upper-bound protocol control, not a memory-method result
  - controlled evaluation remains a mechanism study and does not replace
    TriviaQA
  - no fallback top-1, last-retrieved decay, or Version B was introduced

### EXP-20260613-005: Calibrated G1 Version A-Simple Smoke

- Phase: 8C-alt calibrated G1
- Status: `completed`
- Output:
  `outputs/controlled_memory/EXP-20260613-005-calibrated-g1-vA-simple/`
- Configuration:
  - group `G1_vA_simple`, memory mode `vA_simple`
  - one deterministic exact-code episode
  - `seed=42`, `batch_size=1`, greedy, `max_response_length=64`
  - frozen calibrated prompt and dual strict/relaxed scoring
  - write-age decay, no fallback top-1, Reasoner-only retrieval
  - legacy update policy `replace_oldest`
- Results:
  - valid episodes `1/1`
  - Turn 3 excluded the early fact and gold value
  - raw response began with the unique wrong code `123456`
  - strict parser returned `null`
  - relaxed parser returned `123456` with
    `parser_mode=exact_code_single_candidate`
  - strict exact match `0/1`; relaxed exact match `0/1`
- Memory behavior:
  - one bank persisted across all three turns
  - slot trace was `[4, 8, 8]`
  - final slot count was `8`
  - `memory_write_count=12`
  - `memory_retrieve_count=11`
  - `update_action_trace` showed eight `append` actions followed by four
    legacy `replace` actions
  - `thread_update` was not used
  - stored latent hidden sizes were eight `1536`-dimensional tensors
  - Weaver input counts exactly matched reasoner-to-Weaver input counts
  - retrieved memory therefore remained Reasoner-only
- Runtime:
  - Trigger calls `132`
  - Weaver prompt calls `3`
  - Weaver inference calls `9`
  - latency `6.162 s`
  - no crash, non-finite metric, OOM, CUDA, shape, dtype, or device error
- Interpretation:
  - the calibrated harness executes the legacy Version A-simple path correctly
  - this one-episode smoke is a mechanism check only and does not support a
    performance claim
  - comparisons against G0/G2/G3 should remain cautious because all results are
    single synthetic episodes on an out-of-distribution checkpoint
  - controlled evaluation remains a mechanism study and does not replace
    TriviaQA
  - no fallback top-1, last-retrieved decay, or Version B was introduced

### EXP-20260618-001: Search-R1 Retrieval Service Preflight

- Phase: R4 Search-R1 / TriviaQA infrastructure validation
- Status: `completed_with_caveats`
- Research question: Can the local Search-R1 retrieval service serve the
  MemGen-compatible `/retrieve` schema for TriviaQA dynamic smoke tests?
- Nature: infrastructure preflight only; not a MemGen evaluation and not a
  performance result
- Search-R1 repo:
  `/mnt/18T/baishilong/Search-R1`
- Endpoint:
  `http://127.0.0.1:8000/retrieve`
- MemGen harness route:
  - Search-R1 hard-codes Uvicorn port `8000`
  - MemGen harness used the `--retrieval-endpoint` override
  - Search-R1 was not patched to port `8001`
- Assets:
  - E5 model:
    `/mnt/18T/baishilong/retrieval_assets/e5-base-v2`
  - E5 verification: `AutoTokenizer` and `AutoModel` load succeeded; hidden
    size `768`
  - corpus:
    `/mnt/18T/baishilong/retrieval_assets/wiki-18/wiki-18.jsonl`, valid JSONL,
    about `14G`
  - compressed corpus:
    `/mnt/18T/baishilong/retrieval_assets/wiki-18/wiki-18.jsonl.gz`
  - FAISS index:
    `/mnt/18T/baishilong/retrieval_assets/wiki-18/e5_Flat.index`, about `61G`
  - split index files:
    `/mnt/18T/baishilong/retrieval_assets/wiki-18-index/part_aa` and
    `/mnt/18T/baishilong/retrieval_assets/wiki-18-index/part_ab`
  - extraction caveat: the corpus `.gz` was actually a gzip-compressed tar
    payload; correct extraction used `tar -xOzf`, not plain `gzip -dc`
- Bring-up observations:
  - port `8000` was initially occupied by a user-owned temporary
    `python3 -m http.server 8000 --bind 0.0.0.0`; it was killed after
    verification
  - all-visible-GPU FAISS loading failed because GPU `6` was nearly full
  - `CUDA_VISIBLE_DEVICES=7` failed because one A6000 could not hold the
    about-61G index
  - successful launch used `CUDA_VISIBLE_DEVICES=0,2,3,4,7`
- Schema verification:
  - request:
    `{"queries":["Who was Evan Morris?"],"topk":3,"return_scores":true}`
  - HTTP status `200`
  - response top-level keys: `["result"]`
  - `result[0][0].document.contents` existed
  - `score` existed
  - response shape compatible with MemGen:
    `{"result":[[{"document":{"contents":"Title\nBody"},"score":...}]]}`
- Conclusion:
  - Search-R1 / Wikipedia retrieval is usable for R4 smoke tests on port `8000`
  - endpoint override is the least invasive route
  - this does not establish any model performance claim

### EXP-20260618-002: Disabled-Memory TriviaQA Dynamic Smoke

- Phase: R4 disabled-memory dynamic smoke
- Status: `completed`
- Research question: Can the R4 dynamic harness complete one disabled-memory
  TriviaQA sample with live Search-R1 retrieval and structured artifacts?
- Nature: infrastructure smoke only; not a formal TriviaQA result and not a
  performance experiment
- Output:
  `outputs/r4_triviaqa_dynamic_smoke_disabled_1sample/`
- Configuration:
  - config: `configs/latent_memory/triviaqa.yaml`
  - checkpoint:
    `/home/baishilong/.cache/huggingface/hub/models--Kana-s--MemGen/snapshots/269d9b1741130b94fffa410cdaa3d4bc74081a7f/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model`
  - sample index `0`, sample count `1`, batch size `1`
  - memory mode `disabled`
  - retrieval endpoint `http://127.0.0.1:8000/retrieve`
  - retrieval top-k `3`
  - seed `42`, temperature `0.0`, max response length `1024`
  - `--require-retrieval-ok` enabled
- Result:
  - exit code `0`
  - `evaluate/answer.json` written as JSONL-style records
  - retrieval calls `1`
  - retrieval successes `1`
  - retrieval failures `0`
  - `saw_cannot_find_pages=False`
  - `valid_run=True`
  - `invalid_reason=None`
  - `memory_enabled=False`
  - Claude read-only review: `PASS`
- Caveats:
  - duplicate system prompt appears in the conversation artifact
  - `answer.json` must be read line by line rather than with `json.load`
  - do not treat `reward=1.0` from this one-sample smoke as a performance
    result

### EXP-20260618-003: Version A-Aligned TriviaQA Dynamic Smoke

- Phase: R4 Version A-aligned dynamic smoke
- Status: `completed`
- Research question: Can the Version A-aligned memory path run on one dynamic
  TriviaQA sample with live retrieval?
- Nature: enabled-path infrastructure smoke only; not a formal TriviaQA result
  and not a performance experiment
- Output:
  `outputs/r4_triviaqa_dynamic_smoke_version_a_1sample/`
- Configuration:
  - config: `configs/latent_memory/triviaqa.yaml`
  - checkpoint:
    `/home/baishilong/.cache/huggingface/hub/models--Kana-s--MemGen/snapshots/269d9b1741130b94fffa410cdaa3d4bc74081a7f/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model`
  - sample index `0`, sample count `1`, batch size `1`
  - memory mode `version_a_aligned`
  - retrieval endpoint `http://127.0.0.1:8000/retrieve`
  - retrieval top-k `3`
  - seed `42`, temperature `0.0`, max response length `1024`
  - `--require-retrieval-ok` enabled
- Retrieval result:
  - retrieval calls `1`
  - retrieval successes `1`
  - retrieval failures `0`
  - `valid_run=True`
  - Claude read-only review: `PASS`
- Memory result:
  - `memory_enabled=True`
  - `memory_write_count=2`
  - `memory_retrieve_count=1`
  - `retrieved_latent_count=0`
  - `new_latent_count=16`
  - `slot_count=2`
  - default `threshold=0.7`
  - `max_score` about `0.044`
  - `threshold_passed=False`
- Interpretation:
  - Version A enabled path and memory write path were validated on this smoke
  - non-empty retrieved-memory path was not triggered at default threshold on
    sample `0`
  - artifacts did not directly assert Reasoner-only injection; they recorded
    memory-bank behavior consistent with the Version A path
  - do not treat `reward=1.0` from this one-sample smoke as a performance
    result
- Caveat:
  - duplicate system prompt appears in the conversation artifact

### EXP-20260618-004: Retrieval-Positive Version A Diagnostic

- Phase: R4 retrieval-positive diagnostic
- Status: `completed_diagnostic_only`
- Research question: Can non-empty retrieved latent memory be exercised under a
  controlled threshold override?
- Nature:
  - controlled diagnostic only
  - not a formal TriviaQA result
  - not a default Version A setting
  - not a performance experiment
- Output:
  `outputs/r4_triviaqa_dynamic_diagnostic_version_a_threshold001_1sample/`
- Configuration:
  - base config copied to:
    `outputs/r4_triviaqa_dynamic_diagnostic_version_a_threshold001_1sample/triviaqa_threshold001.yaml`
  - original source and original config were not modified
  - copied YAML was provenance only because the R4 harness hard-codes the
    Version A-aligned memory config
  - effective diagnostic override was in-memory:
    `latent_memory_bank.threshold: 0.7 -> 0.01`
  - sample index `0`, sample count `1`, batch size `1`
  - memory mode `version_a_aligned`
  - retrieval endpoint `http://127.0.0.1:8000/retrieve`
  - retrieval top-k `3`
  - seed `42`, temperature `0.0`, max response length `1024`
  - `--require-retrieval-ok` enabled
- Result:
  - exit code `0`
  - `memory_enabled=True`
  - `memory_write_count=2`
  - `memory_retrieve_count=1`
  - `retrieved_latent_count=8`
  - `new_latent_count=16`
  - `slot_count=1`
  - `threshold=0.01`
  - `max_score=0.04365994428860699`
  - `threshold_passed=True`
  - `update_action_trace=["insert", "replace_matched"]`
  - `retrieved_indices=[0]`
  - Claude read-only review: `PASS`
- Interpretation:
  - non-empty retrieved latent memory can be exercised under a controlled
    diagnostic threshold
  - this does not justify changing the default threshold or making performance
    claims
  - threshold `0.01` must not be used as a formal performance setting without a
    separate decision
- Caveats:
  - duplicate system prompt appears in the conversation artifact
  - artifacts show retrieval and memory-bank behavior; they do not separately
    assert Reasoner-only injection

### EXP-20260618-005: LatentMemoryBank Scoring / Recency Semantics Audit

- Phase: R4 mechanism audit
- Status: `completed`
- Research question: Does the active LatentMemoryBank retrieval path match the
  intended last-retrieved-age design?
- Nature: read-only audit; no model runs
- Scope: `memgen/model/latent_memory_bank.py`
- Key findings:
  - score formula: `score = similarity * exp(-decay_alpha * age)`
  - exact age: `age = max(0, retrieval_step - slot.last_retrieved_step)`
  - Δt_i = last-retrieved age (NOT retrieval count, NOT insertion age)
  - `_retrieval_step` is enabled retrieval-turn counter
  - `_step` is write count, used for created_step/stale checks only
  - `access_count` is incremented for returned slots but not used in scoring
  - successful retrieval updates `last_retrieved_step`
  - thread update eviction: largest last-retrieved age, tie-break by
    earliest created_step then smallest index
  - debug exports consistent with semantics
  - tests exist for all key behaviors
- Caveat: config comment calls threshold "cosine similarity threshold" but
  implementation compares against decayed retrieval score (terminology mismatch)

### EXP-20260618-006: Default-Threshold Natural Trigger Scan (samples 1..5)

- Phase: R4 default-threshold diagnostic
- Status: `completed`
- Research question: Does default `threshold=0.7` trigger non-empty retrieval
  on TriviaQA samples 1..5?
- Output: `outputs/r4_triviaqa_default_threshold_scan_version_a_s1_5/`
- Configuration: memory-mode `version_a_aligned`, threshold `0.7`, samples 1..5
- Result: 5/5 valid, retrieval 5/5, natural triggers 0/5
  - max_score values roughly 0.02–0.045 range
- Interpretation: default threshold 0.7 consistently blocks retrieval on
  TriviaQA despite memory writes occurring

### EXP-20260618-007: Threshold Calibration Score Scan (samples 0..19)

- Phase: R4 threshold calibration
- Status: `completed`
- Research question: What is the decayed retrieval score scale for TriviaQA
  samples 0..19 under default threshold?
- Output: `outputs/r4_triviaqa_threshold_calibration_score_scan_s0_20/`
- Summary: `threshold_calibration_summary.json`
- Score distribution:
  - min: 0.0102, max: 0.0539, mean: 0.0356, median: 0.0368
  - p25: 0.0300, p75: 0.0441
- Hypothetical trigger rates:
  - t=0.01: 100%, t=0.02: 90%, t=0.03: 75%, t=0.04: 40%
  - t=0.05: 5%, t=0.10: 0%, t=0.70: 0%
- Interpretation:
  - default 0.7 far above observed range
  - threshold 0.04 selected as first calibrated candidate (moderate 40%
    trigger rate, no reward inspection)

### EXP-20260618-008: Threshold=0.04 Calibrated Behavior Scan (samples 0..19)

- Phase: R4 behavior validation
- Status: `completed`
- Research question: Does threshold=0.04 actually activate Version A
  retrieved-memory injection on samples 0..19?
- Output: `outputs/r4_triviaqa_threshold_calibrated_behavior_t004_s0_20/`
- Summary: `threshold_behavior_summary.json`
- Configuration: in-memory threshold override 0.04, no source/config changed
- Result: 20/20 valid, 8/20 triggered (exactly matched offline estimate)
  - total retrieved_latent: 64
  - replace_matched: 8, insert-only: 12
  - slot_count: {1: 8, 2: 12}
- Interpretation: behavior validation only, not performance evidence

### EXP-20260618-009: Held-Out Exploratory Comparison (samples 20..39)

- Phase: R4 held-out exploratory
- Status: `completed`
- Research question: Does Version A t=0.04 differ from disabled on held-out
  TriviaQA samples 20..39?
- Output: `outputs/r4_triviaqa_heldout_s20_39_*`
- Summary: `outputs/r4_triviaqa_heldout_s20_39_comparison_summary.json`
- Calibration: samples 0..19; held-out: 20..39; threshold fixed at 0.04
- Result: 20/20 valid both runs
  - disabled `compute_reward`: 0.60 (12/20)
  - Version A t=0.04: 0.55 (11/20)
  - only change: sample 21 (1.0→0.0)
  - 6/20 memory-triggered, total retrieved: 88
- Interpretation: one regression, no rescue; exploratory only

### EXP-20260618-010: Sample 21 Regression Case Study

- Phase: R4 case study
- Status: `completed`
- Research question: Why did sample 21 regress under Version A t=0.04?
- Question: "What Michelle Pfeiffer movie got a boost from the Coolio song
  Gangsta's Paradise?"
- Disabled: "Dangerous Minds" (reward 1.0)
- Version A t=0.04: "Gangsta's Paradise" (reward 0.0)
- External retrieval identical; docs clearly contained correct answer
- Version A memory: writes=2, retrieved=8, max_score=0.0534, replace_matched
- Likely cause: memory-induced regression
  - retrieved latent amplified salient query/song entity instead of
    evidence-grounded movie answer
- Memory timing hypothesis:
  - first insert: before Search-R1 evidence
  - later replace_matched: after external evidence
  - retrieved latent from pre-evidence query context injected into
    post-evidence answer generation

### EXP-20260618-011: Triggered Held-Out Audit (samples 20..39)

- Phase: R4 triggered audit
- Status: `completed`
- Research question: What effect did memory triggering have on reward outcomes
  for samples 20..39?
- Triggered samples: 20, 21, 34, 36, 37, 39
- Summary: helpful=0, harmful=1 (21), neutral=3, neutral/unclear=2
- Mechanism finding: pre-evidence latent seeded from query context,
  retrieved during post-evidence answer generation, amplifying query entities

### EXP-20260618-012: Fresh Held-Out Rescue/Regression Scan (samples 40..79)

- Phase: R4 rescue/regression scan
- Status: `completed`
- Research question: Does Version A t=0.04 rescue any disabled-wrong answers
  on fresh held-out samples 40..79?
- Output: `outputs/r4_triviaqa_rescue_scan_s40_79_*`
- Summary: `outputs/r4_triviaqa_rescue_scan_s40_79_summary.json`
- Fresh samples 40..79; threshold fixed at 0.04; no threshold tuning
- Result: 40/40 valid both runs
  - disabled: 0.575 (23/40)
  - Version A: 0.600 (24/40), diff +0.025
  - rescue: 1 (sample 53 "Seymour Hersh"), regression: 0
  - memory-triggered: 12/40, retrieved: 120
- Notable rescue sample 53:
  - journalist who told of My Lai massacre
  - disabled: "Normand Poirier" → Version A: "Seymour Hersh"
  - max_score: 0.0441, replace_matched
- Interpretation: Version A can rescue; not only harmful

### EXP-20260618-013: Combined Held-Out Interpretation (samples 20..79)

- Phase: R4 combined analysis
- Status: `completed`
- Research question: What is the net effect across all 60 held-out samples?
- Samples 20..39: disabled 12/20, Version A 11/20, rescue=0, regression=1
- Samples 40..79: disabled 23/40, Version A 24/40, rescue=1, regression=0
- Combined 20..79: disabled 35/60, Version A 35/60
  - net gain: 0
  - rescue: 1, regression: 1
- Interpretation:
  - Version A t=0.04 can both rescue and regress; mixed behavior
  - no net improvement across 60 held-out samples
  - effect fragile and sample-dependent
  - do NOT claim improvement or failure; evidence shows neutral with isolated
    effects in both directions

### EXP-20260619-014: Expanded R4 TriviaQA Paired Evaluation (samples 80..179)

- Phase: R4 exploratory paired evaluation
- Status: `completed`
- Research question: Does Version A t=0.04 improve over disabled on a
  larger held-out TriviaQA slice 80..179?
- Output: `outputs/r4_triviaqa_paired_s80_179_*`
- Configuration:
  - checkpoint: `Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model`
  - dataset: TriviaQA validation / `rc.wikipedia.nocontext`
  - retrieval: local Search-R1 endpoint
  - threshold: 0.04
  - top_k: 1
  - batch_size: 1
- Result: 100/100 valid both runs
  - disabled: 47/100
  - Version A t=0.04: 47/100
  - rescue: 1 (sample 83)
  - regression: 1 (sample 82)
  - stable correct: 46
  - stable wrong: 52
  - threshold-passed: 37/100
  - net gain: 0
- Interpretation:
  - exploratory R4 evidence only; not formal target-task benchmark
  - Version A shows sparse steering but no net gain on the larger held-out slice
  - result strengthens the case for a suppress-pre-evidence-write ablation

### EXP-20260619-015: Disabled TriviaQA Full Validation Aggregate

- Phase: R4 disabled full baseline
- Status: `completed`
- Research question: Can the disabled-memory TriviaQA harness complete the
  full validation split end-to-end with the corrected retry chunks?
- Output: `outputs/r4_triviaqa_full_chunks/disabled_s*`
- Configuration:
  - checkpoint: `Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model`
  - dataset: TriviaQA validation / `rc.wikipedia.nocontext`
  - retrieval: local Search-R1 endpoint
  - batch_size: 1
  - temperature: 0.0
  - seed: 42
- Result: 7993/7993 samples covered with no missing or duplicate sample IDs
  - disabled correct: 5148/7993
  - disabled accuracy: 0.6440635556
  - the original stuck `disabled_s7000_7992` chunk was preserved in place and
    excluded from the final aggregate; the tail was re-run via smaller retry
    chunks
  - retry chunks:
    - 7000..7499: 500/500 valid, 0 retrieval-blocked
    - 7500..7799: 295/300 valid, 5 retrieval-blocked
    - 7800..7992: 193/193 valid, 0 retrieval-blocked
- Interpretation:
  - this is an operational/full-coverage disabled baseline, not a Version A
    comparison and not a formal claim about the enabled mechanism
  - the full disabled path now has complete artifacts for the validation split
  - aggregate denominator is all `7993` samples
  - full validity accounting: valid `7970`, invalid/retrieval-blocked `23`

### EXP-20260620-016: Version A Full TriviaQA Validation Rerun

- Phase: R4 full target-task evaluation
- Status: `completed_negative_result`
- Research question: Does the current Version A session-local latent memory
  bank improve over disabled MemGen on the complete TriviaQA validation split?
- Output:
  `outputs/r4_triviaqa_full_version_a_t004_chunks_250_fullrerun/`
- Execution:
  - 32 chunks, 250 samples each except final chunk `7750..7992` with 243
  - all chunks completed with `run_config.json`, `evaluate/answer.json`,
    `summary.json`, and `memory_trace.json`
- Configuration:
  - memory mode: `version_a_aligned`
  - threshold: `0.04`
  - top_k: `1`
  - batch_size: `1`
  - seed: `42`
  - temperature: `0.0`
  - max_response_length: `1024`
  - retrieval_topk: `3`
  - dataset: TriviaQA validation / `rc.wikipedia.nocontext`
  - checkpoint: TriviaQA Weaver-SFT
- Denominator rule: all `7993` samples, including invalid/retrieval-blocked
  samples
- Result:
  - correct: `5092/7993`
  - accuracy: `0.6370574252`
  - valid: `7970`
  - invalid/retrieval-blocked: `23`
  - missing: `0`
  - duplicates: `0`
- Interpretation: complete enabled-path result; the paired comparison is
  recorded separately in EXP-20260620-017.

### EXP-20260620-017: Disabled vs Version A Full Paired Comparison

- Phase: R4 full paired comparison
- Status: `completed_negative_result`
- Inputs:
  - disabled: `outputs/r4_triviaqa_full_chunks/`
  - Version A:
    `outputs/r4_triviaqa_full_version_a_t004_chunks_250_fullrerun/`
- Analysis output:
  `outputs/r4_triviaqa_full_version_a_t004_analysis/`
- Alignment:
  - `7993` paired sample IDs, range `0..7992`
  - missing `0`, duplicates `0`
  - question mismatches `0`, gold-answer mismatches `0`
- Result:

| Mode | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Disabled | 5148 | 7993 | 0.6440635556 |
| Version A | 5092 | 7993 | 0.6370574252 |

- Accuracy delta: `-0.0070061304` (`-0.7006` percentage points)
- Net correct change: `-56`
- Paired transitions:

| Transition | Count |
|---|---:|
| Rescue | 53 |
| Regression | 109 |
| Stable correct | 5039 |
| Stable wrong | 2792 |

- Memory summary:
  - mean writes: `2.102965`
  - mean retrieve attempts: `1.102965`
  - mean retrieved latent count: `2.973602`
  - median retrieved latent count: `0`
  - samples with retrieve attempts: `7971`
  - samples receiving latent injection / threshold passed: `2417`
  - mean per-sample max score: `0.034162`
  - maximum score: `0.082211`
  - action occurrences: insert `13838`, replace_matched `2971`
- Interpretation:
  - Version A is worse by 56 correct answers
  - the mechanism is active and produces real rescues, but regressions are
    approximately `2.06x` as frequent
  - classify as a negative full TriviaQA result, not an inert mechanism

### EXP-20260620-018: Version A Full Post-Hoc Failure Analysis

- Phase: R4 artifact-only failure analysis
- Status: `completed`
- Inputs:
  - `outputs/r4_triviaqa_full_version_a_t004_analysis/paired_per_sample.jsonl`
  - Version A full `memory_trace.json` and saved conversations
- Outputs:
  - `outputs/r4_triviaqa_full_version_a_t004_analysis/failure_analysis.json`
  - `outputs/r4_triviaqa_full_version_a_t004_analysis/failure_analysis.md`
- Score-bucket net gains:
  - no score: `0`
  - `<0.04`: `0`
  - `0.04..0.045`: `+2`
  - `0.045..0.05`: `-12`
  - `0.05..0.055`: `-27`
  - `0.055..0.06`: `-14`
  - `>=0.06`: `-5`
- Repeated-injection result:
  - retrieved latents `8`: rescue `45`, regression `59`, net `-14`
  - retrieved latents `16`: rescue `7`, regression `9`, net `-2`
  - retrieved latents `24`: rescue `1`, regression `3`, net `-2`
  - retrieved latents `32+`: rescue `0`, regression `38`, net `-38`
  - retrieve count `4+`: rescue `2`, regression `44`, net `-42`
- Regression taxonomy:
  - verbose_malformed `42`
  - retrieval_confusion `26`
  - answer_to_question_term `20`
  - over_specific_or_under_specific `11`
  - entity_substitution `9`
  - unknown_other `1`
- Rescue taxonomy:
  - evidence_entity_fix `30`
  - answer_specificity_fix `11`
  - incomplete_to_answer `7`
  - unknown_other `4`
  - normalization_fix `1`
- Interpretation:
  - repeated latent injection is the strongest observed failure signal
  - `max_score` is not calibrated as answer correctness/confidence
  - simple threshold increases to `0.05`, `0.055`, or `0.06` are not
    supported by the score-bucket results
  - current Version A is mechanism-active but policy-unstable
- Follow-up status: paused by user; no ablation started.

### EXP-20260620-019: MAB-1A No-API Real-Data Smoke

- Status: `completed_infrastructure_smoke`
- Scope: local `factconsolidation_sh_6k`, one context, first query, no model or
  external API.
- Artifact: `outputs/mab/no_api_smoke/20260620T015554Z-455306d-fact-sh-6k-real-local/`
- Result: local parquet loading, official chunking, templates, and metric path
  validated. This is infrastructure evidence, not a benchmark score.
- Evidence note: `benchmarks/memoryagentbench_no_api_smoke.md`.

### EXP-20260620-020: MAB-2 Full-History Bank-off

- Status: `completed_valid_one_context`
- Run ID: `20260620T034034Z-factconsolidation-sh-6k-onectx`
- Result: original MemGen full-history rebuild, official scoring, and absence of
  the added LatentMemoryBank validated on one context.
- Evidence note: `benchmarks/memoryagentbench_mab2_bank_off_run.md`.

### EXP-20260620-021: MAB-3 Full-History Bank-on

- Status: `completed_valid_one_context`
- Run ID: `20260620T085407Z-factconsolidation-sh-6k-onectx`
- Result: session-local bank lifecycle and Reasoner-only injection boundary
  validated; default threshold produced no retrieved latent injection.
- Evidence note: `benchmarks/memoryagentbench_mab3_bank_on_full_history_run.md`.

### EXP-20260620-022: MAB-3A Shared-Threshold Ablation

- Status: `completed_valid_diagnostic`
- Artifact: `outputs/mab/memgen_bank_on_threshold_ablation/20260620T103852Z-factconsolidation-sh-6k-onectx/`
- Result: low shared thresholds activated retrieval on the one-context
  full-history case. This is mechanism evidence, not performance evidence.
- Evidence note: `benchmarks/memoryagentbench_mab3a_threshold_ablation.md`.

### EXP-20260620-023: MAB-4A Compressed-Memory Exploratory Run

- Status: `completed_exploratory_one_context`
- Artifact: `outputs/mab/memgen_bank_on_compressed_memory/20260620T111903Z-factconsolidation-sh-6k-onectx/`
- Result: query chunk and acknowledgement history were excluded while latent
  retrieval remained available.
- Evidence note: `benchmarks/memoryagentbench_mab4a_compressed_memory.md`.

### EXP-20260620-024: Paired Low-Threshold n10 Attempt

- Status: `completed_with_dataset_limitation`
- Artifact: `outputs/mab/paired_bank_off_vs_low_threshold_bank_on/20260620T114425Z-factconsolidation-sh-6k-n10/`
- Result: the local source contained only one matching context, so this is a
  one-context paired case and not n10 evidence.
- Evidence note:
  `benchmarks/memoryagentbench_paired_bank_off_vs_low_threshold_bank_on_n10.md`.

### EXP-20260620-025: Local MAB Task Availability Audit

- Status: `completed_read_only_audit`
- Result: detective_qa provided 10 local rows suitable for a compressed-memory
  pilot, but full-history prompts were over the 32,768-token capacity.
- Evidence note: `benchmarks/memoryagentbench_local_task_availability.md`.

### EXP-20260620-026: MemGen Over-Context Diagnostic

- Status: `completed_diagnostic`
- Artifact: `outputs/mab/memgen_over_context_behavior/20260620T133105Z-over-context/over_context_diagnostic.json`
- Result: original full-history inference has no explicit over-context guard;
  detective_qa preflight exceeded capacity and generation was not called.
- Evidence note: `benchmarks/memgen_over_context_behavior.md`.

### EXP-20260621-001: MAB-5A DetectiveQA Compressed-Memory n10

- Phase: MAB-5A compressed-memory benchmark preservation
- Status: `completed`
- Research question: Does LatentBank help on `detective_qa` when the original full-history prompt is over capacity?
- Output: `outputs/mab/compressed_memory_detectiveqa_n10/20260621T013454Z-detectiveqa-compressed-n10/`
- Configuration:
  - split: `Long_Range_Understanding`
  - subtask: `detective_qa`
  - query mode: `first-query-only`
  - threshold: `0.03`
  - top_k: `1`
  - max_slots: `8`
  - batch_size: `1`
- Result:
  - valid contexts: `10/10`
  - Bank-off accuracy: `0.0`
  - Bank-on accuracy: `0.0`
  - delta: `0.0`
  - output changed: `10/10`
  - retrieval active in all contexts
  - no cross-context leakage
  - query writes: `0`
- Mechanism note:
  - retrieved scores were roughly `0.030-0.064`
  - final slot counts stayed low, consistent with over-merge / over-compression
  - current `thread_update` compares `candidate_inputs_embeds` with existing `slot.key`
    before Weaver emits the new latent, so one threshold currently couples
    retrieval visibility and write/update behavior
- Interpretation:
  - mechanism is active but produced no official exact-match gain
  - `output_changed=10` is activation evidence, not improvement
  - next experiment is MAB-5C decoupled retrieve/update thresholds, not another
    shared-threshold-only ablation
- Detailed evidence:
  `benchmarks/memoryagentbench_mab5a_detectiveqa_compressed_n10.md`.
