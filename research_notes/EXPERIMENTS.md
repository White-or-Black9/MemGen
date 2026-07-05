# Experiment Log

## Current Paper-Facing Experiment Summary (2026-07-05)

This section is the current paper-facing result index. Detailed historical
experiment records below are preserved and should not be interpreted as
expanding the current claim beyond `research_notes/PAPER_SCOPE.md`.

### EventQA-65536: Main Positive Evidence

- Protocol: local MemoryAgentBench EventQA-65536 `frozen_context_bank`, five
  contexts, 500 questions per repeat, unchanged non-strict prompt/parser/scorer.
- Frozen P7 Bank-on EM: `0.197+-0.020`.
- Bank-off EM: `0.008`.
- Frozen P7 Bank-on recall: `0.254+-0.028`.
- Bank-off recall: `0.178`.
- P7 versus P6:
  - EM: `+0.0280`;
  - recall: `-0.0044`;
  - format failures: `-44.4` per 500-question run.
- Context 4 limitation:
  - P7 EM mean: `0.006`;
  - recall mean: `0.228`;
  - format failures: `93.8/100`.
- Interpretation: P7 supports the scoped claim of improved long-context event
  reasoning on EventQA, but gains are not uniform across contexts.

### LoCoMo-QA: Diagnostic / Limitation Evidence

- Scope: session-level paired pilot, two conversations, 304 QA rows per mode.
- Disabled EM: `0`.
- P7 EM: `0`.
- Disabled token F1: `0.01834`.
- P7 token F1: `0.02084`.
- All `304/304` paired rows are exact-match wrong.
- P7 no-context denial: `138/304`.
- P7 refusal: `153/304`.
- Query retrieval is active on every P7 row.
- `retrieved_latent_count=16` on every P7 row.
- `query_write_count=0` on every P7 row.
- Interpretation: the protocol is mechanically correct, but there is no
  positive multi-turn QA improvement. LoCoMo is limitation evidence about
  latent-only exact conversational fact recovery.

### Remaining Paper-Facing Runs

- method-separable EventQA Bank-off/P7 cost;
- text-summary memory baseline;
- BM25 top-2 retrieved-text/RAG baseline;
- 16-token matched-budget baseline;
- P7 no-query-retrieval ablation;
- final unified table package.

## Current Paper-Method Boundary (2026-07-04)

- The current paper-facing main method is frozen to P7 session-local latent
  memory bank on the EventQA / MAB-6B track.
- Fixed P7 parameters:
  `retrieve_threshold=0.05`, `update_threshold=0.10`, `max_slots=16`,
  `top_k=2`, `decay_alpha=0.05`.
- Treat future formal comparisons and paper-preparation summaries as centered
  on P7.
- Utility gate, tuple suppression, top-1 fallback, score-margin gate, learned
  utility prediction, and non-oracle harmful-memory detection are not current
  method implementations.
- EventQA harmful attribution artifacts are mechanism-analysis evidence only.
  They document a limitation of P7 and a future-work direction; they do not
  constitute a method improvement experiment.
- Current evidence supports P7 as the best tested candidate on this track, not
  as final proof of general long-context improvement.

## Latest EventQA Attribution Evidence Boundary (2026-07-04)

- `EXP-20260704-001` and `EXP-20260704-002` are oracle diagnostic
  counterfactuals on one frozen P7 context-4 bank.
- They support feasibility of ordered-tuple harmful-memory attribution, not a
  deployable gate and not general long-context performance improvement.
- Official scorer/parser behavior is unchanged. The original frozen bank and
  canonical outputs are read-only inputs.
- Follow-up attribution expansion is paused. These runs are not part of the
  frozen current main method implementation.

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
- MAB-5C decoupled retrieve/update thresholds has completed; it preserved
  full slot growth while keeping query-time retrieval active in every context.
- MAB-6B Weaver-space bank has completed on detective_qa n10. It improved
  official exact match from `0.0` to `0.1` on the fixed slice while keeping
  `query_write_count=0`, keeping cross-context leakage `false`, and avoiding
  `reasoner_to_weaver` reprojection for retrieved memory.
- The accepted EventQA A/B/C sweep is now complete across all 15 runs.
  Config A (`0.03/0.05/8/1`) is best with Bank-on `114/500 = 0.228` versus
  Bank-off `4/500 = 0.008`; Config B and C force `15-16` slot construction
  but reduce Bank-on EM. Treat this as strong exploratory compressed
  frozen-context bridge evidence, not a direct official long-context baseline
  comparison.
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

Historical pre-MAB mechanism snapshot (superseded by later MAB-6B/EventQA
experiments):

- Reasoner-only retrieved-memory injection.
- Retrieved memory does not enter Weaver.
- Stored memory is reasoner-space `latent_inputs_embeds`.
- Memory is session-local.
- Enabled memory requires `batch_size=1`.
- Retrieval uses last-retrieved decay with no fallback top-1.

Historical next experiment gate:

- R4 infrastructure validation is complete with caveats. Before any larger
  TriviaQA run, decide whether to keep default `threshold=0.7` and search for
  naturally matching samples, or design a threshold calibration / ablation plan.
- Version B was deferred at this snapshot.

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
| EXP-20260622-001 | 2026-06-22 | MAB-5B | Does raising the shared threshold to 0.05 improve the compressed detective_qa n10 result? | `completed` | Both exact match 0.0; final slot counts rose to 8 in every context; retrieval stayed active in every context; output_changed dropped to 5 |
| EXP-20260622-002 | 2026-06-22 | MAB-5C | Does decoupling retrieval and update thresholds preserve slot growth while restoring retrieval density? | `completed` | Both exact match 0.0; final slot counts stayed at 8 in every context; query-time retrieval stayed active in every context; retrieved latents remained Reasoner-only |
| EXP-20260623-001 | 2026-06-23 | MAB-5D | Does increasing max_slots from 8 to 16 reduce eviction churn without changing exact-match behavior? | `completed` | Both exact match 0.0; final slot counts rose to 16 in every context; capacity eviction dropped versus MAB-5C; query-time retrieval stayed active in every context |
| EXP-20260625-001 | 2026-06-25 | MAB-6A | Does routing retrieved memory into Weaver change the mechanism shape on detective_qa n10 without enabling writes or fallback? | `completed_exploratory` | Both exact match 0.0; output_changed stayed 10/10; retrieved memory entered Weaver; raw retrieved memory did not enter Reasoner directly; query writes stayed 0 |
| EXP-20260625-002 | 2026-06-25 | MAB-6B | Does storing Weaver-space memory and querying in Weaver space avoid the MAB-6A projection round trip and change benchmark behavior? | `completed_exploratory` | Bank-off exact match 0.0 and Bank-on exact match 0.1; output_changed stayed 10/10; storage/query space moved to Weaver; query writes stayed 0 |
| EXP-20260626-001 | 2026-06-26 | MAB-6B-FR format repair | Does a final-query answer-only prefix improve output control without changing the Weaver-space bank mechanism? | `completed_exploratory` | Canonical 10/10-valid run increased Bank-on clean-option outputs from 3/10 to 6/10 but changed Bank-on EM from 0.1 to 0.0; format was not the only bottleneck |
| EXP-20260626-002 | 2026-06-26 | MAB-6B-FR threshold diagnostic | Does update_threshold control one-slot collapse independently of retrieve_threshold? | `completed_with_artifact_recovery` | Recovered traces show ut=0.05 ended at one slot and ut>=0.08 ended at eight slots; ut=0.08 retained 2/10 EM, but per-setting manifests are invalid after a postprocessing KeyError |
| EXP-20260626-003 | 2026-06-26 | MAB-6B-FR capacity diagnostic | With ut=0.08 and top_k=1, what storage capacity best balances slot diversity and selection noise? | `completed_exploratory` | Bank-on EM was 0.1/0.2/0.0 for cap8/16/32; capacity changed slot counts and evictions as intended; cap16 was best on n10 |
| EXP-20260626-004 | 2026-06-26 | MAB-6B-FR top-k diagnostic | Does broader final-query retrieval improve Weaver-space-bank accuracy at retrieve_threshold=0.03? | `completed_exploratory` | top_k=1/2/4/8 gave Bank-on EM 0.1/0.0/0.0/0.0; realized retrieval was 1/2/3/3 slots because thresholding capped top_k=4/8 |
| EXP-20260629-001 | 2026-06-29 | MAB-6B-FR retrieval-threshold relaxation | Does relaxing retrieve_threshold let top_k=4 realize four retrieved slots before quality is judged? | `completed_exploratory` | All eight settings completed sequentially on GPU 5; no top_k=4 run reached 32 query-turn latent tokens in all 10 contexts; top_k=4 remained mechanism-inconclusive and top_k=1 stayed preferred |
| EXP-20260629-002 | 2026-06-29 | EventQA frozen-context single-context run | Does the benchmark-conformant `frozen_context_bank` protocol show a positive signal on EventQA context_index=0 before any multi-context scaling? | `completed_exploratory` | Context was memorized once, all 100 queries reused the same frozen bank with zero query writes, Bank-off EM was 0.00, and Bank-on EM reached 0.22 on the single evaluated context |
| EXP-20260629-003 | 2026-06-29 | EventQA frozen-context all-5-context run | Does the benchmark-conformant `frozen_context_bank` protocol retain a positive signal across all 5 EventQA 65536 contexts? | `completed_exploratory` | All 5 contexts completed under isolated per-context runs; Bank-off EM was 4/500 and Bank-on EM was 83/500; protocol invariants held, but every context still collapsed to one slot |
| EXP-20260630-001 | 2026-06-30 | EventQA frozen-context config sweep | Which of Config A/B/C is the best EventQA frozen-context setting after runtime-integrity repair? | `completed_exploratory` | All 15 runs completed; Config A (`0.03/0.05/8/1`) was best at Bank-on `114/500`, while Config B/C forced 15-16-slot construction but reduced Bank-on EM to `72/500` and `67/500` |
| EXP-20260630-003 | 2026-06-30 | EventQA ctx4 reproducibility diagnostic | Does the catastrophic ctx4 failure from the accepted end-to-end Config B `top_k=2` run reproduce under a standalone rerun? | `completed_diagnostic_only` | No exact reproduction: standalone ctx4 reached `11/100` EM with 52 format failures and 44 Chinese outputs, and the dominant pair changed from `(1,0):100` to `{(3,0):84,(0,3):16}` |
| EXP-20260630-004 | 2026-06-30 | EventQA end-to-end top_k=4 ablation | Does Config B remain competitive when the mechanism uses `top_k=4` during both bank construction and frozen-bank query retrieval? | `completed_negative_result` | Bank-on EM fell to `63/500`, recall to `0.164`, format failures rose to `208`, Chinese outputs rose to `156`, and realized retrieval stayed at 3 slots / 24 latents rather than 4 slots / 32 latents |

## Recorded Experiments

### EXP-20260704-001: EventQA Harmful Attribution Smoke q0-9

- Status: completed; follow-up represented by `EXP-20260704-002`
- Artifact:
  `outputs/mab/eventqa_harmful_memory_attribution_smoke/20260704T001049Z-p7-context4-q0-9/`
- Source run:
  `outputs/mab/eventqa_p7_rt005_ut010_cap16_topk2/20260702T084825Z-eventqa-65536-version-b-weaver-space-bank-n5`
- Setup: frozen `context_4.pt`, context 4, questions `0..9`, official
  scorer/parser unchanged, pristine bank clone per question and condition.
- Conditions: `full`, `drop-slot:0`, `drop-slot:1`, `drop-tuple:1,0`,
  `slot-only:0`, `slot-only:1`, `tuple-only:1,0`.
- Replay: `10/10` full-bank questions matched.
- Key metrics:
  - full and tuple-only `[1,0]`: EM `0/10`, recall `0.20`, no-gold `8/10`,
    format failures `10/10`;
  - drop-tuple `[1,0]`: EM `3/10`, recall `0.30`, no-gold `7/10`, format
    failures `1/10`;
  - slot-only 0: EM `3/10`, format failures `3/10`;
  - slot-only 1: EM `5/10`, format failures `0/10`.
- Observation: initial evidence supported a tuple-level interaction rather than
  either slot being independently sufficient for the collapse.
- Caveat: single bank and `n=10`; exploratory only.
- Evidence: `replay_validation.json` and `attribution_summary.json` in the
  artifact directory.

### EXP-20260704-002: EventQA Harmful Attribution Context-4 q0-99

- Status: completed; further attribution expansion paused
- Artifact:
  `outputs/mab/eventqa_harmful_memory_attribution_context4_full/20260704T001824Z-p7-context4-q0-99/`
- Setup: same source run and frozen context-4 bank as
  `EXP-20260704-001`, questions `0..99`, same seven conditions.
- Replay: `100/100` full-bank questions matched on official EM, recall,
  retrieved original slot IDs, raw prediction hash, parsed prediction, and
  format flags.
- Key metrics:
  - full: EM `0/100`, recall `0.19`, no-gold `81/100`, format failures
    `98/100`;
  - drop-slot 0: EM `1/100`, recall `0.34`, no-gold `66/100`, format failures
    `89/100`;
  - drop-slot 1: EM `3/100`, recall `0.14`, no-gold `86/100`, format failures
    `70/100`;
  - drop-tuple `[1,0]`: EM `15/100`, recall `0.15`, no-gold `85/100`, format
    failures `2/100`;
  - slot-only 0: EM `30/100`, recall `0.31`, no-gold `69/100`, format
    failures `35/100`;
  - slot-only 1: EM `26/100`, recall `0.30`, no-gold `70/100`, format
    failures `15/100`;
  - tuple-only `[1,0]`: identical aggregate to full.
- Observation: full retrieval selected `[1,0]` on all 100 questions;
  tuple-only `[1,0]` reproduced the collapse; dropping that ordered tuple
  yielded 15 rescues and 96 format improvements, with no EM regressions.
- Interpretation: clear feasibility evidence for an ordered tuple-level
  harmful interaction in this bank. The remaining no-gold and recall results
  show that tuple removal is not a complete answer-quality solution.
- Caveats: one frozen bank, context 4 only, oracle diagnostic, no cross-repeat
  evidence, no non-oracle utility policy, and no final paper-level claim.
- Evidence: `replay_validation.json`, `attribution_summary.json`,
  `attribution_per_context.json`, and `attribution_per_question.jsonl` in the
  artifact directory.

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

### EXP-20260622-001: MAB-5B Raised Shared-threshold DetectiveQA n10

- Phase: MAB-5B diagnostic benchmark
- Status: `completed`
- Research question: Does raising the shared threshold from `0.03` to `0.05`
  reduce over-merge / over-compression enough to change the detective_qa n10
  compressed-memory result?
- Hypothesis: A higher shared threshold should keep retrieval active while
  allowing more slots to accumulate before replacement.
- Baseline/comparator: `20260621T013454Z-detectiveqa-compressed-n10`
- Code revision: current working-tree state on `rlm-memory-bank`
- Working tree state: existing notes and prepared runner/test present before the
  run; no code edits were required for execution
- Environment: `/home/baishilong/miniconda3/envs/memgen`, Python 3.10.20,
  PyTorch 2.12.0+cu126, CUDA 12.6, single RTX A6000 selected via
  `CUDA_VISIBLE_DEVICES=2`
- Dataset and split: `Long_Range_Understanding / detective_qa`, 10 contexts
- Inputs/session definition: first-query-only, compressed Bank-off vs Bank-on,
  read-only query phase, session-local memory, no fallback, no Weaver injection
- Configuration: `threshold=0.05`, `top_k=1`, `max_slots=8`,
  `retrieve_policy=threshold_topk`, `update_policy=thread_update`
- Random seed: 42
- Batch size: 1
- Output directory:
  `outputs/mab/raised_shared_threshold_detectiveqa_n10/20260622T073545Z-detectiveqa-raised-shared-threshold-n10`
- Run notes:
  - the checked-in runner now inserts the repository root into `sys.path`, so
    the direct repo-root invocation is the canonical entry point

#### Observations

- Both compressed Bank-off and Bank-on exact match remained `0.0`.
- Final slot counts increased to `[8, 8, 8, 8, 8, 8, 8, 8, 8, 8]`.
- Mean final slot count reached `8.0`.
- Retrieval stayed active in all 10 contexts, but retrieved latent count fell to
  `200` from MAB-5A's `2248`.
- Query write count remained `0`.
- Cross-context leakage remained absent.
- Retrieved memory remained Reasoner-only and never entered Weaver.
- Retrieved score range was approximately `0.050-0.064`.
- `matched_replace_count`, `thread_insert_count`, and `capacity_evict_count`
  were not exposed in the current artifact schema.

#### Conclusion

- Hypothesis supported: partially.
- Interpretation: Raising the shared threshold clearly increased slot counts
  without breaking retrieval activity, which makes the simple baseline much
  stronger than MAB-5A.
- Exact match did not improve, so this remains mechanism evidence rather than
  performance evidence.
- Follow-up: MAB-5C is complete; keep any future fallback or Weaver-conditioning
  refinement separate from this diagnostic result.
- Related baseline: `20260621T013454Z-detectiveqa-compressed-n10`

### EXP-20260622-002: MAB-5C Decoupled Retrieval-Update Thresholds DetectiveQA n10

- Phase: MAB-5C diagnostic benchmark
- Status: `completed`
- Research question: Can separate retrieval and update thresholds keep
  MAB-5B-style slot growth while restoring MAB-5A-style retrieval density?
- Hypothesis: `retrieve_threshold=0.03` and `update_threshold=0.05` should
  preserve low retrieval-score visibility while allowing write-back to favor
  new threads over over-merge.
- Baseline/comparator: `20260621T013454Z-detectiveqa-compressed-n10` and
  `20260622T073545Z-detectiveqa-raised-shared-threshold-n10`
- Code revision: current working-tree state on `rlm-memory-bank`
- Working tree state: prepared runner/test present before the run; no code
  edits were required for execution
- Environment: `/home/baishilong/miniconda3/envs/memgen`, Python 3.10.20,
  PyTorch 2.12.0+cu126, CUDA 12.6, single RTX A6000 selected via
  `CUDA_VISIBLE_DEVICES=2`
- Dataset and split: `Long_Range_Understanding / detective_qa`, 10 contexts
- Inputs/session definition: first-query-only, compressed Bank-off vs Bank-on,
  read-only query phase, session-local memory, no fallback, no Weaver injection
- Configuration: `threshold=0.03`, `retrieve_threshold=0.03`,
  `update_threshold=0.05`, `top_k=1`, `max_slots=8`,
  `retrieve_policy=threshold_topk`, `update_policy=thread_update`
- Output directory:
  `outputs/mab/decoupled_thresholds_detectiveqa_n10/20260622T140741Z-detectiveqa-decoupled-thresholds-n10`
- Provenance:
  - preliminary non-canonical runtime-patch run:
    `outputs/mab/decoupled_thresholds_detectiveqa_n10/20260622T131149Z-detectiveqa-decoupled-thresholds-n10`
  - canonical checked-in-runner rerun:
    `outputs/mab/decoupled_thresholds_detectiveqa_n10/20260622T140741Z-detectiveqa-decoupled-thresholds-n10`

#### Observations

- Both compressed Bank-off and Bank-on exact match remained `0.0`.
- Final slot counts stayed at `[8, 8, 8, 8, 8, 8, 8, 8, 8, 8]`.
- Mean final slot count remained `8.0`.
- Retrieval stayed active in all 10 contexts.
- Query-turn retrieval stayed active in all 10 contexts.
- Query write count remained `0`.
- Cross-context leakage remained absent.
- Retrieved memory remained Reasoner-only and never entered Weaver.
- Retrieved latent count was `2288`, which is much closer to the dense
  MAB-5A regime than to MAB-5B.
- Query-turn retrieved latent count was `80` total, with 8 latents in each
  query turn.
- Construction-time retrieval count was `306`.
- `matched_replace_count=36`, `thread_insert_count=80`,
  `capacity_evict_count=210`.

#### Conclusion

- Hypothesis supported: yes for mechanism shape, no for exact-match gain.
- Interpretation: The decoupled thresholds achieved the intended split.
  Retrieval remained active in every context while the bank still grew to full
  capacity, and the canonical checked-in runner rerun is the source of truth.
- This is diagnostic evidence, not a performance win.

### EXP-20260623-001: MAB-5D Capacity16 DetectiveQA n10

- Phase: MAB-5D diagnostic benchmark
- Status: `completed`
- Research question: Does increasing `max_slots` from 8 to 16 reduce eviction
  churn while preserving the decoupled-threshold mechanism on detective_qa n10?
- Hypothesis: keeping `retrieve_threshold=0.03` and `update_threshold=0.05`
  while doubling `max_slots` should raise final slot counts to 16 in every
  context and reduce capacity eviction, but not improve official exact match.
- Baseline/comparator: `20260622T140741Z-detectiveqa-decoupled-thresholds-n10`
  and `20260622T073545Z-detectiveqa-raised-shared-threshold-n10`
- Code revision: current working-tree state on `rlm-memory-bank`
- Working tree state: prepared runner/test present before the run; no code
  edits were required for execution
- Environment: `/home/baishilong/miniconda3/envs/memgen`, Python 3.10.20,
  PyTorch 2.12.0+cu126, CUDA 12.6, single RTX A6000 selected via
  `CUDA_VISIBLE_DEVICES=2`
- Dataset and split: `Long_Range_Understanding / detective_qa`, 10 contexts
- Inputs/session definition: first-query-only, compressed Bank-off vs Bank-on,
  read-only query phase, session-local memory, no fallback, no Weaver injection
- Configuration: `threshold=0.03`, `retrieve_threshold=0.03`,
  `update_threshold=0.05`, `top_k=1`, `max_slots=16`,
  `retrieve_policy=threshold_topk`, `update_policy=thread_update`
- Output directory:
  `outputs/mab/capacity16_detectiveqa_n10/20260623T022140Z-detectiveqa-capacity16-n10`
- Non-canonical earlier attempt:
  `outputs/mab/capacity16_detectiveqa_n10/20260623T015929Z-detectiveqa-decoupled-thresholds-n10`

#### Observations

- Both compressed Bank-off and Bank-on exact match remained `0.0`.
- Final slot counts rose to `[16, 16, 16, 16, 16, 16, 16, 16, 16, 16]`.
- Mean final slot count reached `16.0`.
- Retrieval stayed active in all 10 contexts.
- Query-turn retrieval stayed active in all 10 contexts.
- Query write count remained `0`.
- Cross-context leakage remained absent.
- Retrieved memory remained Reasoner-only and never entered Weaver.
- Retrieved latent count was `2272`, only slightly lower than MAB-5C.
- Query-turn retrieved latent count was `80` total, with 8 latents in each
  query turn.
- Construction-time retrieval count was `306`.
- `matched_replace_count=33`, `append_insert_count=160`,
  `capacity_evict_count=133`.
- Context 6 provides a relaxed diagnostic example:
  gold answer `C. Misty Sketches` versus bank-on prediction
  `答案：C. Misty Sketches\n答案`, but official exact match remained `0`.

#### Conclusion

- Hypothesis supported: yes for capacity/eviction behavior, no for exact-match
  gain.
- Interpretation: Increasing `max_slots` from 8 to 16 raises final slot counts
  to the new capacity in every context and reduces eviction churn, but it does
  not improve official exact match.
- This is mechanism-positive and performance-neutral/negative.
- Follow-up: the next meaningful mechanism direction is MAB-6A / Version B
  retrieved-memory-to-Weaver conditioning, if and only if it remains isolated
  from Version A.
- Related baseline: `20260622T140741Z-detectiveqa-decoupled-thresholds-n10`

### EXP-20260625-001: MAB-6A Version B Weaver-conditioned Memory DetectiveQA n10

- Phase: MAB-6A exploratory benchmark
- Status: `completed_exploratory`
- Research question: Does routing retrieved memory into Weaver, rather than
  injecting raw retrieved memory directly into Reasoner, produce a distinct
  mechanism shape on detective_qa n10 while preserving Version A guardrails?
- Hypothesis: Version B routing will be mechanism-active, keep query writes at
  zero, avoid cross-context leakage, and change outputs even if exact match does
  not improve.
- Baseline/comparator:
  `outputs/mab/decoupled_thresholds_detectiveqa_n10/20260622T140741Z-detectiveqa-decoupled-thresholds-n10`
- Canonical artifact:
  `outputs/mab/version_b_weaver_conditioned_detectiveqa_n10/20260625T023822Z-detectiveqa-version-b-weaver-conditioned-n10`
- Earlier failed/intermediate artifacts:
  `20260625T021750Z-detectiveqa-version-b-weaver-conditioned-n10`,
  `20260625T021830Z-detectiveqa-version-b-weaver-conditioned-n10`,
  `20260625T022818Z-detectiveqa-version-b-weaver-conditioned-n10`
- Configuration:
  - `threshold=0.03`
  - `retrieve_threshold=0.03`
  - `update_threshold=0.05`
  - `max_slots=8`
  - `top_k=1`
  - `retrieve_policy=threshold_topk`
  - `update_policy=thread_update`
  - `retrieved_memory_to_weaver=True`
  - query mode `first-query-only`
  - query phase `read-only`
  - full-history detective_qa `over_capacity_invalid`

#### Observations

- Valid contexts: `10/10`
- Bank-off exact match: `0.0`
- Bank-on exact match: `0.0`
- `output_changed=10`
- Query-turn retrieval active contexts: `10`
- Final slot counts: `[8, 8, 8, 8, 8, 8, 8, 8, 8, 8]`
- Retrieved memory entered Weaver in all valid contexts.
- Raw retrieved memory did not enter Reasoner directly.
- `weaver_conditioning_token_count=80` across the 10 contexts.
- `fused_latent_generated=True`
- Query writes remained `0`; query write attempts remained `0`.
- Cross-context leakage remained `false`.
- One over-capacity warning was emitted at `132726 > 131072` while estimating
  the invalid full-history path, but no full-history detective_qa generation was
  run or scored.

#### Conclusion

- Hypothesis supported: partially.
- Interpretation: Version B routing is mechanism-active and isolated as
  intended, but it did not improve official exact match relative to MAB-5C.
- This is not a performance win. Version A remains the default.
- Follow-up: if more Version B work is approved, do failure analysis first
  rather than another threshold or capacity sweep.

### EXP-20260626-001: MAB-6B-FR Final-query Format Repair

- Phase: MAB-6B-FR output-control diagnostic
- Status: `completed_exploratory`
- Research question: Can a constrained final-query answer-only prefix improve
  output controllability without changing chunk processing or the Weaver-space
  memory-bank mechanism?
- Configuration:
  - `retrieve_threshold=0.03`
  - `update_threshold=0.05`
  - `max_slots=8`
  - `top_k=1`
  - format repair applied to Bank-off and Bank-on final queries only
- Canonical artifact:
  `outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_format_repair/20260626T014628Z-detectiveqa-version-b-weaver-space-bank-format-repair-n10`
- Invalid precursor artifact:
  `outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_format_repair/20260626T014442Z-detectiveqa-version-b-weaver-space-bank-format-repair-n10`

#### Observations

- Canonical validity: 10/10 contexts; precursor validity: 0/10 after
  `RecursionError: maximum recursion depth exceeded` in every context.
- Bank-off EM: `0.0`; Bank-on EM: `0.0`; improved: `0`; regressed: `0`;
  `output_changed=10`.
- Final slot counts stayed at one in every context; write actions were
  `insert=10`, `replace_matched=316`, `capacity_evict=0`.
- Query-turn retrieval remained one slot / eight latent tokens per context.
- Relative to the no-repair MAB-6B run, Bank-on clean-option outputs increased
  from 3/10 to 6/10, while Bank-on EM changed from 0.1 to 0.0.
- Invariants: query writes and attempts `0`; leakage `false`; retrieved latents
  entered Weaver; raw retrieved latents did not enter Reasoner.
- Missing root-level artifacts: no format-repair aggregate, per-context export,
  summary Markdown, or run log is present.

#### Conclusion

- Format repair improved surface-form control but did not improve correctness.
- Memory content compression, retrieval selection, and Weaver utilization
  remain plausible bottlenecks.
- The memory bank changed all outputs without reliably moving predictions
  toward correct answers.

### EXP-20260626-002: MAB-6B-FR Threshold-only Diagnostic

- Phase: MAB-6B-FR memory-formation diagnostic
- Status: `completed_with_artifact_recovery`
- Research question: With `retrieve_threshold` fixed, does
  `update_threshold` control one-slot collapse versus multi-slot formation?
- Configuration:
  - fixed `retrieve_threshold=0.03`, `max_slots=8`, `top_k=1`
  - swept `update_threshold={0.05,0.08,0.10,0.12}`
- Artifact root:
  `outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_threshold_diagnostic/`
- Primary recovered reports:
  `threshold_diagnostic_aggregate.json`,
  `threshold_diagnostic_per_context.jsonl`, and
  `threshold_diagnostic_summary.md`

#### Observations

- Recovered Bank-on EM for ut 0.05/0.08/0.10/0.12 was
  `0.20/0.20/0.00/0.00`; Bank-off EM was `0.00` throughout.
- `ut=0.05`: all final slot counts were 1; write actions were
  `insert=10`, `replace_matched=316`, `capacity_evict=0`.
- `ut=0.08/0.10/0.12`: all final slot counts were 8; each setting recorded
  `insert=80`, `replace_matched=246`, `capacity_evict=0`.
- Query-turn retrieval remained one slot / eight latent tokens per context.
- Recovered invariants: query writes and attempts `0`; leakage `false`;
  configured routing is retrieved latents to Weaver without raw retrieved
  latents entering Reasoner.
- Artifact caveat: every per-setting `paired_results.json` and `manifest.json`
  reports 0/10 valid because postprocessing raised
  `KeyError: 'memory_retrieved_latent_count'`. The root aggregate was later
  recomputed from complete generated predictions and bank-debug rows using the
  benchmark scorer. No threshold run log is present.

#### Conclusion

- `update_threshold` controls the observed single-slot collapse.
- `update_threshold=0.08` is the current working value because it forms eight
  slots while retaining the best recovered EM in this sweep.
- These are strong slot-mechanism traces but recovered, not clean canonical,
  performance evidence.

### EXP-20260626-003: MAB-6B-FR Capacity Diagnostic

- Phase: MAB-6B-FR storage-capacity diagnostic
- Status: `completed_exploratory`
- Research question: With multi-slot formation enabled and final-query
  retrieval fixed to one slot, how does capacity affect storage dynamics and
  output quality?
- Configuration:
  - fixed `retrieve_threshold=0.03`, `update_threshold=0.08`, `top_k=1`
  - swept `max_slots={8,16,32}`
- Artifact root:
  `outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_capacity_diagnostic/`
- Primary reports: root `capacity_diagnostic_aggregate.*`,
  `capacity_diagnostic_per_context.*`, `capacity_diagnostic_summary.md`, and
  `run_capacity_full.log`; nested `smoke_test/` is excluded from the full result.

#### Observations

- All three settings completed 10/10 valid contexts with status 0.
- Bank-on EM for cap8/cap16/cap32 was `0.10/0.20/0.00`; Bank-off EM was `0.00`.
- Final slots were all 8, all 16, and `[22,24,24,28,24,21,25,32,32,20]`.
- Raw per-context write traces give insert counts `80/160/252`, matched
  replacements `60/53/54`, and capacity evictions `186/113/20`.
- Final-query retrieval stayed at one slot / eight latent tokens for every
  capacity.
- Format-clean Bank-on outputs were `8/10`, `9/10`, and `8/10`.
- Invariants held: query writes and attempts `0`; leakage `false`; retrieved
  latents entered Weaver; raw retrieved latents did not enter Reasoner.
- Artifact caveat: the root aggregate/summary generic action fields are
  zero/empty because they read the wrong field family; the counts above are
  recomputed from `bank_on_write_action_counts` in per-context artifacts.

#### Conclusion

- `max_slots` is wired correctly: effective capacity rises and eviction churn
  falls as configured capacity increases.
- More storage is not always better. cap16 is the current preferred balance
  under top_k=1, while cap32 likely enlarges the retrieval-noise surface without
  increasing final-query retrieval breadth.
- n10 and run-to-run variance prevent a final performance claim.

### EXP-20260626-004: MAB-6B-FR Top-k Diagnostic

- Phase: MAB-6B-FR retrieval-breadth diagnostic
- Status: `completed_exploratory`
- Research question: Does increasing top-k improve correctness once the bank
  forms multiple slots and uses the preferred cap16 capacity?
- Configuration:
  - fixed `max_slots=16`, `retrieve_threshold=0.03`, `update_threshold=0.08`
  - swept `top_k={1,2,4,8}`
- Artifact root:
  `outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_topk_diagnostic/`
- Primary reports: `topk_diagnostic_aggregate.*`,
  `topk_diagnostic_per_context.*`, `topk_diagnostic_summary.md`, and
  `run_topk_full.log`

#### Observations

- Final aggregate Bank-on EM for top_k 1/2/4/8 was
  `0.10/0.00/0.00/0.00`; Bank-off EM was `0.00` throughout.
- Realized final-query retrieval was 1, 2, 3, and 3 slots, corresponding to
  8, 16, 24, and 24 latent tokens per context.
- The implemented `threshold_topk` policy filters scores by
  `retrieve_threshold` before applying top-k, so top_k=4/8 were not force-top-k
  interventions.
- Bank-on format failures were 1/10, 6/10, 4/10, and 6/10. Top_k=4 produced
  four empty outputs; top_k=8 produced three.
- All final settings completed 10/10 valid contexts; query writes and attempts
  were `0`; leakage was `false`; retrieved latents entered Weaver; raw
  retrieved latents did not enter Reasoner.
- Run-to-run caveat: an earlier valid top_k=1 artifact scored 2/10; the final
  sweep's top_k=1 rerun scored 1/10 and is the row used by the aggregate.

#### Conclusion

- More retrieved latent tokens did not improve EM and reduced output control
  relative to top_k=1 under `retrieve_threshold=0.03`.
- This result does not show that force-top-k is harmful because thresholding
  capped realized retrieval at three slots for top_k=4/8.
- Follow-up: hold `max_slots=16`, `update_threshold=0.08`, and `top_k=4`, then
  sweep `retrieve_threshold={0.03,0.02,0.01,0.00}` and require 32 final-query
  latent tokens before judging Weaver multi-latent utilization. Include a
  same-run control with `max_slots=16`, `update_threshold=0.08`, `top_k=1`, and
  `retrieve_threshold=0.03` to separate threshold-relaxation effects from the
  observed `1/10` versus `2/10` top_k=1 run variance.

### EXP-20260629-001: MAB-6B-FR Retrieval-Threshold Relaxation Diagnostic

- Phase: MAB-6B-FR retrieval-threshold relaxation
- Status: `completed_exploratory`
- Research question: Does relaxing `retrieve_threshold` let top_k=4 realize the
  intended four-slot / 32-latent final-query retrieval before answer-quality
  claims are made?
- Configuration:
  - fixed `max_slots=16`, `update_threshold=0.08`
  - swept paired settings:
    `retrieve_threshold={0.03,0.02,0.01,0.005}` x `top_k={1,4}`
  - all workers executed sequentially on GPU 5
- Script:
  `scripts/eval/mab6b_weaver_space_bank_detectiveqa_n10_retrieve_threshold_relaxation.py`
- Artifact root:
  `outputs/mab/version_b_weaver_space_bank_detectiveqa_n10_retrieve_threshold_relaxation/20260629T005555Z-full-sweep/`
- Primary reports:
  - `retrieve_threshold_relaxation_aggregate.json`
  - `retrieve_threshold_relaxation_per_context.jsonl`
  - `research_notes/benchmarks/memoryagentbench_mab6b_fr_retrieve_threshold_relaxation.md`

#### Observations

- All eight settings succeeded; no worker failed and no worker wrote the
  benchmark note directly.
- top_k=1 results:
  - `rt=0.03`: Bank-on EM `0.0`, final-query retrieval `8` latent tokens in
    all contexts, Bank-on format failures `5/10`
  - `rt=0.02`: Bank-on EM `0.1`, final-query retrieval `8` latent tokens in
    all contexts, Bank-on format failures `3/10`
  - `rt=0.01`: Bank-on EM `0.1`, final-query retrieval `8` latent tokens in
    all contexts, Bank-on format failures `3/10`
  - `rt=0.005`: Bank-on EM `0.1`, final-query retrieval `8` latent tokens in
    all contexts, Bank-on format failures `1/10`
- top_k=4 results:
  - `rt=0.03`: Bank-on EM `0.0`, final-query retrieval `24` latent tokens in
    all contexts
  - `rt=0.02`: Bank-on EM `0.0`, final-query retrieval `24` latent tokens in
    all contexts
  - `rt=0.01`: Bank-on EM `0.0`, mixed final-query retrieval
    `24/32` latent tokens by context
  - `rt=0.005`: Bank-on EM `0.0`, mixed final-query retrieval
    `24/32` latent tokens by context
- No top_k=4 run reached 32 query-turn retrieved latent tokens in all 10
  contexts, so no top_k=4 row qualifies as a valid force-top-k quality
  comparison.
- top_k=4 also showed weaker output control:
  - Bank-on format failures were `6/10`, `8/10`, `7/10`, and `3/10`
  - empty Bank-on outputs were `5`, `4`, `3`, and `2`
- All settings preserved the key invariants:
  query writes and attempts `0`, cross-context leakage `false`, retrieved
  latents entered Weaver, and raw retrieved latents did not enter Reasoner.

#### Conclusion

- Retrieval-threshold relaxation increased realized top_k=4 retrieval in some
  contexts, but not enough to produce a clean four-slot intervention.
- top_k=4 remains mechanism-inconclusive and must not be interpreted as a
  valid quality comparison.
- top_k=1 remains the preferred diagnostic setting for expansion to EventQA.
- Among the relaxed top_k=1 settings, `retrieve_threshold=0.005` had the
  cleanest output surface, while `0.02`, `0.01`, and `0.005` each reached
  Bank-on EM `1/10`.
- Recommended cautious setting for a future EventQA diagnostic after the
  runtime-config repair:
  `retrieve_threshold=0.005`, `update_threshold=0.08`, `top_k=1`,
  `max_slots=16`. This is not the configuration of EXP-20260629-002 or
  EXP-20260629-003.
- This remains exploratory mechanism evidence on n10 and does not support a
  benchmark-improvement claim.

### EXP-20260629-002: EventQA Frozen-Context Single-Context Run

- Phase: EventQA exploratory benchmark-conformant pilot
- Status: `completed_exploratory`
- Research question: does the benchmark-conformant `frozen_context_bank`
  protocol show a positive signal on EventQA `context_index=0` before any
  multi-context scaling?
- Configuration:
  - actual runtime `retrieve_threshold=0.03`
  - actual runtime `update_threshold=0.05`
  - `top_k=1`
  - actual runtime `max_slots=8`
  - `generation_max_length=40`
  - `eventqa_protocol=frozen_context_bank`
  - `requested_contexts=1`
  - no `question_limit`, so all 100 questions in `context_index=0` were
    evaluated
- Configuration correction: the runner wrote intended constants
  `retrieve_threshold=0.005`, `update_threshold=0.08`, and `max_slots=16` to
  the manifest but passed the DetectiveQA `0.03/0.05/8` config to runtime.
  This experiment must be attributed to the actual runtime values above.
- Command:
  `CUDA_VISIBLE_DEVICES=3 /home/baishilong/miniconda3/envs/memgen/bin/python scripts/eval/mab6b_weaver_space_bank_eventqa_65536_n5.py --requested-contexts 1 --skip-research-note --eventqa-protocol frozen_context_bank`
- Script:
  `scripts/eval/mab6b_weaver_space_bank_eventqa_65536_n5.py`
- Artifact root:
  `outputs/mab/version_b_weaver_space_bank_eventqa_65536_n5/20260629T121408Z-eventqa-65536-version-b-weaver-space-bank-n5/`
- Primary reports:
  - `eventqa_aggregate.json`
  - `eventqa_per_question.jsonl`
  - `eventqa_per_context.jsonl`
  - `diagnostics.jsonl`
  - `research_notes/benchmarks/memoryagentbench_mab6b_fr_eventqa_65536_n5.md`

#### Observations

- Protocol validation:
  - `context_memorization_count=1`
  - `same_frozen_bank_reused_across_queries=true`
  - all 100 queries shared the same frozen bank instance
  - bank snapshot never changed after query
  - total query write delta `0`
  - max query write delta `0`
  - blocked query write attempts total `100` with distribution `{1:100}`
- Mechanism shape:
  - construction `chunk_count=17`
  - construction `final_slot_count=1`
  - `true_insert_count=1`
  - `true_matched_replace_count=16`
  - `true_capacity_evict_count=0`
  - `true_replace_old_slot_count=0`
  - candidate slot count before top-k distribution `{1:100}`
  - retrieved indices distribution `{(0,):100}`
  - retrieved latent count distribution `{8:100}`
  - raw candidate score min / max / mean
    `0.04947 / 0.05574 / 0.05229`
- Result on `context_index=0`:
  - Bank-off substring EM `0/100 = 0.00`
  - Bank-on substring EM `22/100 = 0.22`
  - Bank-off `eventqa_recall` `15/100 = 0.15`
  - Bank-on `eventqa_recall` `22/100 = 0.22`
  - improved / regressed / unchanged `22 / 0 / 78`
  - `output_changed_count=100`
  - bank-off / bank-on format failures `83 / 19`
  - bank-off / bank-on Chinese-script outputs `36 / 0`
- Representative improved examples:
  - Q2 bank-on exactly output
    `Debbie expressed her boredom with the talk of war.`
  - Q3 bank-on exactly output
    `Debbie mentioned her mother, Lucian O'Kerry, during the conversation.`
  - Q49 bank-on exactly output
    `Marianne felt joy and a sense of ownership standing on the foundation of his new plantation.`
- Representative unchanged wrong examples:
  - early Q0 remained wrong
  - late Q99 remained wrong

#### Conclusion

- This is the first benchmark-conformant EventQA positive signal for the
  `frozen_context_bank` protocol.
- The result is still only single-context evidence and must not be promoted to
  a final benchmark-improvement claim.
- Bank-on improved official EM over the compressed-bridge Bank-off baseline on
  `context_index=0`.
- The main mechanism risk remains construction-time single-slot collapse: the

### EXP-20260629-003: EventQA Frozen-Context All-5-Context Run

- Phase: EventQA exploratory benchmark-conformant expansion
- Status: `completed_exploratory`
- Research question: does the benchmark-conformant `frozen_context_bank`
  protocol retain a positive signal across all 5 local EventQA 65536 contexts?
- Configuration:
  - actual runtime `retrieve_threshold=0.03`
  - actual runtime `update_threshold=0.05`
  - `top_k=1`
  - actual runtime `max_slots=8`
  - `generation_max_length=40`
  - `eventqa_protocol=frozen_context_bank`
  - one isolated context per process via `--context-index`
- Configuration correction:
  - the preserved manifests recorded the intended cautious values
    `retrieve_threshold=0.005`, `update_threshold=0.08`, and `max_slots=16`
  - runtime instead received the imported DetectiveQA config
    `retrieve_threshold=0.03`, `update_threshold=0.05`, and `max_slots=8`
  - the all-5 result remains valid frozen-context evidence, but it must be
    attributed only to the actual runtime config
- Scheduling support:
  - the runner now accepts `--context-index` so one EventQA context can be
    forced into one isolated process / output root for safe multi-GPU parallel
    evaluation
  - run metadata now records `context_index` and
    `selected_context_indices`
- Commands / run roots:
  - `ctx0`:
    `CUDA_VISIBLE_DEVICES=6 ... --context-index 0 --output-root outputs/mab/version_b_weaver_space_bank_eventqa_65536_n5_ctx0`
    -> `outputs/mab/version_b_weaver_space_bank_eventqa_65536_n5_ctx0/20260629T131415Z-eventqa-65536-version-b-weaver-space-bank-n5/`
  - `ctx1`:
    `CUDA_VISIBLE_DEVICES=0 ... --context-index 1 --output-root outputs/mab/version_b_weaver_space_bank_eventqa_65536_n5_ctx1`
    -> `outputs/mab/version_b_weaver_space_bank_eventqa_65536_n5_ctx1/20260629T131413Z-eventqa-65536-version-b-weaver-space-bank-n5/`
  - `ctx2`:
    `CUDA_VISIBLE_DEVICES=5 ... --context-index 2 --output-root outputs/mab/version_b_weaver_space_bank_eventqa_65536_n5_ctx2`
    -> `outputs/mab/version_b_weaver_space_bank_eventqa_65536_n5_ctx2/20260629T131413Z-eventqa-65536-version-b-weaver-space-bank-n5/`
  - `ctx3`:
    `CUDA_VISIBLE_DEVICES=3 ... --context-index 3 --output-root outputs/mab/version_b_weaver_space_bank_eventqa_65536_n5_ctx3`
    -> `outputs/mab/version_b_weaver_space_bank_eventqa_65536_n5_ctx3/20260629T133550Z-eventqa-65536-version-b-weaver-space-bank-n5/`
  - `ctx4`:
    `CUDA_VISIBLE_DEVICES=0 ... --context-index 4 --output-root outputs/mab/version_b_weaver_space_bank_eventqa_65536_n5_ctx4`
    -> `outputs/mab/version_b_weaver_space_bank_eventqa_65536_n5_ctx4/20260629T133555Z-eventqa-65536-version-b-weaver-space-bank-n5/`
- Primary reports per run:
  - `eventqa_aggregate.json`
  - `eventqa_per_question.jsonl`
  - `eventqa_per_context.jsonl`
  - `diagnostics.jsonl`
  - `manifest.json`
  - `run_config.json`
  - `paired_results.json`

#### Observations

- Protocol validation held in all 5 contexts:
  - `context_memorization_count=1`
  - `same_frozen_bank_reused_across_queries=true`
  - all 100 queries per context shared the same frozen bank instance
  - `bank_snapshot_changed_after_query=false`
  - total query write delta `0`
  - max query write delta `0`
  - blocked query write attempts total `500` with distribution `{1:500}`
  - `cross_context_leakage_detected=false`
- Overall result across `500` questions:
  - Bank-off substring EM `4/500 = 0.008`
  - Bank-on substring EM `83/500 = 0.166`
  - absolute improvement `+0.158`
  - Bank-off `eventqa_recall=0.178`
  - Bank-on `eventqa_recall=0.208`
  - improved / regressed / unchanged `81 / 2 / 417`
  - `output_changed_count=500`
  - bank-off / bank-on format failures `377 / 173`
  - bank-off / bank-on Chinese-script outputs `189 / 30`
- Per-context EM:
  - `ctx0`: `0/100 -> 17/100`
  - `ctx1`: `0/100 -> 3/100`
  - `ctx2`: `0/100 -> 19/100`
  - `ctx3`: `3/100 -> 21/100`
  - `ctx4`: `1/100 -> 23/100`
- Per-context improved / regressed / unchanged:
  - `ctx0`: `17 / 0 / 83`
  - `ctx1`: `3 / 0 / 97`
  - `ctx2`: `19 / 0 / 81`
  - `ctx3`: `19 / 1 / 80`
  - `ctx4`: `23 / 1 / 76`
- Mechanism shape remained collapsed in all 5 contexts:
  - each context had `17` construction chunks
  - each context ended with `final_slot_count=1`
  - each context had `true_insert_count=1` and
    `true_matched_replace_count=16`
  - `true_capacity_evict_count=0` and `true_replace_old_slot_count=0` in all
    contexts
  - aggregate candidate slot count before top-k `{1:500}`
  - aggregate retrieved indices `{(0,):500}`
  - aggregate retrieved latent count `{8:500}`
  - aggregate raw candidate score min / max / mean
    `0.04529 / 0.07478 / 0.05641`
- Representative preserved examples:
  - improved:
    `ctx0 q1`, `ctx1 q13`, `ctx2 q3`, `ctx3 q1`, `ctx4 q0`
  - regressed:
    `ctx3 q42`, `ctx4 q6`
  - unchanged wrong examples preserved across early / middle / late positions

#### Conclusion

- This is strong exploratory evidence that Bank-on improves over the
  compressed-bridge Bank-off baseline under the benchmark-conformant
  `frozen_context_bank` lifecycle under runtime `0.03/0.05/1/8`.
- It is not a final benchmark-improvement claim and it is not an official full
  long-context baseline comparison.
- The gain cannot yet be attributed to diverse slot retrieval because every
  context still collapsed to one final memory slot.
- The next recommended mechanism study is a frozen-context slot-collapse /
  matched-replacement diagnostic rather than more immediate full EventQA runs.
  bank behaved like one compressed latent memory slot rather than diverse event
  slots.
- Follow-up boundary: preserve this result and do not run the remaining 4
  contexts until explicitly approved.

### EXP-20260630-001: EventQA Frozen-context A/B/C Config Sweep

- Phase: EventQA frozen-context config sweep
- Status: `completed_exploratory`
- Research question:
  which EventQA frozen-context configuration is best after the runner-side
  runtime-integrity repair?
- Hypothesis:
  the historical actual control (`0.03/0.05/8/1`) may still outperform the
  larger-bank multi-slot candidates, so low retrieve-threshold transfer from
  detective_qa should not be assumed better for EventQA.
- Baseline/comparator:
  Config A versus Config B versus Config C under the same EventQA protocol and
  five-context coverage.
- Environment:
  `memgen` Python environment, host CUDA runtime, one EventQA process per GPU.
- Dataset and split:
  MemoryAgentBench Accurate Retrieval EventQA-65536, `context_index=0..4`,
  `100` questions per context, total `500` questions per config.
- Common configuration:
  `eventqa_protocol=frozen_context_bank`, `generation_max_length=40`,
  `top_k=1`, `--skip-research-note`.
- Configs:
  - Config A:
    `retrieve_threshold=0.03`, `update_threshold=0.05`, `max_slots=8`
  - Config B:
    `retrieve_threshold=0.03`, `update_threshold=0.09`, `max_slots=16`
  - Config C:
    `retrieve_threshold=0.005`, `update_threshold=0.09`, `max_slots=16`
- Artifact roots:
  - Config A:
    `outputs/mab/eventqa_frozen_context_bank_cfgA_ctx{0..4}/...`
  - Config B:
    `outputs/mab/eventqa_frozen_context_bank_cfgB_ctx{0..4}/...`
  - Config C:
    `outputs/mab/eventqa_frozen_context_bank_cfgC_ctx{0..4}/...`

#### Observations

- Integrity:
  - all 15 runs completed
  - `manifest.json` and `run_config.json` matched intended config for every run
  - `query_write_count_delta total/max = 0 / 0`
  - `bank_snapshot_changed_after_query=false`
  - `cross_context_leakage_detected=false`
  - canonical detective note SHA / mtime unchanged
- Config A overall:
  - Bank-off `4/500 = 0.008`
  - Bank-on `114/500 = 0.228`
  - Bank-off recall `0.178`
  - Bank-on recall `0.266`
  - improved / regressed / unchanged `113 / 3 / 384`
  - per-context Bank-on EM
    `0.18 / 0.42 / 0.21 / 0.17 / 0.16`
  - final slot distribution `{1:500}`
  - retrieved indices `{(0,):500}`
  - candidate slot distribution `{1:500}`
  - Bank-on format failures `123`
  - Bank-on Chinese-script outputs `23`
- Config B overall:
  - Bank-on `72/500 = 0.144`
  - Bank-on recall `0.202`
  - improved / regressed / unchanged `72 / 4 / 424`
  - final slot distribution `{16:500}`
  - retrieved indices `{(0,):400,(4,):100}`
  - Bank-on format failures `141`
  - Bank-on Chinese-script outputs `120`
- Config C overall:
  - Bank-on `67/500 = 0.134`
  - Bank-on recall `0.208`
  - improved / regressed / unchanged `66 / 3 / 431`
  - final slot distribution `{15:100,16:400}`
  - retrieved indices `{(0,):300,(5,):100,(12,):100}`
  - Bank-on format failures `165`
  - Bank-on Chinese-script outputs `116`
- Mechanism:
  - Config B and C force `15-16` slot construction
  - query-time retrieval still returns exactly one slot in all three settings
    because `retrieved_latent_count` stayed `{8:500}` and `top_k=1`

#### Conclusion

- Hypothesis supported: yes.
- Interpretation:
  Config A is the best EventQA setting in this sweep. Multi-slot construction
  under `top_k=1` hurt EventQA in this compressed frozen-context bridge rather
  than helping it.
- Boundary:
  this uses the official EventQA substring-exact-match / Accuracy scorer, but
  it is still not a direct official full-context baseline comparison.
- Follow-up:
  keep Config A for the next EventQA setting. If a later approved study still
  wants multi-slot benefit, design a query-time retrieval intervention that
  actually returns more than one slot.

### EXP-20260630-002: EventQA End-to-End Config B `top_k=2` Ablation

- Phase: EventQA exploratory multi-slot ablation
- Status: `completed_exploratory`
- Research question:
  does Config B improve when the mechanism uses `top_k=2` during both bank
  construction and frozen-bank query retrieval across all five contexts?
- Interpretation boundary:
  this is end-to-end Config B with `top_k=2`, not the same frozen bank queried
  with one extra slot, because `top_k` affects both construction and query
  retrieval.
- Configuration:
  `retrieve_threshold=0.03`, `update_threshold=0.09`, `max_slots=16`,
  `top_k=2`, `generation_max_length=40`,
  `eventqa_protocol=frozen_context_bank`, `requested_contexts=5`.
- Artifact:
  `outputs/mab/eventqa_configB_allctx_topk2/20260630T084500Z-eventqa-65536-version-b-weaver-space-bank-n5`

#### Observations

- Global result:
  Bank-off EM `4/500`; Bank-on EM `109/500`; Bank-on recall `0.290`;
  Bank-on format failures `131`; Chinese-script outputs `98`;
  final slots `{15:100,16:400}`.
- Versus Config B `top_k=1`:
  EM `+37`, recall `+0.088`, format failures `-10`, Chinese outputs `-22`.
- Versus Config A `top_k=1`:
  EM `-5`, recall `+0.024`, format failures `+8`, Chinese outputs `+75`.
- Per-context Bank-on EM:
  `19/100`, `45/100`, `22/100`, `22/100`, `1/100`.
  Deltas versus Config B `top_k=1` are `+15`, `+30`, `+20`, `+1`, `-29`.
- Retrieval:
  pairs `{(1,0):399,(0,1):101}`; candidate slots
  `{16:400,15:100}`; retrieved latent count `{16:500}`. Routing remained
  fixed per context except for one pair-order swap in context 1.
- Integrity:
  one host-access GPU-2 launch, no failed attempt, `500/500` valid questions,
  matching manifest/runtime config, total/max query write delta `0/0`, blocked
  query writes `500`, changed snapshots `0`, leakage `0`, errors `0`.

#### Context-4 Failure Analysis

- Exact baselines:
  Config A `top_k=1` reached `16/100` EM, recall `0.26`, 56 format failures,
  and 22 Chinese outputs; Config B `top_k=1` reached `30/100` EM, recall
  `0.30`, 5 format failures, and 1 Chinese output.
- Config B `top_k=2` retained recall `0.30` but collapsed to `1/100` EM with
  94 format failures and 83 Chinese outputs.
- Chinese/format intersection is `82`; 29 format failures still contain the
  full gold answer in raw output. A malformed first-line prefix is parsed as
  the answer, explaining the recall/EM gap.
- The ctx4 pair `(1,0)` is returned for all 100 queries. Final-score top-1/top-2
  margins have mean `0.01382` and range `0.01208-0.01626`.
- Reconstructed raw cosine also ranks local slots `(1,0)` first for all
  queries, so the ordering is not only recency or index tie-breaking.
- Both local slots are repeatedly refreshed during construction: slot 1 has
  `created_step=2`, `access_count=15`, age `0`; slot 0 has
  `created_step=3`, `access_count=14`, age `0`. Construction records 16
  inserts and 1 matched replacement with no eviction.
- Final-slot chunk provenance, slot keys, and query vectors are absent, so the
  current artifacts cannot distinguish pathological slot content from query
  representation collapse.

#### Conclusion

- End-to-end Config B `top_k=2` is a strong positive multi-slot signal because
  gains generalize across contexts 0-3 and overall EM approaches Config A.
- It does not replace Config A as the output-stable setting because context 4
  catastrophically regresses and global Chinese outputs remain much higher.
- `top_k=4` remains deferred.
- Primary follow-up: add score-decomposition and slot/chunk-provenance
  diagnostics, then rerun Config B `top_k=2` on context 4 only before changing
  retrieval behavior.

### EXP-20260630-003: EventQA Config B `top_k=2` ctx4 Standalone Reproducibility Diagnostic

- Phase: EventQA exploratory reproducibility diagnostic
- Status: `completed_diagnostic_only`
- Research question:
  does the catastrophic ctx4 collapse from the accepted all-context Config B
  `top_k=2` run reproduce when the same nominal configuration is rerun on
  `context_index=4` only?
- Interpretation boundary:
  this is a standalone reproducibility and stability diagnostic. It does not
  replace the accepted all-context Config B `top_k=2` ablation.
- Configuration:
  `retrieve_threshold=0.03`, `update_threshold=0.09`, `max_slots=16`,
  `top_k=2`, `generation_max_length=40`,
  `eventqa_protocol=frozen_context_bank`, `requested_contexts=1`,
  `context_index=4`.
- Artifact:
  `outputs/mab/eventqa_configB_ctx4_topk2_rerun/20260630T121127Z-eventqa-65536-version-b-weaver-space-bank-n5`

#### Observations

- Integrity:
  `100/100` valid questions; runtime and manifest matched the intended config;
  total / max query write-count delta `0/0`; changed snapshots `0`; leakage
  `0`; errors `0`.
- Result:
  Bank-off EM `1/100`; Bank-on EM `11/100`; Bank-off recall `0.19`;
  Bank-on recall `0.28`; format failures `52`; Chinese outputs `44`;
  final slot count `16`; peak CUDA memory max / mean `11.17 / 9.46 GiB`.
- Versus the accepted all-context Config B `top_k=2` ctx4 result:
  EM `+10`, recall `-0.02`, format failures `-42`, Chinese outputs `-39`.
- Versus Config B `top_k=1` ctx4:
  EM `-19`, recall `-0.02`, format failures `+47`, Chinese outputs `+43`.
- Retrieval changed materially:
  retrieved pairs `{(3,0):84,(0,3):16}` instead of `{(1,0):100}`.
  Top-1/top-2 margins are nearly tied, with mean `0.00123` versus `0.01382`
  in the all-context ctx4 run.
- Construction shape is nominally the same as the accepted all-context ctx4
  run (`16` inserts, `1` matched replacement, no eviction), but the dominant
  local slot pair changed and the bank summary differs.
- Failure profile:
  44 Chinese outputs; 52 format failures; 36 are both Chinese and format
  failures; 18 format failures still contain the full gold answer; 17 cases
  are recall-positive but EM-negative.

#### Conclusion

- The catastrophic ctx4 collapse did not reproduce exactly.
- Config B `top_k=2` ctx4 still underperforms Config B `top_k=1` ctx4 badly,
  but the all-context `1/100` failure mode is not a stable deterministic
  outcome under the same nominal configuration.
- The dominant local slot pair changes across reruns, which points to
  construction-path or routing instability rather than pure same-bank
  generation noise.
- Primary follow-up:
  add score-decomposition and slot/chunk-provenance diagnostics, then rerun
  ctx4 only. Keep `top_k=4` deferred.

### EXP-20260630-004: EventQA End-to-End Config B `top_k=4` Negative Ablation

- Phase: EventQA exploratory top-k ablation
- Status: `completed_negative_result`
- Research question:
  does Config B remain competitive when the mechanism uses `top_k=4` during
  both bank construction and frozen-bank query retrieval across all five
  contexts?
- Interpretation boundary:
  this is a valid end-to-end mechanism test because `top_k` intentionally
  affects both construction-time memory update and query-time retrieval.
  It is not a same-bank query-only comparison.
- Configuration:
  `retrieve_threshold=0.03`, `update_threshold=0.09`, `max_slots=16`,
  `top_k=4`, `generation_max_length=40`,
  `eventqa_protocol=frozen_context_bank`, `requested_contexts=5`.
- Artifact:
  `outputs/mab/eventqa_configB_allctx_topk4/20260630T124028Z-eventqa-65536-version-b-weaver-space-bank-n5`

#### Observations

- Integrity:
  `500/500` valid questions; total / max query write-count delta `0/0`;
  blocked query write attempts `500`; changed snapshots `0`; leakage `0`;
  errors `0`.
- Global result:
  Bank-off EM `4/500`; Bank-on EM `63/500`; Bank-off recall `0.178`;
  Bank-on recall `0.164`; format failures `208`; Chinese outputs `156`;
  final slots `{15:100,16:400}`.
- Retrieval:
  tuples `{(0,1,2):300,(1,0,2):100,(2,4,10):100}`;
  top-1 `{0:300,1:100,2:100}`;
  top-2 `{1:300,0:100,4:100}`;
  top-3 `{2:400,10:100}`;
  top-4 `{}`;
  retrieved latent count `{24:500}`;
  candidate slots before top-k `{15:100,16:400}`.
- Key mechanism finding:
  although `top_k=4` was requested, thresholded retrieval realized only 3
  slots on every query, so retrieved latent count stayed `24` instead of `32`.
- Versus Config A `top_k=1`:
  EM `-51`, recall `-0.102`, format failures `+85`, Chinese outputs `+133`.
- Versus Config B `top_k=1`:
  EM `-9`, recall `-0.038`, format failures `+67`, Chinese outputs `+36`.
- Versus Config B `top_k=2`:
  EM `-46`, recall `-0.126`, format failures `+77`, Chinese outputs `+58`.
- Per-context Bank-on EM:
  `2/100`, `4/100`, `21/100`, `16/100`, `20/100`.
  Deltas versus Config B `top_k=2` are `-17`, `-41`, `-1`, `-6`, `+19`.

#### Conclusion

- `top_k=4` is a negative end-to-end ablation result.
- It is worse than Config B `top_k=2`, worse than Config B `top_k=1`, and far
  worse than Config A `top_k=1`.
- It increases format failures and Chinese outputs while keeping routing fixed
  per context.
- It should not be kept as a candidate setting.
- Do not scale `top_k` further.
- Primary follow-up:
  return to score-decomposition and slot/chunk-provenance diagnostics,
  especially for unstable contexts `ctx1` and `ctx4`.
