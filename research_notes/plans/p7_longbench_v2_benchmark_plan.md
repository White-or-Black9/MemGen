# P7 LongBench v2 Benchmark Plan

Date: 2026-07-12 (closed 2026-07-18)
Status: closed as a protocol-clean negative retrieval diagnostic; no fallback benchmark is active

## Objective

Test whether the frozen P7 session-local latent bank generalizes beyond
EventQA to realistic long-context reasoning without changing P7, retraining
MemGen, or weakening the existing paper evidence boundary.

LongBench v2 is the first candidate because it combines realistic long
contexts with a uniform four-choice answer contract and deterministic
accuracy. This benchmark can add long-context generalization evidence, but it
cannot by itself establish multi-session conversational memory.

## Current Decision

- Reopen benchmark planning only.
- Keep EventQA as the sole current positive paper anchor.
- Select a stratified LongBench v2 subset as the next candidate.
- Do not implement a runner, download data, or launch GPU work until the
  corresponding phase is explicitly approved.
- Do not tune P7 thresholds or other method settings against LongBench v2.

## Historical Candidate Order

1. LongBench v2 stratified subset: primary next trial.
2. BABILong QA1-QA5: controlled-length fallback if LongBench v2 is invalid or
   mechanism-negative.
3. MemBench: schema and scorer audit before any implementation decision.
4. InfiniteBench deterministic subset: optional super-long-context stress
   test.
5. LongBench v1 selective: low-cost sanity check only, not a main paper row.

RULER-QA2, FactConsolidation, and DetectiveQA remain closed under their latest
decisions. LongMemEval, LongMemEval-V2, MemoryArena, and EvoMemBench are outside
this bounded campaign because their judge, trajectory, multimodal, or agentic
execution requirements do not fit a low-risk frozen-P7 adapter.

By DEC-0089, the listed fallback candidates are paused. They must not be
started automatically from this historical plan.

## Non-Negotiable Method Boundary

- Frozen P7 only:
  `retrieve_threshold=0.05`, `update_threshold=0.10`, `max_slots=16`,
  `top_k=2`, `decay_alpha=0.05`, Weaver-space storage, no fallback top-1.
- Trigger, Weaver, and Reasoner remain frozen.
- Batch size remains `1`.
- One LongBench v2 item owns one session-local bank.
- The bank resets before construction and after the item.
- Context is ingested in ordered, sentence-preserving chunks.
- The bank is snapshotted after construction.
- Query-time retrieval is allowed for P7, but query-time writes are blocked.
- `p7_no_query_retrieval` must use identical construction and differ only by
  disabling query retrieval.
- No cross-item memory sharing is allowed.

## Evaluation Methods

Required in the first effectiveness comparison:

1. `disabled_window_fit`: original full-context behavior, only on examples
   whose rendered prompt fits the verified Reasoner capacity.
2. `p7`: frozen construction plus query-time latent retrieval.
3. `p7_no_query_retrieval`: identical construction with retrieval disabled at
   question time.
4. `bm25_top_k`: deterministic retrieved-text baseline using the same source
   chunks.
5. `matched_text_budget`: retrieved text truncated to a declared budget that
   matches the P7 query-time latent injection budget as closely as the current
   tokenizer contract permits.

For over-capacity items, do not report a query-only or truncated Disabled line
as if it were a full-context baseline. Record it separately as
`disabled_over_capacity` and compare P7 primarily against the explicit-memory
controls and the no-query ablation.

## Sample Scope

### Included Domains

- long-dialogue history understanding;
- multi-document QA;
- long structured-data understanding.

### Deferred Domains

- code-repository understanding, because code ability is a major confound;
- long in-context learning, because test-time task induction is not the
  current memory claim;
- single-document QA unless needed to fill a length or difficulty stratum.

### Length and Difficulty Strata

- use the official `short`, `medium`, and `long` labels;
- independently record rendered token counts with the frozen MemGen tokenizer;
- prioritize samples above 32K rendered tokens while keeping a small
  window-fitting control stratum;
- cap the first smoke and bounded manifests at 262,144 rendered ChatML tokens;
  retain larger rows in the audit inventory but defer them as extreme-cost
  stress cases;
- balance `easy` and `hard` where the selected domains permit it.

The smoke should contain 18 fixed examples, with six per included domain. The
bounded manifest should contain 60 fixed examples: 24 multi-document, 18
long-dialogue, and 18 structured-data rows. IDs must be frozen before GPU
execution.

## Phase 0: Read-Only Dataset Audit

Goal: determine whether LongBench v2 has a valid, reproducible local evaluation
contract before implementation.

Tasks:

1. Obtain the official dataset through its documented Hugging Face source.
2. Record revision, file hashes, license, row count, and schema.
3. Validate unique IDs, answer membership in `A/B/C/D`, and non-empty context,
   question, and choices.
4. Inventory domain, sub-domain, difficulty, official length label, word count,
   and MemGen rendered token count.
5. Produce the proposed smoke and bounded-evaluation ID manifests.
6. Confirm that every selected item can be deterministically scored without an
   LLM judge or manual adjudication.

Required artifacts:

- `outputs/longbench_v2/dataset_audit.json`;
- `outputs/longbench_v2/dataset_audit.md`;
- `configs/eval/longbench_v2_p7_smoke_ids.json`;
- `configs/eval/longbench_v2_p7_bounded_ids.json`.

Phase 0 passes only if the official dataset identity, license, schema, answer
contract, and selected IDs are all frozen and reproducible.

## Phase 1: Adapter and Scorer Contract

Start only after Phase 0 approval.

Tasks:

1. Add a dataset adapter that loads only frozen IDs and emits normalized item
   records.
2. Add deterministic option extraction with strict `A/B/C/D` scoring and a
   separately reported relaxed option-text diagnostic.
3. Add ordered context chunking with exact source offsets and tokenizer counts.
4. Add runner helpers for reset, construction, snapshot/restore, read-only
   query execution, and post-item reset.
5. Add tests for dataset identity, scorer behavior, chunk reconstruction,
   method comparability, query-write blocking, and snapshot invariance.

No GPU inference is part of Phase 1. Stop after tests and a no-model fixture
run, then request approval for the smoke.

## Phase 2: Three-Method Smoke

Run the frozen 12-20 item smoke first with:

- `disabled_window_fit` where valid;
- `p7`;
- `p7_no_query_retrieval`.

Required metrics and diagnostics:

- strict option accuracy and invalid-output count;
- per-domain, per-length, and per-difficulty counts;
- construction chunks, writes, updates, replacements, and final slots;
- retrieval-positive queries, selected slots, and retrieved latent count;
- query-write count and bank snapshot equality;
- construction, query, and end-to-end latency;
- peak GPU memory if the existing instrumentation supports a comparable
  per-method measurement.

## Smoke Promotion Gate

Promote only if all conditions hold:

- every method covers the same frozen item IDs applicable to that comparator;
- option extraction is deterministic and invalid-output behavior is explained;
- all P7 items have nonzero construction writes;
- P7 has query-time retrieval on at least one item and retrieval is not
  systematically zero within every included domain;
- every query has `query_write_count=0` and an unchanged frozen snapshot;
- every item resets to an empty bank after completion;
- P7 construction exactly matches the no-query ablation;
- no hidden response-length, prompt, scorer, or context-budget mismatch exists.

Accuracy on 12-20 items is a directional signal only. It is not a paper result
and is not sufficient by itself for promotion.

## Stop Conditions

Stop LongBench v2 without threshold tuning if any of the following occurs:

- official data or license cannot be frozen reproducibly;
- deterministic option scoring cannot be maintained;
- selected methods do not receive aligned questions and choices;
- the adapter silently truncates construction context;
- P7 construction writes are absent;
- P7 retrieval is zero across the smoke, repeating the RULER mechanism failure;
- query writes occur or snapshots change;
- the observed comparison depends on an invalid over-capacity Disabled line;
- prompt or output-format failure dominates the result and cannot be repaired
  locally without changing the method.

## Phase 3: Explicit-Memory Controls

If the smoke passes, add BM25 and matched-budget retrieved-text controls on the
same frozen smoke IDs. Reuse EventQA accounting conventions where valid, but do
not reuse EventQA prompts, chunks, or metrics without a LongBench-specific
contract test.

Promote to a 60-100 item bounded run only if P7 retrieval remains active and
the comparison is protocol-clean after adding these controls.

## Phase 4: Bounded Evaluation and Decision

Run all approved methods on the frozen bounded ID manifest. Aggregate overall
accuracy plus domain, length, difficulty, capacity, retrieval, format, latency,
and failure-mode slices.

Possible decisions:

- `promote_appendix`: protocol-clean evidence that supports a narrow
  LongBench-v2 generalization statement;
- `retain_diagnostic`: mechanically valid but weak, mixed, or underpowered;
- `drop_candidate`: mechanism-negative, comparator-invalid, or dominated by
  format failure.

Do not modify `paper/` until this decision is recorded and a separate paper
update is approved.

## Fallback Gates

### BABILong

Use QA1-QA5 at controlled 8K, 16K, 32K, and 64K lengths. Require deterministic
exact-answer scoring and nonzero P7 retrieval. Its role is controlled mechanism
evidence, not realistic-agent evidence.

### MemBench

Perform a CPU-only audit of download provenance, license, schema, official
scorer, participation/observation splits, factual/reflective splits, and noise
length construction. Do not implement P7 until the audit proves a clean
question-answer interface that does not require explicit textual-memory
operations unavailable to P7.

### InfiniteBench

Consider only deterministic short-output tasks such as `En.MC`, `En.Dia`, and
`Retrieve.KV`. Exclude summarization and long arithmetic. Treat it as a
super-long stress test rather than an agent-memory benchmark.

### LongBench v1

Use only a small LongBench-E subset such as passage retrieval/counting and one
multi-document QA task. Do not run the full 4,750-example suite or present it
as the primary new benchmark.

## Final Closeout (2026-07-18)

Phase 0 and Phase 1 completed on 2026-07-12. Dataset provenance, frozen
manifests, deterministic adapter/scorer contracts, chunk reconstruction, and
no-model lifecycle validation are complete. Phase 2 model-facing integration
and the frozen 18-item smoke also completed. The initial 60-item recovery is
historical; the final valid run used the shared `constrained_choice_v3`
protocol and merged at:
`outputs/longbench_v2/constrained_choice_v3_comparison_merged/20260718T035900Z/artifact.json`.

The merged `60`-item comparison is complete and contract-valid: P7 retrieval
is positive on `60/60`, no-query retrieval is zero, all outputs are valid, and
P7 versus no-query is `17/60` versus `17/60` with paired wins/losses/ties
`5/5/50` (two-sided exact sign-test `p=1.0`). The 11-item window-fit Disabled
slice is `1/11`. Therefore this plan is closed: do not run BM25, matched-text
controls, seed expansion, a larger LongBench v2 sample, or a fallback candidate
under this campaign. EventQA supplementary planning is now the only active
research-planning route.
