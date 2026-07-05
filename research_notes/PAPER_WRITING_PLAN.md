# Paper Writing Roadmap

Date: 2026-07-05

This is the authoritative writing roadmap for the current experiment package.
It complements `research_notes/PAPER_SCOPE.md`: that file fixes the claim;
this file maps evidence into manuscript sections and identifies writing gates.

## 1. Working Title

**Session-Local Latent Memory Banks for Long-Context Reasoning in MemGen**

## 2. Abstract-Level Thesis

MemGen can generate and consume latent representations but lacks an explicit
session-local mechanism for retaining and reusing useful latent context across
a long frozen episode. We add a bounded Weaver-space latent memory bank at
inference time. Without retraining Trigger, Weaver, or Reasoner, frozen P7
improves EventQA-65536 long-context next-event reasoning over the compressed
Bank-off path. The result is scoped to EventQA and is limited by severe
context-specific failures and weak exact conversational fact recovery on a
LoCoMo diagnostic.

## 3. Main Contributions

1. An inference-time session-local latent memory bank for MemGen with bounded
   write, retrieve, matched update, replacement, reset, and query-time
   read-only behavior.
2. A frozen-context evaluation protocol that reuses one context-local bank
   across questions while preventing query-time writes and cross-sample memory
   sharing.
3. Five-repeat EventQA evidence showing higher P7 EM and fewer format failures
   than P6, and a large gain over compressed Bank-off.
4. Failure analysis showing nonuniform context behavior, memory-conditioned
   generation corruption, and a harmful tuple interaction on one frozen
   context-4 bank.
5. A diagnostic boundary result showing that active latent retrieval does not
   by itself solve exact multi-session conversational fact recovery.

## 4. Method Section Plan

### 4.1 Frozen MemGen components

- Define Trigger, Weaver, and Reasoner roles.
- State that all pretrained components remain frozen.
- Clarify that P7 is an inference-time addition, not a new training recipe.

### 4.2 Weaver-space session-local bank

- Define one bank per session/context and no cross-sample sharing.
- Describe eight latent vectors per slot, bounded capacity 16, and top-2 query
  retrieval.
- Explain Reasoner-only consumption of retrieved memory.

### 4.3 Lifecycle and policies

- Write, query-key construction, thresholded retrieval, matched update,
  insertion, capacity replacement, and reset.
- Report P7 thresholds and decay exactly.
- Keep implementation symbols and local paths in the appendix.

### 4.4 Frozen-bank query protocol

- Construct once from the long EventQA context.
- Reuse the same frozen snapshot across questions.
- Block query writes and verify snapshot invariance.
- Distinguish construction-time retrieval/update from query-time read-only use.

## 5. Experiment Section Plan

### 5.1 Benchmark and task contract

- EventQA-65536 is the main benchmark.
- Each context is about 64k tokens, chunked by the local official MAB path.
- Each query exposes an event prefix and six candidate next events.
- Primary metric is official normalized substring EM; recall and format
  failures are supporting metrics.

### 5.2 Comparators

- Disabled / compressed Bank-off.
- P6 lower-update-threshold comparator.
- Frozen P7.
- Pending: text-summary, BM25 top-2 RAG, 16-token matched-budget RAG, and
  P7-no-query-retrieval.

### 5.3 Repetition and reporting

- P7 and P6 use five repeats over the same five contexts / 500 questions.
- Report mean, population standard deviation, range, and per-context values.
- Do not average incomparable historical configurations into the main table.

### 5.4 Diagnostic benchmark

- Mention LoCoMo only after the main EventQA evaluation.
- Explain its different open-ended latent-only query contract.
- Present it as a limitation boundary, not a generalization result.

## 6. Main Table Plan

Rows:

1. Disabled / Bank-off.
2. Text-summary memory.
3. BM25 top-2 retrieved-text/RAG.
4. 16-token matched-budget retrieved text.
5. P6.
6. Frozen P7.

Effectiveness columns:

- repeats;
- EM;
- recall;
- format failures;
- helpful and harmful transitions.

Cost columns or aligned cost table:

- construction latency;
- query generation latency;
- end-to-end latency;
- peak GPU memory;
- query evidence positions/tokens;
- output tokens.

Do not combine effectiveness and cost if the latter remains method-inseparable.

## 7. Figure Plan

1. Method figure: frozen MemGen plus session-local bank lifecycle and
   construction/query boundary.
2. Main result figure: Bank-off, P6, and P7 EM/recall with repeat dispersion.
3. Context figure: per-context P7/P6/Bank-off results highlighting context 4.
4. Transition/failure figure: helpful, harmful, format-harm, parser-sensitive,
   and no-gold outcomes.
5. Optional cost-effectiveness figure after method-separable costs and explicit
   text baselines are complete.

## 8. Analysis Section Plan

- P7 versus P6 threshold sensitivity.
- Helpful versus harmful Bank-off-to-Bank-on transitions.
- Retrieval-active versus correct/incorrect output behavior.
- Prompt/format sensitivity using strict and first-line negative ablations.
- No-gold versus parser-sensitive failures.
- Context-4 fixed routing and harmful tuple attribution.
- Explicit-text versus latent evidence after RAG/summary rows exist.
- Cost and memory analysis only after separable instrumentation.

## 9. Limitation Section Plan

- EventQA is one closed-set next-event benchmark, not general proof.
- The query exposes prior events and six candidate answers.
- P7 is not uniformly beneficial; context 4 remains near zero EM.
- Harmful tuple attribution is single-bank oracle evidence, not a deployed fix.
- LoCoMo has no positive exact QA result despite active retrieval.
- Latent-only evidence is difficult to decode into exact dates, people,
  preferences, and cross-session relations.
- Current cost fields from paired EventQA artifacts are not method-separable.

## 10. Appendix Plan

- Exact P7 configuration and lifecycle pseudocode.
- Prompt templates and local official scorer verification.
- Per-repeat and per-context EventQA tables.
- P4 and historical sensitivity configurations with comparability labels.
- Strict/first-line prompt ablations.
- No-query-retrieval and optional no-update ablations when available.
- Failure examples and full context-4 diagnostics.
- Harmful attribution conditions and single-bank caveat.
- LoCoMo pipeline, prompt, output-contract, answer, and diagnostics audits.
- Reproducibility metadata and extended cost accounting.

## 11. Results Ready To Write

- P7 method and frozen configuration.
- EventQA benchmark and frozen-context protocol.
- P7 five-repeat effectiveness result.
- P7 versus P6 comparison.
- Prompt/scorer validation.
- Context-wise results.
- Format-failure and no-gold analysis.
- Context-4 limitation.
- Harmful tuple attribution with oracle/single-bank caveat.
- LoCoMo limitation interpretation.

## 12. Results Not Ready Yet

- Final EventQA main comparison table.
- Method-separable latency and peak-memory claims.
- Text-summary comparison.
- BM25/RAG comparison.
- Matched-budget comparison.
- No-query-retrieval component attribution.
- Final cost-effectiveness figure.
- Final unified artifact/table package.

## 13. Required Remaining Experiments

1. Standalone Bank-off and P7 cost measurement.
2. Text-summary memory baseline.
3. Deterministic BM25 top-2 retrieved-text baseline.
4. Query-position-matched 16-token retrieved-text baseline.
5. P7-no-query-retrieval ablation.

No P7/P6 five-repeat effectiveness rerun is required by default.

## 14. Recommended Experiment Order

1. Freeze final schemas and build a no-inference effectiveness aggregator.
2. Validate method-separable cost instrumentation on EventQA context 0,
   questions 0-9, Disabled and P7.
3. Run the full comparable cost pass.
4. Smoke and run BM25 top-2 RAG.
5. Run 16-token matched-budget RAG.
6. Freeze and run the text-summary baseline.
7. Run P7-no-query-retrieval.
8. Build unified tables, figures, and final claim audit.

## 15. Stop Conditions For Dropping Or Demoting Evidence

- Different question, prompt, generation, parser, or scorer contracts are used
  without explicit labeling.
- P7 parameters or frozen components change.
- Query writes are nonzero or the frozen snapshot changes.
- RAG/summary evidence silently truncates or exceeds capacity.
- Cost fields remain combined or are measured under uncontrolled GPU load.
- Required retrieval IDs, prompt lengths, or injected token counts are absent.
- A baseline has incomplete contexts/questions or incompatible output schema.
- A result is single-run or exploratory but is being promoted as repeated main
  evidence.

When a stop condition holds, move the evidence to appendix/diagnostic status or
drop the associated claim. Do not write around the gap.
