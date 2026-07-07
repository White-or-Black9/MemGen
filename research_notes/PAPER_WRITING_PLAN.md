# Paper Writing Roadmap

Date: 2026-07-05

`paper/outline.md` fixes the paper framing and section structure. This roadmap
maps verified evidence and unresolved experiments into that outline.

## 1. Working Title

**Inference-Time Latent Memory Management for Long-Horizon LLM Agents**

## 2. Abstract-Level Thesis

Long-horizon LLM agents need to preserve and selectively reuse historical
information across inference steps. MemGen generates latent representations
but lacks an explicit session-local mechanism for managing them over a
session. We add a bounded Weaver-space latent memory bank with write,
retrieval, update, replacement, and reset operations. Frozen P7 provides the
current positive evidence on EventQA-65536 without retraining Trigger, Weaver,
or Reasoner; this evidence remains benchmark-scoped and context-dependent.

## 3. Outline-Aligned Contributions

1. An inference-time latent memory management mechanism for MemGen-style LLM
   agents.
2. A session-local latent memory bank with explicit write, retrieval, update,
   replacement, and reset operations.
3. Evaluation on long-context reasoning tasks covering task performance and
   internal memory behavior; EventQA-65536 is the current positive evidence.

The frozen-bank protocol, five-repeat P7 comparison, and failure analysis are
evidence supporting these contributions rather than separate top-level
contributions.

## 4. Research Questions

- RQ1: disabled-path compatibility.
- RQ2: meaningful memory lifecycle behavior during inference.
- RQ3: long-context reasoning benefit from session-level latent reuse.
- RQ4: effects of thresholds, top-k, capacity, and replacement policy.

## 5. Method Section Plan

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

## 6. Experiment Section Plan

### 5.1 Benchmark and task contract

- EventQA-65536 is the main benchmark.
- Each context is about 64k tokens, chunked by the local official MAB path.
- Each query exposes an event prefix and six candidate next events.
- Primary metric is official normalized substring EM; recall and format
  failures are supporting metrics.

### 5.2 Comparators

- Disabled / compressed Bank-off.
- Same-model text-summary memory.
- BM25 top-2 retrieved-text/RAG.
- 16-token matched-budget retrieved text.
- P6 lower-update-threshold comparator.
- P7-no-query-retrieval.
- Frozen P7.

### 5.3 Repetition and reporting

- P7 and P6 use five repeats over the same five contexts / 500 questions.
- Report mean, population standard deviation, range, and per-context values.
- Do not average incomparable historical configurations into the main table.

### 5.4 Diagnostic benchmark

- If retained, mention LoCoMo only after the main EventQA evaluation.
- Explain its different open-ended latent-only query contract.
- Present it as a limitation boundary, not a generalization result.

## 7. Main Table Plan

Rows:

1. Disabled / Bank-off.
2. Text-summary memory.
3. BM25 top-2 retrieved-text/RAG.
4. 16-token matched-budget retrieved text.
5. P6.
6. P7 no-query-retrieval.
7. Frozen P7.

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

## 8. Figure Plan

1. Method figure: frozen MemGen plus session-local bank lifecycle and
   construction/query boundary.
2. Main result figure: main EventQA comparison package with repeat boundaries
   clearly labeled.
3. Context figure: per-context P7/P6/Bank-off results highlighting context 4.
4. Transition/failure figure: helpful, harmful, format-harm, parser-sensitive,
   and no-gold outcomes.
5. Optional cost-effectiveness figure after method-separable costs and explicit
   text baselines are complete.

## 9. Analysis Section Plan

- P7 versus P6 threshold sensitivity.
- Helpful versus harmful Bank-off-to-Bank-on transitions.
- Retrieval-active versus correct/incorrect output behavior.
- Prompt/format sensitivity using strict and first-line negative ablations.
- No-gold versus parser-sensitive failures.
- Context-4 fixed routing and harmful tuple attribution.
- Explicit-text versus latent evidence using the completed control package.
- Cost and memory analysis using the separable instrumentation package and the
  text-summary caveat.

## 10. Limitation Section Plan

- EventQA is one closed-set next-event benchmark, not general proof.
- The query exposes prior events and six candidate answers.
- P7 is not uniformly beneficial; context 4 remains near zero EM.
- Harmful tuple attribution is single-bank oracle evidence, not a deployed fix.
- LoCoMo has no positive exact QA result despite active retrieval.
- Latent-only evidence is difficult to decode into exact dates, people,
  preferences, and cross-session relations.
- Text-summary cost is diagnostic-only because it was measured under shared-GPU
  contention.

## 11. Appendix Plan

- Exact P7 configuration and lifecycle pseudocode.
- Prompt templates and local official scorer verification.
- Per-repeat and per-context EventQA tables.
- P4 and historical sensitivity configurations with comparability labels.
- Strict/first-line prompt ablations.
- No-query-retrieval and optional no-update ablations when available.
- Failure examples and full context-4 diagnostics.
- Harmful attribution conditions and single-bank caveat.
- LoCoMo pipeline, prompt, output-contract, answer, and diagnostics audits,
  packaged in Appendix A with unreliable construction-side cost fields
  excluded.
- Reproducibility metadata and extended cost accounting.

## 12. Results Ready To Write

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

## 13. Results Not Ready Yet

- No required result row remains missing for the current scoped draft.
- A standalone cost-effectiveness figure is optional; Table 2 already records
  the paper-facing cost evidence.

## 14. Required Remaining Experiments

No new EventQA experiment is currently required for the main comparison package.

The required packaging and writing are complete:

1. EventQA main/evidence tables are rendered in the draft.
2. Paper-facing cost rows are fixed in Table 2; text-summary cost remains
   excluded because of shared-GPU contention.
3. The claim audit is reflected in the abstract, results, limitations, and
   conclusion.

No P7/P6 five-repeat effectiveness rerun is required by default.

## 15. Recommended Experiment Order

1. Completed: no-inference effectiveness aggregator.
2. Completed: method-separable cost smoke and full five-context pass.
3. Completed: BM25 top-2 RAG baseline.
4. Completed: 16-token matched-budget RAG baseline.
5. Completed: same-model text-summary baseline.
6. Completed: P7-no-query-retrieval ablation.
7. Completed: build unified final comparison package and claim audit.
8. Completed: turn the package into paper tables/figures and scoped manuscript
   claims.
9. Deferred by user choice: independent skeptical review.

## 16. Stop Conditions For Dropping Or Demoting Evidence

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
