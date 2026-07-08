# Memory Benchmark Campaign Design

## 1. Objective

Extend the frozen P7 evidence package from its current EventQA anchor into a
controlled benchmark campaign that evaluates four distinct claims:

1. event and historical-information reuse;
2. fact consolidation, update, and replacement;
3. long-range clue integration under constrained memory capacity;
4. robustness relative to explicit-text and disabled-memory controls.

The campaign includes benchmark integration, smoke validation, full experiment
execution, aggregation, statistical analysis, and paper-table updates.

## 2. Fixed Method and Safety Boundary

The campaign evaluates the current frozen P7 method without modifying its model
or training behavior:

- session-local Weaver-space latent memory bank;
- `retrieve_threshold=0.05`;
- `update_threshold=0.10`;
- `max_slots=16`;
- `top_k=2`;
- `decay_alpha=0.05`;
- no cross-sample memory;
- enabled inference uses `batch_size=1`;
- no Weaver, Trigger, or Reasoner retraining;
- Disabled uses the original `latent_memory_bank=None` path;
- each benchmark context owns one bank and resets it at context completion or
  failure.

Benchmark adapters, prompts, parsers, scorers, configs, and aggregators remain
outside the training path. Any proposed method change discovered during the
campaign is recorded as future work rather than introduced into P7.

## 3. Benchmark Roles

### 3.1 EventQA: Main Benchmark 1

EventQA remains the primary positive benchmark. Existing validated P7,
Disabled, P6, text-summary, BM25, matched-budget, no-query-retrieval, cost, and
diagnostic artifacts are reused. They are not rerun unless an integrity audit
finds a concrete comparability defect.

EventQA supports claims about event-memory reuse and query-time latent
retrieval. It is also the primary platform for retrieval-component ablations:
top-k, retrieval threshold, and query-retrieval removal. New ablations must use
the existing frozen-context protocol and official scorer.

### 3.2 FactConsolidation: Main Benchmark 2 Candidate

FactConsolidation is the next implementation priority. The campaign begins with
SH and MH 6K smoke cases, then advances to 32K and 64K only after protocol and
metric validation.

It evaluates:

- accumulation of facts across ordered chunks;
- replacement of stale or conflicting facts;
- matched-thread update behavior;
- capacity eviction;
- retrieval and update threshold separation.

FactConsolidation enters the main table only if the final evaluation has enough
independent units for a defensible aggregate. If the released/local data expose
too few independent contexts, it becomes a mechanism/ablation table instead of
being presented as broad performance evidence.

### 3.3 DetectiveQA: Stress Test

DetectiveQA measures distributed clue integration and long-range reasoning. It
uses the existing compressed-memory protocol because the original full-history
inputs exceed the checkpoint capacity. Over-capacity full-history outputs are
never scored or treated as a baseline.

DetectiveQA is reported as a stress test regardless of whether P7 improves task
accuracy. Its primary outputs are accuracy, length/capacity breakdowns,
retrieval activation, slot utilization, and failure analysis.

### 3.4 BABILong: Conditional Optional Test

BABILong is not implemented initially. It is activated only if DetectiveQA
cannot yield a valid length-versus-reasoning stress curve or lacks sufficient
task diversity. BABILong would then provide controlled fact-chain and context
length scaling, not session-memory evidence.

## 4. Experimental Controls

Every new benchmark must include, where technically meaningful:

- Disabled: original MemGen path with no bank object;
- frozen P7;
- P7 without query-time retrieval;
- an explicit-text retrieval baseline when a fair source-text mapping exists;
- a matched-budget text baseline when latent-token and rendered-token budgets
  can be made comparable.

Mechanism ablations are split between benchmarks:

- EventQA: retrieval removal, top-k, and retrieval threshold;
- FactConsolidation: update threshold, capacity, decay, and replacement policy;
- DetectiveQA: capacity and context-length stress.

Random retrieval is included only as a bounded diagnostic because it changes
the inference input distribution and is not a production baseline.

## 5. Evaluation and Logging Contract

### 5.1 Task Metrics

- EventQA: official substring exact match, EventQA recall, format failures.
- FactConsolidation: official substring exact match, separated by SH/MH,
  context length, and conflict/update structure.
- DetectiveQA: official exact match, with normalized or substring checks marked
  diagnostic-only.
- BABILong, if activated: official per-task accuracy and length curves.

### 5.2 Memory Metrics

Each enabled run records:

- write attempts and successful writes;
- retrieval calls and non-empty retrievals;
- retrieved slot and latent-token counts;
- insert, matched-replace, and capacity-eviction counts;
- final and peak slot occupancy;
- unique-slot utilization and access concentration;
- query-time writes;
- reset success and cross-context leakage checks;
- construction, query, and amortized end-to-end latency;
- incremental peak GPU allocation.

Similarity-threshold pass rate is not called memory hit rate. Evidence hit rate
is reported only when source-chunk provenance allows a retrieved slot to be
matched to benchmark gold evidence.

## 6. Phase Gates

### Phase 0: Contract Freeze

Audit current branch state, P7 settings, existing EventQA artifacts, official
benchmark configs, dataset counts, metrics, and protected files. Produce a
row-level target matrix before implementation.

Exit gate: every planned row has a benchmark split, method, metric, repeat
count, artifact schema, and paper destination.

### Phase 1: FactConsolidation Adapter and Smoke

Implement a thin evaluation-only adapter and scorer path for SH/MH 6K. Validate
chunk order, prompts, answer extraction, official scoring, bank reset, Disabled
equivalence, query-read-only behavior, and provenance logging.

Exit gate: paired Disabled/P7 smoke artifacts pass schema and invariant checks.

### Phase 2: FactConsolidation Scale Decision

Audit independent context/query counts and run SH/MH 32K and 64K only after
capacity preflight. Determine whether the benchmark can support a main-table
aggregate or must remain a mechanism table.

Exit gate: explicit GO to main-table evaluation or documented downgrade to
ablation/mechanism evidence.

### Phase 3: FactConsolidation Full Runs and Ablations

Run the accepted lengths and repeats, followed by update-threshold, capacity,
decay, and replacement-policy ablations. Full runs use separate processes and
serialized or contention-controlled GPU assignment.

Exit gate: complete artifacts, reproducible aggregates, no leakage, and a
stable interpretation of update/replacement behavior.

### Phase 4: EventQA Evidence Reuse and Targeted Ablations

Verify existing canonical rows and run only explicitly missing ablations. Do
not rerun the validated five-repeat P7/Disabled package by default.

Exit gate: final EventQA main and ablation rows are complete and traceable to
canonical artifacts.

### Phase 5: DetectiveQA Stress Evaluation

Run bounded compressed-memory stress evaluations by context length, clue
structure, and capacity. Preserve the over-capacity full-history condition as
invalid rather than truncating it silently.

Exit gate: a valid stress curve or a documented decision to activate BABILong.

### Phase 6: Optional BABILong

If activated, select a small set of fact-chain tasks and at least three context
lengths. Use the same Disabled/P7 and budget-control principles.

Exit gate: controlled length curve with official accuracy, or a documented
NO-GO without paper-facing claims.

### Phase 7: Unified Analysis

Aggregate task, memory, cost, and invariant metrics. Separate primary results,
mechanism evidence, stress evidence, and negative boundaries. Use paired
analysis where methods share question identities and report uncertainty across
independent contexts or repeats.

Exit gate: all paper-facing numbers reproduce from machine-readable aggregate
artifacts.

### Phase 8: Paper Update

Update the manuscript only after Phase 7 passes:

- main effectiveness table: EventQA and eligible FactConsolidation rows;
- ablation table: EventQA retrieval and FactConsolidation update/replacement;
- stress/appendix table: DetectiveQA and optional BABILong;
- method and evaluation sections;
- limitations and claim boundary;
- references and artifact provenance.

No benchmark is promoted from appendix to the main table solely because its
runner completed.

## 7. Decision Rules

- A smoke success proves pipeline validity, not method effectiveness.
- A single-context FactConsolidation result cannot support a general benchmark
  claim.
- DetectiveQA remains stress evidence even if it yields a positive delta.
- Long-context and memory-management effects are separated with Disabled,
  explicit-text, matched-budget, and no-retrieval controls.
- Negative or zero results are retained when the protocol is valid.
- Any model or training change requires a separate design and cannot be folded
  into this campaign.

## 8. Deliverables

The completed campaign produces:

1. benchmark adapters and configs confined to evaluation surfaces;
2. tests for loaders, prompts, scorers, reset, and aggregation;
3. smoke and full-run artifacts with invariant diagnostics;
4. canonical JSON and Markdown aggregate packages;
5. a final benchmark decision matrix;
6. updated main, ablation, stress, and appendix tables;
7. revised manuscript claims and limitations tied to validated evidence.

## 9. Execution Order

The required order is:

`contract freeze -> FactConsolidation smoke -> scale decision -> FactConsolidation full/ablation -> EventQA targeted completion -> DetectiveQA stress -> optional BABILong -> unified analysis -> paper update`.

Only one phase is active at a time. Completion and evidence are reported before
the next phase begins.
