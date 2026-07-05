# P7 Consolidated Formal Experiment Plan

Date: 2026-07-04

## Scope

- Planning only
- No GPU runs
- No code implementation
- No dataset downloads
- No P7 changes
- No model-code changes
- No `paper/` changes

## Frozen P7

- `retrieve_threshold = 0.05`
- `update_threshold = 0.10`
- `max_slots = 16`
- `top_k = 2`
- `decay_alpha = 0.05`
- Weaver-space bank path / MAB-6B-style mechanism
- session-local latent memory bank
- no Trigger / Weaver retraining
- no utility gate
- no tuple suppression
- no top-1 fallback
- no cross-sample memory sharing
- query-time writes blocked for frozen-bank QA protocols

## 1. Final Benchmark Decision

### Main Benchmark 1: EventQA

- Role: long-context reasoning anchor
- Why it stays:
  - already integrated in MemGen
  - frozen P7 and P6 five-repeat evidence already exists
  - current table inventory is concrete and reusable
  - provides the strongest current long-context anchor with known cost and failure patterns
- Current status:
  - P7 five-repeat row is reusable now
  - P6 five-repeat row is reusable now
  - P4, strict, first-line, prompt/scorer verification, context-4, and harmful-attribution analyses are already packaged or partially packaged

### Main Benchmark 2: LoCoMo-QA Local Subset

- Role: multi-session conversational-memory anchor
- Why it is selected:
  - true multi-turn / multi-session structure
  - locally available through `/mnt/18T/sunyanjia/AMEM/A-mem-main/data/locomo10.json`
  - has reference QA answers and category labels
  - deterministic exact match and token F1 are available
  - no GPT / LLM judge is required for QA-only scope
  - maps cleanly to frozen P7 at the protocol level
- Why it is better than the other audited candidates:
  - it directly supports the paper title’s conversational-memory side
  - it is locally actionable now
  - it does not require falling back to judge-only scoring

### Deferred: LongMemEval

- Status: deferred, not abandoned
- Why deferred:
  - benchmark fit is excellent
  - available only through MAB-converted local path
  - formal scoring depends on GPT-4o / LLM-as-judge
  - no MemGen-native runner exists yet
- Why not abandoned:
  - it remains the most aligned future expansion benchmark for long-term conversational memory
  - prediction-only and qualitative use remain possible after runner work

### Deferred: MSC-MemFuse-MC10

- Why deferred:
  - not locally available in this audit
  - no resolved schema or evaluator path
  - benchmark identity is not concrete enough to drive implementation

### Supplementary But Not Current Priority

#### RULER-QA2

- Useful for long-context retrieval / QA stress
- Not enough for the missing conversational-memory axis

#### FactConsolidation-SH

- Useful for update / conflict diagnostics
- Not a replacement for multi-session conversational memory

#### DetectiveQA

- Useful for appendix / mechanism evidence
- Not strong enough as the second main paper benchmark

## 2. LoCoMo-QA Implementation Roadmap

### Scope Lock

- QA only
- Exclude:
  - event summarization
  - dialogue generation
  - multimodal generation

### Dataset Path

- Local path:
  - `/mnt/18T/sunyanjia/AMEM/A-mem-main/data/locomo10.json`

### Data Preservation Requirements

- Preserve:
  - conversation/sample IDs
  - question IDs if present or stable derived IDs if absent
  - question text
  - category labels
  - references
  - evidence fields if available

### Metrics

- deterministic exact match
- deterministic token F1

### Frozen P7 Protocol Mapping

- one conversation sample = one session-local latent bank
- sequential construction-time ingestion over all sessions in order
- bank frozen before QA starts
- query-time retrieval allowed
- query-time writes blocked
- enforce `query_write_count == 0`
- support multiple questions over the same frozen bank

### Required Output Behavior

- one output record per question
- one summary record per conversation
- one aggregate summary over the selected subset
- explicit flag for invalid output / parse failure
- category-wise aggregation

## 3. LoCoMo-QA Next Implementation Stages

### Stage A: No-GPU Adapter/Scorer Plan

- define MemGen-side file map
- define local subset loader contract
- define stable output schema
- define deterministic scorer policy
- define frozen-bank multi-question protocol

### Stage B: No-GPU Adapter/Scorer Implementation

- implement loader / adapter
- implement deterministic scorer
- implement output schema
- implement tests for metadata preservation and `query_write_count == 0` accounting

### Stage C: LoCoMo-QA Disabled + P7 Smoke Run

- target:
  - local `locomo10` subset only
  - 1 sample
  - a few representative questions
- required checks:
  - loader correctness
  - bank reset correctness
  - frozen-bank question loop correctness
  - `query_write_count == 0`
  - latency / memory logging

### Stage D: LoCoMo-QA Disabled + P7 Pilot

- modest pilot over the local subset
- validate:
  - score stability
  - category breakdowns
  - invalid output rate
  - cost logging completeness

### Stage E: LoCoMo-QA Baseline Protocol

- run comparable baseline families:
  - Disabled / Bank-off
  - P6
  - P7
  - text-summary memory baseline
  - RAG / retrieved-text baseline
  - matched-budget baseline

### Stage F: LoCoMo-QA Formal Run

- freeze subset, protocol, output schema, and scoring
- run the full selected local subset under the final benchmark stack
- aggregate main table, appendix table, and cost table outputs

## 4. EventQA Remaining Work

- package comparable Disabled / Bank-off latency + peak-memory row
- implement text-summary baseline
- implement RAG / retrieved-text baseline
- implement matched-budget baseline
- build final unified EventQA aggregation table

## 5. Shared Baseline Protocol

### Disabled / Bank-off

- Purpose:
  - rule out gains from the base model alone
  - establish no-memory baseline

### P6

- Purpose:
  - rule out that P7 wins are accidental or only due to variance
  - provide closest frozen-threshold comparator

### P7

- Purpose:
  - final frozen main method

### Text-Summary Memory Baseline

- Purpose:
  - rule out that performance gains come from any compact memory text rather than latent bank structure

### RAG / Retrieved-Text Baseline

- Purpose:
  - rule out that simple retrieved raw text explains the gains
  - compare latent bank against explicit retrieval injection

### Matched-Budget Baseline

- Purpose:
  - rule out a compute/token-budget confound
  - compare under matched injected-text or retrieval budget constraints

### Shared Benchmark Coverage

- Apply the same baseline family to:
  - EventQA
  - LoCoMo-QA
- Keep protocol naming and cost columns aligned across both benchmarks

## 6. Cost Analysis Plan

### Required Metrics

- end-to-end latency
- generation latency if available
- peak GPU memory
- Trigger call count
- Weaver call count
- construction write count
- construction retrieve count if applicable
- query retrieval active count
- retrieved latent count
- final slot count
- `query_write_count`
- output token count
- injected text token count for RAG / summary baselines
- CPU bank-size estimate if available

### Reporting Policy

- report per-question raw metrics
- report per-run aggregated means
- report max peak-memory where relevant
- keep cost tables directly comparable across Disabled, P6, P7, and explicit-text baselines

## 7. Failure Analysis Plan

### EventQA

- per-context EM / recall
- no-gold rate
- format failure
- context-4 diagnostics
- harmful attribution as limitation analysis

### LoCoMo-QA

- category-wise EM / F1
- single-hop vs multi-hop if labels support it
- temporal / causal / adversarial categories if present
- retrieval-active vs retrieval-inactive cases
- P7-correct / Disabled-wrong cases
- P7-wrong / Disabled-correct harmful cases
- invalid output analysis

## 8. Execution Readiness Criteria

- no method changes required
- LoCoMo local path resolved
- LoCoMo QA-only scope fixed
- deterministic scorer fixed
- output schema fixed
- frozen-bank multi-question protocol fixed
- `query_write_count == 0` can be asserted
- EventQA missing rows are explicitly tracked
- shared baseline naming is fixed across benchmarks

## 9. Safety Constraints

- Do not run GPU experiments
- Do not implement code yet
- Do not download datasets
- Do not modify P7
- Do not modify model code
- Do not modify `paper/`
- Only create planning documents
