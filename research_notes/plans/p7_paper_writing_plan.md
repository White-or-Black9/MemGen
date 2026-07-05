# P7 Paper Writing Plan

Date: 2026-07-04

## Paper Target

`Latent Memory Bank Improves Long-Context and Multi-Turn Reasoning.`

## Writing Principle

- Do not overclaim from EventQA alone.
- Keep the paper story explicitly two-axis:
  - long-context reasoning
  - multi-turn / multi-session conversational memory
- Treat frozen P7 as the final method unless later evidence forces a stop decision.

## Section-by-Section Plan

### 1. Introduction

#### Key claims

- latent memory banks can improve long-context reasoning without requiring retraining of memory utilities
- the same frozen mechanism can transfer to multi-session conversational memory
- explicit-text alternatives are not sufficient controls unless summary, RAG, and matched-budget baselines are included

#### Required tables / figures

- high-level benchmark/task figure
- method overview figure
- teaser main-results table placeholder

#### Required experiment outputs

- final benchmark stack
- P7 method diagram
- main benchmark results from EventQA and LoCoMo-QA

#### Missing evidence

- final LoCoMo-QA results
- explicit-text baseline results

#### Draft status

- can draft now in outline form
- final wording must wait for LoCoMo-QA and baseline results

### 2. Related Work

#### Key claims

- prior long-context QA benchmarks do not fully cover multi-session conversational memory
- prior conversational-memory benchmarks often rely on explicit memory or judge-heavy evaluation
- the paper occupies the intersection of long-context reasoning and conversational memory

#### Required tables / figures

- optional comparison matrix of benchmark properties

#### Required experiment outputs

- none strictly required for first draft

#### Missing evidence

- none blocking a draft

#### Draft status

- can draft now

### 3. Method

#### Key claims

- frozen P7 uses a session-local latent memory bank with fixed thresholds and bounded slots
- query-time writes are blocked for frozen-bank QA protocols
- no Trigger / Weaver retraining or auxiliary utility gates are used

#### Required tables / figures

- method block diagram
- protocol diagram for construction vs query

#### Required experiment outputs

- final frozen P7 specification
- benchmark protocol mapping notes for EventQA and LoCoMo-QA

#### Missing evidence

- none for the method description itself

#### Draft status

- can draft now

### 4. Experimental Setup

#### Key claims

- EventQA is the long-context anchor
- LoCoMo-QA is the multi-session anchor
- LongMemEval is deferred because judge availability blocks formal scoring
- baseline family is shared across benchmarks

#### Required tables / figures

- benchmark table
- baseline/protocol table

#### Required experiment outputs

- final benchmark decision
- final baseline definitions
- LoCoMo QA-only protocol

#### Missing evidence

- final implemented LoCoMo runner details
- final subset freeze for LoCoMo formal run

#### Draft status

- can draft now as a protocol section
- final command/config specifics wait for implementation

### 5. Main Results

#### Key claims

- P7 improves over Disabled and relevant comparators on EventQA
- P7 also improves on multi-session conversational memory in LoCoMo-QA
- gains persist against explicit-text baselines

#### Required tables / figures

- EventQA main table
- LoCoMo-QA main table
- combined benchmark summary table

#### Required experiment outputs

- final EventQA aggregation table
- final LoCoMo-QA formal table
- shared baseline results

#### Missing evidence

- EventQA missing baselines
- LoCoMo implementation and runs

#### Draft status

- must wait for experiments

### 6. Ablation and Analysis

#### Key claims

- prompt/scorer path is validated
- strict and first-line prompt ablations are negative controls
- context-specific failures and category-specific failures expose the method boundary

#### Required tables / figures

- EventQA appendix/ablation table
- LoCoMo category table
- selected qualitative case table

#### Required experiment outputs

- EventQA strict / first-line / P4 / context-4 / harmful-attribution artifacts
- LoCoMo category-wise results
- P7-correct / Disabled-wrong and harmful failure examples

#### Missing evidence

- LoCoMo category-wise outputs
- shared baseline comparison cases

#### Draft status

- partial draft possible now for EventQA side
- final section must wait for LoCoMo results

### 7. Cost and Efficiency

#### Key claims

- P7 gains should be interpreted jointly with latency and memory cost
- latent-bank gains should be compared against explicit-text baselines under matched budget

#### Required tables / figures

- cost table across benchmarks and baselines
- optional accuracy-cost frontier figure

#### Required experiment outputs

- packaged EventQA disabled cost row
- EventQA and LoCoMo shared cost metrics
- matched-budget baseline outputs

#### Missing evidence

- most explicit-text baseline cost rows
- LoCoMo cost rows

#### Draft status

- outline only now
- final section must wait for experiments

### 8. Limitations

#### Key claims

- EventQA still has context-specific failures
- harmful attribution cases exist
- LoCoMo formal scope is initially limited to QA-only
- LongMemEval is deferred by judge dependency

#### Required tables / figures

- limitation table or compact bullet summary

#### Required experiment outputs

- EventQA context-4 and harmful-attribution artifacts
- LoCoMo invalid-output and harmful-case analysis
- LongMemEval audit summary

#### Missing evidence

- LoCoMo limitation outputs

#### Draft status

- partial draft possible now
- full section waits for LoCoMo runs

### 9. Conclusion

#### Key claims

- frozen latent memory banks improve long-context reasoning
- the same mechanism extends to multi-session conversational QA
- future work includes judge-dependent LongMemEval and broader conversational benchmarks

#### Required tables / figures

- none required

#### Required experiment outputs

- final main results and limitation summary

#### Missing evidence

- final LoCoMo and baseline results

#### Draft status

- must wait for experiments beyond a high-level placeholder

## What Can Be Drafted Now

- Introduction outline
- Related Work
- Method
- most of Experimental Setup
- EventQA side of Ablation and Analysis
- partial Limitations

## What Must Wait

- Main Results
- Cost and Efficiency
- LoCoMo-heavy parts of Ablation and Analysis
- final Conclusion
- final claim wording in Introduction and Abstract

## Evidence Needed Before Full Draft Freeze

- EventQA missing baseline rows
- EventQA unified final aggregation table
- LoCoMo adapter/scorer implementation
- LoCoMo Disabled and P7 smoke
- LoCoMo pilot
- LoCoMo baseline protocol runs
- LoCoMo formal run
- shared cost tables
