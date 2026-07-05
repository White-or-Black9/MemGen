# P7 Benchmark Replacement And Supplement Audit

Goal: audit replacement and supplementary benchmarks for the frozen P7 formal experiment without downgrading the paper target, which remains **Latent Memory Bank Improves Long-Context and Multi-Turn Reasoning**.

Constraints respected:
- no GPU runs
- no runner implementation
- no P7 changes
- no model-code changes
- no `paper/` edits

## Trusted State

- EventQA remains the current long-context anchor and its reusable P7/P6 five-repeat rows do not need rerunning for this phase.
- LongMemEval is locally available only through the MemoryAgentBench-converted `longmemeval_s*` path, but formal scoring depends on `gpt-4o`, so it is not the immediate formal main benchmark.
- The local benchmark inventory is dominated by MemoryAgentBench:
  - split files:
    - `/mnt/18T/baishilong/datasets/MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet`
    - `/mnt/18T/baishilong/datasets/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet`
    - `/mnt/18T/baishilong/datasets/MemoryAgentBench/data/Long_Range_Understanding-00000-of-00001.parquet`
    - `/mnt/18T/baishilong/datasets/MemoryAgentBench/data/Test_Time_Learning-00000-of-00001.parquet`
  - configs:
    - `.../Accurate_Retrieval/Ruler/QA/Ruler_qa1_197k.yaml`
    - `.../Accurate_Retrieval/Ruler/QA/Ruler_qa2_421k.yaml`
    - `.../Conflict_Resolution/Factconsolidation_{sh,mh}_{6k,32k,64k,262k}.yaml`
    - `.../Long_Range_Understanding/Detective_QA.yaml`
    - `.../Test_Time_Learning/Recsys/Recsys_redial_full.yaml`
- Existing MemGen runner patterns are already present for:
  - one-context MAB bank-off / bank-on / compressed-memory pilots:
    - `scripts/eval/mab2_bank_off.py`
    - `scripts/eval/mab3_bank_on_full_history.py`
    - `scripts/eval/mab4a_compressed_memory.py`
    - `scripts/eval/mab2_mab_bridge.py`
  - detective_qa compressed and P7-proximal Weaver-space runs:
    - `scripts/eval/mab5a_detectiveqa_compressed_n10.py`
    - `scripts/eval/mab5b_raised_shared_threshold_detectiveqa_n10.py`
    - `scripts/eval/mab5c_decoupled_thresholds_detectiveqa_n10.py`
    - `scripts/eval/mab5d_capacity16_detectiveqa_n10.py`
    - `scripts/eval/mab6b_weaver_space_bank_detectiveqa_n10.py`
  - EventQA frozen-bank main runner:
    - `scripts/eval/mab6b_weaver_space_bank_eventqa_65536_n5.py`

## Local Availability Snapshot

Verified local source counts from cached MemoryAgentBench parquet files:

- Accurate Retrieval
  - `eventqa_65536`: 5 contexts
  - `eventqa_131072`: 5 contexts
  - `eventqa_full`: 5 contexts
  - `longmemeval_s*`: 5 contexts
  - `ruler_qa1_197K`: 1 context
  - `ruler_qa2_421K`: 1 context
- Conflict Resolution
  - `factconsolidation_sh_6k`: 1 context
  - `factconsolidation_sh_32k`: 1 context
  - `factconsolidation_sh_64k`: 1 context
  - `factconsolidation_sh_262k`: 1 context
  - `factconsolidation_mh_6k`: 1 context
  - `factconsolidation_mh_32k`: 1 context
  - `factconsolidation_mh_64k`: 1 context
  - `factconsolidation_mh_262k`: 1 context
- Long Range Understanding
  - `detective_qa`: 10 contexts
  - `infbench_sum_eng_shots2`: 100 contexts
- Test Time Learning
  - `recsys_redial_full`: 1 context
  - `icl_*`: one source each

## Candidate Findings

The detailed field-by-field matrix is exported in:
- `outputs/mab/p7_benchmark_recommendation_matrix.json`

The high-signal conclusions are:

### 1. RULER-QA1 / RULER-QA2

- Best immediate second-main candidate for the long-context side.
- Strong reasons:
  - local data and YAML configs already exist
  - deterministic `substring_exact_match`
  - benchmark family is already accepted inside MemoryAgentBench under `Accurate_Retrieval`
  - protocol maps cleanly to P7:
    - one long context = one session-local bank
    - sequential construction ingestion
    - frozen bank reused across many questions
    - query-time writes blocked
- Main caveats:
  - only one local context exists for each of `ruler_qa1_197K` and `ruler_qa2_421K`
  - this is long-context retrieval/reasoning, not conversational multi-session memory
  - MemGen has no native RULER runner yet, so adapter work is still required
- Paper role:
  - viable second main benchmark for long-context reasoning
  - not a replacement for the conversational-memory half of the paper target

### 2. FactConsolidation-SH / MH

- Best immediate diagnostic for update / conflict behavior.
- Strong reasons:
  - local data and configs exist in all 8 variants
  - deterministic `substring_exact_match`
  - directly probes conflict resolution and knowledge-update behavior
  - P7 frozen `update_threshold=0.10` and session-local bank can be evaluated without changing the method
- Main caveats:
  - only one local context per source
  - not a broad benchmark by itself
  - current historical MemGen evidence is mostly one-context / exploratory
- Paper role:
  - diagnostic or limitation-analysis benchmark
  - useful to document whether P7 update/write policy helps or harms under conflicting information

### 3. DetectiveQA

- Most engineering-ready supplementary benchmark because the MemGen runner lineage already exists.
- Strong reasons:
  - local `detective_qa` has 10 contexts
  - exact-match scoring is deterministic
  - many MemGen scripts/tests already target this task
  - existing evidence already shows bank-off/bank-on paired protocol, zero query writes, and P7-adjacent Weaver-space runs
- Main caveats:
  - historical scores are weak, including many zero-EM diagnostics before EventQA
  - original full-history path is over-capacity invalid; only compressed-memory variants are meaningful
  - the task is long-range understanding, not multi-session conversational memory
- Paper role:
  - appendix or mechanism diagnostic
  - not the preferred second main benchmark

### 4. Recsys Redial Full

- Conceptually attractive because it is long conversational preference memory with deterministic `Recall@5`.
- Why it is not the immediate choice:
  - only one local context
  - enormous context (`~5.6M` characters locally)
  - no MemGen-native runner or parser path
  - output must be constrained to movie IDs or reliably mapped movie titles
- Paper role:
  - future supplementary benchmark if a bounded adapter is approved

### 5. ICL TTL tasks

- Deterministic and local, but weak fit for the current paper message.
- They test test-time in-context classification more than long-context memory reasoning.
- Paper role:
  - skip for the current formal package unless later needed as lightweight robustness appendix

### 6. InfBench Summarization

- Local, but final metric uses `gpt-4o` judging.
- Same practical blocker class as LongMemEval.
- Paper role:
  - future only

## Special Attention Findings

### MSC-MemFuse-MC10

- Local availability: not found in this repository, local dataset cache, or the nearby benchmark tree.
- Public identifier status in this audit: unresolved. I did not find a reliable local artifact or a stable public benchmark handle that lets me verify dataset format or scorer from the current workspace.
- If this benchmark really is:
  - multi-session conversation memory
  - 10-way multiple choice
  - deterministic accuracy
  then it would be an unusually attractive second-main candidate because:
  - one sample could map naturally to one session-local P7 bank
  - output could be constrained to an option index or exact option text
  - final scoring could be simple accuracy without an LLM judge
- Current decision:
  - do not select it now
  - the benchmark identity, code path, and local data path must be resolved first
- Paper value today:
  - `future work / blocked candidate`

### FactConsolidation-SH

- This is the cleanest currently local update/conflict diagnostic.
- Deterministic scoring is available through MemoryAgentBench `substring_exact_match`.
- P7 can be evaluated without method changes:
  - session reset per context
  - normal construction-time writes
  - query-time retrieval permitted
  - query-time writes blocked in the formal comparison protocol
- Main limitation:
  - local cache exposes one context per source, so it is diagnostic evidence rather than a broad benchmark row family

### HaluMem

- Local availability: not found.
- Public benchmark profile:
  - operation-level benchmark with separate memory extraction, memory updating, and memory question-answering tasks
  - focused on memory hallucinations, conflicts, omissions, and fabrication
- Relevance to latent-memory P7:
  - QA-only evaluation might be indirectly adaptable
  - extraction/update tasks are not a clean direct fit because frozen P7 does not emit explicit textual memory objects as its main interface
- Practical implication:
  - HaluMem is better matched to explicit-text memory systems than to a latent-memory bank, unless a proxy protocol is invented
  - that proxy would no longer be a clean like-for-like benchmark evaluation
- Paper value today:
  - `future reliability diagnostic`, not immediate formal benchmark

## External Non-Local Candidates

These candidates were not found locally and would require new dataset import plus new runner/eval work:

- LoCoMo QA subset
  - strongest external conceptual fit after LongMemEval for multi-session conversational memory
  - but not locally present and would still need a MemGen-native adapter
- LongBench v2
  - deterministic multiple-choice accuracy and strong long-context value
  - not memory-specific and not conversational
- BABILong
  - deterministic long-context reasoning benchmark
  - not conversational or memory-specific
- MADial-Bench
  - dialogue-generation benchmark with richer human-like evaluation dimensions
  - likely poor fit for a deterministic immediate formal package
- DialSim / LongDialQA
  - relevant long-term dialogue understanding angle
  - but simulator-style overhead raises implementation risk
- MemBench
  - interesting broad memory benchmark, but not locally present and likely larger adapter surface
- MemoryArena
  - strong future agentic-memory benchmark, but much farther from current MemGen evaluation conventions
- EvoMemBench / Evo-Memory
  - ambitious streaming / evolving-memory frameworks, but far beyond the current frozen-P7 formal scope
- StreamBench
  - oriented toward continuous improvement / online learning rather than the present session-local evaluation target

## Recommendation

### Top 3 for this project

1. `RULER-QA2` as the preferred second main benchmark candidate.
2. `FactConsolidation-SH` as the primary update/conflict diagnostic.
3. `DetectiveQA` as the lowest-risk supplementary appendix benchmark because the runner lineage already exists.

### Second main benchmark choice

- Choose `RULER-QA2` first.
- Why `QA2` over `QA1`:
  - larger configured context budget (`421k` vs `197k`)
  - stronger paper-facing value for the long-context half of the claim
  - still deterministic and already local
- Caveat:
  - because it is not conversational, it supplements EventQA on long-context reasoning but does not replace the missing conversational-memory benchmark.

### Diagnostic choice

- Choose `FactConsolidation-SH`.
- It is the best fit for testing whether frozen P7 update/write behavior remains useful or harmful under contradiction and revision pressure.

### Postpone

- Keep `LongMemEval` as future / prediction-only / qualitative until judge access exists.
- Postpone `MSC-MemFuse-MC10` until the benchmark identity and local path are resolved.
- Postpone `HaluMem`, `LoCoMo`, `MADial-Bench`, `DialSim/LongDialQA`, `MemoryArena`, `MemBench`, `EvoMemBench`, `Evo-Memory`, and `StreamBench` because they are not local and/or require a broader adapter surface than the current phase should absorb.

## Replacement Decision

No currently available candidate fully replaces LongMemEval without downgrading the goal.

Reason:
- the long-context side can be strengthened immediately by `RULER-QA2`
- the update/conflict side can be strengthened immediately by `FactConsolidation-SH`
- but the specific **multi-session conversational memory** part of the paper target remains best served by LongMemEval or another equivalent conversational benchmark, and no equally ready deterministic local substitute was confirmed in this audit

## Exact Next Step Before Implementation

Lock the benchmark stack before coding:

1. Promote `RULER-QA2` as the second main benchmark candidate.
2. Promote `FactConsolidation-SH` as the primary diagnostic benchmark.
3. Keep `DetectiveQA` as appendix / mechanism evidence only.
4. Treat LongMemEval as deferred future work, not the immediate formal benchmark.
5. After that decision, write one implementation plan for:
   - a MemGen `RULER-QA` frozen-bank runner for P7
   - a minimal `FactConsolidation-SH` diagnostic runner or adapter
   - shared deterministic output schema, latency logging, and peak-memory logging
