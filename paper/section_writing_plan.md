# Section Writing Plan

This file maps each manuscript section to its evidence and writing gate.

## Abstract

- Writing goal: compress the problem, inference-time method, completed EventQA
  evidence, and strongest limitation into one scoped paragraph.
- Key message: a frozen session-local latent bank improves EventQA event
  reasoning without retraining MemGen components.
- Evidence to cite:
  - `research_notes/PAPER_SCOPE.md`
  - `outputs/mab/eventqa_five_repeat_stability_summary.md`
  - `outputs/mab/eventqa_current_stage_consolidated_summary.md`
- Missing evidence: explicit-text baselines and method-separable cost.
- Overclaim warning: do not imply benchmark generality, RAG superiority, cost
  efficiency, or multi-turn success.
- Readiness: provisional draft now; final version waits for baseline tables.

## 1. Introduction

- Writing goal: make persistent latent reuse the central problem and explain
  why a session-local bank is the minimal inference-time addition to MemGen.
- Key message: the method turns transient Weaver-space latents into bounded,
  reusable session state without retraining.
- Evidence to cite:
  - `research_notes/METHOD.md`
  - `research_notes/DECISIONS.md` (DEC-0077 and DEC-0078)
  - `outputs/mab/paper_scope_claim_redirect.md`
- Missing evidence: verified literature citations and final comparator breadth.
- Overclaim warning: describe the scientific gap, not local engineering history;
  scope the empirical claim to EventQA.
- Readiness: structure and contribution paragraph ready; citations pending.

## 2. Related Work

- Writing goal: position session-local latent memory among latent recurrence,
  long-context inference, RAG, summaries, agent memory, and dialogue memory.
- Key message: this paper studies frozen inference-time latent reuse, not a new
  training objective or a general conversational-memory system.
- Evidence to cite:
  - current method boundary in `research_notes/PAPER_SCOPE.md`
  - verified external literature to be collected separately
- Missing evidence: citation shortlist, closest-neighbor comparison, and exact
  novelty boundary against prior latent-memory systems.
- Overclaim warning: do not claim novelty or superiority from memory; do not
  criticize RAG before the explicit-text baseline is run.
- Readiness: subsection outline ready; prose waits for citation verification.

## 3. Method

- Writing goal: specify the frozen MemGen components, bank state, lifecycle,
  retrieval/update rules, and frozen-bank query contract.
- Key message: P7 is a bounded session-local inference mechanism with no
  Trigger, Weaver, or Reasoner retraining.
- Evidence to cite:
  - `research_notes/METHOD.md`
  - `research_notes/PAPER_SCOPE.md`
  - `outputs/mab/eventqa_p7_non_strict_official_prompt_scorer_verification.md`
- Missing evidence: none for the frozen P7 definition.
- Overclaim warning: exclude utility gate, tuple suppression, top-1 fallback,
  and oracle attribution from the method.
- Readiness: ready to draft now.

## 4. Experimental Setup

- Writing goal: define EventQA, the visible six-candidate query contract,
  frozen context construction, comparators, repetition, and metrics.
- Key message: all completed main rows use the same local official non-strict
  prompt/parser/scorer and context/question set.
- Evidence to cite:
  - `outputs/mab/eventqa_paper_completion_plan.md`
  - `outputs/mab/eventqa_final_table_inventory.md`
  - `research_notes/EXPERIMENTS.md`
- Missing evidence: final summary/RAG/matched-budget protocols and cost
  measurement protocol.
- Overclaim warning: Bank-off is compressed no-bank, not official full-history;
  substring EM is not strict full-string EM.
- Readiness: core setup ready; baseline/cost subsection remains open.

## 5. Main Results

- Writing goal: present P7 versus Bank-off and P7 versus P6 before analyses.
- Key message: P7 has a repeated positive EventQA effect and a better EM/format
  trade-off than P6.
- Evidence to cite:
  - `outputs/mab/eventqa_five_repeat_stability_summary.md`
  - `outputs/mab/eventqa_p7_vs_p6_final_summary.md`
  - `outputs/mab/eventqa_current_stage_consolidated_summary.md`
- Existing numbers:
  - P7 EM `0.197+-0.020`, recall `0.254+-0.028`, format failures
    `121.4+-8.8`;
  - Bank-off EM `0.008`, recall `0.178`;
  - P7-P6 EM `+0.0280`, recall `-0.0044`, format failures `-44.4`.
- Missing evidence: summary, RAG, matched-budget, no-query-retrieval, and valid
  method-separable cost rows.
- Overclaim warning: do not call P7 generally superior or cost efficient; do
  not hide context 4.
- Readiness: current-result subsection ready; final comparison narrative waits.

## 6. Analysis

- Writing goal: explain where gains come from, how outputs fail, and where the
  mechanism is unreliable.
- Key message: latent retrieval is beneficial overall but utility is
  context-dependent and retrieval activity alone does not ensure correctness.
- Evidence to cite:
  - `outputs/mab/eventqa_current_stage_consolidated_summary.md`
  - `outputs/mab/eventqa_p7_context4_failure_diagnosis.md`
  - `outputs/mab/eventqa_format_failure_taxonomy.md`
  - `outputs/mab/eventqa_harmful_memory_attribution_context4_full/`
- Missing evidence: no-query-retrieval component result; explicit-text and cost
  comparisons.
- Overclaim warning: harmful tuple evidence is oracle and single-bank; score
  association is not causal proof.
- Readiness: context/format/transition analysis ready; component/cost analysis waits.

## 7. Limitations

- Writing goal: state the exact empirical and mechanism boundary before the
  reader infers general memory claims.
- Key message: EventQA supports event reasoning, while LoCoMo exposes failure
  of latent-only exact conversational fact recovery.
- Evidence to cite:
  - `outputs/mab/locomo_vs_eventqa_experiment_comparison.md`
  - `outputs/mab/locomo_vs_eventqa_result_gap_analysis.md`
  - `outputs/mab/locomo_qa_full_pipeline_audit.md`
  - EventQA context-4 diagnostics
- Missing evidence: none for the current negative boundary; broader benchmark
  generalization remains absent.
- Overclaim warning: do not frame LoCoMo F1 `+0.00250` as improvement; all 304
  exact-match pairs are wrong.
- Readiness: ready to draft now.

## 8. Conclusion

- Writing goal: restate the scoped method and evidence, then identify the next
  technical frontier.
- Key message: session-local latent reuse helps closed-set EventQA reasoning but
  requires stronger utility control and latent-to-fact decoding for broader use.
- Evidence to cite: main EventQA result and limitation section only.
- Missing evidence: final baseline/cost results may change comparative emphasis.
- Overclaim warning: do not conclude general long-context or multi-turn success.
- Readiness: provisional draft now; freeze last.

## Appendix

- Writing goal: provide reproducibility, complete tables, negative ablations,
  detailed failures, and diagnostic benchmark evidence.
- Key message: the main claim is supported by traceable artifacts and its
  limitations are not hidden.
- Evidence to cite:
  - `outputs/mab/eventqa_final_table_inventory.md`
  - EventQA prompt/scorer verification and per-context artifacts
  - LoCoMo prompt, answer, pipeline, and diagnostics audits
- Missing evidence: final new baseline artifacts and unified manifest.
- Overclaim warning: clearly label historical, exploratory, single-run,
  single-bank, and oracle evidence.
- Readiness: existing appendix modules ready; final assembly waits for new rows.
