# Section Writing Plan

This file maps the reviewed `paper/outline.md` to evidence and writing gates.
The long-horizon LLM-agent framing defines the paper; EventQA defines the
current positive operational evidence.

## Abstract

- Writing goal: compress the problem, inference-time method, completed EventQA
  evidence, and strongest limitation into one scoped paragraph.
- Key message: a frozen session-local latent bank improves EventQA event
  reasoning without retraining MemGen components.
- Evidence to cite:
  - `research_notes/PAPER_SCOPE.md`
  - `outputs/mab/eventqa_five_repeat_stability_summary.md`
  - `outputs/mab/eventqa_current_stage_consolidated_summary.md`
- Completed support: explicit-text baselines, no-query-retrieval ablation, and
  method-separable cost rows.
- Overclaim warning: do not imply benchmark generality, RAG superiority, or
  cost efficiency.
- Readiness: abstract polished against the completed evidence tables.

## 1. Introduction

- Writing goal: make persistent latent reuse the central problem and explain
  why a session-local bank is the minimal inference-time addition to MemGen.
- Key message: the method turns transient Weaver-space latents into bounded,
  reusable session state without retraining.
- Evidence to cite:
  - `research_notes/METHOD.md`
  - `research_notes/DECISIONS.md` (DEC-0077 and DEC-0078)
  - `outputs/mab/paper_scope_claim_redirect.md`
- Completed support: verified closest-neighbor citations in
  `paper/references.bib`; completed explicit-memory comparator package.
- Overclaim warning: describe the scientific gap, not local engineering history;
  scope the empirical claim to EventQA.
- Readiness: introduction positioning, citations, and prose polish complete.

## 2. Related Work

- Writing goal: position session-local latent memory among latent recurrence,
  long-context inference, RAG, summaries, agent memory, and dialogue memory.
- Key message: this paper studies frozen inference-time latent reuse for
  long-horizon agents, not a new training objective.
- Evidence to cite:
  - current method boundary in `research_notes/PAPER_SCOPE.md`
  - verified external literature in `paper/references.bib`
- Completed support: citation shortlist, closest-neighbor comparison, and
  operational novelty boundary against prior latent-memory systems.
- Overclaim warning: do not claim novelty or superiority from memory; do not
  criticize RAG before the explicit-text baseline is run.
- Readiness: subsection prose and verified venue metadata polished.

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
- Readiness: drafted in Limitations and Appendix A with reliable-field and
  excluded-field boundaries recorded.

## 4. Experimental Setup

- Writing goal: define EventQA, the visible six-candidate query contract,
  frozen context construction, comparators, repetition, and metrics.
- Key message: all completed main rows use the same local official non-strict
  prompt/parser/scorer and context/question set.
- Evidence to cite:
  - `outputs/mab/eventqa_paper_completion_plan.md`
  - `outputs/mab/eventqa_final_comparison_package.md`
  - `research_notes/EXPERIMENTS.md`
- Missing evidence: none for the EventQA comparison package itself; only final
  table rendering and figure formatting remain.
- Overclaim warning: Bank-off is compressed no-bank, not official full-history;
  substring EM is not strict full-string EM.
- Readiness: ready.

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
- Missing evidence: none for the completed EventQA comparator package. The only
  remaining gap is converting the package into final paper tables/figures.
- Overclaim warning: do not call P7 generally superior or cost efficient; do
  not hide context 4.
- Readiness: ready.

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
- Missing evidence: none for component attribution and explicit-memory control
  references; all are now present in the unified EventQA package.
- Overclaim warning: harmful tuple evidence is oracle and single-bank; score
  association is not causal proof.
- Readiness: ready, with cost wording caveats.

## 7. Limitations

- Writing goal: state the exact empirical and mechanism boundary before the
  reader infers general memory claims.
- Key message: the verified EventQA result is context-dependent and does not
  establish benchmark-general performance. LoCoMo may be retained as optional
  latent-to-fact limitation evidence.
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
- Overclaim warning: do not conclude benchmark-general success.
- Readiness: provisional draft now; freeze last.

## Appendix

- Writing goal: provide reproducibility, complete tables, negative ablations,
  detailed failures, and diagnostic benchmark evidence.
- Key message: the main claim is supported by traceable artifacts and its
  limitations are not hidden.
- Evidence to cite:
  - `outputs/mab/eventqa_final_table_inventory.md`
  - EventQA prompt/scorer verification and per-context artifacts
  - packaged LoCoMo prompt, answer, pipeline, and diagnostics audit in
    `paper/appendix_locomo_diagnostic.md`
- Missing evidence: none for the scoped appendix modules.
- Overclaim warning: clearly label historical, exploratory, single-run,
  single-bank, and oracle evidence.
- Readiness: LoCoMo boundary appendix assembled; other optional appendix modules
  remain available if selected during final polish.
