# Appendix A Package: LoCoMo-QA Boundary Diagnostic

This package is mirrored into Appendix A of `paper/draft_v0.md`. It records the
source boundary used for the paper-facing table and examples.

## Authoritative Sources

- `outputs/mab/locomo_qa_disabled_vs_p7_answer_comparison.json`
- `outputs/mab/locomo_qa_full_pipeline_audit.json`
- `outputs/mab/locomo_qa_diagnostics_field_provenance.json`
- `outputs/mab/locomo_qa_prompt_inspection.json`
- Session-level row artifacts under
  `outputs/mab/locomo_qa_pilot_session_{disabled,p7}_2conv/`

## Included Fields

- deterministic EM and token F1;
- invalid-output and saved-answer heuristic flags;
- construction chunk/write/retrieval/slot counts from the preserved snapshot;
- query retrieval activity, query-write invariance, and snapshot invariance;
- representative paired rows from the authoritative comparison artifact.

## Explicitly Excluded Fields

- construction latency;
- construction peak GPU memory;
- construction Trigger counts;
- construction Weaver counts;
- any cost comparison between Disabled and P7.

These fields are excluded because the construction-only stop path does not
propagate the required timing, memory, or generation records into the nested
construction diagnostics.

## Claim Boundary

The appendix may state that construction, freezing, retrieval, and write
blocking operated mechanically. It may not state that P7 improves LoCoMo,
multi-session conversational QA, or long-horizon memory in general. Both
methods remain at zero exact match over all 304 paired questions.
