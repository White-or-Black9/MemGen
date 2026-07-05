# MAB-6B-FR EventQA 65536 n5: Current Experiment Record

Updated: 2026-07-03 (Asia/Shanghai)

## Scope and Evidence Boundary

This note is the durable project record for recent EventQA-65536 experiments
under the MAB-6B frozen-context-bank runner. It consolidates the historical
A/B/C/D exploration, the P1--P7 search, reproducibility runs, prompt ablations,
and offline failure analyses. It does not claim a direct official full-history
long-context baseline: Bank-off is the runner's compressed bridge without a
persistent Memory Bank because the rendered full history exceeds capacity.

- Dataset/task: MemoryAgentBench `Accurate_Retrieval/eventqa_65536`.
- Protocol: `frozen_context_bank`, five contexts, 100 questions per context.
- Common generation limit: `40`.
- Primary metric: official EventQA `substring_exact_match` / Accuracy.
- Auxiliary metric: `eventqa_recall`.
- Current evidence source: completed run artifacts and the frozen stage summary
  in `outputs/mab/eventqa_current_stage_consolidated_summary.{md,json}`.
- Protected DetectiveQA note is intentionally not edited:
  `research_notes/benchmarks/memoryagentbench_mab6b_weaver_space_bank.md`.

## Current Decision

1. **P7 non-strict is the paper-level main EventQA candidate among the current
   tested configurations.** Across five repeats it has the best official EM and
   the lowest format-failure burden among P4/P6/P7.
2. **P6 remains the lower-update-threshold comparison and earlier
   recall-oriented baseline.** Its five-repeat mean recall exceeds P7 by only
   `0.0044`, weakening the earlier claim of a meaningful recall advantage.
3. **P4 is lower and is not a main candidate.**
4. **Strict and first-line prompt variants are negative ablations.** Neither
   improves the official result; first-line is especially harmful.
5. **P7 non-strict uses the default EventQA prompt wrapper and the official
   MemoryAgentBench scorer/parser path unchanged.** No parser repair,
   candidate normalization, output repair, strict prompt, or first-line prompt
   is used.
6. **The dominant unresolved failure is no-gold / memory-conditioned generation
   corruption, not parser or first-line mismatch.**
7. **Next mechanism direction:** harmful-slot and harmful-tuple attribution and
   suppression, with utility or routing gates. This work has not started.

## Protocol and Official Prompt/Scorer Verification

P7 non-strict uses:

```text
Based on the context you memorized, complete the task below:

{question}

The event that happens next is:
```

This wrapper is character-identical to the local upstream MemoryAgentBench
EventQA `Long_context_agent` template snapshot. The runner path is:

```text
mab6b_weaver_space_bank_eventqa_65536_n5.py::_score_prediction
  -> scripts/eval/mab2_mab_bridge.py
  -> MemoryAgentBench/utils/eval_other_utils.py::post_process
  -> _process_eventqa_dataset
  -> parse_output / calculate_metrics
```

Bank-off and Bank-on use the same visible query prompt and the same scoring
path. Bank-on memory enters through the latent Weaver path, not visible prompt
text. P7 non-strict has both prompt flags disabled and performs no parser or
post-processing repair.

Common P-series settings are `decay_alpha=0.05`, `requested_contexts=5`,
`eventqa_protocol=frozen_context_bank`, `generation_max_length=40`, and
`retrieved_memory_to_weaver=true`. Diagnostic P-series runs use per-context
reseeding, score decomposition, frozen-bank saving, and bank-transition
diagnostics.

## Table 1. Main Non-strict Families

| Family | Configuration `(rt, ut, slots, top_k, decay)` | Repeats | Bank-on EM mean±std [min,max] | Bank-on recall mean±std [min,max] | Format failures mean±std [min,max] | Helpful / harmful / format harm | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| P7 | `(0.05, 0.10, 16, 2, 0.05)` | 5 | `0.197±0.020 [0.162,0.220]` | `0.254±0.028 [0.212,0.298]` | `121.4±8.8 [108,135]` | `98.4±10.1 / 4.0±0.0 / 28.4±9.3` | Paper-level main candidate |
| P6 | `(0.05, 0.095, 16, 2, 0.05)` | 5 | `0.169±0.018 [0.150,0.194]` | `0.258±0.016 [0.236,0.280]` | `165.8±19.8 [137,187]` | `83.6±9.4 / 3.2±0.7 / 44.6±9.0` | Lower-UT comparison; earlier recall baseline |
| P4 | `(0.03, 0.095, 16, 2, 0.05)` | 2 | `0.149±0.015 [0.134,0.164]` | `0.259±0.007 [0.252,0.266]` | `245.0±50.0 [195,295]` | `74.5±7.5 / 4.0±0.0 / 55.0±4.0` | Lower; control only |

Across these repeat sets, Bank-off is stable at EM `0.008±0.000` and recall
`0.178±0.000`. All main-family runs finish with 16 slots for every query.
Construction totals across five contexts are:

- P4 repeats: `insert=80`, `matched_replace=4`, `capacity_evict=1` each.
- The original three P6 repeats each recorded `insert=80`,
  `matched_replace=4`, `capacity_evict=1`; rep4/rep5 retain complete
  construction diagnostics in their run roots.
- The original three P7 repeats recorded `insert=80`,
  `matched_replace=2/1/2`, `capacity_evict=3/4/3`; rep4/rep5 retain complete
  construction diagnostics in their run roots.

P7 improves official EM and output stability relative to P6 while maintaining
comparable recall (`0.254` versus `0.258`). P6 remains useful for isolating the
update-threshold change, but no longer shows a meaningful recall advantage.

## Table 2. Prompt Ablations

Non-strict rows are repeat means; strict and first-line rows are single runs.
`--` means the frozen stage summary did not retain that auxiliary count.

| Setting | EM | Recall | Format failures | Chinese outputs | Helpful | Harmful | Format harm | Conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P7 non-strict | `0.197` | `0.254` | `121.4` | `151.8` | `98.4` | `4.0` | `28.4` | Main, 5 repeats |
| S7 strict | `0.154` | `0.184` | `126` | `240` | `74` | `5` | `15` | Negative |
| FL-P7 first-line | `0.090` | `0.182` | `193` | `338` | `43` | `11` | `46` | Strong negative |
| P6 non-strict | `0.169` | `0.258` | `165.8` | `189.8` | `83.6` | `3.2` | `44.6` | Lower-UT comparison, 5 repeats |
| S6 strict | `0.088` | `0.224` | `291` | `329` | `42` | `6` | `69` | Negative |
| FL-P6 first-line | `0.146` | `0.256` | `215` | `206` | `66` | `6` | `55` | Negative |
| P4 non-strict | `0.149` | `0.259` | `245.0` | `271.5` | `74.5` | `4.0` | `55.0` | Lower control |
| S4 strict | `0.078` | `0.230` | `244` | `379` | `39` | `8` | `76` | Negative |
| FL-P4 first-line | `0.080` | `0.260` | `296` | `334` | `35` | `8` | `90` | Negative |

Prompt-only format control did not improve official EM. In P7, the lightweight
first-line instruction is worse than both non-strict and strict prompting.
These variants remain negative ablations rather than main protocols.

## Table 3. Failure Taxonomy

Counts aggregate the relevant repeats and classify Bank-on EM failures.

| Family | Parser-sensitive | No-gold | Clean wrong | First-line noise, no gold | First-line noise, later gold | Empty/degenerate | Conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| P7 | `60/1213 (4.9%)` | `1153/1213 (95.1%)` | `715` | `337` | `60` | `87` | Predominantly no-gold/corruption |
| P6 | `149/1245 (12.0%)` | `1096/1245 (88.0%)` | `708` | `370` | `149` | `5` | Predominantly no-gold/corruption |
| P4 | `109/851 (12.8%)` | `742/851 (87.2%)` | `334` | `387` | `109` | `3` | Predominantly no-gold/corruption |

Most failures cannot be recovered by a first-line instruction or parser repair
because the raw output does not contain the gold event. Latent memory can help
correctness but can also destabilize generation: cases where Bank-off is clean
and Bank-on becomes a format failure are `100` for P7, `123` for P6, and `109`
for P4.

## Table 4. Context 4 Limitation

| Family/run | EM | Recall | Format failures | Taxonomy/routing headline | Interpretation |
| --- | ---: | ---: | ---: | --- | --- |
| P7 original | `0.000` | `0.190` | `98` | Dominant tuple `[1,0]` | Collapse |
| P7 rep2 | `0.000` | `0.120` | `94` | Dominant tuple `[1,0]` | Collapse |
| P7 rep3 | `0.000` | `0.140` | `95` | Dominant tuple `[1,0]` | Collapse |
| P7 rep4 | `0.030` | `0.350` | `83` | Complete diagnostics saved | Small EM recovery, still poor |
| P7 rep5 | `0.000` | `0.340` | `99` | Complete diagnostics saved | EM collapse persists |
| P7 mean (5) | `0.006` | `0.228` | `93.8` | `[1,0]` selected `300/300` in original three-run diagnostic subset | Severe limitation |
| P6 rep1 | `0.100` | `0.330` | `60` | Different construction/bank | Better recall and EM |
| P6 rep2 | `0.010` | `0.300` | `87` | Different construction/bank | Unstable |
| P6 rep3 | `0.170` | `0.320` | `43` | Different construction/bank | Best ctx4 repeat |
| P6 rep4 | `0.000` | `0.180` | `97` | Complete diagnostics saved | Collapse |
| P6 rep5 | `0.000` | `0.200` | `98` | Complete diagnostics saved | Collapse |
| P6 mean (5) | `0.056` | `0.266` | `77.0` | Better mean than P7 but unstable | Both new repeats zero EM |

Across the original three-run P7 context-4 taxonomy, `first_line_noise_no_gold`
accounts for approximately `247--249` cases and
`first_line_noise_later_gold` for `40`. The dominant tuple
has format-failure rate `0.957` and no-gold rate `0.867`. In the dominant
`0.08--0.09` top-score bucket (`n=230`), EM is `0`, recall `0.152`, format
failure `0.961`, and no-gold `0.865`. Context 4 is an extreme instance of the
global generation-corruption problem, not a unique parser bug.

## Experiment Inventory

Artifact key: `A` aggregate, `C` per-context JSONL, `Q` per-question JSONL,
`P` paired results, `S` score decomposition, `T` transition diagnostics,
`R` construction provenance, `F` frozen banks, `L` run log. `-` means absent.
All rows marked complete have 500/500 valid questions unless noted.

### Historical and Controlled Runs

| Experiment | Output root | Config `(rt,ut,slots,k)` | On EM / recall / format | Files | Status and decision |
| --- | --- | --- | --- | --- | --- |
| Initial A, 15-run per-context sweep | `outputs/mab/eventqa_frozen_context_bank_cfgA_ctx{0..4}` | `(0.03,0.05,8,1)` | `0.228 / 0.266 / 123` | per-context artifacts | Historical best at that stage; one-slot bridge |
| Initial B, 15-run per-context sweep | `outputs/mab/eventqa_frozen_context_bank_cfgB_ctx{0..4}` | `(0.03,0.09,16,1)` | `0.144 / 0.202 / 141` | per-context artifacts | Lower than A |
| Initial C, 15-run per-context sweep | `outputs/mab/eventqa_frozen_context_bank_cfgC_ctx{0..4}` | `(0.005,0.09,16,1)` | `0.134 / 0.208 / 165` | per-context artifacts | Lower than A/B |
| B top-k2 ctx0 pilot | `outputs/mab/eventqa_configB_ctx0_topk2` | `(0.03,0.09,16,2)` | `0.110 / 0.270 / 67` on 100 questions | `ACQP-----` | Pilot only; superseded by all-context run |
| B top-k2 all-context | `outputs/mab/eventqa_configB_allctx_topk2` | `(0.03,0.09,16,2)` | `0.218 / 0.290 / 131` | `ACQP-----` | Positive multi-slot signal; context-4 unstable |
| B top-k2 ctx4 rerun | `outputs/mab/eventqa_configB_ctx4_topk2_rerun` | `(0.03,0.09,16,2)` | `0.110 / 0.280 / 52` on 100 questions | `ACQP-----` | Dominant tuple changed; instability confirmed |
| B top-k4 | `outputs/mab/eventqa_configB_allctx_topk4` | `(0.03,0.09,16,4)` | `0.126 / 0.164 / 208` | `ACQP-----` | Negative; only three slots realized |
| A rerun | `outputs/mab/eventqa_configA_rt003_ut005_cap8_topk1_rerun` | `(0.03,0.05,8,1)` | `0.118 / 0.148 / 135` | `ACQP-----` | Reproducibility warning; not current anchor |
| B2 rerun | `outputs/mab/eventqa_configB_rt003_ut009_cap16_topk2_rerun` | `(0.03,0.09,16,2)` | `0.118 / 0.262 / 254` | `ACQP-----` | Reproducibility warning |
| D rerun | `outputs/mab/eventqa_configD_rt003_ut009_cap8_topk1` | `(0.03,0.09,8,1)` | `0.156 / 0.234 / 186` | `ACQP-----` | Diagnostic only |
| Controlled A rep1/2 | `outputs/mab/eventqa_controlled_A_rep{1,2}` | `(0.03,0.05,8,1)` | `0.154/0.226`; recall `0.212/0.236`; format `142/53` | `ACQPSTRF-` | Bank hashes differ; unstable Bank-on |
| Controlled B2 rep1/2 | `outputs/mab/eventqa_controlled_B2_rep{1,2}` | `(0.03,0.09,16,2)` | `0.184/0.134`; recall `0.288/0.250`; format `149/192` | `ACQPSTRF-` | Construction instability |
| Controlled D rep1/2 | `outputs/mab/eventqa_controlled_D_rep{1,2}` | `(0.03,0.09,8,1)` | `0.222/0.146`; recall `0.262/0.260`; format `145/189` | `ACQPSTRF-` | Construction instability |

No raw EventQA run with `construction_only=true` was found in the inventoried
roots. Threshold sensitivity is nevertheless preserved in
`outputs/mab/eventqa_score_diagnostics/construction_threshold_sensitivity.json`;
the absence of a distinct construction-only run root is recorded rather than
inferred away.

### P-series Search and Reproducibility

| Experiment | Output root | Config `(rt,ut,slots,k)` | On EM / recall / format | Chinese | Helpful/harmful/format harm | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| P1 | `outputs/mab/eventqa_p1_rt003_ut0095_cap16_topk1` | `(0.03,0.095,16,1)` | `0.172 / 0.232 / 183` | `81` | `83/1/30` | Best top-k1 in bounded P1--P3 |
| P2 | `outputs/mab/eventqa_p2_rt004_ut0095_cap16_topk1` | `(0.04,0.095,16,1)` | `0.080 / 0.190 / 265` | `103` | `40/4/55` | Reject |
| P3 | `outputs/mab/eventqa_p3_rt005_ut0095_cap16_topk1` | `(0.05,0.095,16,1)` | `0.112 / 0.188 / 195` | `101` | `53/1/38` | Reject |
| P4 initial | `outputs/mab/eventqa_p4_rt003_ut0095_cap16_topk2` | `(0.03,0.095,16,2)` | `0.172 / 0.262 / 141` | `221` | `85/3/45` | Control candidate; repeats lower |
| P5 | `outputs/mab/eventqa_p5_rt004_ut0095_cap16_topk2` | `(0.04,0.095,16,2)` | `0.162 / 0.268 / 202` | `226` | `80/3/53` | Below P6 |
| P6 initial diagnostic | `outputs/mab/eventqa_p6_rt005_ut0095_cap16_topk2` | `(0.05,0.095,16,2)` | `0.178 / 0.272 / 156` | `167` | `88/3/47` | Selected for reproducibility |
| P4 repro 1/2 | `outputs/mab/eventqa_p4_repro_rep{1,2}_rt003_ut0095_cap16_topk2` | `(0.03,0.095,16,2)` | `0.134/0.164`; recall `0.252/0.266`; format `295/195` | `288/255` | `67/4/59`; `82/4/51` | Lower control |
| P6 repro 1--5 | `outputs/mab/eventqa_p6_repro_rep{1..5}_rt005_ut0095_cap16_topk2` | `(0.05,0.095,16,2)` | `0.194/0.150/0.166/0.150/0.184`; recall `0.280/0.264/0.266/0.236/0.244`; format `149/187/185/171/137` | `153/196/217/202/181` | New: rep4 `74/3/43`, rep5 `91/3/30` | Lower-UT comparison |
| P7 original/rep2--5 | `outputs/mab/eventqa_p7*_rt005_ut010_cap16_topk2` | `(0.05,0.10,16,2)` | `0.200/0.212/0.162/0.190/0.220`; recall `0.252/0.240/0.212/0.266/0.298`; format `125/119/135/108/120` | `160/129/192/148/130` | New: rep4 `95/4/38`, rep5 `110/4/39` | Paper-level main |

Every P-series run above has `A,C,Q,P,S,T,R,F,L` present and completed.

### Prompt Ablation Roots

| Experiment | Output root | Prompt mode | On EM / recall / format | Decision |
| --- | --- | --- | ---: | --- |
| S4 | `outputs/mab/eventqa_strict_p4_rt003_ut0095_cap16_topk2` | strict | `0.078 / 0.230 / 244` | Negative |
| S6 | `outputs/mab/eventqa_strict_p6_rt005_ut0095_cap16_topk2` | strict | `0.088 / 0.224 / 291` | Negative |
| S7 | `outputs/mab/eventqa_strict_p7_rt005_ut010_cap16_topk2` | strict | `0.154 / 0.184 / 126` | Negative |
| FL-P4 | `outputs/mab/eventqa_firstline_p4_rt003_ut0095_cap16_topk2` | first-line | `0.080 / 0.260 / 296` | Negative |
| FL-P6 | `outputs/mab/eventqa_firstline_p6_rt005_ut0095_cap16_topk2` | first-line | `0.146 / 0.256 / 215` | Negative |
| FL-P7 | `outputs/mab/eventqa_firstline_p7_rt005_ut010_cap16_topk2` | first-line | `0.090 / 0.182 / 193` | Strong negative |

All prompt-ablation roots have `A,C,Q,P,S,T,R,F,L` present and completed. The
strict P6/P7 roots contain abandoned partial timestamp directories; the table
uses the timestamp directory containing the complete 500-question aggregate.

## Chronological Experiment Timeline

1. **Initial A/B/C sweep (2026-06-30).** Tested capacity/update alternatives at
   `top_k=1`. A led with EM `0.228`; larger banks B/C did not help. Historical
   decision: retain A while investigating multi-slot retrieval.
2. **B top-k2 and top-k4.** Top-k2 raised recall and gave a positive multi-slot
   signal, but context 4 collapsed and a standalone rerun changed the dominant
   tuple. Top-k4 regressed and realized only three retrieved slots. Decision:
   avoid top-k4; instrument construction and routing.
3. **Controlled A/B2/D reruns (2026-07-01).** Per-context reseeding preserved
   RNG state, but all frozen-bank hashes differed between repeats and score
   divergence appeared around construction turn 1. Decision: treat Bank-on as
   construction-path unstable and rely on repeat statistics.
4. **Offline score diagnostic.** Scores occupied a narrow range and did not
   cleanly distinguish useful from harmful memory. Fixed thresholds alone were
   judged insufficient; margin/hysteresis/utility-aware routing became plausible.
5. **Bounded P1--P6 search.** Fixed `max_slots=16`, `ut=0.095`, and crossed
   `rt={0.03,0.04,0.05}` with `top_k={1,2}`. P6 was the best single bounded
   candidate (`0.178` EM, `0.272` recall), motivating reproducibility runs.
6. **P4/P6 reproducibility.** The initial three P6 repeats averaged EM `0.170`
   and recall `0.270`; P4 was lower. This motivated retaining P6 as the early
   recall-oriented reference.
7. **P7 update-threshold test and repeats (2026-07-02).** Raising `ut` from
   `0.095` to `0.10` improved mean EM and reduced format failures, but lowered
   recall. Decision: promote P7 as current main; keep P6 as a trade-off.
8. **P7 context-4 diagnosis.** P7 reached zero EM in context 4 across the first
   three repeats with fixed tuple `[1,0]`. After five repeats, context-4 EM is
   `0/0/0/0.03/0`; the limitation remains material.
9. **Format taxonomy and all-context mechanism diagnosis.** Between `87%` and
   `95%` of examined family failures were no-gold. Decision: parser repair and
   first-line prompt constraints cannot address the dominant failure.
10. **Strict prompt ablation.** Strict official-style instructions reduced P7
    EM/recall. Decision: negative ablation.
11. **First-line prompt ablation.** Lightweight first-line instructions further
    reduced P7 EM and increased failures. Decision: strong negative ablation.
12. **Official prompt/scorer verification.** Confirmed P7 non-strict uses the
    local upstream default wrapper and unchanged official scorer/parser path.
13. **Stage consolidation (2026-07-03).** Frozen current conclusion in
    `eventqa_current_stage_consolidated_summary.{md,json}`; this note now makes
    that conclusion durable for project continuation and paper writing.
14. **Five-repeat stability completion (2026-07-03).** Added P7 rep4/rep5 and
    P6 rep4/rep5. P7 now has EM `0.197±0.020`, recall `0.254±0.028`, and
    format failures `121.4±8.8`; P6 has `0.169±0.018`, `0.258±0.016`, and
    `165.8±19.8`. Decision: strengthen P7 to paper-level main candidate and
    retain P6 as the lower-UT comparison rather than a meaningful recall winner.

## Diagnostic Reports and Artifact Pointers

- Frozen stage summary:
  `outputs/mab/eventqa_current_stage_consolidated_summary.{md,json}`
- P7 versus P6/P4 repeat summary:
  `outputs/mab/eventqa_p7_vs_p6_final_summary.{md,json}`
- P7 context-4 diagnosis:
  `outputs/mab/eventqa_p7_context4_failure_diagnosis.{md,json}`
- Format taxonomy:
  `outputs/mab/eventqa_format_failure_taxonomy.{md,json}`
- All-context format/mechanism diagnosis:
  `outputs/mab/eventqa_all_context_format_mechanism_diagnosis.{md,json}`
- P7 default prompt/scorer verification:
  `outputs/mab/eventqa_p7_non_strict_official_prompt_scorer_verification.{md,json}`
- Score diagnostics:
  `outputs/mab/eventqa_score_diagnostics/`
- P6 stability diagnostics:
  `outputs/mab/eventqa_p6_stability_diagnostics/`
- Comprehensive inventory generated with this update:
  `outputs/mab/eventqa_recent_experiments_full_inventory_and_notes_update.{md,json}`
- Five-repeat stability checkpoint:
  `outputs/mab/eventqa_five_repeat_stability_summary.{md,json}`

## Claims Safe for Paper Drafting

- **Supported:** P7 improves repeat-mean official EventQA EM and lowers format
  failures relative to P6 under the unchanged official scorer/parser and local
  upstream default EventQA prompt wrapper.
- **Supported:** P7 maintains recall comparable to P6; the five-repeat mean gap
  is only `0.0044` in P6's favor.
- **Supported negative result:** strict and first-line prompt-only controls do
  not improve official EM in this setting.
- **Supported diagnosis:** the observed family-level failure pool is dominated
  by raw outputs that do not contain the gold event; parser-sensitive cases are
  a minority.
- **Partially supported mechanism hypothesis:** fixed dominant routing and a
  pathological context-local tuple are associated with P7 context-4 failure.
  Causality and a general suppression rule remain unproven.

## Limitations

- P7 is the current main candidate, not a final SOTA claim.
- P7 and P6 remain close on recall; five repeats do not support a meaningful
  P6 recall advantage.
- Frozen banks differ across repeated Bank-on runs despite per-context
  reseeding; repeat statistics are required.
- P7 context 4 remains near zero EM (`0.006` mean); P6 is better there on
  average (`0.056`) but rep4 and rep5 are both zero.
- Fixed routing and narrow score ranges indicate that the current retrieval
  score does not reliably separate useful from harmful memory.
- The compressed Bank-off bridge is not an official full-history baseline.
- Harmful-slot/tuple suppression has not yet been implemented or evaluated.

## Next Steps

Do not continue prompt-only formatting variants and do not modify the official
parser/scorer for the main result. The next focused mechanism stage should:

1. Attribute utility and corruption to selected slots and slot tuples.
2. Test whether failures concentrate in repeatable harmful tuples across banks.
3. Evaluate a minimal opt-in utility gate, routing-margin gate, or harmful-tuple
   suppression mechanism only after the attribution criterion is fixed.
4. Preserve P7 as the paper-level main candidate and P6 as the lower-UT
   comparison / earlier recall-oriented baseline.
5. Use bounded experiments with repeats; do not infer stability from one bank.

## Paper-ready Mini Summary

Under the unchanged official MemoryAgentBench EventQA scorer/parser, P7
non-strict is the current main candidate and uses the default EventQA prompt
wrapper character-identical to the local upstream snapshot. Across five runs,
P7 improves official exact match (`0.197±0.020` vs. `0.169±0.018`) and reduces
format failures (`121.4±8.8` vs. `165.8±19.8`) relative to P6 while maintaining
comparable recall (`0.254±0.028` vs. `0.258±0.016`). P6 is retained as the
lower-update-threshold comparison. Strict and first-line prompt controls are negative
ablations. Error analysis shows that most remaining failures are no-gold,
memory-conditioned generation corruption rather than parser mismatch, making
harmful-slot and harmful-tuple attribution and suppression the next mechanism
direction.

## Project cleanup review checkpoint

- **Date/time:** 2026-07-03 19:40 CST.
- **Current paper-critical files:**
  `scripts/eval/mab6b_weaver_space_bank_eventqa_65536_n5.py`,
  `scripts/eval/mab2_mab_bridge.py`, the memory-bank/model integration under
  `memgen/model/`, `tests/test_mab6b_weaver_space_bank.py`,
  `tests/test_latent_memory_bank.py`,
  `tests/test_latent_memory_bank_integration.py`, and this EventQA note.
- **Current paper-critical outputs:** all P7 and P6 five-repeat run families;
  P4 original/repeat controls; strict and first-line ablation families;
  `eventqa_five_repeat_stability_summary.{md,json}`;
  `eventqa_current_stage_consolidated_summary.{md,json}`;
  `eventqa_p7_vs_p6_final_summary.{md,json}`; official prompt/scorer
  verification; format taxonomy; all-context mechanism diagnosis; and P7
  context-4 diagnosis.
- **Must not be touched:** the protected DetectiveQA canonical note
  `research_notes/benchmarks/memoryagentbench_mab6b_weaver_space_bank.md`,
  accepted canonical result artifacts, official MemoryAgentBench scorer/parser,
  and existing experiment outputs.
- **Cleanup candidates requiring user approval:** three incomplete 8 KB
  `eventqa_A_stability_rep{1,2,3}_rt003_ut005_cap8_topk1` log-only directories;
  surplus frozen-context launch directories after an exact manifest audit;
  root file `1` (nonempty path listing); runtime tmux logs; Python bytecode
  caches; older sweep outputs for archival only; and untracked non-mainline
  diagnostic/plan files.
- **Next recommended action:** review and freeze a paper artifact manifest with
  paths, completion markers, note references, sizes, and checksums; then obtain
  explicit approval for the Level 1 deletion and Level 2 archive lists before
  changing any artifact location.
- **Current result reminder:** P7 non-strict remains the paper-level main
  EventQA candidate at five repeats (`EM 0.197±0.020`,
  `Recall 0.254±0.028`, format failures `121.4±8.8`); P6 remains the
  lower-update-threshold comparison.
