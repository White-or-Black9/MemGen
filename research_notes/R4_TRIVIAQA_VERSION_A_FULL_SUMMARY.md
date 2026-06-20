# R4 TriviaQA Version A Full Evaluation Summary

## Status

- Date preserved: 2026-06-20
- Status: completed negative result; further TriviaQA ablations paused by user
- Resume label: **Version A full TriviaQA negative result, mechanism-active but
  policy-unstable.**
- No follow-up ablation has been started.
- Version B remains deferred.

## Experiment Context

This R4 study evaluates the current session-local Version A latent memory bank
against the disabled-memory MemGen path on the complete TriviaQA validation
split (`rc.wikipedia.nocontext`). Version A stores Reasoner-space
`latent_inputs_embeds`, injects retrieved memory into the Reasoner only, keeps
memory session-local, and does not expose retrieved memory to the Weaver.

The full comparison is paired by sample index. Both modes use all 7,993
validation samples in the denominator, including invalid/retrieval-blocked
samples.

## Configuration

- memory mode: `version_a_aligned`
- threshold: `0.04`
- top_k: `1`
- batch_size: `1`
- seed: `42`
- temperature: `0.0`
- max_response_length: `1024`
- retrieval_topk: `3`
- dataset: TriviaQA validation / `rc.wikipedia.nocontext`
- retrieval: local Search-R1 endpoint used by the completed run
- checkpoint: TriviaQA Weaver-SFT
  `Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model`

## Artifact Index

### Disabled Full Baseline

- root: `outputs/r4_triviaqa_full_chunks/`
- completed original chunks: `disabled_s0000_0999` through
  `disabled_s6000_6999`
- completed retry chunks:
  - `disabled_s7000_7499_retry`
  - `disabled_s7500_7799_retry`
  - `disabled_s7800_7992_retry`
- the original no-artifact `disabled_s7000_7992` attempt was preserved and
  excluded from the final aggregate

### Version A Full Rerun

- root:
  `outputs/r4_triviaqa_full_version_a_t004_chunks_250_fullrerun/`
- 32 chunks, 250 samples each except final chunk `7750..7992` with 243
- every chunk contains:
  - `run_config.json`
  - `evaluate/answer.json`
  - `summary.json`
  - `memory_trace.json`

### Paired and Failure Analysis

- root: `outputs/r4_triviaqa_full_version_a_t004_analysis/`
- `version_a_full_summary.json`
- `paired_transition_table.json`
- `paired_per_sample.jsonl`
- `rescues_top20.json`
- `regressions_top20.json`
- `memory_stats.json`
- `failure_analysis.json`
- `failure_analysis.md`

Raw `evaluate/answer.json` files are JSONL despite the `.json` suffix. Real
sample records contain `sample_index`; the terminal summary object must not be
counted as a sample.

## Coverage and Full Result

| Mode | Correct | Total | Accuracy | Valid | Invalid / blocked | Missing | Duplicates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Disabled | 5148 | 7993 | 0.6440635556 | 7970 | 23 | 0 | 0 |
| Version A | 5092 | 7993 | 0.6370574252 | 7970 | 23 | 0 | 0 |

- accuracy delta: `-0.0070061304` (`-0.7006` percentage points)
- net correct change: `-56`
- question mismatches: `0`
- gold-answer mismatches: `0`
- denominator rule: all `7993` samples for both modes

## Paired Transition Table

| Transition | Count |
|---|---:|
| disabled wrong -> Version A correct (rescue) | 53 |
| disabled correct -> Version A wrong (regression) | 109 |
| stable correct | 5039 |
| stable wrong | 2792 |

Regressions are approximately `2.06x` as frequent as rescues.

## Memory Statistics

- mean writes: `2.102965`
- mean retrieve attempts: `1.102965`
- mean retrieved latent count: `2.973602`
- median retrieved latent count: `0`
- samples with retrieve attempts: `7971`
- samples receiving latent injection: `2417`
- threshold-passed samples: `2417/7993` (`30.24%`)
- mean per-sample max score: `0.034162`
- maximum max score: `0.082211`
- max score available: `7971/7993`
- update action occurrences:
  - `insert`: `13838`
  - `replace_matched`: `2971`

Most common update traces:

| Trace | Samples |
|---|---:|
| `insert -> insert` | 5483 |
| `insert -> replace_matched` | 2058 |
| `insert -> replace_matched -> replace_matched` | 185 |
| `insert -> replace_matched x4` | 88 |
| `insert x5` | 54 |

## Score Bucket Finding

| Version A max score | Net gain |
|---|---:|
| no score | 0 |
| `<0.04` | 0 |
| `0.04 <= score < 0.045` | +2 |
| `0.045 <= score < 0.05` | -12 |
| `0.05 <= score < 0.055` | -27 |
| `0.055 <= score < 0.06` | -14 |
| `>=0.06` | -5 |

`max_score` is not a reliable correctness or confidence signal. The only
slightly positive active score bucket is `0.04..0.045`; every higher bucket is
negative. A threshold-only increase to `0.05`, `0.055`, or `0.06` is therefore
not supported by this post-hoc evidence.

## Injection Count Finding

With `top_k=1`, one retrieved memory slot corresponds to approximately eight
latent tokens. `retrieved_latent_count` is cumulative across injections:

- `0`: no injection
- `8`: one injection
- `16`: two injections
- `24`: three injections
- `32+`: four or more injections

| Retrieved latent count | Rescue | Regression | Net |
|---|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 8 | 45 | 59 | -14 |
| 16 | 7 | 9 | -2 |
| 24 | 1 | 3 | -2 |
| 32+ | 0 | 38 | -38 |

| Retrieve count | Rescue | Regression | Net |
|---|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 45 | 56 | -11 |
| 2 | 6 | 8 | -2 |
| 3 | 0 | 1 | -1 |
| 4+ | 2 | 44 | -42 |

Repeated latent injection is the strongest observed failure signal. The
`32+` latent path produced no rescues and 38 regressions. The `4+` retrieve
path accounts for net `-42` of the overall net `-56` result.

## Failure Taxonomy

The taxonomy is a mutually exclusive, deterministic coarse classification for
diagnostic prioritization. It is not a manually adjudicated causal label.
`retrieval_confusion` requires the wrong answer string to appear in the saved
retrieved evidence.

### Regressions

| Category | Count |
|---|---:|
| verbose / malformed output | 42 |
| retrieval confusion | 26 |
| answer changed to question term | 20 |
| over-specific or under-specific answer | 11 |
| entity substitution | 9 |
| unknown other | 1 |

### Rescues

| Category | Count |
|---|---:|
| evidence entity fix | 30 |
| answer specificity fix | 11 |
| incomplete response to answer | 7 |
| unknown other | 4 |
| normalization fix | 1 |

Representative cases remain sample 53 (rescue: Normand Poirier -> Seymour
Hersh) and sample 21 (regression: Dangerous Minds -> Gangsta's Paradise).

## Interpretation

The current Version A full TriviaQA result is negative but informative.

- The mechanism is active, not inert: it changes answers and produces 53 real
  rescues.
- The current policy is unreliable: 109 regressions dominate the rescues.
- Samples without latent injection are stable relative to disabled.
- Even one eight-latent injection is mildly negative in aggregate.
- Repeated injection is sharply harmful and is the main source of the full-run
  degradation.
- Higher `max_score` does not imply a safer or more useful memory injection.
- The observed behavior is better described as brittle latent steering than
  reliable evidence-grounded memory.

Authoritative short conclusion:

> Version A full TriviaQA negative result, mechanism-active but policy-unstable.

## Paused Status

- Further TriviaQA ablation work is paused by user decision as of 2026-06-20.
- Do not start a threshold-only sweep.
- Do not start Version B.
- Do not reinterpret exploratory 20..179 slices as the final result; the full
  7,993-sample paired result is authoritative.
- Preserve raw outputs and analysis files unchanged.

## Future Ablation Priority

When work resumes, evaluate one change at a time, behind an explicit default-off
config flag, while protecting disabled-path equivalence.

1. Allow at most one latent injection per sample.
2. Cap cumulative `retrieved_latent_count` at `8`.
3. Suppress repeated `replace_matched` / repeated retrieval injection.
4. Add an answer-preserving confidence gate.
5. Delay writes until external evidence is present or use an evidence-aware
   write gate.
6. Calibrate score semantics before considering any threshold-only rerun.

The first recommended continuation is the max-one-injection ablation. It has
not been implemented or started.

## Open Questions

- Does limiting each sample to one injection preserve the 45 single-injection
  rescues while removing repeated-injection failures?
- Is cumulative latent volume itself harmful, or is the failure specifically
  caused by repeated `replace_matched` updates?
- Can an answer-confidence gate protect already-correct short answers without
  freezing incorrect early candidates?
- Can evidence-aware writes reduce question-term copying and retrieved-evidence
  distractor substitution?
- Why is `max_score` anti-correlated with useful behavior in the higher score
  buckets, and what representation similarity does it actually measure?
- Does TriviaQA underestimate the value of session-local memory because most
  samples are independent single-question QA sessions?

## Do Not Forget on Resume

1. Read this file first, then `failure_analysis.md` and
   `version_a_full_summary.json`.
2. The disabled reference is `5148/7993`; Version A is `5092/7993`.
3. Use all 7,993 samples in the denominator, including 23 invalid/blocked runs.
4. Do not count the terminal summary object in JSONL `answer.json` files.
5. `retrieved_latent_count=32+` means repeated injections, not one long memory.
6. Do not assume higher `max_score` means higher answer confidence.
7. Do not start with a broad threshold sweep; the score buckets argue against
   it.
8. The strongest next ablation is max one injection, followed by a cumulative
   eight-latent cap.
9. Keep all new ablations default-off and preserve disabled-path equivalence.
10. Advance only one experimental stage, then stop for confirmation.
