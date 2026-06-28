# MAB-6B Error Analysis: detective_qa Version B Weaver-space Bank n10

## Scope

Artifact-only analysis of:

`outputs/mab/version_b_weaver_space_bank_detectiveqa_n10/20260625T122323Z-detectiveqa-version-b-weaver-space-bank-n10`

No new inference was run. The comparison target is the canonical MAB-6A run:

`outputs/mab/version_b_weaver_conditioned_detectiveqa_n10/20260625T023822Z-detectiveqa-version-b-weaver-conditioned-n10`

## Summary

MAB-6B improved one sample and raised official Bank-on exact match from `0.0`
to `0.1`. The gain came from context `7`, where the Bank-on output contained
the exact gold option string `A. Bad eyesight`. The output was still noisy
(`| A. Bad eyesight\n\n| ...`), but the official scorer accepted it.

Across the nine failed samples, the dominant failure modes were refusal or
no-context answers, template / JSON leakage, language drift, and wrong-option
content. The run is mechanism-active, but the error profile is still dominated
by output-control failures rather than clean multiple-choice answer selection.

## Improved Sample

| context | gold answer | MAB-6A Bank-on | MAB-6B Bank-on | diagnosis |
| --- | --- | --- | --- | --- |
| 7 | `A. Bad eyesight` | `1\n 由于没有提供具体的背景信息或...` | `| A. Bad eyesight\n\n| 果...` | MAB-6B retrieved Weaver-space memory made the correct option text available to generation. The answer format remained noisy, but the exact gold answer string appeared early enough to score correct. |

The improvement is not explained by a stronger retrieval score. Context `7`
had query-turn score `0.056665`, the second-lowest score in the run.

## Failed Sample Classification

| context | gold answer | MAB-6B Bank-on output excerpt | primary failure | secondary signal |
| --- | --- | --- | --- | --- |
| 0 | `C. The Brandt couple` | `In the provided context, there is no mention of...` | refusal/no-context answer | no option selected |
| 1 | `D. Was killed by the mistress, Blacklock` | `"Attempted murder and suicide"\n\nReasoning...` | wrong option | answer-format failure; free-form answer instead of option |
| 2 | `B. He is not satisfied with this job.` | `. (No output needed, as the question is...` | answer-format failure | refusal/no-answer behavior |
| 3 | `A. Abandoned and left behind` | `不需要翻译，直接输出答案和理由。` | language drift | answer-format failure; no option selected |
| 4 | `D. Extorted by Field` | `(The provided context is not relevant to the question...` | refusal/no-context answer | no option selected |
| 5 | `A. Blue Hat Stranger` | `json block: {"answer": "...` | JSON/template leakage | incomplete answer |
| 6 | `C. Misty Sketches` | `�单词表\n\n{"answer": "B...` | JSON/template leakage | wrong option; language drift |
| 8 | `A. Adore` | `: {"answer":"B. Resentment",...` | wrong option | JSON/template leakage |
| 9 | `C. Xīchuān de mèimei` | `No search results found. Unable to provide an...` | refusal/no-context answer | no option selected |

Bucket counts using the primary label:

| bucket | count |
| --- | ---: |
| refusal/no-context answer | 3 |
| wrong option | 2 |
| answer-format failure | 1 |
| JSON/template leakage | 2 |
| language drift | 1 |

## Slot Dynamics

All contexts collapsed to one final slot under the Weaver-space bank path.

| context | final slots | insert | replace_matched | retrieved index | query score |
| --- | ---: | ---: | ---: | --- | ---: |
| 0 | 1 | 1 | 25 | `[0]` | 0.069670 |
| 1 | 1 | 1 | 25 | `[0]` | 0.078031 |
| 2 | 1 | 1 | 25 | `[0]` | 0.071528 |
| 3 | 1 | 1 | 29 | `[0]` | 0.068277 |
| 4 | 1 | 1 | 29 | `[0]` | 0.072921 |
| 5 | 1 | 1 | 27 | `[0]` | 0.080353 |
| 6 | 1 | 1 | 30 | `[0]` | 0.060845 |
| 7 | 1 | 1 | 41 | `[0]` | 0.056665 |
| 8 | 1 | 1 | 50 | `[0]` | 0.053414 |
| 9 | 1 | 1 | 35 | `[0]` | 0.074315 |

Aggregate slot dynamics:

- final slot counts: `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`
- write action counts: `{"insert": 10, "replace_matched": 316}`
- update reason counts: `{"empty_bank": 10, "matched_thread": 316}`
- retrieved indices: always `[0]`
- query score range: `0.053414` to `0.080353`
- correct-sample query score: `0.056665`
- failed-sample query score mean: `0.069928`

This indicates that MAB-6B creates a single rolling Weaver-space memory thread
per context. Capacity pressure disappeared, but thread diversity also
disappeared.

## Retrieval Score Correlation

There is no positive correlation between query-turn retrieval score and
correctness in this n10 run.

- Pearson correlation between query score and exact-match correctness:
  approximately `-0.468`
- The only correct sample had score `0.056665`.
- Failed samples included higher scores up to `0.080353`.

This argues against a simple higher-threshold sweep as the first follow-up. A
threshold increase would likely remove the one correct sample before removing
the highest-score failures.

## MAB-6A vs MAB-6B

| context | gold answer | MAB-6A Bank-on | MAB-6B Bank-on | change |
| --- | --- | --- | --- | --- |
| 0 | `C. The Brandt couple` | Chinese meta-answer about death cause | no-context refusal | still failed |
| 1 | `D. Was killed by the mistress, Blacklock` | Chinese no-options response | free-form wrong answer | still failed |
| 2 | `B. He is not satisfied with this job.` | JSON-like `B...` answer fragment | no-output answer | still failed |
| 3 | `A. Abandoned and left behind` | JSON-like `C...` answer fragment | Chinese instruction/meta response | still failed |
| 4 | `D. Extorted by Field` | motive not stated | no-context refusal | still failed |
| 5 | `A. Blue Hat Stranger` | Chinese meta-answer | JSON leakage | still failed |
| 6 | `C. Misty Sketches` | language drift with `A...` | JSON leakage with `B...` | still failed |
| 7 | `A. Bad eyesight` | no-context / missing-background response | exact gold option appears | improved |
| 8 | `A. Adore` | `no` plus question restatement | JSON wrong option `B. Resentment` | still failed |
| 9 | `C. Xīchuān de mèimei` | Chinese "no answer in brackets" meta-response | no search results refusal | still failed |

MAB-6B changed the failure surface but did not cleanly solve answer formatting.
It improved one context by surfacing the exact option string, while other
contexts remained dominated by meta-responses, refusals, and template leakage.

## Smallest Next Ablation

Recommended smallest next ablation: **answer-format repair**.

Rationale:

- One MAB-6B sample became correct despite noisy formatting, so output format is
  now a bottleneck visible in the successful case as well as the failures.
- Several failures are not evidence-retrieval failures in isolation; they are
  no-context refusals, JSON/template leakage, or language drift.
- Retrieval score does not correlate positively with correctness, so a
  threshold sweep is unlikely to be the clean first move.
- Insert-only would directly test the slot-collapse issue, but it changes
  memory dynamics more broadly than an answer-format repair.
- No construction-time retrieval is useful later for isolating whether the
  rolling single-slot thread is over-writing useful state, but it is a larger
  mechanism change than output repair.

Concrete ablation target:

- keep MAB-6B routing unchanged;
- keep `retrieve_threshold=0.03`, `update_threshold=0.05`, `top_k=1`,
  `max_slots=8`, and `memory_bank_storage_space=weaver`;
- add only a constrained final-answer formatting repair at the query answer
  boundary, if approved as a separate run.

Do not promote MAB-6B to the default path from this n10 result alone. The
official exact-match gain is real on this slice, but it should be replicated
after the output-format failure mode is isolated.
