# EventQA Final Comparison Package

## Main Table

| Method | Repeats | EM | Recall | Format failures | Notes |
|---|---:|---:|---:|---:|---|
| Disabled / compressed Bank-off | 5 | 0.008±0.000 | 0.178±0.000 | 377.0±0.0 | five-repeat Bank-off row reconstructed from frozen P7 paired artifacts |
| Same-model text-summary memory | 1 | 0.012 | 0.078 | 267.0 | one deterministic full pass; negative same-model baseline |
| BM25 top-2 retrieved text | 1 | 0.030 | 0.226 | 265.0 | one deterministic full pass |
| 16-token matched-budget retrieved text | 1 | 0.068 | 0.180 | 347.0 | one deterministic full pass; exact 16-token visible budget |
| P6 non-strict | 5 | 0.169±0.018 | 0.258±0.016 | 165.8±19.8 | five-repeat lower-update-threshold comparator |
| P7 with query retrieval disabled | 1 | 0.008 | 0.178 | 377.0 | one deterministic full pass; all query retrieval disabled |
| Frozen P7 non-strict | 5 | 0.197±0.020 | 0.254±0.028 | 121.4±8.8 | five-repeat main result |

## Cost Table

| Method | End-to-end s | s/question | Peak GPU bytes | Paper-facing | Notes |
|---|---:|---:|---:|:---:|---|
| Disabled / compressed Bank-off | 367.448 | 0.735 | 149836288 | yes | method-separable same-GPU serialized full pass |
| Same-model text-summary memory | 691.345 | 1.383 | 209979904 | no | shared-GPU-confounded; not paper-facing |
| BM25 top-2 retrieved text | 692.845 | 1.386 | 3772054528 | yes | one deterministic full pass |
| 16-token matched-budget retrieved text | 501.761 | 1.004 | 179276800 | yes | one deterministic full pass |
| P7 with query retrieval disabled | 445.004 | 0.890 | 149811712 | yes | one deterministic full pass |
| Frozen P7 non-strict | 387.999 | 0.776 | 180270080 | yes | method-separable same-GPU serialized full pass |

## Claim Audit

- `p7_vs_disabled`: supported; P7 EM/recall 0.1968/0.2536 vs Disabled 0.0080/0.1780
- `p7_vs_p6`: supported; P7-P6 deltas: EM +0.0280, recall -0.0044, format failures -44.4
- `p7_beats_explicit_controls`: supported; P7 exceeds text-summary, BM25 top-2, matched16, and no-query-retrieval on both EM and recall.
- `query_time_retrieval_is_necessary`: supported; No-query-retrieval exactly matches Disabled effectiveness while preserving the constructed bank.
- `p7_cost_superiority`: not supported; P7 adds measured overhead over Disabled, while explicit-text baselines use different cost profiles; no blanket cost-superiority claim is supported.
