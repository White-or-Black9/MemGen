# Submitted-paper baseline: independently recomputed results

Source: `review/nference_Time_Latent_Me.pdf`, pages 5–7. Population standard deviation; 500 unique questions repeated five times.

| Method | EM | Raw recall | Heuristic format flags |
|---|---:|---:|---:|
| p7 | 0.1880 ± 0.0473 | 0.2308 ± 0.0419 | 118.6000 ± 19.1478 |
| recent_text | 0.0340 ± 0.0000 | 0.0960 ± 0.0000 | 211.0000 ± 0.0000 |
| rolling_summary | 0.0120 ± 0.0000 | 0.0780 ± 0.0000 | 267.0000 ± 0.0000 |
| bm25_top2 | 0.0300 ± 0.0000 | 0.2260 ± 0.0000 | 265.0000 ± 0.0000 |
| dense_top2 | 0.0300 ± 0.0000 | 0.2400 ± 0.0000 | 285.0000 ± 0.0000 |
| matched16 | 0.0680 ± 0.0000 | 0.1800 ± 0.0000 | 347.0000 ± 0.0000 |
| no_retrieved_memory_conditioning | 0.0080 ± 0.0000 | 0.1780 ± 0.0000 | 377.0000 ± 0.0000 |
| direct_top1 | 0.0472 ± 0.0079 | 0.1864 ± 0.0087 | 354.6000 ± 21.3317 |
| no_decay | 0.1628 ± 0.0470 | 0.2508 ± 0.0295 | 189.0000 ± 52.7523 |

Format flags are not benchmark parser failures. No parsed-only accuracy is inferred from these counts.

| Method | End-to-end seconds | Seconds/question | Max incremental GiB | Campaign GPU |
|---|---:|---:|---:|---:|
| p7 | 377.06 ± 10.49 | 0.754 ± 0.021 | 0.160 | 5 |
| recent_text | 2688.90 ± 9.50 | 5.378 ± 0.019 | 13.094 | 5 |
| rolling_summary | 598.41 ± 16.34 | 1.197 ± 0.033 | 1.782 | 5 |
| bm25_top2 | 645.76 ± 4.30 | 1.292 ± 0.009 | 3.505 | 5 |
| matched16 | 497.87 ± 18.68 | 0.996 ± 0.037 | 0.159 | 5 |
| dense_top2 | 739.36 ± 5.32 | 1.479 ± 0.011 | 3.505 | 4 |

Timing includes method preparation and queries, excludes model loading. Peak allocation is incremental GPU allocation, not total VRAM or CPU bank storage.

## Gate

PASS_WITH_LIMITATIONS: existing predictions can anchor offline reanalysis; exact historical executable reconstruction is incomplete.
