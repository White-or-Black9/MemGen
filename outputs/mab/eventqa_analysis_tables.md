# EventQA Analysis Tables

## Context-wise Table

| Context | Bank-off EM | Bank-off Recall | P6 EM | P6 Recall | P7 EM | P7 Recall | P7 Format Failures | P7-Bank-off EM | P7-P6 EM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ctx0 | 0.000 | 0.150 | 0.194±0.073 | 0.236±0.018 | 0.248±0.020 | 0.258±0.016 | 3.6±2.9 | +0.248 | +0.054 |
| ctx1 | 0.000 | 0.250 | 0.264±0.086 | 0.388±0.019 | 0.382±0.065 | 0.404±0.035 | 8.2±5.6 | +0.382 | +0.118 |
| ctx2 | 0.000 | 0.150 | 0.180±0.048 | 0.202±0.037 | 0.222±0.017 | 0.234±0.012 | 5.2±1.7 | +0.222 | +0.042 |
| ctx3 | 0.030 | 0.150 | 0.150±0.071 | 0.198±0.020 | 0.126±0.054 | 0.144±0.046 | 10.6±2.4 | +0.096 | -0.024 |
| ctx4 | 0.010 | 0.190 | 0.056±0.068 | 0.266±0.063 | 0.006±0.012 | 0.228±0.098 | 93.8±5.7 | -0.004 | -0.050 |

## Transition Table

| Method | Repeats | Helpful | Harmful | Unchanged | Format-harm | Net gain |
|---|---:|---:|---:|---:|---:|---:|
| P6 non-strict | 5 | 83.6±9.4 | 3.2±0.7 | 413.2±9.9 | 44.6±9.0 | 80.4±8.9 |
| Frozen P7 non-strict | 5 | 98.4±10.1 | 4.0±0.0 | 397.6±10.1 | 28.4±9.3 | 94.4±10.1 |

Format-harm is a diagnostic subset rather than an additional partition bucket.
