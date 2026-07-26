# EventQA no-decay ablation

This is the pre-specified P7 ablation with `decay_alpha=0.0`.  It keeps the
formal P7 contract unchanged: frozen session-local Weaver-space bank, 16
slots, top-k 2, retrieval threshold 0.05, update threshold 0.10, CPU storage,
read-only query-time content writes, the original EventQA prompt/scorer, and
40 generated tokens.

`eventqa_p7_no_decay.py` rejects an explicit `--decay-alpha` override.  The
full runner executes the five aligned process-level passes with seeds
`42,142,242,342,442`; it must be run on an otherwise idle GPU for any
paper-facing cost measurement.  Effectiveness comparison is against formal
P7 only, never against a selected decay sweep.

Minimal smoke command:

```bash
CUDA_VISIBLE_DEVICES=<GPU> /home/baishilong/miniconda3/envs/memgen/bin/python \
  eval/exp/no_decay/eventqa_p7_no_decay.py \
  --output-root outputs/mab/no_decay_smoke \
  --requested-contexts 1 --context-index 0 --question-limit 10 \
  --seed 42 --reseed-per-context --max-slots 16 --top-k 2 \
  --retrieve-threshold 0.05 --update-threshold 0.10 --skip-research-note
```
