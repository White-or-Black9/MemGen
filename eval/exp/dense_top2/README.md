# EventQA dense top-2 retrieved-text baseline

This runner is the semantic counterpart of `bm25_top2`: it indexes only the
current EventQA context with frozen local E5 embeddings, scores every parent
chunk by the maximum cosine of its 500-E5-token windows, and injects the two
selected *full* parent chunks into the unchanged bank-off EventQA prompt.

It is deliberately not a P7 variant: no latent bank, external corpus, answer,
candidate text, ANN index, or query-time write is used.

Run a smoke on an idle GPU:

```bash
CUDA_VISIBLE_DEVICES=<GPU> /home/baishilong/miniconda3/envs/memgen/bin/python \
  eval/exp/dense_top2/eventqa_dense_retrieved_text.py \
  --measurement-scope smoke --context-index 0 --question-limit 10 \
  --embedding-device cpu --output-root outputs/mab/eventqa_dense_top2_smoke
```

The first paper-facing pass must use contexts 0--4 with 100 questions each and
the same base-seed / per-context reseeding schedule as the existing controls.

`run_eventqa_full_pass.sh` performs that one 500-question effectiveness pass
and emits `aggregate.json`.  It deliberately does not label its timing or GPU
memory numbers as paper-facing unless it is launched on an otherwise idle GPU.
