# EventQA dense top-2 retrieved-text baseline

This runner is the semantic counterpart of `bm25_top2`: it indexes only the
current EventQA context with frozen local E5 embeddings, scores every parent
chunk by the maximum cosine of its 500-E5-token windows, and injects the two
selected *full* parent chunks into the unchanged bank-off EventQA prompt.

It is deliberately not a P7 variant: no latent bank, external corpus, answer,
or ANN index is used. The retrieval query is the unchanged official EventQA
question input, including its multiple-choice candidates.

Run a smoke on an idle GPU:

```bash
CUDA_VISIBLE_DEVICES=<GPU> /home/baishilong/miniconda3/envs/memgen/bin/python \
  eval/exp/dense_top2/eventqa_dense_retrieved_text.py \
  --measurement-scope smoke --context-index 0 --question-limit 10 \
  --embedding-device cpu --output-root outputs/mab/eventqa_dense_top2_smoke
```

The paper-facing effect estimate uses five complete passes, each covering
contexts 0--4 with 100 questions per context. The aligned base seeds are
`42,142,242,342,442`, with per-context reseeding. The generator runs with its
persistent latent bank off; the manifest records this baseline contract
explicitly rather than inheriting P7 parser defaults.

`run_eventqa_full_pass.sh` performs that one 500-question effectiveness pass
and emits `aggregate.json`.  It deliberately does not label its timing or GPU
memory numbers as paper-facing unless it is launched on an otherwise idle GPU.

`run_eventqa_dense_top2_effect_repeats.sh <gpu> <run-id>` executes the five
aligned passes serially and writes `repeat_aggregate.json`. It is the only
dense-top-2 launcher intended for the paper effectiveness table.
