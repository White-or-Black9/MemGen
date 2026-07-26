# P7 Top-1 direct latent injection

Construction remains full P7: top-2 Weaver-space retrieval, Weaver integration,
thread update, and the same frozen bank. During a question turn only,
`direct_top1` retrieves at most one threshold-qualified Weaver-space slot,
maps its eight latent vectors through the trained `weaver_to_reasoner` bridge,
and injects them directly into Reasoner. Query-time Weaver is not called.

```bash
PY=/home/baishilong/miniconda3/envs/memgen/bin/python
CUDA_VISIBLE_DEVICES=<GPU> $PY eval/exp/direct_top1/eventqa_p7_direct_top1.py \
  --measurement-scope full --context-index <0-4> --question-limit 100 \
  --seed 42 --output-root outputs/mab/direct_top1/reproduction
```

Run contexts 0--4 in separate processes for one complete pass. The per-question
artifact distinguishes retrieved slots/latents from Reasoner-injected latents
and asserts zero query-time Weaver calls and zero bank mutation.
