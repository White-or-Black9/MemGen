# P7 no retrieved-memory conditioning

This is the P7 query-time conditioning ablation. Construction uses the normal
P7 path. During each read-only EventQA question turn, the bank still computes
retrieval, but the retrieved latents are withheld from Weaver using the native
empty-retrieval path.

Run one full context with:

```bash
PY=/home/baishilong/miniconda3/envs/memgen/bin/python
CUDA_VISIBLE_DEVICES=<GPU> $PY eval/exp/no_retrieved_memory_conditioning/eventqa_p7_no_retrieved_memory_conditioning.py \
  --measurement-scope full --context-index <0-4> --question-limit 100 \
  --output-root outputs/mab/no_retrieved_memory_conditioning/reproduction
```

The output `artifact.json` records the construction-bank fingerprint and, for
every question, retrieved versus conditioned slot/latent counts. It must show
`query_retrieval_disabled=false` and `query_retrieved_memory_conditioning=false`.
