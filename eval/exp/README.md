# EventQA paper experiment reproduction index

This directory is an experiment-oriented entry point for the EventQA results in
`paper/experiment.md` and `paper/result.md`.

The files under each experiment directory are independent copies of the
paper-facing evaluation scripts and configs. The matching directory under
`outputs/mab/<experiment>/` contains symbolic links to the canonical artifacts
used as evidence. This keeps the runnable evaluation entrypoints together
without duplicating large outputs; the original output paths and historical
paper packages remain valid.

Use the project environment for all commands:

```bash
PY=/home/baishilong/miniconda3/envs/memgen/bin/python
GPU=<idle physical GPU index>
```

## P7 latent memory manager

- Runner: `p7/mab6b_weaver_space_bank_eventqa_65536_n5.py`
- Evidence: `outputs/mab/p7/five_repeat_summary.json` and
  `outputs/mab/p7/repeat_{1..5}/`
- Reproduction command for one 500-question pass:

```bash
CUDA_VISIBLE_DEVICES=$GPU $PY eval/exp/p7/mab6b_weaver_space_bank_eventqa_65536_n5.py \
  --requested-contexts 5 --seed 42 --reseed-per-context \
  --output-root outputs/mab/p7/reproduction
```

Run five independent process-level passes with the desired five base seeds and
separate output roots before computing a new repeat aggregate. The historical
five-run aggregate is the evidence for the P7 effectiveness row.

For the controlled continuous cost protocol, use:

```bash
bash eval/exp/p7/run_controlled_cost.sh "$GPU"
```

The valid current cost evidence is linked at
`outputs/mab/p7/controlled_cost_20260721T024259Z/`.

## Capacity-max recent-text MemGen baseline

- Runner: `recent_text/eventqa_memgen_recent_window.py`
- Effect evidence: `outputs/mab/recent_text/effect_first_pass_aggregate.json`
  and `outputs/mab/recent_text/effect_repeats/`
- Cost evidence: `outputs/mab/recent_text/controlled_cost_20260721T013449Z/`

For one context of one effect pass, run:

```bash
CUDA_VISIBLE_DEVICES=$GPU $PY eval/exp/recent_text/eventqa_memgen_recent_window.py \
  --measurement-scope full --context-index <0-4> --question-limit 100 \
  --recent-history-token-budget 32768 --generation-reserve-tokens 40 \
  --seed <base-seed> --output-root outputs/mab/recent_text/reproduction \
  --run-id recent-text-seed<base-seed>-ctx<context>
```

Run contexts 0--4 for each base seed, then aggregate all five
`full_artifact.json` files from one pass with
`recent_text/aggregate.py`. The paper effect estimate uses base seeds 42, 142,
242, 342, and 442. The controlled cost protocol is:

```bash
bash eval/exp/recent_text/run_controlled_cost.sh "$GPU"
```

## Rolling summary, BM25 top-2, and matched-16 controls

| Experiment | Entry scripts | Evidence directory |
|---|---|---|
| Rolling summary | `rolling_summary/construction.py`, `rolling_summary/query.py`, `rolling_summary/aggregate.py` | `outputs/mab/rolling_summary/` |
| BM25 top-2 | `bm25_top2/eventqa_bm25_retrieved_text.py`, `bm25_top2/aggregate.py` | `outputs/mab/bm25_top2/` |
| Matched-16 | `matched16/eventqa_matched16_retrieved_text.py`, `matched16/aggregate.py` | `outputs/mab/matched16/` |

For BM25 or matched-16, run one full context with the corresponding script:

```bash
CUDA_VISIBLE_DEVICES=$GPU $PY eval/exp/<bm25_top2|matched16>/<runner>.py \
  --measurement-scope full --context-index <0-4> --question-limit 100 \
  --output-root outputs/mab/<experiment>/reproduction
```

For rolling summary, first run `construction.py --context-index <0-4>` and
then pass its `construction_artifact.json` to
`query.py --measurement-scope full --context-index <0-4> --question-limit 100
--summary-artifact <path>`. Aggregate one artifact per context with the method
aggregator. `common/run_explicit_controls_repeats.sh` is the historical
repeat-2--5 queue; it intentionally targets its recorded output root and is not
the recommended entry point for a fresh reproduction.

## No-query-retrieval ablation

- Runner: `no_query_retrieval/eventqa_p7_no_query_retrieval.py`
- Evidence: `outputs/mab/no_query_retrieval/effect_aggregate.json` and
  `outputs/mab/no_query_retrieval/effect_full_pass/`

Run contexts 0--4 with `--measurement-scope full --context-index <0-4>
--question-limit 100`, then combine their artifacts using
`no_query_retrieval/aggregate.py`.

## No retrieved-memory conditioning ablation

- Runner: `no_retrieved_memory_conditioning/eventqa_p7_no_retrieved_memory_conditioning.py`
- Output root: `outputs/mab/no_retrieved_memory_conditioning/`

This is distinct from no-query-retrieval: it constructs the same P7 bank and
executes real query-time retrieval, but passes Weaver the project's native
empty-retrieval input. Run one context with `--measurement-scope full
--context-index <0-4> --question-limit 100`; use separate output roots for
each process-level repeat.

## Top-1 direct latent injection

- Runner: `direct_top1/eventqa_p7_direct_top1.py`
- Output root: `outputs/mab/direct_top1/`

Construction remains full P7. At query time this structural control retrieves
at most one threshold-qualified slot, maps its eight Weaver-space latents using
the existing `weaver_to_reasoner` projection, and injects them directly into
Reasoner; it does not run query-time Weaver integration.

## Shared implementation and evidence rules

- `common/eventqa_base.py` is the shared EventQA evaluator and model-loading
  implementation; `common/triviaqa.yaml` is the corresponding model config.
- `eval/exp/<experiment>/` contains copied entrypoint scripts and configs.
  Keep them synchronized deliberately when changing the canonical runner.
- All `outputs/mab/<experiment>/` entries are links, not copied artifacts.
  Resolve a link before deleting or replacing an output.
- The recent-text and P7 controlled cost runs each use one continuous process,
  one model load, 500 questions, and an external-process monitor. They were run
  on different physical RTX A6000 cards, so they are not a same-device absolute
  latency comparison.
