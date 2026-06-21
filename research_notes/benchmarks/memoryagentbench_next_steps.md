# MemoryAgentBench Next Steps

## 1. Decision

Proceed next to:

`MAB-5A: detective_qa Compressed-memory Bank-off vs Bank-on n10`

## 2. Why This Is the Next Step

The original MemGen over-context diagnostic showed that full-history multi-turn prompts are only valid when the rebuilt prompt fits within the model context capacity.

For `detective_qa`, the real-data preflight showed:

- estimated full-history query tokens: `102477`
- context capacity: `32768`
- status: over-capacity
- generation called: `false`

Therefore:

- `Original MemGen Full-history Bank-off` is invalid for this task,
- raw over-capacity full-history behavior should not be used as a baseline,
- compressed-memory is the correct next diagnostic because it tests whether LatentBank can answer when full history cannot fit.

## 3. Valid Baseline Taxonomy

- `Original MemGen Full-history Bank-off`: valid only under capacity
- `Original MemGen Full-history over-capacity`: invalid marker, not run
- `Original MemGen Truncated-history Bank-off`: optional future diagnostic, not the same as full-history
- `Compressed Bank-off`: valid compressed lower bound
- `Compressed Bank-on`: valid LatentBank memory condition

## 4. Recommended Experiment

Use `detective_qa` for the next MAB step, but switch to compressed-memory and keep the paired comparison controlled:

- one deterministic 10-context selection if feasible from the split
- `Bank-off` vs `Bank-on`
- no compressed-memory threshold sweep
- no Trigger/Weaver changes
- no benchmark-core changes
- no commit or push

## 5. Guardrail Recommendation

Before any future full-history run, add a harness-level preflight check that marks the sample invalid when:

- rendered full-history prompt tokens exceed loaded reasoner capacity,
- or latent augmentation would push the effective sequence beyond capacity.

The guard should live in the runner, not in the benchmark core and not as a runtime OOM fallback.

## 6. Git Status

### Before

```text
## rlm-memory-bank...origin/rlm-memory-bank [ahead 8]
 M memgen/model/modeling_memgen.py
?? research_notes/benchmarks/
?? scripts/eval/diagnose_memgen_over_context.py
?? scripts/eval/mab2_bank_off.py
?? scripts/eval/mab2_mab_bridge.py
?? scripts/eval/mab3_bank_on_full_history.py
?? scripts/eval/mab3a_threshold_ablation.py
?? scripts/eval/mab4a_compressed_memory.py
?? scripts/eval/mab_paired_bank_off_vs_low_threshold_bank_on.py
?? tests/test_mab2_bank_off.py
?? tests/test_mab3_bank_on_full_history.py
?? tests/test_mab3a_threshold_ablation.py
?? tests/test_mab4a_compressed_memory.py
?? tests/test_mab_paired_bank_off_vs_low_threshold_bank_on.py
```

## 7. MAB-5A Follow-up

The completed `MAB-5A` compressed-memory run was mechanism-active but not accuracy-improving: Bank-off and Bank-on were both `0.0`, while outputs changed in all 10 contexts and retrieval stayed active.

The next mechanism experiment should be a decoupled retrieve/update threshold design. Do not run another threshold-only ablation yet.

### After

```text
## rlm-memory-bank...origin/rlm-memory-bank [ahead 8]
 M memgen/model/modeling_memgen.py
?? research_notes/benchmarks/
?? scripts/eval/diagnose_memgen_over_context.py
?? scripts/eval/mab2_bank_off.py
?? scripts/eval/mab2_mab_bridge.py
?? scripts/eval/mab3_bank_on_full_history.py
?? scripts/eval/mab3a_threshold_ablation.py
?? scripts/eval/mab4a_compressed_memory.py
?? scripts/eval/mab_paired_bank_off_vs_low_threshold_bank_on.py
?? tests/test_mab2_bank_off.py
?? tests/test_mab3_bank_on_full_history.py
?? tests/test_mab3a_threshold_ablation.py
?? tests/test_mab4a_compressed_memory.py
?? tests/test_mab_paired_bank_off_vs_low_threshold_bank_on.py
?? research_notes/benchmarks/memoryagentbench_next_steps.md
```
