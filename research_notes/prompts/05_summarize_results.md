# Prompt: Summarize Research Results

```text
Summarize the current MemGen research evidence without inventing missing support.

Read:
- research_notes/EXPERIMENTS.md
- research_notes/BASELINE.md
- research_notes/ABLATIONS.md
- research_notes/DECISIONS.md
- research_notes/BUGS.md
- research_notes/PROGRESS.md
- research_notes/PAPER_NOTES.md

Use raw artifacts when needed to verify recorded metrics.

Produce:
1. Research questions evaluated.
2. Baseline and compared variants.
3. Main quantitative results with experiment IDs.
4. Qualitative observations and failure cases.
5. Efficiency results: latency, throughput, and memory.
6. Ablation conclusions.
7. Supported, unsupported, and contradicted claims.
8. Reproducibility gaps and threats to validity.
9. Recommended single next experiment or decision.

Update PAPER_NOTES.md only with evidence-backed claims and experiment references.
Do not modify core code, run a new experiment, or advance to another Phase.
```
