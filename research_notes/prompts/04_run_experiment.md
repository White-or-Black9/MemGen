# Prompt: Run One Experiment

```text
Run exactly one approved experiment within the current approved Phase.

Before running:
- Assign an EXP-YYYYMMDD-NNN ID.
- State the research question, hypothesis, baseline, variant, metrics, and exit rule.
- Record code revision, dirty state, environment, dataset/split, configuration,
  seeds, command, and output directory in research_notes/EXPERIMENTS.md.
- Confirm the run does not modify Weaver or Trigger training.
- Confirm memory is not shared across samples in the current approved
  memory-bank scope unless a later phase explicitly changes it.

During the run:
- Preserve raw logs and outputs.
- Record start/end time, failures, retries, exclusions, and resource use.
- Do not silently change configuration after launch.

After the run:
- Add metrics, observations, anomalies, and artifact paths to EXPERIMENTS.md.
- Compare with the recorded baseline.
- Update BUGS.md for defects/anomalies.
- Update DECISIONS.md only if the evidence changes an important design choice.
- Update PROGRESS.md if this completes the current Phase.

Report the experiment result and evidence boundaries. Stop after this experiment;
do not launch another experiment or Phase without confirmation.
```
