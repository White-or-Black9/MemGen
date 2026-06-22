# MemoryAgentBench Fork Hygiene and MAB-1 Smoke Plan

> Historical pre-smoke plan. MAB-1A later completed; use
> `memoryagentbench_no_api_smoke.md` for evidence and
> `memoryagentbench_runbook.md` for current operations.

Date: 2026-06-19  
MemGen branch context: `rlm-memory-bank`  
MemoryAgentBench checkout: `/mnt/18T/baishilong/benchmarks/MemoryAgentBench`  
Scope: MAB-0.5 fork hygiene and MAB-1 preparation only

No MemoryAgentBench experiment, package installation, MemGen adapter work, MemGen modification, commit, or push was performed during this phase.

## Status Summary

**MAB-0.5 fork hygiene: complete.** The fork checkout is present, clean, synchronized with the freshly fetched official upstream, and on the dedicated local branch `mab-1-official-smoke`.

**MAB-1 execution: NO-GO at this time.** Three prerequisites are missing:

1. Conda environment `MABench` does not exist.
2. HuggingFace dataset `ai-hyz/MemoryAgentBench` is not present in the inspected default cache.
3. Neither `OPENAI_API_KEY` nor `AZURE_OPENAI_API_KEY` is available to the selected official `gpt-4o-mini` agent.

The smoke command is prepared below, but it must not be run until all three prerequisites pass.

## 1. Fork Remote Status

### Checkout creation

The requested path did not exist at the start of MAB-0.5. It was created from the user's fork:

```bash
mkdir -p /mnt/18T/baishilong/benchmarks
git clone git@github.com:White-or-Black9/MemoryAgentBench.git \
  /mnt/18T/baishilong/benchmarks/MemoryAgentBench
git -C /mnt/18T/baishilong/benchmarks/MemoryAgentBench remote add upstream \
  https://github.com/HUST-AI-HYZ/MemoryAgentBench.git
git -C /mnt/18T/baishilong/benchmarks/MemoryAgentBench fetch upstream --prune
```

The clone and upstream fetch completed successfully. No push was performed.

### Verified remotes

```text
origin  git@github.com:White-or-Black9/MemoryAgentBench.git (fetch)
origin  git@github.com:White-or-Black9/MemoryAgentBench.git (push)
upstream https://github.com/HUST-AI-HYZ/MemoryAgentBench.git (fetch)
upstream https://github.com/HUST-AI-HYZ/MemoryAgentBench.git (push)
```

These remotes match the requested topology. No corrective remote command is needed.

Safe correction commands if a future checkout drifts are:

```bash
git remote set-url origin git@github.com:White-or-Black9/MemoryAgentBench.git
git remote set-url upstream https://github.com/HUST-AI-HYZ/MemoryAgentBench.git
git fetch origin --prune
git fetch upstream --prune
```

These commands fetch/update remote metadata only; they do not push or rewrite local history.

### Commit and synchronization status

Freshly verified after fetching upstream:

| Ref | Commit SHA |
|---|---|
| Initial local `main` / `HEAD` | `455306dcabc3842526eb83cd4e225e5d486c5c5d` |
| `origin/main` | `455306dcabc3842526eb83cd4e225e5d486c5c5d` |
| `upstream/main` | `455306dcabc3842526eb83cd4e225e5d486c5c5d` |

Latest upstream commit metadata:

```text
SHA: 455306dcabc3842526eb83cd4e225e5d486c5c5d
Date: 2026-05-21 15:26:59 +0800
Subject: Update README.md
```

Fork divergence relative to `upstream/main`:

```text
behind: 0
ahead:  0
```

The previously inspected commit `455306dcabc3842526eb83cd4e225e5d486c5c5d` exists locally and is the current fork/upstream head. This commit is the pinned MAB-1 source revision.

The MemoryAgentBench working tree was clean before and after branch creation.

## 2. Branch Plan

Do not work directly on `main`. The following local branch was created from the freshly fetched `upstream/main`:

```bash
git -C /mnt/18T/baishilong/benchmarks/MemoryAgentBench \
  switch -c mab-1-official-smoke upstream/main
```

Current branch state:

```text
branch: mab-1-official-smoke
HEAD: 455306dcabc3842526eb83cd4e225e5d486c5c5d
tracking: upstream/main
ahead/behind upstream/main: 0/0
working tree: clean
```

The branch is local only. It has not been committed or pushed. Tracking `upstream/main` is acceptable for read-only synchronization checks, but any eventual publication should push explicitly to `origin/mab-1-official-smoke` only after review.

## 3. Environment Status and Setup Plan

### Conda status

`MABench` is absent from the environment list returned by:

```bash
/home/baishilong/miniconda3/bin/conda env list --json
```

The shell-resolved command `/home/baishilong/bin/conda` is currently broken because its interpreter line contains CRLF (`/bin/bash^M`). Use the explicit real Conda binary above for setup rather than modifying the shim during this phase.

No package installation was attempted, so there are no installation errors to record.

### Proposed isolated setup

Run only after review:

```bash
/home/baishilong/miniconda3/bin/conda create \
  --name MABench python=3.10.16

/home/baishilong/miniconda3/bin/conda run -n MABench \
  python -m pip install torch

cd /mnt/18T/baishilong/benchmarks/MemoryAgentBench
/home/baishilong/miniconda3/bin/conda run -n MABench \
  python -m pip install -r requirements.txt

/home/baishilong/miniconda3/bin/conda run -n MABench \
  python -m pip install "numpy<2"
```

Do not activate or modify the existing `memgen` environment. Before installation, record the command, timestamp, Conda channels, CUDA driver/runtime, and selected PyTorch wheel. After successful installation, record:

```bash
/home/baishilong/miniconda3/bin/conda run -n MABench python --version
/home/baishilong/miniconda3/bin/conda run -n MABench python -m pip freeze
/home/baishilong/miniconda3/bin/conda env export -n MABench
```

### High-risk dependencies

The upstream `requirements.txt` is broad and unpinned. Relevant risks are:

| Dependency | Present where | Risk for MAB-1 |
|---|---|---|
| `flash_attn` | `requirements.txt` | Native compilation and strict PyTorch/CUDA ABI coupling; unnecessary for the selected API smoke but may fail the full install |
| `faiss-gpu` | `requirements.txt` | Wheel availability and CUDA compatibility; unnecessary for long-context API smoke |
| `deepspeed` | `requirements.txt` | Native/CUDA extension and PyTorch version coupling; unnecessary for this smoke |
| `bitsandbytes` | `requirements.txt` | CUDA/runtime compatibility; unnecessary for this smoke |
| `mem0ai` | `requirements.txt` | Large transitive dependency surface and provider/database version conflicts; unused by selected agent |
| Cognee | README workaround; `owlready2` in requirements | README warns that Cognee may need supplemental install/uninstall steps; unused by selected agent |
| Letta | README workaround, not directly pinned in requirements | README warns that Letta may need supplemental install/uninstall steps; unused by selected agent |
| Provider SDKs | `openai`, `anthropic`, `google-genai`, `litellm`, LangChain provider packages | Unpinned API drift and conflicting transitive versions; only `openai` is needed by selected agent |
| CUDA 12 wheel set | explicit `nvidia-*-cu12` requirements | May conflict with host CUDA/PyTorch selection and consume substantial disk space |
| `numpy` | unpinned in requirements | Upstream explicitly requires a final `numpy<2` install to avoid incompatibility |

Installation policy:

- First attempt the README commands exactly in isolated `MABench` and preserve the complete log.
- Do not repair a failure by changing benchmark core logic.
- Do not apply the Cognee/Letta workarounds for this smoke because those agents are not selected.
- If full requirements fail on an irrelevant optional dependency, stop and document the exact error before proposing a minimal smoke-only dependency set. Do not silently omit packages.
- Never install these requirements into the MemGen environment.

## 4. Dataset and Cache Status

### HuggingFace cache root

Observed environment:

```text
HF_HOME: unset
effective cache root: /home/baishilong/.cache/huggingface
```

Checked expected hub snapshot path:

```text
/home/baishilong/.cache/huggingface/hub/datasets--ai-hyz--MemoryAgentBench
```

Status: **not present**.

The legacy/Arrow datasets cache root exists:

```text
/home/baishilong/.cache/huggingface/datasets
```

No path matching `MemoryAgentBench` or `ai-hyz` was found beneath the inspected HuggingFace cache. The dataset therefore does not appear cached.

### `entity2id.json`

No `entity2id.json` was found in:

- The MemoryAgentBench checkout.
- The inspected HuggingFace cache.

This file is required for the Recsys task, not for `Factconsolidation_sh_6k`, so its absence does not block MAB-1. It remains a blocker for any later Recsys phase.

### Proposed cache-only preparation

Dataset access is required before MAB-1. Download/cache it without running a model or printing examples:

```bash
/home/baishilong/miniconda3/bin/conda run -n MABench \
  hf download ai-hyz/MemoryAgentBench --repo-type dataset
```

Then verify filenames/revision and loader metadata only. Do not print contexts, questions, or answers:

```bash
hf cache ls --filter "repo_id=ai-hyz/MemoryAgentBench"
test -d /home/baishilong/.cache/huggingface/hub/datasets--ai-hyz--MemoryAgentBench
```

The official loader requests dataset revision `main`. Record the resolved cached dataset commit in the MAB-1 run manifest to avoid silent dataset drift.

## 5. Selected MAB-1 Smoke Task

Selected official data config:

```text
configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml
```

Verified configuration:

```yaml
dataset: Conflict_Resolution
chunk_size: 4096
context_max_length: 6000
sub_dataset: factconsolidation_sh_6k
generation_max_length: 10
max_test_samples: 1
shots: 0
```

This supplies the required one-context limit. The CLI flag `--max_test_queries_ablation 1` supplies the one-query limit. `--chunk_size_ablation 4096` makes the intended chunk size explicit while preserving the config value.

### Selected official agent

Simplest official supported agent for this smoke:

```text
configs/agent_conf/Long_Context_Agents/Long_context_agent_gpt-4o-mini.yaml
```

Verified relevant fields:

```yaml
agent_name: Long_context_agent_gpt-4o-mini
model: gpt-4o-mini
temperature: 0.7
input_length_limit: 128000
buffer_length: 4000
output_dir: ./outputs/gpt-4o-mini
```

This agent stores memorization chunks in its in-process text context and makes one OpenAI-compatible query for the selected one-query smoke. It requires either:

- `OPENAI_API_KEY`, or
- Azure OpenAI configuration including `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, and `AZURE_OPENAI_API_KEY`, with a matching deployment name.

Credential precheck result:

```text
OPENAI_API_KEY: absent
AZURE_OPENAI_API_KEY: absent
```

Only presence was tested; no secret value was printed. Because the required credential is absent, the official smoke must not be run now.

## 6. Exact Proposed Smoke Command

Expected default output file:

```text
/mnt/18T/baishilong/benchmarks/MemoryAgentBench/
outputs/gpt-4o-mini/Conflict_Resolution/
factconsolidation_sh_6k_unknown_in6000_size10_shots0_max_samples1_results.json
```

This file does not currently exist, and the checkout has no existing output files. Therefore `--force` is safe at the time of this plan, but the command must include a fresh non-overwrite guard immediately before execution.

Exact guarded command:

```bash
cd /mnt/18T/baishilong/benchmarks/MemoryAgentBench

expected_output="outputs/gpt-4o-mini/Conflict_Resolution/factconsolidation_sh_6k_unknown_in6000_size10_shots0_max_samples1_results.json"
test ! -e "$expected_output" || {
  echo "Refusing to overwrite existing smoke output: $expected_output" >&2
  exit 1
}

test -n "${OPENAI_API_KEY:-}${AZURE_OPENAI_API_KEY:-}" || {
  echo "No OpenAI or Azure OpenAI credential is available" >&2
  exit 1
}

/home/baishilong/miniconda3/bin/conda run -n MABench python main.py \
  --agent_config configs/agent_conf/Long_Context_Agents/Long_context_agent_gpt-4o-mini.yaml \
  --dataset_config configs/data_conf/Conflict_Resolution/Factconsolidation_sh_6k.yaml \
  --max_test_queries_ablation 1 \
  --chunk_size_ablation 4096 \
  --force
```

`--force` affects result-resume behavior; it does not itself remove the output. The explicit guard is the protection against overwriting a prior successful or partial smoke. If a previous file exists, archive it to a unique run directory and review it before any rerun. Do not delete it automatically.

No MemGen process, checkpoint, environment, or module is involved in this command.

## 7. Expected Output and JSON Schema

The official serializer writes:

```json
{
  "agent_config": {},
  "dataset_config": {},
  "data": [],
  "metrics": {},
  "time_cost": [],
  "averaged_metrics": {}
}
```

For this one-query smoke, `data` should contain exactly one record. The record is expected to include:

```json
{
  "output": "model prediction",
  "input_len": 0,
  "output_len": 0,
  "memory_construction_time": 0,
  "query_time_len": 0.0,
  "parsed_output": "parsed prediction",
  "exact_match": false,
  "f1": 0.0,
  "substring_exact_match": false,
  "rougeL_f1": 0.0,
  "rougeL_recall": 0.0,
  "rougeLsum_f1": 0.0,
  "rougeLsum_recall": 0.0,
  "answer": ["ground truth"],
  "query": "formatted query",
  "query_id": 0,
  "qa_pair_id": "optional source identifier"
}
```

`qa_pair_id` is optional when absent from the source. For FactConsolidation, the primary benchmark metric is `substring_exact_match`; generic EM, F1, and ROUGE fields are also emitted by the utility. `metrics` contains arrays of per-query values. `averaged_metrics` contains means; the official writer multiplies non-length/non-time metrics by 100.

### Preservation plan

After a successful schema validation, copy artifacts into a unique directory under:

```text
/mnt/18T/baishilong/MemGen/outputs/mab/official_smoke/
```

Recommended layout:

```text
outputs/mab/official_smoke/<YYYYMMDD-HHMMSS>-455306d-fact-sh-6k/
  results.json
  manifest.json
  environment.txt
  stdout.log
  stderr.log
```

Use `cp --no-clobber` or an equivalent destination-existence guard. Never copy directly over an existing run. `manifest.json` should record source/fork/upstream SHA, dataset revision, config paths, limits, command, selected agent, and output checksum. It must not contain API keys or private checkpoint paths.

## 8. Known Risks

- Full upstream requirements may fail on optional CUDA/native packages irrelevant to the selected smoke.
- The host exposes CUDA 11.8 in the current shell while requirements explicitly list CUDA 12 wheels; PyTorch/runtime compatibility must be checked in the isolated environment.
- The shell's first `conda` shim is broken; use `/home/baishilong/miniconda3/bin/conda` explicitly.
- The dataset is not cached and requires network access before loader validation.
- The selected official agent requires an absent API credential.
- Provider model behavior/version can drift even with a fixed repository commit.
- The official agent uses `temperature: 0.7`; this is suitable for pipeline smoke validation but not deterministic baseline evidence.
- `nltk` may try to download `punkt` during chunking; prepare/cache it in `MABench` if loader/chunker validation reports it missing.
- The output filename encodes YAML values but not repository SHA, dataset revision, or timestamp; preservation must use a unique run directory and manifest.
- `--force` can rerun an existing result path; retain the explicit existence guard.
- Exact/substring scoring is format-sensitive. Smoke success means schema/pipeline validity, not that the one prediction must be correct.
- The missing Recsys `entity2id.json` does not affect this smoke but must be resolved before Recsys evaluation.
- Broad package fixes or benchmark-core edits would reduce confidence in an "official" smoke and require explicit review.

## 9. Go/No-Go Recommendation

**Current decision: NO-GO for running MAB-1.**

Fork/source readiness is GO:

- Correct `origin` and `upstream` remotes.
- Fresh upstream fetch completed.
- Fork and upstream synchronized at pinned SHA `455306dcabc3842526eb83cd4e225e5d486c5c5d`.
- Clean dedicated branch `mab-1-official-smoke` created from `upstream/main`.
- No existing smoke result would currently be overwritten.

Execution readiness is NO-GO until:

1. `MABench` is created with Python 3.10.16 and installation logs are reviewed.
2. Imports required by the selected official agent, loader, chunker, and scorer pass in that environment.
3. `ai-hyz/MemoryAgentBench` is cached and its resolved revision is recorded.
4. A loader-only check confirms `Conflict_Resolution` contains `metadata.source == factconsolidation_sh_6k` without printing sample content.
5. NLTK `punkt` is locally available.
6. An OpenAI or Azure OpenAI credential is supplied securely and detected by presence-only checks.
7. The expected output path is rechecked immediately before execution and remains absent.

Once all seven gates pass, proceed with exactly the guarded one-context/one-query command in this note. Stop immediately if dataset loading, API invocation, automatic scoring, or JSON serialization fails. Do not broaden the task, add a judge, involve MemGen, or run another benchmark config during MAB-1.
