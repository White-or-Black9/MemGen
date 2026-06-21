# MemoryAgentBench MAB-1A No-API Smoke

Initial attempt: 2026-06-19  
Real-data resume: 2026-06-20  
MemoryAgentBench branch: `mab-1-official-smoke`  
MemoryAgentBench commit: `455306dcabc3842526eb83cd4e225e5d486c5c5d`  
Target: `Conflict_Resolution` / `factconsolidation_sh_6k`, one context, one query, chunk size 4096  
Status: **complete / `no_api_smoke_success`**

This was a dry-run infrastructure smoke, not a benchmark result. No external LLM API, LLM judge, official agent, MemGen model, or MemGen adapter was used. Neither repository's core logic was modified. No commit or push was performed.

## Executive Result

MAB-1A real-data validation passed using the manually transferred local Parquet file. The run was forced offline and did not contact HuggingFace or any model provider.

- Local file: `/mnt/18T/baishilong/datasets/MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet`.
- File SHA-256: `24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45`.
- Parquet rows: 8.
- Rows matching `metadata.source == factconsolidation_sh_6k`: 1.
- Required fields `context`, `questions`, `answers`, and `metadata`: all present.
- Official real-context chunks: 2.
- Official template formatting: memorization and first-query prompts validated.
- Official real-gold metric: `substring_exact_match=true` for the exact first gold answer.
- Official wrong-answer metric: `substring_exact_match=false` for a deliberate wrong answer.
- Successful manifest: `no_api_smoke_success`.

The successful artifact is a dry-run/no-API validation, not a benchmark score. Full contexts, questions, answers, and prompts were neither printed nor persisted; only bounded snippets, lengths, counts, hashes, and redacted result records were stored.

The 2026-06-19 failed attempts are retained below as historical environment and acquisition evidence.

## 1. Environment Commands

The shell's `/home/baishilong/bin/conda` shim remains unusable because it has a CRLF interpreter line. All environment operations used the real Conda binary:

```bash
/home/baishilong/miniconda3/bin/conda create \
  --name MABench python=3.10.16 -y
```

The first create command exceeded the command observation window after writing its transaction history. A diagnostic repeat with `--json` was issued against the already-created prefix and produced a removal transaction, leaving the registered environment empty. Conda history made this cause explicit. The prefix was repaired with:

```bash
/home/baishilong/miniconda3/bin/conda install \
  -n MABench python=3.10.16 pip -y
```

Verified after repair:

```text
Python 3.10.16
pip 26.1.2
```

The requested minimal install was attempted as one command:

```bash
/home/baishilong/miniconda3/envs/MABench/bin/python -m pip install \
  datasets pandas pyarrow nltk tiktoken rouge-score pyyaml "numpy<2"
```

That command exceeded the observation window while downloading, so the same unchanged package set was completed in smaller transactions:

```bash
/home/baishilong/miniconda3/envs/MABench/bin/python -m pip install \
  "numpy<2" pyyaml nltk tiktoken rouge-score

/home/baishilong/miniconda3/envs/MABench/bin/python -m pip install \
  pandas pyarrow

/home/baishilong/miniconda3/envs/MABench/bin/python -m pip install datasets
```

Importing the official `utils.eval_other_utils` then failed because it unconditionally imports `editdistance`, which was not in the requested minimal list. Only that transitive requirement was added:

```bash
/home/baishilong/miniconda3/envs/MABench/bin/python -m pip install editdistance
```

The full MemoryAgentBench `requirements.txt` was not installed.

## 2. Package Versions

| Package | Version |
|---|---:|
| Python | 3.10.16 |
| `datasets` | 5.0.0 |
| `pandas` | 2.3.3 |
| `pyarrow` | 24.0.0 |
| `nltk` | 3.9.4 |
| `tiktoken` | 0.13.0 |
| `rouge-score` | 0.1.2 |
| `pyyaml` | 6.0.3 |
| `numpy` | 1.26.4 |
| `editdistance` | 0.8.1 |

The environment is isolated at `/home/baishilong/miniconda3/envs/MABench`; no package was installed into the MemGen environment.

## 3. NLTK Tokenizer Data

Initial resource checks found both resources missing:

```text
punkt: missing
punkt_tab: missing
```

Only the required tokenizer resources were downloaded:

```bash
/home/baishilong/miniconda3/envs/MABench/bin/python \
  -m nltk.downloader punkt punkt_tab
```

They were installed under `/home/baishilong/nltk_data`. The official chunker still calls `nltk.download('punkt', quiet=True)` on entry; in offline mode this emits a download warning even when the cached tokenizer remains usable. No benchmark-core change was made to suppress that behavior.

## 4. Historical Dataset Loading Result (2026-06-19)

Requested source:

```text
dataset: ai-hyz/MemoryAgentBench
split: Conflict_Resolution
metadata.source filter: factconsolidation_sh_6k
max selected contexts: 1
```

Cache configuration at execution time:

```text
HF_HOME: unset
HF_ENDPOINT: https://hf-mirror.com
effective cache root: /home/baishilong/.cache/huggingface
dataset cache before/after attempts: absent
```

The official loader path was used where possible:

- `datasets.load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution", revision="main")`
- `utils.eval_data_utils.load_data_huggingface(...)`

No context, question, or answer content was printed.

### Acquisition failures

1. Loader through configured mirror:

```text
FileNotFoundError: couldn't find ai-hyz/MemoryAgentBench on the Hub or local cache
```

2. Explicit download through official endpoint:

```bash
env -u HF_ENDPOINT \
  /home/baishilong/miniconda3/envs/MABench/bin/hf download \
  ai-hyz/MemoryAgentBench --repo-type dataset
```

Result:

```text
SSL: UNEXPECTED_EOF_WHILE_READING
```

3. Explicit download through configured mirror:

```bash
/home/baishilong/miniconda3/envs/MABench/bin/hf download \
  ai-hyz/MemoryAgentBench --repo-type dataset
```

Result:

```text
No local file found; distant resource unavailable through the configured endpoint
```

After these distinct endpoint attempts, further network retries were stopped. The final artifact reran in explicit offline mode to capture a deterministic failed schema:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  /home/baishilong/miniconda3/envs/MABench/bin/python \
  /tmp/mab_no_api_smoke.py \
  --repo /mnt/18T/baishilong/benchmarks/MemoryAgentBench \
  --output-root /mnt/18T/baishilong/MemGen/outputs/mab/no_api_smoke
```

Final failure type:

```text
ConnectionError: couldn't reach ai-hyz/MemoryAgentBench (OfflineModeIsEnabled)
```

Because loading failed, row count, actual columns, metadata keys, and required-field presence could not be recorded from real data.

## 5. Historical Synthetic Chunking Result (2026-06-19)

Target actual-context chunking was not executed because no selected context was available.

A clearly labeled synthetic wiring check used the unchanged official function:

```python
utils.eval_other_utils.chunk_text_into_sentences(
    "Synthetic fact one. Synthetic fact two.",
    chunk_size=4096,
)
```

Synthetic-only result:

```text
chunk count: 1
```

This proves that the official chunker, NLTK resources, and `tiktoken` path execute in `MABench`. It does not validate real context chunk counts, token summaries, or first/last context snippets.

## 6. Historical Synthetic Prompt and Template Result (2026-06-19)

Official module:

```text
utils/templates.py
```

Selected normalization:

```text
dataset template: factconsolidation
agent template: long_context_agent
```

The synthetic check used `get_template()` for both `memorize` and `query`, then formatted safe synthetic content.

Synthetic-only result:

```text
memorization prompt length: 227 characters
memorization prefix (maximum 120 chars): recorded in diagnostics.jsonl
query prompt length: 783 characters
query prefix (maximum 120 chars): recorded in diagnostics.jsonl
```

No full prompt was printed or stored. Real first-chunk/first-question prompt validation remains blocked by dataset access.

## 7. Historical Synthetic Metric Smoke Result (2026-06-19)

The unchanged official task routing was used:

```python
utils.eval_other_utils.post_process(output, answer, dataset_config)
```

Metric fields returned:

- `exact_match`
- `f1`
- `substring_exact_match`
- `rougeL_f1`
- `rougeL_recall`
- `rougeLsum_f1`
- `rougeLsum_recall`

Synthetic-only assertions:

```text
prediction equal to synthetic gold: substring_exact_match = true
deliberately wrong prediction:      substring_exact_match = false
```

These assertions verify automatic metric wiring and positive/negative behavior without an LLM. They do not fulfill the requested first-real-gold-answer check because the dataset did not load. No real answer text was printed or persisted.

## 8. Output Artifacts

Successful real-data dry-run artifact:

```text
outputs/mab/no_api_smoke/20260620T015554Z-455306d-fact-sh-6k-real-local/
  manifest.json
  dry_run_results.json
  diagnostics.jsonl
  environment.txt
```

Manifest status:

```json
"status": "no_api_smoke_success"
```

The artifact remains explicitly labeled:

```text
dry_run_no_api_not_benchmark_result
```

`dry_run_results.json` contains the official-like top-level shape (`agent_config`, `dataset_config`, `data`, `metrics`, `time_cost`, `averaged_metrics`). It contains two redacted dry-run records: exact-first-gold and deliberately-wrong. The real prediction, question, and answer text are replaced with explicit redaction markers.

Retained failed attempts, including the first two local-data harness runs:

```text
outputs/mab/no_api_smoke/20260619T130822Z-455306d-fact-sh-6k/
  failure: missing editdistance import

outputs/mab/no_api_smoke/20260619T130847Z-455306d-fact-sh-6k/
  failure: dataset unavailable through Hub/cache

outputs/mab/no_api_smoke/20260619T131049Z-455306d-fact-sh-6k/
  failure: final offline blocked-state capture before local transfer

outputs/mab/no_api_smoke/20260620T015445Z-455306d-fact-sh-6k-real-local/
  failure: default HuggingFace datasets cache mounted read-only

outputs/mab/no_api_smoke/20260620T015517Z-455306d-fact-sh-6k-real-local/
  failure: harness imposed a stricter <=4096 token assertion than the official chunker guarantees
```

No prior artifact was overwritten.

## 9. Real-Data Validation Result (2026-06-20)

### Local dataset

```text
root: /mnt/18T/baishilong/datasets/MemoryAgentBench
file: data/Conflict_Resolution-00000-of-00001.parquet
format: Apache Parquet
magic: PAR1
size: 1,491,588 bytes
SHA-256: 24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45
```

The run used `datasets.load_dataset("parquet", data_files=..., split="Conflict_Resolution")` against this local file with `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`. `HF_HOME` and `HF_DATASETS_CACHE` were redirected to `/tmp/mab_hf_cache` because the default home cache was read-only inside the execution sandbox.

### Filter and schema

```text
total rows: 8
metadata.source == factconsolidation_sh_6k: 1 row
selected rows: 1
columns: context, questions, answers, metadata
questions in selected row: 100
answer entries in selected row: 100
```

Required fields all validated as present and non-null:

- `context`
- `questions`
- `answers`
- `metadata`

Metadata keys:

- `demo`
- `haystack_sessions`
- `keypoints`
- `previous_events`
- `qa_pair_ids`
- `question_dates`
- `question_ids`
- `question_types`
- `source`

Only lengths and the SHA-256 of the first gold answer were recorded. No full question or answer was printed or stored.

### Official chunking

The first real context was passed unchanged to:

```python
utils.eval_other_utils.chunk_text_into_sentences(context, chunk_size=4096)
```

Observed result:

```text
chunk count: 2
tiktoken lengths: [4319, 2119]
minimum: 2119
maximum: 4319
sum: 6438
chunks above requested size: 1
```

The official chunker computes its running count by summing sentence token lengths, then joins sentences with spaces. The inserted join spaces are not included in the running count, so the first emitted chunk measures 4319 tokens when encoded after joining even though the configured target is 4096. No benchmark logic was changed; this discrepancy is recorded for later adapter parity.

`diagnostics.jsonl` stores only the permitted first 80 characters of the first chunk and last 80 characters of the last chunk.

### Official templates

Official utility:

```text
utils/templates.py:get_template
normalized dataset template: factconsolidation
normalized agent template: long_context_agent
```

Observed prompt lengths:

```text
memorization prompt: 17,616 characters
query prompt: 797 characters
```

The memorization prompt contains the first official chunk, and the query prompt contains the first real question. Artifacts retain only the first 120 characters of each prompt.

### Official metrics

Official utility:

```python
utils.eval_other_utils.post_process(output, answer, dataset_config)
```

Metric fields:

- `exact_match`
- `f1`
- `substring_exact_match`
- `rougeL_f1`
- `rougeL_recall`
- `rougeLsum_f1`
- `rougeLsum_recall`

Required assertions:

```text
exact first gold answer: substring_exact_match = 1 / true
deliberately wrong answer: substring_exact_match = 0 / false
```

All metric values in the result artifact are finite. Prediction and target strings are redacted.

## 10. Errors Encountered

| Stage | Error | Resolution/status |
|---|---|---|
| Conda create observation | First transaction exceeded command observation window | Diagnosed through Conda history |
| Conda diagnostic retry | Repeat `create` emptied the existing prefix | Repaired with explicit `conda install python=3.10.16 pip` |
| Minimal pip install | Combined download exceeded observation window | Same package set completed in smaller transactions |
| Official metric import | `ModuleNotFoundError: editdistance` | Installed only `editdistance==0.8.1` |
| NLTK | `punkt` and `punkt_tab` absent | Downloaded only those resources |
| Configured HF mirror | Dataset not found/retrievable | Unresolved |
| Official HF endpoint | SSL unexpected EOF | Unresolved |
| Final offline run | Dataset cache absent | Expected blocked-state capture |
| Local Parquet first run | Default datasets cache lock path was read-only | Redirected `HF_HOME` and `HF_DATASETS_CACHE` to `/tmp/mab_hf_cache` |
| Local Parquet second run | Harness asserted every emitted chunk was at most 4096 tokens | Removed non-official assertion; recorded actual official chunk lengths and over-limit count |
| Shell startup | Unrelated `gpustat`/Python 3.8 `libffi` traceback on commands | Pre-existing shell-profile noise; did not affect MABench Python checks |

## 11. MAB-1B and Next Recommendation

MAB-1A is complete. The environment, local split/filter contract, official real-data chunking, official prompt formatting, automatic metric routing, and dry-run JSON shape are validated.

MAB-1B official API smoke is still needed eventually to validate the official agent call and serializer end to end, but it was explicitly out of scope for this task and was not run. Its API credential prerequisite remains unchanged.

Recommended next action:

1. Review this artifact and the official-chunker 4096/4319 discrepancy.
2. If API credentials and cost approval become available, prepare MAB-1B as a separate one-context/one-query official-agent smoke.
3. Do not implement the MemGen adapter until the project explicitly advances past the MAB-1 gate.

No adapter implementation, full benchmark, LLM judge, commit, or push is justified by this smoke alone.
