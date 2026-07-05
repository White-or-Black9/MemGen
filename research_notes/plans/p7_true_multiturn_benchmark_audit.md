# P7 True Multi-Turn Benchmark Audit

## Scope

- Goal: identify a true multi-turn / multi-session conversational-memory benchmark that can support the paper target without downgrading it.
- Priority candidates audited here:
  - `MSC-MemFuse-MC10`
  - `LoCoMo-QA` subset
- Constraints respected:
  - no GPU runs
  - no new runner implementation
  - no dataset download
  - no P7 or model-code changes

## Bottom Line

- `MSC-MemFuse-MC10` is not locally available in this workspace audit.
- `LoCoMo-QA` is locally available, but only through an external nearby repo rather than MemGen-native support.
- Of the two, `LoCoMo-QA` is the only credible second-main benchmark candidate today.
- `LoCoMo-QA` can be used without a GPT / LLM judge if the scope is restricted to QA and scored with deterministic metrics such as exact match and token F1.

## Candidate A: MSC-MemFuse-MC10

### Local Availability

- Repo search: no MemGen-side code, config, notes, or runner found.
- Local dataset/cache search: no benchmark artifact found under the inspected local roots.
- Public-name resolution is also weak from this audit: the benchmark identifier is not stable enough here to trust implementation planning yet.

### Current Status

- Local availability: no
- Dataset path: none found
- Expected schema: unresolved
- 10-way multiple choice: unverified
- Deterministic accuracy: unverified
- GPT / LLM judge required: unverified

### Frozen P7 Mapping

- If this benchmark is later confirmed to be a true multi-session MC benchmark, the protocol mapping would likely be:
  - one benchmark example = one session-local bank
  - sequential ingestion over the conversation/session history
  - bank frozen before question answering
  - query-time retrieval allowed
  - query-time writes blocked
  - answer parsed as option index or option text
- Today this remains hypothetical because no local benchmark artifact or schema was found.

### Audit Conclusion

- `MSC-MemFuse-MC10` cannot be selected as the second main benchmark yet.
- The blocker is not just missing runner code; the benchmark artifact and schema are not resolved locally.

## Candidate B: LoCoMo-QA Subset

### Local Availability

- MemGen repo: no native `LoCoMo` runner, config, or dataset copy found.
- External local path found:
  - `/mnt/18T/sunyanjia/AMEM/A-mem-main/data/locomo10.json`
- Supporting local loader / evaluator files found:
  - `/mnt/18T/sunyanjia/AMEM/A-mem-main/load_dataset.py`
  - `/mnt/18T/sunyanjia/AMEM/A-mem-main/test_advanced.py`
  - `/mnt/18T/sunyanjia/AMEM/A-mem-main/test_advanced_robust.py`
  - `/mnt/18T/sunyanjia/AMEM/A-mem-main/utils.py`

### Local Dataset Shape

- File: `locomo10.json`
- Sample count: 10 conversation samples
- First-sample keys:
  - `conversation`
  - `event_summary`
  - `observation`
  - `qa`
  - `sample_id`
  - `session_summary`
- QA count in first sample: 199
- Session-count range across local samples: 19 to 32
- Turn-count range across local samples: 369 to 689

### Available Tasks

- QA is available directly in the local subset.
- Event summarization fields are present.
- Session summary fields are present.
- Conversation data also includes image-caption-expanded turns, so multimodal content exists in the raw conversation structure.

### QA References And Labels

- QA entries include:
  - `question`
  - `answer`
  - `evidence`
  - `category`
  - optional `adversarial_answer`
- The local loader uses:
  - `answer` for normal questions
  - `adversarial_answer` for category `5`
- Category distribution in the local subset:
  - category 1: 282
  - category 2: 321
  - category 3: 96
  - category 4: 841
  - category 5: 446

### Deterministic Scoring

- The external local evaluator already computes deterministic QA metrics.
- Confirmed metrics in local code:
  - exact match
  - token F1
  - ROUGE
  - BLEU
  - BERTScore
  - METEOR
  - sentence similarity
- For MemGen paper use, the robust deterministic subset should focus on:
  - exact match
  - token F1
- Substring match is possible to add later if desired, but it is not required to establish a deterministic QA path.

### GPT / LLM Judge Requirement

- QA subset: no GPT / LLM judge is required in the local external evaluation path.
- Event summarization / dialogue-generation style tasks should be excluded for now from the formal path in this repo.

### Tasks To Exclude For Now

- event summarization
- dialogue generation
- multimodal generation

These are not needed to establish the second-main benchmark path, and they would complicate scoring and protocol alignment.

### Frozen P7 Mapping

- `LoCoMo-QA` maps cleanly to the frozen P7 protocol at the benchmark level:
  - one `LoCoMo` sample = one session-local bank
  - construction-time ingestion runs sequentially over the full multi-session conversation history
  - bank is frozen before the QA loop
  - multiple questions are then asked against the same frozen bank
  - query-time retrieval is allowed
  - query-time writes must be blocked
  - answer output can be scored as free text against the reference answer with deterministic QA metrics

### Compatibility Assessment

- Fit with frozen P7: strong
- Need for method changes: none at the method level
- Need for runner/adapter work: yes
- Main engineering gap:
  - no MemGen-native `LoCoMo` runner exists yet
  - current usable path lives in the external `A-mem` repo
- Main data gap:
  - only a local `locomo10` subset is confirmed in this audit
  - full official local dataset availability was not established without downloading

## Comparison

### Paper Value

- `LoCoMo-QA`: strong
  - directly addresses multi-session conversational memory
  - preserves the paper target better
- `MSC-MemFuse-MC10`: unknown
  - could be strong if real and local, but that is not established here

### Implementation Risk

- `LoCoMo-QA`: medium
  - data and evaluator path exist locally
  - MemGen adapter still needs to be written
  - scoring policy for final paper rows must be fixed
- `MSC-MemFuse-MC10`: high
  - dataset and schema unresolved
  - cannot even start reliable adapter design from local evidence

### Scoring Reliability

- `LoCoMo-QA`: good for QA-only scope
  - deterministic exact match and token F1 are already available
  - no GPT judge needed for that path
- `MSC-MemFuse-MC10`: unknown

### Fit With Frozen P7

- `LoCoMo-QA`: clean benchmark-level fit
- `MSC-MemFuse-MC10`: only hypothetical fit at this stage

### Second-Main Benchmark Suitability

- `LoCoMo-QA`: yes, as the best current candidate
- `MSC-MemFuse-MC10`: no, not until benchmark identity and local availability are resolved

## Recommendation

- Promote `LoCoMo-QA` as the working second-main benchmark candidate.
- Keep the formal scope restricted to the QA subset.
- Use deterministic scoring only in the first implementation pass.
- Treat `MSC-MemFuse-MC10` as unresolved and do not block on it.

## Exact Next Step Before Implementation

Write a narrow implementation plan for a MemGen-native `LoCoMo-QA` adapter/runner that:

- consumes the local `locomo10.json` subset first
- uses one sample per session-local frozen bank
- performs sequential construction ingestion
- freezes the bank before QA
- enforces `query_write_count == 0`
- logs per-question predictions plus exact match and token F1
- excludes summarization and generation tasks
- isolates any future full-dataset path from the smoke subset path
