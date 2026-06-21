# MemoryAgentBench Adapter Strategy Review

Date: 2026-06-20  
MemGen branch: `rlm-memory-bank`  
Scope: source inspection and evaluation strategy only  
MAB evidence anchor: `outputs/mab/no_api_smoke/20260620T015554Z-455306d-fact-sh-6k-real-local`

No code, model, benchmark logic, experiment, commit, or push was performed for this review.

## Executive Summary

The conceptual clarification is correct: original MemGen has Trigger, Weaver, and Reasoner inference-time latent-token generation, but it does not have the added explicit session-level `LatentMemoryBank`. Therefore the primary MAB baseline must not be called a generic "disabled" or "no-memory" baseline.

The concrete source shows that current MemGen multi-turn execution is **message-history based**:

- `MultiTurnInteractionManager` owns `init_prompts` and `inter_histories`.
- On every turn it reconstructs `init_prompt + inter_history`, applies the chat template to the entire conversation, and calls `MemGenModel.generate()` again.
- `MemGenModel.generate()` starts with `current_cache = None` on every call.
- KV cache is used only inside that one generation call and is discarded when the call returns. It is also explicitly invalidated and rebuilt after latent injection.
- No `past_key_values` object is accepted from or returned to the interaction manager.

Accordingly, the natural first baseline is:

**Original MemGen History/KV Bank-off**, more precisely **full-history rebuild with per-call KV only**. It retains all prior MAB chunk turns as messages, while the added session-level bank is absent. Trigger and Weaver remain active according to original MemGen configuration. This is not a no-memory baseline: information remains available in the rendered dialogue history, and original MemGen may still generate transient latent tokens during each response.

Recommended phase order:

1. **MAB-2:** Original MemGen History/KV Bank-off, full-history rebuild, one `run_agent_loop()` per MAB context, `factconsolidation_sh_6k`, one context, one query, `batch_size=1`.
2. **MAB-3:** MemGen + LatentBank V-A Full-history Bank-on with the exact same messages, turn order, decoding, and full-history rebuild. This isolates the additive effect of the bank.
3. **Later:** MemGen + LatentBank V-A Compressed-memory Bank-on. This tests whether the bank can replace prompt history, but only after the additive comparison is valid.

Cross-turn KV reuse should not be implemented in MAB-2. It could reduce repeated prefill cost, but it is not current MemGen behavior and is unsafe around chat-template growth, Trigger/Weaver latent injection, position IDs, and explicit cache invalidation. Rebuilding the full prompt is slower but is the source-faithful and scientifically safest baseline.

## Source Inspection Findings

### Files inspected

- `interactions/base_interaction.py`
- `interactions/multiturn_interaction.py`
- `interactions/singleturn_interaction.py`
- `interactions/tensor_utils.py`
- `memgen/model/modeling_memgen.py`
- `memgen/model/modeling_utils.py`
- `memgen/runner.py`
- `memgen/model/latent_memory_bank.py`
- `configs/latent_memory_bank/default.yaml`
- `tests/test_latent_memory_bank_integration.py`
- `scripts/eval/r4_triviaqa_dynamic_harness.py`

### Message-history ownership

`MultiTurnInteractionManager.run_agent_loop()` is the current multi-turn episode implementation (`interactions/multiturn_interaction.py:132`). At entry it:

- receives `init_prompts` and per-sample environment objects;
- initializes `inter_histories` to a fresh empty list for every batch item (`:146-155`);
- loops for `max_turns` while the sample remains active (`:159-161`).

For every turn, `_build_chat_history()` concatenates the original `init_prompt` with accumulated `inter_history` (`:48-65`). The manager then calls `tokenizer.apply_chat_template()` on that complete message sequence (`:169-176`). After generation, `_update_interaction_history()` appends:

```text
assistant response
user environment observation
```

to `inter_histories` (`:68-83`, `:203-205`). Thus the manager maintains a Python list of structured messages and re-renders the complete conversation every turn.

There is no separate persistent dialogue object inside `MemGenModel`. The message/session state lives in the interaction-manager call and its `InteractionDataProto`.

### KV-cache ownership

`MemGenModel.generate()` accepts `input_ids`, `attention_mask`, generation configuration, and optional `latent_memory_bank`; it does not expose a cross-call `past_key_values` argument (`memgen/model/modeling_memgen.py:405-414`). Every call initializes:

```python
current_cache: DynamicCache = None
```

at `:458-463`.

During token generation, Reasoner forward calls use `use_cache=True` and `past_key_values=current_cache`, then assign `outputs.past_key_values` back to `current_cache` (`:642-668`). This is an **intra-call generation optimization only**.

When Trigger invokes Weaver and latent embeddings are inserted, the code explicitly sets `current_cache = None` so the modified embedding sequence is recomputed (`:620-623`). If maximum augmentation count is reached, the continuation path uses `reasoner.generate(... use_cache=False)` (`:625-638`). The method returns generated token IDs (and optionally an augmentation mask), not a reusable cache (`:676-688`).

Therefore:

- KV cache is used within one assistant generation.
- KV cache is not preserved across MAB chunk turns.
- The current interaction manager neither accepts nor stores a KV cache.
- Calling this baseline "History/KV" must not imply persistent cross-turn KV reuse. Run manifests should say `cross_turn_kv_reuse=false` and `intra_generation_kv_cache=true`.

### Original Trigger/Weaver behavior versus added bank

When `latent_memory_bank is None`, a positive Trigger decision still calls Weaver, maps Weaver output back through `weaver_to_reasoner`, and injects the newly generated `latent_inputs_embeds` into the current Reasoner sequence (`memgen/model/modeling_memgen.py:474-535`). This is original MemGen behavior.

When the added bank is present, the additional path retrieves bank slots, injects retrieved reasoner-space memories into the Reasoner candidate sequence, and writes the newly generated reasoner-space `latent_inputs_embeds` back to the bank (`:540-598`). The retrieved slots are combined only after Weaver has generated the new latent; Weaver input remains the current context-derived candidate sequence (`:513-524`). This is the current Version A boundary: retrieved bank memory enters Reasoner, not Weaver.

This distinction is why "Bank-off" is the correct experimental term. Bank-off does not mean Trigger-off, Weaver-off, latent-token-off, history-off, or KV-off inside generation.

### Session and reset boundary

`InteractionManager._create_session_memory_bank()` returns `None` when the optional configuration is absent or `enabled=false`; otherwise it creates and resets a new `LatentMemoryBank` and enforces `batch_size=1` (`interactions/base_interaction.py:104-120`).

`MultiTurnInteractionManager.run_agent_loop()` creates this bank once at episode entry and passes the same object to every turn (`interactions/multiturn_interaction.py:146-184`). The bank is a local variable and is no longer reachable after the episode returns. Integration tests verify that turns within one episode share the bank and a later `run_agent_loop()` starts from zero slots (`tests/test_latent_memory_bank_integration.py:400-425`).

`run_agent_loop()` is therefore still the correct **current session/reset boundary**, provided one call represents exactly one MAB context and all its chunk and query turns. Calling it once per chunk would be incorrect because it would reset `inter_histories` and recreate/reset the bank each time.

If a later external adapter bypasses `MultiTurnInteractionManager`, it must recreate this lifecycle explicitly with `start_session(context_id)` and `reset_session()` in `finally`. That is a later design option, not the MAB-2 default.

### Context-length caveat

The validated MAB row has two official chunks with measured lengths `[4319, 2119]`. Existing interaction configs commonly specify `max_prompt_length: 4096`, but `MultiTurnInteractionManager` applies the chat template directly to full messages before `generate()` and does not call a prompt-truncation helper in the per-turn generation path. `TensorHelper.max_prompt_length` is used elsewhere for output/tensor assembly, not as an explicit full-history truncation guard at `interactions/multiturn_interaction.py:169-185`.

MAB-2 must therefore preflight the rendered token count against the loaded Reasoner's actual context capacity. It must not assume `max_prompt_length=4096` will truncate or protect generation. If the full query history exceeds the model's declared context window or available memory, stop rather than silently dropping early chunks.

## How Original MemGen Handles Multi-Turn Context

### Direct answers

**Does it rebuild the full prompt history each turn?**  
Yes. It concatenates `init_prompts + inter_histories`, applies the chat template, and tokenizes the entire conversation on every turn.

**Does it maintain a messages list?**  
Yes. `init_prompts` and `inter_histories` are lists of `{role, content}` dictionaries. The latter grows by assistant response and next user observation after each turn.

**Does it use or expose `past_key_values` across turns?**  
No. `past_key_values` is used only inside each `MemGenModel.generate()` call. No cross-turn cache is accepted, returned, or stored by the interaction manager.

**Where is the session boundary?**  
For the current manager architecture, one `run_agent_loop()` call is one episode/session. Its local message history and optional bank are newly initialized at entry.

**Is `run_agent_loop()` still the correct reset boundary?**  
Yes. One MAB item/context should map to one call. All chunk-ingestion turns and the selected query must occur inside it. Session cleanup occurs when the call returns; failure handling in an adapter should additionally clear references in `finally` and assert no bank is retained.

### How MAB turns should map

For the validated two-chunk, one-query sample, the natural dynamic episode is:

```text
initial user turn: official memorization prompt for chunk 1
turn 1 assistant: short memorization acknowledgement generated by original MemGen
environment observation/user turn: official memorization prompt for chunk 2
turn 2 assistant: short memorization acknowledgement generated by original MemGen
environment observation/user turn: official query prompt for question 1
turn 3 assistant: scored answer
episode done -> session/history/bank boundary
```

This requires `max_turns = chunk_count + query_count`, which is 3 for MAB-2. The environment must identify the final assistant answer and must not score memorization acknowledgements.

The safest correctness policy is to preserve the complete structured message list and let the existing manager rebuild it every turn. Acknowledgements should be short and deterministically generated where existing decoding controls permit, but the adapter must not fabricate hidden memory state or skip the chunk-generation call: Trigger/Weaver only process the chunk through `MemGenModel.generate()`.

### Correctness versus runtime

| Strategy | Correctness relative to current source | Runtime | Recommendation |
|---|---|---:|---|
| Full-history rebuild each turn | Exact current multi-turn mechanism | Highest prefill cost; cumulative history is reprocessed | Use for MAB-2 and MAB-3 full-history |
| Cross-turn KV reuse | Not implemented; difficult around injected latent embeddings and cache invalidation | Potentially cheapest incremental prefill | Do not use initially |
| Rebuild only once at final query | Skips original MemGen processing of each chunk and Trigger/Weaver opportunities | Cheaper | Optional one-shot long-context baseline, not natural MAB baseline |
| Drop prior history at query | Changes information availability | Lower query cost | Compressed-memory or strict diagnostic only |

Persistent KV reuse is an optimization experiment, not a baseline requirement. It should be considered only after prompt-identical full-history runs pass and should have an equivalence test against full-history rebuild on the same turns.

## Revised Baseline Taxonomy

| Baseline | Query sees previous chunks in prompt history? | Cross-turn KV preserved? | Added bank created? | Bank writes? | Bank retrieval? | Retrieved latents enter | Scientific question |
|---|---|---|---|---|---|---|---|
| **Original MemGen History/KV Bank-off** | Yes, complete rendered multi-turn history | No; per-call/intra-generation KV only | No | No | No | N/A; original new Weaver latents may enter Reasoner | How does original MemGen perform on MAB when given its natural full dialogue history? |
| **MemGen + LatentBank V-A Full-history Bank-on** | Yes, prompt-identical to Bank-off | No; per-call/intra-generation KV only | Yes, once per context | Yes when Trigger/Weaver produces a write-eligible latent | Yes according to current bank path | Reasoner only; never Weaver | Does the added bank provide incremental benefit beyond information already available in full history? |
| **MemGen + LatentBank V-A Compressed-memory Bank-on** | No full chunk history at query; query retains only explicitly specified system/current-turn context | No initially | Yes, once per context | Yes during chunk ingestion | Yes at query/eligible turns | Reasoner only; never Weaver | Can the added latent bank replace long prompt history while preserving answer quality and reducing query context cost? |
| **Strict no-history Bank-off diagnostic** | No | No | No | No | No | N/A | Lower bound: what can the model answer without chunk history or the added bank? Not an original-MemGen baseline. |
| **Base Qwen full-history** (optional) | Yes, matched structured history | No initially | No | No | No | N/A; no Trigger/Weaver | What does original MemGen's Trigger/Weaver stack add over the underlying Reasoner backbone under matched history? |
| **Long-context MemGen Bank-off** (optional) | Yes, but as one consolidated context+query prompt rather than incremental turns | No cross-turn concept; one generation | No | No | No | N/A; original Trigger/Weaver may operate during the one generation | How does incremental interaction compare with a one-shot long-context presentation using MemGen? |

### Naming rules

- "History/KV" describes information availability, but every table and manifest must state the actual cache policy. For current code: `history_policy=full_rebuild`, `cross_turn_kv_reuse=false`, `intra_generation_kv_cache=true`.
- "Bank-off" means the added `LatentMemoryBank` is absent. It does not mean all memory-like mechanisms are absent.
- "Bank-on" means the added session-level bank exists and is passed across turns in one context.
- "Full-history" and "Compressed-memory" must appear in Bank-on result labels because they answer different questions.
- "Strict no-history" is diagnostic-only and must never replace Original MemGen History/KV Bank-off in the primary table.

## Recommended MAB-2 Target

Implement **Original MemGen History/KV Bank-off** first.

### Fixed contract

- Dataset: local transferred MemoryAgentBench Parquet.
- Split: `Conflict_Resolution`.
- Sub-dataset: `factconsolidation_sh_6k`.
- Scope: one context, first query only.
- Official chunking: unchanged; expected two chunks with validated lengths `[4319, 2119]`.
- Official MAB memorize/query templates and `post_process()` metrics.
- MemGen session: one `MultiTurnInteractionManager.run_agent_loop()` call for the entire context.
- Turns: two memorization turns plus one query turn.
- History: complete message history rebuilt each turn.
- Cross-turn KV reuse: false.
- Added bank: configuration absent or `enabled=false`, producing `latent_memory_bank=None`.
- Original Trigger/Weaver: unchanged and observable where feasible.
- Batch size: 1 for comparability with later Bank-on.
- Output: `results.json`, `diagnostics.jsonl`, `manifest.json`; an environment snapshot is also recommended.
- Status labels: `bank_off_success`, `bank_off_failed`, or equivalent explicit names; never generic `disabled_success`.

### Why this is first

It is the least invasive adapter target and establishes the correct prompt/history contract before adding bank state. It protects original behavior, provides a scoreable output path, reveals context-window/runtime constraints, and gives MAB-3 an exact prompt-identical comparator.

### MAB-2 adapter shape

Prefer a thin external MAB controller plus a MemGen-side evaluation harness that uses existing interaction-manager/model APIs. Do not add MemoryAgentBench logic to `modeling_memgen.py`, Weaver, Trigger, or training code. The MAB controller should load/chunk/format/score; the MemGen harness should own the one-episode message flow and return only generated response/accounting data.

For MAB-2, do not add KV-cache plumbing. Record that cache reuse is absent. The adapter can optimize model loading by keeping the model process alive, but model-process persistence must not preserve per-context history or bank state between samples.

## Recommended MAB-3 Target

Choose **A: MemGen + LatentBank V-A Full-history Bank-on first**.

### Scientific rigor

Full-history Bank-on changes one principal factor relative to MAB-2: the added bank. The dataset item, official chunks, structured messages, prompt history, query, decoding, model checkpoint, and metric remain fixed. The paired comparison asks whether the bank adds value when the same evidence is already available in prompt history.

Compression-first would change at least two factors simultaneously:

1. enable the added bank;
2. remove chunk history from the query prompt.

A gain or loss could not be attributed cleanly to bank quality versus prompt removal. It also introduces additional adapter control because current `MultiTurnInteractionManager` always rebuilds all history.

### Engineering safety

Full-history Bank-on follows the already-tested bank lifecycle: one bank created at `run_agent_loop()` entry, shared across turns, released at episode end. Compressed-memory mode requires an external history-pruning/manual-loop policy while preserving the bank, which is not a direct current manager feature.

MAB-3 must keep:

- `batch_size=1`;
- prompt/history hashes identical to MAB-2 before bank-specific latent insertion;
- retrieved latents in Reasoner only;
- Weaver input derived from current context only;
- one fresh bank with zero initial slots per context;
- post-session reset/no retained bank references.

The full-history comparison may show a ceiling effect. That is acceptable: it is still the clean causal test. Compressed-memory Bank-on becomes the next scientific question after mechanism and prompt parity are proven.

## Adapter Design Implications

### Session API

The adapter should model one context as one episode, not one request per chunk:

```text
start context -> one run_agent_loop()
  chunk 1 -> generation/acknowledgement
  chunk 2 -> generation/acknowledgement
  query 1 -> final scored generation
return -> clear history and optional bank
```

If later requirements force a manual external loop, it must expose explicit `start_session`, `memorize`, `query`, and `reset_session` operations and must preserve a single bank object only inside that session. That manual route should not be used for MAB-2 unless the existing dynamic manager cannot express the MAB episode.

### Prompt parity

Persist structured message objects, not concatenated ad hoc strings. Render them with the MemGen tokenizer's chat template immediately before each generation. Save per-turn hashes and token counts so MAB-2 and MAB-3 Full-history can prove identical textual inputs before bank injection.

The adapter must define acknowledgement handling explicitly. The natural path is to let original MemGen generate a short acknowledgement for each memorization prompt and retain that actual assistant response in history. If acknowledgement normalization is later introduced, it must be identical in all compared baselines and documented as an adapter policy.

### Cache policy

MAB-2/MAB-3 initial cache contract:

```text
cross_turn_kv_reuse: false
intra_generation_kv_cache: true when current MemGen generation path uses it
cache_invalidated_after_latent_injection: true
```

Do not report `kv_cache_used=true` without separating these two scopes. A single boolean is ambiguous.

### Bank-off assertions

For Original MemGen History/KV Bank-off:

- `_create_session_memory_bank()` returns `None`.
- `MemGenModel.generate(... latent_memory_bank=None)` on every turn.
- `bank_created=false`.
- write, retrieval, replacement, and slot counters all equal zero or are explicitly unavailable because no bank object exists.
- Trigger and Weaver remain governed by original model configuration.

### Bank-on assertions

For Version A:

- one bank is created at context start with zero slots;
- all chunk/query turns receive the same bank object;
- writes store detached reasoner-space `latent_inputs_embeds`;
- retrieval counters and score summaries are attributable to turns;
- retrieved slots augment only the Reasoner candidate sequence;
- Weaver input count/hash matches the current-context-derived input and excludes retrieved slots;
- the next context starts with zero slots.

### Compressed-memory implications

Compressed-memory Bank-on should be specified only after MAB-3 Full-history passes. Its query prompt should retain system instructions and the official query but omit prior chunk text and acknowledgement history. Chunk ingestion still requires generation calls so Trigger/Weaver can produce bank writes. The exact retained-message policy must be pre-registered; otherwise "compressed" is not reproducible.

## Diagnostics Requirements

### Required identity and task fields

- `run_id`
- `baseline_name`
- `context_id`
- `query_id`
- `task_name`
- `split`
- `sub_dataset`
- MemGen commit/dirty-state marker
- MemoryAgentBench commit
- local Parquet SHA-256
- model/config identifier without private checkpoint paths

### Required history and cache fields

- `chunk_count`
- `turn_index`
- `input_len`
- `output_len`
- `prompt_history_token_len`
- `full_history_included`
- `history_policy` (`full_rebuild`, `compressed`, or `none`)
- `cross_turn_kv_reuse`
- `intra_generation_kv_cache`
- `kv_cache_reused_token_count` (expected 0 initially)
- per-turn rendered-prompt hash
- context-window capacity and headroom

### Required bank and mechanism fields

- `bank_created`
- `bank_write_count`
- `bank_retrieval_count`
- `bank_slot_count`
- `replacement_count`
- retrieved slot/latent count
- top retrieval scores
- `retrieved_latents_enter_reasoner`
- `retrieved_latents_enter_weaver` (must be false)
- `trigger_count` if observable
- `trigger_positive_count` if observable
- `weaver_call_count` if observable
- Weaver input token counts/hashes if observable

For Bank-off, all bank fields must resolve to false/zero rather than being silently omitted, except score arrays which may be empty.

### Required result and resource fields

- `prediction`
- `gold_answers` in the controlled result artifact; logs may use count/hash/redaction to avoid unnecessary exposure
- official metric fields (`substring_exact_match` primary plus emitted EM/F1/ROUGE)
- per-turn and total latency
- memory-construction versus query latency where meaningful
- peak CUDA memory if available
- errors and stop reason
- final response identification/parsing status

Accuracy alone remains insufficient. A valid Bank-on claim requires evidence that writes occurred, relevant slots were retrieved, the correct injection boundary was respected, and no state crossed contexts. A valid Bank-off claim requires evidence that the added bank was never constructed while full history remained available.

## Stop Criteria

Stop MAB-2 or MAB-3 and mark the run invalid if any condition holds:

- One MAB context is split across multiple `run_agent_loop()` calls without an explicit equivalent session layer.
- The rendered query history omits prior chunks in a baseline labeled full-history.
- A baseline claims cross-turn KV reuse although no `past_key_values` is passed across calls.
- Full rendered history exceeds the Reasoner's declared context capacity or causes silent truncation.
- MAB-2 creates a bank or records any bank write/retrieval/slot.
- Trigger or Weaver is disabled relative to the accepted original MemGen configuration without an explicit ablation label.
- MAB-2 and MAB-3 Full-history differ in message content, turn order, decoding, or query formatting.
- Enabled mode runs with `batch_size != 1`.
- A new context observes nonzero initial bank slots or any prior context identifier.
- Stored bank memory is not detached reasoner-space `latent_inputs_embeds`.
- Retrieved bank memory enters Weaver.
- Bank-on reports accuracy without observable bank lifecycle/write/retrieval diagnostics.
- The final query response cannot be separated from memorization acknowledgements.
- Official `post_process()` cannot score the output JSON.
- The validated local data contract changes unexpectedly: not 8 total rows, not 1 matching row, or not 2 official chunks for this pinned artifact/data file.
- Adapter work would require modifying Weaver training, Trigger training, or benchmark core logic.
- The existing unrelated `modeling_memgen.py` modification changes unexpectedly during adapter work.

## Terminology Corrections

Use the following experiment-facing replacements:

| Avoid | Use |
|---|---|
| `MemGen disabled` | `Original MemGen History/KV Bank-off` |
| `disabled adapter` | `Bank-off adapter` |
| `strict disabled` | `Strict no-history Bank-off diagnostic` |
| `enabled Version A` | `MemGen + LatentBank V-A Full-history Bank-on` or `... Compressed-memory Bank-on` |
| `no-memory baseline` for original MemGen | Do not use; original Trigger/Weaver and prompt history remain available |

Code-level phrases such as `latent_memory_bank.enabled=false` and "disabled path" may remain when referring narrowly to the configuration branch and regression invariant. They must not be used as the scientific baseline name.

Older benchmark notes currently contain phrases such as "MemGen disabled", "disabled adapter", and "strict disabled". They should be interpreted as historical terminology until revised. New manifests, result tables, adapter plans, and the MAB-2/MAB-3 notes must use the taxonomy in this report. This review does not rewrite historical evidence files.

## Dirty Worktree Status

Recorded before writing this review:

```text
## rlm-memory-bank...origin/rlm-memory-bank [ahead 8]
 M memgen/model/modeling_memgen.py
?? research_notes/benchmarks/
```

The existing tracked modification in `memgen/model/modeling_memgen.py` is unrelated to this review. Its current diff adds `trust_remote_code=True` to model/config/tokenizer loading at lines around 719-738 (5 insertions, 5 deletions). It was not touched or reverted.

The benchmark planning notes are untracked under `research_notes/benchmarks/`. This report adds only:

```text
research_notes/benchmarks/memoryagentbench_adapter_strategy_review.md
```

No tracked MemGen source file was changed by this planning task.

## Final Recommendation

Proceed to MAB-2 with **Original MemGen History/KV Bank-off**, implemented as one full-history multi-turn episode per MAB context with no added bank and no cross-turn KV reuse. For the validated sample, use two memorization turns and one scored query turn inside one `run_agent_loop()` call. Preserve original Trigger/Weaver behavior, use `batch_size=1`, preflight actual rendered-history length, and emit official-like results plus mechanism/history/cache diagnostics.

Proceed next to **MemGen + LatentBank V-A Full-history Bank-on**, not compressed-memory first. Require prompt and turn parity with MAB-2 so the bank is the only principal intervention. Only after that comparison is valid should the project test **Compressed-memory Bank-on** as a separate memory-replacement question.

Do not add cross-turn KV-cache reuse, strict no-history as a primary baseline, Base Qwen, or one-shot long-context MemGen to the first implementation milestone. Those are later diagnostics/ablations. Do not modify Weaver training, Trigger training, MemGen model core, or MemoryAgentBench core to complete MAB-2.
