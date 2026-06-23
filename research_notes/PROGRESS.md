# 项目进展

## Current Project State (2026-06-22)

- MAB-5A detective_qa compressed-memory n10 is complete and preserved as the
  current MAB reference baseline.
- Result: 10/10 valid contexts; compressed Bank-off exact match `0.0`;
  compressed Bank-on exact match `0.0`; `output_changed=10`; retrieval active
  in all contexts; `query_write_count=0`; no cross-context leakage.
- Current Version A boundary: retrieved memory enters Reasoner only and does not
  enter Weaver. Written memory is Weaver-generated reasoner-space
  `latent_inputs_embeds`.
- Original full-history detective_qa exceeds the 32,768-token capacity. It is
  `over_capacity_invalid`, was not run in MAB-5A, and must not be silently
  truncated.
- Mechanism diagnosis: with shared `threshold=0.03`, retrieval stayed active
  while `write_back()` repeatedly replaced matched slots. Final slot counts
  were `[1, 2, 2, 5, 6, 5, 6, 7, 4, 7]` across contexts with 25-50 chunks.
- The match score compares a query built from `candidate_inputs_embeds` with
  `slot.key`; it does not compare the new Weaver latent with an old slot.
- Canonical MAB index, results, next steps, runbook, and mechanism plan are under
  `research_notes/benchmarks/`.
- MAB-5B raised shared-threshold runner and test have now been executed on a
  CUDA-capable runtime. Result: 10/10 valid contexts; Bank-off exact match
  `0.0`; Bank-on exact match `0.0`; `output_changed=5`; retrieval active in all
  contexts; final slot counts reached `[8, 8, 8, 8, 8, 8, 8, 8, 8, 8]`; no
  cross-context leakage; query writes `0`; retrieved memory remained
  Reasoner-only.
- MAB-5C decoupled retrieval-update thresholds have now been executed on the
  same detective_qa n10 slice. The preliminary runtime-patch artifact is
  historical only; the canonical checked-in runner rerun produced the source
  of truth. Result: 10/10 valid contexts; Bank-off exact match `0.0`; Bank-on
  exact match `0.0`; `output_changed=10`; retrieval active in all contexts;
  query-turn retrieval active in all contexts; final slot counts reached
  `[8, 8, 8, 8, 8, 8, 8, 8, 8, 8]`; retrieved latents remained Reasoner-only;
  query writes `0`; cross-context leakage `false`.
- Current next action: treat MAB-5C as diagnostic evidence, not a final
  performance win. A capacity-ablation follow-up at `max_slots=16` is the
  next comparison if we want to separate threshold effects from slot-capacity
  effects.
- Old shared-threshold behavior must remain reproducible by default.
- `output_changed` is activation evidence, not improvement. Official exact
  match and relaxed diagnostics remain separately labeled.

## Historical Project State Snapshot (through 2026-06-20)

This section preserves the pre-MAB handoff. Its next-step language is
superseded by the current state above and by
`benchmarks/memoryagentbench_next_steps.md`.

- 当前状态：Version A-aligned 已实现、已提交，并已在 Phase R2 / R2-fix
  修订为 last-retrieved decay；当前没有 formal target-task main result。
- 状态：`completed`
- 最后更新：2026-06-18 notes consolidation
- Canonical revision：`c95e2bd feat: revise Version A-aligned decay to last-retrieved age`
- 衰减 / fallback 实现审计：`completed`
- 方法 / 计划对齐更新：`completed`
- Step 2 结构化检索上下文：`completed`
- Step 3 线程感知写回：`completed`
- Step 4 thread-update 机制 smoke：`completed`
  (`EXP-20260612-024`)
- Phase R2 last-retrieved decay revision：`completed`
- Phase R2-fix retrieval-step / write_back clarification：`completed`
- 当前标准状态：
  - Phase 0-7 是当前 accepted formal result set
  - Phase 8A GSM8K pilot 是 historical / exploratory evidence only
  - Phase 8C-alt controlled study 是 historical / exploratory mechanism
    evidence only
  - Phase 8D-0 / R4-1A 是 infrastructure discovery / preflight only
  - Phase R2 / R2-fix 定义当前 mechanism，但没有产生 formal
    target-task result
  - Version A-aligned current retrieval 使用 last-retrieved decay
  - full-bank `new_thread` eviction 使用 largest `last_retrieved_age`
  - no fallback top-1
  - retrieved memory 保持 Reasoner-only，不进入 Weaver
  - stored memory 是 reasoner-space `latent_inputs_embeds`
  - memory 是 session-local
  - enabled memory 仍要求 `batch_size=1`
  - Version B 未开始
  - Controlled mechanism study 已 closeout
  - TriviaQA 仍无正式 baseline 或正式 performance 结果
  - R4 Search-R1 / TriviaQA infrastructure validation 已完成，带 caveats
  - R4 TriviaQA threshold calibration 已完成：默认 threshold=0.7 在
    calibration samples 0..19 上产生 0/20 个自然触发
  - 衰减评分量表：min 0.010, max 0.054, mean 0.036, median 0.037
  - 已校准 threshold=0.04 作为第一个候选值（仅基于评分分布，非 rewards）
  - held-out samples 20..79（校准后）：
    disabled 35/60 vs Version A t=0.04 35/60，净增益 0
  - 观察到 1 个 rescue（样本 53）和 1 个 regression（样本 21）；无净改进也无净退步
  - 关键机制 caveat：检索到的潜在记忆（retrieved latent memory）是在证据前（pre-evidence）生成的，可能放大查询实体显著性（query-entity salience），而非证据锚定答案
  - 当前推荐下一步：暂时不要扩大规模；对救援案例与回归案例进行基于工件的案例研究，以理解记忆究竟何时帮助 vs 伤害
  - Threshold comment caveat：配置注释称 threshold 为"cosine similarity threshold"，但实际应用于 decayed retrieval score 而非 raw cosine
  - MemoryAgentBench / LongMemEval 只记录为 future candidates
  - Version B 继续 deferred
  - MAB-5A detective_qa compressed-memory n10 已完成：
    10/10 valid, Bank-off `0.0`, Bank-on `0.0`, output_changed `10`, no leakage
    detected, retrieval active in all contexts
  - MAB-5A mechanism finding：retrieved scores roughly `0.030-0.064`, final
    slot counts stayed low (`[1, 2, 2, 5, 6, 5, 6, 7, 4, 7]`), suggesting
    over-merge / over-compression under the current low threshold
  - current MAB follow-up recommendation: preserve the evidence first, then
    test decoupled retrieve/update thresholds in a later mechanism experiment
- 历史解释边界：
  - Phase 8A 记录的是 R2 前 historical write-age mechanism，不得重解释为
    current last-retrieved-decay evidence。
  - Phase 8C-alt controlled runs 可作为 runtime / lifecycle / boundary
    evidence，但不是 target-task performance evidence。
  - Controlled diagnostic subset 只能作为 future mechanism diagnostic；不能替代
    TriviaQA。
- 最新验证：
  - `tests 76/76 passed`
  - Phase R2 / R2-fix 只修改代码、tests 和 notes；没有运行正式实验
  - R4 Search-R1 `/retrieve` schema validation passed on port `8000`
  - R4 disabled-memory TriviaQA 1-sample dynamic smoke completed with
    `valid_run=True`
  - R4 Version A-aligned TriviaQA 1-sample dynamic smoke completed with
    `valid_run=True`
  - R4 threshold-positive diagnostic exercised non-empty retrieved latent memory
    under diagnostic-only `threshold=0.01`
  - R4 LatentMemoryBank scoring/recency semantics audit completed — confirmed
    activation path uses last-retrieved-age decay with exact age =
    `retrieval_step - last_retrieved_step`
  - R4 default-threshold natural trigger scan (samples 1..5) 完成：0/5 触发
  - R4 20-sample threshold calibration score scan (samples 0..19) 完成
  - R4 threshold=0.04 calibrated behavior scan (samples 0..19) 完成：
    8/20 触发，exactly matched offline estimate
  - R4 held-out exploratory comparison (samples 20..39) 完成：
    disabled mean 0.60, Version A t=0.04 mean 0.55, 1 regression
  - R4 sample 21 regression case study 完成：记忆诱导的回归（memory-induced regression）
  - R4 triggered held-out audit (samples 20..39 memory-triggered) 完成：
    0 helpful, 1 harmful, 5 neutral
  - R4 fresh held-out rescue/regression scan (samples 40..79) 完成：
    1 rescue (sample 53), 0 regression, mean diff +0.025
  - R4 combined held-out interpretation (samples 20..79) 完成：net gain 0
  - R4 expanded held-out paired eval (samples 80..179) 完成：
    disabled 47/100 vs Version A t=0.04 47/100，rescue 1，regression 1，
    threshold-passed 37/100，net gain 0
  - R4 disabled full TriviaQA baseline completed after retrying the final
    range in smaller chunks:
    disabled 5148/7993 = 0.6441
    - missing 0, duplicates 0
    - original disabled_s7000_7992 chunk was preserved in its stuck/no-artifact
      state and not used for the final aggregate
    - original completed chunks: 0000..6999
    - retry chunks: 7000..7499, 7500..7799, 7800..7992
    - retry artifact summary:
      - 7000..7499: 500/500 valid, 0 retrieval-blocked
      - 7500..7799: 295/300 valid, 5 retrieval-blocked
      - 7800..7992: 193/193 valid, 0 retrieval-blocked
    - final aggregate confirms the disabled path is operational end-to-end on
      the full TriviaQA validation set
  - 当前结论：Version A shows sparse steering but no net gain on the
    larger held-out slice；这仍然是 exploratory R4 evidence，不是 formal
    target-task benchmark
- 停止条件：已达到；没有明确批准，不要进入新的实现或实验阶段。

## Historical Next Step (superseded by MAB-5C)

The following was the R4 handoff before MAB-5A completed. It is retained for
provenance and is not the current project action.

1. R4 infrastructure validation is complete with caveats.
2. Threshold calibration is complete with caveats:
   - default threshold 0.7 is inappropriate for observed TriviaQA decayed-score
     scale
   - threshold 0.04 is a first calibrated candidate, not an optimal threshold
3. Do NOT immediately run larger benchmarks, tune thresholds, or implement
   timing constraints.
4. Primary next step: read-only case study of rescue sample 53 vs harmful
   sample 21.
   - Goal: understand when memory helps versus hurts before scaling
   - Questions: is memory acting as useful latent prior, evidence-grounded clue,
     query-salience amplifier, or noisy perturbation?
5. Later, after mechanism analysis: prioritize suppress pre-evidence memory
   write / evidence-gated write before any further threshold or retrieval
   ablation.
6. Version B remains deferred until a separate explicit decision.

## 研究目标

在不改变 MemGen 训练流程的前提下，在推理时加入一个可选的 session-level Retrieval-Augmented Recurrent Latent Memory Bank。

## Phase 0 结果

Phase 0 在当前路线图下仍然完成。

## Phase 1 结果

Phase 1 已完成。

- [x] 审计推理入口文件和主 dispatch。
- [x] 审计配置加载和运行时配置传递。
- [x] 审计静态和动态 session / sample / episode 边界。
- [x] 定位 Trigger 调用位置。
- [x] 定位 Weaver 调用位置。
- [x] 定位 latent memory 生成和 Reasoner 注入位置。
- [x] 定位生成输出和评估 hook。
- [x] 标记受保护的 Weaver / Trigger 训练边界。
- [x] 评估候选 LatentMemoryBank 集成点和风险。
- [x] 只更新 research notes。没有修改核心代码或训练流程。

## Phase 2 结果

Phase 2 已完成，但有阻塞性 caveat。

- [x] 检查环境、仓库状态和可运行项目环境。
- [x] 找到一个官方最小评估路线：GSM8K + Qwen2.5-1.5B-Instruct + Weaver-SFT。
- [x] 验证当前 `base` 环境不适合运行 MemGen。
- [x] 验证 `memgen` 环境可以正确初始化项目。
- [x] 验证本地 base-model 和 dataset cache 可以离线使用。
- [x] 通过 `Config -> MemGenModel.from_config -> MemGenRunner.evaluate()` 在 1 个 GSM8K sample 上运行原始评估路径。
- [x] 确认模型加载、数据集加载、交互设置和生成启动都能在推荐环境中工作。
- [x] 确认官方 static-eval 路径当前在结果记录阶段失败，导致 `answer.json` 为空。
- [x] 另外确认：当在 script-only harness 中绕过 recorder 逻辑时，原始生成路径可以产生 completion。
- [x] 再次确认官方 LoRA 加载仍然不可信；本 Phase 的 smoke 结果不能作为有效科学 baseline。
- [x] 只更新 research notes。没有修改核心代码、Weaver 训练或 Trigger 训练逻辑。

## 临时环境对齐结果

临时环境对齐阶段已完成。

- [x] 阅读了 `README.md`、`requirements.txt`、`memgen.yml`、官方 Qwen2.5 GSM8K 评估脚本和 GSM8K 配置。
- [x] 确认当前 shell 是 `base` 环境，Python 为 `3.13.9`，不能用于运行 MemGen。
- [x] 确认已有 `memgen` 环境位于 `/home/baishilong/miniconda3/envs/memgen`。
- [x] 确认已有 `memgen` Python 为 `3.10.20`，符合 README 安装说明。
- [x] 确认所有必需 Python imports 成功，且 `pip check` 没有 broken requirements。
- [x] 确认在文件系统 sandbox 外运行时，`memgen` 环境可以在一张 NVIDIA RTX A6000 上使用 CUDA 和 BF16。
- [x] 确认 GSM8K YAML、本地 Qwen snapshot、官方 MemGen checkpoint 和本地 GSM8K dataset cache 可读。
- [x] 判断在 Repair Phase 前没有必要重建环境或安装 package。
- [x] 记录了环境规格漂移和损坏的 PATH-level `conda` shim。
- [x] 只修改 research notes。没有修改项目代码或环境 package。

## 临时 Repair Phase 结果

临时 Repair Phase 已完成。

- [x] 用原始 nested PEFT 加载路径复现 `BUG-0001`。
- [x] 确认原始路径期望 392 个错误嵌套的 adapter keys，而每个官方 checkpoint 实际包含 112 个训练过的 q/v LoRA tensors。
- [x] 只替换 checkpoint adapter 恢复逻辑；初始 Weaver 和 Trigger 训练 adapter 构造保持不变。
- [x] 对照官方 safetensors 验证所有 112 个 Weaver 和所有 112 个 Trigger tensors，没有 missing、unexpected、shape mismatch 或 value mismatch。
- [x] 复现 `BUG-0002`，其原因是 caller / recorder contract mismatch：runner 传入一个 string 和一个 dictionary，而 recorder 需要两个 lists。
- [x] 规范化 gathered static-eval batches，并保持 sample 顺序和 metric 语义。
- [x] 使用 seed 42 和 batch size 1，在一个 GSM8K sample 上运行官方 `Config -> MemGenModel.from_config -> MemGenRunner.evaluate()` 路径。
- [x] 确认产生非空 `answer.json`，包含一条 prediction 和一条 summary record。
- [x] 确认生成路径调用 Trigger decision entry 85 次、Weaver prompt augmentation 1 次、Weaver inference augmentation 3 次。
- [x] 没有修改 Weaver training、Trigger training、训练脚本、依赖或环境 package。
- [x] 没有进入 Phase 3。

## Phase 3 结果

Phase 3 已完成。

- [x] 接受 `memgen-gsm8k-sft-official-v1` 作为 Original MemGen comparator。
- [x] 固定 comparison set 为 GSM8K `main/test` indices 0 到 19。
- [x] 使用 seed 42、batch size 1、greedy decoding、maximum response length 1024。
- [x] 运行官方 `Config -> MemGenModel.from_config -> MemGenRunner.evaluate()` 路径。
- [x] 产生 20 条非空 prediction records 和 1 条 summary record。
- [x] 在固定 20-sample 子集上记录 mean `compute_reward=0.60`。
- [x] 再次确认 Weaver 和 Trigger adapter 精确加载：112/112。
- [x] 记录 1,722 次 Trigger decision calls、20 次 Weaver prompt calls、43 次 Weaver inference calls。
- [x] 记录总延迟 115.728 秒，平均延迟 5.786 秒 / sample，peak allocated CUDA memory 9,415,716,352 bytes。
- [x] 重放固定 samples 0、1、2，并获得完全相同的 response-token 和 augmentation-mask SHA-256 hashes。
- [x] 将 prediction、verification、TensorBoard 和 metric-contract artifacts 归档到 `outputs/baseline/`。
- [x] Phase 3 没有修改核心方法或训练代码。
- [x] 没有进入 Phase 4。

## Phase 4 结果

Phase 4 已完成。

- [x] 在 `memgen/model/latent_memory_bank.py` 中添加独立的 `LatentMemoryBankConfig`、`LatentMemorySlot` 和 `LatentMemoryBank`。
- [x] 添加默认禁用的 `configs/latent_memory_bank/default.yaml`。
- [x] 在 `tests/test_latent_memory_bank.py` 中添加 16 个标准库 unit tests。
- [x] 实现 disabled 和 empty-bank 的 no-op 行为。
- [x] 实现 recent-token mean query pooling 和 memory mean key pooling。
- [x] 实现带 exponential recency decay 的 cosine similarity。
- [x] 实现 threshold、top-k 和 threshold-plus-top-k retrieval。
- [x] 实现 append、replace-lowest-score 和 replace-oldest capacity 行为。
- [x] 强制 Phase 4 batch size 1 tensor shapes。
- [x] 强制写入时 detach and clone，检索时 detached clone。
- [x] 确认 caller 修改 retrieved tensors 或 nested metadata 不会改变 bank-owned slot state。
- [x] 实现显式 storage 和 retrieval device / dtype movement。
- [x] 定义 `_step` 为 successful memory-write count，而不是 generation-token count。
- [x] 定义 `replace` 为 lowest-`last_score` replacement；当所有 slots 都 unscored 时 fallback 到 oldest slot。
- [x] 添加 debug summary 和 detached state-dict-like snapshots。
- [x] 通过 compilation、YAML parsing 和全部 16 个 unit tests。
- [x] 确认 production inference、`generate()`、runner、trainer 和 training scripts 都没有 import 或 call 该 module。
- [x] 确认 import `MemGenModel` 不会加载 memory-bank module。
- [x] 没有修改现有 GSM8K 配置或原始推理行为。
- [x] 没有进入 Phase 5。

## Phase 5 结果

Phase 5 已完成。

- [x] 将 optional LatentMemoryBank 只集成到 inference。
- [x] 保持 bank 为 session-local，并由每次 interaction-manager `run_agent_loop()` 调用拥有。
- [x] single-turn session 每个 session 使用一个 bank；multi-turn episode 内所有 turns 共享一个 bank。
- [x] 将 bank 显式传入 `MemGenModel.generate()`，没有把任何 bank object 存在 `MemGenModel` 上。
- [x] 通过保持 `latent_memory_bank=None` / `enabled=false` 在原始代码分支上，保留原始 disabled path，不进行新的 retrieval、write、mask 或 tensor-packaging 工作。
- [x] 实现 Version A retrieval：retrieved memory 只注入 Reasoner 路径，永远不会传给 `reasoner_to_weaver()`、`augment_prompt()` 或 `augment_inference()`。
- [x] 只把 `weaver_to_reasoner(...)` 之后的 reasoner-space `latent_inputs_embeds` 写入 bank。
- [x] 添加显式 retrieved-memory attention-mask handling，以及单独的 debug bookkeeping：`memory_write_count`、`memory_retrieve_count`、`retrieved_latent_count`、`new_latent_count`、`slot_count`。
- [x] 拒绝 `enabled=true` 且 `batch_size > 1` 的 evaluation，同时保持 disabled mode 不受限制。
- [x] 添加轻量 integration tests：disabled no-op、empty-bank no-op、session reset、no cross-sample leakage、Reasoner-only injection、reasoner-space writes、dtype/device compatibility、enabled batch-size rejection。
- [x] 通过 `py_compile`、完整 `unittest` 和 `git diff --check`。
- [x] 在 GSM8K samples `0..2` 上运行 disabled-path golden replay `EXP-20260612-010`；response-token hashes、augmentation-mask hashes、Trigger call count、Weaver prompt count 和 Weaver inference count 都与 `EXP-20260611-007` 完全一致。
- [x] 在 GSM8K sample `0` 上运行 enabled debug `EXP-20260612-011`；运行未崩溃，并记录 4 次 writes、3 次 retrievals、24 个 retrieved latent tokens、32 个 newly written latent tokens 和 4 个 resident slots。
- [x] 没有修改 `memgen/trainer/**`、`scripts/train/**`、Weaver training logic、Trigger training logic，也没有实现 Version B。
- [x] 没有进入 Phase 6。

## Phase 6 结果

Phase 6 已完成。

- [x] 在 GSM8K test IDs `0..19` 上，对 frozen Phase 3 baseline `EXP-20260611-006` 运行 20-sample disabled-path equivalence test。
- [x] 使用 seed `42`、batch size `1`、greedy decoding 和 maximum response length `1024`。
- [x] 保持 `latent_memory_bank` disabled，并验证 `memory_bank_debug=null`。
- [x] 确认 `answer.json` 非空，且包含恰好 20 条 prediction records 和 1 条 summary record。
- [x] 确认 summary `compute_reward=0.60`，与 `EXP-20260611-006` 完全一致。
- [x] 确认每个 response-token SHA-256 hash 都与 frozen baseline 匹配。
- [x] 确认每个 augmentation-mask SHA-256 hash 都与 frozen baseline 匹配。
- [x] 确认 Trigger decision calls 完全一致：`1722`。
- [x] 确认 Weaver prompt augmentation calls 完全一致：`20`。
- [x] 确认 Weaver inference augmentation calls 完全一致：`43`。
- [x] 再次确认 adapter loading integrity：Weaver `112/112`、Trigger `112/112`，missing、unexpected、shape 或 value mismatch 都为 0。
- [x] 重新运行 `git diff --check`、`py_compile` 和完整 `unittest`，全部通过。
- [x] 确认 `memgen/trainer/**`、`scripts/train/**`、`memgen/model/weaver.py` 或 `memgen/model/trigger.py` 下没有 diff。
- [x] 没有发现 disabled-path regression，也没有新的 blocking bug。
- [x] 本阶段没有修改核心方法代码。
- [x] 没有进入 Phase 7。

## Phase 7 结果

Phase 7 已完成。

- [x] 只运行 enabled-path bounded debug 和 stability checks；没有做 performance claim。
- [x] 保持 seed `42`、batch size `1`、greedy decoding 和 maximum response length `1024`。
- [x] 确认运行前 `git status` 干净，protected training paths 没有 diff。
- [x] 重新运行 `git diff --check`、`py_compile` 和完整 `unittest`，全部通过。
- [x] 在一个 GSM8K test sample 上运行 Tier 1 smoke enabled mode。
- [x] 在 GSM8K test samples `0..2` 上运行 Tier 2 small stability enabled mode。
- [x] 在 GSM8K test samples `0..4` 上运行 Tier 3 bounded capacity enabled mode。
- [x] 确认所有 enabled runs 都写入非空 `answer.json`，并包含预期数量的 prediction records 和 1 条 summary record。
- [x] 所有 Tier 中都没有观察到 crash、NaN、OOM、CUDA error、shape mismatch、device mismatch 或 dtype mismatch。
- [x] 确认每个 single-turn session 都从 `initial_slots=0` 开始。
- [x] 确认 Tier 2 或 Tier 3 sessions 中没有 cross-sample leakage。
- [x] 确认 retrieved memory 仍然是 Reasoner-only；Weaver input token counts 始终匹配 `reasoner_to_weaver` input token counts。
- [x] 确认 stored latent memories 仍然是 hidden size `1536` 的 reasoner-space tensors。
- [x] 确认 slot storage 显式记录：CPU storage、original device `cuda:0`、original dtype `torch.bfloat16`、stored dtype `torch.bfloat16`。
- [x] 确认 `slot_count` 从未超过 `max_slots=8`。
- [x] 在该 bounded run 中没有观察到 replacement-policy activation，因为最大 per-session slot count 是 `4`。
- [x] 在 Phase 7 后运行 capacity-trigger supplement，设置 `max_slots=2`，在真实 enabled session 中确认 replacement activation。
- [x] 确认 supplement 记录 `append_count=2`、`replace_count=2`、`rejected_write_count=0`，以及 `update_action_trace=["append", "append", "replace", "replace"]`。
- [x] 确认 supplement 保持 `final slot_count=2 <= max_slots=2`，同时 `memory_write_count=4 > max_slots`。
- [x] 解决唯一剩余的 Phase 7 warning：现在已经在真实 enabled debug path 中观察到 replacement policy。
- [x] 记录 Tier 1 stats：writes `4`、retrieves `3`、retrieved latents `24`、new latents `32`、slot count `4`、latency `8.658 s`、peak CUDA memory `9,385,351,168` bytes。
- [x] 记录 Tier 2 per-session stats：
  sample 0 -> writes `4`、retrieves `3`、retrieved latents `24`、new latents `32`、slot count `4`；
  sample 1 -> writes `2`、retrieves `1`、retrieved latents `8`、new latents `16`、slot count `2`；
  sample 2 -> writes `4`、retrieves `3`、retrieved latents `24`、new latents `32`、slot count `4`；
  total latency `14.066 s`、mean latency `4.689 s/sample`、peak CUDA memory `9,385,351,168` bytes。
- [x] 记录 Tier 3 per-session stats：
  sample 0 -> writes `4`、retrieves `3`、retrieved latents `24`、new latents `32`、slot count `4`；
  sample 1 -> writes `2`、retrieves `1`、retrieved latents `8`、new latents `16`、slot count `2`；
  sample 2 -> writes `4`、retrieves `3`、retrieved latents `24`、new latents `32`、slot count `4`；
  sample 3 -> writes `2`、retrieves `1`、retrieved latents `8`、new latents `16`、slot count `2`；
  sample 4 -> writes `4`、retrieves `3`、retrieved latents `24`、new latents `32`、slot count `4`；
  total latency `21.562 s`、mean latency `4.312 s/sample`、peak CUDA memory `9,395,434,496` bytes。
- [x] Phase 7 没有发现新的 blocking bug。
- [x] 本 Phase 只修改 debug harness 和 research notes。
- [x] 添加 debug-only bank summary fields 和 debug-harness CLI overrides，用于 capacity-trigger validation；没有改变 disabled-path 或 training-path 语义。
- [x] 没有修改 `memgen/trainer/**`、`scripts/train/**`、Weaver training logic、Trigger training logic，也没有实现 Version B。
- [x] 没有进入 Phase 8。

## 2026-06-12 - Phase 8A Core Ablation Pilot

- 状态：`completed`
- 总体结果：`PASS`
- 范围：
  - 只做 pilot
  - GSM8K test sample IDs `0..19`
  - `sample_count=20`
  - `seed=42`
  - `batch_size=1`
  - greedy decoding
  - `max_response_length=1024`
  - 没有 latest-k retrieval
  - 没有 random retrieval
  - 没有 Version B
- 比较组：
  - `G0` disabled anchor：复用 `EXP-20260612-013`，frozen baseline reference 为 `EXP-20260611-006`
  - `G1` Version A anchor：`EXP-20260612-019`
  - `G4` cosine retrieval without recency decay：`EXP-20260612-020`
  - `G6` append-only update：`EXP-20260612-021`
  - `G7` replace update：`EXP-20260612-022`
- 结果：
  - `G0`: `compute_reward=0.60` (`12/20`)
  - `G1`: `compute_reward=0.50` (`10/20`)
  - `G4`: `compute_reward=0.50` (`10/20`)
  - `G6`: `compute_reward=0.50` (`10/20`)
  - `G7`: `compute_reward=0.50` (`10/20`)
- 稳定性 / debug 观察：
  - 所有 enabled groups 都产生非空 `answer.json`
  - 所有 enabled groups 都写入 `20` 条 predictions 和 `1` 条 summary
  - 没有 crash、NaN、OOM、CUDA error 或 shape/device/dtype mismatch
  - 所有 enabled groups 中，每个 session 的 `initial_slots=0`
  - 没有观察到 cross-sample leakage
  - retrieved memory 在所有 enabled groups 中仍然是 Reasoner-only，因为 `weaver_input_token_counts` 匹配 `reasoner_to_weaver_input_token_counts`
  - stored latent memories 仍然是 reasoner-space `[8, 1536]` tensors
  - `slot_count` 从未超过 `4`，所以本 pilot 中 `max_slots=8` 没有饱和
- 解释：
  - 该 pilot 不支持任何 performance claim
  - 在这个 20-sample slice 上，所有 enabled variants 都以相同 observed margin 低于 disabled anchor
  - 在本 pilot 内，去掉当前 write-age decay 或在当前实现的 update-policy settings 间切换，并没有改变 `compute_reward`
  - update-policy behavior 没有被有效区分，因为 `max_slots=8` 没有饱和，且 `replace_count=0`
  - 当前 `threshold_topk` 没有 fallback top-1
  - 当前 decay 是 write-age decay，而不是 last-retrieved-turn decay
- 下一步建议：
  - 不要直接把 GSM8K 扩展为 primary main experiment
  - 在 enabled-memory runs 之前，先规划可信的 TriviaQA disabled baseline
  - 在 target-task stability 之后，先评估 method-aligned Version A variants，再考虑 Version B
- Gate：
  - 不要把 Phase 8A 当成 paper-level result
  - Phase 8B 尚未开始
  - Phase 9 尚未开始

## 2026-06-12 - Step 2 结构化检索上下文

- 状态：`completed`
- 添加 immutable `LatentMemoryRetrievalResult`。
- 添加 `retrieve_with_context(...)`，包含：
  - 按原始 slot-index 顺序排列的 full-bank scores
  - pre-filter maximum score 和 argmax index
  - threshold-pass status
  - filtered retrieved indices and scores
  - 当前 memory-write bank step
- 保留 `retrieve(...)` 作为 legacy slot-list API。
- 保留当前 write-age scoring、threshold-without-fallback 行为、detached retrieval copies 和所有已有 write/update policies。
- 没有修改 `MemGenModel.generate()`。
- 没有实现 matched-thread write-back、fallback top-1 或 last-retrieved decay。
- Step 3 仍需要明确批准。

## 2026-06-12 - Step 3 线程感知写回

- 状态：`completed`
- 添加 `update_policy=thread_update`。
- 添加 `write_back(memory, retrieval_result, metadata=None)`。
- 实现：
  - empty bank -> insert
  - high current-query score -> replace current argmax slot
  - low score with capacity -> insert new thread
  - low score at capacity -> evict oldest and insert new thread
- 添加 stale retrieval-step 和 matched-index validation。
- 添加独立 debug counts 和 event traces，用于 thread insertion、matched replacement 和 capacity eviction。
- 只把 `thread_update` policy 与 `retrieve_with_context(...)` 和 `write_back(...)` 集成进 generation。
- 保留 Reasoner-only retrieved-memory injection 和不变的 Weaver inputs。
- 保留 legacy update policies、no-fallback threshold retrieval、write-age decay 和 disabled path。
- 验证：
  - 修改后的 model 和 test files 通过 `py_compile`
  - full unit discovery 通过 `47/47`
  - `git diff --check` 通过
  - disabled golden replay `EXP-20260612-023-step3-disabled-replay` 在所有三个 response-token hashes、所有三个 augmentation-mask hashes、Trigger calls (`193`)、Weaver prompt calls (`3`) 和 Weaver inference calls (`8`) 上与 `EXP-20260611-007` 匹配
  - disabled sessions 没有创建 memory bank，也没有暴露 memory debug state
- 没有进入 Version B。
- Step 4 smoke 仍需要明确批准。

## 2026-06-12 - Step 4 Thread-Update 机制验证

- 状态：`completed`
- 实验：`EXP-20260612-024`
- 输出：
  `outputs/latent_bank_vA/EXP-20260612-024-thread-update-smoke/`
- 真实 enabled inference：
  - 一个 GSM8K test sample 完成
  - answer file 非空，包含一条 prediction 和一条 summary
  - 没有 crash、NaN、OOM、CUDA、shape、device 或 dtype error
  - `memory_write_count=4`
  - `memory_retrieve_count=3`
  - `thread_insert_count=1`
  - `matched_replace_count=3`
  - `capacity_evict_count=0`
  - 观察到的 update reasons：一个 `empty_bank`，三个 `matched_thread`
- 边界：
  - Weaver input counts 与 reasoner-to-Weaver input counts 完全匹配
  - retrieved memory 仍然是 Reasoner-only
  - stored latent shape 仍然是 `[8, 1536]`
  - session 从 `initial_slots=0` 开始
- 受控分支证据：
  - 四个 targeted tests 通过，分别覆盖 empty insert、low-score new-thread insert、high-score matched replacement 和 full-bank oldest eviction
  - full test discovery 通过 `47/47`
  - `git diff --check` 通过
- 范围：
  - Step 4 不需要代码修改
  - 因为 Step 4 中 core 或 generate logic 没有改变，所以不需要重新运行 disabled-path
  - 这是机制验证，不是 performance result
  - 没有 fallback top-1 或 last-retrieved decay
  - Version B 尚未开始
- 建议：
  - 在添加更多 method variants 前，回到 TriviaQA baseline planning

## 2026-06-12 - Phase 8C-alt 受控多轮机制评估

- 状态：`completed_with_negative_smoke`
- 范围：
  - 添加一个 harness-only deterministic three-turn evaluation
  - Turn 3 从 system instruction 和当前 query 重建 visible prompt，不包含 Turn 1 和 Turn 2 history
  - 这是机制研究，不能替代 TriviaQA
- 实现：
  - 添加 `scripts/eval/phase8c_controlled_memory.py`
  - 添加 `tests/test_controlled_multiturn_memory.py`
  - 没有修改 core model、Weaver、Trigger、runner、interaction managers、trainers、training scripts 或 baseline configuration
- 验证：
  - `py_compile` 通过
  - full unit discovery 通过 `56/56`
  - `git diff --check` 通过
- Smoke experiments：
  - `EXP-20260612-025`：失败，因为第一个 harness revision 在 FlashAttention 前没有把 model 移到 CUDA；仅在 harness 中修复
  - `EXP-20260612-026`：G0 disabled 完成一个 three-turn episode，通过所有 leakage checks，没有创建 bank，得分 `0/1`
  - `EXP-20260612-027`：G2 `thread_update` 完成一个 three-turn episode，通过所有 leakage checks，在 turns 之间保持一个 bank，得分 `0/1`
- G2 机制证据：
  - 各 turn 后 slot counts：`[1, 2, 3]`
  - `memory_write_count=12`
  - `memory_retrieve_count=11`
  - `retrieved_latent_count=72`
  - `new_latent_count=96`
  - `thread_insert_count=3`
  - `matched_replace_count=9`
  - `capacity_evict_count=0`
  - stored hidden sizes 仍然是 `1536`
  - Weaver input counts 匹配 reasoner-to-Weaver input counts
- 解释：
  - 受控 session lifecycle 和 Version A-aligned mechanism 可以跨三次独立 prompt calls 运行
  - G0 和 G2 都没有在这个 one-episode smoke 中产生 tagged exact answer
  - negative smoke 不说明方法失败，因为 harness 使用的是 GSM8K-trained checkpoint，任务是 out-of-distribution synthetic task
  - 没有引入 fallback top-1、last-retrieved decay 或 Version B

## 2026-06-13 - G3 Oracle-Visible One-Episode Smoke

- 状态：`completed_with_protocol_failure`
- 实验：`EXP-20260613-001`
- 输出：
  `outputs/controlled_memory/EXP-20260613-001-controlled-g3-oracle-visible/`
- 配置：
  - group `G3_oracle_visible`
  - 一个 deterministic exact-code episode
  - `seed=42`、`batch_size=1`、greedy decoding
  - `max_response_length=64`
  - memory disabled 且 `oracle_visible=true`
- Oracle prompt 检查：
  - Turn 3 显式包含 early fact 和 gold value `770487`
  - Turn 3 prompt length 为 `90` tokens
  - oracle-visible content 符合预期，不视为 leakage
- 结果：
  - raw Turn 3 response 为
    `The access code for Project Lumen is 770487.`
  - response 包含正确 gold value，但缺少必需的 `<answer>...</answer>` tags
  - strict parser output 为 `null`，所以 exact match 是 `0/1`
  - episode 在结构上有效，且无 runtime error
- 解释：
  - checkpoint 能读取 visible oracle fact 并产生正确答案内容
  - 当前 tagged-output protocol 不能被该 checkpoint 可靠遵守，因此 G0/G2 strict exact-match failures 受到 prompt/parser contract 混淆
  - G3 是 oracle visible-context control，不是 memory-method result，也不能与 G0/G2 公平比较
  - 该 controlled study 不能替代 TriviaQA
  - 没有修改 harness、core model、Weaver、Trigger、trainer 或 training-script code
  - 没有引入 fallback top-1、last-retrieved decay 或 Version B
- 建议：
  - 在运行 G1 或任何更大 controlled pilot 前，先审计 prompt/parser scoring contract

## 2026-06-13 - Controlled Parser Calibration

- 状态：`implemented`
- 范围：
  - 只修改 controlled harness、其 tests 和 research notes
  - 没有运行 G0、G1、G2、G3 或 small pilot
  - 没有修改 core model、Weaver、Trigger、runner、interactions、trainers 或 training scripts
- Prompt contract：
  - 所有 groups 现在接收相同 final instruction：
    `Return exactly one line: <answer>VALUE</answer>. Do not include any other text.`
  - G0/G1/G2 的 Turn 3 仍然排除 early fact 和 gold value
  - G3 仍然包含 early fact 和 gold value，作为 oracle-visible positive control
- Scoring contract：
  - `strict_exact_match` 只使用最后一个完整的 `<answer>...</answer>` span
  - `relaxed_exact_match` 在可用时复用 strict candidate
  - exact-code fallback 只接受唯一一个 standalone six-digit candidate
  - zero candidates 产生 `none`；multiple candidates 产生 `ambiguous`
  - semantic-relation fallback 只在 strip outer quotes 和一个 terminal punctuation mark 后，normalize 完整短 response
  - 不允许 LLM judge、gold substring search、gold-guided candidate selection 或 fuzzy semantic matching
  - legacy `exact_match` 只作为 deprecated alias 保留，指向 `strict_exact_match`
- Artifact contract：
  - turn 和 episode records 现在包含 strict 和 relaxed parsed answers、parser success flags、parser mode，以及两种 exact-match metrics
  - summary 和 verification records 现在包含 strict/relaxed counts and rates，以及 parser-success counts
- Evidence reclassification：
  - `EXP-20260612-026`、`EXP-20260612-027` 和 `EXP-20260613-001` 是 pre-parser-calibration smoke runs
  - 它们仍然可用于 runtime、leakage、bank-lifecycle 和 boundary evidence，但不是 calibrated comparison results
- 验证：
  - targeted controlled-harness tests 通过 `22/22`
  - harness 和 controlled test module 通过 `py_compile`
  - full unit discovery 通过 `69/69`
  - `git diff --check` 通过
- 范围边界：
  - controlled evaluation 仍是机制研究，不能替代 TriviaQA
  - 没有引入 fallback top-1、last-retrieved decay 或 Version B

## 2026-06-13 - Calibrated G0/G2/G3 One-Episode Smokes

- 状态：`completed`
- 运行前验证：
  - full unit discovery 通过 `69/69`
  - `git diff --check` 通过
  - protected core、Weaver、Trigger、runner、interaction、trainer 或 training-script 没有 diff
- 共享协议：
  - frozen calibrated prompt 和 dual strict/relaxed scoring
  - 一个 deterministic exact-code episode
  - `seed=42`、`batch_size=1`、greedy decoding
  - `max_response_length=64`
- `EXP-20260613-002` calibrated G0：
  - Turn 3 排除 early fact 和 gold value
  - 没有创建 bank
  - raw response 包含唯一错误 code `123456`
  - strict exact match `0/1`；relaxed exact match `0/1`
- `EXP-20260613-003` calibrated G2：
  - 一个 bank 在所有三个 turns 中持续存在
  - turn 后 slot trace 为 `[1, 2, 3]`
  - 12 writes、11 retrievals、3 thread inserts、9 matched replacements
  - retrieved memory 仍然是 Reasoner-only
  - stored latent hidden sizes 仍然是 `1536`
  - raw response 包含唯一错误 code `123456`
  - strict exact match `0/1`；relaxed exact match `0/1`
- `EXP-20260613-004` calibrated G3：
  - Turn 3 包含 oracle-visible early fact 和 gold value `770487`
  - raw response 是
    `The access code for Project Lumen is 770487.`
  - strict parser 因为没有 tags 而失败
  - relaxed parser 提取唯一 code `770487`
  - strict exact match `0/1`；relaxed exact match `1/1`
- 解释：
  - calibrated parser 能按预期区分 format compliance 和 deterministic answer correctness
  - G3 验证 oracle-visible prompt 和 relaxed exact-code extraction
  - 在 relaxed exact match 下，G0 和 G2 都没有在这个 one-episode smoke 中恢复 hidden fact
  - 这只是机制级证据，不是 performance conclusion
  - controlled evaluation 不能替代 TriviaQA
  - 没有引入 fallback top-1、last-retrieved decay 或 Version B

## 2026-06-13 - Calibrated G1 Version A-Simple One-Episode Smoke

- 状态：`completed`
- 实验：`EXP-20260613-005`
- 输出：
  `outputs/controlled_memory/EXP-20260613-005-calibrated-g1-vA-simple/`
- 配置：
  - group `G1_vA_simple`
  - memory mode `vA_simple`
  - 一个 deterministic exact-code episode
  - `seed=42`、`batch_size=1`、greedy decoding
  - `max_response_length=64`
  - frozen calibrated prompt 和 dual strict/relaxed scoring
- 结果：
  - valid episodes `1/1`
  - Turn 3 排除 early fact 和 gold value
  - raw response 包含唯一错误 code `123456`
  - strict parser 返回 `null`
  - relaxed parser 提取 `123456`
  - strict exact match `0/1`；relaxed exact match `0/1`
- Memory behavior：
  - 一个 bank 在所有三个 turns 中持续存在
  - slot trace 为 `[4, 8, 8]`；final slot count 为 `8`
  - `memory_write_count=12`
  - `memory_retrieve_count=11`
  - legacy `replace_oldest` update path 仍然活跃
  - 未使用 `thread_update`
  - retrieved memory 仍然是 Reasoner-only
  - Weaver input counts 匹配 reasoner-to-Weaver input counts
  - stored latent hidden sizes 仍然是 8 个 `1536` 维 tensors
- Runtime：
  - Trigger calls `132`
  - Weaver prompt calls `3`
  - Weaver inference calls `9`
  - latency `6.162 s`
  - 没有 crash、non-finite metric、OOM、CUDA、shape、dtype 或 device error
- 解释：
  - calibrated harness 可以执行 legacy Version A-simple path
  - 这只是 one-episode mechanism smoke，不是 performance conclusion
  - controlled evaluation 不能替代 TriviaQA
  - 没有引入 fallback top-1、last-retrieved decay 或 Version B

## Phase 8C-alt 收尾与 TriviaQA 重启计划

### Controlled Study 目的

- Phase 8C-alt 是机制研究。
- 它在 TriviaQA infrastructure 不可用时，以低成本验证 memory lifecycle、boundary behavior、parser behavior 和 artifact generation。

### 为什么需要这个研究

- TriviaQA checkpoint 不可用。
- TriviaQA / AgentBank data caches 不可用或未验证。
- Retrieval service 和 search index 不可用。
- 尽管存在这些 blocker，Version A-simple 和 Version A-aligned `thread_update` 仍需要一个有限的 cross-turn runtime check。

### 它验证了什么

- controlled harness 可以端到端运行
- calibrated strict / relaxed parser behavior 可以工作
- G1 legacy Version A-simple path 可以运行
- G2 Version A-aligned `thread_update` path 可以运行
- G2 保持 Reasoner-only injection
- stored latent hidden size 仍然是 `1536`
- G3 oracle-visible positive control 达到 relaxed exact match `1/1`

### 它没有验证什么

- 没有 target-task performance claim
- 没有 general memory benefit claim
- 没有 TriviaQA result
- 没有 Version B result
- 没有 fallback top-1
- 没有 last-retrieved decay

### G0/G1/G2/G3 收尾

| Group | 含义 | Strict EM | Relaxed EM | 主要结果 |
|---|---|---:|---:|---|
| G0 | disabled | 0/1 | 0/1 | 错误唯一 code；无 bank |
| G1 | Version A-simple | 0/1 | 0/1 | 错误唯一 code；legacy path 可运行 |
| G2 | Version A-aligned thread_update | 0/1 | 0/1 | 错误唯一 code；thread-aware path 可运行 |
| G3 | oracle-visible control | 0/1 | 1/1 | relaxed parser 恢复正确未加标签 code |

### G1 vs G2 Memory Behavior

- G1 使用 `update_policy=replace_oldest`。
- G1 slot trace 是 `[4, 8, 8]`。
- G1 填满 capacity 后使用 legacy replacement。
- G2 使用 `update_policy=thread_update`。
- G2 slot trace 是 `[1, 2, 3]`。
- G2 thread inserts 是 `3`。
- G2 matched replacements 是 `9`。
- 两者都保持 Reasoner-only injection 和 `1536` hidden-size storage。

### 当前解释

- G0/G1/G2 得分 `0/1` 并不说明方法无效，因为这是单个 synthetic deterministic episode，而且 checkpoint 对该任务是 out-of-distribution。
- G3 不是 memory result，因为 Turn 3 显式包含 early fact 和 gold value。
- Controlled study results 不能替代 TriviaQA。
- 除非之后明确需要更多 synthetic mechanism evidence，否则现在不建议做 small pilot。

### 当前研究状态

- Version A-simple：已实现且可运行；mechanism smoke 完成；没有 positive task evidence。
- Version A-aligned thread_update：已实现且可运行；boundaries 已验证；没有 positive task evidence。
- Controlled mechanism study：完成。
- TriviaQA：仍被 infrastructure 层面阻塞。
- Version B：未开始。

### Phase 8D TriviaQA 重启 Checklist

- 验证官方 TriviaQA checkpoint
- 验证 `mandarjoshi/trivia_qa` cache
- 验证 `Solaris99/AgentBank/triviaqa` cache
- 验证 `127.0.0.1:8001/retrieve`
- 验证 Search-R1 / Wikipedia index
- 防止 silent `Cannot find corresponding pages` fallback
- 设计 dynamic single-sample harness，带 structured `answer.json`
- 只有 disabled baseline 稳定后，才运行 Version A-aligned enabled smoke

### 最终收尾决定

- Phase 8C-alt 作为 mechanism-study node 关闭。
- 下一条主线是 Phase 8D TriviaQA infrastructure。
- 不要进入 Version B。

## 2026-06-13 - Phase 8D-0 TriviaQA Infrastructure Asset Discovery

- 范围：
  - 只做 read-only asset discovery
  - 不下载、不启动 retrieval service、不运行正式实验
  - 不改代码、不做 Version B、不做 fallback top-1、不做 last-retrieved decay
- 仓库状态：
  - branch `rlm-memory-bank`
  - `git status` 干净
  - Phase 8C-alt 在开始 asset discovery 前保持关闭
- Config / script 审计：
  - `configs/latent_memory/triviaqa.yaml` 设置 base model 为 `Qwen/Qwen2.5-1.5B-Instruct`，但 `model.load_model_path: null`
  - `scripts/eval/qwen2_5_triviaqa.sh` 指向预期 TriviaQA checkpoint 路径：
    `MemGen/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8`
  - `data/triviaqa/builder.py` 同时需要 `mandarjoshi/trivia_qa` 和 `Solaris99/AgentBank`，并使用 `triviaqa` config path
  - `data/utils/retrieval_utils.py` 仍指向 `http://127.0.0.1:8001/retrieve`
  - `memgen/runner.py` dynamic evaluation 通过 `DynamicEvalRecorder` 写入 `conversations.txt`，不会生成 structured `answer.json`
  - 官方 dynamic path 暴露 batch-size configuration，但没有识别到可信的 `sample_count=1` 路径
- Checkpoint discovery：
  - 在 repository、`.cache/`、`outputs/`、`checkpoints/` 或 HuggingFace hub cache 下都没有找到本地 TriviaQA checkpoint snapshot
  - 唯一找到的完整本地 MemGen checkpoint 是 `.cache/baselines/memgen-gsm8k-sft/` 下的 GSM8K SFT checkpoint
  - GSM8K assets 不能被当作 TriviaQA substitute
  - TriviaQA checkpoint status 为 `missing`
- Dataset discovery：
  - 本地没有找到 cached `mandarjoshi/trivia_qa` dataset
  - 本地没有找到 cached `Solaris99/AgentBank` dataset
  - 两个 datasets 的 offline metadata / `[:1]` load attempts 都失败，因为没有 local cache
  - TriviaQA 和 AgentBank 的 dataset status 都是 `missing`
- Retrieval / index discovery：
  - 在 read-only audit 期间，没有正面证据表明存在 live `127.0.0.1:8001` retrieval service
  - 没有找到可信的本地 Search-R1 / Wikipedia retrieval index asset set
  - `data/triviaqa/env.py` 仍允许 retrieval failure 表现为 `Cannot find corresponding pages.`，所以 silent degraded runs 仍然是风险
  - retrieval / search asset status 在真正 staging 并验证 service 和 index 前，实际上是 `missing`
- Dynamic harness status：
  - 官方 dynamic TriviaQA path 存在，但对目标 smoke protocol 来说不完整
  - 当前缺口包括：缺少 structured `answer.json`、没有可信的 `sample_count=1` 控制、没有显式 retrieval success / failure accounting
- Phase 8D-0 结论：
  - TriviaQA disabled `1`-sample smoke 仍然是 `NO-GO`
  - 立即 blocker 是：
    - 缺少官方 TriviaQA checkpoint
    - 缺少 `mandarjoshi/trivia_qa` cache
    - 缺少 `Solaris99/AgentBank` cache
    - 缺少或未验证 retrieval service
    - 缺少或未验证 Search-R1 / Wikipedia index assets
    - dynamic harness 不完整，无法进行 single-sample structured recording
  - 下一个工作项是 Phase 8D infrastructure acquisition / verification，而不是 Version B

## 2026-06-16 - Phase R2 Version A-aligned Last-Retrieved Decay Revision

- 状态：`completed`
- 范围：
  - 修改 `memgen/model/latent_memory_bank.py`
  - 更新 `tests/test_latent_memory_bank.py`
  - 更新 method / decision / progress notes
  - 没有运行正式实验
  - 没有进入 Version B
- 机制修订：
  - 新增 enabled retrieval-turn counter
  - Version A-aligned score 从 write-age decay 改为 last-retrieved decay
  - `last_retrieved_age = current_retrieval_step - slot.last_retrieved_step`
  - 只有最终 selected / returned slots 更新 `last_retrieved_step`
  - below-threshold 和 top-k 未选中的 slots 不更新 `last_retrieved_step`
  - 新插入 slot 和 matched replacement slot 初始化为当前 retrieval step
  - full-bank `new_thread` eviction 从 oldest-created 改为最大
    `last_retrieved_age`
  - eviction tie-break 为 earlier `created_step`，再 lower slot index
- Debug / trace：
  - `debug_summary()` 增加 `retrieval_step`
  - slot debug 增加 `last_retrieved_step` 和 `last_retrieved_age`
  - `write_back_trace` 增加 `retrieval_step`、`eviction_basis` 和
    `evicted_slot_last_retrieved_age`
  - 保留 `last_access_step` 作为兼容字段，语义等同
    `last_retrieved_step`
- 边界：
  - retrieved memory 仍然只进入 Reasoner
  - retrieved memory 仍然不进入 Weaver
  - 没有 fallback top-1
  - disabled path 语义不变
  - enabled memory 仍限制 batch size 1
  - 没有修改 Weaver、Trigger、trainer 或 training scripts
  - Version B 未开始
- 解释：
  - Phase 8A 和 Phase 8C-alt 的既有结果仍属于历史 write-age decay 版本
  - 本阶段没有产生 target-task performance claim

## 2026-06-16 - Phase R2-fix Minor Follow-Up

- 状态：`completed`
- 范围：
  - 只修复 R2 的两个 minor review issues
  - 没有修改核心方法边界
  - 没有运行正式实验
- 修复内容：
  - 添加 `retrieve()` vs `retrieve_with_context()` 的 retrieval-step 递增测试
  - 明确 `write_back()` 创建 replacement / inserted slot 时使用
    `retrieval_result.retrieval_step`
  - 防止在额外 retrieval 发生后出现 `last_retrieved_step` 语义漂移
- 验证：
  - `python -m unittest discover -s tests -v` 通过 `76/76`
  - `git diff --check` 通过
- 边界：
  - 没有引入 fallback top-1
  - 没有让 retrieved memory 进入 Weaver
  - 没有进入 Version B

## 2026-06-16 - Phase R4-1A TriviaQA Environment Configuration Preflight

- 状态：`completed`
- 性质：
  - environment preflight only
  - 不是实验
  - 没有运行 TriviaQA disabled baseline
  - 没有运行 Version A-aligned enabled smoke
  - 没有产生 TriviaQA result
  - 没有 target-task performance claim
- 仓库状态：
  - branch `rlm-memory-bank`
  - preflight HEAD `622d385`
  - `git status` clean
  - R4-0A plan 已提交：
    `622d385 docs: record TriviaQA-first evaluation plan`
- Config / script 结论：
  - `configs/latent_memory/triviaqa.yaml` 使用 base model
    `Qwen/Qwen2.5-1.5B-Instruct`
  - YAML 中 `model.load_model_path: null`
  - `scripts/eval/qwen2_5_triviaqa.sh` 预期 checkpoint：
    `MemGen/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8`
  - 官方 dynamic path 写 `evaluate/conversations.txt`
  - 官方 dynamic path 没有可信 structured `answer.json`
  - 官方 dynamic path 没有可信 `sample_count=1` dynamic harness
- Local assets:
  - base model `Qwen/Qwen2.5-1.5B-Instruct` exists in local HF cache
  - TriviaQA checkpoint missing
  - `mandarjoshi/trivia_qa` cache missing
  - `Solaris99/AgentBank` triviaqa cache missing
  - Search-R1 / Wikipedia index assets not found / uncertain
- Retrieval preflight:
  - MemGen TriviaQA dynamic env does not implement a full retriever internally
  - MemGen calls local endpoint `http://127.0.0.1:8001/retrieve`
  - no listener was observed on port `8001`
  - curl to `/retrieve` failed to connect
  - retrieval endpoint is unavailable
  - retrieval failure can silently degrade into
    `Cannot find corresponding pages.`
- Retrieval-service route understanding:
  - original project points users to Search-R1 for retriever environment setup
  - intended formal route is:
    MemGen dynamic env -> local `/retrieve` endpoint -> Search-R1
    `retrieval_server.py` -> FAISS index + Wikipedia corpus
  - Search-R1 launch is expected to use `e5_Flat.index`, `wiki-18.jsonl`,
    `retriever_name=e5`, `retriever_model=intfloat/e5-base-v2`, and optionally
    `--faiss_gpu`
  - a toy retrieval server may be useful only for engineering smoke / harness
    debugging and cannot support formal TriviaQA results
- Current go/no-go:
  - No-Go for TriviaQA disabled baseline
  - No-Go for Version A-aligned enabled smoke
  - blockers are infrastructure / environment blockers, not Version A-aligned
    mechanism failures
  - no performance claim is supported

## 2026-06-16 - Phase R4-1Plan TriviaQA Infra Preflight and Detailed Execution Plan

- 状态：`completed`
- 范围：
  - record R4-1A preflight results in notes
  - record current TriviaQA infrastructure blockers
  - record Search-R1-compatible retrieval-service decision
  - plan R4-1B through R4-1F
  - no model code changes
  - no tests changes
  - no dataset or checkpoint download
  - no retrieval service startup
  - no dynamic harness implementation
  - no toy retrieval server implementation
- Next planned phases:
  - R4-1B: Dataset and Checkpoint Acquisition / Cache
  - R4-1C: Retrieval Service / Index Configuration
  - R4-1D: Dynamic Single-Sample Structured Harness
  - R4-1E: Disabled Baseline Smoke
  - R4-1F: Version A-aligned Enabled Smoke
- Boundary:
  - R4-1B through R4-1F are environment / harness / smoke phases, not main
    result phases
  - no TriviaQA result exists yet
  - no target-task performance gain claim exists yet
  - controlled diagnostic subset remains separate from this TriviaQA infra plan
  - Version B remains deferred
  - no fallback top-1
  - retrieved memory does not enter Weaver

## 2026-06-17 - Phase R4-1C Retrieval Service / Index Configuration Check

- 状态：`completed`
- 性质：
  - retrieval environment / configuration check only
  - no retrieval service was started
  - no large index or corpus was downloaded
  - no formal experiment was run
  - no TriviaQA disabled baseline was run
  - no Version A-aligned enabled smoke was run
  - no target-task performance claim was made
- R4-1B carry-forward:
  - base model `Qwen/Qwen2.5-1.5B-Instruct` ready
  - `mandarjoshi/trivia_qa`, config `rc.wikipedia.nocontext`, split
    `validation` ready and offline verified
  - `Solaris99/AgentBank`, config `triviaqa`, split `train` ready and offline
    verified
  - TriviaQA checkpoint ready at:
    `/home/baishilong/.cache/huggingface/hub/models--Kana-s--MemGen/snapshots/269d9b1741130b94fffa410cdaa3d4bc74081a7f/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model`
- MemGen retrieval client contract:
  - endpoint: `http://127.0.0.1:8001/retrieve`
  - request payload:
    `{"queries": [...], "topk": 3, "return_scores": true}`
  - expected response shape:
    `{"result": [[{"document": {"contents": "Title\nBody"}, "score": ...}]]}`
  - `data/triviaqa/env.py` catches retrieval exceptions and returns
    `Cannot find corresponding pages.`
  - endpoint and top-k are currently hard-coded in
    `data/utils/retrieval_utils.py`
- R4-1C findings:
  - Search-R1 repo / server not found locally
  - `search_r1/search/retrieval_server.py` not found locally
  - `retrieval_launch.sh` not found locally
  - no `searchr1` / retriever conda env found
  - `faiss` missing from the validated `memgen` environment
  - `pyserini` missing from the validated `memgen` environment
  - `e5_Flat.index` not found
  - `wiki-18.jsonl` not found
  - local `intfloat/e5-base-v2` cache not found
  - `127.0.0.1:8001` endpoint not running
  - upstream Search-R1 `/retrieve` schema appears compatible with MemGen's
    payload / response expectations, but upstream default port is `8000` while
    MemGen expects `8001`
- Current blocker:
  - formal TriviaQA retrieval remains blocked until a Search-R1-compatible
    retrieval service, Wikipedia corpus, FAISS index, and retriever model are
    available and verified
  - silent fallback risk remains because retrieval failures can become
    `Cannot find corresponding pages.`
- Decision / transition:
  - continue formal Search-R1 setup later
  - proceed to R4-1D dynamic single-sample structured harness design because
    the harness is needed regardless of retrieval-service readiness
  - any toy retrieval server remains smoke-only and must not be used for a
    formal TriviaQA result or performance claim
- Boundary:
  - no Version B
  - no fallback top-1
  - retrieved memory does not enter Weaver
  - memory-bank mechanism unchanged
  - no commit or push performed

## 2026-06-17 - Phase R4-1D Dynamic Single-Sample Structured Harness

- 状态：`implemented`
- 性质：
  - harness implementation only
  - no formal TriviaQA experiment was run
  - no TriviaQA disabled baseline was run
  - no Version A-aligned enabled smoke was run
  - no retrieval service was started
  - no dataset, checkpoint, Search-R1 repo, index, or corpus was downloaded
  - no target-task performance claim was made
- Implemented:
  - added `scripts/eval/r4_triviaqa_dynamic_harness.py`
  - added `tests/test_r4_triviaqa_dynamic_harness.py`
  - harness enforces `sample_count=1` and `batch_size=1`
  - harness supports `memory_mode=disabled` and
    `memory_mode=version_a_aligned`
  - harness writes structured `evaluate/answer.json`
  - harness preserves `evaluate/conversations.txt`
  - harness writes `summary.json`, `run_config.json`, and
    `memory_trace.json`
  - harness records retrieval endpoint, top-k, call count, success count,
    failure count, exception details, and
    `Cannot find corresponding pages.` detection
  - harness records top-level `valid_run` and `invalid_reason`
  - strict parser extracts `<answer>...</answer>` and does not use gold-guided
    fuzzy matching or an LLM judge
  - enabled runs preserve `memory_bank_debug` when available; missing memory
    trace is recorded as `null`
- Validation:
  - target unit tests passed with standard-library `unittest`
  - preflight-only CLI artifact write completed under `/tmp` without loading a
    model, dataset, checkpoint, or retrieval service
  - `py_compile` passed for the new harness and tests
- Current blocker:
  - formal TriviaQA retrieval remains blocked until Search-R1-compatible
    retrieval service, Wikipedia corpus, FAISS index, and retriever model are
    available and verified
- Boundary:
  - no Version B
  - no fallback top-1
  - retrieved memory does not enter Weaver
  - memory-bank mechanism unchanged
  - no training code changed
  - no commit or push performed

## 2026-06-18 - Phase R4 Search-R1 / TriviaQA Infrastructure Validation

- 状态：`completed_with_caveats`
- 性质：
  - retrieval service / harness / smoke validation only
  - not a full benchmark
  - not a formal TriviaQA performance result
  - no Version B work
  - no source code, training code, or main project config was modified
- Search-R1 service:
  - repo: `/mnt/18T/baishilong/Search-R1`
  - endpoint: `http://127.0.0.1:8000/retrieve`
  - Search-R1 hard-codes Uvicorn port `8000`
  - MemGen harness used `--retrieval-endpoint
    http://127.0.0.1:8000/retrieve` rather than patching Search-R1 or using
    the MemGen default `8001`
  - `/retrieve` schema test passed
  - compatible response shape:
    `{"result":[[{"document":{"contents":"Title\nBody"},"score":...}]]}`
- Retrieval assets:
  - E5 model:
    `/mnt/18T/baishilong/retrieval_assets/e5-base-v2`
  - E5 was verified separately with `AutoTokenizer` and `AutoModel`;
    hidden size `768`
  - corpus:
    `/mnt/18T/baishilong/retrieval_assets/wiki-18/wiki-18.jsonl`
    valid JSONL, about `14G`
  - compressed corpus:
    `/mnt/18T/baishilong/retrieval_assets/wiki-18/wiki-18.jsonl.gz`
  - FAISS index:
    `/mnt/18T/baishilong/retrieval_assets/wiki-18/e5_Flat.index`
    about `61G`
  - index split files:
    `/mnt/18T/baishilong/retrieval_assets/wiki-18-index/part_aa`
    and
    `/mnt/18T/baishilong/retrieval_assets/wiki-18-index/part_ab`
  - caveat: the corpus `.gz` was actually a gzip-compressed tar payload; the
    correct extraction route was `tar -xOzf`, not plain `gzip -dc`
- Search-R1 GPU / port bring-up:
  - port `8000` was initially occupied by a user-owned temporary
    `python3 -m http.server 8000 --bind 0.0.0.0`; it was killed after
    verification
  - all-visible-GPU FAISS loading failed initially because GPU `6` was nearly
    full
  - `CUDA_VISIBLE_DEVICES=7` failed because one 49GB A6000 could not hold the
    about-61G index
  - successful service launch used:
    `CUDA_VISIBLE_DEVICES=0,2,3,4,7`
- Disabled-memory dynamic smoke:
  - output:
    `outputs/r4_triviaqa_dynamic_smoke_disabled_1sample/`
  - exit code `0`
  - `evaluate/answer.json` written as JSONL-style records
  - `retrieval_call_count=1`
  - `retrieval_success_count=1`
  - `retrieval_failure_count=0`
  - `saw_cannot_find_pages=False`
  - `valid_run=True`
  - `invalid_reason=None`
  - `memory_mode=disabled`
  - `memory_enabled=False`
  - Claude read-only review: `PASS`
  - interpretation: infrastructure smoke only, not a performance result
- Version A-aligned dynamic smoke:
  - output:
    `outputs/r4_triviaqa_dynamic_smoke_version_a_1sample/`
  - exit code `0`
  - `memory_mode=version_a_aligned`
  - `memory_enabled=True`
  - `memory_write_count=2`
  - `memory_retrieve_count=1`
  - `retrieved_latent_count=0`
  - `new_latent_count=16`
  - `slot_count=2`
  - default `threshold=0.7`
  - `max_score` about `0.044`
  - `threshold_passed=False`
  - `retrieval_call_count=1`
  - `retrieval_success_count=1`
  - `valid_run=True`
  - Claude read-only review: `PASS`
  - interpretation: Version A enabled path and memory write path validated,
    but non-empty retrieved-memory path was not triggered at the default
    threshold on sample `0`
- Retrieval-positive diagnostic:
  - output:
    `outputs/r4_triviaqa_dynamic_diagnostic_version_a_threshold001_1sample/`
  - diagnostic-only threshold override:
    `threshold 0.7 -> 0.01`
  - original source and original config were not modified
  - copied YAML under `outputs/` was provenance only; the effective threshold
    override was an in-memory harness override
  - exit code `0`
  - `memory_enabled=True`
  - `memory_write_count=2`
  - `memory_retrieve_count=1`
  - `retrieved_latent_count=8`
  - `new_latent_count=16`
  - `slot_count=1`
  - `threshold=0.01`
  - `max_score=0.04365994428860699`
  - `threshold_passed=True`
  - `update_action_trace=["insert", "replace_matched"]`
  - `retrieved_indices=[0]`
  - Claude read-only review: `PASS`
  - interpretation: confirms the non-empty retrieved-memory path can be
    exercised under a controlled diagnostic threshold; it is not a performance
    result and not default Version A behavior
- Caveats:
  - duplicate system prompt appears in disabled, Version A, and diagnostic
    records; this appears to be a conversation construction artifact and did
    not block smoke validation
  - `answer.json` is JSONL-style and must be read line by line, not with
    `json.load`
  - artifacts do not directly assert Reasoner-only injection; they record
    memory retrieval / memory-bank behavior, while the latent injection remains
    consistent with the Version A Reasoner-only design
  - default threshold `0.7` did not trigger `retrieved_latent_count > 0` on
    sample `0`
  - threshold `0.01` is diagnostic-only and must not be used as a formal
    performance setting without a separate decision
- Scoring audit caveat:
  - config comment says threshold is a "cosine similarity threshold", but
    implementation compares the threshold against decayed retrieval score,
    not raw cosine similarity → terminology / comment caveat only

### R4: LatentMemoryBank Scoring / Recency Semantics Audit

- Read-only audit confirmed the active LatentMemoryBank retrieval path
  matches the intended last-retrieved-age design
- Core implementation: `memgen/model/latent_memory_bank.py`
- Score formula: `score = similarity * exp(-decay_alpha * age)`
- Exact age: `age = max(0, retrieval_step - slot.last_retrieved_step)`
- Therefore Δt_i = last-retrieved age (NOT retrieval count, NOT insertion age,
  NOT successful write count age)
- `_retrieval_step` is an enabled retrieval-turn counter
- `_step` is successful memory write count, used for created_step / stale checks
  / legacy ordering, not scoring
- `access_count` is incremented for returned slots but is not used in scoring
- Successful retrieval updates `last_retrieved_step`, effectively resetting
  last-retrieved age for returned slots
- Insertion/replacement initializes or resets recency
- Thread update eviction uses largest last-retrieved age, with tie-breaks by
  earliest created_step and smallest index
- Debug exports are consistent with this semantics
- tests exist for key behaviors:
  retrieval step increment, score uses last-retrieved age, only returned slots
  refresh, never-retrieved slot age from creation baseline, threshold miss does
  not refresh argmax, full-bank thread update eviction, replacement uses
  triggering retrieval step
- Caveat noted: threshold comment terminology mismatch (cosine vs decayed score)

### R4: Default-Threshold Natural Trigger Scan (Samples 1..5)

- Output: `outputs/r4_triviaqa_default_threshold_scan_version_a_s1_5/`
- memory-mode: `version_a_aligned`, threshold: default `0.7`, samples 1..5
- Result: valid 5/5, retrieval 5/5, natural triggers 0/5
- max_score values roughly 0.02–0.045 range
- Interpretation: default threshold 0.7 did not naturally trigger on this
  small scan; consistent with earlier 1-sample findings

### R4: 20-Sample Threshold Calibration Score Scan (Samples 0..19)

- Output:
  `outputs/r4_triviaqa_threshold_calibration_score_scan_s0_20/`
- Summary:
  `outputs/r4_triviaqa_threshold_calibration_score_scan_s0_20/threshold_calibration_summary.json`
- memory-mode: `version_a_aligned`, threshold: default `0.7`, samples 0..19
- Result: valid 20/20, retrieval 20/20, natural triggers at 0.7: 0/20
- Score distribution (decayed retrieval scores):
  - min: 0.0102, max: 0.0539, mean: 0.0356, median: 0.0368
  - p25: 0.0300, p75: 0.0441
- Offline hypothetical trigger counts:
  - threshold 0.01: 20/20 (100%)
  - threshold 0.02: 18/20 (90%)
  - threshold 0.03: 15/20 (75%)
  - threshold 0.04: 8/20 (40%)
  - threshold 0.05: 1/20 (5%)
  - threshold 0.10: 0/20, threshold 0.70: 0/20
- Interpretation:
  - default threshold 0.7 is far above observed decayed-score range for
    TriviaQA
  - threshold 0.04 selected as first calibrated candidate: moderate expected
    trigger rate (40%), no reward inspection

### R4: Threshold=0.04 Calibrated Behavior Scan (Samples 0..19)

- Output:
  `outputs/r4_triviaqa_threshold_calibrated_behavior_t004_s0_20/`
- Summary:
  `outputs/r4_triviaqa_threshold_calibrated_behavior_t004_s0_20/threshold_behavior_summary.json`
- memory-mode: `version_a_aligned`, in-memory threshold override: 0.04,
  samples 0..19
- Result: valid 20/20, retrieval 20/20
  - samples with `retrieved_latent_count > 0`: 8/20
  - total `retrieved_latent_count`: 64
  - `replace_matched` samples: 8, only-insert samples: 12
  - slot_count distribution: {1: 8, 2: 12}
  - observed trigger count exactly matched offline estimate: 8/20
- Interpretation:
  - threshold=0.04 successfully activates retrieved-memory injection path
  - behavior validation only, not performance evidence

### R4: Held-Out Exploratory Comparison (Samples 20..39)

- Outputs:
  - `outputs/r4_triviaqa_heldout_s20_39_disabled/`
  - `outputs/r4_triviaqa_heldout_s20_39_version_a_t004/`
  - `outputs/r4_triviaqa_heldout_s20_39_comparison_summary.json`
- Calibration samples 0..19; held-out samples 20..39
- Threshold 0.04 fixed before reward evaluation; no post-hoc tuning
- Result: valid 20/20 both runs, retrieval 24/0 both runs
  - disabled `compute_reward`: 0.60 (12/20)
  - Version A t=0.04 `compute_reward`: 0.55 (11/20)
  - only reward change: sample 21 1.0→0.0
  - Version A memory-triggered: 6/20, total retrieved_latent: 88
  - update trace: insert=34, replace_matched=11
- Interpretation:
  - one regression, no rescue in this subset
  - Version A t=0.04 can perturb final answers; not final benchmark evidence

### R4: Sample 21 Regression Case Study

- Question: "What Michelle Pfeiffer movie got a boost from the Coolio song
  Gangsta's Paradise?"
- Gold: "dangerous minds" / variants
- Disabled: "Dangerous Minds" (reward 1.0)
- Version A t=0.04: "Gangsta's Paradise" (reward 0.0)
- External retrieval was identical between runs; docs clearly contained
  the correct answer ("song was on the soundtrack for the 1995 film
  Dangerous Minds")
- Conversation messages identical up to final assistant answer
- Version A memory: writes=2, retrieves=1, retrieved_latent=8, slot_count=1,
  max_score=0.0534, threshold_passed=true, replace_matched
- Likely cause: memory-induced regression
  - retrieved latent memory may have amplified salient query/song entity
    "Gangsta's Paradise" instead of evidence-grounded movie answer
    "Dangerous Minds"
  - threshold 0.05 would NOT block this because score 0.0534 > 0.05
- Memory timing hypothesis:
  - first insert occurs during first generation turn, before Search-R1
    evidence is appended to context
  - later replace_matched occurs after external evidence is present
  - retrieved latent may be seeded from pre-evidence question/query context
    and injected into final evidence-grounded answer generation
  - this can amplify query-entity salience instead of evidence-grounded
    answers

### R4: Triggered Held-Out Audit (Samples 20..39 Memory-Triggered)

- Triggered samples: 20, 21, 34, 36, 37, 39
- Summary: helpful=0, harmful=1 (sample 21), neutral=3 (34, 36, 39),
  neutral/unclear=2 (20, 37)
- Details:
  - sample 20: disabled 0.0, Version A 0.0, answer changed, max_score=0.055,
    retrieved=24, neutral/unclear
  - sample 21: disabled 1.0, Version A 0.0, answer changed, max_score=0.053,
    retrieved=8, harmful
  - sample 34: disabled 0.0, Version A 0.0, answer changed, max_score=0.049,
    retrieved=8, neutral
  - sample 36: disabled 1.0, Version A 1.0, answer unchanged, max_score=0.048,
    retrieved=8, neutral
  - sample 37: disabled 0.0, Version A 0.0, answer unchanged, max_score=0.054,
    retrieved=32, neutral/unclear
  - sample 39: disabled 1.0, Version A 1.0, answer unchanged,
    max_score=0.045, retrieved=8, neutral
- Mechanism finding confirmed: pre-evidence latent may be seeded from
  query/context before Search-R1 evidence is present, then retrieved during
  post-evidence answer generation
- Threshold-only fix appears incomplete; issue related to memory
  timing/content

### R4: Fresh Held-Out Rescue/Regression Scan (Samples 40..79)

- Outputs:
  - `outputs/r4_triviaqa_rescue_scan_s40_79_disabled/`
  - `outputs/r4_triviaqa_rescue_scan_s40_79_version_a_t004/`
  - `outputs/r4_triviaqa_rescue_scan_s40_79_summary.json`
- Fresh samples 40..79, threshold 0.04 fixed before evaluation
- Result: valid 40/40 both runs
  - disabled retrieval 44/0, Version A retrieval 47/0
  - disabled `compute_reward`: 0.575 (23/40)
  - Version A t=0.04 `compute_reward`: 0.600 (24/40)
  - difference: +0.025
  - rescue count: 1 (sample 53), regression count: 0
  - Version A memory-triggered: 12/40, total retrieved_latent: 120
  - answer changed count: 4, answer changed but reward same: 3
- Notable rescue (sample 53):
  - question: "Which journalist first told the world about the My Lai
    massacre?"
  - disabled: "Normand Poirier" (wrong)
  - Version A: "Seymour Hersh" (correct, matches gold aliases)
  - max_score: 0.0441, retrieved_latent: 8, replace_matched
- Interpretation:
  - Version A can produce at least one rescue case — it is not only
    harmful/noisy
  - still exploratory only

### R4: Combined Held-Out Interpretation (Samples 20..79)

- Calibration samples: 0..19
- Held-out exploratory samples: 20..79 (60 total)
- Breakdown:
  - samples 20..39: disabled 12/20, Version A t=0.04 11/20,
    rescue=0, regression=1
  - samples 40..79: disabled 23/40, Version A t=0.04 24/40,
    rescue=1, regression=0
- Combined 20..79:
  - disabled: 35/60
  - Version A t=0.04: 35/60
  - net gain: 0
  - observed rescue: 1
  - observed regression: 1
- Interpretation:
  - Version A t=0.04 can both rescue and regress individual samples
  - current exploratory held-out evidence shows no net improvement across
    60 held-out samples
  - effect is fragile and sample-dependent
  - do not claim improvement; do not claim failure — evidence shows mixed
    behavior

### R4: Current Scientific Interpretation

- Default threshold 0.7 is not appropriate for the observed TriviaQA
  decayed-score scale; no natural retrieval in calibration
- Threshold 0.04 is a reasonable first calibrated candidate, not optimal
- Version A t=0.04 activates retrieved latent memory
- Retrieved latent memory can perturb final answers:
  - it can cause regressions (sample 21)
  - it can rescue wrong answers (sample 53)
- Across held-out 20..79, net effect is currently neutral
- Most important mechanism caveat: memory timing
  - pre-evidence latent memory may be written before Search-R1 evidence,
    then retrieved during post-evidence answer generation
  - this can amplify query/entity salience
- Future work should understand when memory helps vs hurts before scaling

### R4: Recommended Next Work

- Primary next step: read-only case study of rescue sample 53 vs harmful
  sample 21
- Goal: understand why sample 53 was rescued while sample 21 regressed
- Determine whether memory is acting as:
  - useful latent prior
  - evidence-grounded clue
  - query-salience amplifier
  - noisy perturbation
- Do NOT immediately run larger benchmarks, tune thresholds, or implement
  timing constraints
- Possible later variants after mechanism analysis:
  - evidence-grounded memory only
  - suppress pre-evidence memory write
  - retrieve only evidence-grounded slots
  - answer-stage verification/gating
  - threshold ablation after timing/content issue is understood

### R4: Current Caveats / Watchlist

- duplicate system prompt still appears across all R4 runs
  - was not differentiating in sample 21 because both runs shared it
- threshold comment misleading: threshold applies to decayed retrieval
  score, not raw cosine similarity
- threshold=0.04 overrides were in-memory diagnostics, not source/config
  changes
- reward means are exploratory only
- output artifacts are under `outputs/` and should not be staged
- source files remain unchanged
- Search-R1 was alive and retrieval succeeded in all reported runs

## 2026-06-20 - Version A Full TriviaQA Result Preserved; Ablations Paused

- Completed the Version A full TriviaQA rerun across all `7993` validation
  samples using 32 durable chunks under:
  `outputs/r4_triviaqa_full_version_a_t004_chunks_250_fullrerun/`
- Completed the sample-aligned paired comparison against the disabled full
  baseline under `outputs/r4_triviaqa_full_chunks/`.
- Completed artifact-only failure analysis under:
  `outputs/r4_triviaqa_full_version_a_t004_analysis/`
- Headline full result:
  - disabled: `5148/7993 = 0.6440635556`
  - Version A: `5092/7993 = 0.6370574252`
  - delta: `-56` correct, `-0.7006` percentage points
  - transitions: rescue `53`, regression `109`, stable correct `5039`,
    stable wrong `2792`
- Coverage is complete for both modes: missing `0`, duplicates `0`, valid
  `7970`, invalid/retrieval-blocked `23`; denominator remains all `7993`.
- Main failure-analysis result:
  - repeated injection is the strongest failure signal
  - `retrieved_latent_count=32+`: rescue `0`, regression `38`, net `-38`
  - `retrieve_count=4+`: rescue `2`, regression `44`, net `-42`
  - higher `max_score` is not a reliable correctness/confidence signal
- Scientific status:
  - **Version A full TriviaQA negative result, mechanism-active but
    policy-unstable.**
  - current behavior is brittle latent steering rather than reliable
    evidence-grounded memory
- User decision: pause further TriviaQA ablations. No threshold sweep or new
  mechanism ablation has been started.
- Authoritative resume summary:
  `research_notes/R4_TRIVIAQA_VERSION_A_FULL_SUMMARY.md`
- If work resumes, the first recommended experiment is a default-off
  max-one-injection ablation, followed by a cumulative
  `retrieved_latent_count <= 8` ablation. Do not start either without explicit
  confirmation.
