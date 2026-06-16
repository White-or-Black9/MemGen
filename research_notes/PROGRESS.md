# 项目进展

## 当前状态

- 当前状态：Phase R2 已将 Version A-aligned `thread_update` 修订为
  last-retrieved decay；下一步仍是目标任务 baseline / infrastructure 规划。
- 状态：`completed`
- 最后更新：2026-06-12
- 衰减 / fallback 实现审计：`completed`
- 方法 / 计划对齐更新：`completed`
- Step 2 结构化检索上下文：`completed`
- Step 3 线程感知写回：`completed`
- Step 4 thread-update 机制 smoke：`completed`
  (`EXP-20260612-024`)
- Phase R2 last-retrieved decay revision：`completed`
- 停止条件：已达到；没有明确批准，不要进入新的实现或实验阶段。
- 下一步建议：规划并建立 Original MemGen / disabled-memory TriviaQA baseline。
  Version B 仍未开始。

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
