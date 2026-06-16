"""
Session-local 潜在记忆库（LatentMemoryBank）。

===== 架构概览 =====

本模块提供了推理时可选的 session 级检索增强循环潜在记忆机制。
核心设计原则：
- Session-local：每个 bank 实例绑定到一个 session（单轮或多轮），不跨 session 共享。
- 显式传入推理：bank 由 interaction manager 持有，通过参数传入 MemGenModel.generate()，
  不挂在 MemGenModel 上。
- 默认禁用：enabled=false 时完全零开销，不影响原始推理路径。
- 独立模块：Phase 4 中本模块不被任何 production 路径 import。

===== 核心数据结构 =====

LatentMemoryBankConfig : 所有超参数和策略选择。
LatentMemorySlot      : 一个记忆槽位，包含 memory tensor、key tensor、metadata。
LatentMemoryBank      : 按写入次数计步的槽位存储，支持多种检索/更新策略。

===== 检索流程 =====

1. build_query(hidden_states) —— 对最近 pool_last_n 个 token 的 hidden states 做 mean pooling
2. retrieve_with_context(query) —— 对所有 slot 计算 cosine similarity × last-retrieved decay
3. 根据 retrieve_policy 过滤（threshold / topk / threshold_topk）
4. 返回 detached clone，外部修改不影响 bank 内部状态

===== 写入流程 =====

write():
  - 不支持 thread_update 策略（thread_update 必须用 write_back）
  - 未满：append
  - 已满 + append 策略：拒绝写入
  - 已满 + replace_oldest 策略：替换 created_step 最小的 slot
  - 已满 + replace 策略：替换 last_score 最低的 slot（全未评分时退化为 oldest）

write_back():
  - 仅用于 thread_update 策略
  - 空 bank -> insert（新线程）
  - max_score >= threshold -> 替换 argmax slot（匹配线程更新）
  - max_score < threshold 且未满 -> insert（新线程）
  - max_score < threshold 且已满 -> 淘汰 last-retrieved age 最大的 slot，
    insert（新线程 + 容量管理）

===== 与项目架构的关系 =====

- Reasoner-only injection：检索到的记忆仅注入 Reasoner，不入 Weaver。
- Reasoner-space storage：存储的是 weaver_to_reasoner(...) 之后的 latent_inputs_embeds。
- 不对 Weaver/Trigger 训练路径做任何修改。
- _step 按成功写入次数计数，而非 generation token 数。
- _retrieval_step 按 enabled retrieval turn 计数。
- 当前 Version A-aligned decay 是 last-retrieved decay，不是 Version B。
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F


# --- 类型别名：策略与存储选项 ---
# 更新策略：写入时如何管理槽位
UpdatePolicy = Literal["append", "replace", "replace_oldest", "thread_update"]
# 检索策略：如何从 bank 中筛选相关记忆
RetrievePolicy = Literal["threshold", "topk", "threshold_topk"]
# 存储设备：cpu = 强制 CPU 存储, same = 保持原设备
StorageDevice = Literal["cpu", "same"]


@dataclass(frozen=True)
class LatentMemoryBankConfig:
    """记忆库配置（frozen，实例化后不可修改）。

    所有字段都有默认值，默认 disabled，对原始推理路径零影响。
    """

    # ---------- 基本开关 ----------
    enabled: bool = False          # 是否启用记忆库；禁用时 write/retrieve 均为 no-op
    batch_size: int = 1            # 当前仅支持 batch_size=1（Phase 4 约束）

    # ---------- 容量与检索 ----------
    max_slots: int = 8             # 最大槽位数
    top_k: int = 1                 # topk 检索时返回的槽位数
    threshold: float = 0.7         # 余弦相似度阈值，范围 [-1, 1]
    decay_alpha: float = 0.05      # 指数衰减系数：score *= exp(-alpha * age)；alpha=0 表示无衰减

    # ---------- 查询构造 ----------
    pool_last_n: int = 64          # query pooling：取最近 N 个 token 的 hidden states 做 mean

    # ---------- 策略选择 ----------
    update_policy: UpdatePolicy = "replace_oldest"    # 写入更新策略
    retrieve_policy: RetrievePolicy = "threshold_topk" # 检索过滤策略

    # ---------- 存储 ----------
    storage_device: StorageDevice = "cpu"  # slot tensor 的存储设备

    # ---------- 调试 ----------
    debug: bool = True             # 是否启用 debug 统计（不影响计算正确性）

    def __post_init__(self) -> None:
        """构造后校验：确保所有参数在合法范围内。"""
        if self.batch_size != 1:
            raise ValueError(
                "LatentMemoryBank currently supports batch_size=1 only"
            )
        if self.max_slots <= 0:
            raise ValueError("max_slots must be greater than zero")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not -1.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between -1.0 and 1.0")
        if self.decay_alpha < 0.0:
            raise ValueError("decay_alpha must be non-negative")
        if self.pool_last_n <= 0:
            raise ValueError("pool_last_n must be greater than zero")
        if self.update_policy not in {
            "append",
            "replace",
            "replace_oldest",
            "thread_update",
        }:
            raise ValueError(
                "update_policy must be append, replace, replace_oldest, "
                "or thread_update"
            )
        if self.retrieve_policy not in {
            "threshold",
            "topk",
            "threshold_topk",
        }:
            raise ValueError(
                "retrieve_policy must be threshold, topk, or threshold_topk"
            )
        if self.storage_device not in {"cpu", "same"}:
            raise ValueError("storage_device must be cpu or same")


@dataclass
class LatentMemorySlot:
    """一个记忆槽位。

    存储单次 Trigger 写入的 latent memory tensor 及其 key、元数据。
    """

    # --- 核心 tensor ---
    memory: torch.Tensor            # [token_count, hidden_size] 的 latent 表示
    key: torch.Tensor               # memory 的 mean-pooled key，用于相似度计算

    # --- 元数据 ---
    metadata: Dict[str, Any] = field(default_factory=dict)  # 用户自定义元数据

    # --- 生命周期追踪 ---
    # created_step：写入顺序标记，仅用于 legacy replace_oldest 和 eviction tie-break，
    # 不再参与 decay 计算。decay 依据 last_retrieved_step。
    created_step: int = 0           # 创建时的 bank _step（写入次数）
    last_access_step: int = 0       # 兼容旧字段：当前与 last_retrieved_step 同步，不单独参与 decay
    last_retrieved_step: int = 0    # 最后一次被真正选中检索的 retrieval turn；未被选中过的 slot 初始化为创建时的 retrieval_step
    access_count: int = 0           # 被检索的总次数
    last_score: Optional[float] = None  # 最后一次检索的相似度得分

    # --- 来源信息 ---
    original_device: str = "cpu"           # 写入时 tensor 所在设备
    original_dtype: str = "torch.float32"  # 写入时 tensor 的 dtype

    def debug_summary(self, current_retrieval_step: Optional[int] = None) -> Dict[str, Any]:
        """返回 slot 状态的只读摘要（用于 debug 日志和实验记录）。"""
        last_retrieved_age = None
        if current_retrieval_step is not None:
            last_retrieved_age = max(
                0,
                current_retrieval_step - self.last_retrieved_step,
            )
        return {
            "memory_shape": list(self.memory.shape),
            "key_shape": list(self.key.shape),
            "storage_device": str(self.memory.device),
            "storage_dtype": str(self.memory.dtype),
            "original_device": self.original_device,
            "original_dtype": self.original_dtype,
            "created_step": self.created_step,
            "last_access_step": self.last_access_step,
            "last_retrieved_step": self.last_retrieved_step,
            "last_retrieved_age": last_retrieved_age,
            "access_count": self.access_count,
            "last_score": self.last_score,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LatentMemoryRetrievalResult:
    """检索结果的不可变容器。

    包含检索到的 slots 以及完整的 bank 级别 score 信息，
    供 write_back 使用（用于 thread_update 策略的匹配判断）。
    """

    slots: List[LatentMemorySlot]           # 检索到的槽位（detached clone）
    scores: Tuple[float, ...]               # 全部槽位的得分，按原始 slot index 排序
    max_score: Optional[float]              # 最高得分（bank 为空时为 None）
    argmax_index: Optional[int]             # 最高得分对应的 slot index（bank 为空时为 None）
    threshold_passed: bool                  # max_score 是否 >= threshold
    retrieved_indices: Tuple[int, ...]      # 通过过滤的 slot indices
    retrieved_scores: Tuple[float, ...]     # 通过过滤的 slot scores
    bank_step: int                          # 检索时的 bank step，用于 write_back 的防过期校验
    retrieval_step: int = 0                 # 当前 retrieval turn


class LatentMemoryBank:
    """Session-local latent memory storage and retrieval skeleton.

    This class has no global registry and is intentionally not connected to any
    MemGen production inference path in Phase 4.

    ===== 生命周期 =====

    1. 构造：每个 session 创建一个实例（由 interaction manager 持有）
    2. 推理中：
       - retrieve / retrieve_with_context：按相似度检索相关记忆
       - write / write_back：写入新的 latent memory
    3. 结束时：随着 session 对象销毁而释放；或调用 reset() 手动清空

    ===== _step 语义 =====

    _step 记录的是成功写入次数（memory-write count），不是 generation token 数。
    这影响：
    - created_step 的取值
    - last_retrieved_step / last_access_step 的初始值

    _retrieval_step 记录 enabled retrieval turn。
    - age 计算：age = max(0, _retrieval_step - slot.last_retrieved_step)
    - decay 计算：score = similarity * exp(-decay_alpha * age)
    - 只有最终 selected / returned slots 更新 last_retrieved_step
    - created_step 不再参与 decay，仅保留为写入顺序标记和 eviction tie-break

    ===== 隔离保证 =====

    - write 时：detach + clone 后存储，断开与推理计算图的联系
    - retrieve 时：返回 detached clone，外部修改不影响 bank 内部状态
    - metadata：deepcopy，嵌套结构也完全隔离
    """

    def __init__(self, config: Optional[LatentMemoryBankConfig] = None) -> None:
        self.config = config or LatentMemoryBankConfig()
        self._slots: List[LatentMemorySlot] = []

        # ===== 核心计数器 =====
        # 成功写入次数，不是 token 数
        self._step = 0
        # retrieval turn 计数器，与 _step（写入次数）独立；
        # 每次 enabled 检索入口递增一；age 计算：current - slot.last_retrieved_step
        self._retrieval_step = 0

        # ===== 写入/检索统计 =====
        self._memory_write_count = 0        # 总写入次数
        self._memory_retrieve_count = 0     # 总检索次数
        self._retrieved_latent_count = 0    # 检索到的 latent token 总数
        self._new_latent_count = 0          # 新写入的 latent token 总数

        # ===== legacy 更新策略追踪 =====
        self._append_count = 0              # append 次数
        self._replace_count = 0             # replace 次数（所有策略的总替换数）
        self._rejected_write_count = 0      # 写入被拒绝次数（仅 append-full 场景）
        self._last_update_action: Optional[str] = None
        self._update_action_trace: List[str] = []

        # ===== thread_update 策略追踪 =====
        self._thread_insert_count = 0       # 新线程插入次数
        self._matched_replace_count = 0     # 匹配线程替换次数
        self._capacity_evict_count = 0      # 因容量满而淘汰 slot 的次数
        self._last_write_back: Optional[Dict[str, Any]] = None
        self._write_back_trace: List[Dict[str, Any]] = []

    # ==================================================================
    # 基本操作：容量、重置
    # ==================================================================

    def __len__(self) -> int:
        """返回当前槽位数。"""
        return len(self._slots)

    def reset(self) -> None:
        """重置 bank：清空所有槽位和计数器（等价于 clear）。"""
        self.clear()

    def clear(self) -> None:
        """清空所有槽位并将所有计数器归零。"""
        self._slots.clear()
        self._step = 0
        self._retrieval_step = 0
        self._memory_write_count = 0
        self._memory_retrieve_count = 0
        self._retrieved_latent_count = 0
        self._new_latent_count = 0
        self._append_count = 0
        self._replace_count = 0
        self._rejected_write_count = 0
        self._last_update_action = None
        self._update_action_trace = []
        self._thread_insert_count = 0
        self._matched_replace_count = 0
        self._capacity_evict_count = 0
        self._last_write_back = None
        self._write_back_trace = []

    # ==================================================================
    # Query / Key 构造
    # ==================================================================

    def build_query(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """根据 hidden states 序列构造检索用的 query。

        取最近 pool_last_n 个 token 的 hidden states，做 mean pooling。
        返回 1D tensor [hidden_size]，已 detach。
        """
        states = self._normalize_memory_tensor(hidden_states, "hidden_states")
        pooled_states = states[-self.config.pool_last_n :]
        return pooled_states.mean(dim=0).detach()

    def build_key(self, memory: torch.Tensor) -> torch.Tensor:
        """根据 memory tensor 构造存储用的 key。

        对 memory 的所有 token 维度做 mean pooling。
        返回 1D tensor [hidden_size]，已 detach。
        """
        states = self._normalize_memory_tensor(memory, "memory")
        return states.mean(dim=0).detach()

    # ==================================================================
    # 检索接口
    # ==================================================================

    def retrieve(
        self,
        query_or_hidden_states: torch.Tensor,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[LatentMemorySlot]:
        """检索相关记忆（legacy 兼容接口）。

        内部调用 retrieve_with_context() 并只返回 .slots。
        返回的是 detached clone，修改返回值不会影响 bank 内部状态。
        """
        return self.retrieve_with_context(
            query_or_hidden_states,
            device=device,
            dtype=dtype,
        ).slots

    def retrieve_with_context(
        self,
        query_or_hidden_states: torch.Tensor,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> LatentMemoryRetrievalResult:
        """检索相关记忆，返回完整的结构化检索结果（包含 all scores、max、argmax 等）。

        检索步骤：
        1. 如果 disabled 或 bank 为空 -> 返回空结果
        2. 如果输入是 1D tensor，直接作为 query；如果是 2D/3D，通过 build_query() 构造
        3. 对每个 slot 计算 cosine_similarity(query, slot.key)
        4. 乘以 last-retrieved decay: score = similarity * exp(-decay_alpha * age)
        5. 根据 retrieve_policy 过滤（threshold / topk / threshold_topk）
        6. 为每个选中 slot 构造 detached clone，同时更新 slot 的访问统计

        参数：
            query_or_hidden_states: 1D query 或 2D/3D hidden states 序列
            device: 输出 tensor 的目标设备（None = 跟随 query）
            dtype: 输出 tensor 的目标 dtype（None = 跟随 query）

        返回：
            LatentMemoryRetrievalResult：包含 slots、全部 scores、max_score 等完整信息
        """
        # --- disabled 或空 bank：直接返回空 ---
        # disabled 路径不递增 retrieval_step（零开销），但 enabled 空 bank 仍然递增，
        # 以保证后续写入的 slot 获得一致的时间基准（age 不会负值）。
        if not self.config.enabled or not self._slots:
            retrieval_step = self._retrieval_step
            if self.config.enabled:
                self._retrieval_step += 1
                retrieval_step = self._retrieval_step
            return LatentMemoryRetrievalResult(
                slots=[],
                scores=(),
                max_score=None,
                argmax_index=None,
                threshold_passed=False,
                retrieved_indices=(),
                retrieved_scores=(),
                bank_step=self._step,
                retrieval_step=retrieval_step,
            )
        # 每次 enabled 检索入口恰推进一个 retrieval turn。retrieve() 委托进来不自增第二遍。
        self._retrieval_step += 1
        retrieval_step = self._retrieval_step

        # --- 输入校验与 query 构造 ---
        if not isinstance(query_or_hidden_states, torch.Tensor):
            raise TypeError("query_or_hidden_states must be a torch.Tensor")
        if query_or_hidden_states.ndim == 1:
            if query_or_hidden_states.shape[0] == 0:
                raise ValueError("query cannot be empty")
            if not query_or_hidden_states.is_floating_point():
                raise TypeError("query must use a floating-point dtype")
            query = query_or_hidden_states.detach()
        else:
            # 2D 或 3D：通过 build_query 从 hidden states 构造 query
            query = self.build_query(query_or_hidden_states)

        # --- 对每个 slot 计算相似度得分 ---
        scored_slots = []
        for index, slot in enumerate(self._slots):
            if slot.key.shape != query.shape:
                raise ValueError(
                    "query/key hidden size mismatch: "
                    f"query={tuple(query.shape)}, slot[{index}].key="
                    f"{tuple(slot.key.shape)}"
                )
            score_device = query.device
            score_dtype = query.dtype
            # 将 slot key 搬到 query 所在的 device/dtype 上计算
            slot_key = slot.key.to(device=score_device, dtype=score_dtype)
            similarity = F.cosine_similarity(
                query.unsqueeze(0),
                slot_key.unsqueeze(0),
                dim=-1,
            ).item()
            # last-retrieved decay：age 使用 retrieval_step - slot.last_retrieved_step，
            # 而非 created_step。原因是 write-age decay 不能反映实际使用频率：
            # 一个很早写入但频繁被检索的 slot 应该比一个刚写入但从未被检索的 slot 得分更高。
            # last_retrieved_step 记录的是真正被选中返回给 Reasoner 的上一次 retrieval turn，
            # 并非所有参与评分计算的访问。
            age = max(0, retrieval_step - slot.last_retrieved_step)
            score = similarity * math.exp(-self.config.decay_alpha * age)
            scored_slots.append((score, index, slot))

        # --- 构建按原始 index 排序的 scores 元组 ---
        scores = [0.0] * len(self._slots)
        for score, index, _ in scored_slots:
            scores[index] = score

        # --- 排序：按 score 降序，同等 score 保留较小原始 index ---
        scored_slots.sort(key=lambda item: (-item[0], item[1]))
        max_score, argmax_index, _ = scored_slots[0]
        selected_slots = scored_slots

        # --- threshold 过滤 ---
        # threshold 全不过时 selected_slots 为空，后续返回空 slots 给 Reasoner。
        # 这里不做 fallback top-1：低于 threshold 时即使有 argmax，也不返回任何 slot，
        # 因为低相似度记忆对生成有害。argmax_index 仅保留用于 matched_thread 判断。
        if self.config.retrieve_policy in {"threshold", "threshold_topk"}:
            selected_slots = [
                item
                for item in selected_slots
                if item[0] >= self.config.threshold
            ]

        # --- top-k 截断 ---
        if self.config.retrieve_policy in {"topk", "threshold_topk"}:
            selected_slots = selected_slots[: self.config.top_k]

        # --- 构造输出 slot（detached clone）并更新访问统计 ---
        # 只有最终进入 selected_slots 的 slot 会更新 last_retrieved_step。
        # 参与 scoring 但被 threshold/top-k 过滤掉的 slot 不刷新，
        # 从而它们的 last_retrieved_age 继续增长，下次检索时衰减更大。
        # 这是 Version A-aligned 的行为：仅真正返回给 Reasoner 的 slot 算"被检索到"。
        output_device = torch.device(device) if device is not None else query.device
        output_dtype = dtype if dtype is not None else query.dtype
        retrieved = []
        for score, _, slot in selected_slots:
            # 更新 bank 内 slot 的访问统计
            slot.last_retrieved_step = retrieval_step
            slot.last_access_step = retrieval_step
            slot.access_count += 1
            slot.last_score = score
            # 构造 detached clone 返回给 caller
            retrieved.append(
                LatentMemorySlot(
                    memory=slot.memory.to(
                        device=output_device,
                        dtype=output_dtype,
                    ).detach().clone(),
                    key=slot.key.to(
                        device=output_device,
                        dtype=output_dtype,
                    ).detach().clone(),
                    metadata=deepcopy(slot.metadata),
                    created_step=slot.created_step,
                    last_access_step=slot.last_access_step,
                    last_retrieved_step=slot.last_retrieved_step,
                    access_count=slot.access_count,
                    last_score=score,
                    original_device=slot.original_device,
                    original_dtype=slot.original_dtype,
                )
            )

        # --- 更新检索统计 ---
        self._memory_retrieve_count += 1
        self._retrieved_latent_count += sum(slot.memory.shape[0] for slot in retrieved)

        return LatentMemoryRetrievalResult(
            slots=retrieved,
            scores=tuple(scores),
            max_score=max_score,
            argmax_index=argmax_index,
            threshold_passed=max_score >= self.config.threshold,
            retrieved_indices=tuple(index for _, index, _ in selected_slots),
            retrieved_scores=tuple(score for score, _, _ in selected_slots),
            bank_step=self._step,
            retrieval_step=retrieval_step,
        )

    # ==================================================================
    # 写入接口
    # ==================================================================

    def write(
        self,
        memory: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """写入新记忆（legacy 更新策略：append / replace / replace_oldest）。

        注意：不支持 thread_update 策略，thread_update 必须使用 write_back()。

        写入逻辑：
        - 未满容量：直接 append
        - 已满 + append：拒绝写入，记录 rejected
        - 已满 + replace_oldest：替换 created_step 最小的 slot
        - 已满 + replace：替换 last_score 最低的 slot（全未评分时退化为 replace_oldest）

        返回 True 表示成功写入，False 表示被拒绝（仅 append-full 场景）。
        """
        if not self.config.enabled:
            return False
        if self.config.update_policy == "thread_update":
            raise ValueError(
                "write does not support update_policy='thread_update'; "
                "use write_back instead"
            )

        normalized = self._normalize_memory_tensor(memory, "memory")

        # --- append 策略 + 已满：拒绝写入 ---
        if (
            len(self._slots) >= self.config.max_slots
            and self.config.update_policy == "append"
        ):
            self._rejected_write_count += 1
            self._last_update_action = "reject_append_full"
            self._update_action_trace.append(self._last_update_action)
            return False

        new_slot = self._create_slot(normalized, metadata)
        # 不传 retrieval_step，_create_slot 回退到 self._retrieval_step。
        # 这对 write() 路径是可接受的，因为 write() 不与特定 retrieval 绑定。

        # --- 未满：直接追加 ---
        if len(self._slots) < self.config.max_slots:
            self._slots.append(new_slot)
            self._append_count += 1
            self._last_update_action = "append"
            self._update_action_trace.append(self._last_update_action)
            return True

        # --- 已满：选择替换目标 ---
        if self.config.update_policy == "replace_oldest":
            # 替换最旧的 slot（最小的 created_step）
            replace_index = min(
                range(len(self._slots)),
                key=lambda index: self._slots[index].created_step,
            )
        else:
            # replace 策略：优先替换 last_score 最低的
            scored_indices = [
                index
                for index, slot in enumerate(self._slots)
                if slot.last_score is not None
            ]
            if not scored_indices:
                # 全未评分：退化为 oldest
                replace_index = min(
                    range(len(self._slots)),
                    key=lambda index: self._slots[index].created_step,
                )
            else:
                replace_index = min(
                    range(len(self._slots)),
                    key=lambda index: (
                        self._slots[index].last_score
                        if self._slots[index].last_score is not None
                        else float("-inf")
                    ),
                )
        self._slots[replace_index] = new_slot
        self._replace_count += 1
        self._last_update_action = "replace"
        self._update_action_trace.append(self._last_update_action)
        return True

    def write_back(
        self,
        memory: torch.Tensor,
        retrieval_result: LatentMemoryRetrievalResult,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """thread_update 策略的写入入口。

        与 write() 不同，write_back 需要当前 query 的完整检索结果，
        以便判断是"匹配已有线程"还是"新线程"。

        决策逻辑（按优先级）：
        1. bank 为空 -> insert（新线程，reason="empty_bank"）
        2. max_score >= threshold -> 替换 argmax slot（匹配线程更新，reason="matched_thread"）
        3. max_score < threshold 且未满 -> insert（新线程，reason="new_thread"）
        4. max_score < threshold 且已满 -> 淘汰 last-retrieved age 最大的 slot 后 insert
           （新线程，reason="new_thread_bank_full"）

        返回 True 表示写入成功。
        """
        if not self.config.enabled:
            return False
        if self.config.update_policy != "thread_update":
            raise ValueError(
                "write_back requires update_policy='thread_update'"
            )
        if not isinstance(retrieval_result, LatentMemoryRetrievalResult):
            raise TypeError(
                "retrieval_result must be a LatentMemoryRetrievalResult"
            )

        # --- 防过期校验：确保 retrieval_result 来自当前 bank 状态 ---
        if retrieval_result.bank_step != self._step:
            raise ValueError(
                "stale retrieval_result: "
                f"bank_step={retrieval_result.bank_step}, current_step={self._step}"
            )
        if retrieval_result.scores != () and len(retrieval_result.scores) != len(self._slots):
            raise ValueError(
                "retrieval_result scores do not match current bank slot count"
            )

        # --- 判断是否匹配已有线程 ---
        matched = (
            bool(self._slots)
            and retrieval_result.max_score is not None
            and retrieval_result.max_score >= self.config.threshold
        )
        matched_index = retrieval_result.argmax_index
        if matched and (
            matched_index is None
            or matched_index < 0
            or matched_index >= len(self._slots)
        ):
            raise ValueError(
                "retrieval_result.argmax_index is invalid for matched replacement"
            )

        normalized = self._normalize_memory_tensor(memory, "memory")
        # 使用 retrieval_result.retrieval_step 而非 self._retrieval_step：
        # 如果在 retrieve_with_context() 和 write_back() 之间发生了另一次检索，
        # self._retrieval_step 会大于触发此次 write_back 的 retrieval step。
        # 绑定到 retrieval_result 的 step 保证新 slot 的时间基准与得分计算一致。
        new_slot = self._create_slot(
            normalized,
            metadata,
            retrieval_step=retrieval_result.retrieval_step,
        )

        # --- 状态变量，用于构造 debug event ---
        replaced_slot_index = None
        replaced_slot_score = None
        evicted_slot_index = None
        evicted_slot_last_retrieved_age = None
        eviction_basis = None
        inserted_new_thread = False

        if not self._slots:
            # 情况 1：空 bank -> 插入
            self._slots.append(new_slot)
            self._append_count += 1
            self._thread_insert_count += 1
            write_action = "insert"
            update_reason = "empty_bank"
            inserted_new_thread = True
        elif matched:
            # 情况 2：匹配线程 -> 替换 argmax slot
            replaced_slot_index = matched_index
            replaced_slot_score = retrieval_result.max_score
            self._slots[matched_index] = new_slot
            self._replace_count += 1
            self._matched_replace_count += 1
            write_action = "replace_matched"
            update_reason = "matched_thread"
        elif len(self._slots) < self.config.max_slots:
            # 情况 3：未满 + 新线程 -> 插入
            self._slots.append(new_slot)
            self._append_count += 1
            self._thread_insert_count += 1
            write_action = "insert"
            update_reason = "new_thread"
            inserted_new_thread = True
        else:
            # 情况 4：已满 + 新线程 -> 淘汰 last-retrieved age 最大者，插入
            # 使用 largest last_retrieved_age 而非 oldest created_step：
            # 一个创建早但频繁被检索的 slot 比一个创建晚但从未被检索的 slot
            # 更值得保留。last_retrieved_age 衡量的是"多久未被实际使用"，
            # 比 write-age 更能反映 slot 的当前价值。
            # Tie-break（确定性的）：
            #   largest age -> earliest created_step -> smallest index
            # 确定性保证同一状态下多次 eviction 结果一致，便于实验复现。
            evicted_slot_index = max(
                range(len(self._slots)),
                key=lambda index: (
                    retrieval_result.retrieval_step
                    - self._slots[index].last_retrieved_step,
                    -self._slots[index].created_step,
                    -index,
                ),
            )
            evicted_slot_last_retrieved_age = max(
                0,
                retrieval_result.retrieval_step
                - self._slots[evicted_slot_index].last_retrieved_step,
            )
            eviction_basis = "last_retrieved_age"
            self._slots[evicted_slot_index] = new_slot
            self._replace_count += 1
            self._thread_insert_count += 1
            self._capacity_evict_count += 1
            write_action = "evict_oldest_insert"
            update_reason = "new_thread_bank_full"
            inserted_new_thread = True

        # --- 记录 debug event ---
        event = {
            "matched_slot_index": matched_index,
            "max_score": retrieval_result.max_score,
            "threshold_passed": retrieval_result.threshold_passed,
            "retrieved_indices": list(retrieval_result.retrieved_indices),
            "retrieved_scores": list(retrieval_result.retrieved_scores),
            "write_action": write_action,
            "replaced_slot_index": replaced_slot_index,
            "replaced_slot_score": replaced_slot_score,
            "evicted_slot_index": evicted_slot_index,
            "evicted_slot_last_retrieved_age": evicted_slot_last_retrieved_age,
            "eviction_basis": eviction_basis,
            "update_reason": update_reason,
            "inserted_new_thread": inserted_new_thread,
            "retrieval_bank_step": retrieval_result.bank_step,  # 用于检测 stale write_back
            "retrieval_step": retrieval_result.retrieval_step,  # 触发本次 write_back 的 retrieval turn
        }
        self._last_update_action = write_action
        self._update_action_trace.append(write_action)
        self._last_write_back = deepcopy(event)
        self._write_back_trace.append(deepcopy(event))
        return True

    # ==================================================================
    # 调试与状态导出
    # ==================================================================

    def debug_summary(self) -> Dict[str, Any]:
        """返回完整的 debug 摘要，包含配置、计数器、所有 slot 摘要。

        用于实验记录中的 memory bank 状态 snapshot。
        """
        return {
            "config": asdict(self.config),
            "enabled": self.config.enabled,
            "step": self._step,
            "retrieval_step": self._retrieval_step,  # 当前 retrieval turn，用于验证 last-retrieved decay
            "slot_count": len(self._slots),
            "memory_write_count": self._memory_write_count,
            "memory_retrieve_count": self._memory_retrieve_count,
            "retrieved_latent_count": self._retrieved_latent_count,
            "new_latent_count": self._new_latent_count,
            "append_count": self._append_count,
            "replace_count": self._replace_count,
            "rejected_write_count": self._rejected_write_count,
            "last_update_action": self._last_update_action,
            "update_action_trace": list(self._update_action_trace),
            "thread_insert_count": self._thread_insert_count,
            "matched_replace_count": self._matched_replace_count,
            "capacity_evict_count": self._capacity_evict_count,
            "last_write_back": deepcopy(self._last_write_back),
            "write_back_trace": deepcopy(self._write_back_trace),
            "slots": [
                slot.debug_summary(self._retrieval_step)
                for slot in self._slots
            ],
        }

    def state_dict(self) -> Dict[str, Any]:
        """返回 detached 的 bank 状态快照（不是训练 checkpoint）。

        所有 tensor 都已 detach + clone，保证与原 bank 完全隔离。
        用于 debug 对比和验证，不用于训练恢复。
        """
        return {
            "config": asdict(self.config),
            "step": self._step,
            "retrieval_step": self._retrieval_step,
            "slots": [
                {
                    "memory": slot.memory.detach().clone(),
                    "key": slot.key.detach().clone(),
                    "metadata": deepcopy(slot.metadata),
                    "created_step": slot.created_step,
                    "last_access_step": slot.last_access_step,
                    "last_retrieved_step": slot.last_retrieved_step,
                    "access_count": slot.access_count,
                    "last_score": slot.last_score,
                    "original_device": slot.original_device,
                    "original_dtype": slot.original_dtype,
                }
                for slot in self._slots
            ],
        }

    # ==================================================================
    # 内部方法
    # ==================================================================

    def _create_slot(
        self,
        normalized: torch.Tensor,
        metadata: Optional[Dict[str, Any]],
        retrieval_step: Optional[int] = None,
    ) -> LatentMemorySlot:
        """根据归一化后的 memory tensor 创建新 slot。

        retrieval_step 参数（Phase R2 新增）：
        - write() 路径不传 retrieval_step，回退到 self._retrieval_step。
          这对 write() 是可接受的，因为它不与特定 retrieval 绑定。
        - write_back() 路径显式传入 retrieval_result.retrieval_step。
          这样即使 retrieve_with_context() 和 write_back() 之间发生了
          其他检索，新 slot 的 last_retrieved_step 仍然绑定到触发
          本次 write_back 的 retrieval turn，与得分计算使用的 step 一致。
        - last_access_step 和 last_retrieved_step 总是同步到此值。

        执行步骤：
        1. 记录原始 device 和 dtype
        2. 根据 storage_device 配置决定存储位置（cpu / same）
        3. detach + clone 建立隔离副本
        4. 构造 key（mean pooling + clone）
        5. 递增 _step 和统计计数器

        此方法会修改 bank 状态（递增 _step 和计数器）。
        """
        original_device = str(normalized.device)
        original_dtype = str(normalized.dtype)

        # detach + clone：断开计算图并创建独立副本
        stored_memory = normalized.detach().clone()
        if self.config.storage_device == "cpu":
            stored_memory = stored_memory.to("cpu")

        # key 也做 clone，与 memory 同设备/dtype
        stored_key = self.build_key(stored_memory).to(
            device=stored_memory.device,
            dtype=stored_memory.dtype,
        ).clone()

        # 递增写入计数器
        self._step += 1
        self._memory_write_count += 1
        self._new_latent_count += stored_memory.shape[0]
        # 确定新 slot 的 last_retrieved_step：
        # - 如果 caller 显式传入了 retrieval_step（write_back 路径），使用该值
        # - 否则（write 路径）使用当前 bank 的 self._retrieval_step
        slot_retrieval_step = (
            self._retrieval_step
            if retrieval_step is None
            else retrieval_step
        )

        return LatentMemorySlot(
            memory=stored_memory,
            key=stored_key,
            metadata=deepcopy(metadata or {}),
            created_step=self._step,
            last_access_step=slot_retrieval_step,
            last_retrieved_step=slot_retrieval_step,
            original_device=original_device,
            original_dtype=original_dtype,
        )

    @staticmethod
    def _normalize_memory_tensor(
        tensor: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        """输入校验与归一化。

        - 3D [1, tokens, hidden] -> 压缩为 2D [tokens, hidden]
        - 2D [tokens, hidden] -> 直接通过
        - 拒绝非浮点、空维度等非法输入
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim == 3:
            if tensor.shape[0] != 1:
                raise ValueError(
                    f"{name} currently supports batch_size=1 only; "
                    f"received shape {tuple(tensor.shape)}"
                )
            tensor = tensor.squeeze(0)
        if tensor.ndim != 2:
            raise ValueError(
                f"{name} must have shape [tokens, hidden] or "
                f"[1, tokens, hidden]; received {tuple(tensor.shape)}"
            )
        if tensor.shape[0] == 0 or tensor.shape[1] == 0:
            raise ValueError(f"{name} cannot contain an empty dimension")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        return tensor
