from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Literal, Optional, Union

import torch
import torch.nn.functional as F


UpdatePolicy = Literal["append", "replace", "replace_oldest"]
RetrievePolicy = Literal["threshold", "topk", "threshold_topk"]
StorageDevice = Literal["cpu", "same"]


@dataclass(frozen=True)
class LatentMemoryBankConfig:
    enabled: bool = False
    batch_size: int = 1
    max_slots: int = 8
    top_k: int = 1
    threshold: float = 0.7
    decay_alpha: float = 0.05
    pool_last_n: int = 64
    update_policy: UpdatePolicy = "replace_oldest"
    retrieve_policy: RetrievePolicy = "threshold_topk"
    storage_device: StorageDevice = "cpu"
    debug: bool = True

    def __post_init__(self) -> None:
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
        if self.update_policy not in {"append", "replace", "replace_oldest"}:
            raise ValueError(
                "update_policy must be append, replace, or replace_oldest"
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
    memory: torch.Tensor
    key: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_step: int = 0
    last_access_step: int = 0
    access_count: int = 0
    last_score: Optional[float] = None
    original_device: str = "cpu"
    original_dtype: str = "torch.float32"

    def debug_summary(self) -> Dict[str, Any]:
        return {
            "memory_shape": list(self.memory.shape),
            "key_shape": list(self.key.shape),
            "storage_device": str(self.memory.device),
            "storage_dtype": str(self.memory.dtype),
            "original_device": self.original_device,
            "original_dtype": self.original_dtype,
            "created_step": self.created_step,
            "last_access_step": self.last_access_step,
            "access_count": self.access_count,
            "last_score": self.last_score,
            "metadata": dict(self.metadata),
        }


class LatentMemoryBank:
    """Session-local latent memory storage and retrieval skeleton.

    This class has no global registry and is intentionally not connected to any
    MemGen production inference path in Phase 4.
    """

    def __init__(self, config: Optional[LatentMemoryBankConfig] = None) -> None:
        self.config = config or LatentMemoryBankConfig()
        self._slots: List[LatentMemorySlot] = []
        # Counts successful memory writes, not generation tokens.
        self._step = 0
        self._memory_write_count = 0
        self._memory_retrieve_count = 0
        self._retrieved_latent_count = 0
        self._new_latent_count = 0
        self._append_count = 0
        self._replace_count = 0
        self._rejected_write_count = 0
        self._last_update_action: Optional[str] = None
        self._update_action_trace: List[str] = []

    def __len__(self) -> int:
        return len(self._slots)

    def reset(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._slots.clear()
        self._step = 0
        self._memory_write_count = 0
        self._memory_retrieve_count = 0
        self._retrieved_latent_count = 0
        self._new_latent_count = 0
        self._append_count = 0
        self._replace_count = 0
        self._rejected_write_count = 0
        self._last_update_action = None
        self._update_action_trace = []

    def build_query(self, hidden_states: torch.Tensor) -> torch.Tensor:
        states = self._normalize_memory_tensor(hidden_states, "hidden_states")
        pooled_states = states[-self.config.pool_last_n :]
        return pooled_states.mean(dim=0).detach()

    def build_key(self, memory: torch.Tensor) -> torch.Tensor:
        states = self._normalize_memory_tensor(memory, "memory")
        return states.mean(dim=0).detach()

    def retrieve(
        self,
        query_or_hidden_states: torch.Tensor,
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[LatentMemorySlot]:
        """Return detached slot copies that cannot mutate bank-owned state."""
        if not self.config.enabled or not self._slots:
            return []

        if not isinstance(query_or_hidden_states, torch.Tensor):
            raise TypeError("query_or_hidden_states must be a torch.Tensor")
        if query_or_hidden_states.ndim == 1:
            if query_or_hidden_states.shape[0] == 0:
                raise ValueError("query cannot be empty")
            if not query_or_hidden_states.is_floating_point():
                raise TypeError("query must use a floating-point dtype")
            query = query_or_hidden_states.detach()
        else:
            query = self.build_query(query_or_hidden_states)
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
            slot_key = slot.key.to(device=score_device, dtype=score_dtype)
            similarity = F.cosine_similarity(
                query.unsqueeze(0),
                slot_key.unsqueeze(0),
                dim=-1,
            ).item()
            age = max(0, self._step - slot.created_step)
            score = similarity * math.exp(-self.config.decay_alpha * age)
            scored_slots.append((score, index, slot))

        scored_slots.sort(key=lambda item: item[0], reverse=True)
        if self.config.retrieve_policy in {"threshold", "threshold_topk"}:
            scored_slots = [
                item for item in scored_slots if item[0] >= self.config.threshold
            ]
        if self.config.retrieve_policy in {"topk", "threshold_topk"}:
            scored_slots = scored_slots[: self.config.top_k]

        output_device = torch.device(device) if device is not None else query.device
        output_dtype = dtype if dtype is not None else query.dtype
        retrieved = []
        for score, _, slot in scored_slots:
            slot.last_access_step = self._step
            slot.access_count += 1
            slot.last_score = score
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
                    access_count=slot.access_count,
                    last_score=score,
                    original_device=slot.original_device,
                    original_dtype=slot.original_dtype,
                )
            )
        self._memory_retrieve_count += 1
        self._retrieved_latent_count += sum(slot.memory.shape[0] for slot in retrieved)
        return retrieved

    def write(
        self,
        memory: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.config.enabled:
            return False

        normalized = self._normalize_memory_tensor(memory, "memory")
        if (
            len(self._slots) >= self.config.max_slots
            and self.config.update_policy == "append"
        ):
            self._rejected_write_count += 1
            self._last_update_action = "reject_append_full"
            self._update_action_trace.append(self._last_update_action)
            return False

        original_device = str(normalized.device)
        original_dtype = str(normalized.dtype)
        stored_memory = normalized.detach().clone()
        if self.config.storage_device == "cpu":
            stored_memory = stored_memory.to("cpu")
        stored_key = self.build_key(stored_memory).to(
            device=stored_memory.device,
            dtype=stored_memory.dtype,
        ).clone()

        self._step += 1
        self._memory_write_count += 1
        self._new_latent_count += stored_memory.shape[0]
        new_slot = LatentMemorySlot(
            memory=stored_memory,
            key=stored_key,
            metadata=deepcopy(metadata or {}),
            created_step=self._step,
            last_access_step=self._step,
            original_device=original_device,
            original_dtype=original_dtype,
        )

        if len(self._slots) < self.config.max_slots:
            self._slots.append(new_slot)
            self._append_count += 1
            self._last_update_action = "append"
            self._update_action_trace.append(self._last_update_action)
            return True
        if self.config.update_policy == "replace_oldest":
            replace_index = min(
                range(len(self._slots)),
                key=lambda index: self._slots[index].created_step,
            )
        else:
            scored_indices = [
                index
                for index, slot in enumerate(self._slots)
                if slot.last_score is not None
            ]
            if not scored_indices:
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

    def debug_summary(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "enabled": self.config.enabled,
            "step": self._step,
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
            "slots": [slot.debug_summary() for slot in self._slots],
        }

    def state_dict(self) -> Dict[str, Any]:
        """Return a detached debug snapshot; this is not a training checkpoint."""
        return {
            "config": asdict(self.config),
            "step": self._step,
            "slots": [
                {
                    "memory": slot.memory.detach().clone(),
                    "key": slot.key.detach().clone(),
                    "metadata": deepcopy(slot.metadata),
                    "created_step": slot.created_step,
                    "last_access_step": slot.last_access_step,
                    "access_count": slot.access_count,
                    "last_score": slot.last_score,
                    "original_device": slot.original_device,
                    "original_dtype": slot.original_dtype,
                }
                for slot in self._slots
            ],
        }

    @staticmethod
    def _normalize_memory_tensor(
        tensor: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
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
