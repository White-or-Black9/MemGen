"""M2 experiment-only, count-matched selectors and replayable CPU snapshots.

No production bank/configuration changes. Native retrieval determines the count
and scores; policies only change the selected set and its presentation order.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
from types import MethodType

import torch
import torch.nn.functional as F

from memgen.model.latent_memory_bank import LatentMemoryBank

POLICIES = ("full", "random", "last_written", "cosine_only", "recency_only")
REFERENCE = "native_order"


def policy_seed(seed, context, query):
    identity = f"m2-random/v1:{seed}:{context}:{query}"
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")


def select_indices(policy, *, full_indices, cosine, ages, last_write, rng_seed):
    """All selectors choose native full's count; ties use stable slot index."""
    if policy not in POLICIES + (REFERENCE,):
        raise ValueError(f"unknown policy {policy}")
    size, n = len(cosine), len(full_indices)
    if len(ages) != size or len(last_write) != size:
        raise ValueError("incomplete slot metadata")
    if n > size or len(set(full_indices)) != n or any(i not in range(size) for i in full_indices):
        raise ValueError("invalid native selection/capacity")
    if any(not math.isfinite(v) for v in cosine):
        raise ValueError("nonfinite cosine")
    if any(v is None for v in last_write):
        raise ValueError("missing construction last-write provenance")
    if policy in ("full", REFERENCE):
        chosen = list(full_indices)
    elif policy == "random":
        chosen = random.Random(rng_seed).sample(range(size), n)
    elif policy == "last_written":
        chosen = sorted(range(size), key=lambda i: (-last_write[i], i))[:n]
    elif policy == "cosine_only":
        chosen = sorted(range(size), key=lambda i: (-cosine[i], i))[:n]
    else:
        chosen = sorted(range(size), key=lambda i: (ages[i], i))[:n]
    return chosen if policy == REFERENCE else sorted(chosen)


def derive_last_write(events, slot_count):
    last = [None] * slot_count
    for event in events:
        index = event.get("target_slot_index")
        if index is not None and event.get("write_action") in {
            "insert", "replace_matched", "evict_oldest_insert",
        }:
            if index not in range(slot_count):
                raise ValueError("construction slot index out of range")
            last[index] = int(event["write_count_after_write"])
    if any(v is None for v in last):
        raise ValueError("incomplete construction last-write provenance")
    return last


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(v) for v in value)
    # Dataclass slots require explicit tensor copies as well.
    if hasattr(value, "memory") and hasattr(value, "key"):
        return replace(value, memory=_cpu_copy(value.memory), key=_cpu_copy(value.key),
                       metadata=deepcopy(value.metadata))
    return deepcopy(value)


def capture_bank(bank):
    state = {}
    for key, value in bank.__dict__.items():
        if callable(value):
            if getattr(value, "__func__", None) is not getattr(type(bank), key, None):
                raise ValueError("capture bank only after temporary trace hooks are removed")
        else:
            state[key] = value
    return _cpu_copy(state)


def restore_bank(snapshot):
    bank = LatentMemoryBank(snapshot["config"])
    bank.__dict__.update(_cpu_copy(snapshot))
    return bank


def fingerprint(snapshot):
    digest = hashlib.sha256()

    def walk(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                digest.update(str(key).encode())
                walk(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(str(len(value)).encode())
            for item in value:
                walk(item)
        elif hasattr(value, "__dataclass_fields__"):
            walk(vars(value))
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode())

    walk(snapshot)
    return digest.hexdigest()


def capture_rng():
    import numpy as np
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("snapshot CUDA RNG topology differs")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_snapshot(path, bank_state, rng_state, last_write, provenance):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "m2-bank/v1", "bank": bank_state, "rng": rng_state,
               "last_write": last_write, "provenance": provenance,
               "bank_sha256": fingerprint(bank_state)}
    torch.save(payload, path)


def load_snapshot(path):
    # Only locally generated, trusted snapshots. torch.load uses pickle.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "m2-bank/v1":
        raise ValueError("unsupported snapshot schema")
    if fingerprint(payload["bank"]) != payload["bank_sha256"]:
        raise ValueError("snapshot bank hash mismatch")
    if len(payload["last_write"]) != len(payload["bank"]["_slots"]) or any(
        v is None for v in payload["last_write"]
    ):
        raise ValueError("snapshot last-write metadata incomplete")
    return payload


@contextmanager
def policy_adapter(bank, policy, *, last_write, seed, context, query, trace):
    """Temporary per-instance hook. The existing read-only proxy restores state."""
    original = bank.retrieve_with_context
    had_override = "retrieve_with_context" in bank.__dict__
    rng_id = policy_seed(seed, context, query)

    def retrieve(self, query_states, **kwargs):
        # Native call preserves exact threshold/top-k and dtype behavior.
        before = [(s.last_retrieved_step, s.last_access_step, s.access_count, s.last_score)
                  for s in self._slots]
        native = original(query_states, **kwargs)
        vector = query_states.detach() if query_states.ndim == 1 else self.build_query(query_states)
        cosine = [float(F.cosine_similarity(vector[None], s.key.to(vector)[None]).item())
                  for s in self._slots]
        ages = [max(0, native.retrieval_step - item[0]) for item in before]
        selected = select_indices(policy, full_indices=native.retrieved_indices,
                                  cosine=cosine, ages=ages, last_write=last_write, rng_seed=rng_id)
        token_lengths = {int(s.memory.shape[0]) for s in self._slots}
        if len(token_lengths) > 1:
            raise ValueError("unequal slot lengths violate count-matched latent budget")
        # Undo native access metadata before applying the experimental selection.
        for slot, old in zip(self._slots, before):
            slot.last_retrieved_step, slot.last_access_step, slot.access_count, slot.last_score = old
        device = kwargs.get("device") or vector.device
        dtype = kwargs.get("dtype") or vector.dtype
        slots = []
        for index in selected:
            slot = self._slots[index]
            slot.last_retrieved_step = slot.last_access_step = native.retrieval_step
            slot.access_count += 1
            slot.last_score = native.scores[index]
            slots.append(replace(slot, memory=slot.memory.to(device=device, dtype=dtype).detach().clone(),
                                 key=slot.key.to(device=device, dtype=dtype).detach().clone(),
                                 metadata=deepcopy(slot.metadata)))
        latent_count = sum(s.memory.shape[0] for s in slots)
        if latent_count != sum(s.memory.shape[0] for s in native.slots):
            raise ValueError("latent budget mismatch")
        trace.append({"policy": policy, "random_seed": rng_id, "n": len(selected),
                      "selected_indices": selected, "native_indices": list(native.retrieved_indices),
                      "latent_count": latent_count, "cosine": cosine, "ages": ages,
                      "decay": [math.exp(-self.config.decay_alpha * age) for age in ages],
                      "final_scores": list(native.scores), "last_write": list(last_write),
                      "full_overlap": len(set(selected) & set(native.retrieved_indices)),
                      "last_written_overlap": len(set(selected) & set(select_indices(
                          "last_written", full_indices=native.retrieved_indices, cosine=cosine,
                          ages=ages, last_write=last_write, rng_seed=rng_id)))})
        return replace(native, slots=slots, retrieved_indices=tuple(selected),
                       retrieved_scores=tuple(native.scores[i] for i in selected))

    bank.retrieve_with_context = MethodType(retrieve, bank)
    try:
        yield
    finally:
        # Do not leave bound-method attributes: snapshot capture rejects them.
        if had_override:
            bank.retrieve_with_context = original
        else:
            del bank.retrieve_with_context
