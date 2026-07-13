"""Shared helpers for pre-prune mask vs trained-oracle recall experiments."""

from __future__ import annotations

import math
from typing import Any, List, Set, Tuple

import torch

TOPK_EVAL = 1000
TOPK_PART = "decoder-weight"
DEFAULT_MIN_PRE_TOPK = 1e-6


def dense_pre_topk_grid(*, min_topk: float = DEFAULT_MIN_PRE_TOPK) -> List[float]:
    """Dense pre_topk grid: 1.0 plus 3×10^n and 10^n down to ``min_topk``."""
    if not (0.0 < min_topk <= 1.0):
        raise ValueError(f"min_topk must be in (0, 1], got {min_topk!r}")
    values = [1.0]
    min_exp = int(math.floor(math.log10(min_topk)))
    for exponent in range(-1, min_exp - 1, -1):
        decade = 10.0**exponent
        values.append(3.0 * decade)
        values.append(decade)
    return sorted(
        {float(v) for v in values if v + 1e-15 >= min_topk},
        reverse=True,
    )


def ref_recall_metrics(heuristic: Set[int], ref: Set[int]) -> Tuple[float, float]:
    """Recall/precision vs a fixed-size oracle top-k (``TOPK_EVAL``)."""
    if not ref:
        return 0.0, 0.0
    inter = len(heuristic & ref)
    recall = inter / TOPK_EVAL
    precision = inter / len(heuristic) if heuristic else 0.0
    return recall, precision


def topk_base_global(model: Any, *, topk: int, part: str) -> Set[int]:
    local_indices = model.gradiend.get_topk_weights(part=part, topk=topk)
    if not local_indices:
        return set()
    base_map = model.gradiend._get_base_global_index_map()
    idx_t = torch.as_tensor(local_indices, dtype=torch.long)
    return {int(x) for x in base_map[idx_t].tolist()}


def all_kept_base_global(model: Any) -> Set[int]:
    """Map every kept GRADIEND input dim to base-global indices."""
    local_indices = torch.arange(int(model.gradiend.input_dim), dtype=torch.long)
    base_map = model.gradiend._get_base_global_index_map()
    return {int(x) for x in base_map[local_indices].tolist()}
