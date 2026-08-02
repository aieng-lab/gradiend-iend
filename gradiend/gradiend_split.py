"""GRADIEND component split configuration and resolution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass(frozen=True)
class GradiendSplit:
    """
    User-facing component split configuration for one resolved GRADIEND input space.

    ``Signal`` and ``SignalScope`` decide which measurements exist. ``GradiendSplit``
    only decides how the already-resolved, flattened signal coordinates are grouped
    into virtual GRADIEND components.

    The split is therefore deliberately signal-agnostic. A split mode should not
    decide whether the model measures gradients, activations, or a future
    attribution signal, and it should not decide which layers/modules/parameters are
    eligible. It receives the ordered signal-space mapping produced by the resolver
    and turns that mapping into non-overlapping component slices.
    """

    mode: str = "none"

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {"none", "single", "tensors"}:
            raise ValueError(
                "GradiendSplit.mode must be 'none', 'single', or 'tensors', "
                f"got {self.mode!r}"
            )
        object.__setattr__(self, "mode", mode)

    @classmethod
    def none(cls) -> "GradiendSplit":
        """Use a standard, unpartitioned GradiendModel."""
        return cls("none")

    @classmethod
    def single(cls) -> "GradiendSplit":
        """Use partitioned GRADIEND behavior with one full input-space partition."""
        return cls("single")

    @classmethod
    def by_tensor(cls) -> "GradiendSplit":
        """
        Use one component per resolved tensor entry in the selected signal space.

        Here, "tensor" means the natural tensor-valued unit emitted by the
        signal-space resolver after applying the selected ``Signal`` and
        ``SignalScope``. This split does not select a signal and does not select a
        scope; it only partitions whatever signal space was already selected.

        Examples:

        - For ``Signal.gradient()``, the resolved tensor entries are selected base
          model parameter tensors. ``by_tensor()`` therefore trains one virtual
          component per included parameter tensor, not one component per scalar
          weight.
        - For ``Signal.activation(...)``, the resolved tensor entries are selected
          activation sources/sites after token selection and flattening metadata have
          been applied. ``by_tensor()`` therefore trains one virtual component per
          included activation tensor/site, not one component per upstream parameter
          tensor.

        This is the lowest-level generic split currently exposed. Higher-level
        semantic splits such as "by layer" should be implemented as separate
        resolver/grouping modes instead of overloading ``by_tensor()``.

        Use ``GradiendSplit.single()`` when the full resolved signal space should be
        represented as one component, and ``GradiendSplit.none()`` when no
        partitioned-component behavior should be used at all.
        """
        return cls("tensors")

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GradiendSplit":
        if not isinstance(data, Mapping):
            raise TypeError(f"GradiendSplit.from_dict expected mapping, got {type(data).__name__}")
        return cls(str(data.get("mode") or "none"))


def coerce_gradiend_split(value: Any) -> Optional[GradiendSplit]:
    """Convert public split shorthand into a GradiendSplit."""
    if value is None:
        return None
    if isinstance(value, GradiendSplit):
        return value
    if isinstance(value, str):
        return GradiendSplit(value)
    if isinstance(value, Mapping):
        return GradiendSplit.from_dict(value)
    raise TypeError(
        "gradiend_split must be GradiendSplit, str, dict, or None, "
        f"got {type(value).__name__}"
    )


def _num_selected_from_spec(spec: Mapping[str, Any]) -> int:
    shape = tuple(int(dim) for dim in spec.get("shape", ()))
    repr_kind = str(spec.get("repr") or "all")
    if repr_kind == "all":
        return int(math.prod(shape))
    if repr_kind == "mask":
        mask = spec.get("mask")
        if not torch.is_tensor(mask):
            raise TypeError("param_map mask spec requires a tensor-valued 'mask'")
        return int(mask.to(dtype=torch.bool).sum().item())
    if repr_kind == "indices":
        indices = spec.get("indices")
        if not torch.is_tensor(indices):
            raise TypeError("param_map indices spec requires a tensor-valued 'indices'")
        return int(indices.numel())
    if repr_kind == "empty":
        return 0
    raise ValueError(f"Unknown param_map repr {repr_kind!r}")


def resolve_gradiend_components(
    param_map: Mapping[str, Mapping[str, Any]],
    split: Any,
    *,
    input_dim: Optional[int] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    Resolve component slices from an ordered signal-space mapping.

    The mapping intentionally has the same shape as the historic ``param_map``,
    but callers may pass activation-space or future signal-space entries as long as
    each value exposes the same selected-width metadata. Component resolution is
    purely over the flattened signal space and must not inspect or infer the signal
    kind.
    """
    normalized = coerce_gradiend_split(split)
    if normalized is None or normalized.mode == "none":
        return None
    if normalized.mode == "single":
        if input_dim is None:
            total = sum(_num_selected_from_spec(spec) for spec in param_map.values())
        else:
            total = int(input_dim)
        return [{"id": "full", "start": 0, "end": total}]

    if not isinstance(param_map, Mapping):
        raise TypeError(f"param_map must be a mapping, got {type(param_map).__name__}")

    components: List[Dict[str, Any]] = []
    offset = 0
    for name, spec in param_map.items():
        width = _num_selected_from_spec(spec)
        if width > 0:
            components.append({"id": str(name), "start": offset, "end": offset + width})
        offset += width

    if input_dim is not None and offset != int(input_dim):
        raise ValueError(
            f"Resolved component widths sum to {offset}, but input_dim={int(input_dim)}"
        )
    if not components:
        raise ValueError("GradiendSplit produced no non-empty components")
    return components


__all__ = [
    "GradiendSplit",
    "coerce_gradiend_split",
    "resolve_gradiend_components",
]
