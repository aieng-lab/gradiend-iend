"""Canonical names for convergence / checkpoint-selection metrics.

Users may spell the same metric several ways (``"auroc"``, ``"min_auc"``, ``"E"``).
Every component that compares metric names goes through :func:`normalize_metric_name`
so the alias table exists exactly once.
"""

from __future__ import annotations

from typing import Any, FrozenSet

METRIC_ALIASES = {
    "auroc": "roc_auc",
    "auc": "roc_auc",
    "roc-auc": "roc_auc",
    "min_auc": "min_auc_n_o",
    "auc_min": "min_auc_n_o",
    "roc_auc_min": "min_auc_n_o",
    "min_auc_no": "min_auc_n_o",
    "min(auc_n,auc_o)": "min_auc_n_o",
    "corr": "correlation",
    "e": "encoding_e",
    "encoding-e": "encoding_e",
    "encodinge": "encoding_e",
}

#: Metrics that are AUC-like (self-orienting; rank by raw value, not ``|value|``).
AUC_METRICS: FrozenSet[str] = frozenset({"roc_auc", "min_auc_n_o", "encoding_e"})

#: Metrics whose validation needs one-pole rival factual rows to be encoded.
RIVAL_METRICS: FrozenSet[str] = AUC_METRICS


def normalize_metric_name(name: Any, default: str = "correlation") -> str:
    """Return the canonical metric id for a user-facing name or alias (``default`` if empty)."""
    key = str(name if name else default).strip().lower()
    return METRIC_ALIASES.get(key, key)


def metric_needs_rivals(name: Any) -> bool:
    """Whether periodic validation for ``name`` must encode rival-class factual rows."""
    return normalize_metric_name(name) in RIVAL_METRICS


__all__ = [
    "AUC_METRICS",
    "METRIC_ALIASES",
    "RIVAL_METRICS",
    "metric_needs_rivals",
    "normalize_metric_name",
]
