"""Sentinel values and helpers for TextPredictionTrainer ``split_col`` modes."""

from __future__ import annotations

from typing import Optional

# Vocabulary-held-out: group by factual token (``split_group_key``) before assigning splits.
HELDOUT_SPLIT_COL = "heldout"


def is_heldout_split_mode(split_col: Optional[str]) -> bool:
    return split_col == HELDOUT_SPLIT_COL


def is_random_resplit_mode(split_col: Optional[str]) -> bool:
    """``split_col=None``: assign train/validation/test by random row shuffle."""
    return split_col is None


def uses_data_split_column(split_col: Optional[str]) -> bool:
    """True when splits are read from a column in the input data."""
    return not is_heldout_split_mode(split_col) and not is_random_resplit_mode(split_col)


def loading_split_col(split_col: Optional[str]) -> Optional[str]:
    """Column passed to unified loaders; ``None`` when the trainer assigns splits."""
    if is_heldout_split_mode(split_col) or is_random_resplit_mode(split_col):
        return None
    return split_col


def data_split_column(split_col: Optional[str], *, default: str = "split") -> str:
    """Physical split column name in merged or per-class tabular data."""
    if uses_data_split_column(split_col):
        return str(split_col)
    return default
