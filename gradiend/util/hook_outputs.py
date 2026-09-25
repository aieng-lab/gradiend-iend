"""Helpers for reading and rewriting the tensor inside a module's (nested) forward output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def first_tensor(value: Any) -> torch.Tensor:
    """Return the first tensor found in a (possibly nested) module output.

    Hugging Face blocks return a tensor, a tuple whose first entry is the hidden state,
    or a ``ModelOutput``-style mapping; this returns that hidden-state tensor.
    """
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            try:
                return first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"Hook output did not contain a tensor, got {type(value).__name__}")


def replace_first_tensor(value: Any, replacement: torch.Tensor) -> Any:
    """Return ``value`` with the tensor selected by :func:`first_tensor` replaced."""
    if torch.is_tensor(value):
        return replacement
    if isinstance(value, tuple):
        out = list(value)
        for i, item in enumerate(out):
            try:
                out[i] = replace_first_tensor(item, replacement)
                return tuple(out)
            except TypeError:
                continue
    if isinstance(value, list):
        out = list(value)
        for i, item in enumerate(out):
            try:
                out[i] = replace_first_tensor(item, replacement)
                return out
            except TypeError:
                continue
    if isinstance(value, dict):
        out = dict(value)
        for key, item in out.items():
            try:
                out[key] = replace_first_tensor(item, replacement)
                return out
            except TypeError:
                continue
    raise TypeError(f"Hook output did not contain a replaceable tensor, got {type(value).__name__}")


__all__ = ["first_tensor", "replace_first_tensor"]
