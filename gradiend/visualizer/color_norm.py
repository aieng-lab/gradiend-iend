"""Shared color normalization helpers for encoded-value visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union


ColorCenter = Union[str, float, int, None]
ColorRange = Union[str, Tuple[float, float], List[float], None]


@dataclass(frozen=True)
class EncodingColorNorm:
    """Resolved diverging color bounds for encoded values."""

    center: float
    vmin: float
    vmax: float
    extent: float

    def normalize(self, value: Any, *, clip: bool = True) -> float:
        """Map a value to ``[-1, 1]`` around ``center``."""
        raw = float(value)
        if raw == self.center:
            scaled = 0.0
        elif raw < self.center:
            denom = max(self.center - self.vmin, 1e-12)
            scaled = -((self.center - raw) / denom)
        else:
            denom = max(self.vmax - self.center, 1e-12)
            scaled = (raw - self.center) / denom
        if clip:
            return max(-1.0, min(1.0, scaled))
        return scaled

    @property
    def legend_label(self) -> str:
        return f"neutral = {self.center:.3g}"


def _flatten_numeric(values: Any) -> List[float]:
    if values is None:
        return []
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "flatten") and not isinstance(values, (str, bytes)):
        try:
            return [float(v) for v in values.flatten().tolist()]
        except Exception:
            pass
    if isinstance(values, dict):
        out: List[float] = []
        for value in values.values():
            out.extend(_flatten_numeric(value))
        return out
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        out = []
        for value in values:
            out.extend(_flatten_numeric(value))
        return out
    if isinstance(values, Real) and not isinstance(values, bool):
        return [float(values)]
    return []


def _resolve_center(
    values: Sequence[float],
    *,
    center: ColorCenter,
    neutral_values: Any = None,
) -> float:
    if center is None or center == "zero":
        return 0.0
    if isinstance(center, Real) and not isinstance(center, bool):
        return float(center)
    if center == "neutral":
        neutral = _flatten_numeric(neutral_values)
        if not neutral:
            raise ValueError("color_center='neutral' requires neutral_values or neutral_value")
        return sum(neutral) / len(neutral)
    if center == "mean":
        if not values:
            return 0.0
        return sum(values) / len(values)
    raise ValueError("color_center must be 'zero', 'neutral', 'mean', a number, or None")


def resolve_encoding_color_norm(
    values: Any,
    *,
    neutral_values: Any = None,
    neutral_value: Optional[float] = None,
    center: ColorCenter = "zero",
    color_range: ColorRange = "symmetric",
    extent: Optional[float] = 1.0,
) -> EncodingColorNorm:
    """
    Resolve diverging encoded-value color bounds.

    ``center='neutral'`` uses the mean of ``neutral_values`` or ``neutral_value`` as
    the neutral color. With ``color_range='symmetric'`` and ``extent=1.0``, the
    displayed bounds are ``center - 1`` and ``center + 1``.
    """
    numeric_values = _flatten_numeric(values)
    if neutral_values is None and neutral_value is not None:
        neutral_values = [neutral_value]
    center_value = _resolve_center(numeric_values, center=center, neutral_values=neutral_values)

    if isinstance(color_range, (tuple, list)):
        if len(color_range) != 2:
            raise ValueError("color_range tuple/list must contain exactly two values")
        vmin = float(color_range[0])
        vmax = float(color_range[1])
    elif color_range in {None, "symmetric"}:
        if extent is None:
            distances = [abs(v - center_value) for v in numeric_values]
            if neutral_values is not None:
                distances.extend(abs(v - center_value) for v in _flatten_numeric(neutral_values))
            resolved_extent = max([1.0, *distances])
        else:
            resolved_extent = float(extent)
        if resolved_extent <= 0:
            raise ValueError("extent must be > 0")
        vmin = center_value - resolved_extent
        vmax = center_value + resolved_extent
    elif color_range == "auto":
        if numeric_values:
            vmin = min(numeric_values)
            vmax = max(numeric_values)
            if vmin == vmax:
                vmin = center_value - 1.0
                vmax = center_value + 1.0
            else:
                vmin = min(vmin, center_value)
                vmax = max(vmax, center_value)
        else:
            vmin = center_value - 1.0
            vmax = center_value + 1.0
    else:
        raise ValueError("color_range must be 'symmetric', 'auto', a (vmin, vmax) pair, or None")

    if not (vmin < center_value < vmax):
        raise ValueError(
            f"Color bounds must contain the center value: vmin={vmin}, center={center_value}, vmax={vmax}"
        )
    return EncodingColorNorm(
        center=float(center_value),
        vmin=float(vmin),
        vmax=float(vmax),
        extent=max(float(center_value - vmin), float(vmax - center_value)),
    )


def diverging_rgb(normalized: float) -> str:
    """Return a blue-white-red hex color for a normalized value in ``[-1, 1]``."""
    value = max(-1.0, min(1.0, float(normalized)))
    if value < 0:
        t = value + 1.0
        start = (55, 126, 184)
        end = (255, 255, 255)
    else:
        t = value
        start = (255, 255, 255)
        end = (215, 48, 39)
    rgb = tuple(round(start[i] + (end[i] - start[i]) * t) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


__all__ = [
    "ColorCenter",
    "ColorRange",
    "EncodingColorNorm",
    "diverging_rgb",
    "resolve_encoding_color_norm",
]
