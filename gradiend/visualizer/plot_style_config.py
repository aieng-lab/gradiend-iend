"""User-facing plot style configuration for GRADIEND matplotlib plots."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional, Union

UseLatexPreference = Union[bool, Literal["auto"]]
TransitionArrowStyle = Literal["auto", "latex", "unicode", "ascii"]

ENV_USE_LATEX = "GRADIEND_PLOT_USE_LATEX"
ENV_FONT_PATH = "GRADIEND_PLOT_FONT_PATH"
ENV_FONT_FAMILY = "GRADIEND_PLOT_FONT_FAMILY"
ENV_LATEX_PREAMBLE_EXTRA = "GRADIEND_PLOT_LATEX_PREAMBLE_EXTRA"
ENV_TRANSITION_ARROWS = "GRADIEND_PLOT_TRANSITION_ARROWS"

_active_config: Optional["PlotStyleConfig"] = None


@dataclass
class PlotStyleConfig:
    """Matplotlib plot rendering options for GRADIEND visualizations.

    GRADIEND always appends ``\\usepackage{amsmath}`` and ``\\usepackage{amssymb}`` when
    ``text.usetex`` is enabled (required for ``\\rightleftarrows`` transition arrows).
    Set **only your extra** packages in ``latex_preamble_extra``.
    """

    use_latex: UseLatexPreference = "auto"
    latex_preamble_extra: str = ""
    font_path: Optional[str] = None
    font_family: Optional[str] = None
    transition_arrows: TransitionArrowStyle = "auto"

    @classmethod
    def from_env(cls) -> "PlotStyleConfig":
        """Build config from ``GRADIEND_PLOT_*`` environment variables."""
        return cls(
            use_latex=_parse_use_latex_env(),
            latex_preamble_extra=os.environ.get(ENV_LATEX_PREAMBLE_EXTRA, "") or "",
            font_path=os.environ.get(ENV_FONT_PATH) or None,
            font_family=os.environ.get(ENV_FONT_FAMILY) or None,
            transition_arrows=_parse_transition_arrows_env(),
        )


@dataclass(frozen=True)
class PlotStyleStatus:
    """Result of :func:`configure_plot_style`."""

    configured: bool
    text_usetex: bool
    latex_usable: bool
    transition_arrows: TransitionArrowStyle


def _parse_use_latex_env() -> UseLatexPreference:
    raw = os.environ.get(ENV_USE_LATEX)
    if raw is None or not str(raw).strip():
        return "auto"
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"auto", "default"}:
        return "auto"
    raise ValueError(f"{ENV_USE_LATEX} must be auto, 1, or 0; got {raw!r}")


def _parse_transition_arrows_env() -> TransitionArrowStyle:
    raw = os.environ.get(ENV_TRANSITION_ARROWS)
    if raw is None or not str(raw).strip():
        return "auto"
    value = str(raw).strip().lower()
    if value in {"auto", "latex", "unicode", "ascii"}:
        return value  # type: ignore[return-value]
    raise ValueError(
        f"{ENV_TRANSITION_ARROWS} must be auto, latex, unicode, or ascii; got {raw!r}"
    )


def get_active_plot_style() -> PlotStyleConfig:
    """Return the active plot style, defaulting to :meth:`PlotStyleConfig.from_env`."""
    global _active_config
    if _active_config is None:
        _active_config = PlotStyleConfig.from_env()
    return _active_config


def set_active_plot_style(config: PlotStyleConfig) -> None:
    """Install *config* as the active plot style for label helpers."""
    global _active_config
    _active_config = config


def reset_active_plot_style() -> None:
    """Reset active config (for tests)."""
    global _active_config
    _active_config = None


def resolve_transition_arrow_mode(*, use_latex: Optional[bool] = None) -> str:
    """Return ``latex``, ``unicode``, or ``ascii`` for transition label arrows."""
    if use_latex is True:
        return "latex"
    if use_latex is False:
        style = get_active_plot_style()
        if style.transition_arrows == "latex":
            return "unicode"
        if style.transition_arrows in {"unicode", "ascii"}:
            return style.transition_arrows
        return "unicode"
    style = get_active_plot_style()
    mode = style.transition_arrows
    if mode == "auto":
        try:
            import matplotlib as mpl

            return "latex" if mpl.rcParams.get("text.usetex") else "unicode"
        except Exception:
            return "unicode"
    if mode == "latex":
        try:
            import matplotlib as mpl

            return "latex" if mpl.rcParams.get("text.usetex") else "unicode"
        except Exception:
            return "unicode"
    return mode
