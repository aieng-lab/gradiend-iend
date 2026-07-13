"""
Matplotlib style defaults for GRADIEND plots.

Environment variables (see :class:`~gradiend.visualizer.plot_style_config.PlotStyleConfig`):

``GRADIEND_PLOT_USE_LATEX``
    ``auto`` (default): enable ``text.usetex`` when LaTeX is usable.
    ``1``/``true``: force LaTeX (falls back with a warning if unavailable).
    ``0``/``false``: disable LaTeX.

``GRADIEND_PLOT_FONT_PATH``
    Path to a ``.ttf``/``.otf`` font file for ``font.family``.

``GRADIEND_PLOT_FONT_FAMILY``
    Explicit Matplotlib font family, for example ``sans-serif``. This is useful
    for paper plots that need LaTeX symbols while keeping sans-serif typography.

``GRADIEND_PLOT_LATEX_PREAMBLE_EXTRA``
    Extra LaTeX preamble lines (GRADIEND always adds ``amsmath`` and ``amssymb`` when usetex is on).

``GRADIEND_PLOT_TRANSITION_ARROWS``
    ``auto`` (default), ``latex``, ``unicode``, or ``ascii`` for ``A -> B`` labels.

Primary entry point: :func:`configure_plot_style`.
"""

from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

from gradiend.util.logging import get_logger
from gradiend.visualizer.plot_style_config import (
    ENV_FONT_FAMILY,
    ENV_FONT_PATH,
    ENV_LATEX_PREAMBLE_EXTRA,
    ENV_TRANSITION_ARROWS,
    ENV_USE_LATEX,
    PlotStyleConfig,
    PlotStyleStatus,
    get_active_plot_style,
    reset_active_plot_style,
    set_active_plot_style,
)

logger = get_logger(__name__)

_LATEX_COMMANDS = ("latex", "pdflatex", "xelatex", "lualatex")
_CONFIGURED = False
_LATEX_USABLE: Optional[bool] = None


def _latex_on_path() -> bool:
    return any(shutil.which(command) for command in _LATEX_COMMANDS)


def _latex_command_paths() -> Dict[str, Optional[str]]:
    return {command: shutil.which(command) for command in _LATEX_COMMANDS}


def _ensure_latex_symbol_preamble() -> None:
    """Ensure matplotlib's LaTeX preamble includes packages for GRADIEND arrow glyphs."""
    import matplotlib as mpl

    preamble = str(mpl.rcParams.get("text.latex.preamble", "") or "")
    additions: list[str] = []
    if "amsmath" not in preamble:
        additions.append(r"\usepackage{amsmath}")
    if "amssymb" not in preamble:
        additions.append(r"\usepackage{amssymb}")
    if not additions:
        return
    mpl.rcParams["text.latex.preamble"] = (preamble + "\n" + "\n".join(additions)).strip()
    _clear_texmanager_cache()


def _ensure_amsmath_preamble() -> None:
    """Backward-compatible alias for :func:`_ensure_latex_symbol_preamble`."""
    _ensure_latex_symbol_preamble()


def _apply_latex_preamble_extra(extra: str) -> None:
    extra = str(extra or "").strip()
    if not extra:
        return
    import matplotlib as mpl

    preamble = str(mpl.rcParams.get("text.latex.preamble", "") or "")
    if extra in preamble:
        return
    mpl.rcParams["text.latex.preamble"] = (preamble + "\n" + extra).strip()
    _clear_texmanager_cache()


def _clear_texmanager_cache() -> None:
    try:
        from matplotlib.texmanager import TexManager

        cache = getattr(TexManager, "_texcache", None)
        if isinstance(cache, dict):
            cache.clear()
    except Exception:
        pass


def _latex_usable() -> bool:
    global _LATEX_USABLE
    if _LATEX_USABLE is not None:
        return _LATEX_USABLE
    if not _latex_on_path():
        _LATEX_USABLE = False
        return False
    fig = None
    plt = None
    try:
        import tempfile

        import matplotlib as mpl
        from matplotlib import pyplot as plt
        from matplotlib.texmanager import TexManager

        _ensure_latex_symbol_preamble()
        TexManager()

        with mpl.rc_context({"text.usetex": True}):
            fig, ax = plt.subplots(figsize=(1, 1))
            from gradiend.visualizer.labels import transition_bidi_arrow

            # Group labels in comparison heatmaps automatically grow to 14 pt.
            # TeX may map that size to a different font (for example tcss1440)
            # than the default 12 pt title, and incomplete TeX installations can
            # therefore pass a default-size probe but fail later at savefig.
            ax.set_title(
                f"Test {transition_bidi_arrow(use_latex=True)}",
                fontsize=14,
            )
            with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
                fig.savefig(tmp.name, bbox_inches="tight")
        _LATEX_USABLE = True
    except Exception as exc:
        logger.debug("LaTeX found on PATH but matplotlib usetex smoke test failed: %s", exc)
        _LATEX_USABLE = False
    finally:
        if fig is not None and plt is not None:
            plt.close(fig)
    return _LATEX_USABLE


def _resolve_use_latex(config: PlotStyleConfig) -> bool:
    preference = config.use_latex
    if preference is False:
        return False
    if preference is True:
        if _latex_usable():
            return True
        logger.warning(
            "%s=true but LaTeX is not usable; continuing without usetex",
            ENV_USE_LATEX,
        )
        return False
    return _latex_usable()


def _apply_custom_font(font_path: str) -> None:
    from matplotlib import font_manager

    path = Path(font_path).expanduser()
    if not path.is_file():
        logger.warning("%s is not a file: %s", ENV_FONT_PATH, path)
        return
    resolved = str(path.resolve())
    font_manager.fontManager.addfont(resolved)
    name = font_manager.FontProperties(fname=resolved).get_name()
    import matplotlib as mpl

    mpl.rcParams["font.family"] = name
    sans = mpl.rcParams.get("font.sans-serif", [])
    if isinstance(sans, str):
        sans = [sans]
    mpl.rcParams["font.sans-serif"] = [name, *[f for f in sans if f != name]]
    logger.info("Using plot font %r from %s", name, resolved)


def _apply_font_family(font_family: str) -> None:
    import matplotlib as mpl

    family = str(font_family or "").strip()
    if family:
        mpl.rcParams["font.family"] = family


def _prefer_tex_safe_serif_font() -> None:
    """Avoid TeX sans-serif font-map lookups that are often missing on clusters."""
    import matplotlib as mpl

    family = mpl.rcParams.get("font.family")
    families = [family] if isinstance(family, str) else list(family or [])
    normalized = {str(value).lower() for value in families}
    if normalized and normalized != {"sans-serif"}:
        return
    mpl.rcParams["font.family"] = "serif"


def configure_plot_style(
    config: Optional[PlotStyleConfig] = None,
    *,
    force: bool = False,
) -> PlotStyleStatus:
    """Apply GRADIEND matplotlib defaults (idempotent unless *force*).

    Args:
        config: Style options. Defaults to :meth:`PlotStyleConfig.from_env` on first call.
        force: Re-apply usetex, font, and preamble even when already configured.

    Returns:
        Summary of the active matplotlib text settings.
    """
    global _CONFIGURED
    import matplotlib as mpl

    style = config or get_active_plot_style()
    set_active_plot_style(style)

    if not _CONFIGURED or force:
        mpl.rcParams["text.usetex"] = bool(_resolve_use_latex(style))

        font_family = style.font_family
        font_path = style.font_path
        if font_family and str(font_family).strip():
            _apply_font_family(str(font_family).strip())
        elif font_path and str(font_path).strip():
            _apply_custom_font(str(font_path).strip())
        elif mpl.rcParams.get("text.usetex"):
            _prefer_tex_safe_serif_font()

        _CONFIGURED = True

    if mpl.rcParams.get("text.usetex"):
        _ensure_latex_symbol_preamble()
        _apply_latex_preamble_extra(style.latex_preamble_extra)

    return PlotStyleStatus(
        configured=_CONFIGURED,
        text_usetex=bool(mpl.rcParams.get("text.usetex")),
        latex_usable=_latex_usable(),
        transition_arrows=style.transition_arrows,
    )


def configure_matplotlib_style(*, force: bool = False) -> PlotStyleStatus:
    """Apply GRADIEND matplotlib defaults (deprecated alias).

    Prefer :func:`configure_plot_style`.
    """
    warnings.warn(
        "configure_matplotlib_style() is deprecated; use configure_plot_style() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return configure_plot_style(force=force)


def _check_configured_font(font_path: Optional[str]) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "env_value": font_path,
        "path": None,
        "exists": False,
        "is_file": False,
        "valid_extension": None,
        "usable": font_path is None or not str(font_path).strip(),
        "font_name": None,
        "error": None,
    }
    if font_path is None or not str(font_path).strip():
        return status

    path = Path(str(font_path).strip()).expanduser()
    status["path"] = str(path)
    status["exists"] = path.exists()
    status["is_file"] = path.is_file()
    status["valid_extension"] = path.suffix.lower() in {".ttf", ".otf"}
    if not path.is_file():
        status["usable"] = False
        status["error"] = "Configured font path is not a file."
        return status
    if not status["valid_extension"]:
        status["usable"] = False
        status["error"] = "Configured font path should point to a .ttf or .otf file."
        return status

    try:
        from matplotlib import font_manager

        status["font_name"] = font_manager.FontProperties(fname=str(path.resolve())).get_name()
        status["path"] = str(path.resolve())
        status["usable"] = True
    except Exception as exc:
        status["usable"] = False
        status["error"] = str(exc)
    return status


def _font_family_text(font_family: Any) -> str:
    if isinstance(font_family, (list, tuple)):
        return ", ".join(str(value) for value in font_family)
    return str(font_family)


def _check_current_font() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "family": None,
        "available": False,
        "resolved_path": None,
        "error": None,
    }
    try:
        import matplotlib as mpl
        from matplotlib import font_manager

        family = mpl.rcParams.get("font.family")
        status["family"] = family
        font_path = font_manager.findfont(
            font_manager.FontProperties(family=family),
            fallback_to_default=True,
        )
        status["resolved_path"] = font_path
        status["available"] = bool(font_path)
    except Exception as exc:
        status["available"] = False
        status["error"] = str(exc)
    return status


def _format_plot_environment_status(status: Dict[str, Any]) -> str:
    latex = status["latex"]
    font = status["font"]
    mpl = status["matplotlib"]
    style = status.get("style", {})
    command_names = [name for name, path in latex["commands"].items() if path]
    command_text = ", ".join(command_names) if command_names else "none"
    font_env_text = font["env_value"] if font.get("env_value") else "unset"
    family_env_text = font.get("family_env_value") if font.get("family_env_value") else "unset"
    configured_font_text = (
        f"configured={font['font_name']} ({font['path']})"
        if font.get("font_name")
        else "configured=default matplotlib font"
        if font.get("usable")
        else f"configured=unusable ({font.get('error')})"
    )
    current_font = font["current"]
    current_font_text = (
        f"current={_font_family_text(current_font['family'])}, "
        f"current_available={current_font['available']}"
    )
    if current_font.get("resolved_path"):
        current_font_text += f", resolved={current_font['resolved_path']}"
    lines = [
        f"GRADIEND plot environment: {'OK' if status['ok'] else 'ISSUES'}",
        (
            "  LaTeX: "
            f"preference={latex['preference']}, on_path={latex['on_path']}, "
            f"usable={latex['usable']}, resolved_text_usetex={latex['resolved_text_usetex']}, "
            f"commands={command_text}"
        ),
        (
            "  Style: "
            f"transition_arrows={style.get('transition_arrows', 'auto')}, "
            f"{ENV_FONT_FAMILY}={family_env_text}, "
            f"preamble_extra={'set' if style.get('latex_preamble_extra') else 'unset'}"
        ),
        f"  Font: usable={font['usable']}, {font['env_var']}={font_env_text}, "
        f"{configured_font_text}, {current_font_text}",
        (
            "  Matplotlib: "
            f"available={mpl['available']}, version={mpl['version']}, "
            f"style_configured={mpl['style_configured']}, "
            f"current_text_usetex={mpl['current_text_usetex']}"
        ),
    ]
    if status["warnings"]:
        lines.append("  Warnings:")
        lines.extend(f"    - {warning}" for warning in status["warnings"])
    if status["info"]:
        lines.append("  Info:")
        lines.extend(f"    - {message}" for message in status["info"])
    return "\n".join(lines)


def check_plot_environment(*, print_status: bool = True, apply_style: bool = True) -> Dict[str, Any]:
    """Print and return GRADIEND plot rendering environment status.

    When ``apply_style=True``, calls :func:`configure_plot_style` with
    :meth:`PlotStyleConfig.from_env`.
    """
    env_config: PlotStyleConfig
    latex_preference_error = None
    try:
        env_config = PlotStyleConfig.from_env()
    except ValueError as exc:
        env_config = PlotStyleConfig()
        latex_preference_error = str(exc)

    command_paths = _latex_command_paths()
    latex_on_path = any(path for path in command_paths.values())
    latex_usable = _latex_usable() if latex_on_path else False
    resolved_usetex = _resolve_use_latex(env_config) if latex_preference_error is None else False

    font_status = _check_configured_font(env_config.font_path)

    matplotlib_status: Dict[str, Any] = {
        "available": False,
        "version": None,
        "current_text_usetex": None,
        "style_configured": _CONFIGURED,
        "current_font_family": None,
        "current_font_sans_serif": None,
        "error": None,
    }

    def _refresh_matplotlib_status() -> None:
        try:
            import matplotlib as mpl

            matplotlib_status.update(
                {
                    "available": True,
                    "version": getattr(mpl, "__version__", None),
                    "current_text_usetex": bool(mpl.rcParams.get("text.usetex")),
                    "style_configured": _CONFIGURED,
                    "current_font_family": mpl.rcParams.get("font.family"),
                    "current_font_sans_serif": mpl.rcParams.get("font.sans-serif"),
                    "error": None,
                }
            )
        except Exception as exc:
            matplotlib_status["error"] = str(exc)

    _refresh_matplotlib_status()

    style_status: Optional[PlotStyleStatus] = None
    if apply_style and matplotlib_status["available"] and latex_preference_error is None:
        try:
            style_status = configure_plot_style(env_config, force=False)
            _refresh_matplotlib_status()
        except Exception as exc:
            font_status["usable"] = False
            font_status["error"] = str(exc)
            _refresh_matplotlib_status()

    current_font_status = _check_current_font() if matplotlib_status["available"] else {
        "family": None,
        "available": False,
        "resolved_path": None,
        "error": "matplotlib is not available.",
    }
    font_status["current"] = current_font_status
    pref = env_config.use_latex
    forced_latex_but_unusable = pref is True and not latex_usable
    active_usetex = matplotlib_status["current_text_usetex"]
    usetex_mismatch = matplotlib_status["available"] and active_usetex != resolved_usetex
    active_family = current_font_status.get("family")
    configured_font_name = font_status.get("font_name")
    font_style_not_applied = bool(
        font_status["usable"]
        and configured_font_name
        and not apply_style
        and not matplotlib_status["style_configured"]
    )
    expected_font_missing = bool(
        font_status["usable"]
        and configured_font_name
        and matplotlib_status["style_configured"]
        and configured_font_name not in (
            active_family if isinstance(active_family, (list, tuple)) else [active_family]
        )
    )
    font_unavailable = matplotlib_status["available"] and not current_font_status["available"]
    ok = (
        latex_preference_error is None
        and matplotlib_status["available"]
        and font_status["usable"]
        and not font_unavailable
        and not expected_font_missing
        and not forced_latex_but_unusable
        and not (usetex_mismatch and apply_style)
    )
    warnings_list = []
    info = []
    if latex_preference_error is not None:
        warnings_list.append(latex_preference_error)
    if forced_latex_but_unusable:
        warnings_list.append(f"{ENV_USE_LATEX}=true but LaTeX is not usable by matplotlib.")
    if env_config.font_path and str(env_config.font_path).strip() and not font_status["usable"]:
        warnings_list.append(str(font_status["error"]))
    if font_unavailable:
        warnings_list.append(
            "Matplotlib cannot resolve the current font.family "
            f"({_font_family_text(current_font_status['family'])}): {current_font_status['error']}"
        )
    if font_style_not_applied:
        info.append(
            "GRADIEND style has not been applied in this process yet; "
            f"plots will register {ENV_FONT_PATH} when rendering."
        )
    if expected_font_missing:
        warnings_list.append(
            f"{ENV_FONT_PATH} resolved to {configured_font_name!r}, but matplotlib is currently "
            f"using {_font_family_text(active_family)!r}."
        )
    if not matplotlib_status["available"]:
        warnings_list.append(f"matplotlib is not available: {matplotlib_status['error']}")
    if usetex_mismatch:
        if not apply_style and not matplotlib_status["style_configured"]:
            info.append(
                "GRADIEND style has not been applied in this process yet; "
                f"plots will set text.usetex={resolved_usetex} when rendering."
            )
        else:
            warnings_list.append(
                "Matplotlib text.usetex does not match the GRADIEND LaTeX preference "
                f"(current={active_usetex}, resolved={resolved_usetex})."
            )

    latex_pref_label = (
        "force_on"
        if pref is True
        else "force_off"
        if pref is False
        else "auto"
    )

    status = {
        "ok": ok,
        "warnings": warnings_list,
        "info": info,
        "style": {
            "transition_arrows": env_config.transition_arrows,
            "font_family": env_config.font_family,
            "font_family_env_var": ENV_FONT_FAMILY,
            "font_family_env_value": os.environ.get(ENV_FONT_FAMILY),
            "latex_preamble_extra": bool((env_config.latex_preamble_extra or "").strip()),
            "configured": style_status.configured if style_status else _CONFIGURED,
            "text_usetex": style_status.text_usetex if style_status else active_usetex,
        },
        "latex": {
            "env_var": ENV_USE_LATEX,
            "env_value": os.environ.get(ENV_USE_LATEX),
            "preference": latex_pref_label,
            "preference_error": latex_preference_error,
            "commands": command_paths,
            "on_path": latex_on_path,
            "usable": latex_usable,
            "resolved_text_usetex": resolved_usetex,
        },
        "font": {
            "env_var": ENV_FONT_PATH,
            "family_env_var": ENV_FONT_FAMILY,
            "family_env_value": os.environ.get(ENV_FONT_FAMILY),
            **font_status,
        },
        "matplotlib": matplotlib_status,
    }
    if print_status:
        print(_format_plot_environment_status(status))
    return status


def disable_usetex_for_axis_text(ax: Any = None) -> None:
    """Render plain tick/group labels literally; keep usetex on intentional math."""
    import matplotlib as mpl

    from gradiend.visualizer.labels import label_contains_matplotlib_latex

    if not mpl.rcParams.get("text.usetex"):
        return
    if ax is None:
        return
    global_usetex = bool(mpl.rcParams.get("text.usetex"))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_usetex(
            global_usetex if label_contains_matplotlib_latex(label.get_text()) else False
        )
    for text in ax.texts:
        text.set_usetex(
            global_usetex if label_contains_matplotlib_latex(text.get_text()) else False
        )
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_usetex(
            global_usetex if label_contains_matplotlib_latex(label.get_text()) else False
        )
    title = ax.get_title()
    if title:
        ax.set_title(title, usetex=label_contains_matplotlib_latex(title))


def reset_matplotlib_style_config() -> None:
    """Reset one-time configuration (for tests)."""
    global _CONFIGURED, _LATEX_USABLE
    _CONFIGURED = False
    _LATEX_USABLE = None
    reset_active_plot_style()
