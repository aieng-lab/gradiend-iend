"""Transition arrow helpers in the main visualizer labels API."""

from __future__ import annotations

import pytest

from gradiend.visualizer.labels import (
    format_transition_label,
    label_contains_matplotlib_latex,
    transition_bidi_arrow,
    transition_directed_arrow,
)


def test_transition_arrows_use_latex_when_requested():
    assert transition_bidi_arrow(use_latex=True) == r"$\rightleftarrows$"
    assert transition_directed_arrow(use_latex=True) == r"$\rightarrow$"
    assert transition_bidi_arrow(use_latex=False) == " ↔ "
    assert transition_directed_arrow(use_latex=False) == " → "


def test_format_transition_label_directed_ascii_without_latex():
    from gradiend.visualizer.plot_style_config import PlotStyleConfig, reset_active_plot_style, set_active_plot_style

    reset_active_plot_style()
    set_active_plot_style(PlotStyleConfig(transition_arrows="ascii"))
    assert format_transition_label("M -> F", use_latex=False) == "M -> F"
    assert format_transition_label("M->F", use_latex=False) == "M -> F"


def test_format_transition_label_directed_with_latex():
    assert format_transition_label("M -> F", use_latex=True) == "M$\\rightarrow$F"


def test_format_transition_label_converts_unicode_arrows_for_latex():
    assert format_transition_label("Neut.Acc ↔ Neut.Dat", use_latex=True) == (
        "Neut.Acc$\\rightleftarrows$Neut.Dat"
    )
    assert format_transition_label("source → target", use_latex=True) == (
        "source$\\rightarrow$target"
    )


def test_format_transition_label_converts_unicode_arrows_for_ascii():
    from gradiend.visualizer.plot_style_config import PlotStyleConfig, reset_active_plot_style, set_active_plot_style

    reset_active_plot_style()
    set_active_plot_style(PlotStyleConfig(transition_arrows="ascii"))
    assert format_transition_label("a ↔ b", use_latex=False) == "a <-> b"
    assert format_transition_label("a → b", use_latex=False) == "a -> b"


def test_format_transition_label_idempotent_for_existing_math():
    label = r"he$\rightleftarrows$she"
    assert format_transition_label(label, use_latex=True) == label
    assert label_contains_matplotlib_latex(label) is True


def test_format_transition_label_follows_matplotlib_usetex(monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    from gradiend.visualizer.plot_style_config import PlotStyleConfig, reset_active_plot_style, set_active_plot_style

    reset_active_plot_style()
    set_active_plot_style(PlotStyleConfig(transition_arrows="auto"))
    monkeypatch.setitem(mpl.rcParams, "text.usetex", True)
    assert format_transition_label("a -> b") == "a$\\rightarrow$b"
    monkeypatch.setitem(mpl.rcParams, "text.usetex", False)
    assert format_transition_label("a -> b") == "a → b"


def test_format_transition_label_respects_ascii_transition_arrows(monkeypatch):
    from gradiend.visualizer.plot_style_config import PlotStyleConfig, reset_active_plot_style, set_active_plot_style

    reset_active_plot_style()
    set_active_plot_style(PlotStyleConfig(transition_arrows="ascii"))
    assert format_transition_label("a -> b") == "a -> b"
