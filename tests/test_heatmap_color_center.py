"""Tests for neutral-centered heatmap color normalization."""

from __future__ import annotations

import pytest


def test_plot_comparison_heatmap_accepts_neutral_color_center():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    from matplotlib import pyplot as plt

    from gradiend.visualizer.heatmaps import plot_comparison_heatmap

    _, ax = plot_comparison_heatmap(
        {
            "matrix": [[-0.6, 0.4, 1.4]],
            "model_ids": ["row"],
            "column_ids": ["low", "neutral", "high"],
            "measure": "gradiend_transition_cross_encoding_mean",
        },
        color_center="neutral",
        neutral_value=0.4,
        color_extent=1.0,
        show=False,
        return_data=False,
        return_fig_ax=True,
    )

    mesh = ax.collections[0]
    assert mesh.norm.vcenter == pytest.approx(0.4)
    assert mesh.norm.vmin == pytest.approx(-0.6)
    assert mesh.norm.vmax == pytest.approx(1.4)
    plt.close("all")


def test_plot_comparison_heatmap_rejects_center_with_log_scale():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    from gradiend.visualizer.heatmaps import plot_comparison_heatmap

    with pytest.raises(ValueError, match="color_center"):
        plot_comparison_heatmap(
            {"matrix": [[1.0]], "model_ids": ["row"], "column_ids": ["col"]},
            color_center=0.4,
            scale="log",
            show=False,
        )
