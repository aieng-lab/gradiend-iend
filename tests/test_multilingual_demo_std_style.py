"""Plot-style regression tests for multilingual cross-encoding seed dispersion."""

from gradiend.visualizer.multilingual_demo_labels import (
    demo_encoding_std_heatmap_style_kwargs,
)


def test_cross_encoding_std_style_uses_sequential_scaled_range():
    base_style = {
        "percentages": True,
        "cmap": "coolwarm",
        "vmin": -1.0,
        "vmax": 1.0,
        "annot_fmt": ".1f",
        "cbar_y_pad": -0.62,
        "cbar_label": "Encoding (%)",
    }

    std_style = demo_encoding_std_heatmap_style_kwargs(base_style)

    assert std_style["percentages"] is True
    assert std_style["cmap"] == "Reds"
    assert "vmin" not in std_style
    assert "vmax" not in std_style
    assert "fmt" not in std_style
    assert "annot_fmt" not in std_style
    assert std_style["cbar_y_pad"] == -0.62
    assert std_style["cbar_label"] == "Seed std. (%)"
    assert base_style["cmap"] == "coolwarm"
