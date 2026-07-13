import pytest

from gradiend.visualizer.heatmaps import plot_comparison_heatmap


def test_normalized_cross_encoding_positive_expands_default_vmax():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    try:
        plot_comparison_heatmap(
            {
                "measure": "cross_encoding_positive_mean",
                "row_normalized_by_diagonal": True,
                "model_ids": ["a", "b"],
                "matrix": [[1.0, 1.4], [0.6, 1.0]],
            },
            show=False,
            return_data=True,
        )
        mesh = plt.gcf().axes[0].collections[0]
        assert mesh.norm.vmin == pytest.approx(0.0)
        assert mesh.norm.vmax == pytest.approx(1.4)
    finally:
        plt.close("all")


def test_normalized_cross_encoding_difference_uses_symmetric_signed_bounds():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    try:
        plot_comparison_heatmap(
            {
                "measure": "cross_encoding_positive_minus_negative",
                "row_normalized_by_diagonal": True,
                "model_ids": ["a", "b"],
                "matrix": [[1.0, -1.8], [0.25, 1.0]],
            },
            show=False,
            return_data=True,
        )
        mesh = plt.gcf().axes[0].collections[0]
        assert mesh.norm.vmin == pytest.approx(-1.8)
        assert mesh.norm.vmax == pytest.approx(1.8)
    finally:
        plt.close("all")


def test_cross_encoding_difference_uses_data_driven_signed_bounds_without_normalization():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    try:
        plot_comparison_heatmap(
            {
                "measure": "cross_encoding_positive_minus_negative",
                "model_ids": ["a", "b"],
                "matrix": [[0.9, -1.8], [0.25, 1.2]],
            },
            show=False,
            return_data=True,
        )
        mesh = plt.gcf().axes[0].collections[0]
        assert mesh.norm.vmin == pytest.approx(-1.8)
        assert mesh.norm.vmax == pytest.approx(1.8)
    finally:
        plt.close("all")


def test_anchor_aligned_encoding_uses_signed_bounds_and_coolwarm():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    try:
        plot_comparison_heatmap(
            {
                "measure": "anchor_aligned_encoding_factual_mean",
                "model_ids": ["A", "B"],
                "column_ids": ["A", "B"],
                "matrix": [[0.74, -0.23], [0.50, 0.86]],
            },
            show=False,
            return_data=True,
        )
        mesh = plt.gcf().axes[0].collections[0]
        assert mesh.norm.vmin == pytest.approx(-1.0)
        assert mesh.norm.vmax == pytest.approx(1.0)
        assert mesh.cmap.name == "coolwarm"
    finally:
        plt.close("all")


def test_plot_comparison_heatmap_sets_axis_labels():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    try:
        plot_comparison_heatmap(
            {
                "measure": "anchor_aligned_encoding_factual_mean",
                "model_ids": ["A", "B"],
                "column_ids": ["A", "B"],
                "matrix": [[0.74, -0.23], [0.50, 0.86]],
            },
            xlabel="Probe feature",
            ylabel="Orienting feature",
            show=False,
            return_data=True,
        )
        ax = plt.gcf().axes[0]
        assert ax.get_xlabel() == "Probe feature"
        assert ax.get_ylabel() == "Orienting feature"
    finally:
        plt.close("all")


def test_plot_comparison_heatmap_cbar_percent_label_survives_usetex(monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    monkeypatch.setitem(mpl.rcParams, "text.usetex", True)
    try:
        plot_comparison_heatmap(
            {
                "measure": "anchor_aligned_encoding_factual_mean",
                "model_ids": ["A", "B"],
                "column_ids": ["A", "B"],
                "matrix": [[0.74, -0.23], [0.50, 0.86]],
            },
            cbar_label="Encoding (%)",
            show=False,
            return_data=True,
        )
        cbar_ax = plt.gcf().axes[1]
        assert cbar_ax.get_ylabel() == "Encoding (%)"
    finally:
        plt.close("all")


def test_plot_comparison_heatmap_disables_usetex_for_plain_cbar_ticks(monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import gradiend.visualizer.plot_style as plot_style_module

    plot_style_module.reset_matplotlib_style_config()
    monkeypatch.setattr(plot_style_module, "_latex_usable", lambda: True)
    monkeypatch.setitem(mpl.rcParams, "text.usetex", True)
    try:
        plot_comparison_heatmap(
            {
                "measure": "anchor_aligned_encoding_factual_mean",
                "model_ids": ["A", "B"],
                "column_ids": ["A", "B"],
                "matrix": [[0.74, -0.23], [0.50, 0.86]],
            },
            show=False,
            return_data=True,
        )
        cbar_ax = plt.gcf().axes[1]
        assert mpl.rcParams["text.usetex"] is True
        assert cbar_ax.get_yticklabels()
        assert all(label.get_usetex() is False for label in cbar_ax.get_yticklabels())
    finally:
        plt.close("all")


def test_plot_comparison_heatmap_marks_plain_non_converged_labels_when_usetex_enabled(monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from gradiend.visualizer.labels import NON_CONVERGENCE_MARKER
    import gradiend.visualizer.plot_style as plot_style_module

    class _TrainerStub:
        def __init__(self, converged):
            self._converged = converged

        def get_training_stats(self):
            return {"convergence_info": {"converged": self._converged}}

    plot_style_module.reset_matplotlib_style_config()
    monkeypatch.setattr(plot_style_module, "_latex_usable", lambda: True)
    monkeypatch.setitem(mpl.rcParams, "text.usetex", True)
    try:
        _, ax = plot_comparison_heatmap(
            {
                "measure": "anchor_aligned_encoding_factual_mean",
                "model_ids": ["ok", "bad"],
                "column_ids": ["ok", "bad"],
                "matrix": [[1.0, 0.0], [0.0, 1.0]],
            },
            models={"ok": _TrainerStub(True), "bad": _TrainerStub(False)},
            show=False,
            return_fig_ax=True,
            return_data=False,
        )
        ylabels = [label.get_text() for label in ax.get_yticklabels()]
        assert ylabels == ["ok", f"bad {NON_CONVERGENCE_MARKER}"]
        assert all(label.get_usetex() is False for label in ax.get_yticklabels())
    finally:
        plt.close("all")
