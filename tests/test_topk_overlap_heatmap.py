import math

import pytest

from gradiend.visualizer.topk.pairwise_heatmap import (
    plot_topk_overlap_heatmap,
    plot_topk_overlap_heatmap_with_correlation,
)
from gradiend.visualizer.topk.venn_ import compute_topk_sets


class _DummyTopKModel:
    def __init__(self, weights, total_count=None):
        self._weights = list(weights)
        self._total_count = int(total_count) if total_count is not None else len(self._weights)

    def get_topk_weights(self, part="decoder-weight", topk=100):
        if isinstance(topk, float):
            k = max(1, int(math.ceil(topk * self._total_count)))
        else:
            k = int(topk)
        return self._weights[:k]


def _make_models():
    return {
        "A": _DummyTopKModel([1, 2, 3, 4]),
        "B": _DummyTopKModel([3, 4, 5, 6]),
    }


def _make_fractional_models():
    return {
        "A": _DummyTopKModel([1, 2], total_count=200),
        "B": _DummyTopKModel([2, 3, 4], total_count=300),
    }


def test_compute_topk_sets_uses_primary_seed_model_by_default():
    from gradiend.trainer.core.seed_models import SeedModelGroup

    models = {
        "A": SeedModelGroup(
            [
                _DummyTopKModel([1, 2, 3]),
                _DummyTopKModel([9, 10, 11]),
            ],
            seed_values=[42, 43],
        ),
        "B": _DummyTopKModel([2, 3, 4]),
    }

    per_model, intersection, union = compute_topk_sets(models, topk=3)

    assert per_model["A"] == [1, 2, 3]
    assert intersection == [2, 3]
    assert union == [1, 2, 3, 4]


def test_compute_topk_sets_can_union_seed_model_group():
    from gradiend.trainer.core.seed_models import SeedModelGroup

    models = {
        "A": SeedModelGroup(
            [
                _DummyTopKModel([1, 2, 3]),
                _DummyTopKModel([3, 4, 5]),
            ],
            seed_values=[42, 43],
        ),
        "B": _DummyTopKModel([3, 5, 6]),
    }

    per_model, intersection, union = compute_topk_sets(
        models,
        topk=3,
        seed_group_policy="union",
    )

    assert per_model["A"] == [1, 2, 3, 4, 5]
    assert intersection == [3, 5]
    assert union == [1, 2, 3, 4, 5, 6]


def test_overlap_heatmap_percentage_fraction_bounds_match_colorbar():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    try:
        result = plot_topk_overlap_heatmap(
            _make_models(),
            topk=4,
            value="intersection_frac",
            percentages=True,
            vmin=0.0,
            vmax=1.0,
            show=False,
            return_data=True,
        )
        mesh = plt.gcf().axes[0].collections[0]

        assert result["matrix"][0][1] == 50.0
        assert mesh.norm.vmin == 0.0
        assert mesh.norm.vmax == 100.0
    finally:
        plt.close("all")


def test_overlap_heatmap_can_use_existing_axis_and_return_fig_ax():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        data, returned_fig, returned_ax = plot_topk_overlap_heatmap(
            _make_models(),
            topk=4,
            show=False,
            return_data=True,
            return_fig_ax=True,
            ax=ax,
        )

        assert data["matrix"][0][1] == 50.0
        assert returned_fig is fig
        assert returned_ax is ax
        ax.set_title("Custom heatmap")
        assert ax.get_title() == "Custom heatmap"
    finally:
        plt.close("all")


def test_overlap_heatmap_formats_transition_group_labels_like_tick_labels(monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    from gradiend.visualizer.labels import format_transition_label

    monkeypatch.setattr(
        "gradiend.visualizer.heatmaps.base.format_transition_label",
        lambda label: format_transition_label(label, use_latex=True),
    )
    try:
        _, fig, _ = plot_topk_overlap_heatmap(
            _make_models(),
            topk=4,
            pretty_groups={"left ↔ right": ["A", "B"]},
            show=False,
            return_data=True,
            return_fig_ax=True,
        )

        group_labels = [text.get_text() for axis in fig.axes for text in axis.texts]
        assert group_labels.count(r"left$\rightleftarrows$right") == 2
    finally:
        plt.close("all")


def test_overlap_heatmap_marks_pretty_labels_from_explicit_convergence_map():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    from gradiend.visualizer.labels import NON_CONVERGENCE_MARKER

    try:
        _, ax = plot_topk_overlap_heatmap(
            {
                "run_ok": _DummyTopKModel([1, 2, 3, 4]),
                "run_bad": _DummyTopKModel([3, 4, 5, 6]),
            },
            topk=4,
            order=["run_ok", "run_bad"],
            row_label_mapping={"run_ok": "Pretty OK", "run_bad": "Pretty bad"},
            column_label_mapping={"run_ok": "Pretty OK", "run_bad": "Pretty bad"},
            converged_by_id={"run_ok": True, "run_bad": False},
            show=False,
            return_data=False,
            return_fig_ax=True,
        )

        assert [label.get_text() for label in ax.get_yticklabels()] == [
            "Pretty OK",
            f"Pretty bad {NON_CONVERGENCE_MARKER}",
        ]
        assert [label.get_text() for label in ax.get_xticklabels()] == [
            "Pretty OK",
            f"Pretty bad {NON_CONVERGENCE_MARKER}",
        ]
    finally:
        plt.close("all")


def test_overlap_heatmap_cbar_y_pad_moves_colorbar_relative_to_heatmap():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    try:
        fig, ax = plot_topk_overlap_heatmap(
            _make_models(),
            topk=4,
            cbar_shrink=0.2,
            cbar_y_pad=-0.25,
            show=False,
            return_data=False,
            return_fig_ax=True,
        )
        cbar_ax = fig.axes[1]
        heatmap_height = ax.get_position().height
        centered_cbar_y = ax.get_position().y0 + (heatmap_height - cbar_ax.get_position().height) / 2

        assert cbar_ax.get_position().y0 == pytest.approx(
            centered_cbar_y - 0.25 * heatmap_height,
            abs=0.02,
        )
    finally:
        plt.close("all")


def test_overlap_heatmap_percentage_count_bounds_match_colorbar():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    try:
        result = plot_topk_overlap_heatmap(
            _make_models(),
            topk=4,
            value="intersection",
            percentages=True,
            vmin=0.0,
            vmax=4.0,
            show=False,
            return_data=True,
        )
        mesh = plt.gcf().axes[0].collections[0]

        assert result["matrix"][0][1] == 50.0
        assert mesh.norm.vmin == 0.0
        assert mesh.norm.vmax == 100.0
    finally:
        plt.close("all")


def test_overlap_heatmap_with_correlation_accepts_annot_fmt_when_metrics_missing(monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(
        "gradiend.visualizer.topk.pairwise_heatmap._extract_best_correlation_for_models",
        lambda models: {},
    )

    try:
        result = plot_topk_overlap_heatmap_with_correlation(
            _make_models(),
            topk=4,
            annot_fmt=".0f",
            show=False,
            return_data=True,
        )

        assert result["matrix"][0][1] == pytest.approx(0.5)
    finally:
        plt.close("all")


def test_overlap_heatmap_fractional_topk_uses_actual_resolved_set_sizes():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    result = plot_topk_overlap_heatmap(
        _make_fractional_models(),
        topk=0.01,
        value="intersection_frac",
        percentages=False,
        show=False,
        return_data=True,
    )

    assert result["resolved_topk"] == {"A": 2, "B": 3}
    assert result["matrix"][0][1] == pytest.approx(0.5)


def test_overlap_heatmap_fractional_topk_percentages_use_pairwise_smaller_set():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    result = plot_topk_overlap_heatmap(
        _make_fractional_models(),
        topk=0.01,
        value="intersection",
        percentages=True,
        show=False,
        return_data=True,
    )

    assert result["matrix"][0][1] == pytest.approx(50.0)


def test_overlap_heatmap_rejects_custom_count_bounds_for_nonuniform_percentage_sets():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    with pytest.raises(ValueError, match="Custom vmin/vmax"):
        plot_topk_overlap_heatmap(
            _make_fractional_models(),
            topk=0.01,
            value="intersection",
            percentages=True,
            vmin=0.0,
            vmax=2.0,
            show=False,
            return_data=True,
        )
