import dataclasses
import math

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def ablation():
    from experiments import gender_de_pre_prune_topk_ablation as mod

    return mod


def test_default_mode_is_mask_recall(ablation):
    args = ablation.build_arg_parser().parse_args([])
    assert not args.full_grid


def test_recall_only_flag_is_accepted(ablation):
    args = ablation.build_arg_parser().parse_args(["--recall-only"])
    assert not args.full_grid


def test_default_grid_schedules_dense_pre_topk(ablation):
    from experiments.pre_prune_mask_recall import dense_pre_topk_grid

    expected = dense_pre_topk_grid()
    args = ablation.build_arg_parser().parse_args([])

    assert args.pre_topk_values == expected
    assert args.pre_topk_values[0] == pytest.approx(1.0)
    assert args.pre_topk_values[-1] == pytest.approx(1e-6)
    missing = ablation._missing_grid_cells(
        [],
        pairs=[ablation.DER_DIE_PAIRS[0]],
        sources=["diff"],
        n_samples_values=[16],
        pre_topk_values=args.pre_topk_values,
    )
    assert any(cell["pre_topk"] == pytest.approx(0.3) for cell in missing)
    assert any(cell["pre_topk"] == pytest.approx(1e-6) for cell in missing)


def test_dense_pre_topk_grid_extends_to_one_millionth():
    from experiments.pre_prune_mask_recall import dense_pre_topk_grid

    values = dense_pre_topk_grid()
    assert len(values) == 13
    assert values[0] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(1e-6)
    assert values[1:4] == pytest.approx([0.3, 0.1, 0.03])
    assert values[-3:] == pytest.approx([1e-5, 3e-6, 1e-6])


def test_decade_pre_topk_helper(ablation):
    assert ablation._is_decade_pre_topk(0.1)
    assert ablation._is_decade_pre_topk(0.01)
    assert not ablation._is_decade_pre_topk(0.3)
    assert ablation._decade_pre_topk_values([1.0, 0.3, 0.1, 0.03, 0.01]) == [1.0, 0.1, 0.01]


def test_ref_recall_metrics_matches_topk_intersection(ablation):
    ref = set(range(1000))
    heuristic = set(range(400, 1400))
    recall, precision = ablation._ref_recall_metrics(heuristic, ref)
    assert recall == pytest.approx(0.6)
    assert precision == pytest.approx(600 / 1000)


def test_has_success_recall_only_does_not_require_converged(ablation, monkeypatch):
    run_id = ablation._grid_run_id(ablation.DER_DIE_PAIRS[0], "diff", 16, 0.1)
    pair_key = ablation._pair_slug(ablation.DER_DIE_PAIRS[0])
    idx_path = "indices.json"
    monkeypatch.setattr(ablation.os.path, "isfile", lambda path: path == idx_path)
    results = [
        ablation.GridResult(
            pair=pair_key,
            run_id=run_id,
            pre_topk=0.1,
            pre_source="diff",
            pre_n_samples=16,
            ref_recall=0.42,
            topk_indices_file=idx_path,
            mask_recall=True,
            converged=False,
        )
    ]
    assert ablation._has_success(results, pair_key, run_id, recall_only=True)
    assert not ablation._has_success(results, pair_key, run_id, recall_only=False)


def test_row_plottable_allows_high_recall_for_mask_recall_rows(ablation):
    row = ablation.GridResult(
        pair="masc_nom_fem_nom",
        run_id="test",
        pre_topk=0.1,
        ref_recall=0.995,
        topk_indices_file="indices.json",
        converged=True,
        kept_dim=100,
        mask_recall=True,
    )
    assert ablation._row_plottable(row, require_converged=True)


def test_pre_topk_axis_ticks_use_decades_only(ablation):
    grid = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 1e-6]
    ticks = ablation._pre_topk_axis_ticks(grid)
    assert 0.3 not in ticks
    assert 0.03 not in ticks
    assert 0.1 in ticks
    assert 1e-6 in ticks
    assert 1.0 in ticks


def test_ref_recall_plot_xlim_follows_plotted_data(ablation):
    assert ablation._ref_recall_plot_xlim_lo([0.003, 0.1, 0.3]) == pytest.approx(0.003)
    assert ablation._ref_recall_plot_xlim_lo([1e-5, 0.1]) == pytest.approx(1e-5)


def test_ref_recall_plot_ylim_tight_to_data(ablation):
    y_lo, y_hi = ablation._ref_recall_plot_ylim([0.97, 0.99, 1.0])
    assert y_lo > 0.9
    assert y_hi <= 1.01
    assert y_lo > 0.5


def test_format_pre_topk_axis_tick_includes_three_tenths(ablation):
    assert ablation._format_pre_topk_axis_tick(0.3) == "0.3"
    assert ablation._format_pre_topk_axis_tick(0.03) == "0.03"
    assert ablation._format_pre_topk_axis_tick(1e-6) == "1e-6"
    assert ablation._format_pre_topk_axis_tick(3e-5) == "3e-5"


def test_order_pre_sources(ablation):
    assert ablation._order_pre_sources(["diff", "factual", "alternative"]) == [
        "factual",
        "alternative",
        "diff",
    ]


def test_ref_recall_plot_bridges_last_point_to_baseline(ablation, monkeypatch):
    summaries = [
        ablation.ConfigSummary(
            pre_topk=0.1,
            pre_source="diff",
            pre_n_samples=16,
            mean_ref_recall=0.45,
            mean_kept_dim=100.0,
            mean_cross_overlap=0.2,
            mean_encoder_correlation=0.9,
            n_pairs=1,
        )
    ]
    captured = {}
    monkeypatch.setattr(
        ablation,
        "_save_figure",
        lambda fig, output_path, **kwargs: captured.setdefault("fig", fig),
    )

    ablation.plot_ref_recall_vs_pre_topk(
        summaries,
        output_path="unused.pdf",
        sources=["diff"],
        pre_topk_values=[1.0, 0.3, 0.1, 0.01],
    )

    ax = captured["fig"].axes[0]
    curve_lines = [line for line in ax.get_lines() if len(line.get_xdata()) > 1]
    assert len(curve_lines) == 1
    assert curve_lines[0].get_xdata()[-1] == pytest.approx(1.0)
    assert curve_lines[0].get_ydata()[-1] == pytest.approx(1.0)
    assert ax.get_xlim()[0] == pytest.approx(0.1 * 0.85)
    y_lo, y_hi = ax.get_ylim()
    assert y_lo > 0.0
    assert y_hi - y_lo < 0.7


def test_ref_recall_plot_uses_one_inline_titled_legend(ablation, monkeypatch):
    summaries = [
        ablation.ConfigSummary(
            pre_topk=topk,
            pre_source=source,
            pre_n_samples=n_samples,
            mean_ref_recall=recall,
            mean_kept_dim=100.0,
            mean_cross_overlap=0.2,
            mean_encoder_correlation=0.9,
            n_pairs=2,
        )
        for source in ("factual", "alternative", "diff")
        for n_samples, recall in ((1, 0.4), (2, 0.5))
        for topk in (0.1,)
    ]
    captured = {}
    monkeypatch.setattr(
        ablation,
        "_save_figure",
        lambda fig, output_path, **kwargs: captured.setdefault("fig", fig),
    )

    ablation.plot_ref_recall_vs_pre_topk(
        summaries,
        output_path="unused.pdf",
        sources=["factual", "alternative", "diff"],
        pre_topk_values=[1.0, 0.1, 0.01, 0.001],
    )

    fig = captured["fig"]
    assert [ax.get_title() for ax in fig.axes] == [
        "source=factual",
        "source=alternative",
        "source=diff",
    ]
    tick_labels = {label.get_text() for ax in fig.axes for label in ax.get_xticklabels()}
    assert "0.3" not in tick_labels
    assert "0.03" not in tick_labels
    assert len(fig.legends) == 1
    legend = fig.legends[0]
    assert legend.get_title().get_text() == ""
    assert [text.get_text() for text in legend.get_texts()] == [
        "n_samples",
        "1",
        "2",
    ]
    assert legend._ncols == 3
    assert all(ax.get_legend() is None for ax in fig.axes)
    assert fig._supylabel.get_text() == "Recall"


def test_ref_recall_plot_includes_configured_dense_topk_values(ablation, monkeypatch):
    summaries = [
        ablation.ConfigSummary(
            pre_topk=topk,
            pre_source="diff",
            pre_n_samples=16,
            mean_ref_recall=0.5,
            mean_kept_dim=100.0,
            mean_cross_overlap=0.2,
            mean_encoder_correlation=0.9,
            n_pairs=1,
        )
        for topk in (0.3, 0.1)
    ]
    captured = {}
    monkeypatch.setattr(
        ablation,
        "_save_figure",
        lambda fig, output_path, **kwargs: captured.setdefault("fig", fig),
    )

    ablation.plot_ref_recall_vs_pre_topk(
        summaries,
        output_path="unused.pdf",
        sources=["diff"],
        pre_topk_values=[1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001],
    )

    ax = captured["fig"].axes[0]
    tick_labels = {label.get_text() for label in ax.get_xticklabels()}
    assert "0.3" not in tick_labels
    assert "0.03" not in tick_labels
    assert "0.1" in tick_labels
    assert "1e-3" not in tick_labels
    plotted_xs = {
        float(x)
        for line in ax.get_lines()
        if len(line.get_xdata()) > 1
        for x in line.get_xdata()
    }
    assert any(math.isclose(x, 0.3) for x in plotted_xs)
    assert any(math.isclose(x, 1.0) for x in plotted_xs)


def test_mask_recall_row_with_large_kept_dim_is_plottable(ablation):
    row = ablation.GridResult(
        pair="masc_nom_fem_nom",
        run_id="test",
        pre_topk=0.3,
        ref_recall=0.99,
        topk_indices_file="indices.json",
        converged=True,
        kept_dim=32_547_226,
        mask_recall=True,
    )
    assert ablation._row_plottable(row, require_converged=True)


def test_backfill_results_from_topk_indices(ablation, tmp_path):
    output_dir = tmp_path / "run"
    pair = ablation.DER_DIE_PAIRS[0]
    pair_key = ablation._pair_slug(pair)
    index_dir = output_dir / "topk_indices" / ablation._pair_child_id(pair)
    index_dir.mkdir(parents=True)
    baseline = {
        "run_id": f"{ablation._pair_child_id(pair)}/ref_pre_topk_1_000000",
        "pair": pair_key,
        "oracle": True,
        "indices": [1, 2, 3],
    }
    cell = {
        "run_id": f"{ablation._pair_child_id(pair)}/pre_src_diff_n_16_topk_0_300000",
        "pair": pair_key,
        "mask_recall": True,
        "indices": [2, 3, 4],
    }
    import json

    with open(index_dir / "ref_pre_topk_1_000000.json", "w", encoding="utf-8") as handle:
        json.dump(baseline, handle)
    with open(index_dir / "pre_src_diff_n_16_topk_0_300000.json", "w", encoding="utf-8") as handle:
        json.dump(cell, handle)

    n = ablation.backfill_results_from_topk_indices(str(output_dir), pair)
    assert n == 2
    results = ablation._load_results(ablation._default_pair_results_path(str(output_dir), pair))
    assert len(results) == 2
    pruned = next(r for r in results if r.pre_topk == pytest.approx(0.3))
    assert pruned.ref_recall == pytest.approx(2 / ablation.TOPK_EVAL)
    assert pruned.mask_recall is True


def test_backfill_skips_existing_rows_without_loading_index_files(ablation, tmp_path, monkeypatch):
    output_dir = tmp_path / "run"
    pair = ablation.DER_DIE_PAIRS[0]
    pair_key = ablation._pair_slug(pair)
    index_dir = output_dir / "topk_indices" / ablation._pair_child_id(pair)
    index_dir.mkdir(parents=True)
    import json

    cell = {
        "run_id": f"{ablation._pair_child_id(pair)}/pre_src_diff_n_16_topk_0_300000",
        "pair": pair_key,
        "mask_recall": True,
        "ref_recall": 0.99,
        "ref_precision": 0.5,
        "kept_dim": 123,
        "indices": [2, 3, 4],
    }
    with open(index_dir / "ref_pre_topk_1_000000.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "run_id": f"{ablation._pair_child_id(pair)}/ref_pre_topk_1_000000",
                "indices": [1, 2, 3],
            },
            handle,
        )
    with open(index_dir / "pre_src_diff_n_16_topk_0_300000.json", "w", encoding="utf-8") as handle:
        json.dump(cell, handle)

    assert ablation.backfill_results_from_topk_indices(str(output_dir), pair) == 2
    loads = {"count": 0}
    real_load = json.load

    def counting_load(handle, *args, **kwargs):
        loads["count"] += 1
        return real_load(handle, *args, **kwargs)

    monkeypatch.setattr(ablation.json, "load", counting_load)
    assert ablation.backfill_results_from_topk_indices(str(output_dir), pair) == 0
    assert loads["count"] == 1  # pair_results only; no topk_indices JSON reload


def test_summarize_configs_skips_index_load_for_single_pair(ablation, monkeypatch):
    pair_key = ablation._pair_slug(ablation.DER_DIE_PAIRS[0])

    def fail_load(_path):
        raise AssertionError("index files should not be loaded for single-pair summaries")

    monkeypatch.setattr(ablation, "_load_topk_indices", fail_load)
    rows = [
        ablation.GridResult(
            pair=pair_key,
            run_id="run",
            pre_topk=0.1,
            pre_source="diff",
            pre_n_samples=16,
            ref_recall=0.5,
            topk_indices_file="indices.json",
            converged=True,
            kept_dim=100,
            mask_recall=True,
        )
    ]
    summaries = ablation._summarize_configs(rows, require_converged=True)
    assert len(summaries) == 1
    assert summaries[0].mean_ref_recall == pytest.approx(0.5)


def test_resolve_plot_grid_results_merges_default_run_dir(ablation, tmp_path):
    import json

    sparse_dir = tmp_path / "sparse"
    sparse_dir.mkdir()
    pair = ablation.DER_DIE_PAIRS[0]
    pair_key = ablation._pair_slug(pair)
    sparse_pair = sparse_dir / "pair_results"
    sparse_pair.mkdir(parents=True)
    baseline = ablation.GridResult(
        pair=pair_key,
        run_id=ablation._baseline_run_id(pair),
        pre_topk=1.0,
        ref_recall=1.0,
        topk_indices_file="baseline.json",
        converged=True,
        kept_dim=100,
    )
    with open(sparse_pair / f"{pair_key}.json", "w", encoding="utf-8") as handle:
        json.dump([dataclasses.asdict(baseline)], handle)

    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()
    pruned = ablation.GridResult(
        pair=pair_key,
        run_id=ablation._grid_run_id(pair, "diff", 16, 0.1),
        pre_topk=0.1,
        pre_source="diff",
        pre_n_samples=16,
        ref_recall=0.42,
        topk_indices_file="pruned.json",
        converged=True,
        kept_dim=10,
    )
    with open(fallback_dir / "pre_topk_grid_results.json", "w", encoding="utf-8") as handle:
        json.dump([dataclasses.asdict(pruned)], handle)

    results = ablation.resolve_plot_grid_results(
        str(sparse_dir),
        [pair],
        require_converged=True,
        fallback_output_dir=str(fallback_dir),
    )
    run_ids = {row.run_id for row in results}
    assert ablation._baseline_run_id(pair) in run_ids
    assert ablation._grid_run_id(pair, "diff", 16, 0.1) in run_ids


def test_resolve_plot_grid_results_skips_fallback_for_parameter_ablation_dir(ablation, tmp_path):
    import json

    isolated_dir = tmp_path / "runs" / "pruning_parameter_ablation" / "german_de_v3"
    pair_results = isolated_dir / "pair_results"
    pair_results.mkdir(parents=True)
    pair = ablation.DER_DIE_PAIRS[0]
    pair_key = ablation._pair_slug(pair)
    baseline = ablation.GridResult(
        pair=pair_key,
        run_id=ablation._baseline_run_id(pair),
        pre_topk=1.0,
        ref_recall=1.0,
        topk_indices_file="baseline.json",
        converged=True,
        kept_dim=100,
    )
    with open(pair_results / f"{pair_key}.json", "w", encoding="utf-8") as handle:
        json.dump([dataclasses.asdict(baseline)], handle)

    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()
    pruned = ablation.GridResult(
        pair=pair_key,
        run_id=ablation._grid_run_id(pair, "diff", 16, 0.1),
        pre_topk=0.1,
        pre_source="diff",
        pre_n_samples=16,
        ref_recall=0.42,
        topk_indices_file="pruned.json",
        converged=True,
        kept_dim=10,
    )
    with open(fallback_dir / "pre_topk_grid_results.json", "w", encoding="utf-8") as handle:
        json.dump([dataclasses.asdict(pruned)], handle)

    results = ablation.resolve_plot_grid_results(
        str(isolated_dir),
        [pair],
        require_converged=True,
        fallback_output_dir=str(fallback_dir),
    )
    run_ids = {row.run_id for row in results}
    assert ablation._baseline_run_id(pair) in run_ids
    assert ablation._grid_run_id(pair, "diff", 16, 0.1) not in run_ids
