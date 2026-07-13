#!/usr/bin/env python3
"""
Generate synthetic cross-encoding explanation figures for the docs.

Overview plots from trained models still come from
``experiments/multilingual_gradiend_demo_small.py`` (see docs/img/README.md).
This script only writes **synthetic** walkthrough figures.

Usage:
    python scripts/generate_cross_encoding_matrix_doc_figures.py
    python scripts/generate_cross_encoding_matrix_doc_figures.py --output-dir docs/img
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from gradiend.comparison.anchor_aligned import compute_anchor_aligned_encoding_matrix
from gradiend.comparison.cross_encoding import compute_gradiend_transition_cross_encoding_matrix
from synthetic_cross_encoding_fixture import (
    FEATURE_LABELS,
    FEATURE_ORDER,
    FEATURE_PLOT_GROUPS,
    PAIR_BY_ID,
    SOURCE_BY_ID,
    TRAINER_LABELS,
    TRAINER_ORDER,
    TRAINER_PLOT_GROUPS,
    WORKED_DIAGONAL,
    WORKED_OFF_DIAGONAL,
    _family_of,
    _parse_transition,
    build_synthetic_encoder_summary,
    cell_contribution_rows,
    dummy_trainers,
    preanchor_highlight_sets,
    synthetic_transition_order,
    transition_label_mapping,
)
from gradiend.visualizer.heatmaps.base import plot_comparison_heatmap
from gradiend.visualizer.heatmaps.highlight import highlight_heatmap_cells

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "img"

SYNTHETIC_FIGURES: Tuple[Tuple[str, str], ...] = (
    ("cross_encoding_synthetic_preanchor_overview.png", "Full GRADIEND × transition matrix"),
    ("cross_encoding_synthetic_oriented_overview.png", "Full synthetic oriented matrix (counterfactual)"),
    ("cross_encoding_synthetic_preanchor_diagonal_highlight.png", "Pre-anchor contributors for diagonal cell"),
    ("cross_encoding_synthetic_oriented_diagonal_highlight.png", "Oriented diagonal cell highlight"),
    ("cross_encoding_synthetic_aggregation_diagonal.png", "Aggregation table (diagonal)"),
    ("cross_encoding_synthetic_preanchor_offdiag_highlight.png", "Pre-anchor contributors for off-diagonal cell"),
    ("cross_encoding_synthetic_oriented_offdiag_highlight.png", "Oriented off-diagonal cell highlight"),
    ("cross_encoding_synthetic_aggregation_offdiag.png", "Aggregation table (off-diagonal)"),
)


def _require_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def _plot_style_kwargs(*, vmin: float = -1.0, vmax: float = 1.0) -> Dict[str, Any]:
    return {
        "cmap": "coolwarm",
        "vmin": vmin,
        "vmax": vmax,
        "annot": True,
        "annot_fmt": ".2f",
        "annot_fontsize": 7,
        "tick_label_fontsize": 9,
        "axis_label_fontsize": 11,
        "group_label_fontsize": 10,
        "show": False,
        "return_fig_ax": True,
        "return_data": True,
    }


def _save_fig(fig, path: Path, *, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)


def _close_fig(fig) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def _transition_labels() -> Dict[str, str]:
    return transition_label_mapping(synthetic_transition_order())


def _compute_oriented_payload(
    encoder_summary: Dict[str, Any],
    *,
    alignment: str,
) -> Dict[str, Any]:
    payload = compute_anchor_aligned_encoding_matrix(
        pair_by_id=PAIR_BY_ID,
        encoder_summary=encoder_summary,
        feature_classes=list(FEATURE_ORDER),
        alignment=alignment,
        aggregate="mean",
        source_by_id=SOURCE_BY_ID,
    )
    payload["pretty_groups"] = FEATURE_PLOT_GROUPS
    return payload


def _compute_preanchor_payload(encoder_summary: Dict[str, Any]) -> Dict[str, Any]:
    transitions = synthetic_transition_order()
    payload = compute_gradiend_transition_cross_encoding_matrix(
        dummy_trainers(),
        trainer_order=list(TRAINER_ORDER),
        transition_order=transitions,
        encoder_summary=encoder_summary,
    )
    payload["pretty_groups"] = TRAINER_PLOT_GROUPS
    return payload


def _oriented_plot_kwargs(**extra: Any) -> Dict[str, Any]:
    return {
        **_plot_style_kwargs(),
        "order": list(FEATURE_ORDER),
        "pretty_groups": FEATURE_PLOT_GROUPS,
        "row_label_mapping": FEATURE_LABELS,
        "column_label_mapping": FEATURE_LABELS,
        "xlabel": "Probe feature (counterfactual)",
        "ylabel": "Orienting feature",
        **extra,
    }


def _preanchor_plot_kwargs(**extra: Any) -> Dict[str, Any]:
    return {
        **_plot_style_kwargs(),
        "order": list(TRAINER_ORDER),
        "pretty_groups": TRAINER_PLOT_GROUPS,
        "row_label_mapping": TRAINER_LABELS,
        "column_label_mapping": _transition_labels(),
        "xlabel": "Input transition",
        "ylabel": "GRADIEND",
        **extra,
    }


def plot_synthetic_preanchor_overview(
    preanchor: Dict[str, Any],
    output_path: Path,
) -> None:
    _require_matplotlib()
    n_cols = len(preanchor["columns"])
    style = _preanchor_plot_kwargs(
        title="Synthetic GRADIEND × transition (pre-anchor, full pool)",
        figsize=(max(14, 0.45 * n_cols), 4.8),
    )
    _, fig, _ax = plot_comparison_heatmap(preanchor, **style)
    _save_fig(fig, output_path)
    _close_fig(fig)


def plot_synthetic_oriented_overview(
    encoder_summary: Dict[str, Any],
    output_path: Path,
) -> None:
    _require_matplotlib()
    payload = _compute_oriented_payload(encoder_summary, alignment="counterfactual")
    style = _oriented_plot_kwargs(
        title="Synthetic oriented matrix (counterfactual probes)",
        figsize=(8.0, 6.8),
    )
    _, fig, _ax = plot_comparison_heatmap(payload, **style)
    _save_fig(fig, output_path)
    _close_fig(fig)


def plot_preanchor_highlight(
    preanchor: Dict[str, Any],
    contrib: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
) -> None:
    _require_matplotlib()
    trainer_ids, transition_ids = preanchor_highlight_sets(contrib, same_family_only=True)
    n_cols = len(preanchor["columns"])
    style = _preanchor_plot_kwargs(
        title=title,
        figsize=(max(14, 0.45 * n_cols), 4.8),
    )
    _, fig, ax = plot_comparison_heatmap(preanchor, **style)
    highlight_heatmap_cells(
        ax,
        row_labels=preanchor["rows"],
        col_labels=preanchor["columns"],
        row_subset=trainer_ids,
        col_subset=transition_ids,
    )
    _save_fig(fig, output_path)
    _close_fig(fig)


def plot_oriented_cell_highlight(
    oriented: Dict[str, Any],
    *,
    anchor: str,
    column: str,
    title: str,
    output_path: Path,
) -> None:
    _require_matplotlib()
    style = _oriented_plot_kwargs(
        title=title,
        figsize=(8.0, 6.8),
    )
    _, fig, ax = plot_comparison_heatmap(oriented, **style)
    highlight_heatmap_cells(
        ax,
        row_labels=oriented["rows"],
        col_labels=oriented["columns"],
        cells=[(anchor, column)],
        linewidth=3.5,
    )
    _save_fig(fig, output_path)
    _close_fig(fig)


def _format_float(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def plot_aggregation_table(
    contrib: pd.DataFrame,
    *,
    anchor: str,
    column: str,
    matrix_value: float,
    output_path: Path,
) -> None:
    from matplotlib import pyplot as plt_module

    if contrib.empty:
        raise ValueError("No contributions to render")

    display = contrib.copy()
    # Walkthrough tables focus on same-family transitions (cross-family terms are ≈0).
    anchor_family = _family_of(str(display["anchor_class"].iloc[0]))
    same_family = display["transition_id"].astype(str).map(
        lambda tid: all(_family_of(part) == anchor_family for part in _parse_transition(tid))
    )
    display = display.loc[same_family]
    display["raw"] = display["source_mean"].astype(float)
    display["sign"] = display["sign"].astype(float).map(lambda s: f"{int(s):+d}")
    display["aligned"] = display["aligned_mean"].astype(float)

    headers = ["GRADIEND", "Transition", "Raw mean", "Anchor sign", "Signed value"]
    table_rows: List[List[str]] = []
    labels = _transition_labels()
    for _, row in display.sort_values(["trainer_id", "transition_id"]).iterrows():
        tid = str(row["transition_id"])
        table_rows.append(
            [
                TRAINER_LABELS.get(str(row["trainer_id"]), str(row["trainer_id"])),
                labels.get(tid, tid),
                _format_float(row["raw"]),
                str(row["sign"]),
                _format_float(row["aligned"]),
            ]
        )

    per_trainer = (
        display.groupby("trainer_id", dropna=False)["aligned"]
        .mean()
        .reset_index(name="trainer_mean")
    )
    overall = float(display["aligned"].mean())

    fig, ax = plt_module.subplots(figsize=(10, 0.55 * (len(table_rows) + 4)))
    ax.axis("off")
    anchor_label = FEATURE_LABELS.get(anchor, anchor)
    column_label = FEATURE_LABELS.get(column, column)
    ax.set_title(
        f"Aggregation for orienting={anchor_label}, probe={column_label} → M={_format_float(matrix_value)}",
        fontsize=12,
        pad=12,
    )

    table = ax.table(
        cellText=table_rows,
        colLabels=headers,
        loc="upper center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    summary_lines = [
        "Per GRADIEND: mean of signed values over contributing transitions.",
        "Matrix entry: mean across GRADIENDs whose pair contains the orienting feature.",
    ]
    summary_lines.append(
        "Per GRADIEND means: "
        + ", ".join(
            f"{TRAINER_LABELS.get(str(row['trainer_id']), row['trainer_id'])}={_format_float(row['trainer_mean'])}"
            for _, row in per_trainer.iterrows()
        )
    )
    summary_lines.append(f"Overall mean (matrix cell): {_format_float(overall)}")

    fig.text(0.5, 0.02, "\n".join(summary_lines), ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    _save_fig(fig, output_path)
    _close_fig(fig)


def generate_synthetic_doc_figures(output_dir: Path) -> List[Path]:
    """Write all synthetic explanation figures; return output paths."""
    encoder_summary_full = build_synthetic_encoder_summary(full_pool=True)
    encoder_summary_oriented = build_synthetic_encoder_summary(full_pool=False)
    preanchor = _compute_preanchor_payload(encoder_summary_full)
    oriented = _compute_oriented_payload(
        encoder_summary_oriented,
        alignment=WORKED_DIAGONAL["alignment"],
    )

    written: List[Path] = []

    preanchor_overview_path = output_dir / SYNTHETIC_FIGURES[0][0]
    plot_synthetic_preanchor_overview(preanchor, preanchor_overview_path)
    written.append(preanchor_overview_path)

    overview_path = output_dir / SYNTHETIC_FIGURES[1][0]
    plot_synthetic_oriented_overview(encoder_summary_full, overview_path)
    written.append(overview_path)

    for spec, preanchor_name, oriented_name, table_name in (
        (
            WORKED_DIAGONAL,
            SYNTHETIC_FIGURES[2][0],
            SYNTHETIC_FIGURES[3][0],
            SYNTHETIC_FIGURES[4][0],
        ),
        (
            WORKED_OFF_DIAGONAL,
            SYNTHETIC_FIGURES[5][0],
            SYNTHETIC_FIGURES[6][0],
            SYNTHETIC_FIGURES[7][0],
        ),
    ):
        anchor = spec["anchor"]
        column = spec["column"]
        contrib = cell_contribution_rows(oriented, anchor=anchor, column=column)
        matrix_df = pd.DataFrame(
            oriented["matrix"],
            index=oriented["rows"],
            columns=oriented["columns"],
        )
        matrix_value = float(matrix_df.loc[anchor, column])

        pre_path = output_dir / preanchor_name
        plot_preanchor_highlight(
            preanchor,
            contrib,
            title=f"Pre-anchor contributors → ({FEATURE_LABELS.get(anchor, anchor)}, {FEATURE_LABELS.get(column, column)})",
            output_path=pre_path,
        )
        written.append(pre_path)

        ori_path = output_dir / oriented_name
        plot_oriented_cell_highlight(
            oriented,
            anchor=anchor,
            column=column,
            title=f"Oriented cell ({FEATURE_LABELS.get(anchor, anchor)}, {FEATURE_LABELS.get(column, column)})",
            output_path=ori_path,
        )
        written.append(ori_path)

        table_path = output_dir / table_name
        plot_aggregation_table(
            contrib,
            anchor=anchor,
            column=column,
            matrix_value=matrix_value,
            output_path=table_path,
        )
        written.append(table_path)

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic cross-encoding doc explanation figures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG output (default: docs/img).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_synthetic_doc_figures(args.output_dir)
    print(f"Wrote {len(paths)} synthetic doc figures to {args.output_dir}:")
    for path in paths:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
