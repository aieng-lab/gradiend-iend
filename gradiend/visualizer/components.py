"""Component-level plot sidecars for explicit GRADIEND/ACTIEND splits."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gradiend.visualizer.encoder_distributions import plot_encoder_distributions
from gradiend.visualizer.labels import CORRELATION_LABEL, escape_matplotlib_usetex_text
from gradiend.visualizer.plot_optional import _require_matplotlib
from gradiend.util.logging import get_logger

logger = get_logger(__name__)


def _safe_component_label(value: Any) -> str:
    text = str(value if value is not None else "component")
    if text.startswith("activation:"):
        text = text[len("activation:"):]
    return text


def _component_sort_key(index: Any, component_id: Any) -> Tuple[int, str]:
    try:
        return int(index), str(component_id)
    except (TypeError, ValueError):
        return 10**9, str(component_id)


def _component_dir_from_output(output: Optional[str], trainer: Optional[Any] = None) -> Optional[str]:
    if output:
        return os.path.join(os.path.dirname(os.path.normpath(output)) or ".", "components")
    experiment_dir = getattr(trainer, "experiment_dir", None) if trainer is not None else None
    if callable(experiment_dir):
        experiment_dir = experiment_dir()
    if experiment_dir and str(experiment_dir).strip():
        return os.path.join(os.path.normpath(str(experiment_dir)), "components")
    return None


def _stem_from_output(output: Optional[str], default: str) -> str:
    if not output:
        return default
    stem = os.path.splitext(os.path.basename(os.path.normpath(output)))[0]
    return stem or default


def _path_with_format(path: str, img_format: str) -> str:
    ext = img_format if str(img_format).startswith(".") else f".{img_format}"
    return os.path.splitext(path)[0] + ext


def _format_suffix(img_format: str) -> str:
    return str(img_format or "png").lstrip(".")


def _component_metadata_from_df(component_df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for _, row in component_df.iterrows():
        component_id = str(row.get("component_id"))
        if component_id in rows:
            continue
        component_index = row.get("component_index")
        label = row.get("component_label") if "component_label" in component_df.columns else component_id
        rows[component_id] = {
            "component_index": component_index,
            "component_id": component_id,
            "component_label": _safe_component_label(label if label is not None else component_id),
        }
    return sorted(rows.values(), key=lambda item: _component_sort_key(item.get("component_index"), item.get("component_id")))


def plot_encoder_component_overview(
    *,
    components: Dict[str, Any],
    output: str,
    img_format: str = "png",
    dpi: Optional[int] = None,
    log_saved: bool = True,
) -> str:
    """Save a compact overview of encoder metrics for each visible component."""
    metrics_by_component = components.get("metrics_by_component") if isinstance(components, dict) else None
    if not isinstance(metrics_by_component, dict) or not metrics_by_component:
        return ""

    plt = _require_matplotlib()
    rows = []
    for component_id, metrics in metrics_by_component.items():
        if not isinstance(metrics, dict):
            continue
        corr = metrics.get("correlation")
        rows.append({
            "component_index": metrics.get("component_index"),
            "component_id": str(component_id),
            "component_label": _safe_component_label(metrics.get("component_label", component_id)),
            "correlation": float(corr) if isinstance(corr, (int, float)) else None,
        })
    rows = sorted(rows, key=lambda item: _component_sort_key(item.get("component_index"), item.get("component_id")))
    if not rows:
        return ""

    labels = [
        str(int(row["component_index"]))
        if isinstance(row.get("component_index"), (int, float)) and not pd.isna(row.get("component_index"))
        else str(i)
        for i, row in enumerate(rows)
    ]
    values = [row["correlation"] if row["correlation"] is not None else 0.0 for row in rows]
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in values]

    width = max(6.0, min(16.0, 0.32 * len(rows) + 4.0))
    fig, ax = plt.subplots(1, 1, figsize=(width, 3.2))
    ax.bar(labels, values, color=colors, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Component index")
    ax.set_ylabel(CORRELATION_LABEL)
    ax.set_title("Encoder component correlations")
    ax.grid(axis="y", alpha=0.25)
    if len(labels) > 24:
        ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    else:
        ax.tick_params(axis="x", labelrotation=45 if len(labels) > 10 else 0)

    summary = components.get("summary") if isinstance(components, dict) else None
    if isinstance(summary, dict) and summary.get("n_components"):
        text_parts = []
        if summary.get("n_converged") is not None:
            text_parts.append(f"converged {summary.get('n_converged')}/{summary.get('n_components')}")
        if summary.get("correlation_mean") is not None:
            text_parts.append(f"mean r={float(summary['correlation_mean']):.3f}")
        if text_parts:
            ax.text(
                0.99,
                0.02,
                "; ".join(text_parts),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
            )

    out_path = _path_with_format(output, img_format)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_kwargs: Dict[str, Any] = {"bbox_inches": "tight"}
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)
    if log_saved:
        logger.info("Saved encoder component overview plot: %s", out_path)
    return out_path


def plot_encoder_component_artifacts(
    *,
    trainer: Any,
    component_df: Optional[pd.DataFrame],
    components: Optional[Dict[str, Any]],
    output: Optional[str],
    show: bool = False,
    img_format: str = "png",
    dpi: Optional[int] = None,
    plot_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Save default encoder component plots under ``components/``.

    This function only consumes the already-computed component sidecar DataFrame
    and metrics. It performs no encoding and has no effect for ordinary non-split
    runs where no component DataFrame exists.
    """
    if component_df is None or component_df.empty:
        return []
    if "component_index" not in component_df.columns or "component_id" not in component_df.columns:
        return []

    component_dir = _component_dir_from_output(output, trainer=trainer)
    if component_dir is None:
        logger.debug("Skipping encoder component plots because no output path or experiment_dir is available.")
        return []
    os.makedirs(component_dir, exist_ok=True)

    img_format = _format_suffix(img_format)
    stem = _stem_from_output(output, "encoder_analysis")
    paths: List[str] = []
    if isinstance(components, dict) and components.get("metrics_by_component"):
        overview = plot_encoder_component_overview(
            components=components,
            output=os.path.join(component_dir, f"{stem}_components.{img_format}"),
            img_format=img_format,
            dpi=dpi,
            log_saved=False,
        )
        if overview:
            paths.append(overview)

    base_kwargs = dict(plot_kwargs or {})
    for key in ("output", "output_dir", "show", "title", "return_fig_ax", "img_format", "dpi", "log_saved"):
        base_kwargs.pop(key, None)
    base_kwargs.setdefault("target_and_neutral_only", True)

    metadata = _component_metadata_from_df(component_df)
    if not metadata:
        if paths:
            logger.info("Saved %d encoder component plot(s) under %s", len(paths), component_dir)
        return paths
    for item in metadata:
        component_id = item["component_id"]
        component_index = item.get("component_index")
        try:
            component_number = int(component_index)
        except (TypeError, ValueError):
            component_number = len(paths)
        group = component_df[component_df["component_id"].astype(str) == component_id].copy()
        if group.empty:
            continue
        component_output = os.path.join(component_dir, f"{stem}_component_{component_number:03d}.{img_format}")
        title = f"Component {component_number}: {item['component_label']}"
        path = plot_encoder_distributions(
            trainer=trainer,
            encoder_df=group,
            output=component_output,
            show=bool(show),
            title=title,
            img_format=img_format,
            dpi=dpi,
            log_saved=False,
            **base_kwargs,
        )
        if path:
            paths.append(path)
    if paths:
        logger.info("Saved %d encoder component plot(s) under %s", len(paths), component_dir)
    return paths


def _normalize_step_keys(d: Dict[str, Any]) -> Dict[int, Any]:
    out: Dict[int, Any] = {}
    for key, value in (d or {}).items():
        try:
            out[int(key)] = value
        except (TypeError, ValueError):
            continue
    return out


def _component_training_payload(training_stats: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    components = training_stats.get("components") if isinstance(training_stats, dict) else None
    if not isinstance(components, dict):
        return {}
    normalized = _normalize_step_keys(components)
    return {
        step: value
        for step, value in normalized.items()
        if isinstance(value, dict) and isinstance(value.get("metrics_by_component"), dict)
    }


def _component_training_metadata(components_by_step: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for step in sorted(components_by_step):
        metrics_by_component = components_by_step[step].get("metrics_by_component") or {}
        for component_id, metrics in metrics_by_component.items():
            if not isinstance(metrics, dict):
                continue
            component_key = str(component_id)
            rows[component_key] = {
                "component_index": metrics.get("component_index", rows.get(component_key, {}).get("component_index")),
                "component_id": component_key,
                "component_label": _safe_component_label(
                    metrics.get("component_label", rows.get(component_key, {}).get("component_label", component_key))
                ),
            }
    return sorted(rows.values(), key=lambda item: _component_sort_key(item.get("component_index"), item.get("component_id")))


def plot_training_component_overview(
    *,
    training_stats: Dict[str, Any],
    output: Optional[str],
    img_format: str = "png",
    dpi: Optional[int] = None,
    log_saved: bool = True,
    title: str = "Component correlations over training",
    show: bool = False,
) -> str:
    """Save an overview plot with one correlation curve per component."""
    components_by_step = _component_training_payload(training_stats)
    metadata = _component_training_metadata(components_by_step)
    if not components_by_step or not metadata:
        return ""

    plt = _require_matplotlib()
    summary_by_step = _normalize_step_keys(training_stats.get("component_summary") or {})
    series: Dict[str, List[Tuple[int, float]]] = {item["component_id"]: [] for item in metadata}
    convergence_counts: List[Tuple[int, float, float]] = []

    for step in sorted(components_by_step):
        payload = components_by_step[step]
        metrics_by_component = payload.get("metrics_by_component") or {}
        for item in metadata:
            metrics = metrics_by_component.get(item["component_id"])
            if not isinstance(metrics, dict):
                continue
            corr = metrics.get("correlation")
            if isinstance(corr, (int, float)):
                series[item["component_id"]].append((step, float(corr)))
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else summary_by_step.get(step)
        if isinstance(summary, dict):
            n_converged = summary.get("n_converged")
            n_components = summary.get("n_components")
            if isinstance(n_converged, (int, float)) and isinstance(n_components, (int, float)):
                convergence_counts.append((step, float(n_converged), float(n_components)))

    has_counts = bool(convergence_counts)
    fig, axes = plt.subplots(2 if has_counts else 1, 1, figsize=(8.5, 5.0 if has_counts else 3.2), sharex=True)
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    ax = axes[0]
    for item in metadata:
        component_id = item["component_id"]
        points = series.get(component_id) or []
        if not points:
            continue
        xs, ys = zip(*points)
        index = item.get("component_index")
        try:
            prefix = str(int(index))
        except (TypeError, ValueError):
            prefix = component_id
        label = escape_matplotlib_usetex_text(f"{prefix}: {item['component_label']}")
        ax.plot(xs, ys, marker=".", markersize=3, linewidth=1.2, label=label, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_ylabel(CORRELATION_LABEL)
    ax.set_title(escape_matplotlib_usetex_text(title))
    ax.grid(True, alpha=0.3)
    if len(metadata) <= 48:
        ncol = 1 if len(metadata) <= 12 else 2 if len(metadata) <= 28 else 3
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6, ncol=ncol)

    if has_counts:
        ax_count = axes[1]
        xs = [point[0] for point in convergence_counts]
        ys = [point[1] for point in convergence_counts]
        totals = [point[2] for point in convergence_counts]
        total = max(totals) if totals else 0.0
        ax_count.plot(xs, ys, marker="o", linewidth=1.5, color="#2ca02c", label="converged components")
        ax_count.set_ylim(bottom=0.0, top=max(total, max(ys) if ys else 1.0) + 0.5)
        ax_count.set_ylabel("Converged")
        ax_count.set_xlabel("Step")
        ax_count.grid(True, alpha=0.3)
        ax_count.legend(loc="best", fontsize=8)
    else:
        ax.set_xlabel("Step")

    fig.tight_layout()
    out_path = _path_with_format(output, img_format) if output else ""
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        save_kwargs: Dict[str, Any] = {"bbox_inches": "tight"}
        if dpi is not None:
            save_kwargs["dpi"] = dpi
        fig.savefig(out_path, **save_kwargs)
    if show:
        plt.show()
    plt.close(fig)
    if out_path and log_saved:
        logger.info("Saved component convergence overview plot: %s", out_path)
    return out_path


def _component_run_info(
    *,
    component_id: str,
    components_by_step: Dict[int, Dict[str, Any]],
    original_run_info: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    component_stats: Dict[str, Any] = {
        "scores": {},
        "mean_by_class": {},
        "mean_by_feature_class": {},
    }
    for step in sorted(components_by_step):
        metrics_by_component = components_by_step[step].get("metrics_by_component") or {}
        metrics = metrics_by_component.get(component_id)
        if not isinstance(metrics, dict):
            continue
        corr = metrics.get("correlation")
        if isinstance(corr, (int, float)):
            component_stats["scores"][step] = float(corr)
        for key in ("mean_by_class", "mean_by_feature_class"):
            value = metrics.get(key)
            if isinstance(value, dict) and value:
                component_stats[key][step] = value
    component_stats = {key: value for key, value in component_stats.items() if value}
    if not component_stats:
        return None

    scores = component_stats.get("scores") or {}
    best_step = None
    best_corr = None
    if scores:
        best_step = max(scores, key=lambda step: abs(float(scores[step])))
        best_corr = float(scores[best_step])
    label_mapping = (original_run_info.get("training_stats") or {}).get("label_value_to_class_name")
    if label_mapping is not None:
        component_stats["label_value_to_class_name"] = label_mapping
    return {
        "training_stats": component_stats,
        "best_score_checkpoint": {
            "global_step": best_step,
            "correlation": best_corr,
        },
    }


def plot_training_component_artifacts(
    *,
    trainer: Optional[Any],
    run_info: Dict[str, Any],
    output: Optional[str],
    experiment_dir: Optional[str] = None,
    show: bool = False,
    img_format: str = "png",
    dpi: Optional[int] = None,
    plot_mean_by_class: bool = True,
    plot_mean_by_feature_class: Optional[bool] = None,
    plot_correlation: bool = True,
    class_spread: Any = None,
    label_name_mapping: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Save component convergence sidecars from already-collected training stats.

    No model inference is performed. Ordinary non-split runs have no component
    history and return an empty list.
    """
    training_stats = run_info.get("training_stats") if isinstance(run_info, dict) else None
    if isinstance(training_stats, dict) and "training_stats" in training_stats:
        training_stats = training_stats["training_stats"]
    if not isinstance(training_stats, dict):
        return []
    components_by_step = _component_training_payload(training_stats)
    if not components_by_step:
        return []

    base_output = output
    if base_output is None and experiment_dir and str(experiment_dir).strip():
        base_output = os.path.join(os.path.normpath(str(experiment_dir)), f"training_convergence.{img_format}")
    component_dir = _component_dir_from_output(base_output, trainer=trainer)
    if component_dir is None:
        logger.debug("Skipping component convergence plots because no output path or experiment_dir is available.")
        return []
    os.makedirs(component_dir, exist_ok=True)

    img_format = _format_suffix(img_format)
    stem = _stem_from_output(base_output, "training_convergence")
    paths: List[str] = []
    overview = plot_training_component_overview(
        training_stats=training_stats,
        output=os.path.join(component_dir, f"{stem}_components.{img_format}"),
        img_format=img_format,
        dpi=dpi,
        log_saved=False,
    )
    if overview:
        paths.append(overview)

    from gradiend.visualizer.convergence import plot_training_convergence

    for item in _component_training_metadata(components_by_step):
        component_id = item["component_id"]
        run = _component_run_info(
            component_id=component_id,
            components_by_step=components_by_step,
            original_run_info=run_info,
        )
        if run is None:
            continue
        try:
            component_number = int(item.get("component_index"))
        except (TypeError, ValueError):
            component_number = len(paths)
        output_path = os.path.join(component_dir, f"{stem}_component_{component_number:03d}.{img_format}")
        path = plot_training_convergence(
            trainer=None,
            training_stats=run,
            plot_mean_by_class=plot_mean_by_class,
            plot_mean_by_feature_class=plot_mean_by_feature_class,
            plot_correlation=plot_correlation,
            class_spread=class_spread,
            best_step=True,
            label_name_mapping=label_name_mapping,
            output=output_path,
            show=bool(show),
            title=f"Component {component_number}: {item['component_label']}",
            img_format=img_format,
            dpi=dpi,
            highlight_non_convergence=False,
            return_fig_ax=False,
            log_saved=False,
        )
        if path:
            paths.append(path)
    if paths:
        logger.info("Saved %d component convergence plot(s) under %s", len(paths), component_dir)
    return paths


__all__ = [
    "plot_encoder_component_artifacts",
    "plot_encoder_component_overview",
    "plot_training_component_artifacts",
    "plot_training_component_overview",
]
