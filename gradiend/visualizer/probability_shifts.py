"""
Probability shifts plot: target token probabilities vs learning rate.

Each subplot is keyed by **factual** class (``label_class`` / ``factual_id``):
``probs_by_dataset["3PL"]["3SG"]`` is P(3SG) on rows where the factual class is 3PL.
Strengthening class T with ``decoder_eval_prob_on_other_class`` selects P(T) on the
other factual class's panel (star on that curve).

Requires matplotlib. If missing, raises ImportError with install instructions.
"""

import os
import math
from typing import Any, Dict, List, Optional, Tuple

from gradiend.visualizer.plot_optional import _require_matplotlib
from gradiend.model._source_target import (
    intervention_feature_factor_from_encoding_direction,
    resolve_model_source,
    resolve_model_signal_kind,
    resolve_model_target,
)
from gradiend.util.logging import get_logger
from gradiend.visualizer.labels import escape_matplotlib_usetex_text

logger = get_logger(__name__)


def _with_base_point(
    xs: List[float],
    ys: List[float],
    base_x: float,
    base_y: float,
) -> Tuple[List[float], List[float]]:
    """Return (x, y) with the base point included, sorted by x."""
    points = sorted(zip(list(xs) + [base_x], list(ys) + [base_y]), key=lambda t: t[0])
    return [p[0] for p in points], [p[1] for p in points]


def _probability_or_nan(
    probs_by_dataset: Dict[str, Any],
    dataset_class: str,
    class_name: str,
) -> float:
    """Return a plotted probability, preserving missing cells as NaN."""
    dataset_probs = probs_by_dataset.get(dataset_class, {})
    if not isinstance(dataset_probs, dict) or class_name not in dataset_probs:
        return math.nan
    value = dataset_probs[class_name]
    if value is None:
        return math.nan
    return float(value)


def _require_probability_cell(
    probs_by_dataset: Dict[str, Any],
    dataset_class: str,
    class_name: str,
    *,
    context: str,
) -> float:
    value = _probability_or_nan(probs_by_dataset, dataset_class, class_name)
    if math.isnan(value):
        raise ValueError(
            f"Missing decoder probability for selected metric in {context}: "
            f"probs_by_dataset[{dataset_class!r}][{class_name!r}] is absent. "
            "Do not interpret missing decoder panels as probability 0.0; re-run "
            "decoder evaluation with full factual-class coverage."
        )
    return value


def _apply_lr_xscale(ax: Any, scale: str, linthresh: Optional[float]) -> None:
    """Apply learning-rate x-axis scale (log or symmetric log around zero)."""
    if scale == "log":
        ax.set_xscale("log")
    elif scale == "symlog":
        ax.set_xscale("symlog", linthresh=linthresh if linthresh is not None else 1e-6, base=10)
    else:
        ax.set_xscale("linear")


def _lr_axis_config(
    lrs: List[float],
) -> Tuple[str, Optional[float], float, float, float, List[float], List[str]]:
    """
    Choose x-axis scale and tick layout for learning-rate sweeps.

    - All LRs > 0: standard log scale, base anchor at min(lr)/10.
    - Otherwise: symmetric log (symlog) so lr=0 and negative LRs remain readable.
    """
    if not lrs:
        raise ValueError("No learning rates to plot")

    def _power_ticks(min_abs: float, max_abs: float) -> List[float]:
        lo = math.ceil(math.log10(min_abs))
        hi = math.floor(math.log10(max_abs))
        return [10.0 ** exp for exp in range(lo, hi + 1)]

    def _power_label(value: float) -> str:
        if value == 0:
            return "base"
        sign = "-" if value < 0 else ""
        exp = int(round(math.log10(abs(value))))
        return f"${sign}10^{{{exp}}}$"

    if all(lr > 0 for lr in lrs):
        min_lr = min(lrs)
        max_lr = max(lrs)
        lr0_x = min_lr / 10.0
        x_min, x_max = lr0_x * 0.5, max_lr * 1.5
        decade_ticks = _power_ticks(min_lr, max_lr)
        x_ticks = [lr0_x] + decade_ticks
        x_labels = ["base"] + [_power_label(tick) for tick in decade_ticks]
        return "log", None, lr0_x, x_min, x_max, x_ticks, x_labels

    lr0_x = 0.0
    abs_nonzero = [abs(lr) for lr in lrs if lr != 0]
    linthresh = min(abs_nonzero) / 10.0 if abs_nonzero else 1e-6
    linthresh = max(linthresh, 1e-12)

    neg_lrs = [lr for lr in lrs if lr < 0]
    pos_lrs = [lr for lr in lrs if lr > 0]
    neg_ticks = [-tick for tick in reversed(_power_ticks(min(abs(lr) for lr in neg_lrs), max(abs(lr) for lr in neg_lrs)))] if neg_lrs else []
    pos_ticks = _power_ticks(min(pos_lrs), max(pos_lrs)) if pos_lrs else []
    x_ticks = neg_ticks + [lr0_x] + pos_ticks
    x_labels = [_power_label(tick) for tick in neg_ticks] + ["base"] + [_power_label(tick) for tick in pos_ticks]

    if neg_lrs and pos_lrs:
        x_min, x_max = min(neg_lrs) * 1.5, max(pos_lrs) * 1.5
    elif neg_lrs:
        x_min, x_max = min(neg_lrs) * 1.5, linthresh * 2
    elif pos_lrs:
        x_min, x_max = -linthresh * 2, max(pos_lrs) * 1.5
    else:
        x_min, x_max = -linthresh * 2, linthresh * 2

    return "symlog", linthresh, lr0_x, x_min, x_max, x_ticks, x_labels


def plot_probability_shifts(
    trainer: Optional[Any] = None,
    decoder_results: Optional[Dict[str, Any]] = None,
    plotting_data: Optional[Dict[str, Any]] = None,
    class_ids: Optional[List[str]] = None,
    target_class: Optional[str] = None,
    increase_target_probabilities: bool = True,
    output: Optional[str] = None,
    show: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    highlight_non_convergence: Optional[bool] = None,
    return_fig_ax: bool = False,
    split: Optional[Any] = None,
    training_like_df: Optional[Any] = None,
    neutral_df: Optional[Any] = None,
    **kwargs: Any
) -> Any:
    """
    Plot decoder probability shifts vs learning rate for a single target (target_class).

    All plots use the same feature factor (the one that pushes toward target_class, as in rewrite_base_model).
    Base model (lr=0) is prepended to each line; x-tick at lr=0 is labeled "base".

    Subplots:

    - LMS: Language Modeling Score (one line)
    - One per dataset: P(class) on that dataset. The metric used to select the learning rate is

      highlighted (thicker curve, labeled "selection") and the selected (lr, value) is marked with a star.

    Args:
        trainer: Trainer instance (used to get model/data if needed)
        decoder_results: Decoder evaluation result dict (summary entries at top level, e.g. result['3SG'], plus 'grid')
        plotting_data: Extended grid data from analyze_decoder_for_plotting (with probs_by_dataset)
        class_ids: Classes to plot (defaults to all_classes or target_classes from trainer)
        target_class: Target class to focus on (same as rewrite_base_model target_class). Default: first target class.
        increase_target_probabilities: If True, plot for strengthen (default). If False, use weaken summary (target_class_weaken).
        output: Path to save plot. If None and trainer.experiment_dir is set, saves there.
        show: Whether to display the plot
        figsize: Figure size in inches. If None, auto-calculated.
        highlight_non_convergence: Accepted for API compatibility.
        return_fig_ax: If True, return ``(fig, axes)`` and leave the figure open for
            caller-side customization.
        split: Decoder-eval split to evaluate/re-derive against when
            ``decoder_results``/``plotting_data`` aren't already supplied
            (defaults to ``"test"`` several layers down if omitted). Has no
            effect when both are already given.
        training_like_df: Optional evaluation frame to reuse (paired with
            ``neutral_df``) when ``plotting_data`` isn't already supplied --
            pass the same frame you evaluated the decoder against, or this
            function derives its own (narrower, trainer-internal) one. Has
            no effect when ``plotting_data`` is already given.
        neutral_df: Optional neutral frame to reuse, paired with
            ``training_like_df``.
        **kwargs: Additional plotting options. Supported keys include:
            ``title``/``suptitle`` for a figure-level title, ``img_format``, ``dpi``,
            and ``use_cache``.

    Returns:
        Path to saved plot file (or empty string if not saved). If ``return_fig_ax=True``,
        returns ``(fig, axes)``.
    """
    plt = _require_matplotlib()

    # Get decoder_results and plotting_data. split/training_like_df/
    # neutral_df are explicit parameters (not folded into **kwargs)
    # specifically so a caller invoking this function directly (bypassing
    # the trainer's own plot_probability_shifts wrapper, which already
    # threads these through) can't have them silently dropped -- see
    # CLAUDE.md in the study repo for the production bug this exact
    # mistake caused when it happened at a different layer.
    if decoder_results is None and trainer is not None:
        # evaluate_decoder's own `split` defaults to "test", not None --
        # None is a distinct value downstream (e.g. the decoder cache key
        # treats split=None as "none", not "test"). Only pass split when
        # this function's own caller actually gave one.
        decoder_results = trainer.evaluate_decoder(
            use_cache=kwargs.get("use_cache"),
            training_like_df=training_like_df,
            neutral_df=neutral_df,
            **({"split": split} if split is not None else {}),
        )

    if plotting_data is None and trainer is not None:
        plotting_data = trainer.analyze_decoder_for_plotting(
            decoder_results=decoder_results,
            class_ids=class_ids,
            use_cache=kwargs.get("use_cache"),
            intervention_kwargs=(decoder_results or {}).get("intervention_kwargs"),
            split=split,
            training_like_df=training_like_df,
            neutral_df=neutral_df,
        )
    
    if decoder_results is None or plotting_data is None:
        raise ValueError("Must provide decoder_results and plotting_data, or trainer to generate them")
    
    grid = plotting_data.get("plotting_data", plotting_data.get("grid", {}))
    _reserved = {"grid", "plot_path", "plot_paths"}
    summary = {k: v for k, v in decoder_results.items() if k not in _reserved}
    
    # Classes that actually have data in the grid (e.g. from decoder_eval_targets)
    classes_in_grid = set()
    for entry in grid.values():
        probs_by_dataset = entry.get("probs_by_dataset", {})
        for dataset_probs in probs_by_dataset.values():
            if isinstance(dataset_probs, dict):
                classes_in_grid.update(dataset_probs.keys())
    
    # Determine classes to plot
    if class_ids is None:
        if trainer is not None:
            class_ids = trainer.all_classes if trainer.all_classes else trainer.target_classes
        else:
            class_ids = sorted(list(classes_in_grid))
    
    # Only show classes that have data in the grid (exclude classes not in decoder_eval_targets)
    class_ids = [c for c in class_ids if c in classes_in_grid]
    
    if not class_ids:
        raise ValueError(
            "No classes to plot. Provide class_ids or ensure grid contains probs_by_dataset "
            "for the requested classes (classes without decoder_eval_targets are excluded)."
        )
    
    # Resolve target_class and summary key (strengthen vs weaken)
    if target_class is None:
        if trainer is not None and getattr(trainer, "target_classes", None):
            target_class = trainer.target_classes[0]
        elif summary:
            candidates = [k for k in summary.keys() if k != "lms" and not k.endswith("_weaken")]
            target_class = candidates[0] if candidates else None
        if target_class is None and class_ids:
            target_class = list(class_ids)[0]
    if not increase_target_probabilities and target_class and not target_class.endswith("_weaken"):
        summary_key = f"{target_class}_weaken"
    else:
        summary_key = target_class
    base_metric = (target_class[:-7] if target_class and target_class.endswith("_weaken") else target_class)
    if target_class is not None and class_ids and base_metric not in class_ids and summary_key not in (summary or {}):
        target_class = list(class_ids)[0]
        base_metric = target_class
        summary_key = target_class if increase_target_probabilities else f"{target_class}_weaken"
    if target_class is None:
        raise ValueError("No target_class available. Pass target_class (e.g. '3SG') or ensure class_ids/summary are set.")
    
    # Extract learning rates and probabilities from grid
    lr_data = {}  # {feature_factor: {lr: {dataset_class: {class_name: prob, ...}, ...}, ...}, ...}
    metrics_data = {}  # {feature_factor: {lr: {metric: value, ...}, ...}, ...}
    
    for candidate_id, entry in grid.items():
        if candidate_id == "base":
            continue
        
        # Extract feature_factor and lr
        if isinstance(candidate_id, tuple) and len(candidate_id) == 2:
            feature_factor, lr = candidate_id
        elif isinstance(candidate_id, dict):
            feature_factor = candidate_id.get("feature_factor")
            lr = candidate_id.get("learning_rate")
        else:
            continue
        
        if feature_factor is None or lr is None:
            continue
        
        # Extract probs_by_dataset
        probs_by_dataset = entry.get("probs_by_dataset", {})
        if not probs_by_dataset:
            continue
        
        # Store probabilities by feature_factor -> lr -> dataset_class -> class_name
        if feature_factor not in lr_data:
            lr_data[feature_factor] = {}
        if lr not in lr_data[feature_factor]:
            lr_data[feature_factor][lr] = {}
        
        lr_data[feature_factor][lr] = probs_by_dataset
        
        # Extract metrics (LMS, etc.); be strict about expected keys
        if feature_factor not in metrics_data:
            metrics_data[feature_factor] = {}
        if lr not in metrics_data[feature_factor]:
            metrics_data[feature_factor][lr] = {}
        
        if "lms" in entry:
            lms_val = entry["lms"]
            if isinstance(lms_val, dict):
                if "lms" in lms_val:
                    metrics_data[feature_factor][lr]["lms"] = float(lms_val["lms"])
                elif "perplexity" in lms_val:
                    metrics_data[feature_factor][lr]["lms"] = float(lms_val["perplexity"])
                else:
                    raise KeyError("Decoder grid entry 'lms' dict missing both 'lms' and 'perplexity' keys.")
            else:
                metrics_data[feature_factor][lr]["lms"] = float(lms_val)
    
    if not lr_data:
        raise ValueError("No valid grid data found for plotting. Ensure grid contains entries with probs_by_dataset.")
    
    # Determine dataset classes from grid data
    dataset_classes = set()
    for feature_factor_data in lr_data.values():
        for lr_data_entry in feature_factor_data.values():
            dataset_classes.update(lr_data_entry.keys())
    dataset_classes = sorted(list(dataset_classes))
    
    if not dataset_classes:
        raise ValueError("No dataset classes found in grid data.")
    
    feature_factors = sorted(set(lr_data.keys()))
    
    # ff per class for strengthen plots (same rule as evaluate_decoder; see _source_target.py).
    class_to_feature_factor: Dict[str, float] = {}
    if trainer is not None and hasattr(trainer, "get_model"):
        try:
            model = trainer.get_model()
            direction = getattr(model, "feature_class_encoding_direction", None)
            if isinstance(direction, dict):
                source = resolve_model_source(model, trainer)
                target = resolve_model_target(model, trainer)
                signal_kind = resolve_model_signal_kind(model, trainer)
                for class_name in class_ids:
                    if class_name in direction:
                        class_to_feature_factor[class_name] = intervention_feature_factor_from_encoding_direction(
                            direction[class_name],
                            source,
                            target,
                            signal_kind=signal_kind,
                        )
        except ValueError as exc:
            if "ACTIEND activation interventions currently require target='diff'" in str(exc):
                raise
        except Exception:
            pass
    if not class_to_feature_factor and feature_factors:
        default_ff = feature_factors[0]
        class_to_feature_factor = {class_name: default_ff for class_name in class_ids}
    
    # Strengthen: plot only the derived ff for this target class (never another class's orientation).
    is_weaken = summary_key and summary_key.endswith("_weaken")
    ff = class_to_feature_factor.get(base_metric)
    if is_weaken and summary_key and summary_key in (summary or {}):
        ff = summary[summary_key].get("feature_factor")
    if ff is None:
        ff = feature_factors[0] if feature_factors else None
    if ff is None or ff not in lr_data:
        derived = class_to_feature_factor.get(base_metric)
        raise ValueError(
            f"No grid data for strengthen target_class={target_class!r} with feature_factor={derived!r}. "
            f"Grid was evaluated for feature_factors={feature_factors}. "
            f"Re-run evaluate_decoder(target_class={target_class!r}, use_cache=False)."
        )
    
    # lr=0 (base) anchor on the x-axis
    lrs_ff = sorted(lr_data[ff].keys())
    x_scale, linthresh, lr0_x, x_min, x_max, x_ticks, x_labels = _lr_axis_config(lrs_ff)
    
    # Base model data
    base_entry = grid.get("base", {})
    if not base_entry or "lms" not in base_entry:
        raise KeyError("Decoder grid is missing base entry with 'lms'; cannot plot LMS panel reliably.")
    base_probs_by_dataset = base_entry.get("probs_by_dataset", {})
    lm = base_entry["lms"]
    if isinstance(lm, dict):
        if "lms" in lm:
            base_lms_val = float(lm["lms"])
        elif "perplexity" in lm:
            base_lms_val = float(lm["perplexity"])
        else:
            raise KeyError("Base entry 'lms' dict missing both 'lms' and 'perplexity' keys.")
    else:
        base_lms_val = float(lm)
    
    def _with_base(xs, ys, x0, y0):
        """Include base point in line data, sorted by x."""
        return _with_base_point(xs, ys, x0, y0)
    
    # Subplots: 1) LMS, 2+) Dataset probability shifts (selection star on counterfactual or factual line)
    lrs = sorted(lr_data[ff].keys())
    other_classes = [c for c in class_ids if c != base_metric]
    n_subplots = 1 + len(dataset_classes)
    if figsize is None:
        figsize = (8, 2 * n_subplots)
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
    if n_subplots == 1:
        axes = [axes]
    title = kwargs.get("title", kwargs.get("suptitle"))
    suptitle_fontsize = kwargs.get("suptitle_fontsize")
    if title:
        fig.suptitle(
            escape_matplotlib_usetex_text(str(title)),
            fontsize=suptitle_fontsize,
            y=0.995,
        )

    # Plot 1: LMS (one line) — use green to distinguish from probability lines (blue, orange)
    ax_lms = axes[0]
    lms_vals = [metrics_data[ff][lr].get("lms", 0.0) for lr in lrs]
    base_lms = base_lms_val if base_lms_val is not None else (lms_vals[0] if lms_vals else 0.0)
    x_lms, y_lms = _with_base(lrs, lms_vals, lr0_x, base_lms)
    ax_lms.plot(x_lms, y_lms, marker="o", label="LMS", alpha=0.7, color="#2ca02c")
    ax_lms.set_ylabel("LMS")
    ax_lms.set_title("LMS (Language Modeling Score)")
    _apply_lr_xscale(ax_lms, x_scale, linthresh)
    ax_lms.grid(True, alpha=0.3)
    
    # Selection metric (strengthen 3SG → P(3SG) on factual 3PL panel, 3SG curve).
    # Panel keys are factual label_class; see evaluate_base_model probs_by_dataset contract.
    selection_metric_class = base_metric
    if is_weaken:
        selection_dataset_class = base_metric
    elif getattr(getattr(trainer, "config", None), "decoder_eval_prob_on_other_class", True):
        selection_dataset_class = other_classes[0] if other_classes else base_metric
    else:
        selection_dataset_class = base_metric
    selection_metric_label = escape_matplotlib_usetex_text(f"P({selection_metric_class})")
    selected_lr = None
    if summary_key in (summary or {}):
        selected_lr = summary[summary_key].get("learning_rate")
    if selected_lr is not None:
        selected_probs_by_dataset = lr_data[ff].get(selected_lr)
        if selected_probs_by_dataset is None:
            raise ValueError(
                f"Selected decoder learning_rate={selected_lr!r} is absent from plotted grid "
                f"for feature_factor={ff!r}."
            )
        _require_probability_cell(
            selected_probs_by_dataset,
            selection_dataset_class,
            selection_metric_class,
            context=f"feature_factor={ff!r}, learning_rate={selected_lr!r}",
        )

    # Plot 2+: Dataset probability shifts — P(3PL) and P(3SG) on each dataset; highlight selection metric
    missing_probability_cells = set()
    for dataset_idx, dataset_class in enumerate(dataset_classes):
        ax = axes[1 + dataset_idx]
        is_selection_dataset = dataset_class == selection_dataset_class
        for class_name in class_ids:
            probs_c = []
            for lr in lrs:
                prob = _probability_or_nan(lr_data[ff][lr], dataset_class, class_name)
                if math.isnan(prob):
                    missing_probability_cells.add((dataset_class, class_name, lr))
                probs_c.append(prob)
            base_p = _probability_or_nan(base_probs_by_dataset, dataset_class, class_name)
            if math.isnan(base_p):
                missing_probability_cells.add((dataset_class, class_name, "base"))
            x_p, y_p = _with_base(lrs, probs_c, lr0_x, base_p)
            # Emphasize the curve that is the selection metric (used to choose learning rate)
            is_selection_curve = is_selection_dataset and class_name == selection_metric_class
            ax.plot(
                x_p,
                y_p,
                marker="o",
                label=escape_matplotlib_usetex_text(class_name),
                alpha=0.7,
                linewidth=2.5 if is_selection_curve else 1.5,
            )
        if summary_key in (summary or {}) and is_selection_dataset:
            selected_lr = summary[summary_key].get("learning_rate")
            if selected_lr is not None:
                # Use same ff as plotted curves so star lies exactly on the selection-metric curve
                sp = _require_probability_cell(
                    lr_data[ff].get(selected_lr, {}),
                    dataset_class,
                    selection_metric_class,
                    context=f"feature_factor={ff!r}, learning_rate={selected_lr!r}",
                )
                ax.scatter([selected_lr], [sp], marker="*", s=280, zorder=5, alpha=0.95, color="red", label="Selected")
        ax.set_ylabel("Probability")
        ax.set_title(escape_matplotlib_usetex_text(f"Dataset: {dataset_class} — P(class)"))
        _apply_lr_xscale(ax, x_scale, linthresh)
        ax.grid(True, alpha=0.3)
        if dataset_idx == len(dataset_classes) - 1:
            ax.set_xlabel("Learning Rate")
    
    for ax in axes:
        ax.set_xlim(left=x_min, right=x_max)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        try:
            from matplotlib.ticker import NullFormatter

            ax.xaxis.set_minor_formatter(NullFormatter())
        except Exception:
            pass

    if missing_probability_cells:
        examples = sorted(missing_probability_cells, key=lambda item: tuple(map(str, item)))[:5]
        logger.warning(
            "Decoder probability-shift plot has %d missing probability cell(s); plotting them as NaN gaps. "
            "Examples: %s",
            len(missing_probability_cells),
            examples,
        )

    # Vertical line at selected learning rate (all subplots)
    if selected_lr is not None:
        for ax in axes:
            ax.axvline(x=selected_lr, color="gray", linestyle="--", alpha=0.7, zorder=1)
    
    # Shared class legend for dataset probability plots, always above the panels.
    # With a figure title, leave a thin band under the title for the legend.
    if len(dataset_classes) > 0:
        handles, labels = [], []
        for ax in axes[1:]:
            ax_handles, ax_labels = ax.get_legend_handles_labels()
            handles.extend(ax_handles)
            labels.extend(ax_labels)
        seen = set()
        unique_handles, unique_labels = [], []
        for h, l in zip(handles, labels):
            if l == "Selected":
                continue
            if l not in seen:
                seen.add(l)
                unique_handles.append(h)
                unique_labels.append(l)
        fig.legend(
            unique_handles,
            unique_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.97 if title else 1.0),
            ncol=min(len(unique_labels), 6),
            fontsize=8,
            frameon=False,
        )

    if title and len(dataset_classes) > 0:
        layout_rect = [0, 0, 1, 0.90]
    elif title:
        layout_rect = [0, 0, 1, 0.94]
    elif len(dataset_classes) > 0:
        layout_rect = [0, 0, 1, 0.94]
    else:
        layout_rect = [0, 0, 1, 1]
    plt.tight_layout(rect=layout_rect)
    
    # Save plot
    output_path = None
    if output is not None:
        output_path = output
    elif trainer is not None and hasattr(trainer, "experiment_dir") and trainer.experiment_dir:
        # Construct output path
        experiment_dir = trainer.experiment_dir
        run_id = getattr(trainer, "run_id", None)
        img_format = kwargs.get("img_format", "png")
        
        if run_id:
            output_dir = os.path.join(experiment_dir, run_id)
        else:
            output_dir = experiment_dir
        
        filename = f"decoder_probability_shifts.{img_format}"
        output_path = os.path.join(output_dir, filename)
    
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_path, dpi=kwargs.get("dpi", 150), bbox_inches="tight")
        logger.info(f"Saved probability shifts plot to {output_path}")
    
    if show:
        plt.show()
    if return_fig_ax:
        return fig, axes
    if not show:
        plt.close(fig)
    
    return output_path or ""
