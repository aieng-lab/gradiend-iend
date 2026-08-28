"""
Decoder evaluation: grid search over feature_factor and lr, best training_args selection.

DecoderEvaluator runs the grid + cache + summary algorithm; it takes a trainer
(protocol: id, _get_decoder_eval_dataframe, _evaluate_model_for_decoder, _model_for_decoder_eval)
and uses it for data and model access.
"""

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

import pandas as pd
import torch
from gradiend.util.tqdm_utils import gradiend_tqdm

from gradiend.evaluator.decoder_eval_utils import (
    convert_results_to_dict,
    convert_results_to_list,
    parse_grid_candidate_id,
)
from gradiend.model import ModelWithGradiend
from gradiend.model._source_target import (
    intervention_feature_factor_from_encoding_direction,
    resolve_model_source,
    resolve_model_signal_kind,
    resolve_model_target,
)
from gradiend.util.paths import resolve_decoder_grid_cache_path
from gradiend.util.logging import get_logger

logger = get_logger(__name__)

# Panel keys in probability-shift plots: factual label_class / factual_id only.
PROBS_BY_DATASET_GROUPING = "label_class"


@contextmanager
def _decoder_grid_model_context(
    model_with_gradiend: Any,
    *,
    base_model: Any,
    learning_rate: float,
    feature_factor: Union[float, List[float]],
    part: str,
    intervention_kwargs: Optional[Mapping[str, Any]] = None,
):
    """Yield the base model inside the public temporary intervention API."""
    kwargs = dict(intervention_kwargs or {})
    if (
        "direction" not in kwargs
        and (kwargs.get("token_selector") == "encoder_direction" or kwargs.get("activation_gate") == "encoder_direction")
    ):
        kwargs["direction"] = feature_factor
    if (
        "target_encoding" not in kwargs
        and (kwargs.get("token_selector") == "encoder_range" or kwargs.get("activation_gate") == "encoder_range")
    ):
        kwargs["target_encoding"] = feature_factor
    with model_with_gradiend.intervene(
        value=learning_rate,
        feature_factor=feature_factor,
        part=part,
        **kwargs,
    ):
        yield base_model


def _normalize_decoder_intervention_kwargs(
    *,
    token_selector: Optional[Any] = None,
    activation_gate: Optional[Any] = None,
    activation_modules: Optional[Any] = None,
    threshold: Optional[float] = None,
    direction: Optional[Any] = None,
    target_encoding: Optional[Any] = None,
    tolerance: Optional[float] = None,
    intervention_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize application-policy kwargs for decoder-grid interventions.

    ``evaluate_decoder`` historically used ACTIEND's encoder-direction selector
    for temporary intervention evaluation.  Keep that behavior as the default,
    but allow callers to make the orthogonal application-policy axes explicit
    (token scope, encoder gate, activation modules, etc.).
    """
    out: Dict[str, Any] = {}
    if intervention_kwargs:
        out.update(dict(intervention_kwargs))
    direct_values = {
        "token_selector": token_selector,
        "activation_gate": activation_gate,
        "activation_modules": activation_modules,
        "threshold": threshold,
        "direction": direction,
        "target_encoding": target_encoding,
        "tolerance": tolerance,
    }
    for key, value in direct_values.items():
        if value is not None:
            out[key] = value
    if "token_selector" not in out:
        out["token_selector"] = "all" if out.get("activation_gate") is not None else "encoder_direction"
    if "threshold" not in out:
        out["threshold"] = 0.5
    return out


def _jsonable_decoder_intervention_kwargs(value: Any) -> Any:
    """Convert decoder intervention kwargs into a stable JSON-compatible fingerprint."""
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable_decoder_intervention_kwargs(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable_decoder_intervention_kwargs(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _default_decoder_raw_output_path(cache_file: Optional[str]) -> Optional[str]:
    """Default CSV path for per-sample decoder grid probabilities."""
    if cache_file is None:
        return None
    parent = os.path.dirname(os.path.normpath(cache_file))
    raw_parent = os.path.dirname(parent) if os.path.basename(parent) == "decoder_grids" else parent
    stem = os.path.splitext(os.path.basename(cache_file))[0]
    return os.path.join(raw_parent, "decoder_raw", f"{stem}_raw_samples.csv")


def _append_decoder_raw_rows(
    frames: List[pd.DataFrame],
    results: Any,
    *,
    grid_id: str,
    feature_factor: Optional[float],
    learning_rate: Optional[float],
) -> None:
    """Move per-sample rows from an eval result into the combined raw frame list."""
    if not isinstance(results, dict):
        return
    per_row_df = results.pop("_decoder_per_row_df", None)
    if per_row_df is None or not hasattr(per_row_df, "empty") or per_row_df.empty:
        return
    frame = per_row_df.copy()
    frame.insert(0, "grid_id", grid_id)
    frame.insert(1, "feature_factor", feature_factor)
    frame.insert(2, "learning_rate", learning_rate)
    frames.append(frame)


def _write_decoder_raw_rows(raw_output_path: Optional[str], frames: Sequence[pd.DataFrame]) -> Optional[str]:
    """Persist combined per-sample decoder grid probabilities, if requested."""
    if raw_output_path is None or not frames:
        return None
    parent = os.path.dirname(raw_output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    raw_df = pd.concat(list(frames), ignore_index=True)
    raw_df.to_csv(raw_output_path, index=False)
    logger.info("Saved decoder per-sample raw results to %s", raw_output_path)
    return raw_output_path


def _cell_lms(entry: Optional[Mapping[str, Any]]) -> Optional[float]:
    """Read a grid cell's scalar LMS value (``entry["lms"]["lms"]``, per default_extract_candidates)."""
    if not isinstance(entry, Mapping):
        return None
    lms = entry.get("lms")
    if isinstance(lms, Mapping):
        val = lms.get("lms")
        return float(val) if isinstance(val, (int, float)) else None
    if isinstance(lms, (int, float)):
        return float(lms)
    return None


def _bisect_refine_lms_boundary(
    *,
    relevant_results: Dict[Any, Dict[str, Any]],
    pairs: List[Tuple[float, float]],
    lrs: List[float],
    classes_to_eval: Sequence[str],
    class_to_ff: Mapping[str, float],
    selector: Any,
    refine_points: int,
    evaluate_pair: Callable[[float, float], Dict[str, Any]],
) -> None:
    """Binary-search each class's LMS-gate boundary, mutating ``relevant_results``/``pairs``/``lrs`` in place.

    For each class's polarity (``class_to_ff[cls]``), scans the already-evaluated
    learning rates (ascending) for the first adjacent pair where ``lms`` crosses
    ``selector.ratio * base_lms`` — the boundary :class:`LMSThresholdPolicy`-style
    selectors actually optimize around (restrict to lms-passing candidates, then
    argmax metric — so under the usual "metric increases, lms decreases with
    |lr|" pattern, the selected point already sits on this boundary). Bisects
    that interval by geometric midpoint for up to ``refine_points`` steps,
    evaluating each new point via ``evaluate_pair`` and narrowing toward
    whichever side is still on the passing side.

    No-ops (per class, or entirely) when: ``selector`` has no ``ratio``
    attribute (nothing to define the boundary against — not every
    ``SelectionPolicy`` has one); base lms is missing; a class's grid has no
    visible pass→fail transition among adjacent points (already uniform,
    nothing to refine); or a class's polarity was already covered by an
    earlier class in ``classes_to_eval`` sharing the same ``feature_factor``.
    """
    ratio = getattr(selector, "ratio", None)
    if ratio is None:
        return
    base_lms = _cell_lms(relevant_results.get("base"))
    if base_lms is None:
        return
    cutoff = float(ratio) * base_lms

    seen_ff: set = set()
    for cls in classes_to_eval:
        ff = class_to_ff.get(cls)
        if ff is None or ff in seen_ff:
            continue
        seen_ff.add(ff)

        same_ff_lrs = sorted({lr for (f, lr) in pairs if f == ff})
        if len(same_ff_lrs) < 2:
            continue
        lms_by_lr = {lr: _cell_lms(relevant_results.get((ff, lr))) for lr in same_ff_lrs}

        lo = hi = None
        for a, b in zip(same_ff_lrs, same_ff_lrs[1:]):
            la, lb = lms_by_lr.get(a), lms_by_lr.get(b)
            if la is None or lb is None:
                continue
            if la >= cutoff > lb:
                lo, hi = a, b
                break
        if lo is None:
            continue

        for _ in range(refine_points):
            mid = (lo * hi) ** 0.5 if lo > 0 and hi > 0 else (lo + hi) / 2.0
            if mid <= lo or mid >= hi or (ff, mid) in relevant_results:
                break
            result = evaluate_pair(ff, mid)
            pairs.append((ff, mid))
            lrs.append(mid)
            mid_lms = _cell_lms(result)
            if mid_lms is None:
                break
            if mid_lms >= cutoff:
                lo = mid
            else:
                hi = mid


def _decoder_split_cache_key(split: Any) -> str:
    if split is None:
        return "none"
    if isinstance(split, str):
        return split
    if isinstance(split, Sequence) and not isinstance(split, (str, bytes)):
        return "_".join(str(item) for item in split)
    return str(split)


def _refresh_probs_by_dataset_for_plotting(
    trainer: Any,
    *,
    model_with_gradiend: Any,
    base_model: Any,
    tokenizer: Any,
    relevant_results: Dict[Any, Dict[str, Any]],
    training_like_df: Any,
    neutral_df: Any,
    part: str,
    intervention_kwargs: Optional[Mapping[str, Any]] = None,
    max_size_training_like: Optional[int] = None,
    max_size_neutral: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
) -> None:
    """Replace probs_by_dataset on every grid entry using full factual-grouped evaluation.

    Decoder selection may evaluate a row subset; legacy caches may have grouped by
    alternative_id while using the same key names. Never merge — always replace for plots.
    """
    for id_key, entry in list(relevant_results.items()):
        if id_key == "base":
            rewritten = nullcontext(base_model)
        else:
            parsed = parse_grid_candidate_id(id_key, entry)
            if parsed is None:
                continue
            ff, lr = parsed
            rewritten = _decoder_grid_model_context(
                model_with_gradiend,
                base_model=base_model,
                learning_rate=lr,
                feature_factor=ff,
                part=part,
                intervention_kwargs=intervention_kwargs,
            )

        with rewritten as eval_model:
            result = trainer.evaluate_base_model(
                eval_model,
                tokenizer,
                use_cache=False,
                training_like_df=training_like_df,
                neutral_df=neutral_df,
                max_size_training_like=max_size_training_like,
                max_size_neutral=max_size_neutral,
                eval_batch_size=eval_batch_size,
                # Plot refresh only needs probs_by_dataset; never rewrite
                # experiment_dir/decoder_row_wise_scores.csv per grid cell.
                export_row_wise_csv=False,
            )

        if isinstance(result, dict) and result.get("probs_by_dataset"):
            entry["probs_by_dataset"] = result["probs_by_dataset"]
            entry["_probs_by_dataset_grouping"] = PROBS_BY_DATASET_GROUPING


def _plot_all_target_classes(
    trainer: Any,
    summary: Dict[str, Any],
    relevant_results: Dict[Any, Dict[str, Any]],
    *,
    experiment_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    show: Optional[bool] = None,
    increase_target_probabilities: bool = True,
    plot_keys_override: Optional[List[str]] = None,
    plot_kwargs: Optional[Dict[str, Any]] = None,
    intervention_kwargs: Optional[Mapping[str, Any]] = None,
    training_like_df: Optional[Any] = None,
    neutral_df: Optional[Any] = None,
    split: Optional[Any] = None,
) -> List[str]:
    """Plot probability shifts once per target class for the given direction. Returns list of saved plot paths.
    When show is None and plot was requested, defaults to True so the plot is displayed.
    When plot_keys_override is set (e.g. when user passed target_class), only those keys are plotted,
    so we do not plot for internal result keys (e.g. 3PL when strengthening 3SG).

    ``training_like_df``/``neutral_df``/``split`` are forwarded to
    ``trainer.plot_probability_shifts`` -> ``analyze_decoder_for_plotting``
    so the plot-refresh pass reuses the SAME frame the decoder grid was
    actually evaluated against, instead of silently re-deriving a
    (potentially narrower, one-pole-scoped) frame from the trainer's own
    internal data. Confirmed production bug when this wasn't threaded
    through: a bisection-refined grid cell's plot-refresh call re-derived
    training_like_df with split defaulting to "test" and no caller frame
    reused, replacing every rival class's panel with the trainer's own
    internal one-pole view -- see CLAUDE.md in the study repo.
    """
    if plot_keys_override is not None:
        plot_keys = [k for k in plot_keys_override if k in summary]
    elif increase_target_probabilities:
        plot_keys = [k for k in summary.keys() if not k.endswith("_weaken")]
    else:
        plot_keys = [k for k in summary.keys() if k.endswith("_weaken")]
    if not plot_keys:
        return []
    decoder_results = {
        **summary,
        "grid": relevant_results,
        "intervention_kwargs": dict(intervention_kwargs or {}),
    }
    cfg = getattr(trainer, "config", None)
    img_format = getattr(cfg, "img_format", "png") if cfg else "png"
    base_plot_kwargs = dict(plot_kwargs or {})
    if show is not None:
        base_plot_kwargs["show"] = show
    elif "show" not in base_plot_kwargs:
        base_plot_kwargs["show"] = True
    base_plot_kwargs.setdefault("img_format", img_format)
    paths: List[str] = []
    explicit_output = base_plot_kwargs.get("output") is not None
    for key in plot_keys:
        target_class = key[:-7] if key.endswith("_weaken") else key
        output_path = None
        default_output_path = None
        if experiment_dir and str(experiment_dir).strip() and not explicit_output:
            safe_name = str(target_class).replace("/", "_").replace("\\", "_").replace(":", "_")
            out_dir = os.path.join(experiment_dir, run_id or "")
            output_path = os.path.join(out_dir, f"decoder_probability_shifts_{safe_name}.{img_format}")
            if len(plot_keys) == 1:
                default_output_path = os.path.join(out_dir, f"decoder_probability_shifts.{img_format}")
        call_kwargs = dict(base_plot_kwargs)
        if output_path is not None:
            call_kwargs["output"] = output_path
        # training_like_df/neutral_df/split are explicit parameters all the
        # way down this call chain (trainer.plot_probability_shifts ->
        # evaluator.plot_probability_shifts -> visualizer.plot_probability_
        # shifts -> analyze_decoder_for_plotting), each treating an explicit
        # None the same as "not supplied" -- so passing them unconditionally
        # here is safe and requires no "only if not None" dance. This used
        # to be a **kwargs bag with a hand-picked subset silently forwarded
        # at one layer (visualizer.py) and not others, which is exactly what
        # let this same data go missing in production -- see CLAUDE.md in
        # the study repo.
        path = trainer.plot_probability_shifts(
            decoder_results=decoder_results,
            target_class=target_class,
            increase_target_probabilities=increase_target_probabilities,
            training_like_df=training_like_df,
            neutral_df=neutral_df,
            split=split,
            **call_kwargs,
        )
        if path:
            paths.append(path)
            if default_output_path and os.path.normpath(path) != os.path.normpath(default_output_path):
                try:
                    os.makedirs(os.path.dirname(default_output_path), exist_ok=True)
                    shutil.copyfile(path, default_output_path)
                    paths.append(default_output_path)
                    logger.info("Saved default decoder probability-shift plot to %s", default_output_path)
                except Exception as e:
                    logger.warning("Could not save default decoder probability-shift plot %s: %s", default_output_path, e)
    return paths


# ============================
# Feature-factor defaults (contract: gradiend.model._source_target)
# ============================


def derive_default_feature_factor(
    trainer: Any,
    model_with_gradiend: Any = None,
    class_name: str = None,
) -> float:
    """
    Derive a single default feature factor for decoder eval.

    CONTRACT (do not change without explicit design review):

    - Gradient-space GRADIEND uses the historical weight-rewrite convention:

      factual/diff sources use ``-feature_class_encoding_direction[class_name]``;
      alternative sources use ``+feature_class_encoding_direction[class_name]``.

    - Activation-space ACTIEND uses direct activation-displacement semantics,

      so its feature factor is the negated GRADIEND weight-rewrite feature
      factor. For the usual ``target="diff"`` steering setup, that means
      factual/diff sources use ``+feature_class_encoding_direction[class_name]``
      and alternative sources use ``-feature_class_encoding_direction[class_name]``.

    - Rewrite/hook orientation is **only** this ``feature_factor`` ×

      decoder(latent); never flip LR. ``SignalScope`` chooses activation sites
      and does not participate in the sign convention.

    Args:
        trainer: Trainer-like object used as fallback for model loading and
            class-label inference.
        model_with_gradiend: Optional loaded model or model path. If omitted,
            ``trainer.get_model()`` is used when available.
        class_name: Target feature class for which the strengthening direction
            should be derived. Required.

    Returns:
        Feature factor that pushes decoder evaluation toward ``class_name``.

    Raises:
        ValueError: If ``class_name`` is omitted or no direction can be inferred.
    """
    model = model_with_gradiend
    if model is None and hasattr(trainer, "get_model"):
        model = trainer.get_model()
    elif isinstance(model, str):
        trust_remote_code = getattr(getattr(trainer, "_training_args", None), "trust_remote_code", False)
        model = ModelWithGradiend.from_pretrained(model, trust_remote_code=trust_remote_code)

    if class_name is None:
        raise ValueError("class_name is required to derive a default feature factor.")

    direction = getattr(model, "feature_class_encoding_direction", None) if model is not None else None
    source = resolve_model_source(model, trainer)
    target = resolve_model_target(model, trainer)
    signal_kind = resolve_model_signal_kind(model, trainer)
    if isinstance(direction, dict) and class_name in direction:
        return intervention_feature_factor_from_encoding_direction(
            direction[class_name],
            source,
            target,
            signal_kind=signal_kind,
        )

    # Fallback: derive from trainer encoding labels / pair when model was created
    # before data load (e.g. in-memory after train)
    labels_fn = getattr(trainer, "get_feature_class_encoding_labels", None)
    class_labels = labels_fn() if callable(labels_fn) else None
    if not class_labels:
        pair = getattr(trainer, "pair", None)
        if pair and len(pair) >= 2:
            class_labels = {pair[0]: 1.0, pair[1]: -1.0}
            classes = getattr(trainer, "target_classes", None) or getattr(trainer, "all_classes", None) or []
            for c in classes:
                if c not in class_labels:
                    class_labels[c] = 0.0
    if class_labels and class_name in class_labels:
        return intervention_feature_factor_from_encoding_direction(
            class_labels[class_name],
            source,
            target,
            signal_kind=signal_kind,
        )

    raise ValueError(
        "Cannot derive default feature factor for class '%s': model does not have feature_class_encoding_direction (%s) or class not found in it." % (class_name, direction)
    )


def derive_feature_factor_for_class(
    trainer: Any,
    model_with_gradiend: Any = None,
    class_name: Optional[str] = None,
) -> float:
    """Derive the feature factor that pushes decoder evaluation toward one class.

    Args:
        trainer: Trainer-like object used for model and class-direction context.
        model_with_gradiend: Optional loaded model or model path.
        class_name: Target feature class. Required.

    Returns:
        Feature factor that strengthens ``class_name``.
    """
    if class_name is None:
        raise ValueError("class_name is required to derive a feature factor.")
    return derive_default_feature_factor(trainer, model_with_gradiend, class_name=class_name)


def default_decoder_feature_factors(
    trainer: Any,
    model_with_gradiend: Any = None,
) -> List[float]:
    """Return default feature factors for all trainer target classes.

    Args:
        trainer: Trainer-like object used to resolve target classes and model
            direction metadata.
        model_with_gradiend: Optional loaded model or model path. If omitted,
            ``trainer.get_model()`` is used by the per-class resolver.

    Returns:
        List of feature factors, one per target class.

    Raises:
        ValueError: If target classes or feature directions cannot be inferred.
    """
    classes = getattr(trainer, "target_classes", None)
    if not classes and hasattr(trainer, "config"):
        classes = getattr(trainer.config, "target_classes", None)
    if not classes and hasattr(trainer, "get_target_feature_classes"):
        classes = trainer.get_target_feature_classes()

    if not classes:
        raise ValueError(
            "Could not derive default feature factors automatically: target_classes are not set. "
            "Set target_classes on TextPredictionConfig (e.g. target_classes=['class_a', 'class_b']), "
            "or ensure your training data is loaded and contains exactly two classes so they can be inferred."
        )

    return [derive_feature_factor_for_class(trainer, model_with_gradiend, cls) for cls in classes]


# ============================
# Metric selection
# ============================
CandidateId = Any  # e.g. (feature_factor, lr)


@dataclass(frozen=True)
class Candidate:
    id: CandidateId
    lms: float
    metrics: Mapping[str, float]  # scalar metrics used for selection


@dataclass(frozen=True)
class BaseContext:
    base_lms: Optional[float]


class SelectionPolicy(Protocol):
    def select(self, metric: str, candidates: Sequence[Candidate], ctx: BaseContext) -> Optional[Candidate]:
        """Return the selected candidate for ``metric`` or ``None``.

        Args:
            metric: Metric name to optimize.
            candidates: Candidate grid entries available for this metric.
            ctx: Base evaluation context, currently carrying base LMS.
        """
        ...


@dataclass(frozen=True)
class LMSThresholdPolicy:
    """
    Restrict to candidates with lms >= ratio * base_lms, then pick argmax(metric).

    Fallback when none pass threshold:
      pick smallest |lr| among candidates with lr != 0
      (keeps your previous behavior, but expressed as a policy).
    """
    ratio: float = 0.99
    lr_from_id: Callable[[CandidateId], float] = (
        lambda cid: float(cid["learning_rate"]) if isinstance(cid, dict) else float(cid[1])
    )
    require_base_lms: bool = True

    def select(self, metric: str, candidates: Sequence[Candidate], ctx: BaseContext) -> Optional[Candidate]:
        """Select the best candidate above the LMS threshold for ``metric``.

        Args:
            metric: Metric name to optimize.
            candidates: Candidate grid entries available for this metric.
            ctx: Base evaluation context with ``base_lms``.
        """
        base_lms = ctx.base_lms
        if base_lms is None:
            if self.require_base_lms:
                raise ValueError("Base model lms missing; cannot apply LMSThresholdPolicy.")
            return max(candidates, key=lambda c: c.metrics.get(metric, float("-inf")), default=None)

        cutoff = self.ratio * base_lms
        passing = [c for c in candidates if c.lms >= cutoff]
        if passing:
            return max(passing, key=lambda c: c.metrics.get(metric, float("-inf")), default=None)

        non_zero = [c for c in candidates if self.lr_from_id(c.id) != 0]
        if not non_zero:
            return None
        return min(non_zero, key=lambda c: abs(self.lr_from_id(c.id)), default=None)


@dataclass(frozen=True)
class LMSTimesMetricPolicy:
    """Pick argmax(metric * lms)."""
    def select(self, metric: str, candidates: Sequence[Candidate], ctx: BaseContext) -> Optional[Candidate]:
        """Select the candidate maximizing ``metric * lms``.

        Args:
            metric: Metric name to optimize.
            candidates: Candidate grid entries available for this metric.
            ctx: Base evaluation context. Accepted for policy compatibility.
        """
        return max(
            candidates,
            key=lambda c: c.metrics.get(metric, float("-inf")) * c.lms,
            default=None,
        )


def default_extract_candidates(results: Mapping[Any, Mapping[str, Any]]) -> Tuple[List[Candidate], BaseContext]:
    """
    Convert current `results` format to a list of Candidates.

    Metrics convention:

      - probs -> keys "<class_name>" (strengthen selection scalar; counterfactual or same-panel)
      - probs_factual -> "<class_name>_weaken" only (1 - P(class) on class dataset; weaken mode)

      - any scalar numeric field at top-level of entry (excluding lms/probs/probs_factual) -> metric with same key
    """
    base_lms = float(results["base"]["lms"]["lms"]) if "base" in results else None

    candidates: List[Candidate] = []
    for cid, entry in results.items():
        if cid == "base":
            continue

        lms = float(entry["lms"]["lms"])
        metrics: Dict[str, float] = {}

        probs = entry.get("probs") or {}
        for cls, p in probs.items():
            metrics[str(cls)] = float(p)

        probs_factual = entry.get("probs_factual") or {}
        for cls, p in probs_factual.items():
            metrics[f"{cls}_weaken"] = 1.0 - float(p)
            # Same-panel strengthen may only populate probs_factual (legacy grids).
            cls_key = str(cls)
            if cls_key not in metrics:
                metrics[cls_key] = float(p)

        for k, v in entry.items():
            if k in ("probs", "lms", "probs_factual", "probs_by_dataset"):
                continue
            if isinstance(v, (int, float)):
                metrics[k] = float(v)
                # Legacy top-level ``<class>_factual`` → strengthen metric ``<class>``.
                key = str(k)
                if key.endswith("_factual"):
                    base = key[: -len("_factual")]
                    if base and base not in metrics:
                        metrics[base] = float(v)

        candidates.append(Candidate(id=entry["id"], lms=lms, metrics=metrics))

    return candidates, BaseContext(base_lms=base_lms)


def _decoder_results_support_metrics(
    results: Mapping[Any, Mapping[str, Any]],
    metrics: Sequence[str],
    *,
    extractor: Callable[[Mapping[Any, Mapping[str, Any]]], Tuple[List[Candidate], BaseContext]] = default_extract_candidates,
) -> Tuple[bool, List[str]]:
    """Whether cached decoder-grid rows expose every metric needed for summarization."""
    if not metrics:
        return True, []
    try:
        candidates, _ = extractor(results)
    except Exception:
        return False, list(metrics)
    available: set = set()
    for candidate in candidates:
        available.update(candidate.metrics.keys())
    missing = [m for m in metrics if m not in available]
    return (not missing), missing


def _decoder_cache_selection_context(
    *,
    prob_on_other_class: bool,
    increase_target_probabilities: bool,
    metrics_for_summary: Sequence[str],
    classes_to_eval: Sequence[str],
) -> Dict[str, Any]:
    """Fingerprint for decoder-grid cache invalidation when selection contract changes."""
    return {
        "prob_on_other_class": bool(prob_on_other_class),
        "increase_target_probabilities": bool(increase_target_probabilities),
        "metrics_for_summary": sorted(str(m) for m in metrics_for_summary),
        "classes_to_eval": sorted(str(c) for c in classes_to_eval),
    }


def _decoder_cache_selection_matches(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    cached = payload.get("selection_context")
    if not isinstance(cached, dict):
        return True
    return all(cached.get(key) == value for key, value in expected.items())


def _decoder_cache_payload(
    *,
    part: str,
    split_cache_key: str,
    max_size_training_like: Any,
    max_size_neutral: Any,
    feature_factors: Sequence[float],
    lrs: Sequence[float],
    intervention_cache_fingerprint: Any,
    results: Any,
    selection_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "part": part,
        "split": split_cache_key,
        "max_size_training_like": max_size_training_like,
        "max_size_neutral": max_size_neutral,
        "feature_factors": list(feature_factors),
        "lrs": list(lrs),
        "intervention_kwargs": intervention_cache_fingerprint,
        "results": results,
    }
    if selection_context is not None:
        payload["selection_context"] = dict(selection_context)
    return payload


def compute_metric_summaries(
    results: Mapping[Any, Mapping[str, Any]],
    metrics: Sequence[str],
    *,
    selector: SelectionPolicy,
    extractor: Callable[[Mapping[Any, Mapping[str, Any]]], Tuple[List[Candidate], BaseContext]] = default_extract_candidates,
    feature_factor_from_id: Callable[[CandidateId], float] = lambda cid: cid[0],
    lr_from_id: Callable[[CandidateId], float] = lambda cid: cid[1],
    empty_default_id: str = "base",
    class_to_ff: Optional[Mapping[str, float]] = None,
    explicit_feature_factors: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Summarize multiple metrics in one pass.

    For strengthen metrics (e.g. ``"3SG"``): only consider candidates whose
    ``feature_factor`` strengthens that class (``class_to_ff[metric]``).

    For metrics ending in "_weaken" (e.g. "3PL_weaken"): only consider candidates whose
    feature_factor pushes toward the *other* class (weaken = use opposite direction).
    class_to_ff must map class_name -> feature_factor that strengthens it. The weaken
    summary selects from candidates with ff in {class_to_ff[c] for c != base_class}.

    Args:
        results: Raw decoder-grid result mapping, including a ``"base"`` entry
            with LMS and one entry per candidate.
        metrics: Metric names to summarize, e.g. target class ids or
            ``"<class>_weaken"``.
        selector: Selection policy used to choose the best candidate per metric.
        extractor: Function converting raw results into candidates and base LMS
            context.
        feature_factor_from_id: Function extracting feature factor from a
            candidate id.
        lr_from_id: Function extracting learning rate from a candidate id.
        empty_default_id: Fallback id used when no candidate can be selected.
        class_to_ff: Optional mapping from class id to its strengthening feature
            factor. Used to restrict strengthen and weaken summaries to candidates
            with the correct rewrite orientation per class.
        explicit_feature_factors: True when the caller passed an explicit
            ``feature_factors=`` to ``evaluate_decoder`` rather than letting it
            derive a default sweep from ``class_to_ff``. When True,
            ``class_to_ff`` is not used to filter candidates for a metric --
            the caller already chose exactly which feature_factor(s) to
            evaluate (e.g. an opposite-polarity random control), and this
            function must respect that instead of silently re-deriving its
            own expectation and rejecting a grid the caller built on purpose.

    Returns:
      {metric: {"value", "feature_factor", "learning_rate", "id", "strengthen"}}
    """
    candidates, ctx = extractor(results)

    available_metrics: set = set()
    for c in candidates:
        available_metrics.update(c.metrics.keys())
    normalized_metrics: List[str] = list(metrics)

    missing = [m for m in normalized_metrics if m not in available_metrics]
    if missing:
        raise ValueError(
            "Requested metrics not present in decoder results: %s. Available metrics: %s"
            % (missing, sorted(available_metrics))
        )

    if not candidates:
        return {m: {"value": 0.0, "feature_factor": 0.0, "learning_rate": 0.0, "id": empty_default_id} for m in normalized_metrics}

    def _fallback_when_none(cands: List[Candidate]) -> Optional[Candidate]:
        """Pick candidate with lr != 0 and smallest absolute value when selector returns None."""
        non_zero = [c for c in cands if lr_from_id(c.id) != 0]
        if not non_zero:
            return None
        return min(non_zero, key=lambda c: abs(lr_from_id(c.id)))

    def _candidates_for_metric(metric: str) -> List[Candidate]:
        if not class_to_ff:
            return list(candidates)
        if explicit_feature_factors:
            # class_to_ff's job is disambiguating which subset of a grid the
            # package *itself* derived a default sweep for counts as
            # "strengthen"/"weaken" for this metric. When the caller instead
            # passed an explicit feature_factors=[...] (e.g. an
            # opposite-polarity random-control re-query deliberately
            # evaluating a class's *non-canonical* direction on purpose),
            # that derivation never happened -- the grid already is exactly
            # what was explicitly requested, and second-guessing it against
            # class_to_ff[metric] would silently override the caller's own
            # manual parameter instead of respecting it. See gradiend-sae's
            # CLAUDE.md ("opposite-polarity causal control failed") for the
            # concrete crash this caused: pronoun_number's
            # actiend:plural-singular:plural random control passed
            # feature_factors=[opp_ff], and this function's old unconditional
            # class_to_ff[metric] filter found zero matching candidates and
            # raised -- even though the caller had explicitly asked for
            # exactly that ff and needed no disambiguation at all.
            #
            # This is deliberately narrower than "grid has only one ff
            # present": an *auto-derived* single-sided grid (class_to_ff
            # never consulted by the caller) still must raise -- see
            # test_strengthen_missing_feature_factor_error_lists_grid_ffs.
            return list(candidates)
        if metric.endswith("_weaken"):
            base = metric[:-7]
            strengthen_ff = class_to_ff.get(base)
            if strengthen_ff is None:
                return list(candidates)
            other_ffs = {float(ff) for cls, ff in class_to_ff.items() if cls != base}
            if not other_ffs:
                return list(candidates)
            return [c for c in candidates if feature_factor_from_id(c.id) in other_ffs]
        strengthen_ff = class_to_ff.get(metric)
        if strengthen_ff is None:
            return list(candidates)
        target_ff = float(strengthen_ff)
        return [c for c in candidates if float(feature_factor_from_id(c.id)) == target_ff]

    # Basic sanity: we must have a valid base LMS for LMS-aware policies.
    if ctx.base_lms is None:
        raise ValueError(
            "Base LMS is missing in decoder results; expected 'base' entry with nested 'lms[\"lms\"]'. "
            "Cannot apply LMS-based selection policy."
        )

    summary: Dict[str, Dict[str, Any]] = {}
    for metric in normalized_metrics:
        filtered = _candidates_for_metric(metric)
        logger.debug(
            "compute_metric_summaries: metric=%r class_to_ff=%s candidate_ffs=%s "
            "filtered_ffs=%s",
            metric,
            class_to_ff,
            sorted({float(feature_factor_from_id(c.id)) for c in candidates}),
            sorted({float(feature_factor_from_id(c.id)) for c in filtered}),
        )
        chosen = selector.select(metric, filtered, ctx) if filtered else None
        if chosen is None:
            chosen = _fallback_when_none(filtered) if filtered else None
        if chosen is None:
            if not metric.endswith("_weaken") and class_to_ff and metric in class_to_ff:
                available_ffs = sorted(
                    {float(feature_factor_from_id(c.id)) for c in candidates}
                )
                logger.error(
                    "compute_metric_summaries: no candidate for strengthen metric=%r -- "
                    "class_to_ff=%s (required ff=%s), grid candidate ffs=%s, "
                    "filtered (post _candidates_for_metric) ffs=%s",
                    metric,
                    class_to_ff,
                    class_to_ff[metric],
                    available_ffs,
                    sorted({float(feature_factor_from_id(c.id)) for c in filtered}),
                )
                raise ValueError(
                    "No decoder grid candidates for strengthen class %r "
                    "(required feature_factor=%s). Grid feature_factors present: %s. "
                    "Pass feature_factors that include the strengthen factor for this class, "
                    "or re-run evaluate_decoder(target_class=%r, use_cache=False)."
                    % (metric, class_to_ff[metric], available_ffs, metric)
                )
            summary[metric] = {
                "value": 0.0,
                "feature_factor": 0.0,
                "learning_rate": 0.0,
                "id": empty_default_id,
                "strengthen": not metric.endswith("_weaken"),
                "lms": float(ctx.base_lms),
                "base_lms": float(ctx.base_lms),
            }
            continue

        summary[metric] = {
            "value": float(chosen.metrics.get(metric, 0.0)),
            "feature_factor": float(feature_factor_from_id(chosen.id)),
            "learning_rate": float(lr_from_id(chosen.id)),
            "id": chosen.id,
            "strengthen": not metric.endswith("_weaken"),
            "lms": float(chosen.lms),
            "base_lms": float(ctx.base_lms),
        }

    return summary


# ============================
# DecoderEvaluator
# ============================


class DecoderEvaluator:
    """
    Decoder evaluation: grid search over (feature_factor, lr), cache, and compute best selection summary.
    Uses trainer for eval dataframe and model evaluation.
    """

    def evaluate_decoder(
        self,
        trainer: Any,
        model_with_gradiend: Any = None,
        feature_factors: Optional[List[float]] = None,
        lrs: Optional[List[float]] = None,
        part: str = "decoder",
        token_selector: Optional[Any] = None,
        activation_gate: Optional[Any] = None,
        activation_modules: Optional[Any] = None,
        threshold: Optional[float] = None,
        direction: Optional[Any] = None,
        target_encoding: Optional[Any] = None,
        tolerance: Optional[float] = None,
        intervention_kwargs: Optional[Mapping[str, Any]] = None,
        output_path: Optional[str] = None,
        raw_output_path: Optional[str] = None,
        selector: Optional[SelectionPolicy] = None,
        summary_extractor: Callable[[Mapping[Any, Mapping[str, Any]]], Tuple[List[Candidate], BaseContext]] = default_extract_candidates,
        summary_feature_factor_from_id: Callable[[CandidateId], float] = lambda cid: cid['feature_factor'] if isinstance(cid, dict) else cid[0],
        summary_lr_from_id: Callable[[CandidateId], float] = lambda cid: cid['learning_rate'] if isinstance(cid, dict) else cid[1],
        summary_empty_default_id: str = "base",
        use_cache: Optional[bool] = None,
        split: Optional[Any] = "test",
        max_size: Optional[int] = None,
        max_size_training_like: Optional[int] = None,
        max_size_neutral: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
        training_like_df: Optional[Any] = None,
        neutral_df: Optional[Any] = None,
        summary_metrics: Optional[Sequence[str]] = None,
        target_class: Optional[Union[str, List[str]]] = None,
        increase_target_probabilities: bool = True,
        plot: bool = False,
        show: Optional[bool] = None,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        refine_points: int = 0,
    ) -> Dict[str, Any]:
        """
        Run decoder grid evaluation and return summary + grid for one direction (strengthen or weaken).

        Only the dataset and feature-factor combinations required for the requested direction are
        computed. Use increase_target_probabilities=True (default) for strengthen, False for weaken.

        Args:
            trainer: Trainer (protocol) with get_model, _model_for_decoder_eval, _get_decoder_eval_dataframe,
                     and evaluate_base_model.
            model_with_gradiend: ModelWithGradiend instance or path. If None, uses trainer.get_model().
            feature_factors: List of feature factors to test. If None, derived from direction and target classes.
            lrs: List of learning rates to test.
            part: which part of GRADIEND is used to derive GRADIEND-modified models (options:  'encoder-weight' |
                'decoder-weight' | 'decoder-bias' | 'decoder-sum' | 'decoder'). All options besides `decoder` are
                independent of the feature factor (e.g., using the encoder weights as update direction), while `decoder`
                computes the update direction via dec(feature_factor) (and is the default).
            token_selector: Optional intervention token selector forwarded to ``ModelWithGradiend.intervene``.
                Defaults to the historical decoder-eval selector ``"encoder_direction"``.
            activation_gate: Optional ACTIEND encoder gate composed with the token selector.
            activation_modules: Optional ACTIEND activation module filter, e.g. one layer name.
            threshold: Optional encoder-selector threshold. Defaults to ``0.5`` when omitted.
            direction: Optional encoder-direction value. When omitted, ACTIEND resolves it from feature_factor.
            target_encoding: Optional encoder-range target. When omitted, ACTIEND resolves it from feature_factor.
            tolerance: Optional encoder-range tolerance.
            intervention_kwargs: Additional low-level kwargs forwarded to ``ModelWithGradiend.intervene``.
                Direct parameters above override matching keys in this mapping.
            output_path: Optional explicit cache path. Overrides experiment_dir-based cache path.
            raw_output_path: Optional CSV path for per-sample decoder probabilities for every
                evaluated grid entry. If omitted and
                ``trainer.config.decoder_eval_export_row_wise_csv`` is True, a sibling
                ``decoder_raw/*_raw_samples.csv`` path is derived from the grid cache path.
            selector: SelectionPolicy, e.g. LMSThresholdPolicy(ratio=0.99) or LMSTimesMetricPolicy().
            summary_extractor: Candidate extractor for summary computation. Use a custom extractor to add
                derived metrics (e.g. bpi, fpi, mpi) to candidates; then pass summary_metrics so they are summarized.
            summary_feature_factor_from_id: Function to extract feature_factor from candidate id.
            summary_lr_from_id: Function to extract lr from candidate id.
            summary_empty_default_id: Fallback id used when no candidate is selected (for comparison with base).
                When the selector returns None, we first try the candidate with learning_rate != 0 and smallest
                absolute value; only if none exists do we use this default (representing the base model).
            use_cache: If True, use cached results when available; if False, recompute.
            split: Dataset split used for training-like decoder evaluation rows.
                Defaults to ``"test"``.
            max_size: Shared evaluation-size alias. If set and
                explicit decoder caps are omitted, caps both training-like decoder
                rows and neutral/LMS rows.
            max_size_training_like: Maximum size for generated training-like eval data.
            max_size_neutral: Maximum size for generated neutral eval data (and LMS text cap).
            eval_batch_size: Common eval batch size used for LMS.
            training_like_df: Optional explicit training-like DataFrame for probability scoring.
            neutral_df: Optional explicit neutral DataFrame for LMS scoring.
            summary_metrics: Optional list of metric names to summarize. If None, uses direction and target classes
                (see increase_target_probabilities).
            target_class: If set, evaluate only for this target class (or list of classes). Restricts
                feature factors and datasets to those needed for the given class(es) for efficiency.
                When None, evaluates for all trainer target classes.
            increase_target_probabilities: If True (default), compute **strengthen** summaries only (keys e.g. "3SG", "3PL").
                If False, compute **weaken** summaries only (keys e.g. "3SG_weaken", "3PL_weaken"). Only the
                dataset–feature-factor combinations required for the chosen direction are evaluated.
            plot: If True, after selection run any missing dataset evaluations needed for plotting,
                update cache incrementally, then call the trainer's plot_probability_shifts.
            show: If True, display the plot (e.g. plt.show()). If False, only save to file. When None
                and plot=True, defaults to True (same as evaluate_encoder: plot implies show).
            plot_kwargs: Optional dict of options forwarded to plot_probability_shifts when plot=True.
                E.g. plot_kwargs=dict(figsize=(5, 3), show=False). The evaluate_decoder ``show`` argument
                overrides plot_kwargs[\"show\"] when set.
            refine_points: If > 0, after the requested ``lrs`` grid is evaluated, binary-search up to
                this many additional points per (strengthen) target class to sharpen the LMS-gate
                boundary the grid found only coarsely. For each class, finds the adjacent pair of
                evaluated learning rates where ``lms`` crosses ``ratio * base_lms`` (the boundary
                ``selector`` — normally :class:`LMSThresholdPolicy` — actually optimizes around), then
                bisects that interval by geometric midpoint, evaluating and narrowing toward the
                boundary each step. Assumes ``lms`` is monotonically non-increasing in ``|lr|`` (more
                aggressive steering degrades fluency) and the target metric is monotonically
                non-decreasing in ``|lr|`` within the passing region — true for typical strengthen
                sweeps. Skips classes whose grid has no visible pass/fail transition among adjacent
                points (nothing to bisect) and any class evaluated under a ``selector`` without a
                ``ratio`` attribute (nothing to define the boundary against). No-op when
                ``increase_target_probabilities=False`` (weaken mode) or ``target_class`` is unset.
                New points are merged into ``grid``/the returned summary exactly like the coarse grid
                (and into the on-disk cache) — this is not a second cache-invalidating call, the
                extra points are computed in the same pass.

        Returns:
            Flat dict with:

              - For strengthen (increase_target_probabilities=True): one entry per target class (e.g. dec_result['3SG']).
              - For weaken (increase_target_probabilities=False): one entry per target class with \"_weaken\" suffix

                (e.g. dec_result['3SG_weaken']).

              - Each summary entry contains selected metric `value`, `feature_factor`, `learning_rate`, `id`,

                a `strengthen` flag, and LMS fields (`lms`, `base_lms`).

              - 'grid': candidate id -> full evaluation results.
              - When plot=True, also 'plot_paths' and 'plot_path'.
        """
        logger.info(f"Starting decoder evaluation with part={part}")
        use_cache = trainer._resolve_artifact_use_cache(use_cache, fallback=False)
        if max_size_training_like is None:
            max_size_training_like = max_size
        if max_size_neutral is None:
            max_size_neutral = max_size
        trainer_config = getattr(trainer, "config", None)
        training_args = getattr(trainer, "_training_args", None)
        if max_size_training_like is None:
            if hasattr(trainer, "_default_from_training_args"):
                max_size_training_like = trainer._default_from_training_args(
                    max_size_training_like, "decoder_eval_max_size_training_like"
                )
            elif training_args is not None:
                max_size_training_like = getattr(training_args, "decoder_eval_max_size_training_like", None)
        if max_size_training_like is None and trainer_config is not None:
            max_size_training_like = getattr(trainer_config, "decoder_eval_lms_max_samples", None)
        if max_size_neutral is None:
            if hasattr(trainer, "_default_from_training_args"):
                max_size_neutral = trainer._default_from_training_args(
                    max_size_neutral, "decoder_eval_max_size_neutral"
                )
            elif training_args is not None:
                max_size_neutral = getattr(training_args, "decoder_eval_max_size_neutral", None)
        if max_size_neutral is None and trainer_config is not None:
            max_size_neutral = getattr(trainer_config, "decoder_eval_lms_max_samples", None)

        if selector is None:
            selector = LMSThresholdPolicy(ratio=0.99)

        resolved_intervention_kwargs = _normalize_decoder_intervention_kwargs(
            token_selector=token_selector,
            activation_gate=activation_gate,
            activation_modules=activation_modules,
            threshold=threshold,
            direction=direction,
            target_encoding=target_encoding,
            tolerance=tolerance,
            intervention_kwargs=intervention_kwargs,
        )
        intervention_cache_fingerprint = _jsonable_decoder_intervention_kwargs(resolved_intervention_kwargs)

        raw_model = model_with_gradiend or trainer.get_model()
        if isinstance(raw_model, str):
            trust_remote_code = getattr(getattr(trainer, "_training_args", None), "trust_remote_code", False)
            raw_model = ModelWithGradiend.from_pretrained(raw_model, trust_remote_code=trust_remote_code)

        target_classes = trainer.get_target_feature_classes()
        if target_class is not None:
            classes_to_eval: List[str] = (
                [target_class] if isinstance(target_class, str) else list(target_class)
            )
            for c in classes_to_eval:
                if c not in (target_classes or []):
                    raise ValueError(
                        f"target_class={target_class!r} must be one of trainer target classes {target_classes}. "
                        f"Got {c!r}."
                    )
        else:
            classes_to_eval = target_classes or []

        # Strengthen summaries must only request classes actually evaluated.
        # ``get_target_feature_classes()`` expands one-pole to claim+CFs for token
        # vocab; those CF names are datasets / weaken rivals, not strengthen metrics.
        # When ``target_class`` (or an explicit ``summary_metrics``) is set, never
        # fall through to the expanded CF list.
        claim_eval = [str(c) for c in classes_to_eval]
        claim_set = set(claim_eval)
        if summary_metrics is not None:
            metrics_for_summary = [
                str(m)
                for m in summary_metrics
                if (str(m).endswith("_weaken") and str(m)[:-7] in claim_set)
                or (not str(m).endswith("_weaken") and str(m) in claim_set)
            ]
            if not metrics_for_summary and claim_eval:
                metrics_for_summary = (
                    list(claim_eval)
                    if increase_target_probabilities
                    else [f"{c}_weaken" for c in claim_eval]
                )
        elif claim_eval:
            if increase_target_probabilities:
                metrics_for_summary = list(claim_eval)
            else:
                metrics_for_summary = [f"{c}_weaken" for c in claim_eval]
        else:
            metrics_for_summary = []
        plot_keys_override = list(classes_to_eval) if target_class is not None else None

        prob_on_other_class = True
        if trainer_config is not None:
            prob_on_other_class = bool(
                getattr(trainer_config, "decoder_eval_prob_on_other_class", True)
            )
        if training_args is not None:
            prob_on_other_class = bool(
                getattr(
                    training_args,
                    "decoder_eval_prob_on_other_class",
                    prob_on_other_class,
                )
            )
        selection_context = _decoder_cache_selection_context(
            prob_on_other_class=prob_on_other_class,
            increase_target_probabilities=increase_target_probabilities,
            metrics_for_summary=metrics_for_summary,
            classes_to_eval=classes_to_eval,
        )

        # Per-class strengthen ff (see gradiend.model._source_target module docstring).
        class_to_ff: Optional[Dict[str, float]] = None
        if target_classes:
            class_to_ff = {
                cls: derive_feature_factor_for_class(trainer, raw_model, cls)
                for cls in target_classes
            }

        # Captured before feature_factors is possibly overwritten by the
        # derivation below -- distinguishes "caller explicitly chose these
        # feature_factors" (e.g. an opposite-polarity random-control
        # re-query, deliberately requesting a class's non-canonical
        # direction) from "this evaluator derived its own default sweep from
        # class_to_ff". Forwarded into compute_metric_summaries so it can
        # tell those two cases apart instead of always trusting class_to_ff.
        explicit_feature_factors = feature_factors is not None

        if feature_factors is None:
            if increase_target_probabilities and classes_to_eval and class_to_ff:
                feature_factors = list(dict.fromkeys(class_to_ff[c] for c in classes_to_eval))
            elif not increase_target_probabilities and classes_to_eval and class_to_ff:
                feature_factors = []
                for c in classes_to_eval:
                    for o in target_classes:
                        if o == c:
                            continue
                        ff = class_to_ff[o]
                        if ff not in feature_factors:
                            feature_factors.append(ff)
                if not feature_factors:
                    feature_factors = default_decoder_feature_factors(trainer, raw_model)
            else:
                feature_factors = default_decoder_feature_factors(trainer, raw_model)

        model_with_gradiend = trainer._model_for_decoder_eval(raw_model)
        path = model_with_gradiend.name_or_path
        base_model = model_with_gradiend.base_model
        tokenizer = model_with_gradiend.tokenizer
        run_id = getattr(trainer, "run_id", None)
        model_id = os.path.basename(path) if path and str(path).startswith("results/models") else path

        if lrs is None:
            lrs = [
                m * 10 ** e
                for e in range(2, -6, -1)
                for m in [5, 2, 1]
                if m * 10 ** e <= 100
            ]

        experiment_dir = trainer.experiment_dir
        cache_file = resolve_decoder_grid_cache_path(experiment_dir, explicit_path=output_path)
        if raw_output_path is None and getattr(trainer_config, "decoder_eval_export_row_wise_csv", False):
            raw_output_path = _default_decoder_raw_output_path(cache_file)
        if use_cache and not cache_file:
            raise ValueError(
                "evaluate_decoder(use_cache=True) requires experiment_dir on the trainer or output_path. "
                "Set experiment_dir on TrainingArguments or pass output_path= to specify the cache location."
            )
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        split_cache_key = _decoder_split_cache_key(split)

        pairs = [(ff, lr) for ff in feature_factors for lr in lrs]
        pairs = sorted(pairs)  # deterministic order for reproducible decoder evaluation
        expected_results = len(pairs) + 1
        logger.debug(
            "Decoder eval feature_factors=%s class_to_ff=%s source=%s",
            list(feature_factors),
            class_to_ff,
            getattr(raw_model, "source", None),
        )

        relevant_results: Dict[Any, Dict[str, Any]] = {}
        all_results: Dict[Any, Dict[str, Any]] = {}
        raw_result_frames: List[pd.DataFrame] = []

        if use_cache and cache_file and os.path.isfile(cache_file):
            try:
                with open(cache_file, "r") as f:
                    payload = json.load(f)
                cached_part = payload.get("part")
                cached_feature_factors = payload.get("feature_factors")
                cached_lrs = payload.get("lrs")
                cached_split = payload.get("split", "test")
                cached_max_size_training_like = payload.get("max_size_training_like")
                cached_max_size_neutral = payload.get("max_size_neutral")
                cached_intervention_kwargs = payload.get("intervention_kwargs")
                cache_matches = (
                    cached_part == part
                    and cached_feature_factors == list(feature_factors)
                    and cached_lrs == list(lrs)
                    and cached_split == split_cache_key
                    and cached_max_size_training_like == max_size_training_like
                    and cached_max_size_neutral == max_size_neutral
                    and cached_intervention_kwargs == intervention_cache_fingerprint
                )
                structural_cache_matches = cache_matches
                if cache_matches and raw_output_path and not os.path.isfile(raw_output_path):
                    logger.info(
                        "Decoder grid cache matches at %s, but raw per-sample CSV is missing at %s; recomputing.",
                        cache_file,
                        raw_output_path,
                    )
                    cache_matches = False
                if cache_matches and not _decoder_cache_selection_matches(payload, selection_context):
                    logger.info(
                        "Decoder cache selection context mismatch at %s; recomputing.",
                        cache_file,
                    )
                    cache_matches = False
                if cache_matches:
                    relevant_results = convert_results_to_dict(payload.get("results", []))
                    if len(relevant_results) < expected_results:
                        logger.info(
                            "Decoder cache at %s has %d/%d grid entries; recomputing.",
                            cache_file,
                            len(relevant_results),
                            expected_results,
                        )
                        cache_matches = False
                    else:
                        metrics_ok, missing_metrics = _decoder_results_support_metrics(
                            relevant_results,
                            metrics_for_summary,
                            extractor=summary_extractor,
                        )
                        if not metrics_ok:
                            logger.info(
                                "Decoder cache at %s lacks required metrics %s; recomputing.",
                                cache_file,
                                missing_metrics,
                            )
                            cache_matches = False
                if cache_matches:
                    logger.info("Using cached decoder grid from %s", cache_file)
                    summary = self.compute_metric_summaries(
                        trainer,
                        relevant_results,
                        selector=selector,
                        metrics=metrics_for_summary,
                        extractor=summary_extractor,
                        feature_factor_from_id=summary_feature_factor_from_id,
                        lr_from_id=summary_lr_from_id,
                        empty_default_id=summary_empty_default_id,
                        class_to_ff=class_to_ff,
                        explicit_feature_factors=explicit_feature_factors,
                    )
                    if not plot:
                        out_cached = {**summary, "grid": relevant_results}
                        if raw_output_path:
                            out_cached["raw_output_path"] = raw_output_path
                        return out_cached
                    # plot=True: get full df and run fill-in + plot. Reuse the
                    # caller's own training_like_df/neutral_df (still their
                    # original values here -- untouched since function entry)
                    # instead of unconditionally re-deriving from the
                    # trainer's internal state, which silently substitutes a
                    # different (often narrower) population. Same bug and
                    # fix as the other _get_decoder_eval_dataframe call site
                    # below -- see CLAUDE.md in the study repo.
                    if training_like_df is None or neutral_df is None:
                        training_like_df, neutral_df = trainer._get_decoder_eval_dataframe(
                            tokenizer,
                            max_size_training_like=max_size_training_like,
                            max_size_neutral=max_size_neutral,
                            split=split,
                            cached_training_like_df=training_like_df,
                            cached_neutral_df=neutral_df,
                        )
                    full_training_like_df = training_like_df
                    dataset_class_col = "label_class" if "label_class" in getattr(training_like_df, "columns", []) else "factual_id"
                    if full_training_like_df is not None and dataset_class_col in getattr(full_training_like_df, "columns", []):
                        _refresh_probs_by_dataset_for_plotting(
                            trainer,
                            model_with_gradiend=model_with_gradiend,
                            base_model=base_model,
                            tokenizer=tokenizer,
                            relevant_results=relevant_results,
                            training_like_df=full_training_like_df,
                            neutral_df=neutral_df,
                            part=part,
                            intervention_kwargs=resolved_intervention_kwargs,
                            max_size_training_like=max_size_training_like,
                            max_size_neutral=max_size_neutral,
                            eval_batch_size=eval_batch_size,
                        )
                    if cache_file:
                        try:
                            payload_update = _decoder_cache_payload(
                                part=part,
                                split_cache_key=split_cache_key,
                                max_size_training_like=max_size_training_like,
                                max_size_neutral=max_size_neutral,
                                feature_factors=feature_factors,
                                lrs=lrs,
                                intervention_cache_fingerprint=intervention_cache_fingerprint,
                                results=convert_results_to_list(relevant_results),
                                selection_context=selection_context,
                            )
                            with open(cache_file, "w") as f:
                                json.dump(payload_update, f, indent=2)
                        except Exception as e:
                            logger.warning("Error writing decoder cache %s: %s", cache_file, e)
                    plot_paths: List[str] = []
                    if hasattr(trainer, "plot_probability_shifts"):
                        try:
                            plot_paths = _plot_all_target_classes(
                                trainer,
                                summary,
                                relevant_results,
                                experiment_dir=getattr(trainer, "experiment_dir", None),
                                run_id=run_id,
                                show=show,
                                increase_target_probabilities=increase_target_probabilities,
                                plot_keys_override=plot_keys_override,
                                plot_kwargs=plot_kwargs,
                                intervention_kwargs=resolved_intervention_kwargs,
                                training_like_df=full_training_like_df,
                                neutral_df=neutral_df,
                                split=split,
                            )
                        except ImportError as e:
                            logger.warning("Skipping decoder probability-shift plots: %s", e)
                    out = {
                        **summary,
                        "grid": relevant_results,
                        "intervention_kwargs": resolved_intervention_kwargs,
                    }
                    if raw_output_path:
                        out["raw_output_path"] = raw_output_path
                    if plot_paths:
                        out["plot_paths"] = plot_paths
                        out["plot_path"] = plot_paths[0] if len(plot_paths) == 1 else None
                    return out
                elif not structural_cache_matches:
                    logger.info("Decoder cache mismatch (part/split/size/feature_factors/lrs); recomputing.")
            except Exception as e:
                logger.warning("Error loading cached decoder results: %s", e)

        if max_size_training_like is None:
            if training_args is not None:
                max_size_training_like = getattr(training_args, "decoder_eval_max_size_training_like", None)
            if max_size_training_like is None and trainer_config is not None:
                max_size_training_like = getattr(trainer_config, "decoder_eval_lms_max_samples", None)
        if max_size_neutral is None:
            if training_args is not None:
                max_size_neutral = getattr(training_args, "decoder_eval_max_size_neutral", None)
            if max_size_neutral is None and trainer_config is not None:
                max_size_neutral = getattr(trainer_config, "decoder_eval_lms_max_samples", None)

        if training_like_df is None or neutral_df is None:
            training_like_df, neutral_df = trainer._get_decoder_eval_dataframe(
                tokenizer,
                max_size_training_like=max_size_training_like,
                max_size_neutral=max_size_neutral,
                split=split,
                cached_training_like_df=training_like_df,
                cached_neutral_df=neutral_df,
            )

        # `full_training_like_df`/`dataset_class_col` are read later for
        # plotting (both the every-class-shown plot branches below) even
        # though `training_like_df` itself is used unfiltered for scoring --
        # see the removed dataset-narrowing note below.
        full_training_like_df = training_like_df
        dataset_class_col = "label_class" if "label_class" in getattr(training_like_df, "columns", []) else "factual_id"
        # NOTE: this used to additionally pre-filter `training_like_df` down
        # to a `required_datasets` subset computed from
        # (target_class/classes_to_eval, tcs, increase_target_probabilities,
        # prob_on_other_class) as a scoring-efficiency optimization ("only
        # score the dataset rows this specific direction needs"). Removed:
        # every real call site's `classes_to_eval` is either (a) a trainer's
        # full claim set (`evaluate_decoder_for_classes` in the study, or a
        # bare `target_class=` covering every pole a pair trainer has) --
        # which already made the old formula recover full coverage, so
        # narrowing was a no-op there -- or (b) a singleton (every direct
        # `target_class="X"` call in this package's own examples, and every
        # opposite-polarity random-control re-query in causal_study.py) --
        # where the old formula unconditionally dropped that one class's own
        # dataset panel. That silent drop is exactly what caused
        # "probs_by_dataset['M']['F'] is absent"-style crashes in production
        # (confirmed on gender_en/pronoun_number/religion_one_pole): the
        # dropped class's own panel is needed by this same call's same-panel
        # weaken-selection (probs_by_dataset[class][class] in trainer.py) and
        # by cross-panel callers this purely-local, single-call vantage point
        # cannot predict. Since no current caller relies on the narrowing for
        # actual performance (it was a no-op whenever safe, and unsafe
        # whenever it did anything), keeping a "fixed" version of it around
        # only preserves complexity with no benefit -- removed outright
        # rather than patched. If a genuine large-multi-class-frame
        # performance need arises later, it should be reintroduced as an
        # explicit, opt-in parameter with its own contract, not a silent
        # default a caller has no way to see or override.

        _LARGE_DATASET = 10000
        if max_size_training_like is None and len(training_like_df) > _LARGE_DATASET:
            logger.warning(
                "decoder eval: max_size_training_like is not set and training data has %d rows. "
                "Computation may be slow. Consider setting decoder_eval_max_size_training_like or max_size_training_like to cap.",
                len(training_like_df),
            )
        if max_size_neutral is None and len(neutral_df) > _LARGE_DATASET:
            logger.warning(
                "decoder eval: max_size_neutral is not set and neutral data has %d rows. "
                "LMS computation may be slow. Set decoder_eval_max_size_neutral or max_size_neutral to cap.",
                len(neutral_df),
            )

        if "base" not in relevant_results or not use_cache:
            logger.debug("Evaluating base model...")
            base_eval_kwargs = {
                "use_cache": use_cache,
                "cache_folder": f"base_split_{split_cache_key}_max_{max_size_training_like}_neutral_{max_size_neutral}",
                "training_like_df": training_like_df,
                "neutral_df": neutral_df,
                "max_size_training_like": max_size_training_like,
                "max_size_neutral": max_size_neutral,
                "eval_batch_size": eval_batch_size,
            }
            if raw_output_path is not None:
                base_eval_kwargs["return_decoder_per_row_df"] = True
            base_results = trainer.evaluate_base_model(base_model, tokenizer, **base_eval_kwargs)
            _append_decoder_raw_rows(
                raw_result_frames,
                base_results,
                grid_id="base",
                feature_factor=None,
                learning_rate=None,
            )
            if isinstance(base_results, dict):
                base_results["id"] = "base"
            all_results["base"] = base_results
            relevant_results["base"] = base_results

        def _evaluate_pair(feature_factor: float, lr: float) -> Dict[str, Any]:
            id_key = (feature_factor, lr)
            logger.info(
                "_evaluate_pair: ff=%r lr=%r id(training_like_df)=%s len(training_like_df)=%s "
                "id(neutral_df)=%s len(neutral_df)=%s",
                feature_factor,
                lr,
                id(training_like_df),
                len(training_like_df) if training_like_df is not None else None,
                id(neutral_df),
                len(neutral_df) if neutral_df is not None else None,
            )
            with _decoder_grid_model_context(
                model_with_gradiend,
                base_model=base_model,
                learning_rate=lr,
                feature_factor=feature_factor,
                part=part,
                intervention_kwargs=resolved_intervention_kwargs,
            ) as modified_model:
                modified_eval_kwargs = {
                    "use_cache": use_cache,
                    "cache_folder": f"split_{split_cache_key}_max_{max_size_training_like}_neutral_{max_size_neutral}_{feature_factor}_{lr}",
                    "model_id": model_id,
                    "training_like_df": training_like_df,
                    "neutral_df": neutral_df,
                    "max_size_training_like": max_size_training_like,
                    "max_size_neutral": max_size_neutral,
                    "eval_batch_size": eval_batch_size,
                }
                if raw_output_path is not None:
                    modified_eval_kwargs["return_decoder_per_row_df"] = True
                modified_results = trainer.evaluate_base_model(modified_model, tokenizer, **modified_eval_kwargs)
            _append_decoder_raw_rows(
                raw_result_frames,
                modified_results,
                grid_id=f"ff={feature_factor}|lr={lr}",
                feature_factor=float(feature_factor),
                learning_rate=float(lr),
            )
            if isinstance(modified_results, dict):
                modified_results["id"] = {"feature_factor": feature_factor, "learning_rate": lr}
            all_results[id_key] = modified_results
            relevant_results[id_key] = modified_results
            return modified_results

        for feature_factor, lr in gradiend_tqdm(
            pairs,
            desc=f"Evaluate GRADIEND {run_id or ''}",
            total=len(pairs),
            position=0,
        ):
            id_key = (feature_factor, lr)
            if id_key in relevant_results and use_cache:
                continue
            _evaluate_pair(feature_factor, lr)

        if refine_points > 0 and increase_target_probabilities and classes_to_eval and class_to_ff:
            logger.info(
                "evaluate_decoder: entering _bisect_refine_lms_boundary refine_points=%r "
                "classes_to_eval=%r class_to_ff=%r id(training_like_df)=%s len(training_like_df)=%s "
                "id(neutral_df)=%s len(neutral_df)=%s",
                refine_points,
                list(classes_to_eval),
                dict(class_to_ff),
                id(training_like_df),
                len(training_like_df) if training_like_df is not None else None,
                id(neutral_df),
                len(neutral_df) if neutral_df is not None else None,
            )
            _bisect_refine_lms_boundary(
                relevant_results=relevant_results,
                pairs=pairs,
                lrs=lrs,
                classes_to_eval=classes_to_eval,
                class_to_ff=class_to_ff,
                selector=selector,
                refine_points=refine_points,
                evaluate_pair=_evaluate_pair,
            )

        summary = self.compute_metric_summaries(
            trainer,
            relevant_results,
            selector=selector,
            metrics=metrics_for_summary,
            extractor=summary_extractor,
            feature_factor_from_id=summary_feature_factor_from_id,
            lr_from_id=summary_lr_from_id,
            empty_default_id=summary_empty_default_id,
            class_to_ff=class_to_ff,
            explicit_feature_factors=explicit_feature_factors,
        )

        plot_paths: List[str] = []
        if plot and full_training_like_df is not None and dataset_class_col in getattr(full_training_like_df, "columns", []):
            _refresh_probs_by_dataset_for_plotting(
                trainer,
                model_with_gradiend=model_with_gradiend,
                base_model=base_model,
                tokenizer=tokenizer,
                relevant_results=relevant_results,
                training_like_df=full_training_like_df,
                neutral_df=neutral_df,
                part=part,
                intervention_kwargs=resolved_intervention_kwargs,
                max_size_training_like=max_size_training_like,
                max_size_neutral=max_size_neutral,
                eval_batch_size=eval_batch_size,
            )
            if cache_file:
                try:
                    payload = _decoder_cache_payload(
                        part=part,
                        split_cache_key=split_cache_key,
                        max_size_training_like=max_size_training_like,
                        max_size_neutral=max_size_neutral,
                        feature_factors=feature_factors,
                        lrs=lrs,
                        intervention_cache_fingerprint=intervention_cache_fingerprint,
                        results=convert_results_to_list(relevant_results),
                        selection_context=selection_context,
                    )
                    with open(cache_file, "w") as f:
                        json.dump(payload, f, indent=2)
                except Exception as e:
                    logger.warning("Error writing decoder cache %s: %s", cache_file, e)
            if hasattr(trainer, "plot_probability_shifts"):
                try:
                    plot_paths = _plot_all_target_classes(
                        trainer,
                        summary,
                        relevant_results,
                        experiment_dir=getattr(trainer, "experiment_dir", None),
                        run_id=run_id,
                        show=show,
                        increase_target_probabilities=increase_target_probabilities,
                        plot_keys_override=plot_keys_override,
                        plot_kwargs=plot_kwargs,
                        intervention_kwargs=resolved_intervention_kwargs,
                        training_like_df=full_training_like_df,
                        neutral_df=neutral_df,
                        split=split,
                    )
                except ImportError as e:
                    logger.warning("Skipping decoder probability-shift plots: %s", e)

        if cache_file and not plot:
            try:
                payload = _decoder_cache_payload(
                    part=part,
                    split_cache_key=split_cache_key,
                    max_size_training_like=max_size_training_like,
                    max_size_neutral=max_size_neutral,
                    feature_factors=feature_factors,
                    lrs=lrs,
                    intervention_cache_fingerprint=intervention_cache_fingerprint,
                    results=convert_results_to_list(relevant_results),
                    selection_context=selection_context,
                )
                with open(cache_file, "w") as f:
                    json.dump(payload, f, indent=2)
            except Exception as e:
                logger.warning("Error writing decoder cache %s: %s", cache_file, e)

        written_raw_output_path = _write_decoder_raw_rows(raw_output_path, raw_result_frames)

        out = {
            **summary,
            "grid": relevant_results,
            "intervention_kwargs": resolved_intervention_kwargs,
        }
        if written_raw_output_path:
            out["raw_output_path"] = written_raw_output_path
        if plot and plot_paths:
            out["plot_paths"] = plot_paths
            out["plot_path"] = plot_paths[0] if len(plot_paths) == 1 else None
        return out

    def compute_metric_summaries(
            self,
            trainer: Any,
            results: Mapping[Any, Mapping[str, Any]],
            *,
            selector: SelectionPolicy,
            metrics: Optional[Sequence[str]] = None,
            extractor: Callable[
                [Mapping[Any, Mapping[str, Any]]], Tuple[List[Candidate], BaseContext]] = default_extract_candidates,
            feature_factor_from_id: Callable[[CandidateId], float] = lambda cid: cid[0],
            lr_from_id: Callable[[CandidateId], float] = lambda cid: cid[1],
            empty_default_id: str = "base",
            class_to_ff: Optional[Mapping[str, float]] = None,
            explicit_feature_factors: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build a per-metric (i.e., target classes) summary for a decoder grid evaluation.

        This is a thin wrapper around the module-level `compute_metric_summaries` that
        resolves default metrics from the trainer and passes through selector and
        extraction behavior.

        Args:
            trainer: Trainer-like object used to resolve default metrics when
                `metrics` is None (via `get_target_feature_classes`).
            results: Mapping from candidate id to evaluation result entries. The
                expected structure is the same as produced by decoder evaluation.
            selector: Policy used to choose the best candidate per metric (e.g., LMS thresholding (LMSThresholdPolicy),
                or metric*lms (LMSTimesMetricPolicy).
            metrics: Optional list of metric names to summarize. If None, uses the
                trainer's target feature classes.
            extractor: Function that converts the raw `results` mapping into
                candidates and a base context.
            feature_factor_from_id: Function to extract feature_factor from a
                candidate id.
            lr_from_id: Function to extract learning_rate from a candidate id.
            empty_default_id: Fallback id used when no candidate is selected (for comparison with base).
                When the selector returns None, we first try the candidate with learning_rate != 0 and smallest
                absolute value; only if none exists do we use this default (representing the base model).
            class_to_ff: Optional mapping from class id to the feature factor
                that strengthens it. Used for strengthen and ``*_weaken`` metrics
                so each summary only considers candidates with the correct
                rewrite orientation for that class.
            explicit_feature_factors: True when the grid's feature_factors were
                explicitly requested by the caller of ``evaluate_decoder``
                (not derived by this evaluator from ``class_to_ff``). See the
                module-level ``compute_metric_summaries``'s docstring.

        Returns:
            A dict keyed by metric name with values containing selected metric
            `value`, `feature_factor`, `learning_rate`, and `id`.
        """
        metrics = list(metrics) if metrics is not None else trainer.get_target_feature_classes()
        return compute_metric_summaries(
            results,
            metrics=metrics,
            selector=selector,
            extractor=extractor,
            feature_factor_from_id=feature_factor_from_id,
            lr_from_id=lr_from_id,
            empty_default_id=empty_default_id,
            class_to_ff=class_to_ff,
            explicit_feature_factors=explicit_feature_factors,
        )
