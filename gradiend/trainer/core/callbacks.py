"""
Training callbacks for GRADIEND models (HF Trainer–style).

Callbacks handle evaluation, checkpointing, normalization, logging,
early stopping, and optional TensorBoard/Wandb logging.

Provided callbacks:

- EvaluationCallback: periodic evaluation (correlation, mean_by_class).
- NormalizationCallback: invert encoding when correlation < -0.5 (single latent).
- CheckpointCallback: save best model and optional periodic checkpoints.
- LoggingCallback: log mean loss over the report window, correlation, class means.
- EarlyStoppingCallback: stop when metric has not improved for N steps.
- TensorBoardCallback: log metrics to TensorBoard (optional).
"""

from typing import Optional, Callable, Dict, Any, Union, List
from abc import ABC

from gradiend.util.logging import get_logger
from gradiend.trainer.core.metric_names import normalize_metric_name
from gradiend.util.component_logging import format_component_convergence_fragment
from gradiend.trainer.core.component_seed import (
    merge_rank_from_summary,
    model_has_component_split,
    payload_from_component_best_states,
    save_component_best_states,
    save_merged_component_best_checkpoint,
    summary_from_component_best_states,
    update_component_best_states,
)
from gradiend.trainer.core.stats import (
    _mean_by_class_for_step,
    _positive_target_class_mean,
    _target_class_mean_stats,
    metric_checkpoint_rank,
)

logger = get_logger(__name__)


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _eval_enabled(config: Any, *, loss_only: bool = False) -> bool:
    if loss_only:
        return False
    return bool(_config_get(config, "do_eval", True))


def _selection_metric_from_config(config: Any, *, use_loss_for_best: bool = False) -> str:
    if use_loss_for_best:
        return "loss"
    explicit = _config_get(config, "selection_metric", None)
    if explicit is None:
        explicit = _config_get(config, "convergent_metric", "correlation")
    return normalize_metric_name(explicit)


def _current_step_correlation(
    *,
    step: int,
    training_stats: Dict[str, Any],
    eval_result: Optional[Dict[str, Any]],
) -> Optional[float]:
    if isinstance(eval_result, dict):
        component_summary = eval_result.get("component_summary")
        if isinstance(component_summary, dict) and component_summary.get("correlation_mean") is not None:
            return float(component_summary["correlation_mean"])
    if isinstance(eval_result, dict) and eval_result.get("correlation") is not None:
        return float(eval_result["correlation"])
    scores = training_stats.get("scores")
    if isinstance(scores, dict) and step in scores and scores[step] is not None:
        return float(scores[step])
    return None


def _hist_float_at_step(hist: Any, step: int) -> Optional[float]:
    if isinstance(hist, dict) and step in hist and hist[step] is not None:
        return float(hist[step])
    if isinstance(hist, (int, float)):
        return float(hist)
    return None


def _current_step_selection_score(
    *,
    step: int,
    metric: str,
    training_stats: Dict[str, Any],
    eval_result: Optional[Dict[str, Any]],
) -> Optional[float]:
    name = normalize_metric_name(metric)
    if name == "roc_auc":
        if isinstance(eval_result, dict) and eval_result.get("roc_auc") is not None:
            return float(eval_result["roc_auc"])
        return _hist_float_at_step(training_stats.get("roc_auc"), step)
    if name == "min_auc_n_o":
        if isinstance(eval_result, dict) and eval_result.get("min_auc_n_o") is not None:
            return float(eval_result["min_auc_n_o"])
        return _hist_float_at_step(training_stats.get("min_auc_n_o"), step)
    if name == "encoding_e":
        if isinstance(eval_result, dict) and eval_result.get("encoding_e") is not None:
            return float(eval_result["encoding_e"])
        return _hist_float_at_step(training_stats.get("encoding_e"), step)
    if name == "loss":
        return None
    return _current_step_correlation(
        step=step,
        training_stats=training_stats,
        eval_result=eval_result,
    )


def _target_mean_converged(mean_by_class: Any, mean_threshold: Any) -> tuple[bool, Optional[float], Optional[float]]:
    return _target_class_mean_stats(mean_by_class, mean_threshold)


def _attach_component_convergence(eval_result: Dict[str, Any], config: Any) -> None:
    components = eval_result.get("components")
    if not isinstance(components, dict):
        return
    metrics_by_component = components.get("metrics_by_component")
    summary = dict(components.get("summary") or {})
    if not isinstance(metrics_by_component, dict) or not metrics_by_component:
        return
    threshold = _config_get(config, "convergent_score_threshold", None)
    mean_threshold = _config_get(config, "convergent_mean_by_class_threshold", None)
    n_converged = 0
    convergence_by_component: Dict[str, Any] = {}
    correlations = []
    min_abs_means = []
    for component_id, metrics in metrics_by_component.items():
        corr = metrics.get("correlation") if isinstance(metrics, dict) else None
        if isinstance(corr, (int, float)):
            correlations.append(float(corr))
        mean_by_class = metrics.get("mean_by_class") if isinstance(metrics, dict) else None
        means_ok, product, min_abs = _target_mean_converged(mean_by_class, mean_threshold)
        if isinstance(min_abs, (int, float)):
            min_abs_means.append(float(min_abs))
        score_ok = threshold is None or (isinstance(corr, (int, float)) and abs(float(corr)) >= float(threshold))
        converged = bool(score_ok and means_ok)
        if converged:
            n_converged += 1
        convergence_by_component[str(component_id)] = {
            "converged": converged,
            "correlation": float(corr) if isinstance(corr, (int, float)) else None,
            "target_mean_product": product,
            "min_target_class_abs_mean": min_abs,
        }
    n_components = int(summary.get("n_components") or len(metrics_by_component))
    summary.update({
        "n_components": n_components,
        "n_converged": n_converged,
        "convergence_rate": float(n_converged / n_components) if n_components else 0.0,
        "convergence_policy": "all",
        "converged": n_components > 0 and n_converged == n_components,
    })
    if correlations:
        sorted_corr = sorted(correlations)
        mid = len(sorted_corr) // 2
        median = (
            sorted_corr[mid]
            if len(sorted_corr) % 2
            else (sorted_corr[mid - 1] + sorted_corr[mid]) / 2.0
        )
        summary.update({
            "correlation_mean": sum(correlations) / len(correlations),
            "correlation_median": median,
            "correlation_min": min(correlations),
            "correlation_max": max(correlations),
            "correlation_abs_mean": sum(abs(v) for v in correlations) / len(correlations),
            "correlation_abs_min": min(abs(v) for v in correlations),
            "correlation_abs_max": max(abs(v) for v in correlations),
        })
    if min_abs_means:
        summary["min_target_class_abs_mean"] = min(min_abs_means)
    components["summary"] = summary
    components["convergence_by_component"] = convergence_by_component
    eval_result["components"] = components
    eval_result["component_summary"] = summary


def _format_loss_for_log(loss: Optional[float]) -> str:
    if loss is None:
        return "None"
    value = float(loss)
    if value != 0.0 and abs(value) < 1e-4:
        return f"{value:.3e}"
    return f"{value:.4f}"



def _is_unrecoverable_eval_error(exc: BaseException) -> bool:
    """Whether an evaluation error means the run can no longer produce results.

    A metric failing on a degenerate batch is worth logging and skipping. Running
    out of device memory is not: allocation failures leave the process unable to
    evaluate anything afterwards, so continuing yields a training run that
    reports success while selecting no checkpoint at all.

    Matched by type where the running torch exposes one, with a message fallback
    because the exception type has moved between torch versions
    (``torch.cuda.OutOfMemoryError`` / ``torch.OutOfMemoryError``).
    """
    import torch

    oom_types = tuple(
        t
        for t in (
            getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None),
            getattr(torch, "OutOfMemoryError", None),
        )
        if isinstance(t, type)
    )
    if oom_types and isinstance(exc, oom_types):
        return True
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda error" in text


class TrainingCallback(ABC):
    """
    Base class for training callbacks (HF Trainer–style lifecycle).

    Subclass and override the hooks you need. All lifecycle hooks are optional
    except on_step_end and on_epoch_end (default no-op for compatibility).
    """

    def on_train_begin(self, config: Dict[str, Any], **kwargs) -> None:
        """Called once at the start of training."""
        pass

    def on_train_end(self, config: Dict[str, Any], **kwargs) -> None:
        """Called once at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, config: Dict[str, Any], **kwargs) -> None:
        """Called at the start of each epoch."""
        pass

    def on_step_begin(self, step: int, config: Dict[str, Any], **kwargs) -> None:
        """Called at the start of each step (before forward)."""
        pass

    def on_step_end(self, step: int, loss: float, model, config: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
        """
        Called at the end of each step. Return value can be used by other callbacks
        (e.g. EvaluationCallback returns eval result for NormalizationCallback).
        """
        pass

    def on_epoch_end(self, epoch: int, model, config: Dict[str, Any], **kwargs) -> None:
        """Called at the end of each epoch."""
        pass


class EvaluationCallback(TrainingCallback):
    """Callback for periodic evaluation during training."""
    
    def __init__(self, evaluate_fn: Callable, n_evaluation: int = 250, do_eval: bool = True):
        """
        Args:
            evaluate_fn: Function to call for evaluation
            n_evaluation: Evaluate every N steps
            do_eval: Whether to perform evaluation
        """
        self.evaluate_fn = evaluate_fn
        self.n_evaluation = n_evaluation
        self.do_eval = do_eval
        self.last_eval_step = -1
    
    def on_step_end(self, step: int, loss: float, model, config: Dict[str, Any], training_stats: Dict[str, Any], **kwargs):
        """Perform evaluation if needed."""
        if not self.do_eval or self.evaluate_fn is None:
            return None
        
        # Check if we should evaluate at this step
        # Evaluate at step 0 (initial), every n_evaluation steps, and at last iteration
        should_eval = (
            (self.n_evaluation > 0 and step % self.n_evaluation == 0) or
            kwargs.get('last_iteration', False)
        )
        
        # Skip if we already evaluated successfully at this step
        if not should_eval or step == self.last_eval_step:
            return None
        
        logger.debug(f'Evaluating at step {step}...')
        
        # Set eval mode only on base_model (not the whole model) to avoid recursion
        # We need gradients enabled for GRADIEND evaluation, so don't use torch.no_grad()
        base_model_was_training = model.base_model.training
        gradiend_was_training = model.gradiend.training

        with model.exclusive_base_gradient_access():
            try:
                # Set eval mode on submodules (not the whole model to avoid recursion)
                model.base_model.eval()
                model.gradiend.eval()

                eval_result = self.evaluate_fn(config=config, training_stats=training_stats)

                if eval_result:
                    _attach_component_convergence(eval_result, config)
                    # Store label_value_to_class_name once (same for all steps); used for display
                    if "label_value_to_class_name" in eval_result:
                        training_stats["label_value_to_class_name"] = eval_result["label_value_to_class_name"]

                    # Update training_stats with evaluation results (tracked over time)
                    for key, value in eval_result.items():
                        if key == "label_value_to_class_name":
                            continue  # already stored above
                        if key not in training_stats:
                            training_stats[key] = {}

                        # Store results by step for tracking over time
                        if isinstance(training_stats[key], dict):
                            training_stats[key][step] = value
                        elif isinstance(training_stats[key], list):
                            training_stats[key].append(value)
                        elif isinstance(training_stats[key], (int, float)):
                            # Convert to dict to track over time
                            training_stats[key] = {step: value}

                    # Ensure correlation is set (and per-step history)
                    current_corr = _current_step_correlation(
                        step=step,
                        training_stats=training_stats,
                        eval_result=eval_result,
                    )
                    if current_corr is not None:
                        if "component_summary" in eval_result and eval_result.get("correlation") is not None:
                            training_stats.setdefault("aggregate_correlation", {})[step] = eval_result["correlation"]
                        val = current_corr
                        training_stats['correlation'] = val
                        if 'scores' not in training_stats:
                            training_stats['scores'] = {}
                        training_stats['scores'][step] = val
                        self.last_eval_step = step

                    # Per-feature-class means at DEBUG, 4 decimals
                    if 'mean_by_feature_class' in eval_result:
                        mbfc = eval_result['mean_by_feature_class']
                        compact = ', '.join(f'{k}={v:.4f}' if isinstance(v, (int, float)) else f'{k}={v}' for k, v in sorted(mbfc.items()))
                        logger.debug(f'Mean by feature class: {compact}')

                    return eval_result
                else:
                    logger.warning(f'Evaluation at step {step} returned no results')
                    return None
            except Exception as e:
                if _is_unrecoverable_eval_error(e):
                    # Resource exhaustion leaves the process degraded: every
                    # later step evaluates nothing, no checkpoint is ever
                    # selected, and the run still exits 0 -- so the job reports
                    # COMPLETED having trained nothing. Fail loudly instead.
                    logger.error(
                        f"Unrecoverable error during evaluation at step {step}; "
                        f"aborting rather than continuing with no evaluation: {e}",
                        exc_info=True,
                    )
                    raise
                logger.error(f"Error during evaluation at step {step}: {e}", exc_info=True)
                return None
            finally:
                # Restore training mode
                if base_model_was_training:
                    model.base_model.train()
                if gradiend_was_training:
                    model.gradiend.train()
    
    def on_epoch_end(self, epoch: int, model, config: Dict[str, Any], **kwargs):
        """No action needed at epoch end."""
        pass


class NormalizationCallback(TrainingCallback):
    """Callback for GRADIEND normalization during training."""
    
    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: Whether to perform normalization
        """
        self.normalize = normalize
    
    def on_step_end(self, step: int, loss: float, model, config: Dict[str, Any], 
                   eval_result: Optional[Dict], training_stats: Dict[str, Any], **kwargs):
        """Perform normalization if needed."""
        if not self.normalize or eval_result is None:
            return
        
        if (
            model.gradiend.latent_dim == 1
            and bool(getattr(model.gradiend, "has_component_split", False))
            and isinstance(eval_result.get("components"), dict)
        ):
            metrics_by_component = eval_result["components"].get("metrics_by_component", {})
            normalized = []
            for component_id, metrics in metrics_by_component.items():
                if not isinstance(metrics, dict):
                    continue
                corr_raw = metrics.get("correlation")
                if corr_raw is None:
                    continue
                corr = float(corr_raw)
                if corr < -0.6:
                    index = metrics.get("component_index")
                    try:
                        model.gradiend._component_view(int(index)).invert_encoding()
                        metrics["correlation"] = -corr
                        for key in (
                            "mean_by_class",
                            "mean_by_feature_class",
                            "mean_by_type",
                            "neutral_mean_by_type",
                        ):
                            means = metrics.get(key)
                            if isinstance(means, dict):
                                metrics[key] = {
                                    k: (-float(v) if isinstance(v, (int, float)) else v)
                                    for k, v in means.items()
                                }
                        normalized.append(component_id)
                    except Exception as exc:
                        logger.warning("Could not normalize component %s: %s", component_id, exc)
            if normalized:
                method_name = getattr(model.gradiend, "method_name", "GRADIEND")
                n_components = len(normalized)
                noun = "component encoding" if n_components == 1 else "component encodings"
                logger.info(
                    "Inverted %s %s %s: %s",
                    n_components,
                    method_name,
                    noun,
                    ", ".join(map(str, normalized)),
                )
                _attach_component_convergence(eval_result, config)
                if isinstance(training_stats.get("components"), dict):
                    training_stats["components"][step] = eval_result.get("components")
                if isinstance(training_stats.get("component_summary"), dict):
                    training_stats["component_summary"][step] = eval_result.get("component_summary")
                corr = _current_step_correlation(step=step, training_stats=training_stats, eval_result=eval_result)
                if corr is not None:
                    training_stats["correlation"] = corr
                    training_stats.setdefault("scores", {})[step] = corr
            return

        if model.gradiend.latent_dim == 1 and 'mean_by_class' in eval_result:
            # Correlation is sign-symmetric and keeps its historical correlation-based
            # normalization. AUC is self-orienting, so explicitly orient the learned
            # encoding: the semantic positive target (numeric label +1) must have mean > 0.
            try:
                selection_metric = _selection_metric_from_config(config)
                is_auc_metric = selection_metric in {"roc_auc", "min_auc_n_o", "encoding_e"}
                positive_target_mean = _positive_target_class_mean(eval_result.get('mean_by_class'))
                corr_raw = eval_result.get('correlation')
                if corr_raw is None and not is_auc_metric:
                    return  # Empty eval data; no correlation to normalize
                corr = float(corr_raw) if isinstance(corr_raw, (int, float)) else None
                should_invert = (
                    isinstance(positive_target_mean, (int, float)) and positive_target_mean < 0.0
                ) if is_auc_metric else (corr is not None and corr < -0.6)
                if should_invert:
                    if is_auc_metric:
                        logger.info(
                            "Inverting AUC-scored encoding since label +1 mean is %.4f < 0",
                            positive_target_mean,
                        )
                    else:
                        logger.info(f'Inverting encoding since correlation is {corr} < -0.5')
                    # CONTRACT: update_direction=False — flip encoder/decoder weights only.
                    # feature_class_encoding_direction is semantic (+1/-1 per class) and must
                    # NOT be negated here. Decoder eval derives ff from that metadata by source.
                    model.invert_encoding(update_direction=False)

                    # After inversion, flip the sign of the recorded correlation so that
                    # logs, checkpoints, and subsequent callbacks see the normalized
                    # (positive) orientation for this step.
                    new_corr = -corr if corr is not None else None
                    if new_corr is not None:
                        eval_result['correlation'] = new_corr
                        training_stats['correlation'] = new_corr

                    # Update per-step history if present
                    scores_hist = training_stats.get('scores')
                    if isinstance(scores_hist, dict) and new_corr is not None:
                        scores_hist[step] = new_corr

                    # Flip mean encodings per class and per feature class to match the new orientation
                    mean_by_class = eval_result.get('mean_by_class')
                    if isinstance(mean_by_class, dict):
                        flipped_means = {
                            k: (-float(v) if isinstance(v, (int, float)) else v)
                            for k, v in mean_by_class.items()
                        }
                        eval_result['mean_by_class'] = flipped_means

                        # If we track mean_by_class over steps, update the current step
                        means_hist = training_stats.get('mean_by_class')
                        if isinstance(means_hist, dict):
                            means_hist[step] = flipped_means

                    for min_key, max_key in (
                        ("min_by_class", "max_by_class"),
                        ("min_by_feature_class", "max_by_feature_class"),
                        ("q1_by_class", "q3_by_class"),
                        ("q1_by_feature_class", "q3_by_feature_class"),
                    ):
                        mins = eval_result.get(min_key)
                        maxs = eval_result.get(max_key)
                        if isinstance(mins, dict) and isinstance(maxs, dict):
                            flipped_mins = {
                                k: (-float(maxs[k]) if isinstance(maxs.get(k), (int, float)) else v)
                                for k, v in mins.items()
                            }
                            flipped_maxs = {
                                k: (-float(mins[k]) if isinstance(mins.get(k), (int, float)) else v)
                                for k, v in maxs.items()
                            }
                            eval_result[min_key] = flipped_mins
                            eval_result[max_key] = flipped_maxs
                            min_hist = training_stats.get(min_key)
                            max_hist = training_stats.get(max_key)
                            if isinstance(min_hist, dict):
                                min_hist[step] = flipped_mins
                            if isinstance(max_hist, dict):
                                max_hist[step] = flipped_maxs

                    mean_by_feature_class = eval_result.get('mean_by_feature_class')
                    if isinstance(mean_by_feature_class, dict):
                        flipped_mbfc = {
                            k: (-float(v) if isinstance(v, (int, float)) else v)
                            for k, v in mean_by_feature_class.items()
                        }
                        eval_result['mean_by_feature_class'] = flipped_mbfc

                        mbfc_hist = training_stats.get('mean_by_feature_class')
                        if isinstance(mbfc_hist, dict):
                            mbfc_hist[step] = flipped_mbfc

                    for mean_key in ("mean_by_type", "neutral_mean_by_type"):
                        mean_by_type = eval_result.get(mean_key)
                        if isinstance(mean_by_type, dict):
                            flipped_mean_by_type = {
                                k: (-float(v) if isinstance(v, (int, float)) else v)
                                for k, v in mean_by_type.items()
                            }
                            eval_result[mean_key] = flipped_mean_by_type
                            mean_type_hist = training_stats.get(mean_key)
                            if isinstance(mean_type_hist, dict):
                                mean_type_hist[step] = flipped_mean_by_type
            except Exception as e:
                logger.warning(f"Error during normalization: {e}")
        elif model.gradiend.latent_dim > 1:
            method_name = getattr(model.gradiend, "method_name", "GRADIEND")
            logger.warning("Normalization is only implemented for single-feature %s models", method_name)
    
    def on_epoch_end(self, epoch: int, model, config: Dict[str, Any], **kwargs):
        """No action needed at epoch end."""
        pass


class CheckpointCallback(TrainingCallback):
    """
    Callback for saving model checkpoints during training.
    
    Behavior:

    - Saves the best model based on ``convergent_metric``
      (``correlation`` / ``roc_auc`` / ``min_auc_n_o``) or loss when ``use_loss_for_best=True``

    - For correlation: ``|correlation|`` by default; ``prefer_convergent_checkpoint`` can
      prefer threshold-satisfying steps

    - For roc_auc: raw one-vs-rest AUROC (higher better); label ``+1`` must have
      positive mean encoding; bipolar magnitude gating remains optional

    - For min_auc_n_o: ``min(auc_n, auc_o)`` so neutrals alone cannot carry selection
    - When use_loss_for_best=True (e.g. supervised_decoder), best = lowest loss
    - Saves periodic checkpoints every checkpoint_interval steps (if checkpoints enabled)
    - Tracks step-0 metrics but does not materialize step-0 as a selectable best checkpoint
    
    Args:
        output: Directory to save checkpoints
        checkpoints: If True, saves checkpoints every checkpoint_interval steps.
        keep_only_best: If True, keeps only the best model checkpoint (default: True)
        checkpoint_interval: Interval in steps for saving checkpoints (default: 5000)
        use_loss_for_best: If True, best checkpoint = lowest loss (e.g. for supervised_decoder where correlation is N/A)
    """

    def __init__(self, output: str, checkpoints: Union[bool, int] = False, keep_only_best: bool = True, checkpoint_interval: int = 5000, use_loss_for_best: bool = False):
        self.output = output
        self.checkpoints = checkpoints
        self.keep_only_best = keep_only_best
        self.use_loss_for_best = use_loss_for_best
        if checkpoints is True:
            self.checkpoint_step = checkpoint_interval
        elif isinstance(checkpoints, int):
            self.checkpoint_step = checkpoints
        else:
            self.checkpoint_step = 0
        self.best_score = None
        self.best_step = None
        self.best_epoch = None
        self._best_rank: Optional[tuple] = None
        self.component_best_states: Dict[str, Dict[str, Any]] = {}
        self._best_merge_rank: Optional[tuple] = None
        self._best_merge_summary: Optional[Dict[str, Any]] = None
        self._selection_metric: str = "loss" if use_loss_for_best else "correlation"

    def on_step_end(self, step: int, loss: float, model, config: Dict[str, Any],
                    training_stats: Dict[str, Any], **kwargs):
        """Save checkpoint if needed."""
        # Always refresh running local-best component slices first. For
        # component-split models these define the best merged checkpoint.
        updated_components = update_component_best_states(
            self.component_best_states,
            model=model,
            eval_result=kwargs.get("eval_result"),
            step=step,
            epoch=kwargs.get("epoch"),
            selection_metric=_selection_metric_from_config(config),
        )
        if updated_components:
            path = save_component_best_states(self.output, self.component_best_states)
            if path:
                logger.debug(
                    "Saved component-best tensor state(s) for %s component(s) to %s",
                    len(updated_components),
                    path,
                )

        component_split = model_has_component_split(model)
        if component_split and self.component_best_states and not self.use_loss_for_best:
            merge_summary = summary_from_component_best_states(self.component_best_states)
            merge_rank = merge_rank_from_summary(merge_summary)
            component_selection_metric = _selection_metric_from_config(config)
            # Component-local states are monotonic under their configured
            # selection metric. Any update therefore creates the new best E
            # merge even when its aggregate correlation is lower.
            is_better = (
                bool(updated_components)
                if component_selection_metric != "correlation"
                else self._best_merge_rank is None or merge_rank > self._best_merge_rank
            )
            score_for_log = (
                None
                if merge_summary is None
                else (
                    sum(
                        float(state.get("encoding_e"))
                        for state in self.component_best_states.values()
                        if isinstance(state.get("encoding_e"), (int, float))
                    ) / len(self.component_best_states)
                    if component_selection_metric == "encoding_e"
                    and self.component_best_states
                    and all(
                        isinstance(state.get("encoding_e"), (int, float))
                        for state in self.component_best_states.values()
                    )
                    else merge_summary.get("correlation_mean")
                )
            )
            if is_better and merge_summary is not None:
                was_first = self._best_merge_rank is None
                old_score = self.best_score
                self._best_merge_rank = merge_rank
                self._best_merge_summary = merge_summary
                self.best_score = score_for_log
                self.best_step = step
                self.best_epoch = kwargs.get("epoch", 0)
                if was_first:
                    logger.debug(
                        "First component-best merge correlation_mean: %s at step %s (%s/%s converged)",
                        f"{score_for_log:.4f}" if isinstance(score_for_log, (int, float)) else score_for_log,
                        step,
                        merge_summary.get("n_converged"),
                        merge_summary.get("n_components"),
                    )
                elif isinstance(score_for_log, (int, float)):
                    logger.debug(
                        "New best component-best merge: correlation_mean=%s at step %s "
                        "(%s/%s converged; previous mean=%s)",
                        f"{score_for_log:.4f}",
                        step,
                        merge_summary.get("n_converged"),
                        merge_summary.get("n_components"),
                        f"{old_score:.4f}" if isinstance(old_score, (int, float)) else old_score,
                    )
                if step > 0:
                    best_output = f"{self.output}_best"
                    merge_payload = payload_from_component_best_states(self.component_best_states)
                    training_info = {
                        "losses": kwargs.get("losses", []),
                        "best_score_checkpoint": {
                            "correlation": score_for_log,
                            "loss": loss,
                            "global_step": step,
                            "epoch": self.best_epoch,
                            "selection": "component_best_merge",
                            "selection_metric": f"component_{component_selection_metric}",
                            "component_best_steps": merge_summary.get("component_best_steps"),
                            "n_converged": merge_summary.get("n_converged"),
                            "n_components": merge_summary.get("n_components"),
                        },
                        "training_stats": training_stats,
                        "training_args": config,
                        "component_best_merge": merge_payload,
                    }
                    save_merged_component_best_checkpoint(
                        model,
                        self.component_best_states,
                        best_output=best_output,
                        training_info=training_info,
                    )
                    logger.debug(
                        "Saved component-best merged checkpoint at step %s "
                        "(correlation_mean: %s)",
                        step,
                        f"{score_for_log:.4f}" if isinstance(score_for_log, (int, float)) else score_for_log,
                    )
                else:
                    logger.debug(
                        "Initial step-0 evaluation updated component-best slices, but is not saved "
                        "as a selectable merged checkpoint; trained or final merge will be used instead."
                    )
        else:
            selection_metric = _selection_metric_from_config(
                config, use_loss_for_best=self.use_loss_for_best
            )
            self._selection_metric = selection_metric
            score = None
            rank = None
            if selection_metric == "loss":
                # Best = lowest loss (e.g. supervised_decoder; correlation not meaningful)
                is_better = self.best_score is None or loss < self.best_score
                score_for_log = loss
            else:
                score = _current_step_selection_score(
                    step=step,
                    metric=selection_metric,
                    training_stats=training_stats,
                    eval_result=kwargs.get("eval_result"),
                )
                # No current-step evaluation result; do not treat stale/sentinel scores as a checkpoint score.
                if score is None:
                    is_better = False
                    score_for_log = None
                else:
                    mean_by_class = _mean_by_class_for_step(
                        training_stats,
                        step,
                        kwargs.get("eval_result"),
                    )
                    rank = metric_checkpoint_rank(
                        step=step,
                        score=score,
                        metric=selection_metric,
                        mean_by_class=mean_by_class,
                        score_threshold=_config_get(config, "convergent_score_threshold", None),
                        mean_threshold=_config_get(config, "convergent_mean_by_class_threshold", None),
                        prefer_convergent=bool(
                            _config_get(config, "prefer_convergent_checkpoint", False)
                        ),
                    )
                    is_better = self._best_rank is None or rank > self._best_rank
                    score_for_log = score

            if is_better:
                was_first = self.best_score is None
                old_score = self.best_score
                self.best_score = loss if selection_metric == "loss" else score
                self.best_step = step
                self.best_epoch = kwargs.get('epoch', 0)
                if rank is not None:
                    self._best_rank = rank

                metric_label = selection_metric
                if was_first and score_for_log is not None:
                    logger.debug(f'First {metric_label}: {score_for_log:.4f} at step {step}')
                elif score_for_log is not None:
                    logger.debug(f'New best {metric_label}: {score_for_log:.4f} at step {step} (previous: {old_score:.4f})')

                if step > 0:
                    best_output = f'{self.output}_best'
                    best_ckpt = {
                        'loss': loss,
                        'global_step': step,
                        'epoch': self.best_epoch,
                        'selection_metric': selection_metric,
                    }
                    if selection_metric == "loss":
                        best_ckpt['correlation'] = None
                        best_ckpt['roc_auc'] = None
                        best_ckpt['min_auc_n_o'] = None
                        best_ckpt['loss'] = self.best_score
                    elif selection_metric in {"roc_auc", "min_auc_n_o", "encoding_e"}:
                        best_ckpt[selection_metric] = self.best_score
                        best_ckpt['correlation'] = _current_step_correlation(
                            step=step,
                            training_stats=training_stats,
                            eval_result=kwargs.get("eval_result"),
                        )
                        ev = kwargs.get("eval_result")
                        if isinstance(ev, dict):
                            if selection_metric != "roc_auc":
                                best_ckpt['roc_auc'] = ev.get("roc_auc")
                            if selection_metric != "min_auc_n_o":
                                best_ckpt['min_auc_n_o'] = ev.get("min_auc_n_o")
                            best_ckpt['roc_auc_neutral'] = ev.get("roc_auc_neutral")
                            best_ckpt['roc_auc_other'] = ev.get("roc_auc_other")
                            best_ckpt['auc_rival'] = ev.get("auc_rival")
                            best_ckpt['class_exclusivity'] = ev.get("class_exclusivity")
                            best_ckpt['neutral_specificity'] = ev.get("neutral_specificity")
                    else:
                        best_ckpt['correlation'] = self.best_score
                        if isinstance(kwargs.get("eval_result"), dict):
                            ev = kwargs["eval_result"]
                            best_ckpt['roc_auc'] = ev.get("roc_auc")
                            best_ckpt['min_auc_n_o'] = ev.get("min_auc_n_o")
                            best_ckpt['roc_auc_neutral'] = ev.get("roc_auc_neutral")
                            best_ckpt['roc_auc_other'] = ev.get("roc_auc_other")
                    training_info = {
                        'losses': kwargs.get('losses', []),
                        'best_score_checkpoint': best_ckpt,
                        'training_stats': training_stats,
                        'training_args': config,
                    }
                    model.save_pretrained(best_output, training=training_info)
                    if score_for_log is not None:
                        logger.debug(
                            f'Saved best model checkpoint at step {step} '
                            f'({metric_label}: {score_for_log:.4f})'
                        )
                else:
                    logger.debug(
                        "Initial step-0 evaluation is currently best, but is not saved as a selectable "
                        "checkpoint; trained or final model state will be used instead."
                    )
        
        # Save periodic checkpoint
        if self.checkpoint_step > 0 and step % self.checkpoint_step == 0 and step > 0:
            checkpoint_output = f'{self.output}_step_{step}'
            model.save_pretrained(checkpoint_output)
            logger.info(f'Saved checkpoint at step {step}')
    
    def on_epoch_end(self, epoch: int, model, config: Dict[str, Any], 
                    training_stats: Dict[str, Any], losses: list, **kwargs):
        """Save final model at end of epoch."""
        selection_metric = getattr(self, "_selection_metric", None) or (
            "loss" if self.use_loss_for_best else "correlation"
        )
        best_score_checkpoint = {
            'global_step': self.best_step,
            'epoch': epoch,
            'selection_metric': selection_metric,
        }
        if selection_metric == "loss" or self.use_loss_for_best:
            best_score_checkpoint['loss'] = self.best_score
            best_score_checkpoint['correlation'] = None
            best_score_checkpoint['roc_auc'] = None
            best_score_checkpoint['min_auc_n_o'] = None
        elif selection_metric in {"roc_auc", "min_auc_n_o", "encoding_e"}:
            best_score_checkpoint[selection_metric] = self.best_score
            best_score_checkpoint['correlation'] = training_stats.get("correlation")
            for key in (
                "roc_auc", "min_auc_n_o", "roc_auc_neutral", "roc_auc_other",
                "auc_rival", "class_exclusivity", "neutral_specificity", "encoding_e",
            ):
                if key == selection_metric:
                    continue
                hist = training_stats.get(key)
                if isinstance(hist, dict) and self.best_step in hist:
                    best_score_checkpoint[key] = hist[self.best_step]
                elif isinstance(hist, (int, float)):
                    best_score_checkpoint[key] = hist
        else:
            best_score_checkpoint['correlation'] = self.best_score
            for key in ("roc_auc", "min_auc_n_o", "roc_auc_neutral", "roc_auc_other"):
                hist = training_stats.get(key)
                if isinstance(hist, dict) and self.best_step in hist:
                    best_score_checkpoint[key] = hist[self.best_step]
                elif isinstance(hist, (int, float)):
                    best_score_checkpoint[key] = hist
        if isinstance(self._best_merge_summary, dict):
            best_score_checkpoint['selection'] = 'component_best_merge'
            best_score_checkpoint['component_best_steps'] = self._best_merge_summary.get('component_best_steps')
            best_score_checkpoint['n_converged'] = self._best_merge_summary.get('n_converged')
            best_score_checkpoint['n_components'] = self._best_merge_summary.get('n_components')
        training_info = {
            'losses': losses,
            'best_score_checkpoint': best_score_checkpoint,
            'training_stats': training_stats,
            'training_args': config,
        }
        # Save full model (including gradiend_context.json) so the directory is always loadable.
        # When save_only_best is True, _handle_keep_only_best may replace this with output_best.
        # For component-split, epoch output is the live training weights; the selectable best
        # remains the merged component-best checkpoint under output_best.
        model.save_pretrained(self.output, training=training_info)
        logger.info(f'Saved model after epoch {epoch + 1} to {self.output}')


def _mean_recent_loss(last_losses, n_loss_report: int) -> Optional[float]:
    """Mean of losses in the current report window (last ``n_loss_report`` steps)."""
    if not last_losses:
        return None
    window = n_loss_report if n_loss_report and n_loss_report > 0 else len(last_losses)
    recent = last_losses[-window:]
    return float(sum(recent) / len(recent))


class LoggingCallback(TrainingCallback):
    """Callback for logging training progress."""
    
    def __init__(self, n_loss_report: int = 100, loss_only: bool = False):
        """
        Args:
            n_loss_report: Log every N steps. Reported Loss is the mean over that window.
            loss_only: If True (e.g. supervised_decoder), correlation is N/A; log loss only.
        """
        self.n_loss_report = n_loss_report
        self.loss_only = loss_only
        # Track best checkpoint across ALL evaluations (including the initial step 0 eval)
        # with the same convergent-preferring rank used by CheckpointCallback.
        self._best_checkpoint_rank: Optional[tuple] = None
        self._best_corr_including_start: Optional[float] = None
    
    def on_step_end(self, step: int, loss: float, model, config: Dict[str, Any],
                    training_stats: Dict[str, Any], **kwargs):
        """Log training progress."""
        last_losses = kwargs.get("last_losses") or ()
        eval_result = kwargs.get("eval_result")
        should_log = (
            step % self.n_loss_report == 0 or
            step == 0 or
            kwargs.get('last_iteration', False)
        )
        # Log if we should log AND (have losses OR have eval results for step 0)
        if should_log and (last_losses or (step == 0 and eval_result is not None)):
            eval_is_enabled = _eval_enabled(config, loss_only=self.loss_only)
            selection_metric = _selection_metric_from_config(config, use_loss_for_best=self.loss_only)
            corr = (
                _current_step_correlation(
                    step=step,
                    training_stats=training_stats,
                    eval_result=eval_result,
                )
                if eval_is_enabled
                else None
            )
            selection_score = (
                _current_step_selection_score(
                    step=step,
                    metric=selection_metric,
                    training_stats=training_stats,
                    eval_result=eval_result,
                )
                if eval_is_enabled and selection_metric != "loss"
                else None
            )

            # Try to get mean encoded values per class for the current step
            mean_by_class_hist = training_stats.get('mean_by_class', {})
            current_means = None
            if isinstance(mean_by_class_hist, dict):
                current_means = mean_by_class_hist.get(step)
            # Label value -> display name (stored once; fallback: step->dict from older runs)
            _lv_to_name_raw = training_stats.get('label_value_to_class_name') or {}
            if not _lv_to_name_raw:
                label_value_to_class_name = {}
            else:
                first_val = next(iter(_lv_to_name_raw.values()), None)
                label_value_to_class_name = first_val if isinstance(first_val, dict) else _lv_to_name_raw

            mean_str = ""
            if isinstance(current_means, dict) and current_means:
                def _mean_display_name(label_val):
                    return label_value_to_class_name.get(
                        label_val,
                        label_value_to_class_name.get(float(label_val), "neutral" if label_val in (0, 0.0) else str(label_val))
                    )

                keys = sorted(current_means.keys(), key=lambda x: (0 if x == 0 else (-1 if x < 0 else 1), x))
                parts = [
                    f"{_mean_display_name(k)}: {float(current_means[k]):.4f}"
                    for k in keys if isinstance(current_means.get(k), (int, float))
                ]
                mean_str = ", ".join(parts)

            neutral_str = ""
            neutral_means = eval_result.get("neutral_mean_by_type") if isinstance(eval_result, dict) else None
            if not isinstance(neutral_means, dict):
                neutral_hist = training_stats.get("neutral_mean_by_type")
                neutral_means = neutral_hist.get(step) if isinstance(neutral_hist, dict) else None
            neutral_abs_means = eval_result.get("abs_mean_by_type") if isinstance(eval_result, dict) else None
            if not isinstance(neutral_abs_means, dict):
                neutral_abs_hist = training_stats.get("abs_mean_by_type")
                neutral_abs_means = neutral_abs_hist.get(step) if isinstance(neutral_abs_hist, dict) else None
            if isinstance(neutral_means, dict) and neutral_means:
                neutral_parts = []
                for key, value in sorted(neutral_means.items(), key=lambda item: str(item[0])):
                    if not isinstance(value, (int, float)):
                        continue
                    part = f"{key}: {float(value):.4f}"
                    if isinstance(neutral_abs_means, dict):
                        abs_value = neutral_abs_means.get(key)
                        if isinstance(abs_value, (int, float)):
                            part += f" abs={float(abs_value):.4f}"
                    neutral_parts.append(part)
                neutral_str = ", ".join(neutral_parts)

            suffix = ""
            if eval_is_enabled and selection_score is not None:
                # Match CheckpointCallback ranking for the active selection metric.
                mean_by_class = _mean_by_class_for_step(training_stats, step, eval_result)
                rank = metric_checkpoint_rank(
                    step=step,
                    score=selection_score,
                    metric=selection_metric,
                    mean_by_class=mean_by_class,
                    score_threshold=_config_get(config, "convergent_score_threshold", None),
                    mean_threshold=_config_get(config, "convergent_mean_by_class_threshold", None),
                    prefer_convergent=bool(
                        _config_get(config, "prefer_convergent_checkpoint", False)
                    ),
                )
                if self._best_checkpoint_rank is None or rank > self._best_checkpoint_rank:
                    suffix = " (new best)"
                    self._best_checkpoint_rank = rank
                    self._best_corr_including_start = float(selection_score)
            # Prefer mean over the report window so class cycling (e.g. M/F vs neutral
            # identity) does not make the logged loss look like a crash.
            reported_loss = _mean_recent_loss(last_losses, self.n_loss_report)
            if reported_loss is None and loss is not None:
                reported_loss = float(loss)
            parts = [f"Step {step}", f"Loss: {_format_loss_for_log(reported_loss)}"]
            if eval_is_enabled:
                corr_str = "N/A" if corr is None else f"{corr:.4f}"
                parts.append(f"Correlation: {corr_str}")
                if selection_metric == "roc_auc":
                    auc_str = "N/A" if selection_score is None else f"{selection_score:.4f}"
                    parts.append(f"AUROC: {auc_str}")
                elif selection_metric == "min_auc_n_o":
                    auc_str = "N/A" if selection_score is None else f"{selection_score:.4f}"
                    parts.append(f"minAUC: {auc_str}")
                    if isinstance(eval_result, dict):
                        auc_n = eval_result.get("roc_auc_neutral")
                        auc_o = eval_result.get("roc_auc_other")
                        detail = []
                        if isinstance(auc_n, (int, float)):
                            detail.append(f"auc_n={float(auc_n):.4f}")
                        if isinstance(auc_o, (int, float)):
                            detail.append(f"auc_o={float(auc_o):.4f}")
                        if detail:
                            parts.append("(" + " ".join(detail) + ")")
                elif selection_metric == "encoding_e":
                    e_str = "N/A" if selection_score is None else f"{selection_score:.4f}"
                    parts.append(f"E: {e_str}")
                    if isinstance(eval_result, dict):
                        detail = []
                        for key, label in (
                            ("roc_auc_neutral", "auc_n"),
                            ("roc_auc_other", "auc_rival"),
                            ("class_exclusivity", "excl"),
                            ("neutral_specificity", "spec"),
                        ):
                            value = eval_result.get(key)
                            if isinstance(value, (int, float)):
                                detail.append(f"{label}={float(value):.4f}")
                        if detail:
                            parts.append("(" + " ".join(detail) + ")")
                component_fragment = ""
                if isinstance(eval_result, dict):
                    component_fragment = format_component_convergence_fragment(
                        eval_result.get("components"),
                        summary=eval_result.get("component_summary"),
                    )
                if component_fragment:
                    parts.append(component_fragment)
                if mean_str:
                    parts.append(f"mean: {mean_str}")
                if neutral_str:
                    parts.append(f"neutral: {neutral_str}")
            logger.info(", ".join(parts) + suffix)
    
    def on_epoch_end(self, epoch: int, model, config: Dict[str, Any], 
                    time_stats: Dict[str, float], **kwargs):
        """Log epoch completion."""
        try:
            import humanize
            import datetime
            
            def humanize_time(seconds):
                return humanize.naturaldelta(datetime.timedelta(seconds=seconds))
            
            total = humanize_time(time_stats.get("total", 0))
            total_epochs = config.get("num_train_epochs", config.get("epochs", 1))
            logger.info(
                f'Epoch {epoch + 1}/{total_epochs} finished in {total}. '
            )
        except ImportError:
            t = time_stats.get("total", 0)
            total_epochs = config.get("num_train_epochs", config.get("epochs", 1))
            logger.info(f'Epoch {epoch + 1}/{total_epochs} finished. Total: {t:.2f}s')

def get_default_callbacks(config: Any) -> List[TrainingCallback]:
    """
    Build the default callback list used by the core training loop.

    Order: Evaluation -> Normalization -> Checkpoint -> Logging.
    Accepts TrainingArguments or a dict (e.g. training_args.to_dict()).
    """
    if hasattr(config, "output_dir"):
        output = config.output_dir or ""
    else:
        output = config.get("output_dir", "")
    supervised_decoder = getattr(config, "supervised_decoder", False) or config.get("supervised_decoder", False)
    selection_metric = _selection_metric_from_config(config, use_loss_for_best=bool(supervised_decoder))
    use_loss_for_best = bool(supervised_decoder) or selection_metric == "loss"
    if hasattr(config, "eval_steps"):
        n_eval = config.eval_steps if config.do_eval and not use_loss_for_best else 0
        do_eval = config.do_eval and not use_loss_for_best  # skip eval for loss-only selection
        evaluate = config.evaluate_fn
        checkpoints = config.save_strategy == "steps"
        keep_only_best = config.save_only_best
        checkpoint_interval = config.save_steps
        normalize_gradiend = config.normalize_gradiend
    else:
        output = config.get("output_dir", "")
        n_eval = config.get("eval_steps", 250) if (config.get("do_eval", True) and not use_loss_for_best) else 0
        do_eval = config.get("do_eval", True) and not use_loss_for_best
        evaluate = config.get("evaluate_fn")
        checkpoints = config.get("save_strategy", "best") == "steps"
        keep_only_best = config.get("save_only_best", True)
        checkpoint_interval = config.get("save_steps", 5000)
        normalize_gradiend = config.get("normalize_gradiend", True)

    return [
        EvaluationCallback(evaluate_fn=evaluate, n_evaluation=n_eval, do_eval=do_eval),
        NormalizationCallback(normalize=normalize_gradiend),
        CheckpointCallback(
            output=output,
            checkpoints=checkpoints,
            keep_only_best=keep_only_best,
            checkpoint_interval=checkpoint_interval,
            use_loss_for_best=use_loss_for_best,
        ),
        LoggingCallback(n_loss_report=n_eval if n_eval > 0 else 100, loss_only=use_loss_for_best),
    ]
