"""Component-aware seed selection and GRADIEND tensor stitching helpers."""

from __future__ import annotations

import copy
import os
import shutil
from typing import Any, Dict, List, Optional, Sequence

import torch


COMPONENT_BEST_DIRNAME = "component_best"
COMPONENT_BEST_FILENAME = "component_best.pt"
COMPONENT_BEST_FORMAT_VERSION = 1


def component_best_staging_dir(output_dir: str) -> str:
    """Sibling directory used while a seed is still training."""
    return f"{output_dir}_{COMPONENT_BEST_DIRNAME}"


def component_best_dir(output_dir: str) -> str:
    """Directory inside a completed seed/model run that stores component-best tensor slices."""
    return os.path.join(output_dir, COMPONENT_BEST_DIRNAME)


def component_best_path(output_dir: str) -> str:
    """Path to the completed component-best tensor payload for a run."""
    return os.path.join(component_best_dir(output_dir), COMPONENT_BEST_FILENAME)


def component_best_staging_path(output_dir: str) -> str:
    """Path to the in-progress component-best tensor payload for a run."""
    return os.path.join(component_best_staging_dir(output_dir), COMPONENT_BEST_FILENAME)


def _target_mean_converged(mean_by_class: Any, mean_threshold: Any) -> tuple[bool, Optional[float], Optional[float]]:
    if not isinstance(mean_by_class, dict):
        return False, None, None
    target_means: List[float] = []
    for label, value in mean_by_class.items():
        try:
            label_value = float(label)
        except (TypeError, ValueError):
            continue
        if label_value == 0.0 or not isinstance(value, (int, float)):
            continue
        target_means.append(float(value))
    if len(target_means) != 2:
        return False, None, None
    product = target_means[0] * target_means[1]
    min_abs = min(abs(value) for value in target_means)
    mean_ok = mean_threshold is None or min_abs >= float(mean_threshold)
    return product < 0 and mean_ok, product, min_abs


def _component_sort_key(component_id: str, payload: Any) -> tuple[int, str]:
    if isinstance(payload, dict) and isinstance(payload.get("component_index"), int):
        return int(payload["component_index"]), str(component_id)
    return 10**9, str(component_id)


def _normalize_step_dict(value: Any) -> Dict[int, Any]:
    if not isinstance(value, dict):
        return {}
    out: Dict[int, Any] = {}
    for key, item in value.items():
        try:
            out[int(key)] = item
        except (TypeError, ValueError):
            continue
    return out


def _numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _selection_score(candidate: Dict[str, Any], selection_metric: str) -> Optional[float]:
	name = str(selection_metric or "correlation").strip().lower()
	if name in {"e", "encoding-e", "encodinge"}:
		name = "encoding_e"
	value = _numeric(candidate.get(name))
	if value is None and isinstance(candidate.get("metrics"), dict):
		value = _numeric(candidate["metrics"].get(name))
	return abs(value) if name == "correlation" and value is not None else value


def _candidate_rank(
    candidate: Dict[str, Any],
    selection_metric: str = "correlation",
) -> tuple[int, float, float, int]:
    corr = _numeric(candidate.get("correlation"))
    min_abs = _numeric(candidate.get("min_target_class_abs_mean"))
    step = candidate.get("global_step")
    try:
        step_int = int(step)
    except (TypeError, ValueError):
        step_int = -1
    return (
        1 if bool(candidate.get("converged")) else 0,
        _selection_score(candidate, selection_metric)
        if _selection_score(candidate, selection_metric) is not None
        else float("-inf"),
        min_abs if min_abs is not None else float("-inf"),
        step_int,
    )


def _component_candidate_from_metrics(
    *,
    component_id: str,
    metrics: Dict[str, Any],
    step: int,
    threshold: Optional[float],
    mean_threshold: Optional[float],
    selection_metric: str = "correlation",
) -> Dict[str, Any]:
    corr = _numeric(metrics.get("correlation"))
    means_ok, target_mean_product, min_abs = _target_mean_converged(
        metrics.get("mean_by_class"),
        mean_threshold,
    )
    score_ok = threshold is None or (corr is not None and abs(corr) >= float(threshold))
    step_ok = step > 0
    converged = bool(step_ok and score_ok and means_ok)
    metrics_copy = copy.deepcopy(metrics)
    metrics_copy["component_id"] = metrics_copy.get("component_id", component_id)
    metrics_copy["best_component_global_step"] = step
    convergence = {
        "component_index": metrics.get("component_index"),
        "component_id": metrics.get("component_id", component_id),
        "component_label": metrics.get("component_label"),
        "converged": converged,
        "correlation": corr,
        "target_mean_product": target_mean_product,
        "min_target_class_abs_mean": min_abs,
        "best_component_global_step": step,
        "selection_metric": f"component_{selection_metric}",
    }
    return {
        "component_index": metrics.get("component_index"),
        "component_id": metrics.get("component_id", component_id),
        "component_label": metrics.get("component_label"),
        "global_step": step,
        "correlation": corr,
        "encoding_e": _numeric(metrics.get("encoding_e")),
        "target_mean_product": target_mean_product,
        "min_target_class_abs_mean": min_abs,
        "converged": converged,
        "metrics": metrics_copy,
        "convergence": convergence,
    }


def _component_by_id(gradiend: Any, component_id: str) -> Any:
    gradiend._require_built()
    for component in getattr(gradiend, "component_slices", ()):
        if str(component.id) == str(component_id):
            return component
    raise KeyError(f"Unknown GRADIEND component id: {component_id!r}")


def extract_gradiend_component_state(model: Any, component_id: str) -> Dict[str, Any]:
    """
    Capture the trainable tensor slices for one explicit split component.

    Component stitching is defined over ``(seed, step, component_id)``
    candidates, not over a seed's aggregate best checkpoint.  The saved state is
    therefore only the component-owned encoder/decoder slice needed to recreate
    that candidate later.
    """
    gradiend = model.gradiend
    component = _component_by_id(gradiend, component_id)
    target_encoder = gradiend.encoder[0].linear
    target_decoder = gradiend.decoder[0].linear
    start = int(component.start)
    end = int(component.end)
    state: Dict[str, Any] = {
        "component_id": str(component.id),
        "component_index": list(getattr(gradiend, "component_slices", ())).index(component),
        "start": start,
        "end": end,
        "encoder_weight": target_encoder.weight[:, start:end].detach().cpu().clone(),
        "decoder_weight": target_decoder.weight[start:end, :].detach().cpu().clone(),
    }
    if target_decoder.bias is not None:
        state["decoder_bias"] = target_decoder.bias[start:end].detach().cpu().clone()
    if target_encoder.bias is not None:
        state["encoder_bias"] = target_encoder.bias.detach().cpu().clone()
    return state


def update_component_best_states(
    best_states: Dict[str, Dict[str, Any]],
    *,
    model: Any,
    eval_result: Optional[Dict[str, Any]],
    step: int,
    epoch: Optional[int] = None,
    selection_metric: str = "correlation",
) -> List[str]:
    """
    Update in-memory component-best states from one evaluation result.

    A component candidate is ranked by its own convergence status and absolute
    correlation, then by the target-class mean margin.  Aggregate correlation is
    deliberately not part of this decision.
    """
    if step <= 0 or not isinstance(eval_result, dict):
        return []
    components = eval_result.get("components")
    if not isinstance(components, dict):
        return []
    metrics_by_component = components.get("metrics_by_component")
    if not isinstance(metrics_by_component, dict) or not metrics_by_component:
        return []
    convergence_by_component = components.get("convergence_by_component")
    if not isinstance(convergence_by_component, dict):
        convergence_by_component = {}

    signature = _component_signature(model.gradiend)
    updated: List[str] = []
    for component_id, metrics in metrics_by_component.items():
        if not isinstance(metrics, dict):
            continue
        component_key = str(component_id)
        convergence = convergence_by_component.get(component_key)
        if not isinstance(convergence, dict):
            convergence = {}
        corr = _numeric(metrics.get("correlation"))
        min_abs = _numeric(convergence.get("min_target_class_abs_mean"))
        candidate = {
            "format_version": COMPONENT_BEST_FORMAT_VERSION,
            "signature": copy.deepcopy(signature),
            "component_id": component_key,
            "component_index": metrics.get("component_index"),
            "component_label": metrics.get("component_label"),
            "global_step": int(step),
            "epoch": epoch,
            "correlation": corr,
            "min_target_class_abs_mean": min_abs,
            "target_mean_product": convergence.get("target_mean_product"),
            "converged": bool(convergence.get("converged")),
            "selection_metric": f"component_{selection_metric}",
            "encoding_e": _numeric(metrics.get("encoding_e")),
            "metrics": copy.deepcopy(metrics),
            "convergence": copy.deepcopy(convergence),
            "state": extract_gradiend_component_state(model, component_key),
        }
        current = best_states.get(component_key)
        if current is None or _candidate_rank(candidate, selection_metric) > _candidate_rank(current, selection_metric):
            best_states[component_key] = candidate
            updated.append(component_key)
    return updated


def save_component_best_states(output_dir: str, best_states: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Persist component-best tensor slices to a sibling staging directory."""
    if not output_dir or not best_states:
        return None
    staging_dir = component_best_staging_dir(output_dir)
    os.makedirs(staging_dir, exist_ok=True)
    path = component_best_staging_path(output_dir)
    torch.save(
        {
            "format_version": COMPONENT_BEST_FORMAT_VERSION,
            "components": best_states,
        },
        path,
    )
    return path


def finalize_component_best_states(output_dir: str) -> Optional[str]:
    """Move staged component-best tensors into the completed run directory."""
    if not output_dir:
        return None
    staging_path = component_best_staging_path(output_dir)
    if not os.path.exists(staging_path):
        return None
    final_dir = component_best_dir(output_dir)
    os.makedirs(final_dir, exist_ok=True)
    final_path = component_best_path(output_dir)
    shutil.copy2(staging_path, final_path)
    try:
        shutil.rmtree(component_best_staging_dir(output_dir))
    except Exception:
        pass
    return final_path


def _torch_load_component_payload(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_component_best_states(output_dir: str) -> Optional[Dict[str, Any]]:
    """Load component-best tensor states for a completed run when available."""
    for path in (component_best_path(output_dir), component_best_staging_path(output_dir)):
        if os.path.exists(path):
            payload = _torch_load_component_payload(path)
            return payload if isinstance(payload, dict) else None
    return None


def apply_saved_component_best_states(model: Any, output_dir: str) -> Optional[List[Dict[str, Any]]]:
    """
    Rewrite ``model`` GRADIEND weights with each component's local-best tensor slice.

    Component-split training tracks a running best merge via CPU-resident component
    slices.  Applying those slices yields the delivered single-seed model.
    """
    if not output_dir or model is None:
        return None
    gradiend = getattr(model, "gradiend", None)
    if gradiend is None or not bool(getattr(gradiend, "has_component_split", False)):
        return None
    payload = load_component_best_states(output_dir)
    if not isinstance(payload, dict):
        return None
    components = payload.get("components")
    if not isinstance(components, dict) or not components:
        return None
    rows = stitch_gradiend_component_states(model, components)
    return rows if rows else None


def model_has_component_split(model: Any) -> bool:
    gradiend = getattr(model, "gradiend", None)
    return bool(gradiend is not None and getattr(gradiend, "has_component_split", False))


def summary_from_component_best_states(best_states: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build a merge summary from the running per-component best candidates (CPU only)."""
    if not best_states:
        return None
    metrics_by_component: Dict[str, Dict[str, Any]] = {}
    convergence_by_component: Dict[str, Dict[str, Any]] = {}
    component_best_steps: Dict[str, Any] = {}
    for component_id, entry in best_states.items():
        if not isinstance(entry, dict):
            continue
        component_key = str(component_id)
        metrics = entry.get("metrics")
        convergence = entry.get("convergence")
        if not isinstance(metrics, dict):
            metrics = {
                "component_id": component_key,
                "component_index": entry.get("component_index"),
                "correlation": entry.get("correlation"),
            }
        if not isinstance(convergence, dict):
            convergence = {
                "component_id": component_key,
                "converged": bool(entry.get("converged")),
                "min_target_class_abs_mean": entry.get("min_target_class_abs_mean"),
                "target_mean_product": entry.get("target_mean_product"),
            }
        metrics_by_component[component_key] = copy.deepcopy(metrics)
        convergence_by_component[component_key] = copy.deepcopy(convergence)
        component_best_steps[component_key] = entry.get("global_step")
    if not metrics_by_component:
        return None
    summary = _summary_from_component_payload(metrics_by_component, convergence_by_component)
    summary["component_best_steps"] = component_best_steps
    summary["convergence_unit"] = "component_best_merge"
    return summary


def merge_rank_from_summary(summary: Optional[Dict[str, Any]]) -> tuple:
    """Rank a running component-best merge for checkpoint selection."""
    if not isinstance(summary, dict):
        return (0, float("-inf"), float("-inf"))
    n_converged = _numeric(summary.get("n_converged"))
    corr = _numeric(summary.get("correlation_mean"))
    min_abs = _numeric(summary.get("min_target_class_abs_mean"))
    return (
        int(n_converged) if n_converged is not None else 0,
        abs(corr) if corr is not None else float("-inf"),
        min_abs if min_abs is not None else float("-inf"),
    )


def snapshot_gradiend_trainable_state(model: Any) -> Dict[str, torch.Tensor]:
    """CPU clone of GRADIEND trainable params (not the base model)."""
    gradiend = model.gradiend
    return {
        name: param.detach().cpu().clone()
        for name, param in gradiend.named_parameters()
    }


def restore_gradiend_trainable_state(model: Any, snapshot: Dict[str, torch.Tensor]) -> None:
    """Restore GRADIEND weights from :func:`snapshot_gradiend_trainable_state`."""
    with torch.no_grad():
        for name, param in model.gradiend.named_parameters():
            saved = snapshot.get(name)
            if saved is None:
                continue
            param.copy_(saved.to(device=param.device, dtype=param.dtype))


def save_merged_component_best_checkpoint(
    model: Any,
    best_states: Dict[str, Dict[str, Any]],
    *,
    best_output: str,
    training_info: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Stitch the running component-best merge into ``model``, save, then restore training weights.

    Only GRADIEND tensors are temporarily rewritten (base model stays put), so GPU
    RAM cost is a CPU snapshot of encoder/decoder weights — not a second base model.
    """
    if not model_has_component_split(model) or not best_states:
        return None
    summary = summary_from_component_best_states(best_states)
    snapshot = snapshot_gradiend_trainable_state(model)
    try:
        stitch_gradiend_component_states(model, best_states)
        model.save_pretrained(best_output, training=training_info)
    finally:
        restore_gradiend_trainable_state(model, snapshot)
        del snapshot
    return summary


def payload_from_component_best_states(best_states: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Full components payload (metrics + convergence + summary) for the running merge."""
    summary = summary_from_component_best_states(best_states)
    if summary is None:
        return None
    metrics_by_component = {}
    convergence_by_component = {}
    for component_id, entry in best_states.items():
        if not isinstance(entry, dict):
            continue
        key = str(component_id)
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        convergence = entry.get("convergence") if isinstance(entry.get("convergence"), dict) else {}
        metrics_by_component[key] = copy.deepcopy(metrics)
        convergence_by_component[key] = copy.deepcopy(convergence)
    return {
        "summary": summary,
        "metrics_by_component": metrics_by_component,
        "convergence_by_component": convergence_by_component,
    }


def _mean_metric_dict(metrics_by_component: Dict[str, Dict[str, Any]], key: str) -> Dict[str, float]:
    values_by_key: Dict[str, List[float]] = {}
    for metrics in metrics_by_component.values():
        value = metrics.get(key) if isinstance(metrics, dict) else None
        if not isinstance(value, dict):
            continue
        for label, raw in value.items():
            if isinstance(raw, (int, float)):
                values_by_key.setdefault(str(label), []).append(float(raw))
    return {
        label: float(sum(values) / len(values))
        for label, values in values_by_key.items()
        if values
    }


def _summary_from_component_payload(
    metrics_by_component: Dict[str, Dict[str, Any]],
    convergence_by_component: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    correlations = [
        float(metrics.get("correlation"))
        for metrics in metrics_by_component.values()
        if isinstance(metrics, dict) and isinstance(metrics.get("correlation"), (int, float))
    ]
    min_abs_means = [
        float(info.get("min_target_class_abs_mean"))
        for info in convergence_by_component.values()
        if isinstance(info, dict) and isinstance(info.get("min_target_class_abs_mean"), (int, float))
    ]
    n_components = len(metrics_by_component)
    n_converged = sum(
        1
        for info in convergence_by_component.values()
        if isinstance(info, dict) and bool(info.get("converged"))
    )
    summary: Dict[str, Any] = {
        "n_components": n_components,
        "n_converged": n_converged,
        "convergence_rate": float(n_converged / n_components) if n_components else 0.0,
        "convergence_policy": "all",
        "converged": n_components > 0 and n_converged == n_components,
    }
    if correlations:
        sorted_corr = sorted(correlations)
        mid = len(sorted_corr) // 2
        summary.update({
            "correlation_mean": float(sum(correlations) / len(correlations)),
            "correlation_median": float(
                sorted_corr[mid]
                if len(sorted_corr) % 2
                else (sorted_corr[mid - 1] + sorted_corr[mid]) / 2.0
            ),
            "correlation_min": float(min(correlations)),
            "correlation_max": float(max(correlations)),
            "correlation_abs_mean": float(sum(abs(value) for value in correlations) / len(correlations)),
            "correlation_abs_min": float(min(abs(value) for value in correlations)),
            "correlation_abs_max": float(max(abs(value) for value in correlations)),
        })
    if min_abs_means:
        summary["min_target_class_abs_mean"] = float(min(min_abs_means))
    return summary


def build_stitched_component_training_run(
    selected_components: Dict[str, Dict[str, Any]],
    source_run_infos_by_component: Dict[str, Dict[str, Any]],
    *,
    base_run_info: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build a synthetic component-history view for a stitched Split GRADIEND.

    The returned run does not claim to be an observed aggregate training run.
    Each component curve is taken from the seed that supplied that component to
    the stitched model, then summary curves are recomputed over the selected
    components at common evaluation steps.
    """
    if not selected_components:
        return None
    components_by_id: Dict[str, Dict[int, Dict[str, Any]]] = {}
    component_meta: Dict[str, Dict[str, Any]] = {}
    label_mapping = None
    for component_id, selection in selected_components.items():
        component_key = str(component_id)
        run_info = source_run_infos_by_component.get(component_key)
        if not isinstance(run_info, dict):
            return None
        training_stats = run_info.get("training_stats") or {}
        if not isinstance(training_stats, dict):
            return None
        if label_mapping is None and isinstance(training_stats.get("label_value_to_class_name"), dict):
            label_mapping = training_stats["label_value_to_class_name"]
        step_payloads = _normalize_step_dict(training_stats.get("components"))
        component_steps: Dict[int, Dict[str, Any]] = {}
        for step, payload in step_payloads.items():
            if not isinstance(payload, dict):
                continue
            metrics_by_component = payload.get("metrics_by_component")
            if not isinstance(metrics_by_component, dict):
                continue
            metrics = metrics_by_component.get(component_key)
            if not isinstance(metrics, dict):
                continue
            convergence_by_component = payload.get("convergence_by_component")
            convergence = (
                convergence_by_component.get(component_key)
                if isinstance(convergence_by_component, dict)
                else None
            )
            metrics_copy = copy.deepcopy(metrics)
            metrics_copy["source_seed"] = selection.get("seed")
            metrics_copy["source_output_dir"] = selection.get("output_dir")
            convergence_copy = copy.deepcopy(convergence) if isinstance(convergence, dict) else {}
            convergence_copy.setdefault("component_index", metrics_copy.get("component_index"))
            convergence_copy.setdefault("component_id", metrics_copy.get("component_id", component_key))
            convergence_copy.setdefault("component_label", metrics_copy.get("component_label"))
            convergence_copy["source_seed"] = selection.get("seed")
            convergence_copy["source_output_dir"] = selection.get("output_dir")
            component_steps[step] = {
                "metrics": metrics_copy,
                "convergence": convergence_copy,
            }
            component_meta.setdefault(component_key, {
                "component_index": metrics_copy.get("component_index", selection.get("component_index")),
                "component_id": metrics_copy.get("component_id", component_key),
                "component_label": metrics_copy.get("component_label", selection.get("component_label")),
                "source_seed": selection.get("seed"),
                "source_output_dir": selection.get("output_dir"),
                "selected_global_step": selection.get("global_step")
                or selection.get("best_component_global_step"),
            })
        if not component_steps:
            return None
        components_by_id[component_key] = component_steps

    common_steps = set.intersection(*(set(steps) for steps in components_by_id.values()))
    if not common_steps:
        return None

    synthetic_stats: Dict[str, Any] = {
        "scores": {},
        "mean_by_class": {},
        "mean_by_feature_class": {},
        "components": {},
        "component_summary": {},
        "synthetic_component_stitching": True,
        "component_stitching_note": (
            "Synthetic selected-component history: component curves are taken from the seed "
            "that supplied each stitched component; aggregate curves are recomputed over "
            "selected components at common evaluation steps."
        ),
        "selected_component_sources": component_meta,
    }
    if label_mapping is None and isinstance((base_run_info or {}).get("training_stats"), dict):
        base_ts = (base_run_info or {}).get("training_stats") or {}
        if isinstance(base_ts.get("label_value_to_class_name"), dict):
            label_mapping = base_ts["label_value_to_class_name"]
    if label_mapping is not None:
        synthetic_stats["label_value_to_class_name"] = copy.deepcopy(label_mapping)

    for step in sorted(common_steps):
        metrics_by_component: Dict[str, Dict[str, Any]] = {}
        convergence_by_component: Dict[str, Dict[str, Any]] = {}
        for component_id in selected_components:
            component_key = str(component_id)
            row = components_by_id[component_key][step]
            metrics_by_component[component_key] = copy.deepcopy(row["metrics"])
            convergence_by_component[component_key] = copy.deepcopy(row["convergence"])
        summary = _summary_from_component_payload(metrics_by_component, convergence_by_component)
        synthetic_stats["components"][step] = {
            "metrics_by_component": metrics_by_component,
            "convergence_by_component": convergence_by_component,
            "summary": summary,
        }
        synthetic_stats["component_summary"][step] = summary
        if isinstance(summary.get("correlation_mean"), (int, float)):
            synthetic_stats["scores"][step] = float(summary["correlation_mean"])
        mean_by_class = _mean_metric_dict(metrics_by_component, "mean_by_class")
        if mean_by_class:
            synthetic_stats["mean_by_class"][step] = mean_by_class
        mean_by_feature_class = _mean_metric_dict(metrics_by_component, "mean_by_feature_class")
        if mean_by_feature_class:
            synthetic_stats["mean_by_feature_class"][step] = mean_by_feature_class

    selected_metrics_by_component: Dict[str, Dict[str, Any]] = {}
    selected_convergence_by_component: Dict[str, Dict[str, Any]] = {}
    component_best_steps: Dict[str, Any] = {}
    for component_id, selection in selected_components.items():
        component_key = str(component_id)
        selected_step = selection.get("global_step") or selection.get("best_component_global_step")
        if selected_step is not None:
            component_best_steps[component_key] = selected_step
        metrics = selection.get("metrics")
        convergence = selection.get("convergence")
        if not isinstance(metrics, dict) and selected_step is not None:
            try:
                selected_step_int = int(selected_step)
            except (TypeError, ValueError):
                selected_step_int = None
            if selected_step_int is not None and component_key in components_by_id:
                row = components_by_id[component_key].get(selected_step_int)
                if isinstance(row, dict):
                    metrics = row.get("metrics")
                    convergence = row.get("convergence")
        if isinstance(metrics, dict):
            selected_metrics_by_component[component_key] = copy.deepcopy(metrics)
        if isinstance(convergence, dict):
            selected_convergence_by_component[component_key] = copy.deepcopy(convergence)
    selected_summary = (
        _summary_from_component_payload(selected_metrics_by_component, selected_convergence_by_component)
        if selected_metrics_by_component
        else None
    )
    if isinstance(selected_summary, dict):
        selected_summary["component_best_steps"] = copy.deepcopy(component_best_steps)
        synthetic_stats["selected_component_summary"] = copy.deepcopy(selected_summary)

    synthetic_stats = {
        key: value
        for key, value in synthetic_stats.items()
        if value or key in {"synthetic_component_stitching", "component_stitching_note"}
    }
    scores = synthetic_stats.get("scores") or {}
    best_step = None
    best_corr = None
    if isinstance(selected_summary, dict):
        steps = {
            value
            for value in component_best_steps.values()
            if isinstance(value, (int, float))
        }
        best_step = next(iter(steps)) if len(steps) == 1 else None
        best_corr = selected_summary.get("correlation_mean")
    elif scores:
        best_step = max(scores, key=lambda step: abs(float(scores[step])))
        best_corr = float(scores[best_step])
    return {
        "training_stats": synthetic_stats,
        "best_score_checkpoint": {
            "global_step": best_step,
            "correlation": best_corr,
            "component_best_steps": copy.deepcopy(component_best_steps),
        },
        "convergence_info": {
            "convergence_unit": "stitched_component_history",
            "synthetic": True,
            "n_components": len(selected_components),
            "common_steps": sorted(common_steps),
            "selected_component_summary": copy.deepcopy(selected_summary),
        },
    }


def component_run_from_training_stats(
    stats: Optional[Dict[str, Any]],
    *,
    threshold: Optional[float],
    mean_threshold: Optional[float],
    selection_metric: str = "correlation",
) -> Optional[Dict[str, Any]]:
    """Extract per-component best-step convergence details from ``training.json`` data.

    Split GRADIEND/ACTIEND selection is component-local: component ``i`` is
    evaluated at the best step for component ``i``.  The seed's aggregate
    ``best_score_checkpoint`` is only a fallback for non-component models and
    must not decide component convergence.
    """
    if not isinstance(stats, dict):
        return None
    training_stats = stats.get("training_stats") or {}
    step_payloads = _normalize_step_dict(training_stats.get("components"))
    if not step_payloads:
        return None

    best_candidates: Dict[str, Dict[str, Any]] = {}
    for step in sorted(step_payloads):
        payload = step_payloads[step]
        if not isinstance(payload, dict):
            continue
        metrics_by_component = payload.get("metrics_by_component")
        if not isinstance(metrics_by_component, dict):
            continue
        for component_id, metrics in metrics_by_component.items():
            if not isinstance(metrics, dict):
                continue
            component_key = str(component_id)
            candidate = _component_candidate_from_metrics(
                component_id=component_key,
                metrics=metrics,
                step=step,
                threshold=threshold,
                mean_threshold=mean_threshold,
                selection_metric=selection_metric,
            )
            current = best_candidates.get(component_key)
            if current is None or _candidate_rank(candidate, selection_metric) > _candidate_rank(current, selection_metric):
                best_candidates[component_key] = candidate

    if not best_candidates:
        return None
    metrics_by_component = {
        component_id: copy.deepcopy(candidate["metrics"])
        for component_id, candidate in best_candidates.items()
    }
    component_convergence = {
        component_id: copy.deepcopy(candidate["convergence"])
        for component_id, candidate in best_candidates.items()
    }
    summary = _summary_from_component_payload(metrics_by_component, component_convergence)
    component_best_steps = {
        component_id: candidate.get("global_step")
        for component_id, candidate in best_candidates.items()
    }
    summary["component_best_steps"] = component_best_steps
    summary["convergence_unit"] = "component_best_step"
    n_components = len(component_convergence)
    if n_components == 0:
        return None
    return {
        "summary": summary,
        "convergence_by_component": component_convergence,
        "metrics_by_component": metrics_by_component,
    }


def summarize_component_seed_runs(
    runs: Sequence[Dict[str, Any]],
    *,
    min_convergent_seeds: Optional[int],
    selection_metric: str = "correlation",
) -> Optional[Dict[str, Any]]:
    """Summarize which components have enough convergent seeds and select source runs."""
    component_ids: Dict[str, Dict[str, Any]] = {}
    candidates_by_component: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        component_convergence = run.get("component_convergence")
        if not isinstance(component_convergence, dict):
            continue
        for component_id, info in component_convergence.items():
            if not isinstance(info, dict):
                continue
            component_key = str(component_id)
            component_ids.setdefault(component_key, {
                "component_index": info.get("component_index"),
                "component_id": info.get("component_id", component_key),
                "component_label": info.get("component_label"),
            })
            if bool(info.get("converged")):
                component_metrics = run.get("component_metrics")
                metrics = (
                    component_metrics.get(component_key)
                    if isinstance(component_metrics, dict)
                    else None
                )
                candidates_by_component.setdefault(component_key, []).append({
                    "seed": run.get("seed"),
                    "output_dir": run.get("output_dir"),
                    "correlation": info.get("correlation"),
                    "encoding_e": (
                        metrics.get("encoding_e") if isinstance(metrics, dict) else None
                    ),
                    "min_target_class_abs_mean": info.get("min_target_class_abs_mean"),
                    "global_step": info.get("best_component_global_step") or info.get("global_step"),
                    "best_component_global_step": info.get("best_component_global_step") or info.get("global_step"),
                    "component_index": info.get("component_index"),
                    "component_id": info.get("component_id", component_key),
                    "component_label": info.get("component_label"),
                    "metrics": copy.deepcopy(metrics) if isinstance(metrics, dict) else None,
                    "convergence": copy.deepcopy(info),
                })

    if not component_ids:
        return None

    min_required = int(min_convergent_seeds or 0)
    counts_by_component: Dict[str, int] = {}
    selected_components: Dict[str, Dict[str, Any]] = {}
    missing_components: List[str] = []
    satisfied = 0
    for component_id, meta in sorted(component_ids.items(), key=lambda item: _component_sort_key(item[0], item[1])):
        candidates = candidates_by_component.get(component_id, [])
        counts_by_component[component_id] = len(candidates)
        if len(candidates) >= min_required:
            satisfied += 1
        else:
            missing_components.append(component_id)
        if candidates:
            selected = max(
                candidates,
                key=lambda item: (
                    _selection_score(item, selection_metric)
                    if _selection_score(item, selection_metric) is not None
                    else float("-inf"),
                    float(item["min_target_class_abs_mean"])
                    if isinstance(item.get("min_target_class_abs_mean"), (int, float))
                    else float("-inf"),
                    int(item["global_step"]) if isinstance(item.get("global_step"), int) else -1,
                ),
            )
            selected_components[component_id] = selected

    counts = list(counts_by_component.values())
    n_components = len(component_ids)
    min_count = min(counts) if counts else 0
    converged = n_components > 0 and min_count >= min_required
    return {
        "convergence_unit": "component_seed",
        "n_components": n_components,
        "n_satisfied_components": satisfied,
        "min_convergent_seeds": min_convergent_seeds,
        "min_convergent_runs_per_component": min_count,
        "component_convergent_count_by_id": counts_by_component,
        "missing_component_ids": missing_components,
        "converged": converged,
        "selected_components": selected_components if converged else {},
        "selection_metric": selection_metric,
    }


def _component_signature(gradiend: Any) -> Dict[str, Any]:
    return {
        "input_dim": int(getattr(gradiend, "input_dim")),
        "latent_dim": int(getattr(gradiend, "latent_dim")),
        "activation": getattr(gradiend, "activation", None),
        "activation_decoder": getattr(gradiend, "activation_decoder", None),
        "bias_encoder": bool(getattr(gradiend, "bias_encoder", True)),
        "bias_decoder": bool(getattr(gradiend, "bias_decoder", False)),
        "components": [
            {
                "id": str(component.id),
                "start": int(component.start),
                "end": int(component.end),
            }
            for component in getattr(gradiend, "component_slices", ())
        ],
    }


def stitch_gradiend_components(
    target_model: Any,
    component_source_models: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Copy selected component tensors from source models into ``target_model``.

    The returned rows document the copy operation. The saved model remains a
    normal GRADIEND checkpoint; component sourcing is only metadata.
    """
    target_gradiend = target_model.gradiend
    target_gradiend._require_built()
    target_components = {
        str(component.id): component
        for component in getattr(target_gradiend, "component_slices", ())
    }
    if not bool(getattr(target_gradiend, "has_component_split", False)):
        raise ValueError("Component stitching requires an explicit GRADIEND component split")
    if bool(getattr(target_gradiend, "bias_encoder", True)) and len(target_components) > 1:
        raise ValueError(
            "Component stitching with multiple components requires bias_encoder=False "
            "because the encoder bias is shared across components."
        )

    target_signature = _component_signature(target_gradiend)
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for component_id, source_model in component_source_models.items():
            component_key = str(component_id)
            if component_key not in target_components:
                raise KeyError(f"Unknown target GRADIEND component id: {component_key!r}")
            source_gradiend = source_model.gradiend
            source_gradiend._require_built()
            if _component_signature(source_gradiend) != target_signature:
                raise ValueError(
                    "Cannot stitch components from seeds with different GRADIEND architecture or component split"
                )
            component = target_components[component_key]
            target_encoder = target_gradiend.encoder[0].linear
            source_encoder = source_gradiend.encoder[0].linear
            target_decoder = target_gradiend.decoder[0].linear
            source_decoder = source_gradiend.decoder[0].linear
            target_encoder.weight[:, component.start:component.end].copy_(
                source_encoder.weight[:, component.start:component.end].to(
                    dtype=target_encoder.weight.dtype,
                    device=target_encoder.weight.device,
                )
            )
            target_decoder.weight[component.start:component.end, :].copy_(
                source_decoder.weight[component.start:component.end, :].to(
                    dtype=target_decoder.weight.dtype,
                    device=target_decoder.weight.device,
                )
            )
            if target_decoder.bias is not None:
                if source_decoder.bias is None:
                    raise ValueError("Cannot stitch decoder bias from a source without decoder bias")
                target_decoder.bias[component.start:component.end].copy_(
                    source_decoder.bias[component.start:component.end].to(
                        dtype=target_decoder.bias.dtype,
                        device=target_decoder.bias.device,
                    )
                )
            if target_encoder.bias is not None:
                if source_encoder.bias is None:
                    raise ValueError("Cannot stitch encoder bias from a source without encoder bias")
                target_encoder.bias.copy_(
                    source_encoder.bias.to(dtype=target_encoder.bias.dtype, device=target_encoder.bias.device)
                )
            rows.append({
                "component_id": component_key,
                "component_index": list(target_components).index(component_key),
                "start": int(component.start),
                "end": int(component.end),
            })
    target_model.gradiend.kwargs.setdefault("component_seed_stitching", []).extend(copy.deepcopy(rows))
    return rows


def _copy_tensor_slice(target: torch.Tensor, source: Any, *, name: str) -> None:
    if not isinstance(source, torch.Tensor):
        raise ValueError(f"Component-best state is missing tensor {name!r}")
    if tuple(target.shape) != tuple(source.shape):
        raise ValueError(
            f"Component-best tensor {name!r} has shape {tuple(source.shape)}, "
            f"expected {tuple(target.shape)}"
        )
    target.copy_(source.to(dtype=target.dtype, device=target.device))


def stitch_gradiend_component_states(
    target_model: Any,
    component_source_states: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Copy selected component tensor states into ``target_model``.

    Unlike ``stitch_gradiend_components``, this does not load the source seed's
    aggregate best checkpoint.  Each source entry is a tensor slice captured at
    the component's own best evaluation step.
    """
    target_gradiend = target_model.gradiend
    target_gradiend._require_built()
    target_components = {
        str(component.id): component
        for component in getattr(target_gradiend, "component_slices", ())
    }
    if not bool(getattr(target_gradiend, "has_component_split", False)):
        raise ValueError("Component stitching requires an explicit GRADIEND component split")
    if bool(getattr(target_gradiend, "bias_encoder", True)) and len(target_components) > 1:
        raise ValueError(
            "Component stitching with multiple components requires bias_encoder=False "
            "because the encoder bias is shared across components."
        )

    target_signature = _component_signature(target_gradiend)
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for component_id, entry in component_source_states.items():
            component_key = str(component_id)
            if component_key not in target_components:
                raise KeyError(f"Unknown target GRADIEND component id: {component_key!r}")
            if not isinstance(entry, dict):
                raise ValueError(f"Component {component_key!r} source state must be a dictionary")
            if entry.get("signature") != target_signature:
                raise ValueError(
                    "Cannot stitch component-best states from a different GRADIEND architecture or component split"
                )
            state = entry.get("state")
            if not isinstance(state, dict):
                raise ValueError(f"Component {component_key!r} source state has no tensor payload")
            component = target_components[component_key]
            start = int(component.start)
            end = int(component.end)
            if int(state.get("start", start)) != start or int(state.get("end", end)) != end:
                raise ValueError(
                    f"Component-best state for {component_key!r} targets slice "
                    f"{state.get('start')}:{state.get('end')}, expected {start}:{end}"
                )
            target_encoder = target_gradiend.encoder[0].linear
            target_decoder = target_gradiend.decoder[0].linear
            _copy_tensor_slice(
                target_encoder.weight[:, start:end],
                state.get("encoder_weight"),
                name=f"{component_key}.encoder_weight",
            )
            _copy_tensor_slice(
                target_decoder.weight[start:end, :],
                state.get("decoder_weight"),
                name=f"{component_key}.decoder_weight",
            )
            if target_decoder.bias is not None:
                _copy_tensor_slice(
                    target_decoder.bias[start:end],
                    state.get("decoder_bias"),
                    name=f"{component_key}.decoder_bias",
                )
            if target_encoder.bias is not None:
                _copy_tensor_slice(
                    target_encoder.bias,
                    state.get("encoder_bias"),
                    name=f"{component_key}.encoder_bias",
                )
            rows.append({
                "component_id": component_key,
                "component_index": list(target_components).index(component_key),
                "start": start,
                "end": end,
                "source_seed": entry.get("seed"),
                "source_output_dir": entry.get("output_dir"),
                "source_global_step": entry.get("global_step"),
                "source_kind": "component_best_state",
            })
    target_model.gradiend.kwargs.setdefault("component_seed_stitching", []).extend(copy.deepcopy(rows))
    return rows


__all__ = [
    "apply_saved_component_best_states",
    "build_stitched_component_training_run",
    "component_run_from_training_stats",
    "extract_gradiend_component_state",
    "finalize_component_best_states",
    "load_component_best_states",
    "merge_rank_from_summary",
    "model_has_component_split",
    "payload_from_component_best_states",
    "restore_gradiend_trainable_state",
    "save_component_best_states",
    "save_merged_component_best_checkpoint",
    "snapshot_gradiend_trainable_state",
    "stitch_gradiend_components",
    "stitch_gradiend_component_states",
    "summarize_component_seed_runs",
    "summary_from_component_best_states",
    "update_component_best_states",
]
