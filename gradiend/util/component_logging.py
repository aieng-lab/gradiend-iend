"""Compact log formatting for component-split GRADIEND convergence."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _component_sort_key(row: Dict[str, Any]) -> Tuple[int, str]:
    index = _as_int(row.get("component_index"))
    return index if index is not None else 10**9, str(row.get("component_id", ""))


def _component_rows(components: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(components, dict):
        return []
    metrics_by_component = components.get("metrics_by_component")
    convergence_by_component = components.get("convergence_by_component")
    if not isinstance(metrics_by_component, dict):
        metrics_by_component = {}
    if not isinstance(convergence_by_component, dict):
        convergence_by_component = {}

    component_ids = list(metrics_by_component.keys())
    for component_id in convergence_by_component:
        if component_id not in metrics_by_component:
            component_ids.append(component_id)

    rows: List[Dict[str, Any]] = []
    for component_id in component_ids:
        metrics = metrics_by_component.get(component_id)
        convergence = convergence_by_component.get(component_id)
        metrics = metrics if isinstance(metrics, dict) else {}
        convergence = convergence if isinstance(convergence, dict) else {}
        converged = convergence.get("converged")
        if not isinstance(converged, bool):
            converged = metrics.get("converged")
        rows.append({
            "component_id": str(component_id),
            "component_index": convergence.get("component_index", metrics.get("component_index")),
            "converged": converged if isinstance(converged, bool) else None,
        })
    return sorted(rows, key=_component_sort_key)


def best_step_component_payload(training_stats: Dict[str, Any], best_score_checkpoint: Dict[str, Any]) -> Optional[dict]:
    """Return the component metrics payload recorded at the best checkpoint step."""
    best_step = best_score_checkpoint.get("global_step") if isinstance(best_score_checkpoint, dict) else None
    if best_step is None or not isinstance(training_stats, dict):
        return None
    components = training_stats.get("components")
    if not isinstance(components, dict):
        return None
    return components.get(best_step) or components.get(str(best_step))


def component_summary_from_payload(
    components: Optional[Dict[str, Any]],
    summary: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if isinstance(summary, dict) and summary.get("n_components"):
        return summary
    if isinstance(components, dict) and isinstance(components.get("summary"), dict):
        payload_summary = components["summary"]
        if payload_summary.get("n_components"):
            return payload_summary
    return None


def _format_component_id_list(
    component_ids: List[str],
    *,
    label: str,
    max_ids: int,
) -> str:
    if not component_ids:
        return ""
    shown = ", ".join(component_ids[:max_ids])
    if len(component_ids) <= max_ids:
        return f"{label} [{shown}]"
    return f"{label} {len(component_ids)} components (first {max_ids}: [{shown}])"


def format_component_convergence_fragment(
    components: Optional[Dict[str, Any]] = None,
    *,
    summary: Optional[Dict[str, Any]] = None,
    max_ids: int = 3,
    show_blockers: bool = False,
) -> str:
    """
    Return a compact component convergence fragment for logs.

    Middle cases stay terse. Near the edges, component ids are useful: show the
    first converged ids when fewer than four converged, or the remaining ids
    when fewer than four are still non-convergent.
    """
    summary = component_summary_from_payload(components, summary)
    rows = _component_rows(components)
    if summary is None:
        if not rows:
            return ""
        n_components = len(rows)
        n_converged = sum(1 for row in rows if row.get("converged") is True)
    else:
        n_components = _as_int(summary.get("n_components")) or len(rows)
        n_converged = _as_int(summary.get("n_converged"))
        if n_converged is None:
            n_converged = sum(1 for row in rows if row.get("converged") is True)
    if n_components <= 0:
        return ""

    base = f"components: {n_converged}/{n_components} converged"
    if not rows:
        return base

    if n_components <= max_ids:
        status = []
        for row in rows:
            if row.get("converged") is True:
                state = "ok"
            elif row.get("converged") is False:
                state = "pending"
            else:
                state = "unknown"
            status.append(f"{row['component_id']}: {state}")
        return f"{base} [{'; '.join(status)}]"

    converged_ids = [row["component_id"] for row in rows if row.get("converged") is True]
    remaining_ids = [row["component_id"] for row in rows if row.get("converged") is not True]
    n_remaining = max(0, n_components - n_converged)
    if show_blockers and n_remaining > 0 and remaining_ids:
        return f"{base}, {_format_component_id_list(remaining_ids, label='remaining', max_ids=max_ids)}"
    if 0 < n_converged <= max_ids and converged_ids:
        return f"{base} [{', '.join(converged_ids[:max_ids])}]"
    if 0 < n_remaining <= max_ids and remaining_ids:
        return f"{base}, remaining [{', '.join(remaining_ids[:max_ids])}]"
    return base


def format_component_seed_summary_fragment(
    summary: Optional[Dict[str, Any]],
    *,
    max_ids: int = 3,
    show_blockers: bool = False,
) -> str:
    """Return a compact multi-seed component convergence fragment."""
    if not isinstance(summary, dict):
        return ""
    n_components = _as_int(summary.get("n_components"))
    n_satisfied = _as_int(summary.get("n_satisfied_components"))
    if n_components is None or n_satisfied is None or n_components <= 0:
        return ""
    base = f"components: {n_satisfied}/{n_components} have required convergent seed(s)"
    counts = summary.get("component_convergent_count_by_id")
    if not isinstance(counts, dict):
        counts = {}
    min_required = _as_int(summary.get("min_convergent_seeds")) or 1
    missing = [str(value) for value in summary.get("missing_component_ids") or []]
    if n_components <= max_ids and counts:
        statuses = [f"{component_id}: {count}" for component_id, count in counts.items()]
        return f"{base} [{'; '.join(statuses)}]"
    n_remaining = n_components - n_satisfied
    remaining = missing or [
        str(component_id)
        for component_id, count in counts.items()
        if not isinstance(count, int) or count < min_required
    ]
    if show_blockers and n_remaining > 0 and remaining:
        return f"{base}, {_format_component_id_list(remaining, label='remaining', max_ids=max_ids)}"
    if 0 < n_satisfied <= max_ids and counts:
        satisfied = [
            str(component_id)
            for component_id, count in counts.items()
            if isinstance(count, int) and count >= min_required
        ]
        if satisfied:
            return f"{base} [{', '.join(satisfied[:max_ids])}]"
    if 0 < n_remaining <= max_ids:
        if remaining:
            return f"{base}, remaining [{', '.join(remaining[:max_ids])}]"
    return base


def format_component_stitching_fragment(stitching: Optional[Dict[str, Any]]) -> str:
    """Return a short human-readable component stitching status."""
    if not isinstance(stitching, dict):
        return ""
    applied = bool(stitching.get("applied"))
    status = str(stitching.get("status") or ("applied" if applied else "not_applied"))
    n_components = _as_int(stitching.get("n_components"))
    reason = stitching.get("reason")
    if applied or status == "applied":
        suffix = f" ({n_components} component(s))" if n_components is not None else ""
        return f"component stitching applied{suffix}"
    if status == "skipped":
        suffix = f": {reason}" if reason else ""
        return f"component stitching skipped{suffix}"
    if status == "failed":
        suffix = f": {reason}" if reason else ""
        return f"component stitching failed{suffix}"
    suffix = f": {reason}" if reason else ""
    return f"component stitching {status}{suffix}"


def strip_component_prefix(fragment: str) -> str:
    """Drop the leading ``components:`` label for embedding in a sentence."""
    prefix = "components: "
    return fragment[len(prefix):] if fragment.startswith(prefix) else fragment


__all__ = [
    "best_step_component_payload",
    "component_summary_from_payload",
    "format_component_convergence_fragment",
    "format_component_seed_summary_fragment",
    "format_component_stitching_fragment",
    "strip_component_prefix",
]
