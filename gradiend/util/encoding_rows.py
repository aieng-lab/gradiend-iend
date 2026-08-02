"""Dependency-light helpers for encoding signal datasets into row dicts."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from gradiend.util.logging import get_logger
from gradiend.util.tqdm_utils import gradiend_tqdm

logger = get_logger(__name__)


def _component_label(component_id: Any) -> str:
    text = str(component_id)
    if text.startswith("activation:"):
        return text[len("activation:"):]
    return text


def visible_component_index(model_with_gradiend: Any) -> List[Dict[str, Any]]:
    """Return public component labels for explicit split models, otherwise an empty list."""
    gradiend = getattr(model_with_gradiend, "gradiend", model_with_gradiend)
    component_slices = getattr(gradiend, "component_slices", ())
    if not isinstance(component_slices, (list, tuple)) or not component_slices:
        return []
    components = list(component_slices)
    return [
        {
            "component_index": index,
            "component_id": str(component.id),
            "component_label": _component_label(component.id),
        }
        for index, component in enumerate(components)
    ]


def _encoded_cell(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "flatten"):
        values = value.flatten().tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        return float(value)
    if len(values) == 1:
        return float(values[0])
    return json.dumps([float(v) for v in values])


def encode_signal_with_visible_components(
    model_with_gradiend: Any,
    signal_tensor: Any,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Encode one already-extracted GRADIEND signal once and derive visible component rows.

    For ``GradiendSplit.none()`` this returns the ordinary scalar/list encoding and no
    component rows. Explicit split modes return the mean component encoding as the
    aggregate value plus one component value per visible component.
    """
    gradiend = getattr(model_with_gradiend, "gradiend", None)
    if gradiend is None:
        return model_with_gradiend.encode(signal_tensor, return_float=True), []
    signal_tensor = signal_tensor.to(gradiend.device_encoder, dtype=gradiend.torch_dtype)
    component_index = visible_component_index(model_with_gradiend)
    if not component_index:
        if hasattr(model_with_gradiend, "encode") and callable(model_with_gradiend.encode):
            return model_with_gradiend.encode(signal_tensor, return_float=True), []
        return _encoded_cell(gradiend.encoder(signal_tensor)), []

    encoded_components = gradiend._encode_components(signal_tensor)
    aggregate = encoded_components.mean(dim=-2)
    component_values: List[Dict[str, Any]] = []
    for item, encoded in zip(component_index, encoded_components):
        component_values.append({**item, "encoded": _encoded_cell(encoded)})
    return _encoded_cell(aggregate), component_values


def gradient_entry_to_encoder_row(
    entry: Dict[str, Any],
    *,
    encoded: float,
    input_type: Optional[str] = "factual",
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one encoder-analysis row from a signal-dataset entry."""
    if input_type == "alternative":
        source_id = entry.get("alternative_id")
        source_token = entry.get("alternative_token")
    elif input_type == "diff":
        source_id = entry.get("feature_class_id")
        source_token = None
    else:
        source_id = entry.get("factual_id")
        source_token = entry.get("factual_token")
    row: Dict[str, Any] = {
        "encoded": encoded,
        "label": float(entry.get("label", 0.0)),
        "source_id": source_id,
        "target_id": entry.get("alternative_id"),
        "factual_id": entry.get("factual_id"),
        "counterfactual_id": entry.get("alternative_id"),
        "transition_id": entry.get("feature_class_id"),
        "feature_class_id": entry.get("feature_class_id"),
        "input_type": input_type,
    }
    if source_token is not None:
        row["source_token"] = source_token
    for token_key in ("factual_token", "alternative_token"):
        if entry.get(token_key) is not None:
            row[token_key] = entry[token_key]
    for text_key in ("text", "template", "input_text", "display_text"):
        if entry.get(text_key) is not None:
            row[text_key] = entry[text_key]
    if entry.get("template") is not None and row.get("masked") is None:
        row["masked"] = entry["template"]
    if entry.get("data_split") is not None:
        row["data_split"] = entry["data_split"]
    if entry.get("neutral_variant") is not None:
        row["neutral_variant"] = entry["neutral_variant"]
    if entry.get("transition_type") is not None:
        row["transition_type"] = entry["transition_type"]
    if overrides:
        row.update(overrides)
    return row


def encode_dataset_to_rows(
    model_with_gradiend: Any,
    dataset: Any,
    row_extractor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    *,
    return_component_rows: bool = False,
) -> Any:
    """
    Encode a SignalTrainingDatasetBase and return per-row dicts for building DataFrames.

    Each row has: encoded, label, source_id, target_id, plus optional fields
    provided by row_extractor (modality-specific, e.g. text).
    Used by EncoderEvaluator and by callers (e.g. _analyze_encoder) when training_rows
    are not available from cache.
    """
    rows: List[Dict[str, Any]] = []
    component_rows: List[Dict[str, Any]] = []
    try:
        total = len(dataset)
    except (TypeError, AttributeError):
        total = None
    for entry in gradiend_tqdm(
        dataset,
        desc="Encoding",
        total=total,
        leave=False,
        ncols=80,
        position=0,
    ):
        signal_tensor = entry["source"]
        encoded_val, component_values = encode_signal_with_visible_components(model_with_gradiend, signal_tensor)
        input_type = getattr(dataset, "source", None)
        row = gradient_entry_to_encoder_row(
            entry,
            encoded=encoded_val,
            input_type=input_type,
        )
        row["eval_group"] = entry.get("eval_group")
        if row_extractor is not None:
            try:
                extra = row_extractor(entry)
                if isinstance(extra, dict) and extra:
                    row.update(extra)
            except Exception as e:
                logger.warning("Row extractor failed: %s", e)
        rows.append(row)
        for component_value in component_values:
            component_row = dict(row)
            component_row.update(component_value)
            component_rows.append(component_row)
    if return_component_rows:
        return rows, component_rows
    return rows


__all__ = [
    "encode_dataset_to_rows",
    "encode_signal_with_visible_components",
    "gradient_entry_to_encoder_row",
    "visible_component_index",
]
