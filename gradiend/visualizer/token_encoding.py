"""Text token highlighting for GRADIEND / ACTIEND encoded signals."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from gradiend.util.encoding_rows import visible_component_index
from gradiend.visualizer.color_norm import (
    ColorCenter,
    ColorRange,
    diverging_rgb,
    resolve_encoding_color_norm,
)


@dataclass(frozen=True)
class TokenEncodingRow:
    """One token and its encoded GRADIEND response."""

    token: str
    encoded: float
    index: int
    token_id: Optional[int] = None
    start: Optional[int] = None
    end: Optional[int] = None
    component_id: Optional[str] = None
    component_label: Optional[str] = None
    masked_text: Optional[str] = None
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "encoded": self.encoded,
            "index": self.index,
            "token_id": self.token_id,
            "start": self.start,
            "end": self.end,
            "component_id": self.component_id,
            "component_label": self.component_label,
            "masked_text": self.masked_text,
            "label": self.label,
        }


def _to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "flatten"):
        values = value.flatten().tolist()
        if len(values) != 1:
            raise ValueError("Token encoding currently requires scalar latent encodings")
        return float(values[0])
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Token encoding currently requires scalar latent encodings")
        return float(value[0])
    return float(value)


def _component_label(component_id: Optional[str]) -> Optional[str]:
    if component_id is None:
        return None
    text = str(component_id)
    return text[len("activation:"):] if text.startswith("activation:") else text


def _resolve_component(model_with_gradiend: Any, component: Any) -> tuple[Any, Optional[str], Optional[str]]:
    if component is None or (isinstance(component, str) and component in {"aggregate", "mean", "full"}):
        return None, None, None
    gradiend = getattr(model_with_gradiend, "gradiend", model_with_gradiend)
    if not getattr(gradiend, "has_component_split", False):
        raise ValueError("component selection requires a component-split GRADIEND model")
    component_index = visible_component_index(model_with_gradiend)
    if isinstance(component, str):
        matches = [
            item for item in component_index
            if item["component_id"] == component or item["component_label"] == component
        ]
        if not matches:
            known = ", ".join(item["component_label"] for item in component_index)
            raise KeyError(f"Unknown GRADIEND component {component!r}. Known components: {known}")
        key = matches[0]["component_id"]
    elif isinstance(component, int):
        key = int(component)
    else:
        raise TypeError(f"component must be str, int, or None, got {type(component).__name__}")
    resolved = gradiend._component_by_key(key)
    return key, str(resolved.id), _component_label(str(resolved.id))


def list_token_encoding_components(model_with_gradiend: Any) -> List[Dict[str, Any]]:
    """Return selectable token-highlighting components for a model."""
    return [
        {"component_index": None, "component_id": "aggregate", "component_label": "aggregate"},
        *visible_component_index(model_with_gradiend),
    ]


def _token_rows(tokenizer: Any, text: str, *, skip_special_tokens: bool, max_length: Optional[int]) -> List[Dict[str, Any]]:
    kwargs: Dict[str, Any] = {
        "return_tensors": "pt",
        "truncation": True,
        "padding": False,
    }
    if max_length is not None:
        kwargs["max_length"] = max_length
    try:
        encoded = tokenizer(text, return_offsets_mapping=True, **kwargs)
        offsets = encoded.get("offset_mapping")
    except TypeError:
        encoded = tokenizer(text, **kwargs)
        offsets = None
    input_ids = encoded["input_ids"][0]
    ids = input_ids.detach().cpu().tolist() if hasattr(input_ids, "detach") else list(input_ids)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    try:
        tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
    except Exception:
        tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in ids]

    spans: Sequence[Any]
    if offsets is not None:
        spans = offsets[0].detach().cpu().tolist() if hasattr(offsets, "detach") else offsets[0]
    else:
        spans = []

    rows: List[Dict[str, Any]] = []
    search_from = 0
    for index, (token_id, token) in enumerate(zip(ids, tokens)):
        if skip_special_tokens and token_id in special_ids:
            continue
        start = end = None
        if spans:
            start, end = int(spans[index][0]), int(spans[index][1])
            if start == end and skip_special_tokens:
                continue
        else:
            clean = str(token).replace("##", "").replace("\u0120", "").replace("\u2581", "")
            found = text.find(clean, search_from)
            if clean and found >= 0:
                start, end = found, found + len(clean)
                search_from = end
        rows.append({"index": index, "token_id": int(token_id), "token": str(token), "start": start, "end": end})
    return rows


def _mask_text(text: str, token: Dict[str, Any], mask_token: str) -> str:
    start = token.get("start")
    end = token.get("end")
    if isinstance(start, int) and isinstance(end, int) and end > start:
        return f"{text[:start]}{mask_token}{text[end:]}"
    return text


def _encode_signal(model_with_gradiend: Any, signal: Any, *, component_key: Any = None) -> float:
    gradiend = getattr(model_with_gradiend, "gradiend", None)
    if gradiend is not None and hasattr(signal, "to"):
        signal = signal.to(gradiend.device_encoder, dtype=gradiend.torch_dtype)
    if component_key is None:
        return _to_float(model_with_gradiend.encode(signal, return_float=False))
    return _to_float(gradiend._component_encoders[component_key](signal))


def _first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for item in value.values():
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"Hook output did not contain a tensor, got {type(value).__name__}")


def _model_device(model_with_gradiend: Any) -> torch.device:
    base = getattr(model_with_gradiend, "base_model", None)
    if base is not None:
        try:
            return next(base.parameters()).device
        except StopIteration:
            pass
    gradiend = getattr(model_with_gradiend, "gradiend", None)
    return getattr(gradiend, "device_encoder", torch.device("cpu"))


def _capture_site_activations(
    model_with_gradiend: Any,
    text: str,
    *,
    max_length: Optional[int],
) -> List[tuple[str, torch.Tensor]]:
    """Run one forward pass and return ``[(site_name, activation), ...]``."""
    from gradiend.signal_space import resolve_activation_modules

    tokenizer = model_with_gradiend.tokenizer
    sites = list(getattr(model_with_gradiend, "activation_site_modules", None) or ())
    if not sites:
        raise ValueError("ACTIEND token encoding requires activation_site_modules on the model")
    base_model = getattr(model_with_gradiend, "base_model", None)
    if base_model is None:
        raise ValueError("ACTIEND token encoding requires model_with_gradiend.base_model")

    module_items = resolve_activation_modules(base_model, sites)
    kwargs: Dict[str, Any] = {"return_tensors": "pt", "truncation": True, "padding": False}
    if max_length is not None:
        kwargs["max_length"] = max_length
    encoded = tokenizer(str(text), **kwargs)
    device = _model_device(model_with_gradiend)
    model_inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in encoded.items()
        if key != "offset_mapping"
    }

    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module: Any, _args: Any, output: Any) -> None:
            captured[name] = _first_tensor(output).detach()

        return hook

    try:
        for name, module in module_items:
            handles.append(module.register_forward_hook(make_hook(name)))
        was_training = bool(getattr(base_model, "training", False))
        base_model.eval()
        with torch.no_grad():
            base_model(**model_inputs)
        if was_training:
            base_model.train()
    finally:
        for handle in handles:
            handle.remove()

    missing = [name for name, _module in module_items if name not in captured]
    if missing:
        raise RuntimeError(f"Activation hooks did not capture expected sites: {missing!r}")
    return [(name, captured[name]) for name, _module in module_items]


def _activation_vector_at_index(
    site_activations: Sequence[tuple[str, torch.Tensor]],
    index: int,
) -> torch.Tensor:
    pieces: List[torch.Tensor] = []
    for name, activation in site_activations:
        if activation.dim() < 3:
            raise ValueError(
                f"ACTIEND position encoding requires sequence activations for site {name!r}; "
                f"got shape {tuple(activation.shape)}"
            )
        if index < 0 or index >= int(activation.shape[1]):
            raise IndexError(
                f"Token index {index} out of range for site {name!r} with seq_len={activation.shape[1]}"
            )
        pieces.append(activation[0, index].reshape(-1))
    return torch.cat(pieces, dim=-1)


def _compute_activation_token_encodings(
    model_with_gradiend: Any,
    text: str,
    *,
    component_key: Any,
    component_id: Optional[str],
    component_label: Optional[str],
    skip_special_tokens: bool,
    max_tokens: int,
    max_length: Optional[int],
) -> List[TokenEncodingRow]:
    """Encode the ACTIEND activation vector at each visible token position."""
    tokenizer = model_with_gradiend.tokenizer
    tokens = _token_rows(
        tokenizer,
        str(text),
        skip_special_tokens=skip_special_tokens,
        max_length=max_length,
    )
    if len(tokens) > int(max_tokens):
        raise ValueError(f"Refusing to score {len(tokens)} tokens; increase max_tokens to continue")

    site_activations = _capture_site_activations(
        model_with_gradiend,
        str(text),
        max_length=max_length,
    )
    rows: List[TokenEncodingRow] = []
    for token in tokens:
        token_text = str(text[token["start"]:token["end"]]) if token.get("start") is not None else token["token"]
        signal = _activation_vector_at_index(site_activations, int(token["index"]))
        rows.append(
            TokenEncodingRow(
                token=token_text,
                encoded=_encode_signal(model_with_gradiend, signal, component_key=component_key),
                index=int(token["index"]),
                token_id=int(token["token_id"]),
                start=token.get("start"),
                end=token.get("end"),
                component_id=component_id,
                component_label=component_label,
                masked_text=None,
                label=None,
            )
        )
    return rows


def _compute_gradient_token_encodings(
    model_with_gradiend: Any,
    text: str,
    *,
    label: Optional[str],
    component_key: Any,
    component_id: Optional[str],
    component_label: Optional[str],
    skip_special_tokens: bool,
    max_tokens: int,
    max_length: Optional[int],
    mask_token: Optional[str],
) -> List[TokenEncodingRow]:
    """Leave-one-out mask + GRADIEND gradient encoding for each visible token."""
    tokenizer = model_with_gradiend.tokenizer
    if not hasattr(model_with_gradiend, "create_gradients"):
        raise ValueError("GRADIEND token encoding requires model_with_gradiend.create_gradients")

    resolved_mask_token = mask_token or getattr(tokenizer, "mask_token", None) or "[MASK]"
    tokens = _token_rows(
        tokenizer,
        str(text),
        skip_special_tokens=skip_special_tokens,
        max_length=max_length,
    )
    if len(tokens) > int(max_tokens):
        raise ValueError(f"Refusing to score {len(tokens)} tokens; increase max_tokens to continue")

    rows: List[TokenEncodingRow] = []
    for token in tokens:
        token_text = str(text[token["start"]:token["end"]]) if token.get("start") is not None else token["token"]
        target_label = str(label) if label is not None else token_text
        masked_text = _mask_text(str(text), token, resolved_mask_token)
        with torch.enable_grad():
            signal = model_with_gradiend.create_gradients(masked_text, target_label)
        rows.append(
            TokenEncodingRow(
                token=token_text,
                encoded=_encode_signal(model_with_gradiend, signal, component_key=component_key),
                index=int(token["index"]),
                token_id=int(token["token_id"]),
                start=token.get("start"),
                end=token.get("end"),
                component_id=component_id,
                component_label=component_label,
                masked_text=masked_text,
                label=target_label,
            )
        )
    return rows


def compute_token_encodings(
    model_with_gradiend: Any,
    text: str,
    *,
    label: Optional[str] = None,
    component: Union[str, int, None] = None,
    skip_special_tokens: bool = True,
    max_tokens: int = 128,
    max_length: Optional[int] = None,
    mask_token: Optional[str] = None,
) -> List[TokenEncodingRow]:
    """
    Compute one encoded value per visible token.

    - ACTIEND (``uses_activations``): encode the activation vector at each token

      position (one forward pass).

    - GRADIEND: mask each token, create gradients for that leave-one-out example,

      and encode the resulting gradient signal.
    """
    tokenizer = getattr(model_with_gradiend, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("compute_token_encodings requires a text model with a tokenizer")
    if not hasattr(model_with_gradiend, "encode"):
        raise ValueError("compute_token_encodings requires model_with_gradiend.encode")

    component_key, component_id, component_label = _resolve_component(model_with_gradiend, component)
    if bool(getattr(model_with_gradiend, "uses_activations", False)):
        return _compute_activation_token_encodings(
            model_with_gradiend,
            text,
            component_key=component_key,
            component_id=component_id,
            component_label=component_label,
            skip_special_tokens=skip_special_tokens,
            max_tokens=max_tokens,
            max_length=max_length,
        )
    return _compute_gradient_token_encodings(
        model_with_gradiend,
        text,
        label=label,
        component_key=component_key,
        component_id=component_id,
        component_label=component_label,
        skip_special_tokens=skip_special_tokens,
        max_tokens=max_tokens,
        max_length=max_length,
        mask_token=mask_token,
    )


def render_token_encoding_html(
    rows: Sequence[Union[TokenEncodingRow, Dict[str, Any]]],
    *,
    color_center: ColorCenter = "zero",
    neutral_values: Any = None,
    neutral_value: Optional[float] = None,
    color_range: ColorRange = "symmetric",
    color_extent: Optional[float] = 1.0,
    title: Optional[str] = None,
) -> str:
    """Render token encoding rows as dependency-light HTML spans."""
    dict_rows = [row.to_dict() if isinstance(row, TokenEncodingRow) else dict(row) for row in rows]
    values = [float(row["encoded"]) for row in dict_rows]
    norm = resolve_encoding_color_norm(
        values,
        neutral_values=neutral_values,
        neutral_value=neutral_value,
        center=color_center,
        color_range=color_range,
        extent=color_extent,
    )
    parts = [
        "<div class=\"gradiend-token-encoding\" "
        "style=\"font-family: system-ui, -apple-system, Segoe UI, sans-serif; line-height: 2.2;\">"
    ]
    if title:
        parts.append(f"<div style=\"font-weight: 600; margin-bottom: 0.4rem;\">{escape(str(title))}</div>")
    parts.append("<div>")
    for row in dict_rows:
        score = float(row["encoded"])
        color = diverging_rgb(norm.normalize(score))
        tooltip_items = [f"token={row.get('token')!r}", f"encoded={score:.4g}"]
        if row.get("component_label"):
            tooltip_items.append(f"component={row['component_label']}")
        title_text = escape("; ".join(tooltip_items), quote=True)
        parts.append(
            f"<span title=\"{title_text}\" "
            f"style=\"background: {color}; border-radius: 0.25rem; padding: 0.12rem 0.2rem; "
            "margin: 0 0.05rem; box-decoration-break: clone; -webkit-box-decoration-break: clone;\">"
            f"{escape(str(row.get('token', '')))}</span>"
        )
    parts.append("</div>")
    parts.append(
        "<div style=\"font-size: 0.8rem; color: #555; margin-top: 0.35rem;\">"
        f"{escape(norm.legend_label)}; bounds [{norm.vmin:.3g}, {norm.vmax:.3g}]"
        "</div>"
    )
    parts.append("</div>")
    return "".join(parts)


def highlight_token_encoding(
    model_with_gradiend: Any,
    text: str,
    *,
    label: Optional[str] = None,
    component: Union[str, int, None] = None,
    interactive: bool = False,
    show: bool = True,
    return_rows: bool = False,
    color_center: ColorCenter = "zero",
    neutral_values: Any = None,
    neutral_value: Optional[float] = None,
    color_range: ColorRange = "symmetric",
    color_extent: Optional[float] = 1.0,
    title: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Compute and render a token-encoding highlight view."""
    if interactive:
        return _interactive_token_encoding(
            model_with_gradiend,
            text,
            label=label,
            component=component,
            show=show,
            color_center=color_center,
            neutral_values=neutral_values,
            neutral_value=neutral_value,
            color_range=color_range,
            color_extent=color_extent,
            title=title,
            **kwargs,
        )

    rows = compute_token_encodings(
        model_with_gradiend,
        text,
        label=label,
        component=component,
        **kwargs,
    )
    html = render_token_encoding_html(
        rows,
        color_center=color_center,
        neutral_values=neutral_values,
        neutral_value=neutral_value,
        color_range=color_range,
        color_extent=color_extent,
        title=title,
    )
    if show:
        try:
            from IPython.display import HTML, display

            display(HTML(html))
        except Exception:
            pass
    if return_rows:
        return html, rows
    return html


def _interactive_token_encoding(
    model_with_gradiend: Any,
    text: str,
    *,
    label: Optional[str],
    component: Union[str, int, None],
    show: bool,
    color_center: ColorCenter,
    neutral_values: Any,
    neutral_value: Optional[float],
    color_range: ColorRange,
    color_extent: Optional[float],
    title: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    try:
        import ipywidgets as widgets
        from IPython.display import HTML, display
    except Exception:
        return highlight_token_encoding(
            model_with_gradiend,
            text,
            label=label,
            component=component,
            interactive=False,
            show=show,
            color_center=color_center,
            neutral_values=neutral_values,
            neutral_value=neutral_value,
            color_range=color_range,
            color_extent=color_extent,
            title=title,
            **kwargs,
        )

    components = list_token_encoding_components(model_with_gradiend)
    options = [(item["component_label"], item["component_id"]) for item in components]
    text_widget = widgets.Textarea(value=str(text), layout=widgets.Layout(width="100%", height="5rem"))
    label_widget = widgets.Text(value="" if label is None else str(label), description="Label")
    component_widget = widgets.Dropdown(options=options, value=component or "aggregate", description="Component")
    output = widgets.Output()

    def refresh(*_):
        with output:
            output.clear_output(wait=True)
            selected_component = component_widget.value
            if selected_component == "aggregate":
                selected_component = None
            html = highlight_token_encoding(
                model_with_gradiend,
                text_widget.value,
                label=label_widget.value or None,
                component=selected_component,
                interactive=False,
                show=False,
                color_center=color_center,
                neutral_values=neutral_values,
                neutral_value=neutral_value,
                color_range=color_range,
                color_extent=color_extent,
                title=title,
                **kwargs,
            )
            display(HTML(html))

    button = widgets.Button(description="Refresh", button_style="primary")
    button.on_click(refresh)
    refresh()
    widget = widgets.VBox([text_widget, widgets.HBox([label_widget, component_widget, button]), output])
    if show:
        display(widget)
    return widget


__all__ = [
    "TokenEncodingRow",
    "compute_token_encodings",
    "highlight_token_encoding",
    "list_token_encoding_components",
    "render_token_encoding_html",
]
