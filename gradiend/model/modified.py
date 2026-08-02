"""Utilities for GRADIEND/ACTIEND-modified models."""

from __future__ import annotations

import json
import os
from types import MethodType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


MODIFIED_CONFIG_NAME = "gradiend_modified_config.json"
MODIFIED_TENSORS_NAME = "gradiend_modified_tensors.pt"
ENCODER_TOKEN_SELECTORS = frozenset({"encoder_abs", "encoder_threshold", "encoder_direction", "encoder_range"})


def _named_modules(model: nn.Module) -> Dict[str, nn.Module]:
    return dict(model.named_modules())


def _resolve_module(model: nn.Module, name: str) -> nn.Module:
    modules = _named_modules(model)
    if name not in modules:
        raise KeyError(f"Activation intervention module {name!r} not found in model")
    return modules[name]


def _first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
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


def _replace_first_tensor(value: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(value):
        return replacement
    if isinstance(value, tuple):
        out = list(value)
        for i, item in enumerate(out):
            try:
                out[i] = _replace_first_tensor(item, replacement)
                return tuple(out)
            except TypeError:
                continue
    if isinstance(value, list):
        out = list(value)
        for i, item in enumerate(out):
            try:
                out[i] = _replace_first_tensor(item, replacement)
                return out
            except TypeError:
                continue
    if isinstance(value, dict):
        out = dict(value)
        for key, item in out.items():
            try:
                out[key] = _replace_first_tensor(item, replacement)
                return out
            except TypeError:
                continue
    raise TypeError(f"Hook output did not contain a replaceable tensor, got {type(value).__name__}")


def _input_context(args: Iterable[Any], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    ctx = dict(kwargs)
    args = tuple(args)
    if args and "input_ids" not in ctx and torch.is_tensor(args[0]):
        ctx["input_ids"] = args[0]
    return ctx


def _strip_gradiend_only_kwargs(kwargs: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Remove hook-only kwargs before the wrapped Hugging Face model sees them."""
    stripped = None
    for key in ("prediction_mask",):
        if key in kwargs:
            if stripped is None:
                stripped = dict(kwargs)
            stripped.pop(key, None)
    return stripped


def _mask_for_selector(
    *,
    selector: Any,
    context: Mapping[str, Any],
    activation: torch.Tensor,
    mask_token_id: Optional[int] = None,
    cls_token_id: Optional[int] = None,
) -> Optional[torch.Tensor]:
    if selector is None or selector in {"all", "mean"}:
        return None
    if activation.dim() < 3:
        raise ValueError(f"Activation token selector {selector!r} requires activation shape (batch, seq, ...)")
    if selector in {"last_token", "current_generation_token"}:
        raise ValueError(
            f"ACTIEND token_selector={selector!r} has been removed because it duplicates "
            "token_selector='prediction' for CLM and is not the prediction site for MLM. "
            "Use token_selector='prediction' for task prediction-site steering, "
            "token_selector='all' for unconditional steering, or an integer selector for a raw positional ablation."
        )
    if isinstance(selector, int):
        mask = torch.zeros(activation.shape[:2], dtype=torch.bool, device=activation.device)
        mask[:, selector] = True
        return mask
    if selector == "prediction":
        prediction_mask = context.get("prediction_mask")
        if torch.is_tensor(prediction_mask):
            return prediction_mask.to(device=activation.device, dtype=torch.bool)
        input_ids = context.get("input_ids")
        if torch.is_tensor(input_ids) and mask_token_id is not None:
            # MLM-style prediction rows expose their prediction slot as the mask
            # token at eval time. Plain neutral LMS rows have no mask token, so
            # this deliberately returns an all-false mask and leaves them
            # unmodified.
            return input_ids.to(device=activation.device).eq(int(mask_token_id))
        if torch.is_tensor(input_ids) or torch.is_tensor(context.get("attention_mask")):
            # Decoder-only CLM prediction is the next token after the visible
            # prefix, so the hook position is the last non-padding prefix token.
            return _clm_prediction_position_mask(context=context, activation=activation)
        raise ValueError(
            "ACTIEND token_selector='prediction' requires prediction_mask, MLM input_ids with mask_token_id, "
            "or CLM-style input_ids/attention_mask so the prediction position can be resolved."
        )
    if selector == "mask":
        input_ids = context.get("input_ids")
        if not torch.is_tensor(input_ids) or mask_token_id is None:
            raise ValueError("ACTIEND token_selector='mask' requires input_ids and mask_token_id")
        return input_ids.to(device=activation.device).eq(int(mask_token_id))
    if selector == "cls":
        input_ids = context.get("input_ids")
        if torch.is_tensor(input_ids) and cls_token_id is not None:
            return input_ids.to(device=activation.device).eq(int(cls_token_id))
        mask = torch.zeros(activation.shape[:2], dtype=torch.bool, device=activation.device)
        mask[:, 0] = True
        return mask
    raise ValueError(f"Unsupported ACTIEND token_selector {selector!r}")


def _clm_prediction_position_mask(*, context: Mapping[str, Any], activation: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(activation.shape[:2], dtype=torch.bool, device=activation.device)
    attention_mask = context.get("attention_mask")
    if torch.is_tensor(attention_mask) and attention_mask.shape == activation.shape[:2]:
        lengths = attention_mask.to(device=activation.device).long().sum(dim=1).clamp_min(1)
        mask[torch.arange(activation.shape[0], device=activation.device), lengths - 1] = True
    else:
        mask[:, -1] = True
    return mask


def _encoder_score(
    activation: torch.Tensor,
    *,
    encoder_weight: torch.Tensor,
    encoder_bias: Optional[torch.Tensor],
    encoder_activation: Optional[str] = None,
) -> torch.Tensor:
    if activation.dim() < 3:
        raise ValueError("ACTIEND encoder selectors require activation shape (batch, seq, ...)")
    weight = encoder_weight.to(device=activation.device, dtype=activation.dtype)
    bias = encoder_bias.to(device=activation.device, dtype=activation.dtype) if encoder_bias is not None else None
    if weight.shape[-1] != activation.shape[-1]:
        raise ValueError(
            f"ACTIEND encoder width {weight.shape[-1]} does not match activation last dimension "
            f"{activation.shape[-1]}"
        )
    encoded = F.linear(activation, weight, bias)
    activation_name = str(encoder_activation or "id").lower()
    if activation_name == "tanh":
        return torch.tanh(encoded)
    if activation_name == "relu":
        return F.relu(encoded)
    if activation_name == "leakyrelu":
        return F.leaky_relu(encoded)
    if activation_name == "smht":
        return F.hardtanh(encoded)
    if activation_name == "elu":
        return F.elu(encoded)
    if activation_name == "gelu":
        return F.gelu(encoded)
    if activation_name == "sigmoid":
        return torch.sigmoid(encoded)
    if activation_name == "silu":
        return F.silu(encoded)
    if activation_name in {"id", "identity", "linear", "none"}:
        return encoded
    raise ValueError(f"Unsupported ACTIEND encoder activation {encoder_activation!r}")


def _direction_tensor(value: Any, *, encoded: torch.Tensor) -> torch.Tensor:
    if value is None:
        if encoded.shape[-1] != 1:
            raise ValueError("ACTIEND encoder_direction selector requires direction for multi-feature encodings")
        value = 1.0
    direction = torch.as_tensor(value, device=encoded.device, dtype=encoded.dtype).flatten()
    if direction.numel() == 1 and encoded.shape[-1] != 1:
        direction = direction.expand(encoded.shape[-1])
    if direction.numel() != encoded.shape[-1]:
        raise ValueError(
            f"ACTIEND encoder direction width {direction.numel()} does not match encoded width {encoded.shape[-1]}"
        )
    return direction


def _encoder_selector_mask(
    activation: torch.Tensor,
    *,
    selector: Any,
    encoder_weight: torch.Tensor,
    encoder_bias: Optional[torch.Tensor],
    encoder_activation: Optional[str],
    threshold: float,
    direction: Any = None,
    target_encoding: Any = None,
    tolerance: float = 0.2,
) -> torch.Tensor:
    encoded = _encoder_score(
        activation,
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        encoder_activation=encoder_activation,
    )
    if selector in {"encoder_abs", "encoder_threshold"}:
        return encoded.abs().amax(dim=-1) > float(threshold)
    if selector == "encoder_direction":
        direction_tensor = _direction_tensor(direction, encoded=encoded)
        projected = (encoded * direction_tensor).sum(dim=-1)
        return projected > float(threshold)
    if selector == "encoder_range":
        if target_encoding is None:
            raise ValueError("ACTIEND token_selector='encoder_range' requires target_encoding")
        target = torch.as_tensor(target_encoding, device=encoded.device, dtype=encoded.dtype).flatten()
        if target.numel() == 1 and encoded.shape[-1] != 1:
            target = target.expand(encoded.shape[-1])
        if target.numel() != encoded.shape[-1]:
            raise ValueError(
                f"ACTIEND encoder target width {target.numel()} does not match encoded width {encoded.shape[-1]}"
            )
        return (encoded - target).abs().amax(dim=-1) < float(tolerance)
    raise ValueError(f"Unsupported ACTIEND encoder selector {selector!r}")


def _selector_mask_for_activation(
    activation: torch.Tensor,
    *,
    selector: Any,
    context: Mapping[str, Any],
    mask_token_id: Optional[int] = None,
    cls_token_id: Optional[int] = None,
    encoder_weight: Optional[torch.Tensor] = None,
    encoder_bias: Optional[torch.Tensor] = None,
    encoder_activation: Optional[str] = None,
    threshold: float = 0.5,
    direction: Any = None,
    target_encoding: Any = None,
    tolerance: float = 0.2,
) -> Optional[torch.Tensor]:
    """Resolve the token mask used by ACTIEND hooks for one activation tensor."""
    if selector in ENCODER_TOKEN_SELECTORS:
        if encoder_weight is None:
            raise ValueError(f"ACTIEND token_selector={selector!r} requires persisted encoder_weight")
        return _encoder_selector_mask(
            activation,
            selector=selector,
            encoder_weight=encoder_weight,
            encoder_bias=encoder_bias,
            encoder_activation=encoder_activation,
            threshold=threshold,
            direction=direction,
            target_encoding=target_encoding,
            tolerance=tolerance,
        )
    return _mask_for_selector(
        selector=selector,
        context=context,
        activation=activation,
        mask_token_id=mask_token_id,
        cls_token_id=cls_token_id,
    )


def _selector_mask_from_application(
    activation: torch.Tensor,
    *,
    application: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
    context: Mapping[str, Any],
) -> Optional[torch.Tensor]:
    """Resolve the ACTIEND selector mask from a persisted hook application config."""
    selector = application.get("token_selector")
    activation_gate = application.get("activation_gate")
    encoder_weight_key = application.get("encoder_weight_key")
    encoder_bias_key = application.get("encoder_bias_key")
    encoder_weight = tensors.get(encoder_weight_key) if encoder_weight_key is not None else None
    encoder_bias = tensors.get(encoder_bias_key) if encoder_bias_key is not None else None
    common = dict(
        context=context,
        mask_token_id=application.get("mask_token_id"),
        cls_token_id=application.get("cls_token_id"),
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        encoder_activation=application.get("encoder_activation"),
        threshold=float(application.get("threshold", 0.5)),
        direction=application.get("direction"),
        target_encoding=application.get("target_encoding"),
        tolerance=float(application.get("tolerance", 0.2)),
    )
    if activation_gate is None:
        return _selector_mask_for_activation(activation, selector=selector, **common)

    if activation_gate not in ENCODER_TOKEN_SELECTORS:
        raise ValueError(f"Unsupported ACTIEND activation_gate {activation_gate!r}")
    if selector in ENCODER_TOKEN_SELECTORS:
        raise ValueError(
            "ACTIEND activation_gate composes with position token selectors only; "
            f"got token_selector={selector!r} and activation_gate={activation_gate!r}"
        )
    base_mask = _mask_for_selector(
        selector=selector,
        context=context,
        activation=activation,
        mask_token_id=application.get("mask_token_id"),
        cls_token_id=application.get("cls_token_id"),
    )
    gate_mask = _selector_mask_for_activation(activation, selector=activation_gate, **common)
    if base_mask is None:
        return gate_mask
    if gate_mask is None:
        return base_mask
    if base_mask.shape != gate_mask.shape:
        raise ValueError(
            f"ACTIEND composed selector mask shape mismatch: token selector produced {tuple(base_mask.shape)}, "
            f"activation gate produced {tuple(gate_mask.shape)}"
        )
    return base_mask & gate_mask


def _position_scope_mask_from_application(
    activation: torch.Tensor,
    *,
    application: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Optional[torch.Tensor]:
    """Resolve only the position part of a composed ACTIEND selector."""
    selector = application.get("token_selector")
    if selector in ENCODER_TOKEN_SELECTORS:
        return None
    return _mask_for_selector(
        selector=selector,
        context=context,
        activation=activation,
        mask_token_id=application.get("mask_token_id"),
        cls_token_id=application.get("cls_token_id"),
    )


def _activation_prefix_size(activation: torch.Tensor) -> int:
    prefix_shape = tuple(activation.shape[:-1])
    if not prefix_shape:
        return 1
    total = 1
    for dim in prefix_shape:
        total *= int(dim)
    return int(total)


def _mask_counts(activation: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[int, int]:
    if mask is None:
        total = _activation_prefix_size(activation)
        return total, total
    if mask.shape != activation.shape[:2]:
        raise ValueError(
            f"ACTIEND token selector mask shape {tuple(mask.shape)} does not match activation prefix "
            f"{tuple(activation.shape[:2])}"
        )
    return int(mask.to(dtype=torch.bool).sum().item()), int(mask.numel())


def _apply_steering_with_mask(
    activation: torch.Tensor,
    vector: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    vector = vector.to(device=activation.device, dtype=activation.dtype).flatten()
    if activation.shape[-1] != vector.numel():
        raise ValueError(
            f"ACTIEND steering vector width {vector.numel()} does not match activation last dimension "
            f"{activation.shape[-1]}"
        )
    view_shape = (1,) * (activation.dim() - 1) + (vector.numel(),)
    if mask is None:
        return activation + vector.reshape(view_shape)
    if mask.shape != activation.shape[:2]:
        raise ValueError(
            f"ACTIEND token selector mask shape {tuple(mask.shape)} does not match activation prefix "
            f"{tuple(activation.shape[:2])}"
        )
    out = activation.clone()
    out[mask] = out[mask] + vector
    return out


def _add_steering(
    activation: torch.Tensor,
    vector: torch.Tensor,
    *,
    selector: Any,
    context: Mapping[str, Any],
    mask_token_id: Optional[int] = None,
    cls_token_id: Optional[int] = None,
    encoder_weight: Optional[torch.Tensor] = None,
    encoder_bias: Optional[torch.Tensor] = None,
    encoder_activation: Optional[str] = None,
    threshold: float = 0.5,
    direction: Any = None,
    target_encoding: Any = None,
    tolerance: float = 0.2,
) -> torch.Tensor:
    mask = _selector_mask_for_activation(
        activation,
        selector=selector,
        context=context,
        mask_token_id=mask_token_id,
        cls_token_id=cls_token_id,
        encoder_weight=encoder_weight,
        encoder_bias=encoder_bias,
        encoder_activation=encoder_activation,
        threshold=threshold,
        direction=direction,
        target_encoding=target_encoding,
        tolerance=tolerance,
    )
    return _apply_steering_with_mask(activation, vector, mask)


def _register_activation_steering_hooks(
    model: nn.Module,
    config: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
) -> List[Any]:
    context: Dict[str, Any] = {}

    def root_pre_hook(_module: nn.Module, args: Any, kwargs: Any) -> Any:
        context.clear()
        context.update(_input_context(args, kwargs))
        stripped = _strip_gradiend_only_kwargs(kwargs)
        if stripped is not None:
            return args, stripped
        return None

    handles = [model.register_forward_pre_hook(root_pre_hook, with_kwargs=True)]
    for intervention in config.get("interventions", []):
        module_name = intervention["module"]
        vector_key = intervention["tensor_key"]
        vector = tensors[vector_key]
        application = dict(intervention.get("application") or {})
        module = _resolve_module(model, module_name)

        def make_hook(steering_vector: torch.Tensor, app: Mapping[str, Any]):
            def hook(_module: nn.Module, _args: Any, output: Any) -> Any:
                activation = _first_tensor(output)
                mask = _selector_mask_from_application(
                    activation,
                    application=app,
                    tensors=tensors,
                    context=context,
                )
                steered = _apply_steering_with_mask(activation, steering_vector, mask)
                return _replace_first_tensor(output, steered)

            return hook

        handles.append(
            module.register_forward_hook(make_hook(vector, application))
        )
    return handles


def remove_hook_handles(handles: Iterable[Any]) -> None:
    """Remove PyTorch hook handles, ignoring handles that were already removed."""
    for handle in list(handles):
        remove = getattr(handle, "remove", None)
        if remove is not None:
            remove()


def activation_steering_config(interventions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the versioned operational config for fixed activation steering hooks."""
    return {
        "format": "gradiend_modified_model",
        "format_version": 1,
        "type": "actiend_activation_hooks",
        "future": {
            "auto_model_from_pretrained": (
                "A future format may register a custom Hugging Face config/model class so "
                "AutoModel.from_pretrained can restore hooks directly."
            )
        },
        "interventions": interventions,
    }


def register_activation_steering_hooks(
    model: nn.Module,
    *,
    interventions: List[Dict[str, Any]],
    tensors: Mapping[str, torch.Tensor],
) -> List[Any]:
    """Register fixed activation steering hooks and return removable handles."""
    return _register_activation_steering_hooks(
        model,
        activation_steering_config(interventions),
        tensors,
    )


def activation_selector_coverage_for_inputs(
    model: nn.Module,
    *,
    interventions: List[Dict[str, Any]],
    tensors: Mapping[str, torch.Tensor],
    args: Iterable[Any] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run one forward pass and summarize which positions ACTIEND selectors choose.

    This is an observer-only diagnostic: it uses the same selector mask resolver
    as ACTIEND steering hooks, but it does not apply the steering vector. For
    multi-site interventions, downstream-site coverage is therefore measured on
    the unmodified base activations, which is usually the desired cheap sanity
    check for neutral firing.
    """
    context: Dict[str, Any] = {}
    records: List[Dict[str, Any]] = []
    forward_kwargs = dict(kwargs or {})

    def root_pre_hook(_module: nn.Module, hook_args: Any, hook_kwargs: Any) -> Any:
        context.clear()
        context.update(_input_context(hook_args, hook_kwargs))
        stripped = _strip_gradiend_only_kwargs(hook_kwargs)
        if stripped is not None:
            return hook_args, stripped
        return None

    handles = [model.register_forward_pre_hook(root_pre_hook, with_kwargs=True)]
    for intervention in interventions:
        module_name = intervention["module"]
        application = dict(intervention.get("application") or {})
        module = _resolve_module(model, module_name)

        def make_observer(name: str, app: Mapping[str, Any]):
            def hook(_module: nn.Module, _args: Any, output: Any) -> None:
                activation = _first_tensor(output)
                mask = _selector_mask_from_application(
                    activation,
                    application=app,
                    tensors=tensors,
                    context=context,
                )
                scope_mask = _position_scope_mask_from_application(
                    activation,
                    application=app,
                    context=context,
                )
                selected, total = _mask_counts(activation, mask)
                candidate_positions, _ = _mask_counts(activation, scope_mask)
                records.append({
                    "module": name,
                    "token_selector": app.get("token_selector"),
                    "selected_positions": selected,
                    "candidate_positions": candidate_positions,
                    "total_positions": total,
                    "coverage": (selected / total) if total else None,
                    "scope_coverage": (selected / candidate_positions) if candidate_positions else None,
                    "activation_shape": tuple(int(dim) for dim in activation.shape),
                })

            return hook

        handles.append(module.register_forward_hook(make_observer(module_name, application)))

    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.no_grad():
            model(*tuple(args), **forward_kwargs)
    finally:
        remove_hook_handles(handles)
        if was_training:
            model.train()

    by_module: Dict[str, Dict[str, Any]] = {}
    for record in records:
        module_name = str(record["module"])
        item = by_module.setdefault(
            module_name,
            {
                "module": module_name,
                "token_selector": record.get("token_selector"),
                "selected_positions": 0,
                "candidate_positions": 0,
                "total_positions": 0,
                "calls": 0,
            },
        )
        item["selected_positions"] += int(record["selected_positions"])
        item["candidate_positions"] += int(record["candidate_positions"])
        item["total_positions"] += int(record["total_positions"])
        item["calls"] += 1

    module_rows: List[Dict[str, Any]] = []
    for item in by_module.values():
        total = int(item["total_positions"])
        candidate_total = int(item["candidate_positions"])
        module_rows.append({
            **item,
            "coverage": (int(item["selected_positions"]) / total) if total else None,
            "scope_coverage": (
                int(item["selected_positions"]) / candidate_total
                if candidate_total
                else None
            ),
        })

    selected_total = sum(int(row["selected_positions"]) for row in records)
    candidate_total = sum(int(row["candidate_positions"]) for row in records)
    position_total = sum(int(row["total_positions"]) for row in records)
    return {
        "selected_positions": selected_total,
        "candidate_positions": candidate_total,
        "total_positions": position_total,
        "coverage": (selected_total / position_total) if position_total else None,
        "scope_coverage": (selected_total / candidate_total) if candidate_total else None,
        "calls": len(records),
        "modules": module_rows,
    }


def apply_activation_steering(
    model: nn.Module,
    *,
    interventions: List[Dict[str, Any]],
    tensors: Mapping[str, torch.Tensor],
) -> nn.Module:
    """Attach fixed activation steering hooks to ``model`` and return it."""
    config = activation_steering_config(interventions)
    handles = _register_activation_steering_hooks(model, config, tensors)
    setattr(model, "_gradiend_modified_config", config)
    setattr(model, "_gradiend_modified_tensors", {k: v.detach().cpu() for k, v in tensors.items()})
    setattr(model, "_gradiend_modified_hook_handles", handles)
    attach_modified_model_save(model)
    return model


def attach_modified_model_save(model: nn.Module) -> nn.Module:
    if hasattr(model, "save_pretrained_modified"):
        return model

    def save_pretrained_modified(self: nn.Module, save_directory: str, **kwargs: Any) -> None:
        save_modified_model(self, save_directory, **kwargs)

    setattr(model, "save_pretrained_modified", MethodType(save_pretrained_modified, model))
    return model


def save_modified_model(model: nn.Module, save_directory: str, **kwargs: Any) -> None:
    """Save a model with GRADIEND/ACTIEND runtime modifications."""
    config = getattr(model, "_gradiend_modified_config", None)
    tensors = getattr(model, "_gradiend_modified_tensors", None)
    if not isinstance(config, Mapping) or not isinstance(tensors, Mapping):
        raise ValueError("model does not carry GRADIEND modified-model metadata")
    os.makedirs(save_directory, exist_ok=True)
    if not hasattr(model, "save_pretrained"):
        raise TypeError("Modified model must provide save_pretrained() to persist base model weights")
    model.save_pretrained(save_directory, **kwargs)
    with open(os.path.join(save_directory, MODIFIED_CONFIG_NAME), "w", encoding="utf-8") as handle:
        json.dump(dict(config), handle, indent=2)
    torch.save({k: v.detach().cpu() for k, v in tensors.items()}, os.path.join(save_directory, MODIFIED_TENSORS_NAME))


def load_modified_model(
    load_directory: str,
    *,
    model_loader: Optional[Callable[[str], nn.Module]] = None,
    **model_loader_kwargs: Any,
) -> nn.Module:
    """Load a model saved by ``save_pretrained_modified``."""
    cfg_path = os.path.join(load_directory, MODIFIED_CONFIG_NAME)
    tensor_path = os.path.join(load_directory, MODIFIED_TENSORS_NAME)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing {MODIFIED_CONFIG_NAME} in {load_directory}")
    if not os.path.isfile(tensor_path):
        raise FileNotFoundError(f"Missing {MODIFIED_TENSORS_NAME} in {load_directory}")
    with open(cfg_path, encoding="utf-8") as handle:
        config = json.load(handle)
    if "format_version" not in config and "version" in config:
        config["format_version"] = config["version"]
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if model_loader is None:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(load_directory, **model_loader_kwargs)
    else:
        model = model_loader(load_directory, **model_loader_kwargs)
    handles = _register_activation_steering_hooks(model, config, tensors)
    setattr(model, "_gradiend_modified_config", config)
    setattr(model, "_gradiend_modified_tensors", tensors)
    setattr(model, "_gradiend_modified_hook_handles", handles)
    attach_modified_model_save(model)
    return model


def activation_interventions_from_gradiend(
    gradiend: Any,
    steering_vector: torch.Tensor,
    *,
    token_selector: Any = None,
    activation_gate: Any = None,
    threshold: float = 0.5,
    direction: Any = None,
    target_encoding: Any = None,
    tolerance: float = 0.2,
    encoder_tensors: Optional[Mapping[str, Mapping[str, torch.Tensor]]] = None,
    mask_token_id: Optional[int] = None,
    cls_token_id: Optional[int] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
    """Split a flat ACTIEND steering vector into per-activation-site hook tensors."""
    if activation_gate is not None and activation_gate not in ENCODER_TOKEN_SELECTORS:
        raise ValueError(f"Unsupported ACTIEND activation_gate {activation_gate!r}")
    interventions: List[Dict[str, Any]] = []
    tensors: Dict[str, torch.Tensor] = {}
    idx = 0
    for i, (name, spec) in enumerate(gradiend.param_map.items()):
        if not name.startswith("activation:"):
            raise ValueError(f"ACTIEND activation mapping expected 'activation:' entries, got {name!r}")
        module_name = name[len("activation:"):]
        shape = tuple(spec["shape"])
        if spec.get("repr") != "all" or len(shape) != 1:
            raise NotImplementedError("ACTIEND hooks currently support full 1D activation-site mappings only")
        n = int(shape[0])
        tensor_key = f"steering_{i}"
        tensors[tensor_key] = steering_vector[idx: idx + n].detach().cpu().reshape(n)
        idx += n
        application = {"axis": "last_dim", "token_selector": token_selector}
        encoder_selector = activation_gate if activation_gate is not None else token_selector
        if encoder_selector in ENCODER_TOKEN_SELECTORS:
            if encoder_tensors is None or name not in encoder_tensors:
                raise ValueError(f"Missing ACTIEND encoder tensors for activation site {name!r}")
            encoder_weight_key = f"encoder_weight_{i}"
            tensors[encoder_weight_key] = encoder_tensors[name]["weight"].detach().cpu()
            application["encoder_weight_key"] = encoder_weight_key
            if activation_gate is not None:
                application["activation_gate"] = activation_gate
            encoder_bias = encoder_tensors[name].get("bias")
            if encoder_bias is not None:
                encoder_bias_key = f"encoder_bias_{i}"
                tensors[encoder_bias_key] = encoder_bias.detach().cpu()
                application["encoder_bias_key"] = encoder_bias_key
            encoder_activation = encoder_tensors[name].get("activation")
            if encoder_activation is not None:
                application["encoder_activation"] = str(encoder_activation)
            application["threshold"] = float(threshold)
            if direction is not None:
                application["direction"] = (
                    list(direction) if isinstance(direction, (list, tuple)) else float(direction)
                )
            if target_encoding is not None:
                application["target_encoding"] = (
                    list(target_encoding) if isinstance(target_encoding, (list, tuple)) else float(target_encoding)
                )
            application["tolerance"] = float(tolerance)
        if mask_token_id is not None:
            application["mask_token_id"] = int(mask_token_id)
        if cls_token_id is not None:
            application["cls_token_id"] = int(cls_token_id)
        interventions.append({
            "module": module_name,
            "tensor_key": tensor_key,
            "application": application,
        })
    if idx != int(steering_vector.numel()):
        raise ValueError(f"Steering vector length mismatch: used {idx}, vector has {steering_vector.numel()}")
    return interventions, tensors
