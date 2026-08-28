"""Signal-space resolution helpers shared by model construction and extractors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from gradiend.model_topology import (
    ModelTopology,
    describe_model_for_topology_error,
    infer_model_topology,
)


@dataclass(frozen=True)
class ResolvedSignalSpace:
    """Concrete flattened input space for one signal."""

    kind: str
    signal_id: str
    input_dim: int
    mapping: Tuple[Dict[str, Any], ...]
    scope: Any = None
    signal: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"Signal space input_dim must be positive, got {self.input_dim}")


@dataclass(frozen=True)
class SignalTrainingPlan:
    """
    Resolved signal-axis training plan.

    Signal and scope are independent user-facing axes. They meet here exactly
    once to define concrete tensor spaces for GRADIEND construction/extraction.
    """

    spaces: Tuple[ResolvedSignalSpace, ...]
    scope: Any = None

    def __post_init__(self) -> None:
        if not self.spaces:
            raise ValueError("SignalTrainingPlan requires at least one signal space")

    @property
    def single_space(self) -> ResolvedSignalSpace:
        if len(self.spaces) != 1:
            raise NotImplementedError(
                f"This construction path currently supports one signal space; got {len(self.spaces)}."
            )
        return self.spaces[0]


def signal_kind(signal: Any) -> str:
    """Return the signal kind from a Signal-like object or shorthand."""
    if signal is None:
        return "gradient"
    if isinstance(signal, str):
        return signal
    if isinstance(signal, Mapping):
        return str(signal.get("kind", "gradient"))
    return str(getattr(signal, "kind", "gradient"))


def signal_id(signal: Any) -> str:
    """Return a stable signal id from a Signal-like object or shorthand."""
    if signal is None:
        return "gradient"
    if isinstance(signal, str):
        return signal
    if isinstance(signal, Mapping):
        return str(signal.get("name") or signal.get("kind") or "gradient")
    return str(getattr(signal, "id", getattr(signal, "kind", "gradient")))


def signal_options(signal: Any) -> Dict[str, Any]:
    """Return options from a Signal-like object."""
    if isinstance(signal, Mapping):
        return dict(signal.get("options") or {})
    options = getattr(signal, "options", None)
    return dict(options or {})


def _iter_signal_axis(signal: Any = None, signals: Any = None) -> Tuple[Any, ...]:
    if signals is None:
        return (signal,) if signal is not None else (None,)
    if isinstance(signals, (str, bytes, Mapping)) or hasattr(signals, "kind"):
        return (signals,)
    if hasattr(signals, "__iter__"):
        return tuple(signals)
    return (signals,)


def _prod_int(values: Any) -> Optional[int]:
    if isinstance(values, int):
        return int(values)
    if isinstance(values, torch.Size):
        values = tuple(values)
    if isinstance(values, (list, tuple)) and values:
        result = 1
        for value in values:
            if not isinstance(value, int):
                return None
            result *= int(value)
        return result
    return None


def infer_static_module_output_dim(module: nn.Module) -> Optional[int]:
    """Infer a module's last activation width without running a forward pass."""
    if isinstance(module, nn.Embedding):
        return int(module.embedding_dim)
    if isinstance(module, nn.Linear):
        return int(module.out_features)
    if isinstance(module, nn.LayerNorm):
        return _prod_int(module.normalized_shape)

    for attr in ("out_features", "embedding_dim", "hidden_size", "nf"):
        value = getattr(module, attr, None)
        if isinstance(value, int) and value > 0:
            return int(value)

    dim = _prod_int(getattr(module, "normalized_shape", None))
    if dim is not None:
        return dim

    for attr in (
        "LayerNorm",
        "layer_norm",
        "norm",
        "ln_f",
        "ln_1",
        "ln_2",
        "input_layernorm",
        "post_attention_layernorm",
        "final_layer_norm",
        "output_layer_norm",
        "sa_layer_norm",
        "word_embeddings",
        "embed_tokens",
        "output",
        "dense",
        "out_proj",
        "proj",
        "projection",
        "c_proj",
    ):
        child = getattr(module, attr, None)
        if isinstance(child, nn.Module):
            dim = infer_static_module_output_dim(child)
            if dim is not None:
                return dim

    children = list(module.children())
    if isinstance(module, nn.Sequential) and children:
        for child in reversed(children):
            dim = infer_static_module_output_dim(child)
            if dim is not None:
                return dim
    return None


def activation_sites_from_scope(scope: Any) -> Tuple[str, ...]:
    """Return activation site patterns from a SignalScope-like object or mapping."""
    if scope is None:
        return ()
    if isinstance(scope, dict):
        raw = scope.get("activation_sites")
    else:
        raw = getattr(scope, "activation_sites", None)
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(item) for item in raw)


def activation_selector_from_scope(scope: Any) -> Optional[Tuple[Any, ...]]:
    """Return semantic activation selector metadata from a SignalScope-like object."""
    if scope is None:
        return None
    if isinstance(scope, Mapping):
        raw = scope.get("activation_selector")
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise TypeError("activation_selector must be a mapping")
        kind = raw.get("kind")
        if kind == "layers":
            layers = raw.get("layers")
            return ("layers", None if layers is None else tuple(int(item) for item in layers))
        if kind in {"embeddings", "word_embedding"}:
            return (kind,)
        raise ValueError(f"Unknown activation_selector kind: {kind!r}")
    return getattr(scope, "activation_selector", None)


def scope_mode(scope: Any) -> str:
    """Return scope preset mode, defaulting to backbone/text-tower scope."""
    if scope is None:
        return "default"
    if isinstance(scope, Mapping):
        return str(scope.get("mode") or "default")
    return str(getattr(scope, "mode", None) or "default")


def scope_params(scope: Any, *, base_model: Optional[nn.Module] = None) -> Optional[Tuple[str, ...]]:
    """Return parameter patterns carried by a SignalScope-like object.

    ``scope.params`` (an explicit include-list) always wins when present. When
    it is absent but the scope carries a semantic ``activation_selector``
    (``SignalScope.layers()``/``.layer()``/``.embeddings()``/``.word_embedding()``)
    and ``base_model`` is given, resolves it into weight-parameter wildcards
    via ``gradient_params_from_selector`` -- see that function's docstring for
    why this matters (a gradient signal used to silently ignore that field
    entirely). ``base_model=None`` (the default, for callers that only need
    ``scope.params`` and have no model handy) preserves the old behavior of
    returning ``None`` when only a selector is set.
    """
    if scope is None:
        return None
    if isinstance(scope, Mapping):
        raw = scope.get("params")
    else:
        raw = getattr(scope, "params", None)
    if raw is not None:
        if isinstance(raw, str):
            return (raw,)
        return tuple(str(item) for item in raw)
    if base_model is not None:
        selector = activation_selector_from_scope(scope)
        if selector is not None:
            return gradient_params_from_selector(base_model, selector)
    return None


def _prefer_text_backbone_if_multimodal(module: nn.Module) -> nn.Module:
    has_vision = any(
        hasattr(module, attr)
        for attr in ("vision_tower", "vision_model", "visual", "vision_encoder")
    )
    if not has_vision:
        return module
    for attr in (
        "language_model",
        "text_model",
        "text_tower",
        "text_encoder",
        "decoder",
        "transformer",
    ):
        candidate = getattr(module, attr, None)
        if isinstance(candidate, nn.Module):
            return candidate
    return module


def _join_module_name(prefix: str, name: str) -> str:
    if not prefix:
        return name
    if not name:
        return prefix
    return f"{prefix}.{name}"


def _find_module_name(base_model: nn.Module, target: nn.Module) -> str:
    target_id = id(target)
    for name, module in base_model.named_modules():
        if id(module) == target_id:
            return name
    return ""


def _prefer_text_backbone_if_multimodal_named(
    base_model: nn.Module,
    module_name: str,
    module: nn.Module,
) -> Tuple[str, nn.Module]:
    selected = _prefer_text_backbone_if_multimodal(module)
    if selected is module:
        return module_name, module
    selected_name = _find_module_name(base_model, selected)
    return selected_name, selected


def _default_scope_root_module(base_model: nn.Module, mode: str) -> nn.Module:
    return _default_scope_root_module_named(base_model, mode)[1]


def _default_scope_root_module_named(base_model: nn.Module, mode: str) -> Tuple[str, nn.Module]:
    if mode == "full":
        return "", base_model
    prefix = getattr(base_model, "base_model_prefix", None)
    if prefix and hasattr(base_model, prefix):
        return _prefer_text_backbone_if_multimodal_named(
            base_model,
            prefix,
            getattr(base_model, prefix),
        )
    if hasattr(base_model, "base_model"):
        root = base_model.base_model
        return _prefer_text_backbone_if_multimodal_named(
            base_model,
            _find_module_name(base_model, root),
            root,
        )
    return _prefer_text_backbone_if_multimodal_named(base_model, "", base_model)


def _get_child_by_path(module: nn.Module, path: str) -> Optional[nn.Module]:
    current: nn.Module = module
    for part in path.split("."):
        child = getattr(current, part, None)
        if not isinstance(child, nn.Module):
            return None
        current = child
    return current


def _append_site_if_static(sites: list[str], module_name: str, module: nn.Module) -> None:
    if module_name and infer_static_module_output_dim(module) is not None and module_name not in sites:
        sites.append(module_name)


def _iter_module_list_items(
    root_name: str,
    root: nn.Module,
    container_paths: Iterable[str],
) -> Iterable[Tuple[str, nn.Module]]:
    for path in container_paths:
        container = _get_child_by_path(root, path)
        if not isinstance(container, (nn.ModuleList, nn.Sequential)):
            continue
        for index, child in enumerate(container):
            if isinstance(child, nn.Module):
                yield _join_module_name(root_name, f"{path}.{index}"), child


def _representation_stream_sites(root_name: str, root: nn.Module) -> Tuple[str, ...]:
    """
    Return activation sites that correspond to model representation stream states.

    For GRADIEND-over-activations, the default scope should measure hidden states
    that features can plausibly live in: combined embedding outputs and
    transformer/block outputs. It should not select implementation internals such
    as position embeddings or individual projection layers merely because they
    have statically inferable widths.
    """
    sites: list[str] = []

    for embedding_attr in ("embeddings", "embed_tokens", "embed", "emb"):
        embedding = getattr(root, embedding_attr, None)
        if isinstance(embedding, nn.Module):
            _append_site_if_static(sites, _join_module_name(root_name, embedding_attr), embedding)
            break

    block_container_paths = (
        "encoder.layer",
        "encoder.layers",
        "encoder.block",
        "encoder",
        "transformer.layer",
        "transformer.h",
        "h",
        "layers",
        "model.layers",
        "decoder.layers",
        "decoder.block",
        "block",
    )
    for name, module in _iter_module_list_items(root_name, root, block_container_paths):
        _append_site_if_static(sites, name, module)

    return tuple(sites)


def _full_scope_boundary_sites(base_model: nn.Module) -> Tuple[str, ...]:
    """Return coarse activation boundaries for an explicit full-model scope."""
    root_name, root = _default_scope_root_module_named(base_model, "default")
    sites = list(_representation_stream_sites(root_name, root))
    for name, module in base_model.named_children():
        if name in sites:
            continue
        if isinstance(module, (nn.ModuleList, nn.Sequential)):
            for index, child in enumerate(module):
                _append_site_if_static(sites, f"{name}.{index}", child)
        else:
            _append_site_if_static(sites, name, module)
    return tuple(sites)


def _topology_default_activation_sites(topology: ModelTopology, *, mode: str) -> Tuple[str, ...]:
    if mode == "full":
        return tuple(
            site
            for site in (
                *topology.embeddings,
                *topology.layers,
                *topology.prediction_heads,
            )
            if site
        )
    return tuple(site for site in (*topology.embeddings, *topology.layers) if site)


def default_activation_sites(base_model: nn.Module, *, mode: str = "default") -> Tuple[str, ...]:
    """Resolve the default/full activation scope to method-level activation sites."""
    if mode not in {"default", "full"}:
        raise ValueError("scope mode must be 'default' or 'full'")
    topology = infer_model_topology(base_model)
    if topology is not None:
        selected = _topology_default_activation_sites(topology, mode=mode)
    elif mode == "full":
        selected = _full_scope_boundary_sites(base_model)
    else:
        root_name, root = _default_scope_root_module_named(base_model, mode)
        selected = _representation_stream_sites(root_name, root)
    if not selected:
        raise ValueError(
            f"Could not infer activation stream sites for SignalScope.{mode}().\n"
            + describe_model_for_topology_error(base_model)
        )
    return tuple(selected)


def _activation_sites_from_selector(base_model: nn.Module, selector: Tuple[Any, ...]) -> Tuple[str, ...]:
    topology = infer_model_topology(base_model)
    if topology is None:
        raise ValueError(
            "Cannot resolve semantic activation scope "
            f"{selector!r} for this model.\n"
            + describe_model_for_topology_error(base_model)
        )

    kind = selector[0]
    if kind == "layers":
        selected_layers = selector[1]
        if selected_layers is None:
            return topology.layers
        sites = []
        for layer in selected_layers:
            if layer >= len(topology.layers):
                raise ValueError(
                    f"SignalScope.layer({layer}) is out of range for {topology.model_type}; "
                    f"available layers are 0..{len(topology.layers) - 1}."
                )
            sites.append(topology.layers[layer])
        return tuple(sites)
    if kind == "embeddings":
        if not topology.embeddings:
            raise ValueError(f"{topology.model_type} topology has no combined embedding activation site")
        return topology.embeddings
    if kind == "word_embedding":
        if topology.word_embedding is None:
            raise ValueError(f"{topology.model_type} topology has no word embedding activation site")
        return (topology.word_embedding,)
    raise ValueError(f"Unknown activation selector: {kind!r}")


def gradient_params_from_selector(base_model: nn.Module, selector: Tuple[Any, ...]) -> Tuple[str, ...]:
    """Translate a semantic ``SignalScope`` selector (``.layers()``/``.layer()``/
    ``.embeddings()``/``.word_embedding()``) into weight-parameter wildcard
    patterns for a gradient signal, via the same architecture-agnostic
    ``ModelTopology`` the activation-signal path already uses.

    ``SignalScope``'s semantic shortcuts are documented as scope metadata
    independent of signal kind, but only the activation-signal path
    (``resolve_activation_modules``, via ``_activation_sites_from_selector``)
    actually resolved them -- ``scope_params``/``scope_mode`` (what a gradient
    signal's weight-parameter scope is built from, see
    ``gradiend/model/core/backbone.py::build_gradiend_from_base_model``) never
    consulted ``activation_selector`` at all, so attaching e.g.
    ``SignalScope.layers()`` to ``Signal.gradient()`` silently resolved to no
    scope restriction whatsoever. Fixed by resolving the same selector into
    module paths via ``_activation_sites_from_selector`` (identical topology,
    identical error messages for out-of-range layers / unsupported
    architectures) and appending ``.*`` so each path becomes a parameter-name
    wildcard matching every weight under that module -- module dotted-paths
    match nothing on their own in ``named_parameters()``, only their leaf
    parameters do.

    Note this is deliberately narrower than "backbone minus embeddings": for
    gpt2, ``topology.layers`` covers only ``transformer.h.*`` (the residual
    blocks), not the final ``transformer.ln_f`` layer norm applied after the
    last block -- that's consistent with ``.layers()``'s own documented
    "transformer-layer outputs" semantics, but callers wanting backbone-minus-
    embeddings specifically (as opposed to residual-blocks-only) should keep
    using an explicit ``SignalScope.from_values(params=[...])`` include-list
    instead.
    """
    sites = _activation_sites_from_selector(base_model, selector)
    return tuple(f"{site}.*" for site in sites)


def resolve_activation_modules(
    base_model: nn.Module,
    activation_sites: Sequence[str],
    *,
    scope: Any = None,
) -> Tuple[Tuple[str, nn.Module], ...]:
    """Resolve exact or wildcard activation site patterns against named modules."""
    if not activation_sites:
        selector = activation_selector_from_scope(scope)
        if selector is None:
            activation_sites = default_activation_sites(base_model, mode=scope_mode(scope))
        else:
            activation_sites = _activation_sites_from_selector(base_model, selector)
    modules = dict(base_model.named_modules())
    selected: Dict[str, nn.Module] = {}
    for pattern in activation_sites:
        if pattern in modules:
            selected[pattern] = modules[pattern]
            continue
        for name, module in modules.items():
            if name and fnmatch(name, pattern):
                selected[name] = module
    if not selected:
        raise ValueError(f"No activation modules matched activation_sites={tuple(activation_sites)!r}")
    return tuple(selected.items())


def resolve_activation_signal_space(base_model: nn.Module, signal: Any, scope: Any) -> ResolvedSignalSpace:
    """Resolve the flattened GRADIEND input space for an activation signal."""
    selector = signal_options(signal).get("token_selector")
    if callable(selector):
        raise ValueError(
            "Cannot statically construct an activation GRADIEND for a callable token_selector, "
            "because it may change the feature width. Use a fixed-width selector such as "
            "'mask', 'cls', 'mean', 'all', or an integer token index."
        )

    sites = activation_sites_from_scope(scope)
    resolved = resolve_activation_modules(base_model, sites, scope=scope)
    mapping = []
    input_dim = 0
    for name, module in resolved:
        dim = infer_static_module_output_dim(module)
        if dim is None:
            raise ValueError(
                f"Cannot statically infer activation width for site {name!r} "
                f"({module.__class__.__name__}). Choose a site with a known output width "
                "or add static dimension support for this module type."
            )
        mapping.append({"name": name, "shape": (int(dim),), "repr": "all"})
        input_dim += int(dim)

    return ResolvedSignalSpace(
        kind="activation",
        signal_id=signal_id(signal),
        input_dim=input_dim,
        mapping=tuple(mapping),
        scope=scope,
        signal=dict(signal.to_dict()) if hasattr(signal, "to_dict") else None,
    )


def resolve_signal_training_plan(
    base_model: nn.Module,
    *,
    training_args: Any = None,
    signal: Any = None,
    signals: Any = None,
    scope: Any = None,
) -> SignalTrainingPlan:
    """Resolve independent signal/scope axes into concrete GRADIEND spaces."""
    if training_args is not None:
        if signal is None:
            signal = getattr(training_args, "signal", None)
        if signals is None:
            signals = getattr(training_args, "signals", None)
        if scope is None:
            scope = getattr(training_args, "signal_scope", None)

    spaces = []
    for item in _iter_signal_axis(signal=signal, signals=signals):
        kind = signal_kind(item)
        if kind == "gradient":
            spaces.append(
                ResolvedSignalSpace(
                    kind="gradient",
                    signal_id=signal_id(item),
                    input_dim=1,
                    mapping=(),
                    scope=scope,
                )
            )
        elif kind == "activation":
            spaces.append(resolve_activation_signal_space(base_model, item, scope))
        else:
            raise NotImplementedError(
                f"Signal training plan resolution for Signal(kind={kind!r}) is not implemented yet."
            )

    return SignalTrainingPlan(spaces=tuple(spaces), scope=scope)
