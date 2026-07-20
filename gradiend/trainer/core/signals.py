"""Signal abstractions for GRADIEND training inputs."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn

from gradiend.signal_space import infer_static_module_output_dim, resolve_activation_modules


SignalKind = str


def _normalize_string_sequence(value: Optional[Union[str, Sequence[str]]], *, name: str) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        items = (value,)
    else:
        items = tuple(value)
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} entries must be non-empty strings")
    return tuple(item.strip() for item in items)


@dataclass(frozen=True)
class Signal:
    """
    Description of what is measured from the base model.

    Scope is deliberately not part of Signal. For example, ``Signal.gradient()``
    says "measure raw gradients"; parameter eligibility remains controlled by
    params/param_map, and component formation remains controlled by GradiendSplit.
    """

    kind: SignalKind
    name: Optional[str] = None
    options: Mapping[str, Any] = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("Signal.kind must be a non-empty string")
        if self.name is not None and (not isinstance(self.name, str) or not self.name.strip()):
            raise ValueError("Signal.name must be None or a non-empty string")
        opts = dict(self.options or {})
        object.__setattr__(self, "kind", self.kind.strip())
        object.__setattr__(self, "name", self.name.strip() if isinstance(self.name, str) else self.name)
        object.__setattr__(self, "options", opts)

    @property
    def id(self) -> str:
        """Stable identifier used as the signal axis key."""
        return self.name or self.kind

    @classmethod
    def gradient(cls, *, name: Optional[str] = None) -> "Signal":
        """Raw gradient signal. Scope is controlled outside the signal."""
        return cls("gradient", name=name)

    @classmethod
    def activation(cls, *, token_selector: Optional[Any] = None, name: Optional[str] = None) -> "Signal":
        """Activation signal. Module/site scope is controlled outside the signal."""
        options: Dict[str, Any] = {}
        if token_selector is not None:
            options["token_selector"] = token_selector
        return cls("activation", name=name, options=options)

    @classmethod
    def activation_gradient(cls, *, token_selector: Optional[Any] = None, name: Optional[str] = None) -> "Signal":
        """Gradient with respect to activations. Module/site scope is controlled outside the signal."""
        options: Dict[str, Any] = {}
        if token_selector is not None:
            options["token_selector"] = token_selector
        return cls("activation_gradient", name=name, options=options)

    @classmethod
    def product(cls, left: str, right: str, *, name: Optional[str] = None) -> "Signal":
        """Derived product signal such as activation x activation-gradient."""
        if not left or not isinstance(left, str):
            raise ValueError("left must be a non-empty signal id")
        if not right or not isinstance(right, str):
            raise ValueError("right must be a non-empty signal id")
        return cls("product", name=name, options={"left": left, "right": right})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "options": dict(self.options or {}),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Signal":
        if not isinstance(data, Mapping):
            raise TypeError(f"Signal.from_dict expected mapping, got {type(data).__name__}")
        return cls(
            str(data["kind"]),
            name=data.get("name"),
            options=data.get("options") or {},
        )


class SignalSet:
    """Ordered collection of uniquely identified signals."""

    def __init__(self, *signals: Union[Signal, Iterable[Signal]]) -> None:
        if len(signals) == 1 and not isinstance(signals[0], Signal):
            maybe_iterable = signals[0]
            if isinstance(maybe_iterable, Iterable):
                signals = tuple(maybe_iterable)  # type: ignore[assignment]
        if not signals:
            raise ValueError("SignalSet requires at least one signal")

        ordered: Dict[str, Signal] = {}
        for signal in signals:
            if not isinstance(signal, Signal):
                raise TypeError(f"SignalSet entries must be Signal instances, got {type(signal).__name__}")
            if signal.id in ordered:
                raise ValueError(f"Duplicate signal id {signal.id!r}")
            ordered[signal.id] = signal
        self._signals = ordered

    @classmethod
    def gradient(cls) -> "SignalSet":
        """Default single raw-gradient signal set."""
        return cls(Signal.gradient())

    def __len__(self) -> int:
        return len(self._signals)

    def __iter__(self) -> Iterator[Signal]:
        return iter(self._signals.values())

    def __contains__(self, signal_id: object) -> bool:
        return signal_id in self._signals

    def __getitem__(self, signal_id: str) -> Signal:
        return self._signals[signal_id]

    @property
    def ids(self) -> Tuple[str, ...]:
        return tuple(self._signals.keys())

    @property
    def is_single(self) -> bool:
        return len(self._signals) == 1

    @property
    def single(self) -> Signal:
        if not self.is_single:
            raise ValueError(f"SignalSet contains {len(self)} signals, not one")
        return next(iter(self._signals.values()))

    def to_list(self) -> Sequence[Dict[str, Any]]:
        return [signal.to_dict() for signal in self]

    @classmethod
    def from_list(cls, data: Sequence[Mapping[str, Any]]) -> "SignalSet":
        return cls(Signal.from_dict(item) for item in data)


def coerce_signal(value: Any) -> Optional[Signal]:
    """Convert user-facing signal shorthand to a Signal instance."""
    if value is None:
        return None
    if isinstance(value, Signal):
        return value
    if isinstance(value, str):
        key = value.strip()
        if key == "gradient":
            return Signal.gradient()
        if key == "activation":
            return Signal.activation()
        if key == "activation_gradient":
            return Signal.activation_gradient()
        raise ValueError(f"Unsupported signal string {value!r}")
    if isinstance(value, Mapping):
        return Signal.from_dict(value)
    raise TypeError(f"signal must be Signal, str, dict, or None, got {type(value).__name__}")


def coerce_signal_set(value: Any) -> Optional[SignalSet]:
    """Convert user-facing signal-set shorthand to a SignalSet."""
    if value is None:
        return None
    if isinstance(value, SignalSet):
        return value
    if isinstance(value, Signal):
        return SignalSet(value)
    if isinstance(value, Mapping):
        return SignalSet(Signal.from_dict(value))
    if isinstance(value, (list, tuple)):
        return SignalSet(coerce_signal(v) for v in value)
    raise TypeError(
        f"signals must be SignalSet, Signal, sequence, dict, or None, got {type(value).__name__}"
    )


def normalize_signal_arguments(*, signal: Any = None, signals: Any = None) -> Tuple[Optional[Signal], SignalSet]:
    """
    Normalize the public ``signal`` / ``signals`` API pair.

    ``signal`` is kept convenient for the common single-signal case, while
    ``signals`` carries the full axis for future multi-signal training.
    """
    normalized_signal = coerce_signal(signal)
    normalized_signals = coerce_signal_set(signals)

    if normalized_signal is None and normalized_signals is None:
        normalized_signal = Signal.gradient()
        normalized_signals = SignalSet(normalized_signal)
    elif normalized_signal is not None and normalized_signals is None:
        normalized_signals = SignalSet(normalized_signal)
    elif normalized_signal is None and normalized_signals is not None:
        normalized_signal = normalized_signals.single if normalized_signals.is_single else None
    else:
        if normalized_signals.is_single and normalized_signals.single == normalized_signal:
            pass
        else:
            raise ValueError(
                "Pass either signal=... or signals=..., unless signals contains exactly the same single signal."
            )

    return normalized_signal, normalized_signals


def require_single_signal(
    *,
    signal: Any = None,
    signals: Any = None,
    context: str = "This path",
) -> Signal:
    """Normalize and validate that a path is operating on exactly one signal."""
    normalized_signal, normalized_signals = normalize_signal_arguments(signal=signal, signals=signals)
    if not normalized_signals.is_single:
        raise NotImplementedError(
            f"{context} currently supports exactly one signal; got signals={normalized_signals.ids!r}."
        )
    return normalized_signals.single


def require_single_gradient_signal(
    *,
    signal: Any = None,
    signals: Any = None,
    context: str = "This path",
) -> Signal:
    """
    Validate the current implementation path.

    Milestone 1 wires signal configuration into the existing raw-gradient
    training path. Activation and multi-signal extraction are represented in the
    API but are intentionally rejected here until their extractors are built.
    """
    normalized_signal, normalized_signals = normalize_signal_arguments(signal=signal, signals=signals)
    if not normalized_signals.is_single:
        raise NotImplementedError(
            f"{context} currently supports exactly one raw gradient signal; "
            f"got signals={normalized_signals.ids!r}."
        )
    selected = normalized_signals.single
    if selected.kind != "gradient":
        raise NotImplementedError(
            f"{context} currently supports only Signal.gradient() for training; "
            f"got Signal(kind={selected.kind!r}, id={selected.id!r})."
        )
    return selected


@dataclass(frozen=True)
class SignalScope:
    """
    Scope/eligibility metadata for signals.

    The first implementation stores metadata only; concrete scope resolution is
    handled by existing params/param_map behavior.
    """

    params: Optional[Tuple[str, ...]] = None
    activation_sites: Optional[Tuple[str, ...]] = None
    mode: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode is not None and self.mode not in {"default", "full"}:
            raise ValueError("SignalScope.mode must be one of None, 'default', or 'full'")

    @classmethod
    def from_values(
        cls,
        *,
        params: Optional[Union[str, Sequence[str]]] = None,
        activation_sites: Optional[Union[str, Sequence[str]]] = None,
        mode: Optional[str] = None,
    ) -> "SignalScope":
        return cls(
            params=_normalize_string_sequence(params, name="params"),
            activation_sites=_normalize_string_sequence(activation_sites, name="activation_sites"),
            mode=mode,
        )

    @classmethod
    def default(cls) -> "SignalScope":
        """Default involved-model scope: backbone/text tower, excluding prediction heads."""
        return cls(mode="default")

    @classmethod
    def full(cls) -> "SignalScope":
        """Full model scope, including prediction heads and non-text towers when present."""
        return cls(mode="full")

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "params": list(self.params) if self.params is not None else None,
            "activation_sites": list(self.activation_sites) if self.activation_sites is not None else None,
        }
        if self.mode is not None:
            data["mode"] = self.mode
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SignalScope":
        if not isinstance(data, Mapping):
            raise TypeError(f"SignalScope.from_dict expected mapping, got {type(data).__name__}")
        return cls.from_values(
            params=data.get("params"),
            activation_sites=data.get("activation_sites"),
            mode=data.get("mode"),
        )


def coerce_signal_scope(value: Any) -> Optional[SignalScope]:
    """Convert user-facing signal scope shorthand to a SignalScope."""
    if value is None:
        return None
    if isinstance(value, SignalScope):
        return value
    if isinstance(value, Mapping):
        return SignalScope.from_dict(value)
    raise TypeError(f"signal_scope must be SignalScope, dict, or None, got {type(value).__name__}")


@dataclass(frozen=True)
class SignalSpace:
    """
    Metadata for one flattened signal space.

    ``mapping`` is intentionally generic. Gradient signals can point to a
    param_map; activation signals can later point to activation-site metadata.
    """

    signal: Signal
    input_dim: int
    mapping: Optional[Any] = None
    scope: Optional[SignalScope] = None

    def __post_init__(self) -> None:
        if not isinstance(self.signal, Signal):
            raise TypeError(f"signal must be Signal, got {type(self.signal).__name__}")
        if not isinstance(self.input_dim, int) or self.input_dim <= 0:
            raise ValueError(f"input_dim must be a positive int, got {self.input_dim!r}")


@dataclass
class SignalBatch:
    """Factual/alternative/diff tensors for one signal extraction batch."""

    factual: Optional[torch.Tensor] = None
    alternative: Optional[torch.Tensor] = None
    diff: Optional[torch.Tensor] = None
    signal_id: str = "gradient"
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.signal_id, str) or not self.signal_id.strip():
            raise ValueError("signal_id must be a non-empty string")
        self.signal_id = self.signal_id.strip()

    @classmethod
    def from_factual_alternative(
        cls,
        factual: Optional[torch.Tensor],
        alternative: Optional[torch.Tensor],
        *,
        signal_id: str = "gradient",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SignalBatch":
        diff = None
        if factual is not None and alternative is not None:
            diff = factual - alternative
        return cls(
            factual=factual,
            alternative=alternative,
            diff=diff,
            signal_id=signal_id,
            metadata=metadata,
        )

    def select(self, key: Optional[str]) -> Optional[torch.Tensor]:
        """Return factual, alternative, diff, or None by source/target keyword."""
        if key is None:
            return None
        if key == "factual":
            return self.factual
        if key == "alternative":
            return self.alternative
        if key == "diff":
            return self.diff
        raise ValueError(f"Unknown signal selection {key!r}")


class GradientSignalExtractor:
    """
    Adapter around the existing gradient_creator callable.

    It is deliberately small: the current datasets keep calling gradient_creator
    directly, while future collection-aware datasets can use this wrapper to
    produce SignalBatch objects.
    """

    def __init__(self, gradient_creator: Callable[[Any], torch.Tensor], *, signal: Optional[Signal] = None) -> None:
        if not callable(gradient_creator):
            raise TypeError("gradient_creator must be callable")
        self.gradient_creator = gradient_creator
        self.signal = signal or Signal.gradient()
        self.signals = SignalSet(self.signal)
        if self.signal.kind != "gradient":
            raise ValueError(f"GradientSignalExtractor requires a gradient signal, got {self.signal.kind!r}")

    def exclusive_signal_access(self):
        model = getattr(self.gradient_creator, "__self__", None)
        if model is not None and hasattr(model, "exclusive_base_gradient_access"):
            return model.exclusive_base_gradient_access()
        return nullcontext()

    def __call__(
        self,
        factual_inputs: Optional[Any] = None,
        alternative_inputs: Optional[Any] = None,
        *,
        requires_factual: bool = True,
        requires_alternative: bool = True,
    ) -> SignalBatch:
        factual = self.gradient_creator(factual_inputs) if requires_factual else None
        alternative = self.gradient_creator(alternative_inputs) if requires_alternative else None
        return SignalBatch.from_factual_alternative(
            factual,
            alternative,
            signal_id=self.signal.id,
        )


class ActivationSignalExtractor:
    """Forward-hook extractor for single activation signals."""

    _NON_MODEL_INPUT_KEYS = frozenset({"prediction_mask"})

    def __init__(
        self,
        model: Any,
        *,
        signal: Optional[Signal] = None,
        scope: Optional[SignalScope] = None,
        tokenizer: Any = None,
        aggregate_batch: str = "mean",
    ) -> None:
        self.model = model
        self.base_model = getattr(model, "base_model", model)
        if not isinstance(self.base_model, nn.Module):
            raise TypeError("ActivationSignalExtractor requires a torch.nn.Module or object with .base_model")
        self.signal = signal or Signal.activation()
        self.signals = SignalSet(self.signal)
        if self.signal.kind != "activation":
            raise ValueError(f"ActivationSignalExtractor requires an activation signal, got {self.signal.kind!r}")
        self.scope = coerce_signal_scope(scope)
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        if aggregate_batch not in {"mean", "none"}:
            raise ValueError("aggregate_batch must be 'mean' or 'none'")
        self.aggregate_batch = aggregate_batch
        self._module_items = self._resolve_modules()

    def _resolve_modules(self) -> Tuple[Tuple[str, nn.Module], ...]:
        sites = tuple(self.scope.activation_sites or ()) if self.scope is not None else ()
        return resolve_activation_modules(self.base_model, sites, scope=self.scope)

    @classmethod
    def _static_module_output_dim(cls, module: nn.Module) -> Optional[int]:
        return infer_static_module_output_dim(module)

    def infer_input_dim_static(self) -> Optional[int]:
        """
        Infer activation signal dimension without a forward pass when reliable.

        Returns None when any selected site has unknown output width or when the
        token selector is callable and may change dimensionality.
        """
        selector = (self.signal.options or {}).get("token_selector")
        if callable(selector):
            return None
        total = 0
        for _name, module in self._module_items:
            dim = self._static_module_output_dim(module)
            if dim is None:
                return None
            total += int(dim)
        return total

    def exclusive_signal_access(self):
        if hasattr(self.model, "exclusive_base_gradient_access"):
            return self.model.exclusive_base_gradient_access()
        return nullcontext()

    @staticmethod
    def _first_tensor(value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value
        if isinstance(value, Mapping):
            for item in value.values():
                try:
                    return ActivationSignalExtractor._first_tensor(item)
                except TypeError:
                    continue
        if isinstance(value, (list, tuple)):
            for item in value:
                try:
                    return ActivationSignalExtractor._first_tensor(item)
                except TypeError:
                    continue
        raise TypeError(f"Hook output did not contain a tensor, got {type(value).__name__}")

    def _input_device(self) -> Optional[torch.device]:
        try:
            return next(self.base_model.parameters()).device
        except StopIteration:
            return None

    def _prepare_inputs(self, inputs: Any) -> Any:
        device = self._input_device()
        if torch.is_tensor(inputs):
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            if device is None:
                return inputs
            return inputs.to(device)
        if isinstance(inputs, Mapping):
            prepared = {}
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    tensor = v.unsqueeze(0) if v.dim() == 1 else v
                    prepared[k] = tensor.to(device) if device is not None else tensor
                else:
                    prepared[k] = v
            return prepared
        return inputs

    def _forward(self, inputs: Any) -> Any:
        prepared = self._prepare_inputs(inputs)
        model_inputs = prepared
        if isinstance(prepared, Mapping):
            model_inputs = {
                key: value
                for key, value in prepared.items()
                if key not in self._NON_MODEL_INPUT_KEYS
            }
        with torch.no_grad():
            if isinstance(model_inputs, Mapping):
                self.base_model(**model_inputs)
            else:
                self.base_model(model_inputs)
        return prepared

    def _select_tokens(self, activation: torch.Tensor, inputs: Any) -> torch.Tensor:
        selector = (self.signal.options or {}).get("token_selector")
        if callable(selector):
            return selector(activation, inputs)
        if activation.dim() < 3:
            if selector in {"cls", "mask"} or isinstance(selector, int):
                raise ValueError(f"token_selector={selector!r} requires activation shape (batch, seq, ...)")
            return activation
        if selector is None or selector in {"all", "mean"}:
            attention_mask = inputs.get("attention_mask") if isinstance(inputs, Mapping) else None
            if torch.is_tensor(attention_mask):
                weights = attention_mask.to(device=activation.device, dtype=activation.dtype)
                while weights.dim() < activation.dim():
                    weights = weights.unsqueeze(-1)
                denom = weights.sum(dim=1).clamp_min(1.0)
                return (activation * weights).sum(dim=1) / denom
            return activation.mean(dim=1)
        if selector == "cls":
            input_ids = inputs.get("input_ids") if isinstance(inputs, Mapping) else None
            cls_token_id = getattr(self.tokenizer, "cls_token_id", None)
            if torch.is_tensor(input_ids) and cls_token_id is not None:
                return self._mean_selected_tokens(
                    activation,
                    input_ids.eq(cls_token_id),
                    selector_name="cls",
                )
            return activation[:, 0, ...]
        if isinstance(selector, int):
            return activation[:, selector, ...]
        if selector == "mask":
            input_ids = inputs.get("input_ids") if isinstance(inputs, Mapping) else None
            if input_ids is None:
                raise ValueError("token_selector='mask' requires inputs with input_ids")
            mask_token_id = getattr(self.tokenizer, "mask_token_id", None)
            if mask_token_id is None:
                raise ValueError("token_selector='mask' requires a tokenizer with mask_token_id")
            input_ids = input_ids.to(device=activation.device)
            selected_positions = input_ids.eq(mask_token_id)
            prediction_mask = inputs.get("prediction_mask") if isinstance(inputs, Mapping) else None
            if (
                torch.is_tensor(prediction_mask)
                and not bool(selected_positions.any(dim=1).all().item())
            ):
                selected_positions = prediction_mask.to(device=activation.device, dtype=torch.bool)
            return self._mean_selected_tokens(
                activation,
                selected_positions,
                selector_name="mask",
            )
        if selector == "prediction":
            prediction_mask = inputs.get("prediction_mask") if isinstance(inputs, Mapping) else None
            if not torch.is_tensor(prediction_mask):
                raise ValueError("token_selector='prediction' requires inputs with prediction_mask")
            return self._mean_selected_tokens(
                activation,
                prediction_mask,
                selector_name="prediction",
            )
        raise ValueError(f"Unsupported activation token_selector {selector!r}")

    @staticmethod
    def _mean_selected_tokens(
        activation: torch.Tensor,
        selected_positions: torch.Tensor,
        *,
        selector_name: str,
    ) -> torch.Tensor:
        selected_positions = selected_positions.to(device=activation.device, dtype=torch.bool)
        if selected_positions.shape[:2] != activation.shape[:2]:
            raise ValueError(
                f"token_selector={selector_name!r} position mask shape {tuple(selected_positions.shape)} "
                f"does not match activation prefix {tuple(activation.shape[:2])}"
            )
        if not bool(selected_positions.any(dim=1).all().item()):
            raise ValueError(f"token_selector={selector_name!r} requires at least one selected token in every row")
        weights = selected_positions.to(dtype=activation.dtype)
        while weights.dim() < activation.dim():
            weights = weights.unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (activation * weights).sum(dim=1) / denom

    def _flatten_site(self, activation: torch.Tensor, inputs: Any) -> torch.Tensor:
        selected = self._select_tokens(activation, inputs)
        if selected.dim() == 0:
            selected = selected.reshape(1, 1)
        elif selected.dim() == 1:
            selected = selected.reshape(1, -1)
        else:
            selected = selected.reshape(selected.shape[0], -1)
        if self.aggregate_batch == "mean" and selected.shape[0] > 1:
            selected = selected.mean(dim=0, keepdim=True)
        return selected

    def _extract(self, inputs: Any) -> torch.Tensor:
        captured: Dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(name: str):
            def hook(_module: nn.Module, _args: Tuple[Any, ...], output: Any) -> None:
                captured[name] = self._first_tensor(output).detach()

            return hook

        try:
            for name, module in self._module_items:
                handles.append(module.register_forward_hook(make_hook(name)))
            prepared_inputs = self._forward(inputs)
        finally:
            for handle in handles:
                handle.remove()
        missing = [name for name, _module in self._module_items if name not in captured]
        if missing:
            raise RuntimeError(f"Activation hooks did not capture expected sites: {missing!r}")
        flattened = [self._flatten_site(captured[name], prepared_inputs) for name, _module in self._module_items]
        signal = torch.cat(flattened, dim=-1)
        if self.aggregate_batch == "mean":
            return signal.squeeze(0)
        return signal

    def infer_input_dim(self, sample_inputs: Any) -> int:
        """Infer flattened activation signal dimensionality from representative inputs."""
        signal = self._extract(sample_inputs)
        if signal.dim() == 0:
            return 1
        return int(signal.shape[-1])

    def __call__(
        self,
        factual_inputs: Optional[Any] = None,
        alternative_inputs: Optional[Any] = None,
        *,
        requires_factual: bool = True,
        requires_alternative: bool = True,
    ) -> SignalBatch:
        factual = self._extract(factual_inputs) if requires_factual else None
        alternative = self._extract(alternative_inputs) if requires_alternative else None
        return SignalBatch.from_factual_alternative(
            factual,
            alternative,
            signal_id=self.signal.id,
        )


__all__ = [
    "Signal",
    "SignalSet",
    "coerce_signal",
    "coerce_signal_set",
    "coerce_signal_scope",
    "normalize_signal_arguments",
    "require_single_signal",
    "require_single_gradient_signal",
    "SignalScope",
    "SignalSpace",
    "SignalBatch",
    "GradientSignalExtractor",
    "ActivationSignalExtractor",
]
