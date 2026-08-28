"""Signal abstractions for GRADIEND training inputs."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from gradiend.signal_space import infer_static_module_output_dim, resolve_activation_modules


SignalKind = str


class ActivationRunningRms:
    """Online RMS scale for activation sites (O(n_sites) floats only).

    Never stores activation tensors — only running mean-of-squares (or a freeze
    after the first batch when ``momentum == 0``).
    """

    def __init__(
        self,
        *,
        reduce: str = "per_site",
        momentum: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        reduce = str(reduce or "per_site").strip().lower()
        if reduce not in {"per_site", "global"}:
            raise ValueError(f"reduce must be 'per_site' or 'global', got {reduce!r}")
        self.reduce = reduce
        self.momentum = float(momentum)
        self.eps = float(eps)
        self._ms: Optional[torch.Tensor] = None  # mean of squares per site
        self._n_updates = 0
        self._frozen = False

    @property
    def n_sites(self) -> int:
        return 0 if self._ms is None else int(self._ms.numel())

    def state_dict(self) -> Dict[str, Any]:
        return {
            "reduce": self.reduce,
            "momentum": self.momentum,
            "eps": self.eps,
            "ms": None if self._ms is None else self._ms.detach().cpu().tolist(),
            "n_updates": int(self._n_updates),
            "frozen": bool(self._frozen),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.reduce = str(state.get("reduce") or self.reduce)
        self.momentum = float(state.get("momentum", self.momentum))
        self.eps = float(state.get("eps", self.eps))
        ms = state.get("ms")
        if ms is None:
            self._ms = None
        else:
            self._ms = torch.tensor(list(ms), dtype=torch.float32)
        self._n_updates = int(state.get("n_updates") or 0)
        self._frozen = bool(state.get("frozen", False))

    def rms(self) -> Optional[torch.Tensor]:
        if self._ms is None:
            return None
        return torch.sqrt(self._ms.clamp_min(self.eps))

    def update_and_scale(self, site_tensors: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """Update running stats from ``site_tensors`` and return scaled copies."""
        if not site_tensors:
            return []
        device = site_tensors[0].device
        dtype = site_tensors[0].dtype
        # Per-site mean square over all non-batch dims (batch × hidden → scalar).
        stats = []
        for t in site_tensors:
            flat = t.detach().float().reshape(-1)
            stats.append(flat.pow(2).mean() if flat.numel() else torch.zeros((), device=device))
        batch_ms = torch.stack(stats).to(device=device)
        if self.reduce == "global":
            g = batch_ms.mean().expand_as(batch_ms)
            batch_ms = g

        if self._ms is None:
            self._ms = batch_ms.detach().float().cpu()
            self._n_updates = 1
            if self.momentum <= 0.0:
                self._frozen = True
        elif not self._frozen:
            cur = batch_ms.detach().float().cpu()
            if self._ms.numel() != cur.numel():
                raise ValueError(
                    f"ActivationRunningRms site count changed: had {self._ms.numel()}, got {cur.numel()}"
                )
            if self.momentum <= 0.0:
                # Already seeded on first batch; freeze.
                self._frozen = True
            else:
                m = float(self.momentum)
                self._ms.mul_(m).add_(cur, alpha=1.0 - m)
                self._n_updates += 1

        scales = torch.sqrt(self._ms.clamp_min(self.eps)).to(device=device, dtype=dtype)
        out: List[torch.Tensor] = []
        for i, t in enumerate(site_tensors):
            s = scales[i] if self.reduce == "per_site" else scales[0]
            out.append(t / s)
        return out

    def unscale_flat(self, flat: torch.Tensor, site_widths: Sequence[int]) -> torch.Tensor:
        """Map a scaled concat vector back to raw activation units (for intervene)."""
        rms = self.rms()
        if rms is None:
            return flat
        if sum(int(w) for w in site_widths) != int(flat.numel()):
            raise ValueError(
                f"unscale_flat width mismatch: flat={flat.numel()}, sites={list(site_widths)}"
            )
        pieces = []
        idx = 0
        rms_d = rms.to(device=flat.device, dtype=flat.dtype)
        for i, w in enumerate(site_widths):
            w = int(w)
            s = rms_d[i] if self.reduce == "per_site" else rms_d[0]
            pieces.append(flat[idx : idx + w] * s)
            idx += w
        return torch.cat(pieces, dim=0)


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
    def activation(
        cls,
        *,
        token_selector: Optional[Any] = None,
        target_token_selector: Optional[Any] = None,
        scale: Optional[str] = None,
        scale_reduce: str = "per_site",
        scale_momentum: float = 0.0,
        scale_eps: float = 1e-6,
        name: Optional[str] = None,
    ) -> "Signal":
        """Activation signal. Module/site scope is controlled outside the signal.

        ``token_selector`` is the encoder/source gather. ``target_token_selector``
        is the decoder/target gather when it differs (mixed-site ACTIEND: source
        ``pre_prediction``, target ``prediction``). Omitted or equal selectors
        keep same-site source and target.

        ``scale`` (optional):

          - ``None`` / omitted: raw activations (historical default)
          - ``"running_rms"``: divide each site (or the concat) by a running RMS

            estimated online from extracted activations. Only O(n_sites) floats
            of state — never buffers activations. Use for CAA/SAE-comparable
            activation magnitude before the ACTIEND autoencoder.
        ``scale_reduce``: ``"per_site"`` (default) or ``"global"``.
        ``scale_momentum``: ``0`` freezes after the first batch; ``(0,1)`` is EMA.
        """
        options: Dict[str, Any] = {}
        if token_selector is not None:
            options["token_selector"] = token_selector
        if target_token_selector is not None and target_token_selector != token_selector:
            options["target_token_selector"] = target_token_selector
        if scale is not None:
            scale_s = str(scale).strip().lower()
            if scale_s in {"", "none", "off", "raw"}:
                pass
            elif scale_s in {"running_rms", "rms"}:
                options["scale"] = "running_rms"
                reduce = str(scale_reduce or "per_site").strip().lower()
                if reduce not in {"per_site", "global"}:
                    raise ValueError(
                        f"Signal.activation scale_reduce must be 'per_site' or 'global', got {scale_reduce!r}"
                    )
                options["scale_reduce"] = reduce
                options["scale_momentum"] = float(scale_momentum)
                options["scale_eps"] = float(scale_eps)
            else:
                raise ValueError(
                    f"Signal.activation scale must be None or 'running_rms', got {scale!r}"
                )
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
    activation_selector: Optional[Tuple[Any, ...]] = None

    def __post_init__(self) -> None:
        if self.mode is not None and self.mode not in {"default", "full"}:
            raise ValueError("SignalScope.mode must be one of None, 'default', or 'full'")
        if self.activation_sites is not None and self.activation_selector is not None:
            raise ValueError("Use either activation_sites or a semantic SignalScope shortcut, not both")
        if self.activation_selector is not None and not self.activation_selector:
            raise ValueError("SignalScope.activation_selector must be non-empty")

    @classmethod
    def from_values(
        cls,
        *,
        params: Optional[Union[str, Sequence[str]]] = None,
        activation_sites: Optional[Union[str, Sequence[str]]] = None,
        mode: Optional[str] = None,
        activation_selector: Optional[Tuple[Any, ...]] = None,
    ) -> "SignalScope":
        return cls(
            params=_normalize_string_sequence(params, name="params"),
            activation_sites=_normalize_string_sequence(activation_sites, name="activation_sites"),
            mode=mode,
            activation_selector=activation_selector,
        )

    @classmethod
    def default(cls) -> "SignalScope":
        """Default involved-model scope: backbone/text tower, excluding prediction heads."""
        return cls(mode="default")

    @classmethod
    def full(cls) -> "SignalScope":
        """Full model scope, including prediction heads and non-text towers when present."""
        return cls(mode="full")

    @classmethod
    def layers(cls, *layers: Union[int, str]) -> "SignalScope":
        """
        Activation scope over transformer-layer outputs.

        ``SignalScope.layers()`` and ``SignalScope.layers("*")`` both mean all
        transformer layers. This selects one concatenated activation space; it
        does not imply one GRADIEND per layer.
        """
        if not layers or layers == ("*",):
            selected: Optional[Tuple[int, ...]] = None
        else:
            values = []
            for layer in layers:
                if isinstance(layer, str):
                    if layer == "*":
                        if len(layers) != 1:
                            raise ValueError("SignalScope.layers('*') cannot be combined with explicit indices")
                        selected = None
                        break
                    if not layer.isdigit():
                        raise ValueError("SignalScope.layers entries must be integer layer indices or '*'")
                    value = int(layer)
                elif isinstance(layer, int):
                    value = layer
                else:
                    raise TypeError("SignalScope.layers entries must be integer layer indices or '*'")
                if value < 0:
                    raise ValueError("SignalScope.layers indices must be non-negative")
                values.append(value)
            else:
                selected = tuple(values)
        return cls(activation_selector=("layers", selected))

    @classmethod
    def layer(cls, layer: int) -> "SignalScope":
        """Activation scope over one transformer-layer output."""
        return cls.layers(layer)

    @classmethod
    def embeddings(cls) -> "SignalScope":
        """Activation scope over the model's combined embedding stream when available."""
        return cls(activation_selector=("embeddings",))

    @classmethod
    def word_embedding(cls) -> "SignalScope":
        """Activation scope over the token embedding lookup module."""
        return cls(activation_selector=("word_embedding",))

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "params": list(self.params) if self.params is not None else None,
            "activation_sites": list(self.activation_sites) if self.activation_sites is not None else None,
        }
        if self.mode is not None:
            data["mode"] = self.mode
        if self.activation_selector is not None:
            kind = str(self.activation_selector[0])
            selector_data: Dict[str, Any] = {"kind": kind}
            if kind == "layers":
                layers = self.activation_selector[1]
                selector_data["layers"] = None if layers is None else list(layers)
            data["activation_selector"] = selector_data
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SignalScope":
        if not isinstance(data, Mapping):
            raise TypeError(f"SignalScope.from_dict expected mapping, got {type(data).__name__}")
        activation_selector = None
        raw_selector = data.get("activation_selector")
        if raw_selector is not None:
            if not isinstance(raw_selector, Mapping):
                raise TypeError("activation_selector must be a mapping")
            kind = raw_selector.get("kind")
            if kind == "layers":
                raw_layers = raw_selector.get("layers")
                layers = None if raw_layers is None else tuple(int(item) for item in raw_layers)
                activation_selector = ("layers", layers)
            elif kind in {"embeddings", "word_embedding"}:
                activation_selector = (kind,)
            else:
                raise ValueError(f"Unknown activation_selector kind: {kind!r}")
        return cls.from_values(
            params=data.get("params"),
            activation_sites=data.get("activation_sites"),
            mode=data.get("mode"),
            activation_selector=activation_selector,
        )

    @classmethod
    def from_signal_space(cls, signal_space: Mapping[str, Any]) -> "SignalScope":
        """Build explicit activation sites from a saved activation ``signal_space``."""
        if not isinstance(signal_space, Mapping):
            raise TypeError(
                f"SignalScope.from_signal_space expected mapping, got {type(signal_space).__name__}"
            )
        kind = str(signal_space.get("kind") or "").strip().lower()
        if kind and kind != "activation":
            raise ValueError(
                f"SignalScope.from_signal_space expects activation signal_space, got {kind!r}"
            )
        sites = []
        for entry in signal_space.get("mapping") or ():
            if not isinstance(entry, Mapping):
                continue
            name = entry.get("name")
            if not name:
                continue
            site = str(name)
            if site.startswith("activation:"):
                site = site[len("activation:") :]
            sites.append(site)
        if not sites:
            return cls.default()
        return cls.from_values(activation_sites=tuple(sites))


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
    """Factual/alternative/diff tensors for one signal extraction batch.

    Same-site training uses ``factual`` / ``alternative`` for both encoder source
    and decoder target. Mixed-site ACTIEND fills ``factual_target`` /
    ``alternative_target`` with a second gather from the same forward
    (e.g. source ``pre_prediction``, target ``prediction``).
    """

    factual: Optional[torch.Tensor] = None
    alternative: Optional[torch.Tensor] = None
    diff: Optional[torch.Tensor] = None
    factual_target: Optional[torch.Tensor] = None
    alternative_target: Optional[torch.Tensor] = None
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
        factual_target: Optional[torch.Tensor] = None,
        alternative_target: Optional[torch.Tensor] = None,
    ) -> "SignalBatch":
        diff = None
        if factual is not None and alternative is not None:
            diff = factual - alternative
        return cls(
            factual=factual,
            alternative=alternative,
            diff=diff,
            factual_target=factual_target,
            alternative_target=alternative_target,
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
        self._activation_width_validated = False
        opts = self.signal.options or {}
        self._scale: Optional[ActivationRunningRms] = None
        if str(opts.get("scale") or "").lower() in {"running_rms", "rms"}:
            self._scale = ActivationRunningRms(
                reduce=str(opts.get("scale_reduce") or "per_site"),
                momentum=float(opts.get("scale_momentum") or 0.0),
                eps=float(opts.get("scale_eps") or 1e-6),
            )
            # Restore frozen stats from a loaded checkpoint when present.
            gradiend = getattr(self.model, "gradiend", None)
            kwargs = dict(getattr(gradiend, "kwargs", None) or {}) if gradiend is not None else {}
            saved = kwargs.get("activation_scale") or (kwargs.get("signal_space") or {}).get(
                "activation_scale"
            )
            if isinstance(saved, Mapping):
                self._scale.load_state_dict(saved)

    def _resolve_modules(self) -> Tuple[Tuple[str, nn.Module], ...]:
        sites = tuple(self.scope.activation_sites or ()) if self.scope is not None else ()
        return resolve_activation_modules(self.base_model, sites, scope=self.scope)

    @classmethod
    def _static_module_output_dim(cls, module: nn.Module) -> Optional[int]:
        return infer_static_module_output_dim(module)

    @staticmethod
    def _positive_int(value: Any) -> Optional[int]:
        try:
            out = int(value)
        except (TypeError, ValueError):
            return None
        return out if out > 0 else None

    def _expected_input_dim(self) -> Tuple[Optional[int], str]:
        gradiend = getattr(self.model, "gradiend", None)
        configured = self._positive_int(getattr(gradiend, "input_dim", None))
        if configured is not None:
            return configured, "configured encoder input_dim"
        inferred = self.infer_input_dim_static()
        if inferred is not None:
            return int(inferred), "static activation-site width inference"
        return None, "unknown"

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

    def _is_mixed_site(self) -> bool:
        opts = self.signal.options or {}
        target = opts.get("target_token_selector")
        return target is not None and target != opts.get("token_selector")

    def _select_tokens(
        self,
        activation: torch.Tensor,
        inputs: Any,
        *,
        selector: Optional[Any] = None,
    ) -> torch.Tensor:
        if selector is None:
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
        if selector in {"pre_prediction", "unfilled_prediction"}:
            # One token before the first prediction-span position (context site).
            # On decoder-only causal LMs this cannot attend to the fill.
            prediction_mask = inputs.get("prediction_mask") if isinstance(inputs, Mapping) else None
            if not torch.is_tensor(prediction_mask):
                raise ValueError(
                    f"token_selector={selector!r} requires inputs with prediction_mask"
                )
            return self._gather_pre_prediction_tokens(
                activation,
                prediction_mask,
                selector_name=str(selector),
            )
        raise ValueError(f"Unsupported activation token_selector {selector!r}")

    @staticmethod
    def _gather_pre_prediction_tokens(
        activation: torch.Tensor,
        prediction_mask: torch.Tensor,
        *,
        selector_name: str,
    ) -> torch.Tensor:
        """Gather residual at index = first prediction_mask True − 1 (clamped)."""
        mask = prediction_mask.to(device=activation.device, dtype=torch.bool)
        if mask.shape[:2] != activation.shape[:2]:
            raise ValueError(
                f"token_selector={selector_name!r} position mask shape {tuple(mask.shape)} "
                f"does not match activation prefix {tuple(activation.shape[:2])}"
            )
        if not bool(mask.any(dim=1).all().item()):
            raise ValueError(
                f"token_selector={selector_name!r} requires at least one selected token in every row"
            )
        # First True along seq; then step one token left (context / pre-fill site).
        first = mask.to(dtype=torch.long).argmax(dim=1)
        pos = (first - 1).clamp(min=0)
        b, _, d = activation.shape[0], activation.shape[1], activation.shape[-1]
        idx = pos.view(b, 1, 1).expand(b, 1, d)
        return activation.gather(1, idx).squeeze(1)

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

    def _flatten_site(
        self,
        activation: torch.Tensor,
        inputs: Any,
        *,
        selector: Optional[Any] = None,
    ) -> torch.Tensor:
        selected = self._select_tokens(activation, inputs, selector=selector)
        if selected.dim() == 0:
            selected = selected.reshape(1, 1)
        elif selected.dim() == 1:
            selected = selected.reshape(1, -1)
        else:
            selected = selected.reshape(selected.shape[0], -1)
        if self.aggregate_batch == "mean" and selected.shape[0] > 1:
            selected = selected.mean(dim=0, keepdim=True)
        return selected

    def _validate_extracted_width_once(
        self,
        *,
        signal: torch.Tensor,
        captured: Dict[str, torch.Tensor],
        flattened: Sequence[torch.Tensor],
    ) -> None:
        if self._activation_width_validated:
            return
        expected, expected_source = self._expected_input_dim()
        if expected is None:
            self._activation_width_validated = True
            return
        actual = int(signal.shape[-1]) if signal.dim() > 0 else 1
        if actual == int(expected):
            self._activation_width_validated = True
            return

        selector = (self.signal.options or {}).get("token_selector")
        site_details = []
        for (name, module), flat in zip(self._module_items, flattened):
            static_dim = self._static_module_output_dim(module)
            site_details.append(
                f"{name} ({module.__class__.__name__}): "
                f"captured_shape={tuple(captured[name].shape)}, "
                f"selected_shape={tuple(flat.shape)}, "
                f"static_width={static_dim if static_dim is not None else 'unknown'}"
            )
        resolved_sites = tuple(name for name, _module in self._module_items)
        raise ValueError(
            f"Activation signal width mismatch: expected {int(expected)} dimension(s) from "
            f"{expected_source}, but the first extracted signal has width {actual}. "
            f"Resolved activation_sites={resolved_sites!r}, token_selector={selector!r}. "
            f"Per-site widths: {'; '.join(site_details)}. "
            "This usually means static module-width inference was wrong, the token selector changes "
            "the activation dimensionality, or the loaded checkpoint does not match "
            "the current base model and signal_scope. Choose an activation site with a stable 1D "
            "hidden width, pass signal_scope=SignalScope.from_values(activation_sites=[...]), or add "
            "explicit static-width support for this module type."
        )

    def _capture(self, inputs: Any) -> Tuple[Dict[str, torch.Tensor], Any]:
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
        return captured, prepared_inputs

    def _cat_flattened(
        self,
        captured: Mapping[str, torch.Tensor],
        prepared_inputs: Any,
        *,
        selector: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        flattened = [
            self._flatten_site(captured[name], prepared_inputs, selector=selector)
            for name, _module in self._module_items
        ]
        if self._scale is not None:
            flattened = self._scale.update_and_scale(flattened)
            self._persist_scale_state()
        signal = torch.cat(flattened, dim=-1)
        if self.aggregate_batch == "mean":
            signal = signal.squeeze(0)
        return signal, flattened

    def _persist_scale_state(self) -> None:
        """Write O(n_sites) RMS stats onto gradiend.kwargs for checkpointing."""
        if self._scale is None:
            return
        gradiend = getattr(self.model, "gradiend", None)
        if gradiend is None:
            return
        kwargs = dict(getattr(gradiend, "kwargs", None) or {})
        kwargs["activation_scale"] = self._scale.state_dict()
        signal_space = dict(kwargs.get("signal_space") or {})
        signal_space["activation_scale"] = kwargs["activation_scale"]
        kwargs["signal_space"] = signal_space
        gradiend.kwargs = kwargs

    def scale_state_dict(self) -> Optional[Dict[str, Any]]:
        return None if self._scale is None else self._scale.state_dict()

    def unscale_steering_vector(self, flat: torch.Tensor) -> torch.Tensor:
        """Undo running-RMS scale so ACTIEND hooks write in raw activation units."""
        if self._scale is None:
            return flat
        widths = []
        for _name, module in self._module_items:
            dim = self._static_module_output_dim(module)
            if dim is None:
                raise ValueError(
                    "Cannot unscale ACTIEND steering without static per-site widths"
                )
            widths.append(int(dim))
        return self._scale.unscale_flat(flat, widths)

    def _extract_source_and_target(
        self,
        inputs: Any,
        *,
        validate_width: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        captured, prepared_inputs = self._capture(inputs)
        source_sel = (self.signal.options or {}).get("token_selector")
        source, source_flat = self._cat_flattened(
            captured, prepared_inputs, selector=source_sel
        )
        if validate_width:
            self._validate_extracted_width_once(
                signal=source,
                captured=captured,
                flattened=source_flat,
            )
        if not self._is_mixed_site():
            return source, None
        target_sel = (self.signal.options or {}).get("target_token_selector")
        target, _target_flat = self._cat_flattened(
            captured, prepared_inputs, selector=target_sel
        )
        return source, target

    def _extract(self, inputs: Any, *, validate_width: bool = True) -> torch.Tensor:
        source, _target = self._extract_source_and_target(
            inputs, validate_width=validate_width
        )
        return source

    def infer_input_dim(self, sample_inputs: Any) -> int:
        """Infer flattened activation signal dimensionality from representative inputs."""
        signal = self._extract(sample_inputs, validate_width=False)
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
        factual = factual_target = None
        alternative = alternative_target = None
        if requires_factual:
            factual, factual_target = self._extract_source_and_target(factual_inputs)
        if requires_alternative:
            alternative, alternative_target = self._extract_source_and_target(
                alternative_inputs
            )
        return SignalBatch.from_factual_alternative(
            factual,
            alternative,
            factual_target=factual_target,
            alternative_target=alternative_target,
            signal_id=self.signal.id,
        )


__all__ = [
    "Signal",
    "SignalSet",
    "ActivationRunningRms",
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
