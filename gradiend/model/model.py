"""
GRADIEND core model definitions (weights-only).

This module defines:

- GradiendModel: weights-only GRADIEND encoder/decoder (no base-model context).

For the parameter mapping-aware variant, see gradiend.model.param_mapped.ParamMappedGradiendModel.
For the variant in combination of a base model, see gradiend.model.model_with_gradiend.ModelWithGradiend.
"""

import copy
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from gradiend.model.layers import LargeLinear
from gradiend.model.utils import get_activation
from gradiend.util.logging import get_logger
from gradiend.util import convert_tuple_keys_recursively

logger = get_logger(__name__)

DEFAULT_INIT_FAN_IN_FLOOR = 10_000
"""Default fan-in floor for fresh GRADIEND-family initialization.

Empirically, ACTIEND components over activation vectors can be much smaller than
classic GRADIEND parameter spaces (for example, a single transformer hidden
state vs millions of model-weight gradients). A raw ``1/sqrt(n)`` init then
starts those activation components with much larger weights. The floor keeps
small signal components on a conservative random scale while leaving larger
GRADIEND spaces unchanged.
"""


def _coerce_init_fan_in_floor(value: Optional[int]) -> Optional[int]:
    """Validate the optional fan-in floor used for fresh GRADIEND initialization."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            "init_fan_in_floor must be a positive int or None, "
            f"got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"init_fan_in_floor must be >= 1 or None, got {value}")
    return int(value)


def gradiend_signal_kind_from_metadata(metadata: Optional[Dict[str, Any]]) -> str:
    """Resolve the persisted signal kind for a GRADIEND-family model."""
    if not isinstance(metadata, dict):
        return "gradient"
    signal_space = metadata.get("signal_space")
    if isinstance(signal_space, dict) and signal_space.get("kind"):
        return str(signal_space["kind"]).strip().lower()
    mapping_kind = metadata.get("mapping_kind")
    if isinstance(mapping_kind, str) and mapping_kind.strip():
        return mapping_kind.strip().lower()
    signal = metadata.get("signal")
    if isinstance(signal, dict) and signal.get("kind"):
        return str(signal["kind"]).strip().lower()
    kind = getattr(signal, "kind", None)
    if isinstance(kind, str) and kind.strip():
        return kind.strip().lower()
    return "gradient"


def gradiend_method_name_from_signal_kind(kind: str) -> str:
    """Return the human-facing method name for a signal kind."""
    normalized = str(kind or "gradient").strip().lower()
    if normalized == "activation":
        return "ACTIEND"
    if normalized == "activation_gradient":
        return "activation-gradient GRADIEND"
    if normalized == "gradient":
        return "GRADIEND"
    return f"{normalized} GRADIEND"


@dataclass(frozen=True)
class GradiendComponent:
    """A non-overlapping contiguous view into the GRADIEND input space."""

    id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("GradiendComponent.id must be a non-empty string")
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError("GradiendComponent.start/end must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("GradiendComponent requires 0 <= start < end")

    @property
    def input_dim(self) -> int:
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GradiendComponent":
        return cls(id=str(data["id"]), start=int(data["start"]), end=int(data["end"]))


def _normalize_component_slices(
    component_slices: Optional[List[Union[GradiendComponent, Dict[str, Any]]]],
    *,
    input_dim: int,
) -> Tuple[GradiendComponent, ...]:
    if component_slices is None:
        return (GradiendComponent("full", 0, int(input_dim)),)
    components = tuple(
        item if isinstance(item, GradiendComponent) else GradiendComponent.from_dict(dict(item))
        for item in component_slices
    )
    if not components:
        raise ValueError("component_slices must contain at least one component")
    seen_ids = set()
    occupied = torch.zeros(int(input_dim), dtype=torch.bool)
    for component in components:
        if component.id in seen_ids:
            raise ValueError(f"Duplicate component id: {component.id!r}")
        seen_ids.add(component.id)
        if component.end > input_dim:
            raise ValueError(
                f"Component {component.id!r} ends at {component.end}, beyond input_dim={input_dim}"
            )
        if occupied[component.start:component.end].any().item():
            raise ValueError(f"Component {component.id!r} overlaps another component")
        occupied[component.start:component.end] = True
    return components


def _is_full_component_config(component_slices: Any, input_dim: int) -> bool:
    """Return True for legacy serialized full-component metadata."""
    if not isinstance(component_slices, list) or len(component_slices) != 1:
        return False
    item = component_slices[0]
    if not isinstance(item, dict):
        return False
    return (
        str(item.get("id")) == "full"
        and int(item.get("start", -1)) == 0
        and int(item.get("end", -1)) == int(input_dim)
    )


def _is_no_component_split_mode(component_split_mode: Any) -> bool:
    return component_split_mode is None or (
        isinstance(component_split_mode, str) and component_split_mode.strip().lower() == "none"
    )


def _coerce_saved_component_slices(
    component_slices: Any,
    component_split_mode: Any,
    input_dim: int,
) -> Any:
    """
    Normalize component metadata loaded from config.json.

    Pre-pruned checkpoints written before lazy-init pruning remapped components may
    store a stale full-space ``end`` while ``architecture.input_dim`` is already pruned.
    """
    if not _is_no_component_split_mode(component_split_mode):
        return component_slices
    if _is_full_component_config(component_slices, input_dim):
        return None
    if (
        isinstance(component_slices, list)
        and len(component_slices) == 1
        and isinstance(component_slices[0], dict)
        and str(component_slices[0].get("id")) == "full"
        and int(component_slices[0].get("end", -1)) > int(input_dim)
    ):
        return None
    return component_slices


class _ComponentAccessor:
    def __init__(self, model: "GradiendModel", kind: str) -> None:
        self._model = model
        self._kind = kind

    def __len__(self) -> int:
        return len(self._model._virtual_component_slices)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, key: Union[int, str]):
        component = self._model._component_by_key(key)
        if self._kind == "encoder":
            return _VirtualComponentEncoder(self._model, component)
        if self._kind == "decoder":
            return _VirtualComponentDecoder(self._model, component)
        raise ValueError(f"Unknown component accessor kind: {self._kind!r}")


class _VirtualComponentEncoder:
    def __init__(self, model: "GradiendModel", component: GradiendComponent) -> None:
        self.model = model
        self.component = component

    @property
    def weight(self) -> torch.Tensor:
        self.model._require_built()
        return self.model.encoder[0].linear.weight[:, self.component.start:self.component.end]

    @property
    def bias(self) -> Optional[torch.Tensor]:
        self.model._require_built()
        return self.model.encoder[0].linear.bias

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 0:
            raise ValueError("Component encoder input must not be scalar")
        if x.shape[-1] == self.model.input_dim:
            x = x[..., self.component.start:self.component.end]
        elif x.shape[-1] != self.component.input_dim:
            raise ValueError(
                f"Component {self.component.id!r} expected final dimension "
                f"{self.component.input_dim} or full dimension {self.model.input_dim}, got {x.shape[-1]}"
            )
        x = x.to(dtype=self.model.torch_dtype, device=self.model.device_encoder)
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.model._require_built()
        x = self._prepare_input(x)
        encoded = F.linear(x, self.weight, self.bias)
        return self.model.encoder[1](encoded)


class _VirtualComponentDecoder:
    def __init__(self, model: "GradiendModel", component: GradiendComponent) -> None:
        self.model = model
        self.component = component

    @property
    def weight(self) -> torch.Tensor:
        self.model._require_built()
        return self.model.decoder[0].linear.weight[self.component.start:self.component.end, :]

    @property
    def bias(self) -> Optional[torch.Tensor]:
        self.model._require_built()
        bias = self.model.decoder[0].linear.bias
        if bias is None:
            return None
        return bias[self.component.start:self.component.end]

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        self.model._require_built()
        if z.dim() == 0:
            raise ValueError("Component decoder input must not be scalar")
        if z.shape[-1] != self.model.latent_dim:
            raise ValueError(
                f"Component decoder expected final dimension {self.model.latent_dim}, got {z.shape[-1]}"
            )
        z = z.to(dtype=self.model.torch_dtype, device=self.model.device_decoder)
        decoded = F.linear(z, self.weight, self.bias)
        return self.model.decoder[1](decoded)


class _VirtualComponent:
    def __init__(self, model: "GradiendModel", component: GradiendComponent) -> None:
        self.model = model
        self.component = component
        self.encoder = _VirtualComponentEncoder(model, component)
        self.decoder = _VirtualComponentDecoder(model, component)

    def invert_encoding(self) -> None:
        """Invert this component's scalar orientation without exposing component plumbing publicly."""
        self.model._require_built()
        if self.model.latent_dim != 1:
            raise ValueError("Component normalization currently requires latent_dim=1")
        n_virtual = len(self.model._virtual_component_slices)
        if self.model.bias_encoder and n_virtual > 1:
            raise ValueError(
                "Per-component normalization with shared encoder bias is undefined; use bias_encoder=False "
                "or a single explicit component."
            )
        with torch.no_grad():
            self.encoder.weight.mul_(-1)
            if self.encoder.bias is not None and n_virtual == 1:
                self.encoder.bias.mul_(-1)
            self.decoder.weight.mul_(-1)
            if self.decoder.bias is not None:
                self.decoder.bias.mul_(-1)


class GradiendModel(nn.Module):
    """
    GRADIEND - GRADIent ENcoder Decoder model implementation (weights-only):
    maps model gradients to a low-dimensional latent space and back.

    Proposed by Drechsel et al. 2025 (https://arxiv.org/abs/2502.01406).

    This class holds ONLY the neural components (encoder/decoder) + utilities that depend solely on
    GRADIEND parameters:

    - forward / forward_encoder (tensor input space)
    - weight-derived importance scores (encoder/decoder/decoder-bias/decoder-sum)
    - internal prune primitive that physically reduces input_dim (slices weights; no mapping logic)
    - save_pretrained / from_pretrained for weights + architecture + metadata

    Saving:

    - Weights: model.safetensors if available, else pytorch_model.bin
    - Config: config.json (format_version=0)
    - Run info: training.json (optional; if kwargs contains "training")

    Use ParamMappedGradiendModel when you need a parameter mapping or dict-of-gradients I/O.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        activation_encoder: str = "tanh",
        activation_decoder: str = "id",
        bias_encoder: bool = True,
        bias_decoder: bool = True,
        torch_dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        device_encoder: Optional[torch.device] = None,
        device_decoder: Optional[torch.device] = None,
        lazy_init: bool = False,
        init_fan_in_floor: Optional[int] = DEFAULT_INIT_FAN_IN_FLOOR,
        component_slices: Optional[List[Union[GradiendComponent, Dict[str, Any]]]] = None,
        component_split_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a weights-only GRADIEND model (i.e., a GRADIEND encoder-decoder without base-model context
        but ).

        Activation functions (case-insensitive):
        tanh, relu, leakyrelu, gelu, silu, elu, sigmoid, smht (hardtanh), id (identity).
        Defaults (paper): encoder tanh, decoder id.

        Args:
            input_dim: Size of the GRADIEND input space (total selected gradient entries).
            latent_dim: Size of the latent bottleneck.
            activation_encoder: Encoder activation name (case-insensitive).
            activation_decoder: Decoder activation name. If falsy, uses encoder activation
                but with decoder-appropriate defaults via get_activation.
            bias_encoder: Whether the encoder linear layer uses a bias term. Enabled by default.
            bias_decoder: Whether the decoder linear layer uses a bias term.
            torch_dtype: dtype used for model parameters.
            device: Optional default device for both encoder and decoder when specific
                devices are not provided.
            device_encoder: Device for encoder parameters.
            device_decoder: Device for decoder parameters.
            lazy_init: If True, do not create encoder/decoder weights here. Build them later
                via prune (with pruned size) or _build_encoder_decoder (full size).
            init_fan_in_floor: Optional lower bound for the fan-in used when
                initializing encoder weights and matching decoder rows. The
                default, ``10000``, was added after empirical ACTIEND runs showed
                that raw activation components such as one transformer hidden
                state can be much smaller than GRADIEND parameter spaces, making
                the usual ``1/sqrt(n)`` initialization too large. The floor keeps
                small components on a conservative random scale while leaving
                larger gradient spaces unchanged. None uses the raw
                component/input fan-in.
            component_slices: Optional non-overlapping contiguous component metadata over
                the GRADIEND input space. None creates one virtual full component used
                internally; the public ``component_slices`` property stays empty when
                ``component_split_mode`` is ``"none"``.
            component_split_mode: Persisted split mode. ``"none"`` means ordinary
                GRADIEND behavior (virtual full span only); other modes expose
                partitions via public ``component_slices``, including a single
                ``"full"`` partition for ``"single"``.
            **kwargs: Additional metadata stored in `self.kwargs` and serialized into config.json metadata
                on save. Non-JSONable values are stringified in a safe way.
        """
        if not isinstance(input_dim, int):
            raise TypeError(f"input_dim must be int, got {type(input_dim).__name__}")
        if not isinstance(latent_dim, int):
            raise TypeError(f"latent_dim must be int, got {type(latent_dim).__name__}")
        if not isinstance(activation_encoder, str):
            raise TypeError(f"activation_encoder must be str, got {type(activation_encoder).__name__}")
        if not isinstance(activation_decoder, str):
            raise TypeError(f"activation_decoder must be str, got {type(activation_decoder).__name__}")
        if not isinstance(bias_encoder, bool):
            raise TypeError(f"bias_encoder must be bool, got {type(bias_encoder).__name__}")
        if not isinstance(bias_decoder, bool):
            raise TypeError(f"bias_decoder must be bool, got {type(bias_decoder).__name__}")
        if not isinstance(lazy_init, bool):
            raise TypeError(f"lazy_init must be bool, got {type(lazy_init).__name__}")
        if component_split_mode is not None and not isinstance(component_split_mode, str):
            raise TypeError(
                f"component_split_mode must be str or None, got {type(component_split_mode).__name__}"
            )

        super().__init__()
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_encoder = device_encoder or device or default_device
        self.device_decoder = device_decoder or device or default_device

        self.latent_dim = int(latent_dim)
        self.input_dim = int(input_dim)
        if self.input_dim <= 0 or self.latent_dim <= 0:
            raise ValueError("Input and latent dimensions must be positive integers.")

        self.activation = activation_encoder.lower()
        self.activation_decoder = activation_decoder
        self.bias_encoder = bool(bias_encoder)
        self.bias_decoder = bool(bias_decoder)
        self.torch_dtype = torch_dtype
        self._lazy_init = bool(lazy_init)
        self.init_fan_in_floor = _coerce_init_fan_in_floor(init_fan_in_floor)
        self._virtual_component_slices = _normalize_component_slices(
            component_slices, input_dim=self.input_dim
        )
        self.component_split_mode = (
            component_split_mode.strip().lower()
            if isinstance(component_split_mode, str) and component_split_mode.strip()
            else ("none" if component_slices is None else "custom")
        )

        self.kwargs = kwargs
        if "base_model" in self.kwargs and hasattr(self.kwargs["base_model"], "name_or_path"):
            self.kwargs["base_model"] = self.kwargs["base_model"].name_or_path

        if activation_decoder:
            self.activation_decoder = activation_decoder
        else:
            self.activation_decoder = self.activation

        if self._lazy_init:
            self.encoder = None
            self.decoder = None
        else:
            activation_fnc = get_activation(self.activation, encoder=True)
            if activation_decoder:
                activation_fnc_decoder = get_activation(activation_decoder)
            else:
                activation_fnc_decoder = get_activation(self.activation, encoder=False)

            self.encoder = nn.Sequential(
                LargeLinear(
                    self.input_dim,
                    self.latent_dim,
                    bias=self.bias_encoder,
                    dtype=torch_dtype,
                    device=self.device_encoder,
                ),
                activation_fnc,
            )
            self.decoder = nn.Sequential(
                LargeLinear(self.latent_dim, self.input_dim, bias=self.bias_decoder, dtype=torch_dtype, device=self.device_decoder),
                activation_fnc_decoder,
            )

            self._initialize_encoder_decoder_parameters()

    def __str__(self) -> str:
        return (
            f"GradiendModel(input_dim={self.input_dim}, latent_dim={self.latent_dim}, "
            f"activation_encoder={self.activation!r}, activation_decoder={getattr(self, 'activation_decoder', self.activation)!r}, "
            f"bias_encoder={self.bias_encoder}, bias_decoder={self.bias_decoder})"
        )

    def _build_encoder_decoder(self, input_dim: int) -> None:
        """
        Instantiate encoder and decoder with given input_dim. Used for lazy init:
        either after prune (with pruned size) or before first use (with full size).
        """
        if self.encoder is not None and self.decoder is not None:
            return
        self.input_dim = int(input_dim)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        # Lazy-init pre-prune builds encoder/decoder at a smaller input_dim than the
        # initial component metadata (full unpruned space). Keep slices consistent.
        if (
            not self._virtual_component_slices
            or any(component.end > self.input_dim for component in self._virtual_component_slices)
        ):
            self._virtual_component_slices = _normalize_component_slices(
                None, input_dim=self.input_dim
            )

        activation_fnc = get_activation(self.activation, encoder=True)
        if self.activation_decoder and self.activation_decoder != self.activation:
            activation_fnc_decoder = get_activation(self.activation_decoder)
        else:
            activation_fnc_decoder = get_activation(self.activation, encoder=False)

        self.encoder = nn.Sequential(
            LargeLinear(
                self.input_dim,
                self.latent_dim,
                bias=self.bias_encoder,
                dtype=self.torch_dtype,
                device=self.device_encoder,
            ),
            activation_fnc,
        )
        self.decoder = nn.Sequential(
            LargeLinear(self.latent_dim, self.input_dim, bias=self.bias_decoder, dtype=self.torch_dtype, device=self.device_decoder),
            activation_fnc_decoder,
        )

        self._initialize_encoder_decoder_parameters()

    def _effective_init_fan_in(self, fan_in: int) -> int:
        """Return the fan-in used for initialization after applying the optional floor."""
        fan_in = int(fan_in)
        if fan_in <= 0:
            raise ValueError(f"fan_in must be positive, got {fan_in}")
        if self.init_fan_in_floor is None:
            return fan_in
        return max(int(self.init_fan_in_floor), fan_in)

    def _init_bound_for_fan_in(self, fan_in: int) -> float:
        """Return the uniform initialization bound for a raw or component fan-in."""
        return 1.0 / math.sqrt(float(self._effective_init_fan_in(fan_in)))

    def _initialize_encoder_decoder_parameters(self) -> None:
        """
        Initialize encoder columns and decoder rows on the same component scale.

        The physical model stores one full encoder/decoder matrix even for split
        GRADIENDs. Component-aware initialization therefore works by applying the
        component fan-in bound to each contiguous component slice. The fan-in
        floor is intentionally simple: it avoids disproportionately large initial
        weights for small activation-space components, which was empirically
        helpful for ACTIEND convergence, without adding signal-specific branches.
        """
        self._require_built()
        enc = self.encoder[0].linear
        dec = self.decoder[0].linear

        full_bound = self._init_bound_for_fan_in(self.input_dim)
        default_linear_bound = 1.0 / math.sqrt(float(self.input_dim))
        virtual_slices = self._virtual_component_slices
        full_component = (
            len(virtual_slices) == 1
            and virtual_slices[0].start == 0
            and virtual_slices[0].end == self.input_dim
        )

        with torch.no_grad():
            # nn.Linear has already initialized encoder weights with the raw
            # full-matrix fan-in. Reinitialize only when the fan-in floor or
            # component partitioning changes that intended scale.
            if (not full_component) or not math.isclose(full_bound, default_linear_bound):
                nn.init.uniform_(enc.weight, -full_bound, full_bound)
                if enc.bias is not None:
                    nn.init.uniform_(enc.bias, -full_bound, full_bound)
                for component in virtual_slices:
                    component_bound = self._init_bound_for_fan_in(component.input_dim)
                    if math.isclose(component_bound, full_bound):
                        continue
                    nn.init.uniform_(
                        enc.weight[:, component.start:component.end],
                        -component_bound,
                        component_bound,
                    )

            overall_scale = float(enc.weight.abs().max().item())
            nn.init.uniform_(dec.weight, -overall_scale, overall_scale)
            if dec.bias is not None:
                nn.init.uniform_(dec.bias, -overall_scale, overall_scale)
            if not full_component:
                for component in virtual_slices:
                    component_scale = float(
                        enc.weight[:, component.start:component.end].abs().max().item()
                    )
                    nn.init.uniform_(
                        dec.weight[component.start:component.end, :],
                        -component_scale,
                        component_scale,
                    )
                    if dec.bias is not None:
                        nn.init.uniform_(
                            dec.bias[component.start:component.end],
                            -component_scale,
                            component_scale,
                        )

    def to(
        self,
        device: Union[str, torch.device, None] = None,
        *,
        device_encoder: Optional[Union[str, torch.device]] = None,
        device_decoder: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "GradiendModel":
        """
        Move encoder and decoder to the requested devices.

        - If device_encoder or device_decoder is provided, moves only those submodules.
        - If device is provided (and no split devices), moves both to that device.
        - If torch_dtype is provided, converts encoder/decoder parameters to that dtype.
        - If device_encoder/device_decoder is None, leaves that submodule's placement unchanged.
        - When encoder/decoder are not yet built (lazy init), only updates target device attributes.
        """
        if torch_dtype is not None:
            self.torch_dtype = torch_dtype
        if device_encoder is not None or device_decoder is not None:
            if device_encoder is not None:
                self.device_encoder = torch.device(device_encoder) if isinstance(device_encoder, str) else device_encoder
                if self.encoder is not None:
                    self.encoder.to(device=self.device_encoder, dtype=torch_dtype)
            if device_decoder is not None:
                self.device_decoder = torch.device(device_decoder) if isinstance(device_decoder, str) else device_decoder
                if self.decoder is not None:
                    self.decoder.to(device=self.device_decoder, dtype=torch_dtype)
            return self
        if device is not None:
            dev = torch.device(device) if isinstance(device, str) else device
            self.device_encoder = dev
            self.device_decoder = dev
            if self.encoder is not None:
                self.encoder.to(device=dev, dtype=torch_dtype)
            if self.decoder is not None:
                self.decoder.to(device=dev, dtype=torch_dtype)
        elif torch_dtype is not None:
            if self.encoder is not None:
                self.encoder.to(dtype=torch_dtype)
            if self.decoder is not None:
                self.decoder.to(dtype=torch_dtype)
        return self

    def cpu(self) -> "GradiendModel":
        """Move encoder and decoder to CPU."""
        return self.to("cpu")

    def cuda(self, device: Optional[Union[int, str, torch.device]] = None) -> "GradiendModel":
        """Move encoder and decoder to CUDA."""
        if device is None:
            return self.to("cuda")
        if isinstance(device, int):
            return self.to(f"cuda:{device}")
        return self.to(device)

    # ----------------- norms -----------------
    def _require_built(self) -> None:
        """Raise if encoder/decoder are not yet built (lazy init)."""
        if self.encoder is None or self.decoder is None:
            raise RuntimeError(
                "Encoder/decoder weights not yet built. For lazy-init gradiend with pre_prune, "
                "call prune() first. Otherwise call _build_encoder_decoder(input_dim)."
            )

    def _ensure_built(self) -> None:
        """Build encoder/decoder with current input_dim if not yet built (lazy init)."""
        if self.encoder is None or self.decoder is None:
            self._build_encoder_decoder(self.input_dim)

    @property
    def component_slices(self) -> Tuple[GradiendComponent, ...]:
        """Public component partitions. Empty for ``GradiendSplit.none()``."""
        if self.component_split_mode == "none":
            return ()
        return self._virtual_component_slices

    @property
    def component_count(self) -> int:
        """Number of public split components (0 when unpartitioned)."""
        return len(self.component_slices)

    @property
    def has_component_split(self) -> bool:
        """Whether the model exposes a public component split over its input space."""
        return self.component_split_mode != "none"

    @property
    def signal_kind(self) -> str:
        """Signal kind represented by this GRADIEND input space."""
        return self.signal_kind_from_metadata(self.kwargs)

    @property
    def uses_gradients(self) -> bool:
        """Whether this model represents gradient-space signals."""
        return self.signal_kind == "gradient"

    @property
    def uses_activations(self) -> bool:
        """Whether this model represents activation-space signals."""
        return self.signal_kind == "activation"

    @property
    def uses_activation_gradients(self) -> bool:
        """Whether this model represents activation-gradient signals."""
        return self.signal_kind == "activation_gradient"

    @property
    def is_gradiend(self) -> bool:
        """Alias for gradient-space GRADIEND semantics."""
        return self.uses_gradients

    @property
    def is_actiend(self) -> bool:
        """Alias for activation-space ACTIEND semantics."""
        return self.uses_activations

    @staticmethod
    def signal_kind_from_metadata(metadata: Optional[Dict[str, Any]]) -> str:
        """Resolve a signal kind from serialized GRADIEND metadata."""
        return gradiend_signal_kind_from_metadata(metadata)

    @staticmethod
    def method_name_from_signal_kind(kind: str) -> str:
        """Return the human-facing method name for a signal kind."""
        return gradiend_method_name_from_signal_kind(kind)

    @staticmethod
    def method_name_from_metadata(metadata: Optional[Dict[str, Any]]) -> str:
        """Resolve the human-facing method name from serialized GRADIEND metadata."""
        return GradiendModel.method_name_from_signal_kind(
            GradiendModel.signal_kind_from_metadata(metadata)
        )

    @property
    def method_name(self) -> str:
        """Human-facing method name for logs and diagnostics."""
        return self.method_name_from_signal_kind(self.signal_kind)

    @property
    def input_unit_name(self) -> str:
        """Human-facing singular unit for one GRADIEND input dimension."""
        kind = self.signal_kind
        if kind == "activation":
            return "activation dimension"
        if kind == "activation_gradient":
            return "activation-gradient dimension"
        if kind == "gradient":
            return "gradient entry"
        return "signal dimension"

    @staticmethod
    def _plural(count: int, singular: str, plural: Optional[str] = None) -> str:
        if plural is None and singular.endswith("entry"):
            plural = f"{singular[:-1]}ies"
        return singular if int(count) == 1 else (plural or f"{singular}s")

    def _component_preview(self, *, max_ids: int = 3) -> str:
        components = tuple(self.component_slices or ())
        if not components:
            return ""
        ids = [str(component.id) for component in components]
        shown = ", ".join(ids[:max_ids])
        if len(ids) <= max_ids:
            return f" [{shown}]"
        return f" [first {max_ids}: {shown}]"

    def describe_training_space(self, *, split_loss: Optional[str] = None) -> str:
        """Return a compact, signal-aware training-start description."""
        feature_text = f"{self.latent_dim} {self._plural(self.latent_dim, 'feature neuron')}"
        unit = self.input_unit_name
        if self.has_component_split:
            aggregation = str(split_loss or "mean")
            return (
                f"Training component-split {self.method_name} over {self.input_dim:,} "
                f"{self._plural(self.input_dim, unit)} across {self.component_count} "
                f"{self._plural(self.component_count, 'component')} with {feature_text} per component "
                f"(loss aggregation={aggregation}){self._component_preview()}."
            )
        return (
            f"Training {self.method_name} over {self.input_dim:,} "
            f"{self._plural(self.input_dim, unit)} with {feature_text}."
        )

    @property
    def _component_encoders(self) -> _ComponentAccessor:
        """Virtual component encoders over ``_virtual_component_slices``."""
        return _ComponentAccessor(self, "encoder")

    @property
    def _component_decoders(self) -> _ComponentAccessor:
        """Virtual component decoders over ``_virtual_component_slices``."""
        return _ComponentAccessor(self, "decoder")

    def _component_by_key(self, key: Union[int, str]) -> GradiendComponent:
        if isinstance(key, int):
            return self._virtual_component_slices[key]
        if isinstance(key, str):
            for component in self._virtual_component_slices:
                if component.id == key:
                    return component
            raise KeyError(f"Unknown GRADIEND component id: {key!r}")
        raise TypeError(f"Component key must be int or str, got {type(key).__name__}")

    def _component_view(self, key: Union[int, str]) -> _VirtualComponent:
        return _VirtualComponent(self, self._component_by_key(key))

    def _iter_components(self):
        """Iterate over virtual component metadata (includes none's full span)."""
        yield from self._virtual_component_slices

    def _with_components(
        self,
        component_slices: List[Union[GradiendComponent, Dict[str, Any]]],
        *,
        component_split_mode: str = "custom",
    ) -> "GradiendModel":
        """Return a lightweight view of this model with different component metadata."""
        view = copy.copy(self)
        view.__dict__ = self.__dict__.copy()
        view._virtual_component_slices = _normalize_component_slices(
            component_slices, input_dim=self.input_dim
        )
        view.component_split_mode = component_split_mode
        return view

    def _without_split(self) -> "GradiendModel":
        """Return a lightweight unpartitioned view (virtual full span only)."""
        return self._with_components(
            [GradiendComponent("full", 0, self.input_dim)],
            component_split_mode="none",
        )

    def _encode_components(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode all default-split components from one full GRADIEND input tensor.

        Returns shape ``(n_components, latent_dim)`` for a single vector and
        ``(..., n_components, latent_dim)`` for batched inputs.
        """
        self._require_built()
        x = self._ensure_input(x)
        encodings = [encoder(x) for encoder in self._component_encoders]
        return torch.stack(encodings, dim=-2)

    def _decode_components(self, z: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Decode one latent tensor through every virtual component decoder."""
        return tuple(decoder(z) for decoder in self._component_decoders)

    def _forward_components(
        self,
        x: torch.Tensor,
        *,
        return_encoded: bool = False,
    ) -> Union[Tuple[torch.Tensor, ...], Tuple[Tuple[torch.Tensor, ...], torch.Tensor]]:
        """
        Forward each default-split component independently.

        Component encoders read only their input slice and component decoders
        reconstruct only their output slice. Encoder/decoder tensors are shared
        with the full GRADIEND model.
        """
        self._require_built()
        x = self._ensure_input(x)
        encoded = self._encode_components(x)
        decoded = tuple(
            decoder(encoded[..., index, :])
            for index, decoder in enumerate(self._component_decoders)
        )
        return (decoded, encoded) if return_encoded else decoded

    def _component_target_slices(self, target: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Slice a full target tensor into default-split component targets."""
        if target.dim() == 0:
            raise ValueError("Component target tensor must not be scalar")
        if target.shape[-1] != self.input_dim:
            if target.numel() == self.input_dim:
                target = target.view(-1)
            else:
                raise ValueError(
                    f"Target tensor has incorrect shape {tuple(target.shape)}, "
                    f"expected final dimension {self.input_dim}"
                )
        return tuple(
            target[..., component.start:component.end]
            for component in self._virtual_component_slices
        )

    def reconstruction_loss(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        criterion: Any,
        aggregation: str = "mean",
        return_encoded: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute the GRADIEND reconstruction loss for full or component-split training.

        Args:
            source: Full signal tensor in GRADIEND input space.
            target: Full target tensor in GRADIEND input space.
            criterion: Loss callable used per reconstruction.
            aggregation: For component-split models, one of ``"mean"``, ``"sum"``,
                ``"size_weighted"``, or ``"full"``. ``"full"`` keeps the classic
                full-vector objective. Non-split models always use the full-vector
                objective.
            return_encoded: If True, also return the full encoding or component
                encodings used for the loss.
        """
        aggregation = str(aggregation or "mean").strip().lower()
        if aggregation not in {"mean", "sum", "size_weighted", "full"}:
            raise ValueError(
                "aggregation must be 'mean', 'sum', 'size_weighted', or 'full', "
                f"got {aggregation!r}"
            )

        if aggregation == "full" or not self.has_component_split:
            decoded, encoded = self(source, return_encoded=True)
            if target.device != decoded.device:
                target = target.to(decoded.device)
            loss = criterion(decoded, target)
            return (loss, encoded) if return_encoded else loss

        decoded_components, encoded_components = self._forward_components(source, return_encoded=True)
        target_components = self._component_target_slices(target)
        losses = []
        for decoded, component_target in zip(decoded_components, target_components):
            if component_target.device != decoded.device:
                component_target = component_target.to(decoded.device)
            losses.append(criterion(decoded, component_target))
        component_losses = torch.stack(losses)

        if aggregation == "sum":
            loss = component_losses.sum()
        elif aggregation == "size_weighted":
            weights = torch.tensor(
                [component.input_dim for component in self._virtual_component_slices],
                dtype=component_losses.dtype,
                device=component_losses.device,
            )
            loss = (component_losses * (weights / weights.sum())).sum()
        else:
            loss = component_losses.mean()

        return (loss, encoded_components) if return_encoded else loss

    @property
    def base_model_id(self) -> str:
        """
        Base model identifier stored in kwargs.

        Raises:
            ValueError: If base_model is missing from kwargs.
        """
        base_model = self.kwargs.get("base_model")
        if not base_model:
            raise ValueError("base_model missing from gradiend.kwargs")
        return base_model

    @property
    def decoder_norm(self) -> float:
        """
        L2 norm of the decoder weight matrix.

        Returns:
            Scalar float of the decoder's weight L2 norm.
        """
        self._require_built()
        return torch.norm(self.decoder[0].weight, p=2).item()

    @property
    def encoder_norm(self) -> float:
        """
        L2 norm of the encoder weight matrix.

        Returns:
            Scalar float of the encoder's weight L2 norm.
        """
        self._require_built()
        return torch.norm(self.encoder[0].weight, p=2).item()

    # ----------------- importance -----------------
    def get_weight_importance(self, part: str = "decoder-weight") -> torch.Tensor:
        """
        Importance per GRADIEND input dimension (length = input_dim), on CPU.
        Args:
            part: Which component to use for importance aggregation:

                - "encoder-weight": L1 over encoder weight columns
                - "decoder-weight": L1 over decoder weight rows
                - "decoder-bias": absolute decoder bias
                - "decoder-sum": absolute(sum(weight_row) + bias)

        Returns:
            1D CPU float tensor of length input_dim, where higher means more
            influential according to the chosen aggregation.
        """
        self._require_built()
        part = (part or "decoder-weight").lower()
        vec = self.get_update_vector(part=part).detach().cpu()

        if part in ("decoder-sum", "decoder-bias"):
            return vec.abs()

        if part == "decoder-weight":
            w = vec.view(self.decoder[0].linear.weight.shape)
            return w.abs().sum(dim=1)
        if part == "encoder-weight":
            w = vec.view(self.encoder[0].linear.weight.shape)
            return w.abs().sum(dim=0)

        raise ValueError(
            f"part must be 'encoder-weight', 'decoder-weight', 'decoder-bias', or 'decoder-sum', got {part!r}"
        )


    def get_update_vector(self, part: str = "decoder-weight") -> torch.Tensor:
        """
        Return a flattened weight-derived update vector.

        Args:
            part: Which component to use for the update vector:

                - "decoder-weight": decoder weight vector (flattened)
                - "decoder-bias": decoder bias vector
                - "decoder-sum": decoder weight vector + bias
                - "encoder-weight": encoder weight vector (flattened)

        Returns:
            1D tensor in GRADIEND input space derived from the requested component.
        """
        self._require_built()
        part = (part or "decoder-weight").lower()

        if part == "decoder-weight":
            return self.decoder[0].linear.weight.flatten()
        if part == "decoder-bias":
            b = self.decoder[0].linear.bias
            if b is None:
                return torch.zeros(self.input_dim, dtype=self.torch_dtype, device=self.decoder[0].linear.weight.device)
            return b
        if part == "decoder-sum":
            w = self.decoder[0].linear.weight
            b = self.decoder[0].linear.bias
            row_sum = w.sum(dim=1)
            return row_sum if b is None else row_sum + b
        if part == "encoder-weight":
            return self.encoder[0].linear.weight.flatten()
        raise ValueError(f"part must be 'encoder-weight', 'decoder-weight', 'decoder-bias', or 'decoder-sum', got {part!r}")

    def get_topk_weights(self, part: str = "decoder-weight", topk: Union[int, float] = 1000) -> List[int]:
        """
        Return the top-k input indices by importance score.

        Args:
            part: Importance source passed to get_weight_importance.
                Options: "encoder-weight", "decoder-weight", "decoder-bias", "decoder-sum".
            topk: Number of indices to return (clipped to input_dim) or a proportion in (0, 1].

        Returns:
            List of input indices (length k) sorted by descending importance.
        """
        if isinstance(topk, bool):
            raise TypeError("topk must be int or float, not bool")
        if isinstance(topk, float):
            if not (0.0 < topk <= 1.0):
                raise ValueError("topk as float must be in (0, 1.0]")
        elif isinstance(topk, int):
            if topk < 0:
                raise ValueError("topk as int must be >= 0")
        else:
            raise TypeError(f"topk must be int or float, got {type(topk).__name__}")

        imp = self.get_weight_importance(part=part)
        if isinstance(topk, float):
            k = int(math.ceil(topk * imp.numel()))
        else:
            k = int(topk)
        k = min(max(k, 1), imp.numel())
        _, idx = torch.topk(imp, k=k, largest=True, sorted=True)
        return idx.tolist()

    def _ensure_input(self, x: torch.Tensor):
        if x.dim() == 0:
            raise ValueError(f"Input tensor must have final dimension {self.input_dim}, got scalar input")

        if x.dim() == 1:
            if x.numel() != self.input_dim:
                raise ValueError(f"Input tensor has incorrect size {x.numel()}, expected {self.input_dim}")
        elif x.shape[-1] == self.input_dim:
            pass
        elif x.numel() == self.input_dim:
            # Legacy convenience: accept shapes like (input_dim, 1) as one vector.
            x = x.view(-1)
        else:
            raise ValueError(
                f"Input tensor has incorrect shape {tuple(x.shape)}, expected final dimension {self.input_dim}"
            )

        if x.dtype != self.torch_dtype:
            x = x.to(self.torch_dtype)

        if x.device != self.device_encoder:
            x = x.to(self.device_encoder)

        return x

    # ----------------- forward -----------------
    def forward(
        self, x: torch.Tensor, return_encoded: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for tensor input already in GRADIEND input space.

        Args:
            x: 1D tensor of shape (input_dim,) representing a flattened
                gradient vector in GRADIEND input space.
            return_encoded: If True, also return the latent encoding.

        Returns:
            If return_encoded is False:
                Decoded tensor of shape (input_dim,).
            If return_encoded is True:
                Tuple (decoded, encoded), where:

                - decoded: tensor of shape (input_dim,)
                - encoded: tensor of shape (latent_dim,)
        """
        self._require_built()
        x = self._ensure_input(x)

        encoded = self.encoder(x)
        if encoded.device != self.device_decoder:
            encoded = encoded.to(self.device_decoder)
        decoded = self.decoder(encoded)

        return (decoded, encoded) if return_encoded else decoded

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder-only forward for tensor input.

        Args:
            x: 1D tensor of shape (input_dim,) in GRADIEND input space.

        Returns:
            Encoded tensor of shape (latent_dim,).
        """
        self._require_built()
        x = self._ensure_input(x)
        return self.encoder(x)

    # ----------------- prune primitive -----------------
    def _prune_input_dims(
        self,
        keep_idx: torch.Tensor,
        *,
        inplace: bool = False,
        return_index_map: bool = False,
        keep_idx_sorted_unique: bool = False,
    ):
        """
        INTERNAL: physically prune input_dim and output_dim by slicing encoder/decoder weights.

        - encoder: slice columns (latent_dim, input_dim) -> (latent_dim, new_in)
        - decoder: slice rows    (input_dim, latent_dim) -> (new_in, latent_dim)
        - bias:   slice entries  (input_dim,)            -> (new_in,)
        """
        if not torch.is_tensor(keep_idx):
            raise TypeError(f"keep_idx must be a torch.Tensor, got {type(keep_idx)}")
        if keep_idx.dim() != 1 or keep_idx.numel() == 0:
            raise ValueError("keep_idx must be a non-empty 1D tensor")
        if keep_idx.dtype not in (torch.int64, torch.long, torch.int32):
            raise TypeError(f"keep_idx must be integer dtype, got {keep_idx.dtype}")

        keep_idx = keep_idx.detach().to(torch.long)
        if not keep_idx_sorted_unique:
            keep_idx = torch.unique(keep_idx, sorted=True)

        m = self if inplace else copy.deepcopy(self)

        enc_ll = m.encoder[0].linear
        dec_ll = m.decoder[0].linear
        enc_w, enc_b = enc_ll.weight, enc_ll.bias
        dec_w, dec_b = dec_ll.weight, dec_ll.bias

        keep_idx_dev = keep_idx.detach().to(dtype=torch.long, device=enc_w.device)
        if keep_idx_dev.min().item() < 0 or keep_idx_dev.max().item() >= m.input_dim:
            raise ValueError(f"keep_idx out of bounds for input_dim={m.input_dim}")

        new_in = int(keep_idx_dev.numel())

        new_enc = LargeLinear(new_in, m.latent_dim, bias=m.bias_encoder, dtype=m.torch_dtype, device=m.device_encoder)
        new_dec = LargeLinear(m.latent_dim, new_in, bias=m.bias_decoder, dtype=m.torch_dtype, device=m.device_decoder)

        with torch.no_grad():
            new_enc.linear.weight.copy_(enc_w[:, keep_idx_dev].to(new_enc.linear.weight.device))
            if new_enc.linear.bias is not None and enc_b is not None:
                new_enc.linear.bias.copy_(enc_b.to(new_enc.linear.bias.device))

            keep_idx_dec = keep_idx_dev if dec_w.device == keep_idx_dev.device else keep_idx_dev.to(dec_w.device)
            new_dec.linear.weight.copy_(dec_w[keep_idx_dec, :].to(new_dec.linear.weight.device))
            if new_dec.linear.bias is not None:
                if dec_b is None:
                    new_dec.linear.bias.zero_()
                else:
                    new_dec.linear.bias.copy_(dec_b[keep_idx_dec].to(new_dec.linear.bias.device))

        m.encoder[0] = new_enc
        m.decoder[0] = new_dec
        m.encoder[0].train(m.training)
        m.decoder[0].train(m.training)
        m.input_dim = new_in
        m._virtual_component_slices = _normalize_component_slices(None, input_dim=new_in)

        return (m, keep_idx.detach().to("cpu").long()) if return_index_map else m

    def prune(
        self,
        *,
        topk: Union[int, float, None] = None,
        threshold: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        part: str = "decoder-weight",
        importance: Optional[torch.Tensor] = None,
        inplace: bool = False,
        return_mask: bool = False,
    ) -> Union["GradiendModel", Tuple["GradiendModel", torch.Tensor]]:
        """
        Physically prune the model (reduce input_dim) by selecting important input dimensions.
        
        Selection order: mask -> threshold -> topk.
        
        Args:
            topk: int (absolute) or float in (0,1] (relative fraction among remaining dims).
            threshold: keep dims with importance >= threshold.
            mask: optional bool tensor of shape (input_dim,) in current input space.
            part: 'encoder-weight' | 'decoder-weight' | 'decoder-bias' | 'decoder-sum' (used when importance is None).
            importance: optional 1D tensor of length input_dim; used instead of get_weight_importance(part) when provided.
            inplace: modify this instance if True, else return a deepcopy.
            return_mask: if True, also return final combined_mask (original input space).
        
        Returns:
            If return_mask is False:
                The pruned GradiendModel (self or a deepcopy depending on `inplace`).
            If return_mask is True:
                Tuple (model, combined_mask) where combined_mask is a bool tensor
                of shape (old_input_dim,) indicating kept dimensions.
        """
        if topk is None and threshold is None and mask is None:
            raise ValueError("At least one of topk, threshold, mask must be provided.")
        if topk is not None:
            if isinstance(topk, bool):
                raise TypeError("topk must be int or float, not bool")
            if isinstance(topk, int) and topk < 0:
                raise ValueError("topk as int must be >= 0")
            if isinstance(topk, float) and not (0.0 < topk <= 1.0):
                raise ValueError("topk as float must be in (0, 1.0]")
            if not isinstance(topk, (int, float)):
                raise TypeError(f"topk must be int or float, got {type(topk).__name__}")
        if threshold is not None:
            if not isinstance(threshold, (int, float)):
                raise TypeError(f"threshold must be float or None, got {type(threshold).__name__}")
            if threshold < 0:
                raise ValueError("threshold must be >= 0")

        # topk=1.0 (float) means no pruning (return self); topk int 1 means keep top-1 dimension
        if topk is not None and isinstance(topk, float) and topk == 1.0:
            m = self if inplace else copy.deepcopy(self)
            if return_mask:
                old_input_dim = int(self.input_dim)
                full_mask = torch.ones(old_input_dim, dtype=torch.bool, device="cpu")
                return m, full_mask
            return m

        old_input_dim = int(self.input_dim)
        combined = torch.ones(old_input_dim, dtype=torch.bool, device="cpu")
        
        if mask is not None:
            if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.shape != (old_input_dim,):
                raise ValueError(f"mask must be bool tensor with shape ({old_input_dim},)")
            combined &= mask.detach().to("cpu")
        
        importance_scores = None
        if threshold is not None or topk is not None:
            if importance is not None:
                importance_scores = importance.detach().to("cpu")
            else:
                importance_scores = self.get_weight_importance(part=part)
            
            if threshold is not None:
                combined &= (importance_scores >= threshold)
            
            if topk is not None:
                if isinstance(topk, float):
                    if not (0.0 < topk <= 1.0):
                        raise ValueError("topk float must be in (0, 1]")
                    k = int(math.ceil(topk * combined.sum().item()))
                else:
                    k = int(topk)
                k = min(max(k, 1), combined.sum().item())
                if k < combined.sum().item():
                    masked_importance = importance_scores.clone()
                    masked_importance[~combined] = float('-inf')
                    _, top_indices = torch.topk(masked_importance, k=k, largest=True, sorted=True)
                    new_combined = torch.zeros(old_input_dim, dtype=torch.bool)
                    new_combined[top_indices] = True
                    combined = new_combined
        
        keep_idx = torch.nonzero(combined, as_tuple=False).squeeze(-1)
        if keep_idx.numel() == 0:
            raise ValueError("Pruning resulted in zero dimensions")
        
        result = self._prune_input_dims(keep_idx, inplace=inplace, return_index_map=return_mask)
        if return_mask:
            pruned_model, index_map = result
            # Convert index_map to bool mask
            final_mask = torch.zeros(old_input_dim, dtype=torch.bool)
            final_mask[index_map] = True
            return pruned_model, final_mask
        return result

    # ----------------- save/load (weights-only) -----------------
    def save_pretrained(self, save_directory: str, use_safetensors: Optional[bool] = None, **kwargs: Any) -> None:
        """
        Save weights + config.json (+ optional training.json).

        Notes:

        - safetensors is used if available unless use_safetensors=False.
        - training info: if kwargs contains "training", it is written to training.json and removed from config metadata.

        Args:
            save_directory: Folder to write model files into.
            use_safetensors: If True, require safetensors. If False, force PyTorch
                bin format. If None, prefer safetensors when available.
            **kwargs: Extra metadata to store in config.json.

        Returns:
            None.
        """
        self._require_built()
        os.makedirs(save_directory, exist_ok=True)

        prefer_safetensors = (use_safetensors is not False)
        used_safetensors = False

        # ---- weights ----
        if prefer_safetensors:
            try:
                from safetensors.torch import save_file
                save_file(self.state_dict(), os.path.join(save_directory, "model.safetensors"))
                used_safetensors = True
            except ImportError as e:
                if use_safetensors is True:
                    raise ImportError("safetensors not installed, cannot save as safetensors") from e

        if not used_safetensors:
            torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

        # ---- config ----
        self.kwargs.update(kwargs)
        meta = self._serialize_kwargs()

        run_info = meta.pop("training", None)

        config = {
            "format_version": 0,
            "architecture": {
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "activation_encoder": self.activation,
                "activation_decoder": self.activation_decoder,
                "bias_encoder": self.bias_encoder,
                "bias_decoder": self.bias_decoder,
                "init_fan_in_floor": self.init_fan_in_floor,
                "torch_dtype": str(self.torch_dtype).replace("torch.", ""),
            },
            "components": (
                [component.to_dict() for component in self.component_slices]
                if self.has_component_split
                else None
            ),
            "component_split_mode": self.component_split_mode,
            "mapping": None,  # filled by ParamMappedGradiendModel; core leaves None
            "metadata": meta,
        }

        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        if run_info is not None:
            with open(os.path.join(save_directory, "training.json"), "w") as f:
                json.dump(run_info, f, indent=2)

    def _serialize_kwargs(self) -> Dict[str, Any]:
        """Serialize kwargs, filtering out non-JSON objects."""
        kwargs = self.kwargs.copy()
        out: Dict[str, Any] = {}
        for k, v in kwargs.items():
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                if hasattr(v, "id"):
                    out[k] = {"id": v.id, "_type": type(v).__name__}
                else:
                    out[k] = {"_type": type(v).__name__, "_repr": str(v)[:120]}
        return convert_tuple_keys_recursively(out)

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
        device_encoder: Optional[torch.device] = None,
        device_decoder: Optional[torch.device] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "GradiendModel":
        """
        Load weights + config.json (weights-only).
        ParamMappedGradiendModel overrides to also load mapping.

        Args:
            load_directory: Directory containing model files.
            device_encoder: Optional device override for encoder parameters.
            device_decoder: Optional device override for decoder parameters.
            torch_dtype: Optional dtype override. If None, uses dtype stored in config.json.

        Returns:
            Instantiated GradiendModel with loaded weights and metadata.
        """
        cfg_path = os.path.join(load_directory, "config.json")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        arch = cfg["architecture"]
        component_slices = cfg.get("components")
        component_split_mode = cfg.get("component_split_mode")
        component_slices = _coerce_saved_component_slices(
            component_slices, component_split_mode, arch["input_dim"]
        )

        # dtype
        if torch_dtype is None:
            td = arch.get("torch_dtype", "float32")
            torch_dtype = getattr(torch, td, torch.float32)

        # weights
        st_path = os.path.join(load_directory, "model.safetensors")
        bin_path = os.path.join(load_directory, "pytorch_model.bin")
        if not os.path.exists(st_path) and not os.path.exists(bin_path):
            model_dir = os.path.join(load_directory, "model")
            st_path = os.path.join(model_dir, "model.safetensors")
            bin_path = os.path.join(model_dir, "pytorch_model.bin")

        if os.path.exists(st_path):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(st_path)
            except ImportError:
                # fallback to bin if present
                if os.path.exists(bin_path):
                    state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
                else:
                    raise
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No model weights found in {load_directory}")

        # instantiate
        meta = cfg.get("metadata") or {}
        model = cls(
            input_dim=arch["input_dim"],
            latent_dim=arch["latent_dim"],
            activation_encoder=arch.get("activation_encoder", "tanh"),
            activation_decoder=arch.get("activation_decoder", "id"),
            bias_encoder=(
                arch["bias_encoder"]
                if "bias_encoder" in arch
                else "encoder.0.linear.bias" in state_dict
            ),
            bias_decoder=arch.get("bias_decoder", True),
            init_fan_in_floor=arch.get("init_fan_in_floor", None),
            torch_dtype=torch_dtype,
            device_encoder=device_encoder,
            device_decoder=device_decoder,
            component_slices=component_slices,
            component_split_mode=component_split_mode,
            **meta,
        )

        # attach training info (optional)
        training_path = os.path.join(load_directory, "training.json")
        if os.path.exists(training_path):
            try:
                with open(training_path, "r") as f:
                    model.kwargs.setdefault("training", json.load(f))
            except Exception:
                pass

        model.load_state_dict(state_dict)
        model.name_or_path = load_directory
        return model


    def pruned_length(self) -> int:
        """
        Return the current input_dim after pruning.

        Returns:
            Current input_dim as an integer.
        """
        return self.input_dim

    def __len__(self) -> int:
        """
        Return the current input_dim after pruning.

        Returns:
            Current input_dim as an integer.
        """
        return self.pruned_length()
