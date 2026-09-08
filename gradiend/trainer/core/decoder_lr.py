"""Memory-safe automatic decoder learning-rate calibration.

The decoder is linear by default, so with a frozen encoder and MSE objective its
local optimum is a small least-squares problem in latent space. Adam already
stores an exponentially averaged gradient for every decoder parameter. This
module combines that existing optimizer state with an EMA of the tiny latent
Hessian to estimate the distance to the decoder optimum without ever creating a
new tensor whose size scales with the decoder output dimension.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn


AUTO_DECODER_LR = "auto"


def is_auto_decoder_lr(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == AUTO_DECODER_LR


def _linear(module: Any) -> Any:
    return getattr(module, "linear", module)


@dataclass(frozen=True)
class _DecoderBlock:
    name: str
    start: int
    end: int

    @property
    def output_dim(self) -> int:
        return self.end - self.start


class AutoDecoderLearningRate:
    """One-shot reachability calibration backed by existing Adam moments.

    Persistent calibration state is independent of decoder output width: one
    ``(latent_dim + bias)^2`` CPU Hessian per logical component. Decoder-sized
    Adam first moments already owned by the optimizer are only read through
    reductions whose outputs are small Gram matrices.
    """

    def __init__(
        self,
        *,
        gradiend: Any,
        optimizer: torch.optim.Optimizer,
        initial_lr: float,
        aggregation: str,
        criterion: Any,
    ) -> None:
        if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            raise ValueError(
                "learning_rate_decoder='auto' currently requires optim='adamw' "
                "or optim='adam'; SGD does not have the Adam first moment used "
                "by the memory-safe estimator"
            )
        if not isinstance(criterion, nn.MSELoss) or criterion.reduction not in {"mean", "sum"}:
            raise ValueError(
                "learning_rate_decoder='auto' requires nn.MSELoss with "
                "reduction='mean' or 'sum'"
            )
        activation = str(getattr(gradiend, "activation_decoder", "id") or "id").lower()
        if activation not in {"id", "identity", "linear"}:
            raise ValueError(
                "learning_rate_decoder='auto' requires an identity/linear decoder; "
                f"got activation_decoder={activation!r}"
            )

        decoder = getattr(gradiend, "decoder", None)
        if decoder is None:
            raise ValueError("Automatic decoder LR requires a built decoder")
        layer = _linear(decoder[0])
        self.weight = layer.weight
        self.bias = layer.bias
        self.latent_dim = int(self.weight.shape[1])
        self.has_bias = self.bias is not None
        self.design_dim = self.latent_dim + int(self.has_bias)
        self.gradiend = gradiend
        self.optimizer = optimizer
        self.initial_lr = float(initial_lr)
        self.aggregation = str(aggregation or "mean").strip().lower()
        self.reduction = str(criterion.reduction)
        self.decoder_group = self._find_decoder_group()
        if float(self.decoder_group.get("weight_decay", 0.0)) != 0.0:
            raise ValueError(
                "learning_rate_decoder='auto' requires weight_decay=0 for the "
                "decoder parameter group because the estimator targets the "
                "unregularized least-squares optimum"
            )
        self.beta1 = float(self.decoder_group.get("betas", (0.9, 0.999))[0])
        self.blocks = self._blocks()
        self.hessian_ema: List[torch.Tensor] = [
            torch.zeros((self.design_dim, self.design_dim), dtype=torch.float64)
            for _ in self.blocks
        ]
        self.observations = 0
        self.selected_lr: Optional[float] = None
        self.last_result: Dict[str, Any] = {
            "status": "warming_up",
            "initial_lr": self.initial_lr,
            "selected_lr": None,
            "calibration_step": None,
            "persistent_state_numel": sum(x.numel() for x in self.hessian_ema),
            "decoder_parameter_count": int(
                self.weight.numel() + (self.bias.numel() if self.bias is not None else 0)
            ),
        }

    def _find_decoder_group(self) -> Dict[str, Any]:
        decoder_ids = {id(self.weight)}
        if self.bias is not None:
            decoder_ids.add(id(self.bias))
        for group in self.optimizer.param_groups:
            ids = {id(parameter) for parameter in group["params"]}
            if decoder_ids <= ids:
                return group
        raise ValueError("Optimizer has no parameter group containing the decoder")

    def _blocks(self) -> Tuple[_DecoderBlock, ...]:
        if bool(getattr(self.gradiend, "has_component_split", False)) and self.aggregation != "full":
            return tuple(
                _DecoderBlock(str(component.id), int(component.start), int(component.end))
                for component in self.gradiend._iter_components()
            )
        return (_DecoderBlock("full", 0, int(self.weight.shape[0])),)

    def _design(self, encoded: torch.Tensor, block_index: int) -> torch.Tensor:
        values = encoded.detach().to(device="cpu", dtype=torch.float64)
        if len(self.blocks) > 1:
            if values.shape[-2] != len(self.blocks) or values.shape[-1] != self.latent_dim:
                raise ValueError(
                    "Component-split automatic decoder LR expected encodings with "
                    f"final shape ({len(self.blocks)}, {self.latent_dim}), got "
                    f"{tuple(values.shape)}"
                )
            values = values[..., block_index, :]
        values = values.reshape(-1, self.latent_dim)
        if not self.has_bias:
            return values
        ones = torch.ones((values.shape[0], 1), dtype=torch.float64)
        return torch.cat((values, ones), dim=1)

    def _aggregation_factor(self, block: _DecoderBlock) -> float:
        if len(self.blocks) == 1:
            return 1.0
        if self.aggregation == "mean":
            return 1.0 / float(len(self.blocks))
        if self.aggregation == "sum":
            return 1.0
        if self.aggregation == "size_weighted":
            return float(block.output_dim) / float(self.weight.shape[0])
        raise ValueError(f"Unsupported decoder loss aggregation: {self.aggregation!r}")

    def observe(self, encoded: torch.Tensor) -> None:
        """Update the tiny Hessian EMA from one ordinary training forward."""
        batch_hessians = []
        for index, block in enumerate(self.blocks):
            design = self._design(encoded, index)
            if design.shape[0] == 0:
                raise ValueError("Automatic decoder LR received an empty latent batch")
            scale = 2.0 * self._aggregation_factor(block)
            if self.reduction == "mean":
                scale /= float(design.shape[0] * block.output_dim)
            batch_hessians.append(scale * (design.T @ design))

        for running, batch in zip(self.hessian_ema, batch_hessians):
            running.mul_(self.beta1).add_(batch, alpha=1.0 - self.beta1)
        self.observations += 1

    @staticmethod
    def _state_step(state: Dict[str, Any]) -> int:
        value = state.get("step", 0)
        if torch.is_tensor(value):
            return int(value.detach().cpu().item())
        return int(value)

    def _moment_gram(self, block: _DecoderBlock) -> Tuple[torch.Tensor, int]:
        weight_state = self.optimizer.state.get(self.weight, {})
        moment_weight = weight_state.get("exp_avg")
        if moment_weight is None:
            raise ValueError("Adam decoder first moment is unavailable; need at least one optimizer step")
        step = self._state_step(weight_state)
        weight_view = moment_weight.detach()[block.start:block.end, :]
        # These contractions return latent-sized tensors. They do not clone or
        # concatenate the decoder-sized first moment.
        ww = (weight_view.T @ weight_view).detach().cpu().to(torch.float64)
        if not self.has_bias:
            return ww, step

        bias_state = self.optimizer.state.get(self.bias, {})
        moment_bias = bias_state.get("exp_avg")
        if moment_bias is None:
            raise ValueError("Adam decoder-bias first moment is unavailable")
        if self._state_step(bias_state) != step:
            raise ValueError("Decoder weight and bias optimizer steps disagree")
        bias_view = moment_bias.detach()[block.start:block.end]
        wb = (weight_view.T @ bias_view).detach().cpu().to(torch.float64)
        bb = torch.dot(bias_view, bias_view).detach().cpu().to(torch.float64)
        gram = torch.empty((self.design_dim, self.design_dim), dtype=torch.float64)
        gram[: self.latent_dim, : self.latent_dim] = ww
        gram[: self.latent_dim, -1] = wb
        gram[-1, : self.latent_dim] = wb
        gram[-1, -1] = bb
        return gram, step

    def estimate(self, *, step: int, total_steps: int) -> Dict[str, Any]:
        """Estimate and apply the decoder reachability-floor LR.

        Rank-deficient latent Hessians leave the dummy LR in place and can be
        retried at a later evaluation boundary.
        """
        remaining_steps = int(total_steps) - int(step)
        if remaining_steps <= 0:
            self.last_result = {
                **self.last_result,
                "status": "no_remaining_steps",
                "calibration_step": int(step),
            }
            return dict(self.last_result)
        if self.observations <= 0:
            return dict(self.last_result)

        correction = 1.0 - self.beta1 ** self.observations
        distance_squared = 0.0
        ranks = []
        condition_numbers = []
        optimizer_steps = []
        for block, running_hessian in zip(self.blocks, self.hessian_ema):
            hessian = running_hessian / correction
            rank = int(torch.linalg.matrix_rank(hessian).item())
            ranks.append(rank)
            if rank < self.design_dim:
                self.last_result = {
                    **self.last_result,
                    "status": "waiting_for_full_rank",
                    "calibration_step": int(step),
                    "hessian_ranks": ranks,
                    "required_rank": self.design_dim,
                }
                return dict(self.last_result)
            condition_numbers.append(float(torch.linalg.cond(hessian).item()))
            gram, optimizer_step = self._moment_gram(block)
            optimizer_steps.append(optimizer_step)
            moment_correction = 1.0 - self.beta1 ** optimizer_step
            gram = gram / (moment_correction * moment_correction)
            inverse = torch.linalg.inv(hessian)
            block_distance_squared = torch.trace(inverse.T @ gram @ inverse).item()
            distance_squared += max(0.0, float(block_distance_squared))

        distance = math.sqrt(distance_squared)
        decoder_parameter_count = int(self.last_result["decoder_parameter_count"])
        reachability_floor = distance / (
            float(remaining_steps) * math.sqrt(float(decoder_parameter_count))
        )
        selected = max(self.initial_lr, reachability_floor)
        if not math.isfinite(selected) or selected <= 0.0:
            raise ValueError(
                "Automatic decoder LR produced a non-finite/non-positive rate: "
                f"distance={distance}, remaining_steps={remaining_steps}, "
                f"decoder_parameter_count={decoder_parameter_count}"
            )
        self.decoder_group["lr"] = float(selected)
        self.selected_lr = float(selected)
        self.last_result = {
            **self.last_result,
            "status": "calibrated",
            "calibration_step": int(step),
            "observations": int(self.observations),
            "remaining_steps": remaining_steps,
            "decoder_distance_to_local_optimum": float(distance),
            "reachability_floor_lr": float(reachability_floor),
            "selected_lr": float(selected),
            "multiplier_over_encoder_lr": float(selected / self.initial_lr),
            "target_reachability": 1.0,
            "hessian_ranks": ranks,
            "hessian_condition_numbers": condition_numbers,
            "optimizer_steps": optimizer_steps,
        }
        return dict(self.last_result)

    def summary(self) -> Dict[str, Any]:
        return dict(self.last_result)


__all__ = [
    "AUTO_DECODER_LR",
    "AutoDecoderLearningRate",
    "is_auto_decoder_lr",
]
