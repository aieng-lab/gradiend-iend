"""Trainer support for already-extracted signal-vector pairs."""

from __future__ import annotations

import copy
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

from gradiend.model import ModelWithGradiend
from gradiend.model.model_with_gradiend import _is_gradiend_checkpoint
from gradiend.trainer.config import TrainerConfig
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.dataset import SignalTrainingDatasetBase
from gradiend.trainer.core.signals import Signal, SignalBatch, SignalSet, require_single_signal
from gradiend.trainer.trainer import Trainer
from gradiend.util import normalize_split_name
from gradiend.util.encoding_rows import encode_dataset_to_rows
from gradiend.util.paths import resolve_encoder_analysis_path


SignalMatrix = Union[torch.Tensor, Sequence[Any]]
SignalSplit = Union[
    Tuple[SignalMatrix, SignalMatrix],
    Mapping[str, SignalMatrix],
]
SignalData = Union[SignalSplit, Mapping[str, SignalSplit]]


def _signal_matrix(value: SignalMatrix, *, name: str) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape (examples, signal_dim), got {tuple(tensor.shape)}")
    if tensor.shape[0] < 1 or tensor.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one non-empty signal vector")
    if not tensor.is_floating_point():
        tensor = tensor.float()
    return tensor.detach()


def _split_matrices(value: SignalSplit, *, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(value, Mapping):
        if "positive" not in value or "negative" not in value:
            raise ValueError(f"Signal split {split!r} requires 'positive' and 'negative' matrices")
        positive_raw, negative_raw = value["positive"], value["negative"]
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        positive_raw, negative_raw = value
    else:
        raise TypeError(
            f"Signal split {split!r} must be (positive, negative) or a mapping with those keys"
        )
    positive = _signal_matrix(positive_raw, name=f"{split}.positive")
    negative = _signal_matrix(negative_raw, name=f"{split}.negative")
    if positive.shape[1] != negative.shape[1]:
        raise ValueError(
            f"Signal widths for split {split!r} must match, got "
            f"{positive.shape[1]} and {negative.shape[1]}"
        )
    return positive, negative


def _normalize_signal_data(data: SignalData) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(data, Mapping) and "positive" not in data and "negative" not in data:
        normalized = {
            normalize_split_name(str(split)): _split_matrices(value, split=str(split))
            for split, value in data.items()
        }
    else:
        normalized = {"train": _split_matrices(data, split="train")}
    if not normalized:
        raise ValueError("Signal data must contain at least one split")
    widths = {int(positive.shape[1]) for positive, _negative in normalized.values()}
    if len(widths) != 1:
        raise ValueError(f"All signal splits must have the same width, got {sorted(widths)}")
    return normalized


@dataclass
class SignalTrainerConfig(TrainerConfig):
    """Configuration for training from already-extracted signal vectors."""

    data: Optional[SignalData] = None
    target_classes: Optional[List[str]] = None
    run_id: Optional[str] = None
    n_features: int = 1


class SignalPairDataset(Dataset):
    """Deterministically pair positive and negative signal vectors.

    Rows expose the ordinary GRADIEND factual/alternative dataset contract.
    ``SignalTrainingDatasetBase`` is responsible for source/target selection,
    ``source='both'`` pole swapping, label inversion, and batching.
    """

    def __init__(
        self,
        positive: SignalMatrix,
        negative: SignalMatrix,
        *,
        positive_class: str,
        negative_class: str,
        batch_size: int = 1,
        max_size: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.positive = _signal_matrix(positive, name="positive")
        self.negative = _signal_matrix(negative, name="negative")
        if self.positive.shape[1] != self.negative.shape[1]:
            raise ValueError(
                "positive and negative signal widths must match, got "
                f"{self.positive.shape[1]} and {self.negative.shape[1]}"
            )
        self.positive_class = str(positive_class)
        self.negative_class = str(negative_class)
        self.batch_size = max(1, int(batch_size))
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        positive_order = torch.randperm(len(self.positive), generator=generator)
        negative_order = torch.randperm(len(self.negative), generator=generator)
        if max_size is not None:
            limit = max(1, int(max_size))
            positive_order = positive_order[:limit]
            negative_order = negative_order[:limit]
        self._positive_order = positive_order
        self._negative_order = negative_order
        self._pair_count = max(len(positive_order), len(negative_order))

    @property
    def signal_dim(self) -> int:
        return int(self.positive.shape[1])

    def __len__(self) -> int:
        return int(self._pair_count)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        positive_index = self._positive_order[index % len(self._positive_order)]
        negative_index = self._negative_order[index % len(self._negative_order)]
        return {
            "factual": self.positive[positive_index],
            "alternative": self.negative[negative_index],
            "label": 1.0,
            "factual_id": self.positive_class,
            "alternative_id": self.negative_class,
            "feature_class_id": self.positive_class,
        }


class _PrecomputedSignalExtractor:
    """Identity extractor implementing the ordinary signal-extractor contract."""

    def __init__(self, signal: Signal) -> None:
        self.signal = signal
        self.signals = SignalSet(signal)

    def exclusive_signal_access(self):
        return nullcontext()

    def __call__(
        self,
        factual_inputs: Optional[torch.Tensor] = None,
        alternative_inputs: Optional[torch.Tensor] = None,
        *,
        requires_factual: bool = True,
        requires_alternative: bool = True,
    ) -> SignalBatch:
        def _as_tensor(value: Optional[Any]) -> Optional[torch.Tensor]:
            if value is None:
                return None
            if torch.is_tensor(value):
                return value
            if isinstance(value, (list, tuple)) and value and all(torch.is_tensor(item) for item in value):
                return torch.stack(list(value), dim=0)
            return torch.as_tensor(value)

        factual = _as_tensor(factual_inputs) if requires_factual else None
        alternative = _as_tensor(alternative_inputs) if requires_alternative else None
        return SignalBatch.from_factual_alternative(
            factual,
            alternative,
            signal_id=self.signal.id,
        )


class SignalTrainer(Trainer):
    """Concrete ``Trainer`` for already-extracted paired signal vectors.

    ``Signal`` describes what the vectors measure; this trainer only supplies
    those vectors to the ordinary training lifecycle. Consequently the same
    trainer supports gradient, activation, activation-gradient, and future
    signal kinds as long as the supplied model maps the same signal kind. It
    inherits argument resolution, callbacks, runtime monitoring, checkpointing,
    cache policy, multi-seed execution, convergence handling, and post-pruning
    from :class:`gradiend.trainer.Trainer`. The existing
    :class:`SignalTrainingDatasetBase` performs batching and source/target
    compilation exactly as it does for other modalities.

    A preconstructed ``ModelWithGradiend`` is required because extracted vectors
    alone do not identify a base model or signal scope. Encoder evaluation is
    supported on supplied signal splits. Decoder/base-
    model evaluation is intentionally unavailable: it requires modality inputs
    and a task-specific intervention evaluator, neither of which can be inferred
    from extracted signal tensors.
    """

    def __init__(
        self,
        model: ModelWithGradiend,
        positive: Optional[SignalMatrix] = None,
        negative: Optional[SignalMatrix] = None,
        *,
        data: Optional[SignalData] = None,
        args: Optional[TrainingArguments] = None,
        config: Optional[SignalTrainerConfig] = None,
        target_classes: Optional[List[str]] = None,
        run_id: Optional[str] = None,
        n_features: int = 1,
        evaluator_class: Optional[type] = None,
    ) -> None:
        if not isinstance(model, ModelWithGradiend):
            raise TypeError("SignalTrainer requires a preconstructed ModelWithGradiend")
        if data is not None and (positive is not None or negative is not None):
            raise ValueError("Pass either data=... or positive=... and negative=..., not both")
        self.config = config or SignalTrainerConfig()
        if data is None and positive is None and negative is None and self.config.data is not None:
            data = self.config.data
        elif data is None:
            if positive is None or negative is None:
                raise ValueError("SignalTrainer requires data=... or both positive and negative matrices")
            data = (positive, negative)
        if self.config.data is not None and self.config.data is not data:
            raise ValueError("Signal data was provided both directly and through config")
        self.config.data = data
        classes = target_classes or self.config.target_classes or ["positive", "negative"]
        if len(classes) != 2:
            raise ValueError(f"SignalTrainer requires exactly two target classes, got {classes!r}")
        self.config.target_classes = [str(value) for value in classes]
        self.config.n_features = int(n_features or self.config.n_features)
        self._signal_data = _normalize_signal_data(data)
        expected_dim = int(getattr(model.gradiend, "input_dim", 0) or 0)
        actual_dim = next(iter(self._signal_data.values()))[0].shape[1]
        if int(actual_dim) != expected_dim:
            raise ValueError(
                f"Signal width {actual_dim} does not match model input_dim {expected_dim}"
            )
        if args is None:
            args = TrainingArguments(signal=Signal(str(model.signal_kind)), source="both", target="diff")
        signal = require_single_signal(
            signal=getattr(args, "signal", None),
            signals=getattr(args, "signals", None),
            context="SignalTrainer",
        )
        model_signal_kind = str(model.signal_kind).strip().lower()
        if signal.kind.strip().lower() != model_signal_kind:
            raise ValueError(
                f"Configured signal kind {signal.kind!r} does not match model signal kind "
                f"{model_signal_kind!r}"
            )
        self._model_template = model
        self._initial_gradiend = copy.deepcopy(model.gradiend)
        self._shared_model_cls = type(model)
        super().__init__(
            model=model,
            target_classes=self.config.target_classes,
            args=args,
            run_id=run_id or self.config.run_id,
            n_features=self.config.n_features,
            evaluator_class=evaluator_class,
        )

    @property
    def default_model_with_gradiend_cls(self) -> type:
        return self._shared_model_cls

    def create_model_with_gradiend(
        self,
        load_directory: str,
        model_with_gradiend_cls: Optional[type] = None,
        **kwargs: Any,
    ) -> ModelWithGradiend:
        """Create or reload only the signal mapping; never duplicate the base model."""
        del model_with_gradiend_cls, kwargs
        if _is_gradiend_checkpoint(load_directory):
            template_gradiend = self._model_template.gradiend
            gradiend = type(template_gradiend).from_pretrained(
                load_directory,
                device_encoder=template_gradiend.device_encoder,
                device_decoder=template_gradiend.device_decoder,
                torch_dtype=template_gradiend.torch_dtype,
            )
        else:
            gradiend = copy.deepcopy(self._initial_gradiend)

        # A shallow nn.Module copy needs independent registry dictionaries before
        # replacing a registered submodule. The base model and modality helpers
        # remain shared; only the small learned signal mapping is independent.
        model = copy.copy(self._model_template)
        model._parameters = self._model_template._parameters.copy()
        model._buffers = self._model_template._buffers.copy()
        model._modules = self._model_template._modules.copy()
        model.gradiend = gradiend
        if getattr(self._model_template, "_gradient_creator", None) is self._model_template:
            model._gradient_creator = model
        model.name_or_path = load_directory
        model._ensure_gradiend_param_map_spec()
        model._sync_base_requires_grad_to_param_map()
        return model

    def get_target_feature_class_ids(self) -> Optional[List[Any]]:
        return list(self.target_classes or [])

    def create_training_data(
        self,
        model_or_tokenizer: Any,
        split: str = "train",
        batch_size: Optional[int] = None,
        max_size: Optional[int] = None,
        **kwargs: Any,
    ) -> SignalPairDataset:
        del model_or_tokenizer, kwargs
        split_name = normalize_split_name(str(split))
        if split_name not in self._signal_data:
            raise ValueError(
                f"No signal data for split {split_name!r}; available: {sorted(self._signal_data)}"
            )
        positive, negative = self._signal_data[split_name]
        seed = int(getattr(self.training_args, "seed", 42) or 42)
        return SignalPairDataset(
            positive,
            negative,
            positive_class=self.target_classes[0],
            negative_class=self.target_classes[1],
            batch_size=batch_size or 1,
            max_size=max_size,
            seed=seed,
        )

    def create_gradient_training_dataset(
        self,
        raw_training_data: SignalPairDataset,
        model_with_gradiend: ModelWithGradiend,
        *,
        cache_dir: Optional[str] = None,
        use_cached_gradients: bool = False,
        **kwargs: Any,
    ) -> SignalTrainingDatasetBase:
        del cache_dir, use_cached_gradients
        args = self.training_args
        signal = require_single_signal(
            signal=kwargs.pop("signal", getattr(args, "signal", None)),
            signals=kwargs.pop("signals", getattr(args, "signals", None)),
            context="SignalTrainer.create_gradient_training_dataset",
        )
        model_signal_kind = str(model_with_gradiend.signal_kind).strip().lower()
        if signal.kind.strip().lower() != model_signal_kind:
            raise ValueError(
                f"Configured signal kind {signal.kind!r} does not match model signal kind "
                f"{model_signal_kind!r}"
            )
        source = kwargs.pop("source", getattr(args, "source", "both"))
        target = kwargs.pop("target", getattr(args, "target", "diff"))
        dtype = kwargs.pop("dtype", model_with_gradiend.gradiend.torch_dtype)
        device = kwargs.pop("device", model_with_gradiend.gradiend.device_encoder)
        return SignalTrainingDatasetBase(
            raw_training_data,
            _PrecomputedSignalExtractor(signal),
            source=source,
            target=target,
            cache_dir=None,
            use_cached_signals=False,
            dtype=dtype,
            device=device,
            timing_steps=kwargs.pop("timing_steps", 0),
            timing_label=kwargs.pop("timing_label", "precomputed-signal"),
            signal=signal,
            **kwargs,
        )

    def _analyze_encoder(
        self,
        model_with_gradiend: ModelWithGradiend,
        split: str = "train",
        neutral_data_df: Optional[pd.DataFrame] = None,
        max_size: Optional[int] = None,
        use_cache: Optional[bool] = None,
        plot: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        del neutral_data_df, plot
        use_cache = self._resolve_artifact_use_cache(use_cache, fallback=False)
        cache_path = (
            resolve_encoder_analysis_path(self.experiment_dir, split=split, max_size=max_size, **kwargs)
            if self.experiment_dir
            else None
        )
        if use_cache and cache_path and os.path.isfile(cache_path):
            return pd.read_csv(cache_path)
        eval_data = self.create_eval_data(
            model_with_gradiend,
            split=split,
            source=getattr(self.training_args, "source", "both"),
            max_size=max_size,
        )
        rows = encode_dataset_to_rows(model_with_gradiend, eval_data)
        for row in rows:
            row["type"] = "training"
        frame = pd.DataFrame(rows)
        if cache_path:
            path = Path(cache_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
            frame.to_csv(tmp, index=False)
            os.replace(tmp, path)
        return frame

    @staticmethod
    def _decoder_unavailable() -> None:
        raise NotImplementedError(
            "SignalTrainer cannot infer decoder/base-model evaluation from extracted signal tensors; "
            "use a modality-specific evaluator with the original inputs"
        )

    def _get_decoder_eval_dataframe(self, *args: Any, **kwargs: Any):
        del args, kwargs
        self._decoder_unavailable()

    def _get_decoder_eval_targets(self) -> Dict[str, List[str]]:
        self._decoder_unavailable()

    def evaluate_base_model(self, model: Any, tokenizer: Any, use_cache: Optional[bool] = None, **kwargs: Any):
        del model, tokenizer, use_cache, kwargs
        self._decoder_unavailable()


__all__ = ["SignalPairDataset", "SignalTrainer", "SignalTrainerConfig"]
