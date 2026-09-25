"""
Signal training datasets: modality-agnostic signal batching and gradient wrapper.

SignalTrainingDatasetBase wraps any training dataset that yields batches with 'factual'
and 'alternative' inputs, runs a signal_extractor on them, and returns source/target
tensors. Modality-specific details are delegated to callers: padding value when batching
variable-length tensors (get_padding_value), and cache_key_fields when caching is used.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from contextlib import nullcontext
from typing import Any, Callable, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from gradiend.util import hash_it
from gradiend.util.logging import get_logger
from gradiend.trainer.core.config import (
    factual_computation_required_keywords,
    alternative_computation_required_keywords,
    source_target_keywords,
    validate_source_target_combination,
)
from gradiend.trainer.core.signals import (
    GradientSignalExtractor,
    SignalSet,
    require_single_gradient_signal,
    require_single_signal,
)

logger = get_logger(__name__)


def _invert_numeric_label(label: Any) -> Any:
    """Return the opposite binary label for alternative-source metadata."""
    if torch.is_tensor(label):
        numeric = label.to(dtype=torch.float32)
        if bool(torch.isin(numeric, torch.tensor([-1.0, 0.0, 1.0], device=numeric.device)).all().item()):
            return -label
        return label
    if isinstance(label, (int, float)):
        return -label if label in (-1, 0, 1) else label
    if isinstance(label, list):
        return [_invert_numeric_label(v) for v in label]
    if isinstance(label, tuple):
        return tuple(_invert_numeric_label(v) for v in label)
    return label


# Metadata pairs swapped when compiling source="both" alternative batches to factual/diff.
_BOTH_SWAP_KEY_PAIRS = (
    ("factual", "alternative"),
    ("factual_token", "alternative_token"),
    ("factual_id", "alternative_id"),
)


def _swap_factual_alternative_batch(batch: dict) -> dict:
    """Return a shallow-copied batch with factual/alternative poles swapped and label inverted."""
    out = dict(batch)
    for left_key, right_key in _BOTH_SWAP_KEY_PAIRS:
        if left_key in out or right_key in out:
            out[left_key], out[right_key] = out.get(right_key), out.get(left_key)
    if "label" in out:
        out["label"] = _invert_numeric_label(out["label"])
    return out


def _both_side_for_batch(index: int, *, n_balance_groups: int = 1) -> str:
    """Choose factual vs alternative pole for ``source='both'`` training.

    Training datasets that set ``balance_column`` (e.g. ``feature_pole``) cycle
    balance groups with ``batch_idx % n_balance_groups``. Pole selection must be
    **orthogonal** to that phase. Using ``batch_idx % 2`` alone locks feature
    batches onto a single pole whenever ``n_balance_groups`` is even — the common
    case of one feature group plus neutral identity (``add_neutral_identity_transitions``).

    Rule: ``visit = batch_idx // n_balance_groups``; even visit → factual, odd →
    alternative. With ``n_balance_groups=1`` this is the historical even/odd rule.
    """
    n = max(1, int(n_balance_groups))
    visit = int(index) // n
    return "factual" if (visit % 2 == 0) else "alternative"


def _both_eval_base_and_side(index: int) -> tuple[int, str]:
    """Map an encoder-eval index to (base_batch_index, side) when expanding poles.

    Used for ``source='both'`` encoder eval and for **one-pole** encoder eval
    (where a single factual class would otherwise yield only one label sign).
    Normal two-pole encoder eval must **not** expand: encode only ``source``.
    """
    base = int(index) // 2
    side = "factual" if (int(index) % 2 == 0) else "alternative"
    return base, side



class SignalTrainingDatasetBase:
    """
    Modality-agnostic dataset for GRADIEND signal extraction and batching.

    Wraps a training dataset that provides items with 'factual' and 'alternative' inputs.
    Batches items, optionally pads variable-length tensors (via get_padding_value),
    runs a signal extractor on factual/alternative, and returns source/target tensors
    (factual, alternative, or diff). When cache_dir and use_cached_signals are set, cache_key_fields must be provided:
    list of batch dict keys whose values are included in the cache hash (e.g. ['input_text', 'label']).
    All listed keys must be present in every batch when caching is used.

    Args:
        training_data: Dataset with __len__ and __getitem__ returning dicts containing
            at least 'factual' and 'alternative' (modality-specific, e.g. tokenizer outputs).
        signal_extractor: Callable producing a SignalBatch from factual/alternative inputs.
        source: 'factual' | 'alternative' | 'diff' | 'both' | None. When None (e.g. supervised_decoder),
            source signals are not computed. ``both`` alternates poles per training batch and compiles to
            the factual/diff path via optional fac↔alt swap. For encoder evaluation (``target=None``),
            encode only the configured ``source`` unless ``expand_encoder_eval_poles=True`` (one-pole)
            or ``source='both'`` (both poles are the source).
        target: Same options or None. When None (e.g. supervised_encoder), target signals are not computed.
            ``source='both'`` requires ``target='diff'`` or ``target=None``.
        expand_encoder_eval_poles: When ``target=None``, expand each base row to factual and
            alternative poles so one-pole data still yields ±1 labels. Must stay False for
            normal two-pole encoder eval (encode ``source`` only).
        cache_dir: Optional directory for caching extracted signals.
        use_cached_signals: If True and cache_dir set, load/save signals.
        cache_key_fields: When caching is used, list of batch keys to include in cache hash.
            Required when cache_dir is set and use_cached_signals is True. All keys must exist in batch.
        dtype: Tensor dtype.
        device: Device for tensors.
        return_metadata: If True, pass through batch 'metadata'.
        get_padding_value: Callable(subkey: str) -> int used when batching variable-length
            tensors (pad_sequence). Default 0. Text uses tokenizer.pad_token_id for 'input_ids'.
    """

    def __init__(
        self,
        training_data: Any,
        signal_extractor: Any,
        *,
        source: str = 'factual',
        target: str = 'diff',
        expand_encoder_eval_poles: bool = False,
        cache_dir: Optional[str] = None,
        use_cached_signals: bool = True,
        cache_key_fields: Optional[List[str]] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        return_metadata: bool = False,
        get_padding_value: Optional[Callable[[str], int]] = None,
        timing_steps: int = 0,
        timing_label: str = "signal",
        signal: Any = None,
        signals: Any = None,
        combine_diff_in_place: bool = False,
    ):
        assert source in source_target_keywords, f'Invalid source {source}, must be one of {source_target_keywords}'
        assert target in source_target_keywords, f'Invalid target {target}, must be one of {source_target_keywords}'
        if source == "both":
            validate_source_target_combination(source, target)

        if cache_dir is not None and use_cached_signals and (not cache_key_fields or len(cache_key_fields) == 0):
            raise ValueError(
                "When cache_dir is set and use_cached_signals is True, cache_key_fields must be provided "
                "(list of batch keys to include in cache hash, e.g. ['input_text', 'label'])."
            )

        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.training_data = training_data
        self.combine_diff_in_place = bool(combine_diff_in_place)
        self.batch_size = getattr(training_data, 'batch_size', None) or 1
        if not callable(signal_extractor):
            raise TypeError(f"signal_extractor must be callable, got {type(signal_extractor).__name__}")
        self.signal_extractor = signal_extractor
        extractor_signal = getattr(signal_extractor, "signal", None)
        extractor_signals = getattr(signal_extractor, "signals", None)
        if signal is None and signals is None and extractor_signal is None and extractor_signals is None:
            raise ValueError(
                f"{self.__class__.__name__} requires signal=..., signals=..., "
                "or a signal_extractor with .signal/.signals metadata."
            )
        self.signal = require_single_signal(
            signal=signal if signal is not None else extractor_signal,
            signals=signals if signals is not None else extractor_signals,
            context=self.__class__.__name__,
        )
        self.signals = SignalSet(self.signal)
        self.source = source
        self.target = target
        self.expand_encoder_eval_poles = bool(expand_encoder_eval_poles)
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.use_cached_signals = use_cached_signals
        self.cache_key_fields = cache_key_fields or []
        self.dtype = dtype
        self.device = device
        self.return_metadata = return_metadata
        self._get_padding_value = get_padding_value if callable(get_padding_value) else (lambda _: 0)
        self.timing_steps = int(timing_steps or 0)
        self.timing_label = timing_label

    @staticmethod
    def _sync_cuda_for_timing() -> None:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    def _expands_poles_for_encoder_eval(self) -> bool:
        """One-pole encoder eval: map each index to a fac/alt pole (not ``source='both'``)."""
        if self.target is not None or self.source is None:
            return False
        if self.source == "both":
            return False
        return self.expand_encoder_eval_poles

    def __len__(self) -> int:
        n_batches = len(self.training_data) // self.batch_size
        # Dual-pole encoder visits: source='both', or one-pole expand flag.
        if self.target is None and (
            self.source == "both" or self.expand_encoder_eval_poles
        ):
            return 2 * n_batches
        return n_batches

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def _n_training_balance_groups(self) -> int:
        """Balance-group count of the wrapped training dataset (at least 1).

        Used so ``source='both'`` pole visits stay orthogonal to balance-group
        cycling. Prefer ``training_data.n_balance_groups``, then
        ``len(training_data.balance_keys)``.
        """
        td = self.training_data
        n = getattr(td, "n_balance_groups", None)
        if n is not None:
            try:
                return max(1, int(n))
            except (TypeError, ValueError):
                pass
        keys = getattr(td, "balance_keys", None)
        if keys is not None:
            try:
                return max(1, len(keys))
            except TypeError:
                pass
        return 1

    def _resolve_encoder_eval_index(self, index: int) -> tuple[int, str]:
        """Resolve an encoder-eval index into (base_batch_index, pole side)."""
        return _both_eval_base_and_side(index)

    def _resolve_both_index(self, index: int) -> tuple[int, str]:
        """Resolve source='both' into (base_batch_index, pole side)."""
        if self.target is None:
            return self._resolve_encoder_eval_index(index)
        return int(index), _both_side_for_batch(
            index, n_balance_groups=self._n_training_balance_groups()
        )

    def _merge_batch(self, indices: list) -> dict:
        """Collect items at indices and merge into one batch; pad variable-length tensors when needed."""
        if len(indices) == 1:
            return self.training_data[indices[0]]

        batch = {}
        for idx in indices:
            data = self.training_data[idx]
            for key in data:
                if key not in batch:
                    batch[key] = []
                batch[key].append(data[key])

        for key in batch:
            if not (isinstance(batch[key], list) and all(isinstance(d, dict) for d in batch[key])):
                continue
            first = batch[key][0]
            first_key = next(iter(first))
            if not hasattr(first[first_key], 'shape'):
                raise NotImplementedError(
                    'Nested dictionary structure in batch without tensor shapes detected. '
                    'This is an unexpected edge case. Please report this issue.'
                )
            needs_padding = any(
                any(d[subkey].shape != first[subkey].shape for d in batch[key])
                for subkey in first
            )
            if needs_padding:
                padded = {}
                for subkey in first:
                    tensors = [d[subkey] for d in batch[key]]
                    padding_value = self._get_padding_value(subkey)
                    padded[subkey] = pad_sequence(tensors, batch_first=True, padding_value=padding_value)
                batch[key] = padded
            else:
                batch[key] = {subkey: torch.stack([d[subkey] for d in batch[key]]) for subkey in first}
        return batch

    def _exclusive_signal_access(self):
        """Serialize tokenization and base signal extraction for one row when needed."""
        access = getattr(self.signal_extractor, "exclusive_signal_access", None)
        if callable(access):
            return access()
        model = getattr(self.signal_extractor, "__self__", None)
        if model is not None and hasattr(model, "exclusive_base_gradient_access"):
            return model.exclusive_base_gradient_access()
        return nullcontext()

    def _exclusive_gradient_access(self):
        """Backward-compatible alias for old gradient-specific subclasses."""
        return self._exclusive_signal_access()

    @staticmethod
    def _all_truthy(value: Any) -> bool:
        if torch.is_tensor(value):
            return bool(value.to(dtype=torch.bool).all().item())
        if isinstance(value, (list, tuple)):
            return bool(value) and all(SignalTrainingDatasetBase._all_truthy(v) for v in value)
        return bool(value)

    @staticmethod
    def _values_equal(left: Any, right: Any) -> bool:
        if torch.is_tensor(left) or torch.is_tensor(right):
            if not (torch.is_tensor(left) and torch.is_tensor(right)):
                return False
            return bool(torch.equal(left, right))
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            if not (isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))):
                return False
            if len(left) != len(right):
                return False
            return all(SignalTrainingDatasetBase._values_equal(l, r) for l, r in zip(left, right))
        return left == right

    def _is_identity_batch(self, batch: dict) -> bool:
        """Return True when every row is an explicit factual==alternative identity transition."""
        if "is_identity_transition" in batch:
            return self._all_truthy(batch["is_identity_transition"])
        if "factual_token" in batch and "alternative_token" in batch:
            return self._values_equal(batch["factual_token"], batch["alternative_token"])
        return False

    def _is_mixed_site(self) -> bool:
        opts = getattr(self.signal, "options", None) or {}
        target = opts.get("target_token_selector")
        return target is not None and target != opts.get("token_selector")

    @staticmethod
    def _pack_cached_side(source: torch.Tensor, target: Optional[torch.Tensor], *, mixed: bool) -> Any:
        if not mixed or target is None:
            return source
        return {"source": source, "target": target}

    @staticmethod
    def _unpack_cached_side(payload: Any) -> tuple:
        if isinstance(payload, dict) and "source" in payload:
            target = payload.get("target")
            source = payload["source"]
            return source, source if target is None else target
        return payload, payload

    @staticmethod
    def _load_cached_side(cache_file: str) -> Optional[tuple]:
        """Load one cached pole as ``(source, target)``; None when there is no cache entry."""
        if not cache_file or not os.path.exists(cache_file):
            return None
        return SignalTrainingDatasetBase._unpack_cached_side(torch.load(cache_file, weights_only=True))

    def _compute_side(self, batch: dict, side: str, cache_file: str, *, mixed: bool) -> tuple:
        """Extract one pole's ``(source, target)`` signals and write the cache entry if configured."""
        inputs = batch[side]
        source, target = self._extract_one_side(inputs, side=side)
        del inputs
        source = source.to(dtype=self.dtype, device=self.device)
        target = target.to(dtype=self.dtype, device=self.device)
        if cache_file:
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(self._pack_cached_side(source, target, mixed=mixed), cache_file)
        return source, target

    def _extract_one_side(self, inputs: Any, *, side: str) -> tuple:
        if side == "factual":
            batch = self.signal_extractor(
                factual_inputs=inputs,
                alternative_inputs=None,
                requires_factual=True,
                requires_alternative=False,
            )
            source = batch.factual
            target = getattr(batch, "factual_target", None)
        elif side == "alternative":
            batch = self.signal_extractor(
                factual_inputs=None,
                alternative_inputs=inputs,
                requires_factual=False,
                requires_alternative=True,
            )
            source = batch.alternative
            target = getattr(batch, "alternative_target", None)
        else:
            raise ValueError(f"Unknown signal side {side!r}")
        if source is None:
            raise RuntimeError(f"Signal extractor returned no {side} tensor for signal {self.signal.id!r}")
        if target is None:
            target = source
        return source, target

    def __getitem__(self, index: int) -> dict:
        # Bounds-check like any indexable dataset: an out-of-range index otherwise
        # resolves to an empty slice (``range(index*bs, min((index+1)*bs, N))`` is
        # empty), _merge_batch returns ``{}``, and the caller only discovers it far
        # downstream as a cryptic ``template=None`` / ``KeyError: 'factual'``. Raise
        # IndexError here so a caller that iterates past ``len(self)`` fails clearly.
        length = len(self)
        if index < 0:
            index += length
        if not (0 <= index < length):
            raise IndexError(
                f"{type(self).__name__} index {index} out of range for length {length}"
            )
        timing_enabled = self.timing_steps > 0 and (index == 0 or (index + 1) % self.timing_steps == 0)
        if timing_enabled:
            self._sync_cuda_for_timing()
        t0 = time.perf_counter() if timing_enabled else 0.0
        t_merge = t0
        t_factual = t0
        t_alternative = t0
        t_combine = t0

        source = self.source
        fetch_index = index
        eval_side = None
        if source == "both":
            validate_source_target_combination(source, self.target)
            fetch_index, eval_side = self._resolve_both_index(index)
        elif self._expands_poles_for_encoder_eval():
            # One-pole encoder eval only: expand both poles so ±1 labels exist.
            # Normal two-pole encoder eval encodes ``source`` alone (no expand).
            fetch_index, eval_side = self._resolve_encoder_eval_index(index)

        indices = list(
            range(
                fetch_index * self.batch_size,
                min((fetch_index + 1) * self.batch_size, len(self.training_data)),
            )
        )

        with self._exclusive_signal_access():
            batch = self._merge_batch(indices)
            # Optional fac↔alt swap presents the chosen pole as the "factual" side
            # for source="both", or as the selected pole for other encoder-eval sources.
            if eval_side == "alternative":
                batch = _swap_factual_alternative_batch(batch)
            if source == "both":
                # Compile source="both" to the existing factual/diff path.
                source = "factual"
            if timing_enabled:
                self._sync_cuda_for_timing()
                t_merge = time.perf_counter()

            cache_file_factual = ''
            cache_file_alternative = ''
            if self.use_cached_signals and self.cache_dir is not None and self.cache_key_fields:
                missing = [k for k in self.cache_key_fields if k not in batch]
                if missing:
                    raise KeyError(
                        f"Cache key requires batch keys {self.cache_key_fields}; missing in batch: {missing}. "
                        "Ensure training_data yields these keys when caching is used."
                    )
                h = hash_it([batch[k] for k in self.cache_key_fields] + [self.dtype, self.signal.to_dict()])
                cache_file_factual = os.path.join(self.cache_dir, f'factual_{h}.pt')
                cache_file_alternative = os.path.join(self.cache_dir, f'alternative_{h}.pt')

            factual_source = factual_target = None
            alternative_source = alternative_target = None
            identity_batch = self._is_identity_batch(batch)
            mixed = self._is_mixed_site()
            use_cache = self.use_cached_signals and self.cache_dir is not None

            if use_cache and cache_file_factual:
                cached = self._load_cached_side(cache_file_factual)
                if cached is not None:
                    factual_source, factual_target = cached
                if not identity_batch:
                    cached = self._load_cached_side(cache_file_alternative)
                    if cached is not None:
                        alternative_source, alternative_target = cached

            requires_factual = source in factual_computation_required_keywords or self.target in factual_computation_required_keywords
            requires_alternative = source in alternative_computation_required_keywords or self.target in alternative_computation_required_keywords
            if identity_batch and requires_alternative:
                requires_factual = True  # the alternative pole reuses the factual signal
            if factual_source is None and requires_factual:
                factual_source, factual_target = self._compute_side(
                    batch, "factual", cache_file_factual if use_cache else "", mixed=mixed
                )
            if timing_enabled:
                self._sync_cuda_for_timing()
                t_factual = time.perf_counter()

            if identity_batch and requires_alternative:
                alternative_source, alternative_target = factual_source, factual_target
            elif alternative_source is None and requires_alternative:
                alternative_source, alternative_target = self._compute_side(
                    batch, "alternative", cache_file_alternative if use_cache else "", mixed=mixed
                )
            if timing_enabled:
                self._sync_cuda_for_timing()
                t_alternative = time.perf_counter()

            if source == 'factual':
                source_tensor = factual_source
            elif source == 'alternative':
                source_tensor = alternative_source
            elif source == 'diff':
                source_tensor = factual_source - alternative_source
            elif source is None:
                source_tensor = None  # e.g. supervised_decoder: only target needed
            else:
                raise ValueError(f'Unknown source: {source}')

            if self.target == 'factual':
                target_tensor = factual_target
            elif self.target == 'alternative':
                target_tensor = alternative_target
            elif self.target == 'diff':
                if source == 'diff' and not mixed:
                    target_tensor = source_tensor.clone()
                elif self.combine_diff_in_place:
                    # Memory-constrained streaming consumers (notably CGA)
                    # relinquish the factual target after this operation.  Reuse
                    # its storage instead of allocating a third full-width
                    # gradient for ``factual - alternative``.
                    target_tensor = factual_target.sub_(alternative_target)
                else:
                    target_tensor = factual_target - alternative_target
            elif self.target is None:
                target_tensor = None
            else:
                raise ValueError(f'Unknown target: {self.target}')

            del factual_source
            del factual_target
            del alternative_source
            del alternative_target

            output = {'source': source_tensor, 'target': target_tensor}
            for key in batch:
                if key not in output and key not in {'metadata', 'factual', 'alternative'}:
                    output[key] = batch[key]
            # Label metadata must describe the signal exposed as source.
            # For binary pairs, the alternative side is the opposite feature class.
            # source="both" already inverted the label during the fac↔alt swap.
            if source == 'alternative' and 'label' in output:
                output['label'] = _invert_numeric_label(output['label'])
            if self.return_metadata and 'metadata' in batch:
                output['metadata'] = batch['metadata']
        if timing_enabled:
            self._sync_cuda_for_timing()
            t_combine = time.perf_counter()
            logger.info(
                "%s row %s timing: merge=%.3fs, factual=%.3fs, alternative=%.3fs, combine=%.3fs, total=%.3fs",
                self.timing_label,
                index + 1,
                t_merge - t0,
                t_factual - t_merge,
                t_alternative - t_factual,
                t_combine - t_alternative,
                t_combine - t0,
            )
        return output


class GradientTrainingDataset(SignalTrainingDatasetBase):
    """
    Backward-compatible dataset for raw-gradient GRADIEND training.

    This wrapper preserves the public constructor while delegating batching,
    caching, and source/target selection to SignalTrainingDatasetBase.
    """

    def __init__(
        self,
        training_data: Any,
        gradient_creator: Any,
        *,
        source: str = 'factual',
        target: str = 'diff',
        expand_encoder_eval_poles: bool = False,
        cache_dir: Optional[str] = None,
        use_cached_gradients: bool = True,
        cache_key_fields: Optional[List[str]] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        return_metadata: bool = False,
        get_padding_value: Optional[Callable[[str], int]] = None,
        timing_steps: int = 0,
        timing_label: str = "gradient",
        signal: Any = None,
        signals: Any = None,
        combine_diff_in_place: bool = False,
    ):
        gradient_signal = require_single_gradient_signal(
            signal=signal,
            signals=signals,
            context="GradientTrainingDataset",
        )
        self.gradient_creator = gradient_creator
        super().__init__(
            training_data,
            GradientSignalExtractor(gradient_creator, signal=gradient_signal),
            source=source,
            target=target,
            expand_encoder_eval_poles=expand_encoder_eval_poles,
            cache_dir=cache_dir,
            use_cached_signals=use_cached_gradients,
            cache_key_fields=cache_key_fields,
            dtype=dtype,
            device=device,
            return_metadata=return_metadata,
            get_padding_value=get_padding_value,
            timing_steps=timing_steps,
            timing_label=timing_label,
            signal=gradient_signal,
            combine_diff_in_place=combine_diff_in_place,
        )
        self.use_cached_gradients = self.use_cached_signals


class PreComputedTrainingDataset(torch.utils.data.IterableDataset):
    """
    Asynchronously precompute rows from another signal dataset.

    This wrapper is intentionally small: a single background thread evaluates the
    wrapped dataset sequentially and keeps a bounded queue of ready rows. It does
    not copy or wrap the model; it only overlaps the next base signal extraction
    with the current GRADIEND optimizer step.

    Safe when base forward/backward uses ModelWithGradiend.exclusive_base_gradient_access
    (default for built-in text models and in-training encoder eval).
    """

    def __init__(self, dataset: Any, *, buffer_size: int = 1):
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {buffer_size}")
        self.dataset = dataset
        self.buffer_size = int(buffer_size)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        q: "queue.Queue[Any]" = queue.Queue(maxsize=self.buffer_size)
        stop_event = threading.Event()
        sentinel = object()

        def _put(value: Any) -> None:
            while not stop_event.is_set():
                try:
                    q.put(value, timeout=0.1)
                    return
                except queue.Full:
                    continue

        def _worker() -> None:
            try:
                for idx in range(len(self.dataset)):
                    if stop_event.is_set():
                        break
                    _put((True, self.dataset[idx]))
            except BaseException as e:
                _put((False, e))
            finally:
                _put((None, sentinel))

        worker = threading.Thread(
            target=_worker,
            name="gradiend-signal-precompute",
            daemon=True,
        )
        worker.start()

        try:
            while True:
                ok, payload = q.get()
                if ok is None:
                    break
                if ok is False:
                    raise payload
                yield payload
        finally:
            stop_event.set()
            worker.join(timeout=1.0)
