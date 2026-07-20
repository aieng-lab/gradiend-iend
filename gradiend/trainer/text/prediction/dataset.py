"""
Prediction datasets: TextBatchedDataset, TextTrainingDataset, create_masked_pair_from_text.
"""

import random
from typing import Any, List, Optional, Tuple

import pandas as pd
import torch

from gradiend.model.core.seq2seq_backbone import SEQ2SEQ_ENCODER_MLM
from gradiend.trainer.text.prediction.seq2seq import (
    SEQ2SEQ_DECODER_SEQUENCE_CLOZE,
    create_seq2seq_decoder_item,
    create_seq2seq_decoder_sequence_item,
    create_seq2seq_mlm_item,
    mask_placeholder_for_tokenizer,
)
from gradiend.trainer.text.common.dataset_base import TextBatchedDatasetBase
from gradiend.trainer.core.dataset import SignalTrainingDatasetBase
from gradiend.trainer.core.unified_schema import (
    UNIFIED_ALTERNATIVE,
    UNIFIED_FACTUAL,
    UNIFIED_MASKED,
    UNIFIED_SPLIT,
)
from gradiend.trainer.core.signals import Signal
from gradiend.util import normalize_split_name
from gradiend.util.logging import suppress_tokenizer_length_warning


def _ids_for_text(tokenizer: Any, text: str) -> List[int]:
    """Tokenize ``text`` without special tokens and return token ids.

    Args:
        tokenizer: Tokenizer used for encoding.
        text: Text to tokenize.
    """
    return tokenizer(str(text), add_special_tokens=False, padding=False)["input_ids"]


def _continuation_ids_from_prefix(tokenizer: Any, prefix: str, continuation: str) -> List[int]:
    """Return token ids contributed by ``continuation`` after ``prefix``.

    Args:
        tokenizer: Tokenizer used for encoding.
        prefix: Text before the continuation.
        continuation: Candidate continuation text.
    """
    prefix_ids = _ids_for_text(tokenizer, prefix)
    full_ids = _ids_for_text(tokenizer, prefix + continuation)
    if full_ids[: len(prefix_ids)] == prefix_ids:
        return full_ids[len(prefix_ids) :]
    full_ids = _ids_for_text(tokenizer, prefix + " " + continuation)
    if full_ids[: len(prefix_ids)] == prefix_ids:
        return full_ids[len(prefix_ids) :]
    return _ids_for_text(tokenizer, continuation)


def _find_subsequence(values: List[int], needle: List[int], start: int = 0) -> int:
    """Return the first index of ``needle`` in ``values`` at or after ``start``."""
    if not needle:
        return -1
    max_start = len(values) - len(needle)
    for idx in range(max(0, start), max_start + 1):
        if values[idx : idx + len(needle)] == needle:
            return idx
    return -1


def _prediction_positions_for_filled_text(
    tokenizer: Any,
    *,
    template: str,
    target: str,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> List[int]:
    """Locate the target span inserted into a filled prediction template."""
    if "[MASK]" not in template:
        raise ValueError("Activation prediction-position selection requires a template with [MASK].")
    prefix, _suffix = str(template).split("[MASK]", 1)
    prefix_ids = _ids_for_text(tokenizer, prefix)
    target_ids = _continuation_ids_from_prefix(tokenizer, prefix, str(target))
    if not target_ids:
        target_ids = _ids_for_text(tokenizer, str(target))
    if not target_ids:
        raise ValueError(f"Could not tokenize prediction target {target!r}.")

    ids = [int(v) for v in input_ids.tolist()]
    if attention_mask is not None:
        valid_len = int(attention_mask.to(dtype=torch.long).sum().item())
        ids = ids[:valid_len]

    prefix_end = 0
    if prefix_ids:
        prefix_start = _find_subsequence(ids, [int(v) for v in prefix_ids])
        if prefix_start >= 0:
            prefix_end = prefix_start + len(prefix_ids)

    start = _find_subsequence(ids, [int(v) for v in target_ids], start=prefix_end)
    if start < 0:
        start = _find_subsequence(ids, [int(v) for v in target_ids])
    if start < 0:
        raise ValueError(
            f"Could not locate filled prediction target {target!r} in tokenized template {template!r}."
        )
    return list(range(start, start + len(target_ids)))


def _stack_text_items(items: List[dict]) -> dict:
    """Stack same-shaped tokenized text items into one batch dictionary."""
    if len(items) == 1:
        return items[0]
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def _targetable_token_text(token: str) -> str:
    """Normalize tokenizer artifacts before testing whether a token is targetable.

    Args:
        token: Tokenizer token string.
    """
    token = str(token).strip()
    return token.lstrip("#").lstrip("Ġ").lstrip("▁").strip()


def create_masked_pair_from_text(
    text: str,
    tokenizer: Any,
    is_decoder_only_model: bool,
    excluded_tokens: Optional[List[str]] = None,
    mask_token: Optional[str] = None,
    min_prefix_tokens: int = 5,
) -> Optional[Tuple[str, str]]:
    """Create a single (masked_text, target_token) pair from raw text.

    For MLM models: masks one random non-special token. For decoder-only models:
    uses prefix up to a random position, inserts [MASK], and uses the next token
    as target.

    Args:
        text: Raw input text to create a masked example from.
        tokenizer: Tokenizer for encoding and decoding.
        is_decoder_only_model: If True, uses prefix+[MASK] mode; else uses MLM mask mode.
        excluded_tokens: Tokens to avoid masking (e.g., target class tokens).
        mask_token: Mask token string for MLM (e.g., "[MASK]").
        min_prefix_tokens: Minimum prefix length for decoder-only mode.

    Returns:
        Tuple of (masked_text, target_token), or None if no valid pair could be created.
    """
    excluded_tokens = excluded_tokens or []
    tokens = tokenizer.tokenize(text)
    if not tokens:
        return None
    if not is_decoder_only_model and mask_token:
        valid_indices = [
            i for i, token in enumerate(tokens)
            if _targetable_token_text(token)
            and not (
                token.startswith("[") and token.endswith("]")
                or (excluded_tokens and any(excl.lower() in token.lower() for excl in excluded_tokens))
            )
        ]
        if not valid_indices:
            return None
        mask_idx = random.choice(valid_indices)
        target_token = tokens[mask_idx]
        tokens[mask_idx] = mask_token
        masked_text = tokenizer.convert_tokens_to_string(tokens)
        return (masked_text, target_token)
    if len(tokens) < 2:
        return None
    valid_k = [
        k for k in range(min_prefix_tokens, len(tokens))
        if _targetable_token_text(tokens[k])
        and not (
            tokens[k].startswith("[") and tokens[k].endswith("]")
            or (excluded_tokens and any(excl.lower() in tokens[k].lower() for excl in excluded_tokens))
        )
    ]
    if not valid_k:
        return None
    split_at = random.choice(valid_k)
    prefix_tokens = tokens[:split_at]
    true_next_token = tokens[split_at]
    prefix_str = tokenizer.convert_tokens_to_string(prefix_tokens)
    next_str = tokenizer.convert_tokens_to_string([true_next_token])
    masked_text = prefix_str + (" [MASK]" if next_str and (next_str[0] in " \t" or next_str[0] == "\u2581") else "[MASK]")
    return (masked_text, true_next_token)


class TextBatchedDataset(TextBatchedDatasetBase):
    """Prediction batched dataset with mask token support for MLM/decoder-only models.

    Adds mask_token and mask_token_id from the tokenizer and implements _create_item
    to produce input_ids, attention_mask, and labels for training.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: Any,
        batch_size: int,
        batch_criterion: Any,
        max_size: Optional[int] = None,
        seed: int = 42,
        shuffle_batches: Optional[bool] = None,
        max_length: int = 256,
        balance_column: Optional[str] = None,
        shuffle_within: Optional[bool] = None,
    ):
        """Initialize the batched dataset.

        Args:
            data: DataFrame with text/label data.
            tokenizer: Tokenizer for encoding.
            batch_size: Batch size.
            batch_criterion: Criterion for batching (e.g., target key).
            max_size: Optional maximum number of samples.
            seed: Random seed.
            shuffle_batches: Whether to shuffle batches.
            max_length: Maximum sequence length.
            balance_column: Column for balancing batches.
            shuffle_within: Whether to shuffle within batches.
        """
        super().__init__(
            data=data,
            tokenizer=tokenizer,
            batch_size=batch_size,
            batch_criterion=batch_criterion,
            max_size=max_size,
            seed=seed,
            shuffle_batches=shuffle_batches,
            max_length=max_length,
            balance_column=balance_column,
            shuffle_within=shuffle_within,
        )
        self.mask_token = getattr(tokenizer, "mask_token", None)
        self.mask_token_id = getattr(tokenizer, "mask_token_id", None)

    def _create_item(self, text: str, target: str):
        """Create a single training item: input_ids, attention_mask, labels.

        For decoder-only: labels at last non-pad position. For MLM: labels at mask positions.

        Args:
            text: Input text or template used for the prediction objective.
            target: Target token/text to predict.
        """
        is_decoder_only_model = getattr(self, "is_decoder_only_model", False)
        prediction_objective = getattr(self, "prediction_objective", None)
        if prediction_objective == "clm_mlm_head":
            target_labels = getattr(self, "mlm_head_target_labels", None)
            if not target_labels:
                raise ValueError(
                    "clm_mlm_head training requires mlm_head_target_labels on the dataset "
                    "(load from the trained decoder MLM head checkpoint)."
                )
            if not self.mask_token:
                raise ValueError("clm_mlm_head requires tokenizer.mask_token.")
            if "[MASK]" not in text:
                raise ValueError("clm_mlm_head training requires a [MASK] placeholder in the input text.")
            expanded_text = text.replace("[MASK]", self.mask_token, 1)
            with suppress_tokenizer_length_warning():
                encoded = self.tokenizer(
                    expanded_text,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                )
            label_map = {str(lab).strip(): idx for idx, lab in enumerate(target_labels)}
            target_str = str(target).strip()
            if target_str not in label_map:
                raise ValueError(
                    f"Target {target_str!r} is not a decoder MLM-head label. Known labels: {target_labels}"
                )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            labels = torch.tensor([label_map[target_str]], dtype=torch.long)
            return {
                "input_ids": input_ids.squeeze(0),
                "attention_mask": attention_mask.squeeze(0),
                "labels": labels,
            }
        if prediction_objective == "clm_sequence_cloze":
            if "[MASK]" not in text:
                raise ValueError("clm_sequence_cloze training requires a [MASK] placeholder.")
            prefix, rhs = text.split("[MASK]", 1)
            expanded_text = text.replace("[MASK]", str(target), 1)
            with suppress_tokenizer_length_warning():
                encoded = self.tokenizer(expanded_text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=self.max_length, padding="max_length")
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            labels = torch.full_like(input_ids, -100)
            prefix_len = len(_ids_for_text(self.tokenizer, prefix))
            candidate_len = len(_continuation_ids_from_prefix(self.tokenizer, prefix, str(target)))
            rhs_window = getattr(self, "rhs_window", -1)
            valid_len = int(attention_mask.sum(dim=1)[0].item())
            if rhs_window is None or rhs_window < 0:
                end = valid_len
            else:
                end = min(valid_len, prefix_len + candidate_len + int(rhs_window))
            start = min(prefix_len, valid_len)
            if start < end:
                labels[:, start:end] = input_ids[:, start:end]
            return {"input_ids": input_ids.squeeze(0), "attention_mask": attention_mask.squeeze(0), "labels": labels.squeeze(0)}
        is_seq2seq_model = getattr(self, "is_seq2seq_model", False)
        if is_seq2seq_model:
            prediction_objective = getattr(self, "prediction_objective", None)
            common = dict(
                masked_text=text,
                label=str(target),
                tokenizer=self.tokenizer,
                base_model=getattr(self, "base_model", None),
            )
            if prediction_objective == SEQ2SEQ_ENCODER_MLM:
                item = create_seq2seq_mlm_item(**common)
            elif prediction_objective == SEQ2SEQ_DECODER_SEQUENCE_CLOZE:
                item = create_seq2seq_decoder_sequence_item(
                    **common,
                    rhs_window=getattr(self, "rhs_window", -1),
                )
            else:
                item = create_seq2seq_decoder_item(**common)
            return {k: v.squeeze(0) for k, v in item.items()}
        if not is_decoder_only_model and self.mask_token and self.mask_token not in text:
            raise ValueError("Input text must contain at least one [MASK] token placeholder.")
        mask_count = 0 if is_decoder_only_model else (text.count(self.mask_token) if self.mask_token else 0)
        target_tokens = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        num_target_tokens = len(target_tokens)
        if is_decoder_only_model:
            expanded_text = text.split("[MASK]")[0] if "[MASK]" in text else text
        else:
            expanded_text = text.replace(self.mask_token, " ".join([self.mask_token] * num_target_tokens), 1) if mask_count == 1 and self.mask_token else text
        with suppress_tokenizer_length_warning():
            encoded = self.tokenizer(expanded_text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = torch.full_like(input_ids, -100)
        if is_decoder_only_model:
            last_idxs = attention_mask.sum(dim=1)
            if hasattr(self.tokenizer, "vocab") and target in self.tokenizer.vocab:
                target_idx = self.tokenizer.vocab[target]
            else:
                target_idx = self.tokenizer(target, add_special_tokens=False)["input_ids"][0]
            for i, last_idx in enumerate(last_idxs):
                if last_idx < input_ids.size(1):
                    labels[i, last_idx - 1] = target_idx
        else:
            mask_token_id = self.tokenizer.convert_tokens_to_ids(self.mask_token)
            mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=False)
            for i, idx in enumerate(mask_positions):
                b, pos = idx.tolist()
                labels[b, pos] = target_tokens[min(i, len(target_tokens) - 1)]
        return {"input_ids": input_ids.squeeze(0), "attention_mask": attention_mask.squeeze(0), "labels": labels.squeeze(0)}


class TextTrainingDataset(TextBatchedDataset):
    """Text training dataset with dict interface for TextGradientTrainingDataset.

    Produces batches with factual and alternative items for gradient-based training,
    including template, input_text, label, and token ids for both options.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: Any,
        batch_size: int,
        is_decoder_only_model: bool = False,
        is_seq2seq_model: bool = False,
        max_size: Optional[int] = None,
        target_key: str = "label",
        balance_column: str = "feature_class_id",
        max_length: int = 256,
        seed: Optional[int] = None,
        prediction_objective: Optional[str] = None,
        rhs_window: int = -1,
        mlm_head_target_labels: Optional[List[str]] = None,
    ):
        """Initialize the training dataset.

        Args:
            data: DataFrame with masked, factual, alternative columns (unified schema).
            tokenizer: Tokenizer for encoding.
            batch_size: Batch size.
            is_decoder_only_model: If True, uses decoder-only ([MASK] after prefix) format.
            is_seq2seq_model: If True, creates encoder-decoder style inputs.
            max_size: Optional maximum number of samples.
            target_key: Key for label/target in data.
            balance_column: Column for balancing (e.g., feature_class_id).
            max_length: Maximum sequence length.
            seed: Random seed for batch ordering (default 42). Use training_args.seed for reproducibility.
            prediction_objective: Optional prediction objective override, such as
                cloze-sequence or seq2seq objectives.
            rhs_window: Right-context token window for sequence-cloze objectives.
        """
        super().__init__(
            data=data,
            tokenizer=tokenizer,
            batch_size=batch_size,
            batch_criterion=target_key,
            max_size=max_size,
            balance_column=balance_column,
            max_length=max_length,
            seed=seed if seed is not None else 42,
        )
        self.target_key = target_key
        self.is_decoder_only_model = is_decoder_only_model
        self.is_seq2seq_model = is_seq2seq_model
        self.prediction_objective = prediction_objective
        self.mlm_head_target_labels = list(mlm_head_target_labels) if mlm_head_target_labels else None
        self.rhs_window = rhs_window
        if is_decoder_only_model and getattr(self.tokenizer, "pad_token", None) is None:
            eos_token = getattr(self.tokenizer, "eos_token", None)
            if eos_token is not None:
                self.tokenizer.pad_token = eos_token

    def __getitem__(self, idx: int):
        """Return a single item with factual/alternative pairs and gradient-ready structure.

        Args:
            idx: Row or batch index resolved by the parent batched dataset.
        """
        entry = super().__getitem__(idx)
        template = entry[UNIFIED_MASKED]
        if self.prediction_objective == "clm_sequence_cloze":
            input_text = template
        elif self.prediction_objective == SEQ2SEQ_DECODER_SEQUENCE_CLOZE:
            input_text = template
        elif self.is_decoder_only_model:
            if self.prediction_objective == "clm_mlm_head":
                input_text = template
            else:
                input_text = template.split("[MASK]")[0] if "[MASK]" in template else template
        elif self.is_seq2seq_model:
            input_text = mask_placeholder_for_tokenizer(template, self.tokenizer)
        elif self.mask_token:
            input_text = template.replace("[MASK]", self.mask_token)
        else:
            input_text = template
        text = template.replace("[MASK]", entry[UNIFIED_FACTUAL])
        label = entry[self.target_key]
        try:
            import numpy as _np
            if isinstance(label, _np.generic):
                label = int(label)
        except Exception:
            pass
        item_factual = self._create_item(input_text, entry[UNIFIED_FACTUAL])
        item_alternative = self._create_item(input_text, entry[UNIFIED_ALTERNATIVE])
        out = {
            "factual": item_factual,
            "alternative": item_alternative,
            "text": text,
            "template": template,
            "input_text": input_text,
            "label": label,
            "factual_token": entry[UNIFIED_FACTUAL],
            "factual_id": entry["factual_id"],
            "alternative_token": entry[UNIFIED_ALTERNATIVE],
            "alternative_id": entry["alternative_id"],
        }
        if "feature_class_id" in entry:
            out["feature_class_id"] = entry["feature_class_id"]
        if UNIFIED_SPLIT in entry.index:
            out["data_split"] = normalize_split_name(str(entry[UNIFIED_SPLIT]))
        elif "split" in entry.index:
            out["data_split"] = normalize_split_name(str(entry["split"]))
        return out


class TextActivationTrainingDataset(SignalTrainingDatasetBase):
    """
    Text-prediction activation signal dataset.

    Unlike gradient signals, plain activations are not label-conditioned. This
    wrapper therefore constructs factual/alternative inputs by filling the
    prediction slot before activation extraction, and adds ``prediction_mask``
    so token_selector="prediction" can select the filled span.
    """

    CACHE_KEY_FIELDS: List[str] = ["template", "factual_token", "alternative_token", "label"]

    def __init__(
        self,
        training_data: Any,
        tokenizer: Any,
        signal_extractor: Any,
        *,
        source: str = "factual",
        target: str = "diff",
        cache_dir: Optional[str] = None,
        use_cached_signals: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        return_metadata: bool = False,
        timing_steps: int = 0,
        timing_label: str = "text-activation",
        signal: Any = None,
        signals: Any = None,
    ):
        if signal is not None:
            signal = self.default_signal(signal)
        elif getattr(signal_extractor, "signal", None) is not None:
            signal = self.default_signal(signal_extractor.signal)
        pad_token_id = getattr(tokenizer, "pad_token_id", 0) if tokenizer is not None else 0

        def get_padding_value(subkey: str) -> int:
            return pad_token_id if "input_ids" in subkey else 0

        super().__init__(
            training_data,
            signal_extractor,
            source=source,
            target=target,
            cache_dir=cache_dir,
            use_cached_signals=use_cached_signals,
            cache_key_fields=self.CACHE_KEY_FIELDS if (cache_dir and use_cached_signals) else None,
            dtype=dtype,
            device=device,
            return_metadata=return_metadata,
            get_padding_value=get_padding_value,
            timing_steps=timing_steps,
            timing_label=timing_label,
            signal=signal,
            signals=signals,
        )
        self.tokenizer = tokenizer

    @staticmethod
    def default_signal(signal: Signal) -> Signal:
        """Text-prediction default for activation signals: select the prediction span."""
        if signal.kind != "activation":
            return signal
        options = dict(signal.options or {})
        if options.get("token_selector") is None:
            options["token_selector"] = "prediction"
            return Signal(signal.kind, name=signal.name, options=options)
        return signal

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _create_filled_prediction_item(self, template: str, target_token: Any) -> dict:
        template = str(template)
        target_token = str(target_token)
        if "[MASK]" not in template:
            raise ValueError("Text activation training requires templates with a [MASK] prediction slot.")
        filled_text = template.replace("[MASK]", target_token, 1)
        max_length = getattr(self.training_data, "max_length", 256)
        if not isinstance(max_length, int) or max_length <= 0:
            max_length = 256
        with suppress_tokenizer_length_warning():
            encoded = self.tokenizer(
                filled_text,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(0)
        positions = _prediction_positions_for_filled_text(
            self.tokenizer,
            template=template,
            target=target_token,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        prediction_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for position in positions:
            if 0 <= position < prediction_mask.numel():
                prediction_mask[position] = True
        if not bool(prediction_mask.any().item()):
            raise ValueError(
                f"Filled prediction target {target_token!r} was truncated out of template {template!r}."
            )
        item = {
            "input_ids": input_ids,
            "prediction_mask": prediction_mask,
        }
        if attention_mask is not None:
            item["attention_mask"] = attention_mask
        return item

    def _filled_side_batch(self, templates: List[Any], target_tokens: List[Any]) -> dict:
        items = [
            self._create_filled_prediction_item(template, target_token)
            for template, target_token in zip(templates, target_tokens)
        ]
        return _stack_text_items(items)

    def _merge_batch(self, indices: list) -> dict:
        batch = super()._merge_batch(indices)
        templates = self._as_list(batch.get("template"))
        factual_tokens = self._as_list(batch.get("factual_token"))
        alternative_tokens = self._as_list(batch.get("alternative_token"))
        if not (len(templates) == len(factual_tokens) == len(alternative_tokens)):
            raise ValueError(
                "Text activation batch metadata lengths must match for template, factual_token, and alternative_token."
            )
        batch["factual"] = self._filled_side_batch(templates, factual_tokens)
        batch["alternative"] = self._filled_side_batch(templates, alternative_tokens)
        return batch
