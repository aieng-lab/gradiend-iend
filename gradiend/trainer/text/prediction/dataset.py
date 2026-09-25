"""
Prediction datasets: TextBatchedDataset, TextTrainingDataset, create_masked_pair_from_text.
"""

import random
import re
from typing import Any, List, Optional, Sequence, Tuple, Union

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
from gradiend.util.positions import (
    assert_labels_on_real_tokens,
    first_real_token_positions,
    last_real_token_positions,
)
from gradiend.util.tokenization import offset_pairs, tokenize_with_offsets

# Dataset-level cloze placeholder (matches TextFilterConfig.mask). Independent of
# tokenizer.mask_token, which is an MLM model special and may be absent (e.g. GPT-2).
DEFAULT_DATASET_MASK_PLACEHOLDER = "[MASK]"


def missing_mask_placeholder_message(template: str, mask_placeholder: str) -> str:
    """Explain a template/placeholder mismatch (common when data uses ``[PRONOUN]``)."""
    text = str(template)
    preview = text if len(text) <= 120 else f"{text[:117]}..."
    return (
        f"Masked template does not contain mask_placeholder={mask_placeholder!r}. "
        f"Got template={preview!r}. "
        f"Set TextPredictionConfig.mask_placeholder (or TrainingArguments.mask_placeholder) "
        f"to match the prediction slot in your data (e.g. '[PRONOUN]')."
    )


def require_mask_placeholder(template: str, mask_placeholder: str) -> str:
    """Return ``template`` or raise if ``mask_placeholder`` is missing."""
    text = str(template)
    placeholder = str(mask_placeholder)
    if placeholder not in text:
        raise ValueError(missing_mask_placeholder_message(text, placeholder))
    return text


def validate_masked_templates_in_dataframe(
    data: pd.DataFrame,
    mask_placeholder: str,
    *,
    masked_col: str = UNIFIED_MASKED,
) -> None:
    """Fail fast when masked templates omit the configured prediction slot."""
    if masked_col not in data.columns or len(data) == 0:
        return
    masked = data[masked_col].astype(str)
    placeholder = str(mask_placeholder)
    missing = ~masked.str.contains(re.escape(placeholder), regex=True)
    if not bool(missing.any()):
        return
    first_bad = masked[missing].iloc[0]
    raise ValueError(missing_mask_placeholder_message(first_bad, placeholder))


def resolve_mask_placeholder(
    *,
    config: Any = None,
    training_args: Any = None,
    default: str = DEFAULT_DATASET_MASK_PLACEHOLDER,
) -> str:
    """Resolve the dataset prediction-slot marker from config and/or training args.

    ``TextPredictionConfig.mask_placeholder`` is the data-schema field. A non-default
    value on either side is accepted; conflicting non-default values raise.
    """
    default = str(default) if default else DEFAULT_DATASET_MASK_PLACEHOLDER

    def _as_placeholder(value: Any) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            return None
        text = value.strip() if hasattr(value, "strip") else value
        return text or None

    config_ph = _as_placeholder(getattr(config, "mask_placeholder", None)) if config is not None else None
    args_ph = (
        _as_placeholder(getattr(training_args, "mask_placeholder", None))
        if training_args is not None
        else None
    )
    config_ph = config_ph or default
    args_ph = args_ph or default
    if config_ph != default and args_ph != default and config_ph != args_ph:
        raise ValueError(
            f"Conflicting mask_placeholder: TextPredictionConfig has {config_ph!r} "
            f"but TrainingArguments has {args_ph!r}."
        )
    if config_ph != default:
        return config_ph
    return args_ph


def _tokenize_classic_mlm_site_targets(
    tokenizer: Any,
    target: Union[str, Sequence[Any]],
    mask_count: int,
) -> List[List[int]]:
    """Resolve classic MLM targets into one token-id list per prediction site.

    A single string target is broadcast to every site. A sequence of targets must
    have length ``mask_count`` (one target per ``[MASK]`` site); a length mismatch
    raises ``ValueError`` with an actionable message.
    """
    if isinstance(target, (list, tuple)):
        site_texts = list(target)
        if len(site_texts) != mask_count:
            raise ValueError(
                "Classic MLM per-site targets must match the number of prediction "
                f"placeholders: got {len(site_texts)} target(s) for {mask_count} "
                f"placeholder(s). Pass one shared target string to broadcast to all "
                f"sites, or a sequence of length {mask_count}. targets={site_texts!r}."
            )
    else:
        site_texts = [target] * mask_count

    site_token_lists: List[List[int]] = []
    for site_idx, site_target in enumerate(site_texts):
        token_ids = tokenizer(str(site_target), add_special_tokens=False, padding=False)["input_ids"]
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        if not token_ids:
            raise ValueError(
                f"Could not tokenize MLM target {site_target!r} for prediction site {site_idx}."
            )
        site_token_lists.append(list(token_ids))
    return site_token_lists


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


def decoder_only_label_token_ids(tokenizer: Any, label: Any) -> List[int]:
    """Single source of truth: token ids that label a decoder-only prediction.

    The label is written onto the LAST prefix token (the template's trailing space)
    and, because causal-LM loss shifts by one, predicted from the token before it.
    The natural next token after a word boundary is the *leading-space* variant
    (``▁she`` / ``Ġshe``), so that is the label: ``tokenizer(" " + label)``.
    ``vocab[label]`` (``she``) is the mid-word variant and is only kept for
    ``label_token_protocol="legacy"``.

    ``TextPredictionModelWithGradiend.create_inputs`` and the training dataset must
    both call this; it returns all ids so callers can apply their own multi-token
    policy (``create_inputs`` rejects them, the dataset uses the first).
    """
    return list(tokenizer(f" {label}", add_special_tokens=False)["input_ids"])


def training_label_token_id(tokenizer: Any, label: Any, protocol: str = "legacy") -> int:
    """The one token id a decoder-only *training* item supervises for ``label``.

    ``protocol`` is ``TrainingArguments.label_token_protocol``:

    - ``"canonical"``: :func:`decoder_only_label_token_ids` (leading-space variant),
      first id for multi-token labels.
    - ``"legacy"``: ``vocab[label]`` (else the first id of ``tokenizer(label)``), the
      rule every artifact trained before 2026-09-25 used.

    The dataset and single-row scorers (e.g. CGA detection) must both resolve the
    label through this function so their gradients live in the same signal space.
    """
    if protocol == "canonical":
        return int(decoder_only_label_token_ids(tokenizer, label)[0])
    if protocol != "legacy":
        raise ValueError(f"label_token_protocol must be 'canonical' or 'legacy', got {protocol!r}")
    label = str(label)
    if hasattr(tokenizer, "vocab") and label in tokenizer.vocab:
        return int(tokenizer.vocab[label])
    return int(tokenizer(label, add_special_tokens=False)["input_ids"][0])


def _find_subsequence(values: List[int], needle: List[int], start: int = 0) -> int:
    """Return the first index of ``needle`` in ``values`` at or after ``start``."""
    if not needle:
        return -1
    max_start = len(values) - len(needle)
    for idx in range(max(0, start), max_start + 1):
        if values[idx : idx + len(needle)] == needle:
            return idx
    return -1


def _target_token_ids_for_fill(tokenizer: Any, target: str, *, prefix: str = "") -> List[int]:
    """Resolve a fill target to one or more vocabulary ids.

    Single vocab pieces (including WordPiece ``##...``) are preferred so they are
    inserted by id. Otherwise the target is tokenized as a surface continuation
    after ``prefix``.
    """
    target = str(target)
    single = _single_vocab_token_id(tokenizer, target)
    if single is not None:
        return [single]
    target_ids = _continuation_ids_from_prefix(tokenizer, prefix, target)
    if not target_ids:
        target_ids = _ids_for_text(tokenizer, target)
    return [int(v) for v in target_ids]


def _pad_1d(
    values: List[int],
    *,
    max_length: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncate/pad ``values`` to ``max_length`` and build a matching attention mask."""
    clipped = values[:max_length]
    attention = [1] * len(clipped) + [0] * (max_length - len(clipped))
    padded = clipped + [pad_id] * (max_length - len(clipped))
    return (
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(attention, dtype=torch.long),
    )


def _char_spans_for_substring(text: str, substring: str) -> List[Tuple[int, int]]:
    """Return inclusive-exclusive character spans of every ``substring`` in ``text``."""
    spans: List[Tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(substring, start)
        if idx < 0:
            break
        spans.append((idx, idx + len(substring)))
        start = idx + len(substring)
    return spans


def _token_spans_covering_char_spans(
    offset_mapping: List[Tuple[int, int]],
    char_spans: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Map character spans to inclusive-exclusive token index spans via offset mapping."""
    token_spans: List[Tuple[int, int]] = []
    for c_start, c_end in char_spans:
        token_indices = [
            i
            for i, (o_start, o_end) in enumerate(offset_mapping)
            if int(o_end) > int(o_start) and int(o_start) < c_end and int(o_end) > c_start
        ]
        if not token_indices:
            continue
        token_spans.append((token_indices[0], token_indices[-1] + 1))
    return token_spans


def _left_truncate_template_keeping_mask(
    tokenizer: Any,
    template: str,
    *,
    mask_placeholder: str,
    max_length: int,
) -> str:
    """Drop left (and unused right) context so ``[MASK]`` fits in ``max_length`` tokens.

    Naive ``truncation=True`` keeps the *start* of the string and can cut the mask
    off on long wiki templates. We instead keep a window that **ends at the first
    mask** (mask at the end of the window).

    The first, not the last: with more than one placeholder, anchoring on the
    last leaves the earlier ones in the prefix as literal ``[MASK]`` text, which
    the model then attends to as context. Truncating after the first gives a
    prefix with no stray placeholder in it. Single-placeholder templates -- the
    normal case -- are unaffected, since first and last coincide.
    """
    template = str(template)
    mask_placeholder = str(mask_placeholder)
    first = template.find(mask_placeholder)
    if first < 0:
        return template
    # RHS after the mask does not affect causal hidden states at the mask span,
    # and dropping it also removes any later placeholders.
    clipped = template[: first + len(mask_placeholder)]
    encoded, raw_offsets = tokenize_with_offsets(
        tokenizer, clipped, add_special_tokens=True, truncation=False
    )
    if raw_offsets is None:
        ids = encoded["input_ids"]
        if len(ids) <= int(max_length):
            return clipped
        approx_chars = max(len(mask_placeholder) + 8, int(max_length) * 3)
        start = max(0, len(clipped) - approx_chars)
        if start > first:
            start = first
        return clipped[start:]

    ids = list(encoded["input_ids"])
    offsets = offset_pairs(raw_offsets)
    if len(ids) <= int(max_length):
        return clipped

    char_spans = _char_spans_for_substring(clipped, mask_placeholder)
    token_spans = _token_spans_covering_char_spans(offsets, char_spans)
    if not token_spans:
        return clipped
    mask_end = token_spans[-1][1]  # exclusive
    start_tok = max(0, mask_end - int(max_length))
    char_start = 0
    for o_start, o_end in offsets[start_tok:]:
        if int(o_end) > int(o_start):
            char_start = int(o_start)
            break
    first_mask_char = char_spans[0][0]
    if char_start > first_mask_char:
        char_start = first_mask_char
    # Token offsets were computed with the original left context.  Removing that
    # context can change how a boundary piece is tokenized (notably for
    # SentencePiece tokenizers and a window beginning mid-word).  Consequently
    # the substring may be one or more tokens longer when it is tokenized again
    # below, which would let ``truncation=True`` remove the trailing mask.
    #
    # Recheck candidate windows using their *actual* tokenization.  Advancing to
    # later original token boundaries only removes left context, so it is safe
    # for a causal prediction site at the trailing mask.  Keep the first window
    # which fits to retain as much context as possible.
    candidate_starts = []
    for offset_start, offset_end in offsets[start_tok:]:
        if int(offset_end) <= int(offset_start):
            continue
        candidate_start = int(offset_start)
        if candidate_start > first_mask_char:
            candidate_start = first_mask_char
        if candidate_start not in candidate_starts:
            candidate_starts.append(candidate_start)
    if char_start not in candidate_starts:
        candidate_starts.insert(0, char_start)

    for candidate_start in candidate_starts:
        candidate = clipped[candidate_start:]
        candidate_ids = tokenizer(
            candidate,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        if len(candidate_ids) <= int(max_length):
            return candidate

    # The mask itself is guaranteed to fit in normal configurations.  If a
    # tokenizer makes even the shortest token-boundary suffix too long, return
    # that suffix and let the caller raise a precise mask-location error.
    return clipped[first_mask_char:]


def _locate_mask_spans_in_encoded(
    *,
    filled_ids: List[int],
    offsets: Optional[List[Tuple[int, int]]],
    char_spans: List[Tuple[int, int]],
    tokenizer: Any,
    mask_placeholder: str,
) -> List[Tuple[int, int]]:
    """Locate mask token spans via offsets, then id-subsequence fallbacks."""
    if offsets is not None:
        token_spans = _token_spans_covering_char_spans(offsets, char_spans)
        if len(token_spans) == len(char_spans):
            return token_spans

    placeholder_ids = [int(v) for v in _ids_for_text(tokenizer, mask_placeholder)]
    if placeholder_ids:
        token_spans = []
        search_from = 0
        for _ in char_spans:
            start = _find_subsequence(filled_ids, placeholder_ids, start=search_from)
            if start < 0:
                break
            token_spans.append((start, start + len(placeholder_ids)))
            search_from = start + len(placeholder_ids)
        if len(token_spans) == len(char_spans):
            return token_spans

    single_id = _single_vocab_token_id(tokenizer, mask_placeholder)
    if single_id is not None:
        token_spans = []
        search_from = 0
        for _ in char_spans:
            try:
                start = filled_ids.index(int(single_id), search_from)
            except ValueError:
                break
            token_spans.append((start, start + 1))
            search_from = start + 1
        if len(token_spans) == len(char_spans):
            return token_spans
    return []


def _mask_placeholder_token_spans(
    tokenizer: Any,
    template: str,
    *,
    mask_placeholder: str,
    max_length: int,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Tokenize a template that still contains the dataset mask and locate those slots.

    The locator is the **dataset** placeholder (``TextFilterConfig.mask``, usually
    ``"[MASK]"``), not ``tokenizer.mask_token``. Models without an MLM special token
    (e.g. GPT-2) must still fill activation templates.

    If the template is longer than ``max_length``, left context is dropped so the
    mask remains inside the window (near the end), instead of HF left-truncation
    which would discard the mask.
    """
    template = _left_truncate_template_keeping_mask(
        tokenizer,
        str(template),
        mask_placeholder=str(mask_placeholder),
        max_length=int(max_length),
    )
    char_spans = _char_spans_for_substring(template, mask_placeholder)
    if not char_spans:
        raise ValueError(missing_mask_placeholder_message(template, mask_placeholder))

    encode_kwargs = dict(
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    with suppress_tokenizer_length_warning():
        encoded, raw_offsets = tokenize_with_offsets(tokenizer, template, **encode_kwargs)

    filled_ids = [int(v) for v in encoded["input_ids"].squeeze(0).tolist()]
    pairs = offset_pairs(raw_offsets) if raw_offsets is not None else None

    token_spans = _locate_mask_spans_in_encoded(
        filled_ids=filled_ids,
        offsets=pairs,
        char_spans=char_spans,
        tokenizer=tokenizer,
        mask_placeholder=mask_placeholder,
    )
    if len(token_spans) == len(char_spans):
        return filled_ids, token_spans

    raise ValueError(
        f"Could not locate dataset mask placeholder {mask_placeholder!r} in tokenized template {template!r}."
    )


def _filled_prediction_from_template(
    tokenizer: Any,
    *,
    template: str,
    target: str,
    max_length: int = 128,
    mask_placeholder: str = DEFAULT_DATASET_MASK_PLACEHOLDER,
) -> dict:
    """Fill a dataset-mask template in token space and return model inputs + prediction_mask.

    The template uses the **dataset** mask placeholder (default ``"[MASK]"``, same as
    ``TextFilterConfig.mask``), which is model-agnostic. Filling never requires
    ``tokenizer.mask_token`` / ``mask_token_id`` (those are MLM specials, not data).
    This is intentional for ACTIEND training: if factual and alternative examples
    both kept the literal MLM mask token at the measured position, their activation
    inputs would be identical there and the factual-vs-alternative activation
    signal would collapse to zero. Filling the slot with the candidate token makes
    the measured hidden state target-conditioned while ``prediction_mask`` keeps
    track of the exact span that was filled.

    Steps:

      1. Tokenize the template with the dataset mask still present; locate each slot.
      2. Resolve ``target`` to vocab id(s).
      3. Splice those id(s) over each mask slot (templates may contain several).
      4. Mark every spliced span in ``prediction_mask``.

    This avoids string-replacing tokenizer artifacts (e.g. ``##ver``) into surface text.
    """
    template = require_mask_placeholder(template, mask_placeholder)
    target = str(target)
    mask_placeholder = str(mask_placeholder)
    # Keep a window ending at [MASK] so long wiki templates are not left-truncated
    # in a way that drops the placeholder (see _left_truncate_template_keeping_mask).
    template = _left_truncate_template_keeping_mask(
        tokenizer,
        template,
        mask_placeholder=mask_placeholder,
        max_length=max_length,
    )

    filled_ids, mask_spans = _mask_placeholder_token_spans(
        tokenizer,
        template,
        mask_placeholder=mask_placeholder,
        max_length=max_length,
    )
    prefix, _suffix = template.split(mask_placeholder, 1)
    target_ids = _target_token_ids_for_fill(tokenizer, target, prefix=prefix)
    if not target_ids:
        raise ValueError(f"Could not tokenize prediction target {target!r}.")

    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)

    # Splice from the right so earlier indices stay valid.
    prediction_spans: List[Tuple[int, int]] = []
    for start, end in reversed(mask_spans):
        filled_ids = filled_ids[:start] + target_ids + filled_ids[end:]
        prediction_spans.append((start, start + len(target_ids)))

    input_ids, attention_mask = _pad_1d(filled_ids, max_length=max_length, pad_id=pad_id)
    prediction_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for start, end in prediction_spans:
        clipped_start = min(max(start, 0), max_length)
        clipped_end = min(max(end, 0), max_length)
        if clipped_end > clipped_start:
            prediction_mask[clipped_start:clipped_end] = True
    if not bool(prediction_mask.any().item()):
        raise ValueError(
            f"Filled prediction target {target!r} was truncated out of template {template!r}."
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prediction_mask": prediction_mask,
    }

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


def _is_wordpiece_continuation(token: str) -> bool:
    """Return True for BERT-style WordPiece continuation pieces (``##...``)."""
    return str(token).startswith("##")


def _single_vocab_token_id(tokenizer: Any, token: str) -> Optional[int]:
    """Return the tokenizer id when ``token`` is a single vocabulary piece.

    Used to fill ``[MASK]`` by id (required for WordPiece targets like ``##ver``),
    instead of string-replacing tokenizer artifacts into surface text.
    """
    token = str(token)
    if not token or not hasattr(tokenizer, "convert_tokens_to_ids"):
        return None
    pieces = None
    if hasattr(tokenizer, "tokenize"):
        try:
            pieces = tokenizer.tokenize(token)
        except Exception:
            pieces = None
    if pieces is not None and pieces != [token]:
        return None
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return None
    if token_id is None:
        return None
    token_id = int(token_id)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and token_id == int(unk_id):
        return None
    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_token = getattr(tokenizer, "pad_token", None)
    if pad_id is not None and token_id == int(pad_id) and token != pad_token:
        return None
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        try:
            recovered = tokenizer.convert_ids_to_tokens(token_id)
        except Exception:
            recovered = None
        if isinstance(recovered, (list, tuple)):
            recovered = recovered[0] if recovered else None
        if recovered is not None and str(recovered) != token:
            return None
    return token_id


def create_masked_pair_from_text(
    text: str,
    tokenizer: Any,
    is_decoder_only_model: bool,
    excluded_tokens: Optional[List[str]] = None,
    mask_token: Optional[str] = None,
    min_prefix_tokens: int = 5,
    mask_placeholder: str = DEFAULT_DATASET_MASK_PLACEHOLDER,
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
        mask_token: The tokenizer's MLM special, used only to decide whether the
            in-place masking branch applies and to detect already-masked input.
            It is NOT written into the template -- see ``mask_placeholder``.
        min_prefix_tokens: Minimum prefix length for decoder-only mode.
        mask_placeholder: Dataset-schema placeholder written into the returned
            template. Masked templates carry the schema placeholder at every
            stage; substituting ``tokenizer.mask_token`` is a tokenization-stage
            concern handled later (``mask_placeholder_for_tokenizer``) and must
            not leak into the dataframe, or
            ``validate_masked_templates_in_dataframe`` rejects rows this
            function itself produced.

    Returns:
        Tuple of (masked_text, target_token), or None if no valid pair could be created.
    """
    # Guard against silent corruption: if `text` already contains a mask
    # placeholder (caller passed an already-masked string, e.g. neutral_data
    # built with `masked`==`text`, with no companion label/token column to
    # take the pre-built-pair fast path), tokenizing and re-splitting it here
    # can land a random split point *inside* the placeholder's own BPE
    # sub-tokens (e.g. "[MASK]" -> "[", "MAS", "K", "]" for gpt2), producing
    # nonsense pairs like ("...is [MAS[MASK]", "K") with no error anywhere.
    # Fail loud instead of training on that.
    placeholder_candidates = {"[MASK]"}
    if mask_token:
        placeholder_candidates.add(str(mask_token))
    for placeholder in placeholder_candidates:
        if placeholder and placeholder in text:
            raise ValueError(
                f"create_masked_pair_from_text: `text` already contains "
                f"{placeholder!r} — this function expects raw, unmasked text "
                f"and derives its own mask position/target from it. Passing "
                f"pre-masked text here silently corrupts the pair (a random "
                f"split point can land inside the placeholder's own "
                f"sub-tokens). If you already have a masked/target pair, "
                f"provide a label/factual/token column on the DataFrame so "
                f"the caller uses it directly instead of calling this "
                f"function."
            )
    excluded_tokens = excluded_tokens or []
    tokens = tokenizer.tokenize(text)
    if not tokens:
        return None
    if not is_decoder_only_model and mask_token:
        valid_indices = [
            i for i, token in enumerate(tokens)
            if _targetable_token_text(token)
            and not _is_wordpiece_continuation(token)
            and not (
                token.startswith("[") and token.endswith("]")
                or (excluded_tokens and any(excl.lower() in token.lower() for excl in excluded_tokens))
            )
        ]
        if not valid_indices:
            return None
        mask_idx = random.choice(valid_indices)
        target_token = tokens[mask_idx]
        # Write the dataset placeholder, never tokenizer.mask_token. Emitting
        # the tokenizer special here produced templates the schema validator
        # rejected, but only for a tokenizer that owns one -- every model used
        # so far is a causal LM without a mask token, which is why the two
        # branches silently disagreed.
        tokens[mask_idx] = str(mask_placeholder)
        masked_text = tokenizer.convert_tokens_to_string(tokens)
        return (masked_text, target_token)
    if len(tokens) < 2:
        return None
    valid_k = [
        k for k in range(min_prefix_tokens, len(tokens))
        if _targetable_token_text(tokens[k])
        and not _is_wordpiece_continuation(tokens[k])
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
    placeholder = str(mask_placeholder)
    leading_space = bool(next_str) and (next_str[0] in ' 	' or next_str[0] == '▁')
    masked_text = prefix_str + ((' ' + placeholder) if leading_space else placeholder)
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
        max_length: int = 128,
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

    def _create_item(self, text: str, target: Union[str, Sequence[Any]]):
        """Create a single training item: input_ids, attention_mask, labels.

        For decoder-only: labels at last non-pad position. For classic encoder
        MLM: every prediction placeholder is an independent supervised site.
        A single target string is broadcast to all sites; a sequence of targets
        must have one entry per site (length mismatch raises ``ValueError``).
        Each site expands to a contiguous mask span of
        ``len(tokenize(site_target))`` tokens.

        Args:
            text: Input text or template used for the prediction objective.
            target: Shared target string, or per-site sequence of targets.
        """
        is_decoder_only_model = getattr(self, "is_decoder_only_model", False)
        prediction_objective = getattr(self, "prediction_objective", None)
        mask_placeholder = getattr(self, "mask_placeholder", DEFAULT_DATASET_MASK_PLACEHOLDER)
        if prediction_objective == "clm_mlm_head":
            target_labels = getattr(self, "mlm_head_target_labels", None)
            if not target_labels:
                raise ValueError(
                    "clm_mlm_head training requires mlm_head_target_labels on the dataset "
                    "(load from the trained decoder MLM head checkpoint)."
                )
            if not self.mask_token:
                raise ValueError("clm_mlm_head requires tokenizer.mask_token.")
            if mask_placeholder not in text:
                raise ValueError(
                    f"clm_mlm_head training requires a {mask_placeholder!r} placeholder in the input text."
                )
            if isinstance(target, (list, tuple)):
                raise ValueError(
                    "Per-site MLM targets (a sequence of targets) are only supported for "
                    "classic encoder MLM; clm_mlm_head expects a single target string."
                )
            expanded_text = text.replace(mask_placeholder, self.mask_token, 1)
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
            if mask_placeholder not in text:
                raise ValueError(
                    f"clm_sequence_cloze training requires a {mask_placeholder!r} placeholder."
                )
            if isinstance(target, (list, tuple)):
                raise ValueError(
                    "Per-site MLM targets (a sequence of targets) are only supported for "
                    "classic encoder MLM; clm_sequence_cloze expects a single target string."
                )
            prefix, rhs = text.split(mask_placeholder, 1)
            expanded_text = text.replace(mask_placeholder, str(target), 1)
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
                # Real tokens begin after the pad block under LEFT padding.
                offset = int(first_real_token_positions(attention_mask[:1])[0])
                labels[:, offset + start:offset + end] = input_ids[:, offset + start:offset + end]
            assert_labels_on_real_tokens(labels, attention_mask, where="clm_sequence_cloze item")
            return {"input_ids": input_ids.squeeze(0), "attention_mask": attention_mask.squeeze(0), "labels": labels.squeeze(0)}
        is_seq2seq_model = getattr(self, "is_seq2seq_model", False)
        if is_seq2seq_model:
            if isinstance(target, (list, tuple)):
                raise ValueError(
                    "Per-site MLM targets (a sequence of targets) are only supported for "
                    "classic encoder MLM; seq2seq training expects a single target string."
                )
            prediction_objective = getattr(self, "prediction_objective", None)
            common = dict(
                masked_text=text,
                label=str(target),
                tokenizer=self.tokenizer,
                base_model=getattr(self, "base_model", None),
                mask_placeholder=mask_placeholder,
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
        if is_decoder_only_model:
            if isinstance(target, (list, tuple)):
                raise ValueError(
                    "Per-site MLM targets (a sequence of targets) are only supported for "
                    "classic encoder MLM; decoder-only training expects a single target string."
                )
            # ``text`` is the CLM prefix (mask slot already removed upstream).
            expanded_text = text
            with suppress_tokenizer_length_warning():
                encoded = self.tokenizer(expanded_text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=self.max_length, padding="max_length")
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            labels = torch.full_like(input_ids, -100)
            last_idxs = attention_mask.sum(dim=1)
            target_idx = training_label_token_id(
                self.tokenizer, target, getattr(self, "label_token_protocol", "legacy")
            )
            last_positions = last_real_token_positions(attention_mask, allow_empty=True)
            for i, last_idx in enumerate(last_idxs):
                if last_idx == 0:
                    # No context token at all: the target is the first word of the text and the tokenizer
                    # adds no BOS (GPT-2, Pythia, Qwen), so the row is entirely padding. Nothing precedes
                    # the target, hence no sentence-dependent signal; the row is kept exactly as it always
                    # was (label on the last position of the fully masked row), not filtered.
                    labels[i, -1] = target_idx
                elif last_idx < input_ids.size(1):
                    # Last REAL token. ``last_idx - 1`` is only that under right
                    # padding; with left padding (Gemma, ...) it points into the pad
                    # block, which made every sentence's gradient identical.
                    labels[i, last_positions[i]] = target_idx
            assert_labels_on_real_tokens(
                labels, attention_mask, where="decoder-only training item", allow_empty_rows=True
            )
            return {"input_ids": input_ids.squeeze(0), "attention_mask": attention_mask.squeeze(0), "labels": labels.squeeze(0)}

        if not self.mask_token:
            raise ValueError("Classic MLM training requires tokenizer.mask_token.")
        mask_count = text.count(self.mask_token)
        if mask_count < 1 and mask_placeholder != self.mask_token:
            mask_count = text.count(mask_placeholder)
            text = text.replace(mask_placeholder, self.mask_token)
        if mask_count < 1:
            raise ValueError(
                "Classic MLM training requires at least one prediction placeholder; "
                f"found 0 occurrences of {mask_placeholder!r} in text={text!r}."
            )
        site_token_lists = _tokenize_classic_mlm_site_targets(self.tokenizer, target, mask_count)
        # Expand each site independently so per-site targets may differ in length.
        expanded_text = text
        for site_tokens in site_token_lists:
            span = " ".join([self.mask_token] * len(site_tokens))
            expanded_text = expanded_text.replace(self.mask_token, span, 1)
        with suppress_tokenizer_length_warning():
            encoded = self.tokenizer(expanded_text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = torch.full_like(input_ids, -100)
        mask_token_id = self.tokenizer.convert_tokens_to_ids(self.mask_token)
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=False)
        expected_masks = sum(len(site_tokens) for site_tokens in site_token_lists)
        if len(mask_positions) != expected_masks:
            raise ValueError(
                "Classic MLM training expected "
                f"{expected_masks} tokenized mask position(s) across {mask_count} site(s), "
                f"got {len(mask_positions)}. text={text!r}, expanded_text={expanded_text!r}."
            )
        offset = 0
        for site_tokens in site_token_lists:
            for tok in site_tokens:
                b, pos = mask_positions[offset].tolist()
                labels[b, pos] = tok
                offset += 1
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
        balance_column: str = "feature_pole",
        max_length: int = 128,
        seed: Optional[int] = None,
        prediction_objective: Optional[str] = None,
        rhs_window: int = -1,
        mlm_head_target_labels: Optional[List[str]] = None,
        mask_placeholder: str = DEFAULT_DATASET_MASK_PLACEHOLDER,
        label_token_protocol: str = "legacy",
    ):
        """Initialize the training dataset.

        Args:
            label_token_protocol: ``"canonical"`` labels decoder-only items with the
                leading-space token (see :func:`decoder_only_label_token_ids`);
                ``"legacy"`` (default here, so direct callers are unchanged) uses
                ``vocab[label]``. The trainer passes ``TrainingArguments.label_token_protocol``.
            data: DataFrame with masked, factual, alternative columns (unified schema).
            tokenizer: Tokenizer for encoding.
            batch_size: Batch size.
            is_decoder_only_model: If True, uses decoder-only ([MASK] after prefix) format.
            is_seq2seq_model: If True, creates encoder-decoder style inputs.
            max_size: Optional maximum number of samples.
            target_key: Key for label/target in data.
            balance_column: Column for balancing (default ``feature_pole``:
                ``pos`` / ``neg`` / ``neutral``).
            max_length: Maximum sequence length.
            seed: Random seed for batch ordering (default 42). Use training_args.seed for reproducibility.
            prediction_objective: Optional prediction objective override, such as
                cloze-sequence or seq2seq objectives.
            rhs_window: Right-context token window for sequence-cloze objectives.
            mask_placeholder: Dataset-level prediction placeholder in masked templates.
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
        self.mask_placeholder = str(mask_placeholder)
        if label_token_protocol not in ("canonical", "legacy"):
            raise ValueError(
                f"label_token_protocol must be 'canonical' or 'legacy', got {label_token_protocol!r}"
            )
        self.label_token_protocol = label_token_protocol
        validate_masked_templates_in_dataframe(
            self.data,
            self.mask_placeholder,
            masked_col=UNIFIED_MASKED,
        )
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
        mask_placeholder = self.mask_placeholder
        template = require_mask_placeholder(entry[UNIFIED_MASKED], mask_placeholder)
        if self.prediction_objective == "clm_sequence_cloze":
            input_text = template
        elif self.prediction_objective == SEQ2SEQ_DECODER_SEQUENCE_CLOZE:
            input_text = template
        elif self.is_decoder_only_model:
            if self.prediction_objective == "clm_mlm_head":
                input_text = template
            else:
                input_text = template.split(mask_placeholder)[0]
        elif self.is_seq2seq_model:
            input_text = mask_placeholder_for_tokenizer(
                template,
                self.tokenizer,
                mask_placeholder=mask_placeholder,
            )
        elif self.mask_token:
            input_text = template.replace(mask_placeholder, self.mask_token)
        else:
            input_text = template
        text = template.replace(mask_placeholder, entry[UNIFIED_FACTUAL])
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
        for key in ("is_identity_transition", "neutral_variant", "transition_type"):
            if key in entry:
                out[key] = entry[key]
        if "feature_pole" in entry:
            out["feature_pole"] = entry["feature_pole"]
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

    The filled span is not just a convenience. For MLM-style models, leaving the
    model's mask token in both factual and alternative inputs would produce the
    same activation at the measured prediction position for both sides; ACTIEND
    would then see no factual/alternative distinction to learn from. Filling with
    the candidate token creates the target-conditioned activation difference that
    the ACTIEND decoder later turns into an additive steering vector.
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
        expand_encoder_eval_poles: bool = False,
        cache_dir: Optional[str] = None,
        use_cached_signals: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        return_metadata: bool = False,
        timing_steps: int = 0,
        timing_label: str = "text-activation",
        signal: Any = None,
        signals: Any = None,
        mask_placeholder: str = DEFAULT_DATASET_MASK_PLACEHOLDER,
        prediction_objective: str = "clm_next_token",
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
            expand_encoder_eval_poles=expand_encoder_eval_poles,
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
        self.mask_placeholder = str(mask_placeholder)
        # Use the package's shared next-token objective. Full target-span loss
        # remains available only when requested explicitly.
        self.prediction_objective = str(prediction_objective or "clm_next_token")
        if self.prediction_objective not in {"clm_target_span", "clm_next_token"}:
            raise ValueError(
                "Activation signals support prediction_objective='clm_target_span' "
                "or 'clm_next_token', got "
                f"{self.prediction_objective!r}"
            )

    @staticmethod
    def default_signal(signal: Signal) -> Signal:
        """Text-prediction default for activation-based signals: select the prediction span."""
        if signal.kind not in ("activation", "activation_gradient"):
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
        max_length = getattr(self.training_data, "max_length", 128)
        if not isinstance(max_length, int) or max_length <= 0:
            max_length = 128
        return _filled_prediction_from_template(
            self.tokenizer,
            template=str(template),
            target=str(target_token),
            max_length=max_length,
            mask_placeholder=self.mask_placeholder,
        )

    def _filled_side_batch(self, templates: List[Any], target_tokens: List[Any]) -> dict:
        items = [
            self._create_filled_prediction_item(template, target_token)
            for template, target_token in zip(templates, target_tokens)
        ]
        return _stack_text_items(items)

    def _merge_batch(self, indices: list) -> dict:
        batch = super()._merge_batch(indices)
        if (
            self.signal is not None
            and self.signal.kind == "activation_gradient"
            and self.prediction_objective == "clm_next_token"
        ):
            # Reuse the TextTrainingDataset item: its single non--ignore label
            # defines the next-token loss. Expose that label
            # position as prediction_mask so ActivationSignalExtractor shifts it
            # left to the matching generating residual position.
            for side in ("factual", "alternative"):
                item = dict(batch[side])
                labels = item.get("labels")
                if not torch.is_tensor(labels):
                    raise ValueError("clm_next_token activation-gradient items require tensor labels")
                item["prediction_mask"] = labels.ne(-100)
                batch[side] = item
            return batch
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
