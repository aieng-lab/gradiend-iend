"""
Decoder evaluation helpers for text prediction (LMS, feature/target token scores).

Used by TextPredictionTrainer.evaluate_base_model. No dependency on
trainer.core.feature_definition to avoid circular imports.

Row-wise mode: for each row, P(dataset_class) = P(factual token), P(other_class) = P(alternative token).
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
import pandas as pd
import torch

from gradiend.model.utils import is_decoder_only_model, prediction_eval_kind
from gradiend.trainer.text.prediction.seq2seq import (
    score_seq2seq_continuation_logprob,
    seq2seq_mlm_probs_at_mask,
)
from gradiend.util.logging import get_logger
from gradiend.util.positions import last_real_token_positions

from gradiend.trainer.text.common.lm_eval import compute_lms


logger = get_logger(__name__)

# Column names for row-wise eval (unified schema)
DEFAULT_FACTUAL_COL = "factual"
DEFAULT_ALTERNATIVE_COL = "alternative"
DEFAULT_DATASET_CLASS_COL = "factual_id"
DEFAULT_OTHER_CLASS_COL = "alternative_id"
RAW_RESULT_METADATA_COLUMNS = (
    "text",
    "masked",
    "factual",
    "alternative",
    "label",
    "label_class",
    "factual_id",
    "alternative_id",
    "split",
    "source",
)


def _csv_scalar(value):
    """Return a compact CSV-friendly scalar for decoder raw-result exports."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _rows_with_source_index(df: pd.DataFrame) -> List[Dict[str, object]]:
    return [
        {**row.to_dict(), "_row_index": idx}
        for idx, row in df.iterrows()
    ]


def _decoder_raw_result_row(
    row: Dict[str, object],
    *,
    key_text: str,
    dataset_class_col: Optional[str],
    dataset_class: object,
    class_probs: Dict[str, float],
    score_kind: str,
) -> Dict[str, object]:
    record: Dict[str, object] = {
        "row_index": _csv_scalar(row.get("_row_index")),
        "dataset_class": _csv_scalar(dataset_class),
        "score_kind": score_kind,
    }
    for col in RAW_RESULT_METADATA_COLUMNS:
        if col in row:
            record[col] = _csv_scalar(row.get(col))
    if key_text in row and key_text not in record:
        record[key_text] = _csv_scalar(row.get(key_text))
    if key_text in row and "masked" not in record:
        record["masked"] = _csv_scalar(row.get(key_text))
    if dataset_class_col and dataset_class_col in row and dataset_class_col not in record:
        record[dataset_class_col] = _csv_scalar(row.get(dataset_class_col))
    for cls, prob in sorted(class_probs.items(), key=lambda item: str(item[0])):
        record[f"p_class_{cls}"] = float(prob)
    return record


def _normalize_token_string(value: str) -> str:
    if value is None:
        return ""
    # GPT-style BPE and SentencePiece encode leading whitespace as marker
    # characters in vocab tokens. Decoder class probabilities are semantic over
    # the surface target (e.g. "He"), so normalize these markers together with
    # ordinary leading whitespace before matching target strings to vocab ids.
    return str(value).lstrip().lstrip("Ġ▁").lstrip().casefold()


def _convert_vocab_token_to_id(tokenizer, token: str) -> Optional[int]:
    if not hasattr(tokenizer, "convert_tokens_to_ids"):
        return None
    token_id = tokenizer.convert_tokens_to_ids(token)
    if isinstance(token_id, list):
        return None
    if token_id is None:
        return None
    unk_id = getattr(tokenizer, "unk_token_id", None)
    unk_token = getattr(tokenizer, "unk_token", None)
    if unk_id is not None and int(token_id) == int(unk_id) and token != unk_token:
        return None
    return int(token_id)


def _unique_token_ids(ids: List[int]) -> List[int]:
    seen = set()
    out = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id in seen:
            continue
        seen.add(token_id)
        out.append(token_id)
    return out


def _candidate_strings_for_label(tokenizer, token_str: str, vocab_norm_map: Dict[str, List[str]]) -> List[str]:
    s = str(token_str).strip()
    norm = _normalize_token_string(s)
    return list(vocab_norm_map.get(norm, []) or [s])


def _target_id_sequences_for_label(tokenizer, token_str: str, vocab_norm_map: Dict[str, List[str]]) -> List[List[int]]:
    if token_str is None or (isinstance(token_str, float) and np.isnan(token_str)):
        return []
    s = str(token_str).strip()
    if not s:
        return []
    _tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    sequences: List[List[int]] = []
    seen = set()
    for cand in _candidate_strings_for_label(_tokenizer, s, vocab_norm_map):
        direct_id = _convert_vocab_token_to_id(_tokenizer, cand)
        ids: List[int] = [direct_id] if direct_id is not None else []
        if not ids and hasattr(_tokenizer, "tokenize"):
            tokenized = _tokenizer.tokenize(cand)
            if tokenized:
                converted = _tokenizer.convert_tokens_to_ids(tokenized)
                if isinstance(converted, list):
                    ids = [int(i) for i in converted if i is not None]
                elif converted is not None:
                    ids = [int(converted)]
        if not ids:
            encoded = _tokenizer(cand, add_special_tokens=False).get("input_ids", [])
            ids = [int(i) for i in encoded]
        if not ids:
            continue
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        sequences.append(ids)
    return sequences


def _single_token_candidate_ids(tokenizer, token_str: str, vocab_norm_map: Dict[str, List[str]]) -> List[int]:
    ids = []
    fallback_id = None
    for seq in _target_id_sequences_for_label(tokenizer, token_str, vocab_norm_map):
        if len(seq) == 1:
            ids.append(seq[0])
        elif fallback_id is None:
            fallback_id = seq[0]
    if ids:
        return _unique_token_ids(ids)
    return [int(fallback_id)] if fallback_id is not None else []


def _token_to_single_id(tokenizer, token_str: str, vocab_norm_map: Dict[str, List[str]]) -> Optional[int]:
    """Resolve a single token string to one token id (single token or first subword). Returns None if invalid."""
    ids = _single_token_candidate_ids(tokenizer, token_str, vocab_norm_map)
    return ids[0] if ids else None


def resolve_prediction_target_token_id(tokenizer, label: str) -> int:
    """
    Resolve a prediction label string to one vocabulary token id.

    Multi-token labels use the first subword token, matching decoder evaluation.
    """
    if label is None or (isinstance(label, float) and np.isnan(label)):
        raise ValueError("Cannot resolve empty prediction label to a token id.")
    s = str(label).strip()
    if not s:
        raise ValueError("Cannot resolve empty prediction label to a token id.")
    _tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    vocab_norm_map = _build_vocab_norm_map(_tokenizer)
    token_id = _token_to_single_id(_tokenizer, s, vocab_norm_map)
    if token_id is None:
        ids = _tokenizer(s, add_special_tokens=False).get("input_ids", [])
        if not ids:
            raise ValueError(f"Could not resolve label {label!r} to a vocabulary token id.")
        token_id = ids[0]
    ids = _tokenizer(s, add_special_tokens=False).get("input_ids", [])
    if len(ids) > 1:
        logger.warning(
            "Label %r tokenized to %d tokens; using first token id %s for decoder-only prediction.",
            label,
            len(ids),
            token_id,
        )
    return token_id


def _build_vocab_norm_map(tokenizer) -> Dict[str, List[str]]:
    vocab = None
    if hasattr(tokenizer, "get_vocab"):
        try:
            vocab = tokenizer.get_vocab()
        except Exception:
            vocab = None
    if vocab is None and hasattr(tokenizer, "vocab"):
        vocab = getattr(tokenizer, "vocab", None)
    if not vocab:
        return {}
    norm_map: Dict[str, List[str]] = defaultdict(list)
    for token in vocab.keys():
        norm_map[_normalize_token_string(token)].append(token)
    return norm_map


def _tokenizer_input_ids(tokenizer, text: str) -> List[int]:
    encoded = tokenizer(str(text), add_special_tokens=False, padding=False)
    ids = encoded.get("input_ids", encoded) if hasattr(encoded, "get") else encoded
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in ids]


def _clm_context_and_gap_from_prefix(prefix: str) -> Tuple[str, str]:
    """Split a CLM prefix into model context and whitespace carried by the target.

    GPT-style tokenizers usually encode leading whitespace as part of the next
    token (e.g. ``" she"`` -> ``Ġshe``). If the raw template prefix ends in a
    space, feeding that space as the final context token asks the model to
    predict after a standalone whitespace token. Instead, score the same surface
    sentence by trimming the context and prepending the removed whitespace to the
    candidate continuation.
    """
    prefix = str(prefix)
    context = prefix.rstrip()
    return context, prefix[len(context):]


def _clm_context_and_gap_from_masked_text(text: str, *, placeholder: str = "[MASK]") -> Tuple[str, str]:
    prefix = str(text).split(placeholder, 1)[0]
    return _clm_context_and_gap_from_prefix(prefix)


def _clm_first_continuation_token_ids(
    tokenizer,
    prefix_context: str,
    gap: str,
    target: str,
    vocab_norm_map: Optional[Dict[str, List[str]]] = None,
) -> List[int]:
    """Resolve CLM continuation token ids for ``target`` in this row context.

    Always include the contextual first continuation token. Also SUM (via the
    caller gathering ``p[ids].sum()``) every single-token vocab piece whose
    surface matches ``target`` after stripping leading space markers and
    casefolding (``he`` / ``He`` / ``Ġhe`` / …). Do not take MAX over variants.
    """
    if target is None or (isinstance(target, float) and np.isnan(target)):
        return []
    continuation = f"{gap}{str(target).lstrip()}"
    ids: List[int] = []
    prefix_ids = _tokenizer_input_ids(tokenizer, prefix_context)
    if prefix_ids:
        full_ids = _tokenizer_input_ids(tokenizer, prefix_context + continuation)
        if full_ids[: len(prefix_ids)] == prefix_ids and len(full_ids) > len(prefix_ids):
            ids.append(int(full_ids[len(prefix_ids)]))
    if not ids:
        continuation_ids = _tokenizer_input_ids(tokenizer, continuation)
        if continuation_ids:
            ids.append(int(continuation_ids[0]))
    if vocab_norm_map is not None:
        ids.extend(_single_token_candidate_ids(tokenizer, target, vocab_norm_map))
    return _unique_token_ids(ids)


def _clm_first_continuation_token_ids_for_targets(
    tokenizer,
    prefix_context: str,
    gap: str,
    targets: List[str],
    vocab_norm_map: Dict[str, List[str]],
) -> List[int]:
    ids: List[int] = []
    for target in targets:
        ids.extend(_clm_first_continuation_token_ids(tokenizer, prefix_context, gap, target, vocab_norm_map))
    return _unique_token_ids(ids)


def _batch_to_device(inputs, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _last_non_padding_positions(logits: torch.Tensor, inputs) -> torch.Tensor:
    attention_mask = inputs.get("attention_mask") if hasattr(inputs, "get") else None
    if attention_mask is None:
        return torch.full((logits.shape[0],), logits.shape[1] - 1, dtype=torch.long, device=logits.device)
    mask = attention_mask.to(device=logits.device)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D attention_mask for CLM decoder scoring, got shape {tuple(mask.shape)}.")
    if logits.ndim != 3 or logits.shape[:2] != mask.shape:
        raise ValueError(
            "CLM decoder scoring received incompatible logits/attention_mask shapes: "
            f"logits={tuple(logits.shape)}, attention_mask={tuple(mask.shape)}."
        )
    try:
        return last_real_token_positions(mask)
    except ValueError as exc:
        raise ValueError("Cannot score CLM next-token probabilities for an empty tokenized prefix.") from exc


def _last_non_padding_logits(logits: torch.Tensor, inputs) -> torch.Tensor:
    last_positions = _last_non_padding_positions(logits, inputs)
    batch_idx = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_idx, last_positions, :]


def _debug_token_label(tokenizer, token_id: int) -> str:
    token_id = int(token_id)
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        token = tokenizer.convert_ids_to_tokens(token_id)
        if token is not None:
            return str(token)
    if hasattr(tokenizer, "decode"):
        try:
            return str(tokenizer.decode([token_id], skip_special_tokens=False))
        except Exception:
            pass
    return f"<id:{token_id}>"


def _debug_token_decoded(tokenizer, token_id: int) -> str:
    token_id = int(token_id)
    if hasattr(tokenizer, "decode"):
        try:
            return str(tokenizer.decode([token_id], skip_special_tokens=False))
        except Exception:
            return f"<id:{token_id}>"
    return f"<id:{token_id}>"


def _print_clm_next_token_debug(
    tokenizer,
    prefix_texts: List[str],
    inputs,
    last_positions: torch.Tensor,
    probs: torch.Tensor,
    *,
    top_k: int,
    raw_prefix_texts: Optional[List[str]] = None,
) -> None:
    input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
    top_k = max(1, min(int(top_k), probs.shape[-1]))
    top_probs, top_ids = torch.topk(probs.detach().cpu(), k=top_k, dim=-1)
    for row_idx, prefix in enumerate(prefix_texts):
        pos = int(last_positions[row_idx].detach().cpu().item())
        context_id = None
        context_token = None
        if input_ids is not None:
            context_id = int(input_ids[row_idx, pos].detach().cpu().item())
            context_token = _debug_token_label(tokenizer, context_id)
        raw_prefix = raw_prefix_texts[row_idx] if raw_prefix_texts is not None else prefix
        print(f"CLM next-token debug [{row_idx}]: prefix={raw_prefix!r}")
        if raw_prefix != prefix:
            print(f"  model_prefix={prefix!r}")
        if context_id is not None:
            context_decoded = _debug_token_decoded(tokenizer, context_id)
            print(
                f"  selected_position={pos}, context_id={context_id}, "
                f"context_token={context_token!r}, context_decoded={context_decoded!r}"
            )
        else:
            print(f"  selected_position={pos}")
        print(f"  top {top_k} next-token predictions:")
        for rank, (token_id, prob) in enumerate(zip(top_ids[row_idx].tolist(), top_probs[row_idx].tolist()), start=1):
            token_label = _debug_token_label(tokenizer, int(token_id))
            token_decoded = _debug_token_decoded(tokenizer, int(token_id))
            print(
                f"    {rank}. id={int(token_id)} token={token_label!r} "
                f"decoded={token_decoded!r} prob={float(prob):.6f}"
            )


def _clm_next_token_probs_for_prefixes(
    model,
    tokenizer,
    prefix_texts: List[str],
    device,
    *,
    max_length: int = 512,
    verbose: bool = False,
    top_k: int = 5,
) -> torch.Tensor:
    """Return next-token probability vectors for CLM prefixes.

    The gathered logit position is the last non-padding prefix token. This is
    the causal-LM next-token site and is independent of tokenizer padding side.
    Set ``verbose=True`` to print per-prefix diagnostics showing the selected
    context position and the top predicted next tokens.
    """
    raw_prefix_texts = list(prefix_texts)
    model_prefix_texts = [
        _clm_context_and_gap_from_prefix(prefix)[0]
        for prefix in raw_prefix_texts
    ]
    inputs = tokenizer(
        model_prefix_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = _batch_to_device(inputs, device)
    outputs = model(**inputs)
    last_positions = _last_non_padding_positions(outputs.logits, inputs)
    batch_idx = torch.arange(outputs.logits.shape[0], device=outputs.logits.device)
    logits = outputs.logits[batch_idx, last_positions, :]
    probs = torch.softmax(logits, dim=-1)
    if verbose:
        _print_clm_next_token_debug(
            tokenizer,
            model_prefix_texts,
            inputs,
            last_positions,
            probs,
            top_k=top_k,
            raw_prefix_texts=raw_prefix_texts,
        )
    return probs


def annotate_text_probability_rows(
    model,
    tokenizer,
    df: pd.DataFrame,
    targets: Dict[str, List[str]],
    *,
    key_text: str = "masked",
    batch_size: int = 16,
    safe_token_map: Optional[Dict[str, str]] = None,
    prefix: str = "",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Return per-row target-token and class probabilities for text prediction data.

    The resulting DataFrame contains columns:

    - ``{prefix}p_target_{safe_token}``
    - ``{prefix}p_class_{class_name}``
    """
    if key_text not in df.columns:
        raise ValueError(
            f"annotate_text_probability_rows() requires column {key_text!r}. Available: {list(df.columns)}"
        )

    token_set = sorted({str(token) for values in targets.values() for token in values if token is not None})
    if safe_token_map is None:
        safe_token_map = {str(idx): token for idx, token in enumerate(token_set)}
    token_to_safe = {token: safe for safe, token in safe_token_map.items()}

    model.eval()
    device = model.device
    _tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    eval_kind = prediction_eval_kind(model)
    vocab_norm_map = _build_vocab_norm_map(_tokenizer)
    mask_token = getattr(tokenizer, "mask_token", None) or getattr(_tokenizer, "mask_token", None)
    mask_token_id = getattr(tokenizer, "mask_token_id", None) or getattr(_tokenizer, "mask_token_id", None)

    token_id_map: Dict[str, List[int]] = {}
    for token in token_set:
        token_ids = _single_token_candidate_ids(_tokenizer, token, vocab_norm_map)
        if not token_ids:
            logger.warning("Skipping unresolved annotation token %r", token)
            continue
        token_id_map[token] = token_ids
    if not token_id_map:
        raise ValueError("annotate_text_probability_rows() could not resolve any target token ids.")

    prob_rows: List[Dict[str, float]] = []
    rows = df.to_dict("records")
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            if eval_kind == "clm_next_token":
                raw_prefixes = [str(r[key_text]).split("[MASK]", 1)[0] for r in batch]
                prefix_specs = [_clm_context_and_gap_from_prefix(prefix) for prefix in raw_prefixes]
                valid = [(i, raw_prefixes[i]) for i, (context, _gap) in enumerate(prefix_specs) if context.strip()]
                example_probs = {}
                if valid:
                    idxs, prefix_texts = zip(*valid)
                    probs = _clm_next_token_probs_for_prefixes(model, tokenizer, list(prefix_texts), device)
                    example_probs = {i: probs[j] for j, i in enumerate(idxs)}
            elif eval_kind == "seq2seq_encoder_mlm":
                texts = [str(r[key_text]) for r in batch]
                probs = seq2seq_mlm_probs_at_mask(model, _tokenizer, texts, device)
                example_probs = {i: probs[i] for i in range(len(batch))}
            else:
                if mask_token is None or mask_token_id is None:
                    raise ValueError("annotate_text_probability_rows() requires tokenizer.mask_token for MLM models.")
                texts = [str(r[key_text]).replace("[MASK]", mask_token) for r in batch]
                inputs = tokenizer(
                    list(texts), return_tensors="pt", padding=True, truncation=True, max_length=512
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                mask_idxs = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=False)
                example_probs = {}
                if len(mask_idxs) > 0:
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    for b_idx, pos in mask_idxs.tolist():
                        example_probs[b_idx] = probs[b_idx, pos, :]

            for b_idx, _row in enumerate(batch):
                p = example_probs.get(b_idx)
                if p is None:
                    prob_rows.append({})
                    continue
                record: Dict[str, float] = {}
                token_probs: Dict[str, float] = {}
                prefix_context, gap = _clm_context_and_gap_from_masked_text(str(_row[key_text]))
                token_items = (
                    (
                        token,
                        _clm_first_continuation_token_ids_for_targets(
                            _tokenizer,
                            prefix_context,
                            gap,
                            [token],
                            vocab_norm_map,
                        ),
                    )
                    for token in token_set
                ) if eval_kind == "clm_next_token" else token_id_map.items()
                for token, token_ids in token_items:
                    if not token_ids:
                        continue
                    safe = token_to_safe[token]
                    value_t = p[token_ids].sum()
                    value = float(value_t.item() if hasattr(value_t, "item") else value_t)
                    token_probs[token] = value
                    record[f"{prefix}p_target_{safe}"] = value
                for class_name, class_tokens in targets.items():
                    record[f"{prefix}p_class_{class_name}"] = float(
                        sum(token_probs.get(str(token), 0.0) for token in class_tokens)
                    )
                prob_rows.append(record)

    return pd.DataFrame(prob_rows), safe_token_map


def compute_probability_shift_score_row_wise(
    model,
    tokenizer,
    df,
    key_text: str = "masked",
    batch_size: int = 16,
    factual_col: str = DEFAULT_FACTUAL_COL,
    alternative_col: str = DEFAULT_ALTERNATIVE_COL,
    dataset_class_col: str = DEFAULT_DATASET_CLASS_COL,
    other_class_col: str = DEFAULT_OTHER_CLASS_COL,
    return_per_row_df: bool = False,
) -> Union[Dict[str, Dict[str, float]], Tuple[Dict[str, Dict[str, float]], pd.DataFrame]]:
    """
    Row-wise decoder eval: for each row, P(dataset_class) = P(factual), P(other_class) = P(alternative).

    DataFrame must have: key_text (masked), factual_col, alternative_col, dataset_class_col (e.g. factual_id),
    other_class_col (e.g. alternative_id). Returns same shape as static eval: {dataset_class: {class_name: mean_prob}}.
    When return_per_row_df=True, also returns a DataFrame with one row per sample: masked, factual, alternative,
    factual_id, alternative_id, dataset_class, other_class, P_factual, P_alternative.
    """
    model.eval()
    device = model.device
    eval_kind = prediction_eval_kind(model)
    _tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    vocab_norm_map = _build_vocab_norm_map(_tokenizer)
    mask_token_id = getattr(tokenizer, "mask_token_id", None) or getattr(_tokenizer, "mask_token_id", None)

    for col in (key_text, factual_col, alternative_col, dataset_class_col, other_class_col):
        if col not in df.columns:
            raise ValueError(
                f"Row-wise decoder eval requires column '{col}' in the DataFrame. "
                f"Available: {list(df.columns)}."
            )

    probs_by_dataset: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_row_records: List[Dict[str, object]] = []
    rows = df.to_dict("records")

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            if eval_kind == "clm_next_token":
                raw_prefixes = [str(r[key_text]).split("[MASK]", 1)[0] for r in batch]
                prefix_specs = [_clm_context_and_gap_from_prefix(prefix) for prefix in raw_prefixes]
                valid = [(i, raw_prefixes[i]) for i, (context, _gap) in enumerate(prefix_specs) if context.strip()]
                if not valid:
                    continue
                idxs, prefix_texts = zip(*valid)
                probs = _clm_next_token_probs_for_prefixes(model, tokenizer, list(prefix_texts), device)
                example_probs = {i: probs[j] for j, i in enumerate(idxs)}
            elif eval_kind == "seq2seq_encoder_mlm":
                texts = [str(r[key_text]) for r in batch]
                probs = seq2seq_mlm_probs_at_mask(model, _tokenizer, texts, device)
                example_probs = {i: probs[i] for i in range(len(batch))}
            else:
                mask_tok = getattr(tokenizer, "mask_token", None) or getattr(_tokenizer, "mask_token", None)
                texts = [r[key_text].replace("[MASK]", mask_tok) for r in batch]
                inputs = tokenizer(
                    list(texts), return_tensors="pt", padding=True, truncation=True, max_length=512
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                if mask_token_id is None:
                    continue
                mask_idxs = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=False)
                if len(mask_idxs) == 0:
                    continue
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                example_probs = {}
                for b_idx, pos in mask_idxs.tolist():
                    example_probs[b_idx] = probs[b_idx, pos, :]

            for b_idx, row in enumerate(batch):
                if b_idx not in example_probs:
                    continue
                p = example_probs[b_idx]
                ds_class = row.get(dataset_class_col)
                other_class = row.get(other_class_col)
                if ds_class is None or other_class is None:
                    continue
                if eval_kind == "clm_next_token":
                    prefix_context, gap = _clm_context_and_gap_from_masked_text(str(row[key_text]))
                    fids = _clm_first_continuation_token_ids_for_targets(
                        _tokenizer, prefix_context, gap, [row.get(factual_col)], vocab_norm_map
                    )
                    aids = _clm_first_continuation_token_ids_for_targets(
                        _tokenizer, prefix_context, gap, [row.get(alternative_col)], vocab_norm_map
                    )
                else:
                    fids = _single_token_candidate_ids(_tokenizer, row.get(factual_col), vocab_norm_map)
                    aids = _single_token_candidate_ids(_tokenizer, row.get(alternative_col), vocab_norm_map)
                if not fids or not aids:
                    continue
                p_f_t = p[fids].sum()
                p_a_t = p[aids].sum()
                p_f = float(p_f_t.item() if hasattr(p_f_t, "item") else p_f_t)
                p_a = float(p_a_t.item() if hasattr(p_a_t, "item") else p_a_t)
                probs_by_dataset[ds_class][ds_class].append(p_f)
                probs_by_dataset[ds_class][other_class].append(p_a)
                if return_per_row_df:
                    per_row_records.append({
                        "masked": row.get(key_text),
                        "factual": row.get(factual_col),
                        "alternative": row.get(alternative_col),
                        "factual_id": ds_class,
                        "alternative_id": other_class,
                        "dataset_class": ds_class,
                        "other_class": other_class,
                        "P_factual": p_f,
                        "P_alternative": p_a,
                    })

    probs_by_dataset_means = {}
    for ds_cls, class_probs in probs_by_dataset.items():
        probs_by_dataset_means[ds_cls] = {
            cls: float(np.mean(vals)) if vals else 0.0 for cls, vals in class_probs.items()
        }
    if not probs_by_dataset_means:
        raise ValueError(
            "Row-wise decoder eval produced no probabilities. Ensure the DataFrame has columns "
            "factual, alternative, factual_id, alternative_id and that tokenizer can tokenize the tokens."
        )
    logger.debug(
        "Decoder eval (row-wise) probability means: %s",
        ", ".join(f"{ds}: {list(p.keys())}" for ds, p in sorted(probs_by_dataset_means.items())),
    )
    if return_per_row_df and per_row_records:
        per_row_df = pd.DataFrame(per_row_records)
        return (probs_by_dataset_means, per_row_df)
    if return_per_row_df:
        return (probs_by_dataset_means, pd.DataFrame())
    return probs_by_dataset_means


def compute_probability_shift_score_clm(
    model,
    tokenizer,
    df,
    targets,
    key_text='masked',
    batch_size=16,
    dataset_class_col=None,
    factual_dataset_class_col=None,
    return_per_row_df: bool = False,
):
    """
    Compute probabilities for all classes evaluated on all datasets.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        df: DataFrame with evaluation data
        targets: Dict mapping class_name -> list of tokens
        key_text: Column name for masked text
        batch_size: Batch size for evaluation
        dataset_class_col: Column name identifying dataset class (e.g., "label_class" or "factual_id").
            If None, uses "label_class" if available, else "factual_id".

    Returns:
        Dict[str, Dict[str, float]]: {dataset_class: {class_name: prob, ...}, ...}
    """
    model.eval()
    device = model.device
    eval_kind = prediction_eval_kind(model)
    _tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
    vocab_norm_map = _build_vocab_norm_map(_tokenizer)

    def create_token_ids(words):
        toks = []
        for w in set(words):
            toks.extend(_single_token_candidate_ids(_tokenizer, w, vocab_norm_map))
        return _unique_token_ids(toks)

    targets_ids = None
    if eval_kind != "clm_next_token":
        targets_ids = {g: create_token_ids(ws) for g, ws in targets.items()}
        for g, ids in targets_ids.items():
            if not ids:
                raise ValueError(f"Target group '{g}' has no valid token ids; check your target words and tokenizer.")

    # Determine dataset class column
    if dataset_class_col is None:
        if "label_class" in df.columns:
            dataset_class_col = "label_class"
        elif "factual_id" in df.columns:
            dataset_class_col = "factual_id"
        else:
            dataset_class_col = None

    # Group probabilities by dataset class: {dataset_class: {class_name: [probs]}}
    probs_by_dataset = defaultdict(lambda: defaultdict(list))
    probs_by_factual_dataset = (
        defaultdict(lambda: defaultdict(list)) if factual_dataset_class_col else None
    )
    rows = _rows_with_source_index(df)
    per_row_records: List[Dict[str, object]] = []
    n_batches = (len(rows) + batch_size - 1) // batch_size

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]

            if eval_kind == "clm_next_token":
                # Decoder/CLM: use next-token logits only (no MLM head)
                raw_prefixes = [str(r[key_text]).split("[MASK]", 1)[0] for r in batch]
                prefix_specs = [_clm_context_and_gap_from_prefix(prefix) for prefix in raw_prefixes]
                valid = [(i, raw_prefixes[i]) for i, (context, _gap) in enumerate(prefix_specs) if context.strip()]
                if not valid:
                    continue
                idxs, prefix_texts = zip(*valid)
                probs = _clm_next_token_probs_for_prefixes(model, tokenizer, list(prefix_texts), device)
                example_probs = {i: probs[j] for j, i in enumerate(idxs)}
            elif eval_kind == "seq2seq_encoder_mlm":
                texts = [str(r[key_text]) for r in batch]
                probs = seq2seq_mlm_probs_at_mask(model, _tokenizer, texts, device)
                example_probs = {i: probs[i] for i in range(len(batch))}
            else:
                texts = [r[key_text].replace('[MASK]', tokenizer.mask_token) for r in batch]
                inputs = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                mask_idxs = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)
                if len(mask_idxs) == 0:
                    continue
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                example_probs = {}
                for b_idx, pos in mask_idxs.tolist():
                    example_probs[b_idx] = probs[b_idx, pos, :]

            for b_idx, row in enumerate(batch):
                if b_idx not in example_probs:
                    continue
                p = example_probs[b_idx]
                
                # Get dataset class for this row
                dataset_class = None
                if dataset_class_col:
                    dataset_class = row.get(dataset_class_col)
                factual_class = None
                if factual_dataset_class_col:
                    factual_class = row.get(factual_dataset_class_col)
                
                class_prob_values: Dict[str, float] = {}
                if eval_kind == "clm_next_token":
                    prefix_context, gap = _clm_context_and_gap_from_masked_text(str(row[key_text]))
                    target_id_items = [
                        (
                            g,
                            _clm_first_continuation_token_ids_for_targets(
                                _tokenizer,
                                prefix_context,
                                gap,
                                ws,
                                vocab_norm_map,
                            ),
                        )
                        for g, ws in targets.items()
                    ]
                else:
                    target_id_items = list((targets_ids or {}).items())

                # Compute probabilities for all classes
                for g, ids in target_id_items:
                    if ids:
                        prob = float(p[ids].sum())  # Sum multiple tokens per class
                        class_prob_values[g] = prob
                        if dataset_class is not None:
                            probs_by_dataset[dataset_class][g].append(prob)
                        else:
                            # Fallback: use class name as dataset identifier
                            probs_by_dataset[g][g].append(prob)
                        if factual_class is not None and probs_by_factual_dataset is not None:
                            probs_by_factual_dataset[factual_class][g].append(prob)
                if return_per_row_df and class_prob_values:
                    per_row_records.append(
                        _decoder_raw_result_row(
                            row,
                            key_text=key_text,
                            dataset_class_col=dataset_class_col,
                            dataset_class=dataset_class,
                            class_probs=class_prob_values,
                            score_kind="target_token_probability",
                        )
                    )

    logger.debug(
        "compute_probability_shift_score_clm: eval_kind=%r dataset_class_col=%r "
        "expected metrics (targets.keys())=%s groups actually populated=%s "
        "metrics populated per group=%s",
        eval_kind,
        dataset_class_col,
        sorted(targets.keys()),
        sorted(str(k) for k in probs_by_dataset.keys()),
        {str(g): sorted(m.keys()) for g, m in probs_by_dataset.items()},
    )
    missing_cells = {
        str(g): sorted(set(targets.keys()) - set(m.keys()))
        for g, m in probs_by_dataset.items()
        if set(targets.keys()) - set(m.keys())
    }
    if missing_cells:
        logger.warning(
            "compute_probability_shift_score_clm: some (group, metric) cells never "
            "got populated despite being in targets -- missing_cells(group -> metrics)=%s. "
            "This means every row in that group failed to produce a non-empty token id "
            "list for that metric (see _clm_first_continuation_token_ids_for_targets), "
            "not a missing-group problem.",
            missing_cells,
        )

    # Compute means per dataset class
    probs_by_dataset_means = {}
    for dataset_class, class_probs in probs_by_dataset.items():
        probs_by_dataset_means[dataset_class] = {
            class_name: float(np.mean(probs)) if probs else 0.0
            for class_name, probs in class_probs.items()
        }

    # Return probs_by_dataset structure; selection filtering is done in evaluate_base_model
    if probs_by_dataset_means:
        means_str = ", ".join(
            f"{ds_cls}: {', '.join(f'{cls}={prob:.4f}' for cls, prob in sorted(cls_probs.items()))}"
            for ds_cls, cls_probs in sorted(probs_by_dataset_means.items())
        )
        logger.debug(f"Decoder eval probability means by dataset: {means_str}")
        if probs_by_factual_dataset is not None:
            factual_means = {
                dataset_class: {
                    class_name: float(np.mean(probs)) if probs else 0.0
                    for class_name, probs in class_probs.items()
                }
                for dataset_class, class_probs in probs_by_factual_dataset.items()
            }
            if return_per_row_df:
                return probs_by_dataset_means, pd.DataFrame(per_row_records)
            return probs_by_dataset_means, factual_means
        if return_per_row_df:
            return probs_by_dataset_means, pd.DataFrame(per_row_records)
        return probs_by_dataset_means
    
    raise ValueError("No valid group probabilities computed; check your data and targets.")


_token_cache = {}


def compute_probability_shift_score_mlm(
    model,
    tokenizer,
    df,
    targets,
    key_text='masked',
    batch_size=16,
    dataset_class_col=None,
    factual_dataset_class_col=None,
    return_per_row_df: bool = False,
):
    """
    Compute probabilities for all classes evaluated on all datasets.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        df: DataFrame with evaluation data
        targets: Dict mapping class_name -> list of tokens
        key_text: Column name for masked text
        batch_size: Batch size for evaluation
        dataset_class_col: Column name identifying dataset class (e.g., "label_class" or "factual_id").
            If None, uses "label_class" if available, else "factual_id".

    Returns:
        Dict[str, Dict[str, float]]: {dataset_class: {class_name: prob, ...}, ...}
    """
    model.eval()
    device = model.device
    if is_decoder_only_model(model):
        return compute_probability_shift_score_clm(
            model, tokenizer, df, targets, key_text, batch_size,
            dataset_class_col, factual_dataset_class_col,
            return_per_row_df=return_per_row_df,
        )

    eval_kind = prediction_eval_kind(model)
    if eval_kind == "seq2seq_encoder_mlm":
        return compute_probability_shift_score_clm(
            model, tokenizer, df, targets, key_text, batch_size,
            dataset_class_col, factual_dataset_class_col,
            return_per_row_df=return_per_row_df,
        )

    _tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
    vocab_norm_map = _build_vocab_norm_map(_tokenizer)

    def create_token_ids(words):
        cache_id = (tokenizer.name_or_path, hash(tuple(sorted(set(words)))))
        if cache_id in _token_cache:
            return _token_cache[cache_id]
        result = []
        seen = set()
        for w in set(words):
            for ids in _target_id_sequences_for_label(_tokenizer, w, vocab_norm_map):
                if not ids:
                    continue
                tup = tuple(ids)
                if tup in seen:
                    continue
                seen.add(tup)
                result.append(list(ids))
        _token_cache[cache_id] = result
        return result

    targets_ids = {g: create_token_ids(ws) for g, ws in targets.items()}
    for g, id_lists in targets_ids.items():
        if not id_lists:
            raise ValueError(f"Target group '{g}' has no valid token ids; check your targets.")

    # Determine dataset class column
    if dataset_class_col is None:
        if "label_class" in df.columns:
            dataset_class_col = "label_class"
        elif "factual_id" in df.columns:
            dataset_class_col = "factual_id"
        else:
            dataset_class_col = None

    # Group probabilities by dataset class: {dataset_class: {class_name: [probs]}}
    probs_by_dataset = defaultdict(lambda: defaultdict(list))
    probs_by_factual_dataset = (
        defaultdict(lambda: defaultdict(list)) if factual_dataset_class_col else None
    )
    rows = _rows_with_source_index(df)
    per_row_records: List[Dict[str, object]] = []
    n_batches = (len(rows) + batch_size - 1) // batch_size

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            texts = [r[key_text] for r in batch]
            targets_by_len = defaultdict(list)
            for g, id_lists in targets_ids.items():
                for ids in id_lists:
                    targets_by_len[len(ids)].append((g, ids))
            example_probs = defaultdict(dict)
            for k, g_and_ids in targets_by_len.items():
                expanded_texts = []
                example_map = []
                for b_idx, text in enumerate(texts):
                    if "[MASK]" not in text:
                        continue
                    masked_text = text.replace("[MASK]", " ".join([tokenizer.mask_token] * k))
                    expanded_texts.append(masked_text)
                    example_map.append(b_idx)
                if not expanded_texts:
                    continue
                inputs = tokenizer(expanded_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                for local_idx, b_idx in enumerate(example_map):
                    mask_positions = (inputs["input_ids"][local_idx] == tokenizer.mask_token_id).nonzero(as_tuple=False).squeeze(-1).tolist()
                    for g, ids in g_and_ids:
                        if len(ids) > len(mask_positions):
                            continue
                        logp = 0.0
                        for j, tok_id in enumerate(ids):
                            pos = mask_positions[j]
                            logp += torch.log(probs[local_idx, pos, tok_id] + 1e-12)
                        p = torch.exp(logp).item()
                        example_probs[b_idx][g] = example_probs[b_idx].get(g, 0.0) + p
            
            for b_idx, row in enumerate(batch):
                per_group_probs = example_probs.get(b_idx, {})
                
                # Get dataset class for this row
                dataset_class = None
                if dataset_class_col:
                    dataset_class = row.get(dataset_class_col)
                factual_class = None
                if factual_dataset_class_col:
                    factual_class = row.get(factual_dataset_class_col)
                
                # Store probabilities grouped by dataset class
                class_prob_values: Dict[str, float] = {}
                for g, p in per_group_probs.items():
                    class_prob_values[g] = float(p)
                    if dataset_class is not None:
                        probs_by_dataset[dataset_class][g].append(p)
                    else:
                        # Fallback: use class name as dataset identifier
                        probs_by_dataset[g][g].append(p)
                    if factual_class is not None and probs_by_factual_dataset is not None:
                        probs_by_factual_dataset[factual_class][g].append(p)
                if return_per_row_df and class_prob_values:
                    per_row_records.append(
                        _decoder_raw_result_row(
                            row,
                            key_text=key_text,
                            dataset_class_col=dataset_class_col,
                            dataset_class=dataset_class,
                            class_probs=class_prob_values,
                            score_kind="target_token_probability",
                        )
                    )

    # Compute means per dataset class
    probs_by_dataset_means = {}
    for dataset_class, class_probs in probs_by_dataset.items():
        probs_by_dataset_means[dataset_class] = {
            class_name: float(np.mean(probs)) if probs else 0.0
            for class_name, probs in class_probs.items()
        }

    # Return probs_by_dataset structure; selection filtering is done in evaluate_base_model
    if probs_by_dataset_means:
        means_str = ", ".join(
            f"{ds_cls}: {', '.join(f'{cls}={prob:.4f}' for cls, prob in sorted(cls_probs.items()))}"
            for ds_cls, cls_probs in sorted(probs_by_dataset_means.items())
        )
        logger.debug(f"Decoder eval probability means by dataset: {means_str}")
        if probs_by_factual_dataset is not None:
            factual_means = {
                dataset_class: {
                    class_name: float(np.mean(probs)) if probs else 0.0
                    for class_name, probs in class_probs.items()
                }
                for dataset_class, class_probs in probs_by_factual_dataset.items()
            }
            if return_per_row_df:
                return probs_by_dataset_means, pd.DataFrame(per_row_records)
            return probs_by_dataset_means, factual_means
        if return_per_row_df:
            return probs_by_dataset_means, pd.DataFrame(per_row_records)
        return probs_by_dataset_means
    
    raise ValueError("No valid group probabilities computed; check your data and targets.")


def _ids_for_text(tokenizer, text: str) -> List[int]:
    return tokenizer(str(text), add_special_tokens=False)["input_ids"]


def _continuation_ids_from_prefix(tokenizer, prefix: str, continuation: str) -> List[int]:
    prefix_ids = _ids_for_text(tokenizer, prefix)
    full_ids = _ids_for_text(tokenizer, prefix + continuation)
    if full_ids[: len(prefix_ids)] == prefix_ids:
        return full_ids[len(prefix_ids) :]
    full_ids = _ids_for_text(tokenizer, prefix + " " + continuation)
    if full_ids[: len(prefix_ids)] == prefix_ids:
        return full_ids[len(prefix_ids) :]
    return _ids_for_text(tokenizer, continuation)


def _limit_rhs_by_tokens(tokenizer, rhs: str, rhs_window: int) -> str:
    if rhs_window is None or rhs_window < 0:
        return rhs
    if rhs_window == 0:
        return ""
    ids = _ids_for_text(tokenizer, rhs)
    if len(ids) <= rhs_window:
        return rhs
    return tokenizer.decode(ids[:rhs_window], skip_special_tokens=True)


def _score_clm_continuation_logprob(model, tokenizer, prefix: str, continuation: str, device) -> float:
    prefix, gap = _clm_context_and_gap_from_prefix(prefix)
    continuation = f"{gap}{str(continuation).lstrip()}"
    prefix_ids = _ids_for_text(tokenizer, prefix)
    continuation_ids = _continuation_ids_from_prefix(tokenizer, prefix, continuation)
    if not prefix_ids or not continuation_ids:
        return float("-inf")
    input_ids = torch.tensor([prefix_ids + continuation_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = torch.log_softmax(outputs.logits, dim=-1)
    logprob = 0.0
    for offset, token_id in enumerate(continuation_ids):
        pred_pos = len(prefix_ids) + offset - 1
        logprob += float(log_probs[0, pred_pos, token_id].item())
    return logprob


def compute_probability_shift_score_clm_sequence(
    model,
    tokenizer,
    df,
    targets,
    key_text='masked',
    batch_size=16,
    dataset_class_col=None,
    rhs_window: int = -1,
    use_row_wise: bool = False,
    return_per_row_df: bool = False,
    score_continuation=None,
):
    """
    Decoder-only sequence-cloze scoring.

    For each candidate, score log P(candidate + RHS_window | prefix), where prefix
    and RHS come from splitting key_text at [MASK]. Aggregates candidate-softmax
    probabilities so downstream probability-shift code can keep its existing shape.
    """
    model.eval()
    device = model.device
    _tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

    if dataset_class_col is None:
        if "label_class" in df.columns:
            dataset_class_col = "label_class"
        elif "factual_id" in df.columns:
            dataset_class_col = "factual_id"

    rows = _rows_with_source_index(df)
    probs_by_dataset = defaultdict(lambda: defaultdict(list))
    per_row_records: List[Dict[str, object]] = []

    def _default_score(m, tok, prefix_text, token_text, rhs_text, dev):
        return _score_clm_continuation_logprob(m, tok, prefix_text, token_text + rhs_text, dev)

    score_fn = score_continuation or _default_score

    with torch.no_grad():
        for row in rows:
            text = str(row.get(key_text, ""))
            if "[MASK]" not in text:
                continue
            prefix, rhs = text.split("[MASK]", 1)
            rhs = _limit_rhs_by_tokens(_tokenizer, rhs, rhs_window)

            if use_row_wise:
                candidate_items = [
                    (str(row.get(DEFAULT_DATASET_CLASS_COL)), str(row.get(DEFAULT_FACTUAL_COL))),
                    (str(row.get(DEFAULT_OTHER_CLASS_COL)), str(row.get(DEFAULT_ALTERNATIVE_COL))),
                ]
            else:
                candidate_items = [
                    (str(class_name), str(token))
                    for class_name, tokens in targets.items()
                    for token in tokens
                    if token is not None
                ]
            candidate_items = [
                (cls, tok)
                for cls, tok in candidate_items
                if cls and cls != "None" and tok and tok != "None"
            ]
            if not candidate_items:
                continue

            scored_items = [
                (
                    cls,
                    score_fn(model, _tokenizer, prefix, token, rhs, device),
                )
                for cls, token in candidate_items
            ]
            finite_items = [(cls, logp) for cls, logp in scored_items if np.isfinite(logp)]
            if not finite_items:
                continue
            classes, values = zip(*finite_items)
            probs = torch.softmax(torch.tensor(values), dim=0).tolist()
            class_probs: Dict[str, float] = defaultdict(float)
            for cls, prob in zip(classes, probs):
                class_probs[cls] += float(prob)

            dataset_class = row.get(dataset_class_col) if dataset_class_col else None
            if dataset_class is None and use_row_wise:
                dataset_class = row.get(DEFAULT_DATASET_CLASS_COL)
            if dataset_class is None:
                for cls, prob in class_probs.items():
                    probs_by_dataset[cls][cls].append(prob)
            else:
                for cls, prob in class_probs.items():
                    probs_by_dataset[dataset_class][cls].append(prob)

            if return_per_row_df:
                record = _decoder_raw_result_row(
                    row,
                    key_text=key_text,
                    dataset_class_col=dataset_class_col,
                    dataset_class=dataset_class,
                    class_probs=dict(class_probs),
                    score_kind="clm_sequence_cloze_candidate_softmax",
                )
                if use_row_wise:
                    ds_cls = row.get(DEFAULT_DATASET_CLASS_COL)
                    other_cls = row.get(DEFAULT_OTHER_CLASS_COL)
                    record.update({
                        "other_class": _csv_scalar(other_cls),
                        "P_factual": class_probs.get(str(ds_cls), 0.0),
                        "P_alternative": class_probs.get(str(other_cls), 0.0),
                    })
                per_row_records.append(record)

    probs_by_dataset_means = {
        dataset_class: {
            class_name: float(np.mean(probs)) if probs else 0.0
            for class_name, probs in class_probs.items()
        }
        for dataset_class, class_probs in probs_by_dataset.items()
    }
    if not probs_by_dataset_means:
        raise ValueError("No valid CLM sequence-cloze probabilities computed; check data, targets, and [MASK].")
    if return_per_row_df:
        return probs_by_dataset_means, pd.DataFrame(per_row_records)
    return probs_by_dataset_means


def evaluate_probability_shift_score(
    model,
    tokenizer,
    targets,
    eval_data_df,
    key_text="masked",
    dataset_class_col=None,
    factual_dataset_class_col=None,
    use_row_wise: bool = False,
    return_per_row_df: bool = False,
    objective: str = "auto",
    rhs_window: int = -1,
) -> Union[Dict[str, Dict[str, float]], Tuple[Dict[str, Dict[str, float]], pd.DataFrame], Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]]:
    """
    Compute probabilities for all classes on all datasets.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        targets: Dict mapping class_name -> list of tokens (ignored if use_row_wise=True)
        eval_data_df: DataFrame with evaluation data (must have factual/alternative if use_row_wise=True)
        key_text: Column name for masked text
        dataset_class_col: Column name identifying dataset class (ignored if use_row_wise=True)
        use_row_wise: If True, compute P(factual) and P(alternative) per row by dataset class.
        return_per_row_df: If True, return (result_dict, per_row_DataFrame) for CSV export.
            Row-wise mode reports P_factual/P_alternative; static target-token mode reports
            one ``p_class_<class>`` column per decoder target class.

    Returns:
        Dict or (Dict, DataFrame): {dataset_class: {class_name: prob, ...}}. If
        return_per_row_df, returns the aggregate dict plus a per-sample DataFrame.
    """
    if objective == "auto":
        objective = prediction_eval_kind(model)
    if objective == "clm_sequence_cloze":
        return compute_probability_shift_score_clm_sequence(
            model,
            tokenizer,
            eval_data_df,
            targets=targets,
            key_text=key_text,
            dataset_class_col=dataset_class_col,
            use_row_wise=use_row_wise,
            return_per_row_df=return_per_row_df,
            rhs_window=rhs_window,
        )
    if objective == "seq2seq_decoder" or objective == "seq2seq_decoder_sequence_cloze":
        include_span_sentinels = objective == "seq2seq_decoder_sequence_cloze"
        return compute_probability_shift_score_clm_sequence(
            model,
            tokenizer,
            eval_data_df,
            targets=targets,
            key_text=key_text,
            dataset_class_col=dataset_class_col,
            use_row_wise=use_row_wise,
            return_per_row_df=return_per_row_df,
            rhs_window=rhs_window,
            score_continuation=lambda m, tok, prefix, token, rhs, dev: score_seq2seq_continuation_logprob(
                m, tok, prefix, token, dev, rhs=rhs, include_span_sentinels=include_span_sentinels
            ),
        )
    if objective == "seq2seq_encoder_mlm":
        if use_row_wise:
            return compute_probability_shift_score_row_wise(
                model,
                tokenizer,
                eval_data_df,
                key_text=key_text,
                factual_col=DEFAULT_FACTUAL_COL,
                alternative_col=DEFAULT_ALTERNATIVE_COL,
                dataset_class_col=DEFAULT_DATASET_CLASS_COL,
                other_class_col=DEFAULT_OTHER_CLASS_COL,
                return_per_row_df=return_per_row_df,
            )
        return compute_probability_shift_score_clm(
            model,
            tokenizer,
            eval_data_df,
            targets=targets,
            key_text=key_text,
            dataset_class_col=dataset_class_col,
            factual_dataset_class_col=factual_dataset_class_col,
        )
    if use_row_wise:
        out = compute_probability_shift_score_row_wise(
            model,
            tokenizer,
            eval_data_df,
            key_text=key_text,
            factual_col=DEFAULT_FACTUAL_COL,
            alternative_col=DEFAULT_ALTERNATIVE_COL,
            dataset_class_col=DEFAULT_DATASET_CLASS_COL,
            other_class_col=DEFAULT_OTHER_CLASS_COL,
            return_per_row_df=return_per_row_df,
        )
        return out
    return compute_probability_shift_score_mlm(
        model, tokenizer, eval_data_df, targets=targets, key_text=key_text,
        dataset_class_col=dataset_class_col, factual_dataset_class_col=factual_dataset_class_col,
        return_per_row_df=return_per_row_df,
    )


__all__ = [
    "resolve_prediction_target_token_id",
    "annotate_text_probability_rows",
    "compute_probability_shift_score_row_wise",
    "compute_probability_shift_score_mlm",
    "compute_probability_shift_score_clm",
    "compute_probability_shift_score_clm_sequence",
    "compute_lms",
    "evaluate_probability_shift_score",
]
