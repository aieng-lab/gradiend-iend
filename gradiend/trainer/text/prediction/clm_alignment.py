"""CLM divergence-point alignment for multi-token factual and alternative targets.

Decoder-only causal language models predict the next token given preceding context.
When factual and alternative target phrases share a common token prefix of length
k (where k < len(factual) and k < len(alternative)), the causal divergence point
occurs at token position k+1 (0-indexed position k). At this position, preceding
shared tokens act as context, and the contrastive supervision is applied to
predict the differing tokens factual[k] vs alternative[k].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd


class ClmAlignmentError(ValueError):
    """Raised when factual and alternative targets cannot be contrastively aligned for CLM."""

    def __init__(
        self,
        message: str,
        *,
        factual: Optional[str] = None,
        alternative: Optional[str] = None,
        context_prefix: Optional[str] = None,
        factual_ids: Optional[Sequence[int]] = None,
        alternative_ids: Optional[Sequence[int]] = None,
        k: Optional[int] = None,
        reason: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.factual = factual
        self.alternative = alternative
        self.context_prefix = context_prefix
        self.factual_ids = tuple(factual_ids) if factual_ids is not None else ()
        self.alternative_ids = tuple(alternative_ids) if alternative_ids is not None else ()
        self.k = k
        self.reason = reason


@dataclass(frozen=True)
class ClmDivergence:
    """Result of aligning factual and alternative targets for decoder-only CLM supervision.

    Attributes:
        shared_ids: Token ids of the shared prefix between factual and alternative.
        factual_id: Divergence token id to predict for the factual target.
        alternative_id: Divergence token id to predict for the alternative target.
        n: Total number of token ids in the factual continuation.
        m: Total number of token ids in the alternative continuation.
    """

    shared_ids: Tuple[int, ...]
    factual_id: int
    alternative_id: int
    n: int
    m: int

    @property
    def k(self) -> int:
        """Length of the shared common prefix."""
        return len(self.shared_ids)


def _ids_for_text(tokenizer: Any, text: str) -> List[int]:
    """Tokenize ``text`` without special tokens and return token ids.

    Args:
        tokenizer: Tokenizer used for encoding.
        text: Text to tokenize.
    """
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    return tok(str(text), add_special_tokens=False)["input_ids"]


def continuation_ids_from_prefix(tokenizer: Any, prefix: str, continuation: str) -> List[int]:
    """Return token ids contributed by ``continuation`` after ``prefix``.

    Handles tokenizer gap / space mechanics by checking full text tokenization against prefix.

    Args:
        tokenizer: Tokenizer used for encoding.
        prefix: Text before the continuation.
        continuation: Candidate continuation text.
    """
    if not continuation:
        return []
    prefix_ids = _ids_for_text(tokenizer, prefix)
    full_ids = _ids_for_text(tokenizer, prefix + continuation)
    if full_ids[: len(prefix_ids)] == prefix_ids:
        return full_ids[len(prefix_ids) :]
    full_ids = _ids_for_text(tokenizer, prefix + " " + continuation)
    if full_ids[: len(prefix_ids)] == prefix_ids:
        return full_ids[len(prefix_ids) :]
    clean_prefix = prefix.rstrip()
    if clean_prefix and clean_prefix != prefix:
        clean_prefix_ids = _ids_for_text(tokenizer, clean_prefix)
        full_ids = _ids_for_text(tokenizer, prefix + continuation)
        if full_ids[: len(clean_prefix_ids)] == clean_prefix_ids:
            return full_ids[len(clean_prefix_ids) :]
        full_ids = _ids_for_text(tokenizer, clean_prefix + " " + continuation.lstrip())
        if full_ids[: len(clean_prefix_ids)] == clean_prefix_ids:
            return full_ids[len(clean_prefix_ids) :]
    return _ids_for_text(tokenizer, continuation)


def align_clm_divergence(
    tokenizer: Any,
    context_prefix: str,
    factual: str,
    alternative: str,
) -> ClmDivergence:
    """Align factual and alternative continuations to find their first divergence point.

    Tokenizes ``factual`` and ``alternative`` as continuations of ``context_prefix``
    (respecting leading-space and BPE boundary behaviors). Let n = len(factual_ids),
    m = len(alternative_ids), and k = length of their longest common prefix.

    The alignment is valid iff strict inequality holds: ``k < n AND k < m``.
    When valid:

    - ``shared_ids`` = factual_ids[:k] (identical to alternative_ids[:k])
    - ``factual_id`` = factual_ids[k]
    - ``alternative_id`` = alternative_ids[k]

    If ``k == n`` or ``k == m``, one phrase is a token prefix of the other (e.g.
    'bride' vs 'bridegroom'). If ``factual_ids == alternative_ids``, the tokens are
    identical. Both cases are invalid and raise :class:`ClmAlignmentError`.

    Args:
        tokenizer: Tokenizer or model with a tokenizer.
        context_prefix: Preceding sentence context (or prefix before the mask).
        factual: Factual target string.
        alternative: Alternative target string.

    Returns:
        ClmDivergence instance containing shared_ids, factual_id, alternative_id, n, m.

    Raises:
        ClmAlignmentError: If targets cannot be contrastively aligned.
    """
    f_str = str(factual)
    a_str = str(alternative)
    f_ids = continuation_ids_from_prefix(tokenizer, context_prefix, f_str)
    a_ids = continuation_ids_from_prefix(tokenizer, context_prefix, a_str)
    n = len(f_ids)
    m = len(a_ids)

    if n == 0 or m == 0:
        raise ClmAlignmentError(
            f"Cannot align empty continuation: factual={f_str!r} (ids={f_ids}), "
            f"alternative={a_str!r} (ids={a_ids}) after prefix={context_prefix!r}.",
            factual=f_str,
            alternative=a_str,
            context_prefix=context_prefix,
            factual_ids=f_ids,
            alternative_ids=a_ids,
            k=0,
            reason="empty_tokenization",
        )

    k = 0
    while k < n and k < m and f_ids[k] == a_ids[k]:
        k += 1

    if k == n and k == m:
        raise ClmAlignmentError(
            f"Factual {f_str!r} and alternative {a_str!r} tokenize to identical token "
            f"sequences {f_ids} after prefix {context_prefix!r}; cannot supervise zero contrast.",
            factual=f_str,
            alternative=a_str,
            context_prefix=context_prefix,
            factual_ids=f_ids,
            alternative_ids=a_ids,
            k=k,
            reason="identical_tokens",
        )

    if k == n or k == m:
        shorter = "factual" if n < m else "alternative"
        longer = "alternative" if n < m else "factual"
        raise ClmAlignmentError(
            f"One target is a token-prefix of the other: {shorter} (n={n}, ids={f_ids if n < m else a_ids}) "
            f"is a prefix of {longer} (m={m}, ids={a_ids if n < m else f_ids}) with common prefix k={k}. "
            f"The shorter side has no valid next-token contrast.",
            factual=f_str,
            alternative=a_str,
            context_prefix=context_prefix,
            factual_ids=f_ids,
            alternative_ids=a_ids,
            k=k,
            reason="prefix_overlap",
        )

    return ClmDivergence(
        shared_ids=tuple(f_ids[:k]),
        factual_id=f_ids[k],
        alternative_id=a_ids[k],
        n=n,
        m=m,
    )


def audit_clm_alignment(
    tokenizer: Any,
    pairs_or_df: Any,
    context_prefix: str = "",
) -> pd.DataFrame:
    """Audit target pairs or DataFrame rows for decoder-only CLM divergence alignment.

    Diagnostic utility to inspect which rows are single-token, multi-token aligned,
    or unalignable (invalid).

    Args:
        tokenizer: Tokenizer used for evaluation/training.
        pairs_or_df: DataFrame with factual/alternative (and optionally masked) columns,
            or an iterable of (factual, alternative) string tuples or dicts.
        context_prefix: Default context prefix if not present in the row.

    Returns:
        DataFrame with columns ['factual', 'alternative', 'prefix', 'factual_ids',
        'alternative_ids', 'n', 'm', 'k', 'status', 'reason'].
        'status' is one of:

        - 'single': n=1, m=1, k=0 (standard single-token)
        - 'aligned': k < n and k < m (multi-token divergence aligned)
        - 'invalid': prefix overlap, identical tokens, or empty tokens
    """
    records: List[Dict[str, Any]] = []

    if isinstance(pairs_or_df, pd.DataFrame):
        df = pairs_or_df
        fact_col = "factual" if "factual" in df.columns else ("label" if "label" in df.columns else None)
        alt_col = "alternative" if "alternative" in df.columns else None
        mask_col = "masked" if "masked" in df.columns else ("text" if "text" in df.columns else None)
        for _, row in df.iterrows():
            f_val = str(row[fact_col]) if fact_col and fact_col in row else ""
            a_val = str(row[alt_col]) if alt_col and alt_col in row else ""
            prefix = context_prefix
            if mask_col and mask_col in row and pd.notna(row[mask_col]):
                raw_text = str(row[mask_col])
                if "[MASK]" in raw_text:
                    prefix = raw_text.split("[MASK]")[0]
                elif not context_prefix:
                    prefix = raw_text
            records.append({"factual": f_val, "alternative": a_val, "prefix": prefix})
    elif isinstance(pairs_or_df, (list, tuple)):
        for item in pairs_or_df:
            if isinstance(item, dict):
                f_val = str(item.get("factual", item.get("label", "")))
                a_val = str(item.get("alternative", ""))
                prefix = item.get("prefix", item.get("masked", context_prefix))
                if "[MASK]" in prefix:
                    prefix = prefix.split("[MASK]")[0]
                records.append({"factual": f_val, "alternative": a_val, "prefix": prefix})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                records.append({"factual": str(item[0]), "alternative": str(item[1]), "prefix": context_prefix})
            else:
                raise ValueError(f"Unrecognized pair format: {item!r}")
    else:
        raise TypeError(f"pairs_or_df must be DataFrame, list, or tuple, got {type(pairs_or_df).__name__}")

    results: List[Dict[str, Any]] = []
    for rec in records:
        f = rec["factual"]
        a = rec["alternative"]
        p = rec["prefix"]
        f_ids = continuation_ids_from_prefix(tokenizer, p, f)
        a_ids = continuation_ids_from_prefix(tokenizer, p, a)
        n = len(f_ids)
        m = len(a_ids)

        if n == 0 or m == 0:
            status = "invalid"
            reason = "empty_tokenization"
            k = 0
        else:
            k = 0
            while k < n and k < m and f_ids[k] == a_ids[k]:
                k += 1
            if k == n and k == m:
                status = "invalid"
                reason = "identical_tokens"
            elif k == n or k == m:
                status = "invalid"
                reason = "prefix_overlap"
            elif n == 1 and m == 1 and k == 0:
                status = "single"
                reason = "single_token"
            else:
                status = "aligned"
                reason = "divergence_aligned"

        results.append({
            "factual": f,
            "alternative": a,
            "prefix": p,
            "factual_ids": f_ids,
            "alternative_ids": a_ids,
            "n": n,
            "m": m,
            "k": k,
            "status": status,
            "reason": reason,
        })

    return pd.DataFrame(results)
