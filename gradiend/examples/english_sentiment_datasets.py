"""Published English NRC-adjective sentiment datasets used by GRADIEND examples."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

EN_SENTIMENT_NRC_HF_DATASET = "aieng-lab/en-sentiment-nrc"
EN_SENTIMENT_NRC_NEUTRAL_HF_DATASET = "aieng-lab/en-sentiment-nrc-neutral"
# Vocabulary-held-out by adjective (package-paper training splits).
EN_SENTIMENT_NRC_HF_SUBSET_SPLIT = "split"
# Split column as written by ``generate_data`` (use for multi-seed re-splits).
EN_SENTIMENT_NRC_HF_SUBSET_DEFAULT = "default"
EN_SENTIMENT_HF_SPLITS = ["train", "validation", "test"]


def load_english_sentiment_neutral_data(split: str = "train") -> pd.DataFrame:
    """Load the published neutral sentiment contexts as a pandas DataFrame."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Loading the published English sentiment neutral dataset requires "
            "`datasets`. Install with: pip install gradiend[data]"
        ) from exc
    return load_dataset(EN_SENTIMENT_NRC_NEUTRAL_HF_DATASET, split=split).to_pandas()


def english_sentiment_hf_training_kwargs(
    *,
    subset: str = EN_SENTIMENT_NRC_HF_SUBSET_SPLIT,
    splits: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Kwargs for ``TextPredictionConfig`` / suite constructors (paper HF recipe)."""
    return {
        "hf_dataset": EN_SENTIMENT_NRC_HF_DATASET,
        "hf_subset": subset,
        "hf_splits": list(splits) if splits is not None else list(EN_SENTIMENT_HF_SPLITS),
    }
