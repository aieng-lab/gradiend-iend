"""Published English pronoun datasets used by GRADIEND examples."""

from __future__ import annotations

import pandas as pd

EN_PRONOUNS_HF_DATASET = "aieng-lab/en-pronouns"
EN_PRONOUN_NEUTRAL_HF_DATASET = "aieng-lab/en-pronoun-neutral"
EN_PRONOUN_HF_SPLITS = ["train", "validation", "test"]


def load_english_pronoun_neutral_data(split: str = "train") -> pd.DataFrame:
    """Load the published neutral pronoun contexts as a pandas DataFrame."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Loading the published English pronoun neutral dataset requires "
            "`datasets`. Install with: pip install gradiend[data]"
        ) from exc
    return load_dataset(EN_PRONOUN_NEUTRAL_HF_DATASET, split=split).to_pandas()
