"""``add_neutral_identity_transitions`` is tri-state: ``None`` follows ``neutral_data``."""

import pandas as pd
import pytest

from gradiend import TextPredictionTrainer, TrainingArguments
from gradiend.trainer.core.unified_schema import (
    UNIFIED_ALTERNATIVE,
    UNIFIED_ALTERNATIVE_CLASS,
    UNIFIED_FACTUAL,
    UNIFIED_FACTUAL_CLASS,
    UNIFIED_MASKED,
    UNIFIED_SPLIT,
    UNIFIED_TRANSITION,
    transition_id,
)
from tests.testing_mocks import MockTokenizer


def _unified_rows() -> pd.DataFrame:
    rows = []
    for cls, tok in (("white", "w1"), ("white", "w2"), ("black", "b1"), ("black", "b2")):
        other = "black" if cls == "white" else "white"
        rows.append(
            {
                UNIFIED_MASKED: "The person is [MASK]",
                UNIFIED_SPLIT: "train",
                UNIFIED_FACTUAL_CLASS: cls,
                UNIFIED_ALTERNATIVE_CLASS: other,
                UNIFIED_FACTUAL: tok,
                UNIFIED_ALTERNATIVE: other,
                UNIFIED_TRANSITION: transition_id(cls, other),
            }
        )
    return pd.DataFrame(rows)


def _trainer(*, args: TrainingArguments, neutral_data=None) -> TextPredictionTrainer:
    trainer = TextPredictionTrainer(
        model="distilbert-base-cased",
        target_classes=["white", "black"],
        neutral_data=neutral_data,
        args=args,
    )
    trainer._combined_data = _unified_rows()
    trainer._data_loaded = True
    return trainer


def test_default_is_auto_none():
    assert TrainingArguments().add_neutral_identity_transitions is None


def test_rejects_non_bool_values():
    with pytest.raises(TypeError, match="add_neutral_identity_transitions"):
        TrainingArguments(add_neutral_identity_transitions="yes")


def test_auto_is_disabled_without_neutral_data():
    trainer = _trainer(args=TrainingArguments(experiment_dir=None))
    assert trainer.neutral_identity_transitions_enabled() is False


def test_auto_is_enabled_with_neutral_data():
    neutral = pd.DataFrame({"text": ["Water is wet today."]})
    trainer = _trainer(args=TrainingArguments(experiment_dir=None), neutral_data=neutral)
    assert trainer.neutral_identity_transitions_enabled() is True


def test_explicit_false_wins_over_neutral_data():
    neutral = pd.DataFrame({"text": ["Water is wet today."]})
    trainer = _trainer(
        args=TrainingArguments(experiment_dir=None, add_neutral_identity_transitions=False),
        neutral_data=neutral,
    )
    assert trainer.neutral_identity_transitions_enabled() is False


def test_training_data_without_neutral_data_works_by_default():
    """The quickstart (data only) must not require ``neutral_data``."""
    trainer = _trainer(args=TrainingArguments(experiment_dir=None))
    dataset = trainer.create_training_data(MockTokenizer(), split="train", batch_size=1)
    assert len(dataset) > 0


def test_explicit_true_without_neutral_data_fails_loudly():
    trainer = _trainer(args=TrainingArguments(experiment_dir=None, add_neutral_identity_transitions=True))
    with pytest.raises(ValueError, match="requires TextPredictionConfig.neutral_data"):
        trainer.create_training_data(MockTokenizer(), split="train", batch_size=1)
