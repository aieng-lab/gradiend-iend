"""Regression tests for single-pass encoder evaluation and dataset balancing.

Covers:
1. Default ``TrainingArguments.encoder_eval_balance is False``.
2. ``create_eval_data(..., max_size=None)`` visits every row strictly once with
   zero duplicate head rows and zero dropped word-ordered tail rows.
3. ``encoder_eval_balance=True`` uses deterministic shuffled stratified
   undersampling (zero duplicates, includes tail words across seeds).
4. ``max_size`` caps per group without cyclic wrap-around duplicates.
5. ``TextBatchedDatasetBase`` enables ``shuffle_within`` by default when
   ``balance_column`` is set even at ``batch_size=1``, while respecting explicit
   ``shuffle_within=False``.
6. ``cross_encoding._gradient_dataset_for_unified_df`` evaluates unequal probe
   classes strictly once (``balance_column=None``).
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import torch

from gradiend.comparison.cross_encoding import _gradient_dataset_for_unified_df
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.unified_schema import (
    UNIFIED_ALTERNATIVE,
    UNIFIED_FACTUAL,
    UNIFIED_MASKED,
)
from gradiend.trainer.text.prediction.dataset import TextTrainingDataset
from gradiend.trainer.text.prediction.trainer import (
    TextPredictionConfig,
    TextPredictionTrainer,
)
from tests.testing_mocks import MockTokenizer


def _make_imbalanced_word_ordered_df() -> pd.DataFrame:
    """Word-ordered BE/AE dataset mimicking the global vocabulary fold bug."""
    rows = []
    # BE: 4 lift + 6 holiday = 10 rows
    for i in range(4):
        rows.append(
            {
                "masked": f"Take the [MASK] be_{i} upstairs.",
                "split": "test",
                "label_class": "BE",
                "label": "lift",
                "alternative_class": "AE",
                "alternative": "elevator",
            }
        )
    for i in range(6):
        rows.append(
            {
                "masked": f"Enjoy the [MASK] be_{i} soon.",
                "split": "test",
                "label_class": "BE",
                "label": "holiday",
                "alternative_class": "AE",
                "alternative": "vacation",
            }
        )
    # AE: 6 elevator + 6 soccer + 6 vacations = 18 rows (tail word "vacations" last!)
    for i in range(6):
        rows.append(
            {
                "masked": f"Take the [MASK] ae_{i} upstairs.",
                "split": "test",
                "label_class": "AE",
                "label": "elevator",
                "alternative_class": "BE",
                "alternative": "lift",
            }
        )
    for i in range(6):
        rows.append(
            {
                "masked": f"Watch the [MASK] ae_{i} match.",
                "split": "test",
                "label_class": "AE",
                "label": "soccer",
                "alternative_class": "BE",
                "alternative": "football",
            }
        )
    for i in range(6):
        rows.append(
            {
                "masked": f"Plan your [MASK] ae_{i} early.",
                "split": "test",
                "label_class": "AE",
                "label": "vacations",
                "alternative_class": "BE",
                "alternative": "holidays",
            }
        )
    return pd.DataFrame(rows)


def _make_trainer_and_model(df: pd.DataFrame, **arg_overrides):
    tokenizer = MockTokenizer()
    for tok in ("lift", "elevator", "holiday", "vacation", "holidays", "vacations", "soccer", "football"):
        tokenizer.vocab[tok] = len(tokenizer.vocab) + 10

    config = TextPredictionConfig(
        data=df,
        target_classes=["BE", "AE"],
        alternative_col="alternative",
        alternative_class_col="alternative_class",
    )
    args = TrainingArguments(
        source="factual",
        target="diff",
        add_neutral_identity_transitions=False,
        add_identity_for_other_classes=False,
        **arg_overrides,
    )
    trainer = TextPredictionTrainer(
        model=SimpleNamespace(tokenizer=tokenizer),
        config=config,
        args=args,
    )
    mock_model = MagicMock()
    mock_model.tokenizer = tokenizer
    mock_model.gradiend.torch_dtype = torch.float32
    mock_model.gradiend.device_encoder = torch.device("cpu")
    mock_model.gradient_creator = MagicMock(return_value=torch.tensor([1.0, 2.0]))
    return trainer, mock_model


def test_training_arguments_default_encoder_eval_balance_is_false() -> None:
    assert TrainingArguments().encoder_eval_balance is False


def test_create_eval_data_default_visits_every_row_once_no_tail_drop() -> None:
    df = _make_imbalanced_word_ordered_df()
    trainer, mock_model = _make_trainer_and_model(df, seed=0)

    eval_ds = trainer.create_eval_data(mock_model, split="test", max_size=None)
    assert eval_ds.training_data.balance_column is None

    items = [eval_ds[i] for i in range(len(eval_ds))]
    assert len(items) == 28

    texts = [it["text"] for it in items]
    assert len(set(texts)) == 28  # zero duplicates

    word_counts = Counter(it["factual_token"] for it in items)
    assert word_counts == {
        "lift": 4,
        "holiday": 6,
        "elevator": 6,
        "soccer": 6,
        "vacations": 6,
    }
    class_counts = Counter(it["factual_id"] for it in items)
    assert class_counts == {"BE": 10, "AE": 18}


def test_create_eval_data_balance_true_undersamples_shuffled_without_duplicates() -> None:
    df = _make_imbalanced_word_ordered_df()
    vacations_seen = False

    for seed in (0, 1, 2):
        trainer, mock_model = _make_trainer_and_model(
            df, encoder_eval_balance=True, seed=seed
        )
        eval_ds = trainer.create_eval_data(mock_model, split="test", max_size=None)
        assert eval_ds.training_data.balance_column is None

        items = [eval_ds[i] for i in range(len(eval_ds))]
        assert len(items) == 20  # 2 * min(10, 18)

        texts = [it["text"] for it in items]
        assert len(set(texts)) == 20  # zero duplicates

        class_counts = Counter(it["factual_id"] for it in items)
        assert class_counts == {"BE": 10, "AE": 10}
        if any(it["factual_token"] == "vacations" for it in items):
            vacations_seen = True

    assert vacations_seen


def test_create_eval_data_max_size_caps_per_group_without_duplicates() -> None:
    df = _make_imbalanced_word_ordered_df()

    # encoder_eval_balance=False: cap larger group at 12, keep all 10 of smaller group.
    trainer_nat, model_nat = _make_trainer_and_model(df, encoder_eval_balance=False, seed=0)
    ds_nat = trainer_nat.create_eval_data(model_nat, split="test", max_size=12)
    items_nat = [ds_nat[i] for i in range(len(ds_nat))]
    assert len(items_nat) == 22
    assert len({it["text"] for it in items_nat}) == 22
    assert Counter(it["factual_id"] for it in items_nat) == {"BE": 10, "AE": 12}

    # encoder_eval_balance=True: cap both groups at min(10, 6) = 6.
    trainer_bal, model_bal = _make_trainer_and_model(df, encoder_eval_balance=True, seed=0)
    ds_bal = trainer_bal.create_eval_data(model_bal, split="test", max_size=6)
    items_bal = [ds_bal[i] for i in range(len(ds_bal))]
    assert len(items_bal) == 12
    assert len({it["text"] for it in items_bal}) == 12
    assert Counter(it["factual_id"] for it in items_bal) == {"BE": 6, "AE": 6}


def test_dataset_base_shuffle_within_defaults_true_for_balanced_bs1_and_respects_false() -> None:
    tok = MockTokenizer()
    data = pd.DataFrame(
        {
            UNIFIED_MASKED: [f"Sentence {i} [MASK]." for i in range(6)],
            UNIFIED_FACTUAL: [f"token_{i}" for i in range(6)],
            UNIFIED_ALTERNATIVE: [f"token_{i + 10}" for i in range(6)],
            "factual_id": ["A"] * 3 + ["B"] * 3,
            "alternative_id": ["B"] * 3 + ["A"] * 3,
            "label": [1] * 3 + [-1] * 3,
            "feature_pole": ["pos"] * 3 + ["neg"] * 3,
        }
    )

    ds_default = TextTrainingDataset(
        data,
        tok,
        batch_size=1,
        balance_column="feature_pole",
        label_token_protocol="no_leading_space",
    )
    assert ds_default.shuffle_within is True

    ds_unbalanced = TextTrainingDataset(
        data,
        tok,
        batch_size=1,
        balance_column=None,
        label_token_protocol="no_leading_space",
    )
    assert ds_unbalanced.shuffle_within is False

    ds_explicit_false = TextTrainingDataset(
        data,
        tok,
        batch_size=2,
        balance_column="feature_pole",
        shuffle_within=False,
        label_token_protocol="no_leading_space",
    )
    assert ds_explicit_false.shuffle_within is False


def test_cross_encoding_gradient_dataset_preserves_all_unequal_class_probes() -> None:
    tok = MockTokenizer()
    unified_df = pd.DataFrame(
        [
            {
                UNIFIED_MASKED: f"Probe row {i} [MASK].",
                UNIFIED_FACTUAL: f"token_{i}",
                UNIFIED_ALTERNATIVE: f"token_{i + 20}",
                "factual_id": "A" if i < 3 else "B",
                "alternative_id": "B" if i < 3 else "A",
                "label": 1.0 if i < 3 else -1.0,
                "feature_class_id": 0 if i < 3 else 1,
            }
            for i in range(10)
        ]
    )

    class _ProbeTrainer:
        def create_gradient_training_dataset(self, raw_ds, model, target=None):
            return raw_ds

    ds = _gradient_dataset_for_unified_df(
        _ProbeTrainer(),
        SimpleNamespace(tokenizer=tok),
        unified_df,
    )
    assert ds.balance_column is None
    assert len(ds) == 10
    visited_texts = [ds[i]["text"] for i in range(len(ds))]
    assert len(set(visited_texts)) == 10
