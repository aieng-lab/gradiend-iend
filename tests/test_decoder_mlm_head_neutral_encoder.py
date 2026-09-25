"""Regression tests: neutral encoder rows under clm_mlm_head must not require MLM-head labels."""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import torch

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.signals import Signal, SignalScope
from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer
from gradiend.trainer.text.prediction.dataset import create_masked_pair_from_text
from tests.testing_mocks import MockTokenizer, SimpleMockModel


class _Gpt2StyleTokenizer(MockTokenizer):
    """Decoder-only tokenizer stub (no native [MASK] token)."""

    mask_token = None
    mask_token_id = None


def _make_clm_mlm_head_trainer() -> TextPredictionTrainer:
    trainer = object.__new__(TextPredictionTrainer)
    trainer._training_args = TrainingArguments(
        experiment_dir="runs/test_decoder_mlm_head_neutral",
        prediction_objective="clm_mlm_head",
        seed=0,
    )
    mock_objective = Mock()
    mock_objective.name = "clm_mlm_head"
    trainer._prediction_objective = lambda _tokenizer=None: mock_objective
    trainer.config = TextPredictionConfig(
        data=pd.DataFrame(),
        target_classes=["masc_nom", "fem_nom"],
        masked_col="masked",
        split_col="split",
    )
    return trainer


def _make_model_with_gradiend(tokenizer: MockTokenizer) -> Mock:
    gradiend = Mock()
    gradiend.torch_dtype = torch.float32
    gradiend.device_encoder = torch.device("cpu")

    model = Mock()
    model.tokenizer = tokenizer
    model.base_model = tokenizer
    model.is_seq2seq_model = False
    model.gradiend = gradiend
    model.forward_clm_gradients = Mock(return_value=torch.randn(16, dtype=torch.float32))
    model.gradient_creator = Mock(side_effect=AssertionError("MLM-head gradient_creator must not be used"))
    model.encode = lambda _grad, **kwargs: 0.42 if kwargs.get("return_float") else torch.tensor(0.42)
    return model


def test_neutral_encoder_gradient_creator_uses_forward_clm_gradients():
    trainer = _make_clm_mlm_head_trainer()
    model = Mock()
    model.forward_clm_gradients = Mock()
    model.gradient_creator = Mock()

    assert trainer._neutral_encoder_gradient_creator(model) is model.forward_clm_gradients


def test_neutral_encoder_gradient_creator_keeps_default_for_other_objectives():
    trainer = _make_clm_mlm_head_trainer()
    mock_objective = Mock()
    mock_objective.name = "clm_next_token"
    trainer._prediction_objective = lambda _tokenizer=None: mock_objective
    model = Mock()
    model.forward_clm_gradients = Mock()
    model.gradient_creator = Mock()

    assert trainer._neutral_encoder_gradient_creator(model) is model.gradient_creator


@pytest.fixture
def train_eval_entries():
    return [
        {
            "template": "Der [MASK] Mann läuft schnell",
            "factual_token": "große",
            "alternative_token": "kleine",
        },
        {
            "template": "Die [MASK] Frau sitzt ruhig",
            "factual_token": "alte",
            "alternative_token": "junge",
        },
    ]


def test_encode_neutral_training_masked_rows_under_clm_mlm_head(train_eval_entries):
    trainer = _make_clm_mlm_head_trainer()
    tokenizer = _Gpt2StyleTokenizer()
    model = _make_model_with_gradiend(tokenizer)

    with patch(
        "gradiend.trainer.text.prediction.trainer.create_masked_pair_from_text",
        wraps=create_masked_pair_from_text,
    ) as masked_pair:
        rows = trainer._encode_neutral_training_masked_rows(
            model,
            train_eval_entries,
            excluded_tokens=["der", "die"],
            factual_token_key="factual_token",
            alternative_token_key="alternative_token",
            max_size=2,
            torch_dtype=torch.float32,
            device=torch.device("cpu"),
        )

    assert masked_pair.call_count == 2
    assert len(rows) == 2
    assert all(row["type"] == "neutral_training_masked" for row in rows)
    assert all(isinstance(row["encoded"], float) for row in rows)
    assert model.forward_clm_gradients.call_count == 2
    model.gradient_creator.assert_not_called()


def test_encode_neutral_dataset_rows_under_clm_mlm_head():
    trainer = _make_clm_mlm_head_trainer()
    tokenizer = _Gpt2StyleTokenizer()
    model = _make_model_with_gradiend(tokenizer)
    neutral_df = pd.DataFrame(
        {
            "text": [
                "Der Hund schläft auf dem Sofa heute Abend.",
                "Die Katze springt über den Zaun im Garten.",
            ],
        }
    )

    rows = trainer._encode_neutral_dataset_rows(
        model,
        neutral_df,
        encoder_kwargs={"text_col": "text"},
        masked_col_name="masked",
        excluded_tokens=["der", "die"],
        max_size=2,
        torch_dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert len(rows) == 2
    assert all(row["type"] == "neutral_dataset" for row in rows)
    assert all(isinstance(row["encoded"], float) for row in rows)
    assert model.forward_clm_gradients.call_count == 2
    model.gradient_creator.assert_not_called()


def test_encode_neutral_dataset_rows_caps_max_size_to_population():
    """max_size larger than the neutral pool must not raise on df.sample."""
    trainer = _make_clm_mlm_head_trainer()
    tokenizer = _Gpt2StyleTokenizer()
    model = _make_model_with_gradiend(tokenizer)
    neutral_df = pd.DataFrame(
        {
            "text": [
                "Der Hund schläft auf dem Sofa heute Abend.",
                "Die Katze springt über den Zaun im Garten.",
            ],
        }
    )

    rows = trainer._encode_neutral_dataset_rows(
        model,
        neutral_df,
        encoder_kwargs={"text_col": "text"},
        masked_col_name="masked",
        excluded_tokens=["der", "die"],
        max_size=200,
        torch_dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert len(rows) == 2
    assert all(row["type"] == "neutral_dataset" for row in rows)


def _make_activation_signal_trainer() -> TextPredictionTrainer:
    trainer = object.__new__(TextPredictionTrainer)
    trainer._training_args = TrainingArguments(
        experiment_dir="runs/test_activation_neutral_signal",
        signal=Signal.activation(),
        signal_scope=SignalScope.from_values(activation_sites=["embeddings"]),
        seed=0,
    )
    mock_objective = Mock()
    mock_objective.name = "mlm"
    trainer._prediction_objective = lambda _tokenizer=None: mock_objective
    trainer.config = TextPredictionConfig(
        data=pd.DataFrame(),
        target_classes=["3SG", "3PL"],
        masked_col="masked",
        split_col="split",
    )
    return trainer


def _make_activation_neutral_model() -> Mock:
    tokenizer = MockTokenizer()
    model = Mock()
    model.tokenizer = tokenizer
    model.base_model = tokenizer
    model.is_seq2seq_model = False
    model.gradiend = Mock(torch_dtype=torch.float32, device_encoder=torch.device("cpu"))
    model.gradient_creator = Mock(side_effect=AssertionError("activation neutral eval must not compute gradients"))
    model.encode = lambda _signal, **kwargs: 0.25 if kwargs.get("return_float") else torch.tensor(0.25)
    return model


def _fake_signal_entries():
    return [
        {
            "source": torch.ones(4, dtype=torch.float32),
            "label": 0,
            "factual_id": "neutral",
            "alternative_id": "neutral",
            "feature_class_id": 0,
            "factual_token": "neutral",
            "alternative_token": "neutral",
            "input_text": "The neutral token appears here.",
            "template": "The [MASK] token appears here.",
        }
    ]


def test_activation_neutral_training_masked_rows_use_signal_aware_dataset_factory():
    trainer = _make_activation_signal_trainer()
    model = _make_activation_neutral_model()
    factory = Mock(return_value=_fake_signal_entries())
    trainer.create_gradient_training_dataset = factory

    with patch(
        "gradiend.trainer.text.prediction.trainer.TextGradientTrainingDataset",
        side_effect=AssertionError("stale gradient-only neutral path"),
    ):
        rows = trainer._encode_neutral_training_masked_rows(
            model,
            [
                {
                    "template": "The [MASK] token appears here today",
                    "factual_token": "factual",
                    "alternative_token": "alternative",
                }
            ],
            excluded_tokens=["factual", "alternative"],
            factual_token_key="factual_token",
            alternative_token_key="alternative_token",
            max_size=1,
            torch_dtype=torch.float32,
            device=torch.device("cpu"),
        )

    assert len(rows) == 1
    assert rows[0]["type"] == "neutral_training_masked"
    assert rows[0]["encoded"] == 0.25
    assert factory.call_count == 1
    model.gradient_creator.assert_not_called()


def test_activation_neutral_dataset_rows_use_signal_aware_dataset_factory():
    trainer = _make_activation_signal_trainer()
    model = _make_activation_neutral_model()
    factory = Mock(return_value=_fake_signal_entries())
    trainer.create_gradient_training_dataset = factory
    neutral_df = pd.DataFrame({"text": ["The neutral token appears here today."]})

    with patch(
        "gradiend.trainer.text.prediction.trainer.TextGradientTrainingDataset",
        side_effect=AssertionError("stale gradient-only neutral path"),
    ):
        rows = trainer._encode_neutral_dataset_rows(
            model,
            neutral_df,
            encoder_kwargs={"text_col": "text"},
            masked_col_name="masked",
            excluded_tokens=[],
            max_size=1,
            torch_dtype=torch.float32,
            device=torch.device("cpu"),
        )

    assert len(rows) == 1
    assert rows[0]["type"] == "neutral_dataset"
    assert rows[0]["encoded"] == 0.25
    assert factory.call_count == 1
    model.gradient_creator.assert_not_called()


def test_activation_neutral_dataset_rows_extract_real_activation_signal():
    trainer = _make_activation_signal_trainer()
    tokenizer = MockTokenizer()
    tokenizer.vocab.update({
        "The": 10,
        "neutral": 11,
        "word": 12,
        "appears": 13,
        "here": 14,
        "today": 15,
    })
    model = Mock()
    model.tokenizer = tokenizer
    model.base_model = SimpleMockModel(vocab_size=200, hidden_size=4)
    model.is_seq2seq_model = False
    model.exclusive_base_gradient_access = lambda: nullcontext()
    model.gradiend = Mock(torch_dtype=torch.float32, device_encoder=torch.device("cpu"))
    model.gradient_creator = Mock(side_effect=AssertionError("activation neutral eval must not compute gradients"))
    encoded_sources = []

    def _encode(signal_tensor, **kwargs):
        encoded_sources.append(signal_tensor.detach().clone())
        return 0.5 if kwargs.get("return_float") else torch.tensor(0.5)

    model.encode = _encode
    neutral_df = pd.DataFrame({"text": ["The neutral word appears here today"]})

    with patch(
        "gradiend.trainer.text.prediction.trainer.TextGradientTrainingDataset",
        side_effect=AssertionError("stale gradient-only neutral path"),
    ):
        rows = trainer._encode_neutral_dataset_rows(
            model,
            neutral_df,
            encoder_kwargs={"text_col": "text"},
            masked_col_name="masked",
            excluded_tokens=[],
            max_size=1,
            torch_dtype=torch.float32,
            device=torch.device("cpu"),
        )

    assert len(rows) == 1
    assert rows[0]["type"] == "neutral_dataset"
    assert rows[0]["encoded"] == 0.5
    assert len(encoded_sources) == 1
    assert encoded_sources[0].shape == (4,)
    assert encoded_sources[0].requires_grad is False
    model.gradient_creator.assert_not_called()
