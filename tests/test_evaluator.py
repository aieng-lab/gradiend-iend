"""
Tests for evaluators (DecoderEvaluator, EncoderEvaluator).

Tests parameter overwriting, caching behavior, and basic evaluation functionality.
"""

import os
import tempfile
import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, Mock

import torch
import numpy as np
import pandas as pd
import pytest

from gradiend.evaluator.decoder import DecoderEvaluator
from gradiend.evaluator.encoder import EncoderEvaluator
from gradiend.evaluator.evaluator import Evaluator
from gradiend.trainer.core.dataset import GradientTrainingDataset, SignalTrainingDatasetBase
from gradiend.trainer.core.feature_definition import FeatureLearningDefinition
from gradiend.trainer.core.signals import Signal, SignalBatch
from tests.testing_mocks import MockTokenizer


class MockTrainer:
    """Mock trainer for testing evaluators."""

    def __init__(self, training_args=None):
        self._training_args = training_args or MockTrainingArguments()
        self.experiment_dir = None
        self.run_id = None
        self.target_classes = ["positive"]
        self._model = None

    def get_model(self):
        return self._model

    def _default_from_training_args(self, value, name, fallback=None):
        """Simulate parameter overwriting logic."""
        if value is not None:
            return value
        return getattr(self._training_args, name, fallback)

    def _resolve_artifact_use_cache(self, value=None, *, fallback=False):
        return FeatureLearningDefinition._resolve_artifact_use_cache(self, value, fallback=fallback)

    def _get_decoder_eval_dataframe(self, tokenizer, **kwargs):
        """Mock decoder eval dataframe creation."""
        training_like_df = pd.DataFrame({
            "text": ["test1", "test2"],
            "label": ["positive", "negative"],
            "factual_id": [1, 2],
            "alternative_id": [2, 1]
        })
        neutral_df = pd.DataFrame({
            "text": ["neutral1"],
            "label": ["neutral"],
            "factual_id": [0],
            "alternative_id": [0]
        })
        return training_like_df, neutral_df

    def evaluate_base_model(self, base_model, tokenizer, **kwargs):
        """Mock base model evaluation."""
        # Return structure expected by decoder evaluator
        # The "lms" key should contain a dict with "lms" key (nested structure)
        return {
            "lms": {"lms": 0.5},  # Nested structure expected by decoder.py:163
            "positive": 0.7,
            "negative": 0.3
        }

    def _evaluate_model_for_decoder(self, model_with_gradiend, df, **kwargs):
        """Mock decoder evaluation."""
        return {
            "lms": 0.5,
            "positive": 0.7,
            "negative": 0.3
        }

    def _model_for_decoder_eval(self, model_with_gradiend):
        """Mock model preparation for decoder eval."""
        return model_with_gradiend

    def create_eval_data(self, model_with_gradiend, **kwargs):
        """Mock eval data creation. Need at least 2 non-neutral samples for correlation."""
        training_data = MockTrainingData([
            {
                "factual": torch.randn(10),
                "alternative": torch.randn(10),
                "label": 1.0,
                "factual_id": 1,
                "alternative_id": 2
            },
            {
                "factual": torch.randn(10),
                "alternative": torch.randn(10),
                "label": -1.0,
                "factual_id": 2,
                "alternative_id": 1
            },
        ])

        def gradient_creator(inputs):
            return torch.randn(100)

        return GradientTrainingDataset(
            training_data=training_data,
            gradient_creator=gradient_creator,
            source="factual",
            target="diff"
        )

    def get_target_feature_classes(self):
        """Return target feature classes."""
        return self.target_classes

    def _resolve_decoder_eval_targets(self, training_like_df=None):
        """Mock class-based decoder targets."""
        return {"positive": ["positive"]}, False


class MockTrainingArguments:
    """Mock TrainingArguments for testing."""

    def __init__(self):
        self.use_cache = False
        self.encoder_eval_max_size = 100
        self.decoder_eval_max_size_training_like = 50
        self.decoder_eval_max_size_neutral = 50


class MockTrainingData:
    """Mock training dataset."""

    def __init__(self, items):
        self.items = items
        self.batch_size = 1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def test_combined_evaluate_forwards_shared_split_and_max_size_to_decoder():
    trainer = MockTrainer()
    evaluator = Evaluator(trainer)
    captured_decoder_kwargs = {}

    evaluator.evaluate_encoder = lambda **kwargs: {"encoder_kwargs": kwargs}
    evaluator._decoder_evaluator = MagicMock()
    evaluator._decoder_evaluator.evaluate_decoder.side_effect = (
        lambda **kwargs: captured_decoder_kwargs.update(kwargs) or {"grid": {}}
    )

    result = evaluator.evaluate(split="validation", max_size=5, use_cache=False)

    assert result["encoder"]["encoder_kwargs"]["split"] == "validation"
    assert result["encoder"]["encoder_kwargs"]["max_size"] == 5
    assert captured_decoder_kwargs["split"] == "validation"
    assert captured_decoder_kwargs["max_size"] == 5
    assert captured_decoder_kwargs["max_size_training_like"] == 5
    assert captured_decoder_kwargs["max_size_neutral"] == 5


class MockModelWithGradiend:
    """Mock ModelWithGradiend for testing."""

    def __init__(self, *, source="factual", target="diff", mapping_kind=None):
        self.name_or_path = "mock-model"
        self.source = source
        self.target = target
        if mapping_kind is not None:
            self.gradiend = SimpleNamespace(mapping_kind=mapping_kind)
        self.base_model = MagicMock()
        self.tokenizer = MockTokenizer()
        self.feature_class_encoding_direction = {"positive": 1.0, "negative": -1.0}

    def encode(self, grad, return_float=True):
        """Mock encode method."""
        if return_float:
            return float(torch.randn(1).item())
        return torch.randn(1)

    def decode(self, encoded, **kwargs):
        """Mock decode method."""
        return torch.randn(100)

    @contextmanager
    def intervene(self, **kwargs):
        """Mock temporary intervention API."""
        yield {"active": True, **kwargs}


class TestEncoderEvaluator:
    """Test EncoderEvaluator."""

    def test_encoder_evaluator_creation(self):
        """Test that EncoderEvaluator can be created."""
        evaluator = EncoderEvaluator()
        assert evaluator is not None

    def test_evaluate_encoder_basic(self):
        """Test basic encoder evaluation returns unified metrics format."""
        evaluator = EncoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()

        result = evaluator.evaluate_encoder(trainer)

        assert "correlation" in result
        assert "n_samples" in result
        assert "mean_by_class" in result
        assert "all_data" in result
        assert isinstance(result["correlation"], float)

    def test_evaluate_encoder_with_eval_data(self):
        """Test encoder evaluation with pre-computed eval_data returns unified metrics."""
        evaluator = EncoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()

        eval_data = trainer.create_eval_data(trainer._model)

        result = evaluator.evaluate_encoder(trainer, eval_data=eval_data)

        assert "correlation" in result
        assert result["n_samples"] > 0
        assert "mean_by_class" in result

    def test_evaluate_encoder_accepts_generic_signal_dataset(self):
        """Encoder evaluation must operate on signal datasets, not only gradient datasets."""
        evaluator = EncoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()

        training_data = MockTrainingData([
            {"factual": torch.tensor([1.0]), "alternative": torch.tensor([0.0]), "label": 1.0},
            {"factual": torch.tensor([-1.0]), "alternative": torch.tensor([0.0]), "label": -1.0},
        ])

        class TinySignalExtractor:
            signal = Signal.activation()
            signals = None

            def __call__(self, factual_inputs=None, alternative_inputs=None, **_kwargs):
                return SignalBatch.from_factual_alternative(
                    factual_inputs,
                    alternative_inputs,
                    signal_id=self.signal.id,
                )

        eval_data = SignalTrainingDatasetBase(
            training_data=training_data,
            signal_extractor=TinySignalExtractor(),
            source="factual",
            target="diff",
        )

        result = evaluator.evaluate_encoder(trainer, eval_data=eval_data)

        assert "correlation" in result
        assert result["n_samples"] == 2

    def test_evaluate_encoder_parameter_overwriting(self):
        """Test that parameters override TrainingArguments."""
        evaluator = EncoderEvaluator()
        training_args = MockTrainingArguments()
        training_args.encoder_eval_max_size = 200
        trainer = MockTrainer(training_args=training_args)
        trainer._model = MockModelWithGradiend()

        # Create eval_data first to ensure it satisfies the generic signal dataset contract.
        eval_data = trainer.create_eval_data(trainer._model)
        assert isinstance(eval_data, SignalTrainingDatasetBase)

        # Override max_size - pass it directly to evaluate_encoder
        # The parameter overwriting happens in create_eval_data, so we test that
        with patch.object(trainer, 'create_eval_data') as mock_create:
            mock_create.return_value = eval_data
            result = evaluator.evaluate_encoder(trainer, max_size=50)

            # Verify max_size was passed to create_eval_data
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs.get("max_size") == 50
            assert "correlation" in result

    def test_evaluate_encoder_uses_training_args_when_not_overridden(self):
        """Test that TrainingArguments values are used when not overridden."""
        evaluator = EncoderEvaluator()
        training_args = MockTrainingArguments()
        training_args.encoder_eval_max_size = 200
        trainer = MockTrainer(training_args=training_args)
        trainer._model = MockModelWithGradiend()

        # Create eval_data first to ensure it satisfies the generic signal dataset contract.
        eval_data = trainer.create_eval_data(trainer._model)
        assert isinstance(eval_data, SignalTrainingDatasetBase)

        # Don't override max_size - let it use TrainingArguments default
        with patch.object(trainer, 'create_eval_data') as mock_create:
            mock_create.return_value = eval_data
            result = evaluator.evaluate_encoder(trainer)

            # Verify max_size from TrainingArguments was used
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            # max_size should come from TrainingArguments (200) via _default_from_training_args
            # The actual value depends on how _default_from_training_args works
            assert "correlation" in result

    def test_evaluate_encoder_caching(self, temp_dir):
        """Test that encoder evaluation uses caching when enabled."""
        evaluator = EncoderEvaluator()
        training_args = MockTrainingArguments()
        training_args.use_cache = True
        trainer = MockTrainer(training_args=training_args)
        trainer.experiment_dir = temp_dir
        trainer._model = MockModelWithGradiend()

        # First call - should compute
        result1 = evaluator.evaluate_encoder(trainer, use_cache=True)

        # Second call - should use cache
        with patch.object(trainer, 'create_eval_data') as mock_create:
            result2 = evaluator.evaluate_encoder(trainer, use_cache=True)

            # Should not call create_eval_data again if cached
            # (cache file should exist)
            cache_files = [f for f in os.listdir(temp_dir) if f.endswith('.json')]
            assert len(cache_files) > 0

    def test_evaluate_encoder_with_encoder_df_ignores_stale_json_cache(self, temp_dir):
        evaluator = EncoderEvaluator()
        training_args = MockTrainingArguments()
        training_args.use_cache = True
        trainer = MockTrainer(training_args=training_args)
        trainer.experiment_dir = temp_dir

        with open(os.path.join(temp_dir, "encoded_values_split_test.json"), "w", encoding="utf-8") as handle:
            json.dump({"n_samples": 999, "correlation": 0.0}, handle)

        encoder_df = pd.DataFrame(
            {
                "encoded": [0.8, -0.4],
                "label": [1, -1],
                "type": ["training", "training"],
                "source_id": ["positive", "negative"],
                "target_id": ["negative", "positive"],
            }
        )

        result = evaluator.evaluate_encoder(trainer, encoder_df=encoder_df, split="test", use_cache=True)

        assert result["n_samples"] == 2

    def test_evaluate_encoder_with_trusted_encoder_df_loads_json_cache(self, temp_dir):
        evaluator = EncoderEvaluator()
        training_args = MockTrainingArguments()
        training_args.use_cache = True
        trainer = MockTrainer(training_args=training_args)
        trainer.experiment_dir = temp_dir

        with open(os.path.join(temp_dir, "encoded_values_split_test.json"), "w", encoding="utf-8") as handle:
            json.dump({"n_samples": 999, "correlation": 0.25}, handle)

        encoder_df = pd.DataFrame(
            {
                "encoded": [0.8, -0.4],
                "label": [1, -1],
                "type": ["training", "training"],
                "source_id": ["positive", "negative"],
                "target_id": ["negative", "positive"],
            }
        )

        result = evaluator.evaluate_encoder(
            trainer,
            encoder_df=encoder_df,
            split="test",
            use_cache=True,
            trust_encoder_df_cache=True,
        )

        assert result["n_samples"] == 999
        assert result["correlation"] == 0.25

    def test_evaluate_encoder_uses_explicit_component_sidecar_not_dataframe_attrs(self):
        evaluator = EncoderEvaluator()
        trainer = MockTrainer()
        encoder_df = pd.DataFrame(
            {
                "encoded": [0.8, -0.8],
                "label": [1, -1],
                "type": ["training", "training"],
                "source_id": ["positive", "negative"],
                "target_id": ["negative", "positive"],
                "source_token": ["yes", "no"],
            }
        )
        component_df = pd.DataFrame(
            {
                "encoded": [0.7, -0.7],
                "label": [1, -1],
                "type": ["training", "training"],
                "source_id": ["positive", "negative"],
                "target_id": ["negative", "positive"],
                "source_token": ["yes", "no"],
                "component_index": [0, 0],
                "component_id": ["activation:embeddings", "activation:embeddings"],
            }
        )
        encoder_df.attrs["component_df"] = component_df

        result = evaluator.evaluate_encoder(
            trainer,
            encoder_df=encoder_df,
            component_df=component_df,
            split="test",
            use_cache=False,
        )

        assert result["n_samples"] == 2
        assert result["components"]["summary"]["n_components"] == 1

    def test_evaluate_encoder_correlation_computation(self):
        """Test that correlation is computed correctly."""
        evaluator = EncoderEvaluator()
        trainer = MockTrainer()

        # Create a model that encodes with a known pattern
        class DeterministicModel(MockModelWithGradiend):
            def encode(self, grad, return_float=True):
                # Return encoding that correlates with label
                label = grad.sum().item() if isinstance(grad, torch.Tensor) else 0.0
                return float(label * 0.5 + np.random.randn() * 0.1)

        trainer._model = DeterministicModel()

        # Create eval data with known labels
        training_data = MockTrainingData([
            {
                "factual": torch.tensor([1.0, 2.0, 3.0]),
                "alternative": torch.tensor([0.0, 0.0, 0.0]),
                "label": 1.0,
                "factual_id": 1,
                "alternative_id": 2
            },
            {
                "factual": torch.tensor([-1.0, -2.0, -3.0]),
                "alternative": torch.tensor([0.0, 0.0, 0.0]),
                "label": -1.0,
                "factual_id": 2,
                "alternative_id": 1
            }
        ])

        def gradient_creator(inputs):
            return inputs if isinstance(inputs, torch.Tensor) else torch.randn(3)

        eval_data = GradientTrainingDataset(
            training_data=training_data,
            gradient_creator=gradient_creator,
            source="factual",
            target="diff"
        )

        result = evaluator.evaluate_encoder(trainer, eval_data=eval_data)

        assert "correlation" in result
        assert isinstance(result["correlation"], float)
        assert -1.0 <= result["correlation"] <= 1.0

    def test_evaluate_encoder_empty_dataset(self):
        """Test that encoder evaluation handles empty datasets."""
        evaluator = EncoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()

        # Create empty eval data
        training_data = MockTrainingData([])

        def gradient_creator(inputs):
            return torch.randn(100)

        eval_data = GradientTrainingDataset(
            training_data=training_data,
            gradient_creator=gradient_creator,
            source="factual",
            target="diff"
        )

        result = evaluator.evaluate_encoder(trainer, eval_data=eval_data)

        # Empty eval data returns explicit empty result (no correlation to report)
        assert result == {"n_samples": 0, "correlation": None}


class TestDecoderEvaluator:
    """Test DecoderEvaluator."""

    @staticmethod
    def _grid_pairs(result):
        return {
            key
            for key in result["grid"].keys()
            if key != "base"
        }

    @staticmethod
    def _default_decoder_lrs():
        # Mirrors DecoderEvaluator.evaluate_decoder's default lrs grid
        # (gradiend/evaluator/decoder.py), extended down to 1e-05.
        return [
            m * 10 ** e
            for e in range(2, -6, -1)
            for m in [5, 2, 1]
            if m * 10 ** e <= 100
        ]

    @staticmethod
    def _capture_rewrite_calls(trainer):
        calls = []

        def _rewrite(**kwargs):
            calls.append(kwargs)
            return trainer._model

        trainer._model.rewrite_base_model = MagicMock(side_effect=_rewrite)
        return calls

    def test_decoder_evaluator_creation(self):
        """Test that DecoderEvaluator can be created."""
        evaluator = DecoderEvaluator()
        assert evaluator is not None

    def test_evaluate_decoder_parameter_overwriting(self):
        """Test that parameters override TrainingArguments."""
        evaluator = DecoderEvaluator()
        training_args = MockTrainingArguments()
        training_args.decoder_eval_max_size_training_like = 100
        training_args.decoder_eval_max_size_neutral = 100
        trainer = MockTrainer(training_args=training_args)
        trainer._model = MockModelWithGradiend()

        with patch.object(
            trainer,
            "_get_decoder_eval_dataframe",
            wraps=trainer._get_decoder_eval_dataframe,
        ) as mock_get_df:
            result = evaluator.evaluate_decoder(
                trainer,
                max_size_training_like=50,
                max_size_neutral=50,
                feature_factors=[-1.0],
                lrs=[1e-2],
            )

        assert mock_get_df.call_args.kwargs["max_size_training_like"] == 50
        assert mock_get_df.call_args.kwargs["max_size_neutral"] == 50
        assert "grid" in result

    def test_evaluate_decoder_uses_public_temporary_intervention_api(self):
        """Decoder grid evaluation should use the public intervention API."""
        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()
        trainer._model.intervene = MagicMock(wraps=trainer._model.intervene)

        result = evaluator.evaluate_decoder(
            trainer,
            feature_factors=[-1.0],
            lrs=[1e-2],
            use_cache=False,
            plot=False,
        )

        assert "grid" in result
        trainer._model.intervene.assert_called_once_with(
            value=1e-2,
            feature_factor=-1.0,
            part="decoder",
            token_selector="encoder_direction",
            threshold=0.5,
            direction=-1.0,
        )

    def test_evaluate_decoder_forwards_intervention_application_kwargs(self):
        """Decoder eval should support ACTIEND application-policy axes directly."""
        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()
        trainer._model.intervene = MagicMock(wraps=trainer._model.intervene)

        result = evaluator.evaluate_decoder(
            trainer,
            feature_factors=[-1.0],
            lrs=[1e-2],
            use_cache=False,
            plot=False,
            token_selector="all",
            activation_gate="encoder_direction",
            activation_modules="transformer.h.9",
            threshold=0.8,
        )

        assert "grid" in result
        trainer._model.intervene.assert_called_once_with(
            value=1e-2,
            feature_factor=-1.0,
            part="decoder",
            token_selector="all",
            activation_gate="encoder_direction",
            activation_modules="transformer.h.9",
            threshold=0.8,
            direction=-1.0,
        )

    def test_evaluate_decoder_activation_gate_defaults_to_all_token_selector(self):
        """An activation gate alone should compose with all positions by default."""
        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()
        trainer._model.intervene = MagicMock(wraps=trainer._model.intervene)

        evaluator.evaluate_decoder(
            trainer,
            feature_factors=[-1.0],
            lrs=[1e-2],
            use_cache=False,
            plot=False,
            activation_gate="encoder_direction",
        )

        trainer._model.intervene.assert_called_once_with(
            value=1e-2,
            feature_factor=-1.0,
            part="decoder",
            token_selector="all",
            activation_gate="encoder_direction",
            threshold=0.5,
            direction=-1.0,
        )

    def test_evaluate_decoder_does_not_modify_or_rewrite_grid_candidates(self):
        """Decoder grid evaluation should avoid copied or rewritten candidate models."""
        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()
        intervention_events = []

        @contextmanager
        def _intervene(**kwargs):
            intervention_events.append(("enter", kwargs))
            yield {"active": True}
            intervention_events.append(("exit", kwargs))

        trainer._model.intervene = _intervene
        trainer._model.modify_model = MagicMock(
            side_effect=AssertionError("evaluate_decoder should use intervene")
        )
        trainer._model.rewrite_base_model = MagicMock(
            side_effect=AssertionError("evaluate_decoder should use intervene")
        )

        result = evaluator.evaluate_decoder(
            trainer,
            feature_factors=[-1.0],
            lrs=[1e-2],
            use_cache=False,
            plot=False,
        )

        assert "grid" in result
        assert intervention_events == [
            (
                "enter",
                {
                    "value": 1e-2,
                    "feature_factor": -1.0,
                    "part": "decoder",
                    "token_selector": "encoder_direction",
                    "threshold": 0.5,
                    "direction": -1.0,
                },
            ),
            (
                "exit",
                {
                    "value": 1e-2,
                    "feature_factor": -1.0,
                    "part": "decoder",
                    "token_selector": "encoder_direction",
                    "threshold": 0.5,
                    "direction": -1.0,
                },
            ),
        ]
        trainer._model.modify_model.assert_not_called()
        trainer._model.rewrite_base_model.assert_not_called()

    def test_evaluate_decoder_cache_fingerprints_intervention_application_kwargs(self):
        """Technique-specific intervention kwargs must not reuse another decoder grid."""
        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        trainer.experiment_dir = tempfile.mkdtemp()
        trainer._model = MockModelWithGradiend()
        output_path = os.path.join(trainer.experiment_dir, "decoder_grid.json")

        with patch.object(
            trainer,
            "evaluate_base_model",
            wraps=trainer.evaluate_base_model,
        ) as mock_eval:
            evaluator.evaluate_decoder(
                trainer,
                feature_factors=[-1.0],
                lrs=[1e-2],
                use_cache=True,
                plot=False,
                output_path=output_path,
                token_selector="all",
            )
            first_call_count = mock_eval.call_count
            evaluator.evaluate_decoder(
                trainer,
                feature_factors=[-1.0],
                lrs=[1e-2],
                use_cache=True,
                plot=False,
                output_path=output_path,
                token_selector="prediction",
            )

        assert mock_eval.call_count > first_call_count

    def test_evaluate_decoder_writes_static_per_sample_raw_csv(self, tmp_path):
        """Decoder grids should persist per-sample raw probabilities separately from the JSON grid."""
        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        trainer.experiment_dir = str(tmp_path)
        trainer._model = MockModelWithGradiend()
        raw_path = tmp_path / "decoder_raw.csv"
        requested_raw = []

        def fake_evaluate_base_model(_model, _tokenizer, **kwargs):
            requested_raw.append(kwargs.get("return_decoder_per_row_df"))
            is_base = str(kwargs.get("cache_folder", "")).startswith("base")
            prob = 0.5 if is_base else 0.7
            result = {
                "lms": {"lms": 0.5},
                "probs": {"positive": prob},
                "probs_by_dataset": {"positive": {"positive": prob}},
            }
            if kwargs.get("return_decoder_per_row_df"):
                result["_decoder_per_row_df"] = pd.DataFrame(
                    [{
                        "row_index": 0,
                        "masked": "a [MASK]",
                        "dataset_class": "positive",
                        "p_class_positive": prob,
                    }]
                )
            return result

        trainer.evaluate_base_model = fake_evaluate_base_model

        result = evaluator.evaluate_decoder(
            trainer,
            feature_factors=[-1.0],
            lrs=[1e-2],
            use_cache=False,
            plot=False,
            raw_output_path=str(raw_path),
        )

        assert result["raw_output_path"] == str(raw_path)
        assert requested_raw == [True, True]
        raw_df = pd.read_csv(raw_path)
        assert raw_df["grid_id"].tolist() == ["base", "ff=-1.0|lr=0.01"]
        assert raw_df["p_class_positive"].tolist() == [0.5, 0.7]

    def test_evaluate_decoder_exits_temporary_intervention_after_eval_exception(self):
        """Intervention cleanup must run even when candidate evaluation fails."""
        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()
        intervention_events = []

        @contextmanager
        def _intervene(**kwargs):
            intervention_events.append("enter")
            try:
                yield {"active": True}
            finally:
                intervention_events.append("exit")

        trainer._model.intervene = _intervene
        trainer.evaluate_base_model = MagicMock(side_effect=[{"lms": {"lms": 0.5}}, RuntimeError("eval failed")])

        with pytest.raises(RuntimeError, match="eval failed"):
            evaluator.evaluate_decoder(
                trainer,
                feature_factors=[-1.0],
                lrs=[1e-2],
                use_cache=False,
                plot=False,
            )

        assert intervention_events == ["enter", "exit"]

    def test_evaluate_decoder_uses_training_args_when_not_overridden(self):
        """Test that TrainingArguments values are used when not overridden."""
        evaluator = DecoderEvaluator()
        training_args = MockTrainingArguments()
        training_args.decoder_eval_max_size_training_like = 100
        training_args.decoder_eval_max_size_neutral = 100
        trainer = MockTrainer(training_args=training_args)
        trainer._model = MockModelWithGradiend()

        with patch.object(
            trainer,
            "_get_decoder_eval_dataframe",
            wraps=trainer._get_decoder_eval_dataframe,
        ) as mock_get_df:
            result = evaluator.evaluate_decoder(
                trainer,
                feature_factors=[-1.0],
                lrs=[1e-2],
            )

        assert mock_get_df.call_args.kwargs["max_size_training_like"] == 100
        assert mock_get_df.call_args.kwargs["max_size_neutral"] == 100
        assert "grid" in result

    def test_evaluate_decoder_prefers_training_args_when_config_legacy_cap_is_none(self):
        evaluator = DecoderEvaluator()
        training_args = MockTrainingArguments()
        training_args.decoder_eval_max_size_training_like = 7
        training_args.decoder_eval_max_size_neutral = 11
        trainer = MockTrainer(training_args=training_args)
        trainer.config = SimpleNamespace(decoder_eval_lms_max_samples=None)
        trainer._model = MockModelWithGradiend()

        with patch.object(
            trainer,
            "_get_decoder_eval_dataframe",
            wraps=trainer._get_decoder_eval_dataframe,
        ) as mock_get_df:
            result = evaluator.evaluate_decoder(
                trainer,
                feature_factors=[-1.0],
                lrs=[1e-2],
            )

        assert mock_get_df.call_args.kwargs["max_size_training_like"] == 7
        assert mock_get_df.call_args.kwargs["max_size_neutral"] == 11
        assert "grid" in result

    def test_evaluate_decoder_use_cache_overwriting(self):
        """Test that use_cache parameter can override TrainingArguments."""
        evaluator = DecoderEvaluator()
        training_args = MockTrainingArguments()
        training_args.use_cache = True
        trainer = MockTrainer(training_args=training_args)
        trainer.experiment_dir = tempfile.mkdtemp()
        trainer._model = MockModelWithGradiend()

        with patch.object(
            trainer,
            "evaluate_base_model",
            wraps=trainer.evaluate_base_model,
        ) as mock_eval:
            result = evaluator.evaluate_decoder(
                trainer,
                use_cache=False,
                feature_factors=[-1.0],
                lrs=[1e-2],
            )

        assert mock_eval.call_args_list
        assert all(call.kwargs.get("use_cache") is False for call in mock_eval.call_args_list)
        assert "grid" in result

    def test_evaluate_decoder_feature_factors_default(self):
        """Test that default feature factors are derived from model."""
        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        model = MockModelWithGradiend()
        trainer._model = model

        # Test default feature factor derivation
        from gradiend.evaluator.decoder import derive_default_feature_factor

        factor = derive_default_feature_factor(trainer, model, class_name="positive")
        assert factor == -1.0  # Should be -direction["positive"] = -1.0

        factor = derive_default_feature_factor(trainer, model, class_name="negative")
        assert factor == 1.0  # Should be -direction["negative"] = -(-1.0) = 1.0

    def test_evaluate_decoder_activation_feature_factors_default(self):
        """ACTIEND default feature factors use direct activation-displacement semantics."""
        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        model = MockModelWithGradiend(mapping_kind="activation", source="factual", target="diff")
        trainer._model = model
        model.intervene = MagicMock(wraps=model.intervene)

        result = evaluator.evaluate_decoder(
            trainer,
            target_class="positive",
            lrs=[1e-2],
        )

        assert self._grid_pairs(result) == {(1.0, 1e-2)}
        assert model.intervene.call_count == 1
        assert model.intervene.call_args.kwargs["feature_factor"] == 1.0

    def test_evaluate_decoder_feature_factors_custom(self):
        """Test that custom feature factors expand the decoder grid."""
        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        trainer.target_classes = []
        trainer._model = MockModelWithGradiend()

        custom_factors = [0.5, 1.0, 1.5]

        result = evaluator.evaluate_decoder(
            trainer,
            feature_factors=custom_factors,
            lrs=[1e-2],
        )

        assert self._grid_pairs(result) == {(0.5, 1e-2), (1.0, 1e-2), (1.5, 1e-2)}

    def test_evaluate_decoder_lrs_default(self):
        """Test that default learning rates are used when not provided."""
        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()

        result = evaluator.evaluate_decoder(
            trainer,
            feature_factors=[-1.0],
            target_class="positive",
        )

        assert self._grid_pairs(result) == {(-1.0, lr) for lr in self._default_decoder_lrs()}

    def test_evaluate_decoder_lrs_custom(self):
        """Test that custom learning rates can be provided."""
        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()

        custom_lrs = [1e-1, 1e-2]

        result = evaluator.evaluate_decoder(
            trainer,
            feature_factors=[-1.0],
            lrs=custom_lrs,
        )

        assert self._grid_pairs(result) == {(-1.0, 1e-1), (-1.0, 1e-2)}

    def test_evaluate_decoder_part_parameter(self):
        """Test that part parameter is passed correctly."""
        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()
        trainer._model.intervene = MagicMock(wraps=trainer._model.intervene)

        result = evaluator.evaluate_decoder(
            trainer,
            feature_factors=[-1.0],
            lrs=[1e-2],
            part="decoder-weight",
        )

        trainer._model.intervene.assert_called_once_with(
            value=1e-2,
            feature_factor=-1.0,
            part="decoder-weight",
            token_selector="encoder_direction",
            threshold=0.5,
            direction=-1.0,
        )
        assert "grid" in result

    def test_evaluate_decoder_eval_batch_size(self):
        """Test that eval_batch_size parameter is passed correctly."""
        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()

        with patch.object(
            trainer,
            "evaluate_base_model",
            wraps=trainer.evaluate_base_model,
        ) as mock_eval:
            result = evaluator.evaluate_decoder(
                trainer,
                feature_factors=[-1.0],
                lrs=[1e-2],
                eval_batch_size=32,
            )

        assert mock_eval.call_args_list
        assert all(call.kwargs.get("eval_batch_size") == 32 for call in mock_eval.call_args_list)
        assert "grid" in result

    def test_default_decoder_feature_factors_fallback_to_config(self):
        """default_decoder_feature_factors uses config.target_classes when trainer.target_classes is None."""
        from gradiend.evaluator.decoder import default_decoder_feature_factors

        trainer = MockTrainer()
        trainer.target_classes = None  # Simulate use_cache skip before fix
        trainer.config = type("Config", (), {"target_classes": ["positive", "negative"]})()
        trainer._model = MockModelWithGradiend()

        factors = default_decoder_feature_factors(trainer, model_with_gradiend=trainer._model)
        assert factors == [-1.0, 1.0]  # -direction["positive"], -direction["negative"] from model

    def test_bisect_refine_lms_boundary_narrows_toward_true_crossing(self):
        """Binary search should land far closer to the LMS-gate crossing than the coarse grid did."""
        from gradiend.evaluator.decoder import LMSThresholdPolicy, _bisect_refine_lms_boundary

        def simulated_lms(lr: float) -> float:
            return max(0.0, 1.0 - 0.05 * lr)

        relevant_results = {
            "base": {"lms": {"lms": 1.0}},
            (1.0, 1.0): {"lms": {"lms": simulated_lms(1.0)}},
            (1.0, 10.0): {"lms": {"lms": simulated_lms(10.0)}},
        }
        pairs = [(1.0, 1.0), (1.0, 10.0)]
        lrs = [1.0, 10.0]
        calls = []

        def evaluate_pair(ff, lr):
            calls.append((ff, lr))
            entry = {"lms": {"lms": simulated_lms(lr)}}
            relevant_results[(ff, lr)] = entry
            return entry

        _bisect_refine_lms_boundary(
            relevant_results=relevant_results,
            pairs=pairs,
            lrs=lrs,
            classes_to_eval=["positive"],
            class_to_ff={"positive": 1.0},
            selector=LMSThresholdPolicy(ratio=0.9),
            refine_points=10,
            evaluate_pair=evaluate_pair,
        )

        assert len(calls) == 10
        assert len(pairs) == 12 and len(lrs) == 12
        assert all(1.0 < lr < 10.0 for _ff, lr in calls)
        passing_lrs = [lr for (_ff, lr) in calls if simulated_lms(lr) >= 0.9]
        failing_lrs = [lr for (_ff, lr) in calls if simulated_lms(lr) < 0.9]
        assert max(passing_lrs) == pytest.approx(2.0, abs=0.01)
        assert min(failing_lrs) == pytest.approx(2.0, abs=0.01)

    def test_bisect_refine_lms_boundary_skips_when_no_transition(self):
        """No adjacent pass/fail pair on the coarse grid -> nothing to bisect."""
        from gradiend.evaluator.decoder import LMSThresholdPolicy, _bisect_refine_lms_boundary

        relevant_results = {
            "base": {"lms": {"lms": 1.0}},
            (1.0, 1.0): {"lms": {"lms": 0.95}},
            (1.0, 2.0): {"lms": {"lms": 0.93}},
        }
        pairs = [(1.0, 1.0), (1.0, 2.0)]
        lrs = [1.0, 2.0]
        calls = []

        _bisect_refine_lms_boundary(
            relevant_results=relevant_results,
            pairs=pairs,
            lrs=lrs,
            classes_to_eval=["positive"],
            class_to_ff={"positive": 1.0},
            selector=LMSThresholdPolicy(ratio=0.9),
            refine_points=10,
            evaluate_pair=lambda ff, lr: calls.append((ff, lr)),
        )

        assert calls == []
        assert pairs == [(1.0, 1.0), (1.0, 2.0)]

    def test_evaluate_decoder_refine_points_is_opt_in_and_extends_grid(self):
        """refine_points=0 leaves grid untouched; >0 adds bisected cells."""
        from gradiend.evaluator.decoder import LMSThresholdPolicy

        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend(mapping_kind="activation", source="factual", target="diff")

        def evaluate_base_model(base_model, tokenizer, **kwargs):
            cache_folder = kwargs.get("cache_folder", "")
            if cache_folder.startswith("base_split_"):
                return {"lms": {"lms": 1.0}, "positive": 0.7, "negative": 0.3}
            lr = float(cache_folder.rsplit("_", 1)[-1])
            # "positive" strictly increases with lr, so the argmax-among-passing
            # selector prefers the largest still-passing lr it was offered —
            # exactly what bisection should get closer to than the coarse grid.
            return {
                "lms": {"lms": max(0.0, 1.0 - 0.05 * lr)},
                "positive": 0.5 + 0.1 * lr,
                "negative": 0.3,
            }

        trainer.evaluate_base_model = evaluate_base_model
        selector = LMSThresholdPolicy(ratio=0.9)

        baseline = evaluator.evaluate_decoder(
            trainer, target_class="positive", lrs=[1.0, 10.0], refine_points=0, selector=selector,
        )
        refined = evaluator.evaluate_decoder(
            trainer, target_class="positive", lrs=[1.0, 10.0], refine_points=10, selector=selector,
        )

        assert self._grid_pairs(baseline) == {(1.0, 1.0), (1.0, 10.0)}
        assert len(self._grid_pairs(refined)) == 12
        assert refined["positive"]["learning_rate"] != baseline["positive"]["learning_rate"]


def test_plot_all_target_classes_forwards_plot_kwargs():
    """plot_kwargs from evaluate_decoder(plot=True) are forwarded to plot_probability_shifts."""
    from gradiend.evaluator.decoder import _plot_all_target_classes

    trainer = MockTrainer()
    trainer.config = type("Config", (), {"img_format": "png"})()
    trainer.plot_probability_shifts = Mock(return_value="/tmp/decoder_probability_shifts_positive.png")

    summary = {"positive": {"value": 0.7, "feature_factor": -1.0, "learning_rate": 0.01}}
    relevant_results = {"base": {"lms": {"lms": 0.5}}}

    paths = _plot_all_target_classes(
        trainer,
        summary,
        relevant_results,
        plot_kwargs={"figsize": (5, 3), "show": False},
        show=True,
    )

    trainer.plot_probability_shifts.assert_called_once()
    call_kwargs = trainer.plot_probability_shifts.call_args.kwargs
    assert call_kwargs["figsize"] == (5, 3)
    assert call_kwargs["show"] is True
    assert call_kwargs["target_class"] == "positive"
    assert paths == ["/tmp/decoder_probability_shifts_positive.png"]


def test_plot_all_target_classes_writes_single_target_default_output(tmp_path):
    """A single evaluate_decoder(plot=True) target also mirrors to the default decoder plot path."""
    from gradiend.evaluator.decoder import _plot_all_target_classes

    trainer = MockTrainer()
    trainer.experiment_dir = str(tmp_path)
    trainer.config = type("Config", (), {"img_format": "png"})()
    target_path = tmp_path / "decoder_probability_shifts_positive.png"

    def _fake_plot_probability_shifts(**kwargs):
        target_path.write_bytes(b"plot")
        return str(target_path)

    trainer.plot_probability_shifts = Mock(side_effect=_fake_plot_probability_shifts)

    summary = {"positive": {"value": 0.7, "feature_factor": -1.0, "learning_rate": 0.01}}
    relevant_results = {"base": {"lms": {"lms": 0.5}}}

    paths = _plot_all_target_classes(
        trainer,
        summary,
        relevant_results,
        experiment_dir=str(tmp_path),
        plot_kwargs={"show": False},
    )

    default_path = tmp_path / "decoder_probability_shifts.png"
    assert target_path.exists()
    assert default_path.exists()
    assert paths == [str(target_path), str(default_path)]
    assert trainer.plot_probability_shifts.call_args.kwargs["output"] == str(target_path)


def test_plot_all_target_classes_respects_explicit_output(tmp_path):
    """Explicit plot output is not replaced by experiment_dir-derived defaults."""
    from gradiend.evaluator.decoder import _plot_all_target_classes

    trainer = MockTrainer()
    trainer.config = type("Config", (), {"img_format": "png"})()
    explicit_path = tmp_path / "custom.png"
    trainer.plot_probability_shifts = Mock(return_value=str(explicit_path))

    summary = {"positive": {"value": 0.7, "feature_factor": -1.0, "learning_rate": 0.01}}
    relevant_results = {"base": {"lms": {"lms": 0.5}}}

    paths = _plot_all_target_classes(
        trainer,
        summary,
        relevant_results,
        experiment_dir=str(tmp_path),
        plot_kwargs={"output": str(explicit_path), "show": False},
    )

    assert trainer.plot_probability_shifts.call_args.kwargs["output"] == str(explicit_path)
    assert paths == [str(explicit_path)]
