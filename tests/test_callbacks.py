"""
Tests for training callbacks (modality-independent).

Tests EarlyStoppingCallback, EvaluationCallback, CheckpointCallback,
LoggingCallback, and NormalizationCallback.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from gradiend.trainer.core.callbacks import (
    TrainingCallback,
    EvaluationCallback,
    NormalizationCallback,
    CheckpointCallback,
    LoggingCallback,
)


class TestTrainingCallback:
    """Test base TrainingCallback class."""
    
    def test_callback_lifecycle(self):
        """Test that callback lifecycle methods can be called."""
        callback = TrainingCallback()
        
        # All methods should be callable without errors
        callback.on_train_begin({})
        callback.on_epoch_begin(0, {})
        callback.on_step_begin(0, {})
        callback.on_step_end(0, 0.5, None, {})
        callback.on_epoch_end(0, None, {})
        callback.on_train_end({})


class TestEvaluationCallback:
    """Test EvaluationCallback."""
    
    def test_evaluation_callback_creation(self):
        """Test EvaluationCallback can be created."""
        evaluate_fn = MagicMock(return_value={"correlation": 0.8})
        callback = EvaluationCallback(evaluate_fn=evaluate_fn, n_evaluation=100, do_eval=True)
        
        assert callback.evaluate_fn == evaluate_fn
        assert callback.n_evaluation == 100
        assert callback.do_eval is True
        assert callback.last_eval_step == -1
    
    def test_evaluation_callback_skips_when_disabled(self):
        """Test that evaluation is skipped when do_eval=False."""
        evaluate_fn = MagicMock()
        callback = EvaluationCallback(evaluate_fn=evaluate_fn, n_evaluation=100, do_eval=False)
        
        result = callback.on_step_end(step=100, loss=0.5, model=None, config={}, training_stats={})
        
        assert result is None
        evaluate_fn.assert_not_called()
    
    def test_evaluation_callback_evaluates_at_intervals(self):
        """Test that evaluation happens at specified intervals."""
        eval_result = {"correlation": 0.8, "mean_by_class": 0.7}
        evaluate_fn = MagicMock(return_value=eval_result)
        callback = EvaluationCallback(evaluate_fn=evaluate_fn, n_evaluation=50, do_eval=True)
        
        model = MagicMock()
        model.base_model.training = True
        
        # Should evaluate at step 0
        result = callback.on_step_end(step=0, loss=0.5, model=model, config={}, training_stats={})
        assert result == eval_result
        assert evaluate_fn.call_count == 1
        
        # Should not evaluate at step 25
        result = callback.on_step_end(step=25, loss=0.5, model=model, config={}, training_stats={})
        assert result is None
        assert evaluate_fn.call_count == 1
        
        # Should evaluate at step 50
        result = callback.on_step_end(step=50, loss=0.5, model=model, config={}, training_stats={})
        assert result == eval_result
        assert evaluate_fn.call_count == 2
    
    def test_evaluation_callback_evaluates_at_last_iteration(self):
        """Test that evaluation happens at last iteration."""
        eval_result = {"correlation": 0.8}
        evaluate_fn = MagicMock(return_value=eval_result)
        callback = EvaluationCallback(evaluate_fn=evaluate_fn, n_evaluation=100, do_eval=True)
        
        model = MagicMock()
        model.base_model.training = True
        
        # Should evaluate at last iteration even if not at interval
        result = callback.on_step_end(
            step=75, loss=0.5, model=model, config={}, training_stats={},
            last_iteration=True
        )
        assert result == eval_result
        evaluate_fn.assert_called_once()

    def test_evaluation_callback_does_not_mark_step_on_failed_eval(self):
        """Failed evaluations must not mark the step as done (allows retry)."""
        evaluate_fn = MagicMock(return_value=None)
        callback = EvaluationCallback(evaluate_fn=evaluate_fn, n_evaluation=100, do_eval=True)

        model = MagicMock()
        model.base_model.training = True

        result = callback.on_step_end(
            step=75, loss=0.5, model=model, config={}, training_stats={},
            last_iteration=True,
        )

        assert result is None
        assert callback.last_eval_step == -1
        evaluate_fn.assert_called_once()

    def test_evaluation_callback_evaluates_on_last_iteration_when_n_evaluation_zero(self):
        """last_iteration should still trigger eval when periodic interval is disabled."""
        eval_result = {"correlation": 0.8}
        evaluate_fn = MagicMock(return_value=eval_result)
        callback = EvaluationCallback(evaluate_fn=evaluate_fn, n_evaluation=0, do_eval=True)

        model = MagicMock()
        model.base_model.training = True

        result = callback.on_step_end(
            step=12, loss=0.5, model=model, config={}, training_stats={},
            last_iteration=True,
        )

        assert result == eval_result
        assert callback.last_eval_step == 12
        evaluate_fn.assert_called_once()


class TestNormalizationCallback:
    """Test NormalizationCallback."""
    
    def test_normalization_callback_creation(self):
        """Test NormalizationCallback can be created."""
        callback = NormalizationCallback()
        assert callback is not None
    
    def test_normalization_callback_inverts_on_negative_correlation(self):
        """Test that normalization inverts encoding when correlation < -0.6."""
        callback = NormalizationCallback()

        model = MagicMock()
        model.gradiend.latent_dim = 1
        model.invert_encoding = MagicMock()

        eval_result = {
            "correlation": -0.7,
            "mean_by_class": {1.0: 0.5, -1.0: -0.5},
            "mean_by_type": {"training": 0.1, "neutral_dataset": 0.2},
            "neutral_mean_by_type": {"neutral_dataset": 0.2},
            "abs_mean_by_type": {"training": 0.1, "neutral_dataset": 0.2},
            "min_by_class": {1.0: 0.4, -1.0: -0.6},
            "max_by_class": {1.0: 0.6, -1.0: -0.4},
            "q1_by_class": {1.0: 0.45, -1.0: -0.55},
            "q3_by_class": {1.0: 0.55, -1.0: -0.45},
        }
        training_stats = {
            "scores": {100: -0.7},
            "mean_by_class": {100: eval_result["mean_by_class"]},
            "mean_by_type": {100: eval_result["mean_by_type"]},
            "neutral_mean_by_type": {100: eval_result["neutral_mean_by_type"]},
            "abs_mean_by_type": {100: eval_result["abs_mean_by_type"]},
            "min_by_class": {100: eval_result["min_by_class"]},
            "max_by_class": {100: eval_result["max_by_class"]},
            "q1_by_class": {100: eval_result["q1_by_class"]},
            "q3_by_class": {100: eval_result["q3_by_class"]},
        }

        callback.on_step_end(
            step=100, loss=0.5, model=model, config={}, training_stats=training_stats,
            eval_result=eval_result,
        )

        model.invert_encoding.assert_called_once_with(update_direction=False)
        assert eval_result["correlation"] == pytest.approx(0.7)
        assert training_stats["correlation"] == pytest.approx(0.7)
        assert training_stats["scores"][100] == pytest.approx(0.7)
        assert eval_result["mean_by_class"][1.0] == pytest.approx(-0.5)
        assert eval_result["mean_by_type"]["training"] == pytest.approx(-0.1)
        assert eval_result["neutral_mean_by_type"]["neutral_dataset"] == pytest.approx(-0.2)
        assert training_stats["neutral_mean_by_type"][100]["neutral_dataset"] == pytest.approx(-0.2)
        assert eval_result["abs_mean_by_type"]["neutral_dataset"] == pytest.approx(0.2)
        assert eval_result["min_by_class"][1.0] == pytest.approx(-0.6)
        assert eval_result["max_by_class"][1.0] == pytest.approx(-0.4)
        assert training_stats["min_by_class"][100][1.0] == pytest.approx(-0.6)
        assert eval_result["q1_by_class"][1.0] == pytest.approx(-0.55)
        assert eval_result["q3_by_class"][1.0] == pytest.approx(-0.45)

    def test_normalization_callback_no_inversion_on_positive_correlation(self):
        """Test that normalization doesn't invert when correlation >= -0.6."""
        callback = NormalizationCallback()

        model = MagicMock()
        model.gradiend.latent_dim = 1
        model.invert_encoding = MagicMock()

        eval_result = {"correlation": 0.5, "mean_by_class": {1.0: 0.5, -1.0: -0.5}}

        callback.on_step_end(
            step=100, loss=0.5, model=model, config={}, training_stats={},
            eval_result=eval_result,
        )

        model.invert_encoding.assert_not_called()
        assert eval_result["correlation"] == pytest.approx(0.5)

    def test_normalization_callback_no_inversion_when_correlation_mildly_negative(self):
        """Correlation below zero but above the inversion threshold must not flip."""
        callback = NormalizationCallback()

        model = MagicMock()
        model.gradiend.latent_dim = 1
        model.invert_encoding = MagicMock()

        eval_result = {"correlation": -0.55, "mean_by_class": {1.0: 0.5, -1.0: -0.5}}

        callback.on_step_end(
            step=100, loss=0.5, model=model, config={}, training_stats={},
            eval_result=eval_result,
        )

        model.invert_encoding.assert_not_called()
        assert eval_result["correlation"] == pytest.approx(-0.55)

    def test_auc_normalization_orients_label_plus_one_mean_positive(self):
        callback = NormalizationCallback()
        model = MagicMock()
        model.gradiend.latent_dim = 1
        model.invert_encoding = MagicMock()
        eval_result = {
            "correlation": 0.7,
            "mean_by_class": {-1.0: -0.8, 0.0: 0.2, 1.0: -0.4},
        }
        training_stats = {
            "scores": {100: 0.7},
            "mean_by_class": {100: dict(eval_result["mean_by_class"])},
        }

        callback.on_step_end(
            step=100,
            loss=0.5,
            model=model,
            config={"convergent_metric": "min_auc_n_o"},
            training_stats=training_stats,
            eval_result=eval_result,
        )

        model.invert_encoding.assert_called_once_with(update_direction=False)
        assert eval_result["mean_by_class"][1.0] == pytest.approx(0.4)
        assert training_stats["mean_by_class"][100][1.0] == pytest.approx(0.4)

    def test_auc_normalization_does_not_use_negative_correlation_for_orientation(self):
        callback = NormalizationCallback()
        model = MagicMock()
        model.gradiend.latent_dim = 1
        model.invert_encoding = MagicMock()
        eval_result = {
            "correlation": -0.9,
            "mean_by_class": {-1.0: -0.4, 1.0: 0.6},
        }

        callback.on_step_end(
            step=100,
            loss=0.5,
            model=model,
            config={"convergent_metric": "roc_auc"},
            training_stats={},
            eval_result=eval_result,
        )

        model.invert_encoding.assert_not_called()
        assert eval_result["mean_by_class"][1.0] == pytest.approx(0.6)


class TestCheckpointCallback:
    """Test CheckpointCallback."""
    
    def test_checkpoint_callback_creation(self, temp_dir):
        """Test CheckpointCallback can be created."""
        callback = CheckpointCallback(
            output=temp_dir,
            checkpoints=False,
            keep_only_best=True,
            checkpoint_interval=100,
            use_loss_for_best=False
        )
        
        assert callback.output == temp_dir
        assert callback.keep_only_best is True
        assert callback.checkpoint_step == 0  # checkpoints=False means no periodic checkpoints
        assert callback.use_loss_for_best is False
    
    def test_checkpoint_callback_saves_best_model(self, temp_dir):
        """Test that checkpoint saves best model based on correlation."""
        callback = CheckpointCallback(
            output=temp_dir,
            checkpoints=False,
            keep_only_best=True,
            use_loss_for_best=False
        )
        
        model = MagicMock()
        model.save_pretrained = MagicMock()
        
        config = {}
        training_stats = {}
        
        # First step with correlation 0.5
        training_stats['correlation'] = 0.5
        callback.on_step_end(
            step=50, loss=0.5, model=model, config=config, training_stats=training_stats,
            eval_result={"correlation": 0.5},
        )
        
        # Should save as best (first one, but only if step > 1)
        # Step 50 > 1, so should save
        assert model.save_pretrained.call_count >= 1
        assert callback.best_score == 0.5
        
        # Second step with better correlation
        training_stats['correlation'] = 0.8
        callback.on_step_end(
            step=100, loss=0.5, model=model, config=config, training_stats=training_stats,
            eval_result={"correlation": 0.8},
        )
        
        # Should save again (better metric)
        assert model.save_pretrained.call_count >= 2
        assert callback.best_score == 0.8
        
        save_count_after_best = model.save_pretrained.call_count

        # Third step with worse correlation
        training_stats['correlation'] = 0.6
        callback.on_step_end(
            step=150, loss=0.5, model=model, config=config, training_stats=training_stats,
            eval_result={"correlation": 0.6},
        )

        assert callback.best_score == 0.8
        assert model.save_pretrained.call_count == save_count_after_best

    def test_checkpoint_callback_prefers_convergent_means_over_peak_correlation(self, temp_dir):
        """With prefer_convergent_checkpoint=True, keep a convergent step over peak |corr|."""
        callback = CheckpointCallback(
            output=temp_dir,
            checkpoints=False,
            keep_only_best=True,
            use_loss_for_best=False,
        )
        model = MagicMock()
        model.save_pretrained = MagicMock()
        config = {
            "convergent_score_threshold": 0.5,
            "convergent_mean_by_class_threshold": 0.5,
            "prefer_convergent_checkpoint": True,
        }

        callback.on_step_end(
            step=100,
            loss=1e-4,
            model=model,
            config=config,
            training_stats={
                "correlation": 0.908,
                "mean_by_class": {100: {1.0: 0.80, -1.0: -0.2937}},
            },
            eval_result={
                "correlation": 0.908,
                "mean_by_class": {1.0: 0.80, -1.0: -0.2937},
            },
        )
        assert callback.best_score == pytest.approx(0.908)
        assert callback.best_step == 100

        callback.on_step_end(
            step=500,
            loss=1e-4,
            model=model,
            config=config,
            training_stats={
                "correlation": 0.858,
                "mean_by_class": {
                    100: {1.0: 0.80, -1.0: -0.2937},
                    500: {1.0: 0.6425, -1.0: -0.9010},
                },
            },
            eval_result={
                "correlation": 0.8584,
                "mean_by_class": {1.0: 0.6425, -1.0: -0.9010},
            },
        )
        assert callback.best_step == 500
        assert callback.best_score == pytest.approx(0.8584)

    def test_checkpoint_callback_default_keeps_peak_correlation(self, temp_dir):
        """Default prefer_convergent_checkpoint=False keeps max |correlation|."""
        callback = CheckpointCallback(
            output=temp_dir,
            checkpoints=False,
            keep_only_best=True,
            use_loss_for_best=False,
        )
        model = MagicMock()
        model.save_pretrained = MagicMock()
        config = {
            "convergent_score_threshold": 0.5,
            "convergent_mean_by_class_threshold": 0.5,
        }

        callback.on_step_end(
            step=100,
            loss=1e-4,
            model=model,
            config=config,
            training_stats={},
            eval_result={
                "correlation": 0.908,
                "mean_by_class": {1.0: 0.80, -1.0: -0.2937},
            },
        )
        callback.on_step_end(
            step=500,
            loss=1e-4,
            model=model,
            config=config,
            training_stats={},
            eval_result={
                "correlation": 0.8584,
                "mean_by_class": {1.0: 0.6425, -1.0: -0.9010},
            },
        )
        assert callback.best_step == 100
        assert callback.best_score == pytest.approx(0.908)

    def test_checkpoint_callback_component_split_uses_merged_local_bests(self, temp_dir):
        """Component-split best checkpoint is the running local-best merge, not one global step."""
        from gradiend.model import GradiendModel
        from gradiend.trainer.core.component_seed import (
            load_component_best_states,
            finalize_component_best_states,
        )

        class _Tiny:
            def __init__(self, gradiend):
                self.gradiend = gradiend

            def save_pretrained(self, path, **kwargs):
                os.makedirs(path, exist_ok=True)
                # Mimic enough of a checkpoint for the test.
                with open(os.path.join(path, "marker.txt"), "w", encoding="utf-8") as handle:
                    handle.write(kwargs.get("training", {}).get("best_score_checkpoint", {}).get("selection", ""))

        gradiend = GradiendModel(
            input_dim=4,
            latent_dim=1,
            bias_encoder=False,
            bias_decoder=True,
            activation_decoder="id",
            device=torch.device("cpu"),
            component_slices=[
                {"id": "left", "start": 0, "end": 2},
                {"id": "right", "start": 2, "end": 4},
            ],
            component_split_mode="tensors",
        )
        model = _Tiny(gradiend)
        callback = CheckpointCallback(
            output=temp_dir,
            checkpoints=False,
            keep_only_best=True,
            use_loss_for_best=False,
        )

        def eval_at(step, left_corr, right_corr, left_fill, right_fill):
            with torch.no_grad():
                gradiend.encoder[0].linear.weight[:, :2].fill_(left_fill)
                gradiend.encoder[0].linear.weight[:, 2:].fill_(right_fill)
                gradiend.decoder[0].linear.weight[:2, :].fill_(left_fill)
                gradiend.decoder[0].linear.weight[2:, :].fill_(right_fill)
            return {
                "correlation": 0.1,  # deliberately weak global score
                "components": {
                    "metrics_by_component": {
                        "left": {"component_index": 0, "correlation": left_corr},
                        "right": {"component_index": 1, "correlation": right_corr},
                    },
                    "convergence_by_component": {
                        "left": {
                            "converged": left_corr >= 0.5,
                            "min_target_class_abs_mean": left_corr,
                        },
                        "right": {
                            "converged": right_corr >= 0.5,
                            "min_target_class_abs_mean": right_corr,
                        },
                    },
                    "summary": {
                        "n_components": 2,
                        "n_converged": int(left_corr >= 0.5) + int(right_corr >= 0.5),
                        "correlation_mean": 0.5 * (left_corr + right_corr),
                    },
                },
            }

        # Step 10: left is strong, right weak. Global corr is tiny.
        callback.on_step_end(
            step=10,
            loss=1.0,
            model=model,
            config={},
            training_stats={},
            eval_result=eval_at(10, 0.9, 0.1, 3.0, 0.0),
        )
        assert callback.best_score == pytest.approx(0.5)
        assert os.path.isdir(f"{temp_dir}_best")
        with open(os.path.join(f"{temp_dir}_best", "marker.txt"), encoding="utf-8") as handle:
            assert handle.read() == "component_best_merge"

        # Step 20: right improves; merge should update and keep left's earlier fill via stitch.
        callback.on_step_end(
            step=20,
            loss=1.0,
            model=model,
            config={},
            training_stats={},
            eval_result=eval_at(20, 0.2, 0.8, 0.0, 5.0),
        )
        assert callback.best_score == pytest.approx(0.85)
        assert callback._best_merge_summary["n_converged"] == 2

        finalize_component_best_states(temp_dir)
        payload = load_component_best_states(temp_dir)
        assert payload["components"]["left"]["global_step"] == 10
        assert payload["components"]["right"]["global_step"] == 20

    def test_checkpoint_callback_tracks_but_does_not_save_step_zero_best_model(self, temp_dir):
        """Initial step-0 best metrics must not become the selected checkpoint."""
        callback = CheckpointCallback(
            output=temp_dir,
            checkpoints=False,
            keep_only_best=True,
            use_loss_for_best=False
        )

        model = MagicMock()
        model.save_pretrained = MagicMock()

        training_stats = {"correlation": 0.9}

        callback.on_step_end(
            step=0, loss=0.0, model=model, config={}, training_stats=training_stats,
            eval_result={"correlation": 0.9},
        )

        model.save_pretrained.assert_not_called()
        assert callback.best_step == 0
        assert callback.best_score == 0.9

    def test_checkpoint_callback_ignores_unevaluated_sentinel_correlation(self, temp_dir):
        """Global sentinel/default correlation must not become a best checkpoint score."""
        callback = CheckpointCallback(
            output=temp_dir,
            checkpoints=False,
            keep_only_best=True,
            use_loss_for_best=False
        )

        model = MagicMock()
        model.save_pretrained = MagicMock()

        callback.on_step_end(
            step=50,
            loss=0.5,
            model=model,
            config={},
            training_stats={"correlation": -1.0, "scores": {}},
            eval_result=None,
        )

        model.save_pretrained.assert_not_called()
        assert callback.best_step is None
        assert callback.best_score is None
    
    def test_checkpoint_callback_saves_periodic_checkpoints(self, temp_dir):
        """Test that checkpoint saves periodic checkpoints when checkpoints=True."""
        callback = CheckpointCallback(
            output=temp_dir,
            checkpoints=True,  # Enable periodic checkpoints
            checkpoint_interval=50,
            keep_only_best=False,
            use_loss_for_best=False
        )
        
        model = MagicMock()
        model.save_pretrained = MagicMock()
        
        config = {}
        training_stats = {'correlation': 0.5}
        
        # Should save at every checkpoint interval
        for step in [50, 100, 150]:
            callback.on_step_end(
                step=step, loss=0.5, model=model, config=config, training_stats=training_stats
            )
        
        # Should save periodic checkpoints (3 times) plus best model saves
        assert model.save_pretrained.call_count >= 3
    
    def test_checkpoint_callback_saves_at_epoch_end(self, temp_dir):
        """Test that checkpoint saves at epoch end."""
        callback = CheckpointCallback(
            output=temp_dir,
            checkpoints=False,
            keep_only_best=False,
            use_loss_for_best=False
        )
        
        model = MagicMock()
        model.save_pretrained = MagicMock()
        
        config = {}
        training_stats = {}
        losses = [0.5, 0.4, 0.3]
        
        # Should save at epoch end (callback calls model.save_pretrained, not model.gradiend)
        callback.on_epoch_end(epoch=0, model=model, config=config, training_stats=training_stats, losses=losses)
        
        assert model.save_pretrained.call_count == 1


class TestLoggingCallback:
    """Test LoggingCallback."""

    def test_logging_callback_creation(self):
        """Test LoggingCallback can be created."""
        callback = LoggingCallback(n_loss_report=100, loss_only=False)

        assert callback.n_loss_report == 100
        assert callback.loss_only is False

    def test_logging_callback_logs_loss(self):
        """Test that logging callback emits mean loss over the report window."""
        callback = LoggingCallback(n_loss_report=50, loss_only=True)
        training_stats = {}
        with patch("gradiend.trainer.core.callbacks.logger") as mock_logger:
            callback.on_step_end(
                step=50, loss=0.5, model=None, config={"do_eval": False}, training_stats=training_stats,
                last_losses=[0.5, 0.4, 0.3],
            )

        mock_logger.info.assert_called_once()
        message = mock_logger.info.call_args[0][0]
        assert message.startswith("Step 50,")
        # Mean of [0.5, 0.4, 0.3] — not the instantaneous last-step loss.
        assert "Loss: 0.4000" in message
        assert "Correlation:" not in message

    def test_logging_callback_averages_over_report_window_not_full_buffer(self):
        """When last_losses is longer than n_loss_report, only the recent window is averaged."""
        callback = LoggingCallback(n_loss_report=2, loss_only=True)

        with patch("gradiend.trainer.core.callbacks.logger") as mock_logger:
            callback.on_step_end(
                step=2,
                loss=1.0,
                model=None,
                config={"do_eval": False},
                training_stats={},
                last_losses=[9.0, 9.0, 1.0, 3.0],
            )

        message = mock_logger.info.call_args[0][0]
        assert "Loss: 2.0000" in message

    def test_logging_callback_uses_scientific_notation_for_tiny_nonzero_loss(self):
        """Tiny nonzero losses should not be hidden as 0.0000."""
        callback = LoggingCallback(n_loss_report=50, loss_only=True)

        with patch("gradiend.trainer.core.callbacks.logger") as mock_logger:
            callback.on_step_end(
                step=50,
                loss=3.2e-8,
                model=None,
                config={"do_eval": False},
                training_stats={},
                last_losses=[3.2e-8],
            )

        mock_logger.info.assert_called_once()
        message = mock_logger.info.call_args[0][0]
        assert "Loss: 3.200e-08" in message
        assert "Loss: 0.0000" not in message

    def test_logging_callback_reports_initial_eval_loss_as_none(self):
        """Initial evaluation has no optimizer loss yet."""
        callback = LoggingCallback(n_loss_report=50, loss_only=False)
        training_stats = {"correlation": 0.8}

        with patch("gradiend.trainer.core.callbacks.logger") as mock_logger:
            callback.on_step_end(
                step=0,
                loss=None,
                model=None,
                config={"do_eval": True},
                training_stats=training_stats,
                eval_result={"correlation": 0.8},
            )

        mock_logger.info.assert_called_once()
        message = mock_logger.info.call_args[0][0]
        assert "Step 0," in message
        assert "Loss: None" in message
        assert "Loss: 0.0000" not in message

    def test_logging_callback_logs_metrics(self):
        """Test that logging callback records correlation and marks new best runs."""
        callback = LoggingCallback(n_loss_report=50, loss_only=False)
        training_stats = {"correlation": 0.8}

        with patch("gradiend.trainer.core.callbacks.logger") as mock_logger:
            callback.on_step_end(
                step=50, loss=0.5, model=None, config={"do_eval": True}, training_stats=training_stats,
                eval_result={"correlation": 0.8},
                last_losses=[0.5, 0.4, 0.3],
            )

        mock_logger.info.assert_called_once()
        message = mock_logger.info.call_args[0][0]
        assert "Step 50," in message
        assert "Correlation: 0.8000" in message
        assert "(new best)" in message

    def test_logging_callback_reports_neutral_mean_when_available(self):
        """Neutral identity/eval rows should be visible in progress logs."""
        callback = LoggingCallback(n_loss_report=50, loss_only=False)
        training_stats = {
            "correlation": 0.8,
            "mean_by_class": {50: {-1.0: -0.5, 1.0: 0.5}},
            "neutral_mean_by_type": {50: {"neutral_dataset": 0.03}},
            "abs_mean_by_type": {50: {"neutral_dataset": 0.12}},
        }

        with patch("gradiend.trainer.core.callbacks.logger") as mock_logger:
            callback.on_step_end(
                step=50,
                loss=0.5,
                model=None,
                config={"do_eval": True},
                training_stats=training_stats,
                eval_result={
                    "correlation": 0.8,
                    "neutral_mean_by_type": {"neutral_dataset": 0.03},
                    "abs_mean_by_type": {"neutral_dataset": 0.12},
                },
                last_losses=[0.5],
            )

        message = mock_logger.info.call_args[0][0]
        assert "neutral: neutral_dataset: 0.0300 abs=0.1200" in message

    def test_logging_callback_does_not_report_sentinel_correlation_without_eval(self):
        """Unevaluated steps must not display the initialized -1.0 as measured correlation."""
        callback = LoggingCallback(n_loss_report=50, loss_only=False)
        training_stats = {"correlation": -1.0, "scores": {}}

        with patch("gradiend.trainer.core.callbacks.logger") as mock_logger:
            callback.on_step_end(
                step=50, loss=0.5, model=None, config={"do_eval": True}, training_stats=training_stats,
                eval_result=None,
                last_losses=[0.5, 0.4, 0.3],
            )

        mock_logger.info.assert_called_once()
        message = mock_logger.info.call_args[0][0]
        assert "Step 50," in message
        assert "Correlation: N/A" in message
        assert "-1.0000" not in message
        assert "(new best)" not in message

    def test_logging_callback_ignores_eval_result_when_eval_is_disabled(self):
        """do_eval=False means progress logging must not expose evaluation metrics."""
        callback = LoggingCallback(n_loss_report=50, loss_only=False)
        training_stats = {"correlation": 0.8, "scores": {50: 0.8}}

        with patch("gradiend.trainer.core.callbacks.logger") as mock_logger:
            callback.on_step_end(
                step=50, loss=0.5, model=None, config={"do_eval": False}, training_stats=training_stats,
                eval_result={"correlation": 0.8},
                last_losses=[0.5, 0.4, 0.3],
            )

        mock_logger.info.assert_called_once()
        message = mock_logger.info.call_args[0][0]
        assert "Step 50," in message
        assert "Loss: 0.4000" in message
        assert "Correlation:" not in message
        assert "(new best)" not in message


# Note: EarlyStoppingCallback is not yet implemented in the codebase
# Tests will be added when it is implemented
