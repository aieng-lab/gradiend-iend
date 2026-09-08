"""Decoder grid cache validation and auto-invalidation helpers."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from gradiend.evaluator.decoder import (
    DecoderPlotDataError,
    DecoderEvaluator,
    _decoder_cache_selection_context,
    _decoder_cache_selection_matches,
    _decoder_results_support_metrics,
)
from tests.test_decoder_strengthen_weaken_dataset import TrainerForSamePanelStrengthenTest


def _grid_entry(probs, probs_factual, entry_id="c1"):
    return {
        "id": entry_id,
        "lms": {"lms": 0.99},
        "probs": probs,
        "probs_factual": probs_factual,
    }


def test_decoder_results_support_metrics_detects_missing_strengthen_key():
    results = {
        "base": {"id": "base", "lms": {"lms": 0.99}, "probs": {}, "probs_factual": {}},
        "c1": _grid_entry({"neutral": 0.01}, {"IO": 0.8, "neutral": 0.99}),
    }
    ok, missing = _decoder_results_support_metrics(results, ["IO"])
    assert ok is True
    assert missing == []

    broken = {
        "base": {"id": "base", "lms": {"lms": 0.99}, "probs": {}, "probs_factual": {}},
        "c1": _grid_entry({"neutral": 0.01}, {"neutral": 0.99}),
    }
    ok, missing = _decoder_results_support_metrics(broken, ["IO"])
    assert ok is False
    assert missing == ["IO"]


def test_decoder_cache_selection_matches_requires_context_when_present():
    expected = _decoder_cache_selection_context(
        prob_on_other_class=False,
        increase_target_probabilities=True,
        metrics_for_summary=["IO"],
        classes_to_eval=["IO"],
        refine_points=10,
    )
    # A legacy cache has no refinement contract and must not satisfy the new
    # refined default. It remains reusable for an explicit coarse-grid call.
    assert _decoder_cache_selection_matches({}, expected) is False
    coarse = dict(expected, refine_points=0)
    assert _decoder_cache_selection_matches({}, coarse) is True
    assert _decoder_cache_selection_matches({"selection_context": expected}, expected) is True
    wrong = dict(expected)
    wrong["prob_on_other_class"] = True
    assert _decoder_cache_selection_matches({"selection_context": wrong}, expected) is False


def test_cache_hit_missing_only_plot_metadata_never_runs_decoder_evaluation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        evaluator = DecoderEvaluator()
        trainer = TrainerForSamePanelStrengthenTest()
        trainer.experiment_dir = str(tmp_path)
        trainer.plot_probability_shifts = Mock(return_value=str(tmp_path / "plot.png"))

        context = _decoder_cache_selection_context(
            prob_on_other_class=False,
            increase_target_probabilities=True,
            metrics_for_summary=["IO"],
            classes_to_eval=["IO"],
            refine_points=10,
        )
        common_panels = {
            "IO": {"IO": 0.85},
            "SUBJECT": {"IO": 0.15},
        }
        cache_payload = {
            "part": "decoder",
            "split": "test",
            "max_size_training_like": 50,
            "max_size_neutral": 50,
            "feature_factors": [-1.0],
            "lrs": [0.01],
            "intervention_kwargs": {
                "token_selector": "encoder_direction",
                "threshold": 0.5,
            },
            "selection_context": context,
            "results": [
                {
                    "id": "base",
                    "lms": {"lms": 0.99},
                    "probs": {"neutral": 0.01},
                    "probs_factual": {"IO": 0.75, "neutral": 0.99},
                    "probs_by_dataset": common_panels,
                },
                {
                    "id": {"feature_factor": -1.0, "learning_rate": 0.01},
                    "lms": {"lms": 0.98},
                    "probs": {"neutral": 0.02},
                    "probs_factual": {"IO": 0.85, "neutral": 0.98},
                    "probs_by_dataset": common_panels,
                },
            ],
        }
        cache_file = tmp_path / "decoder_grid_cache.json"
        cache_file.write_text(json.dumps(cache_payload), encoding="utf-8")

        result = evaluator.evaluate_decoder(
            trainer,
            target_class="IO",
            increase_target_probabilities=True,
            feature_factors=[-1.0],
            lrs=[0.01],
            plot=True,
            show=False,
            use_cache=True,
        )

        assert trainer._evaluate_base_model_calls == []
        assert result["IO"]["selection_dataset_class"] is None
        trainer.plot_probability_shifts.assert_called_once()
        migrated = json.loads(cache_file.read_text(encoding="utf-8"))
        migrated_base = next(row for row in migrated["results"] if row["id"] == "base")
        assert "_selection_dataset_by_metric" not in migrated_base

        # Even a plot-data failure after a valid cache hit must propagate as a
        # read-only plotting error, never fall through to grid recomputation.
        trainer.plot_probability_shifts = Mock(
            side_effect=DecoderPlotDataError("missing cached plot panels")
        )
        with pytest.raises(DecoderPlotDataError, match="will not trigger"):
            evaluator.evaluate_decoder(
                trainer,
                target_class="IO",
                increase_target_probabilities=True,
                feature_factors=[-1.0],
                lrs=[0.01],
                plot=True,
                show=False,
                use_cache=True,
            )
        assert trainer._evaluate_base_model_calls == []


def test_stale_decoder_cache_missing_metric_recomputes():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        evaluator = DecoderEvaluator()
        trainer = TrainerForSamePanelStrengthenTest()
        trainer.experiment_dir = str(tmp_path)

        stale_context = _decoder_cache_selection_context(
            prob_on_other_class=True,
            increase_target_probabilities=True,
            metrics_for_summary=["IO"],
            classes_to_eval=["IO"],
            refine_points=10,
        )
        cache_payload = {
            "part": "decoder",
            "split": "test",
            "max_size_training_like": 50,
            "max_size_neutral": 50,
            "feature_factors": [-1.0],
            "lrs": [0.01],
            "intervention_kwargs": {
                "token_selector": "encoder_direction",
                "threshold": 0.5,
            },
            "selection_context": stale_context,
            "results": [
                {
                    "id": "base",
                    "lms": {"lms": 0.99},
                    "probs": {"neutral": 0.01},
                    "probs_factual": {"neutral": 0.99},
                },
                {
                    "id": {"feature_factor": -1.0, "learning_rate": 0.01},
                    "lms": {"lms": 0.98},
                    "probs": {"neutral": 0.02},
                    "probs_factual": {"neutral": 0.98},
                },
            ],
        }
        cache_file = tmp_path / "decoder_grid_cache.json"
        cache_file.write_text(json.dumps(cache_payload), encoding="utf-8")

        result = evaluator.evaluate_decoder(
            trainer,
            target_class="IO",
            increase_target_probabilities=True,
            feature_factors=[-1.0],
            lrs=[0.01],
            plot=False,
            use_cache=True,
        )

        assert "IO" in result
        assert result["IO"]["value"] == 0.85
        assert len(trainer._evaluate_base_model_calls) >= 1

        refreshed = json.loads(cache_file.read_text(encoding="utf-8"))
        assert refreshed["selection_context"]["prob_on_other_class"] is False
