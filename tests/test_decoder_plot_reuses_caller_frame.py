"""
Regression tests for decoder plotting data provenance. Decoder evaluation
must store the complete factual-grouped panels during the grid sweep, and
plotting must consume those stored panels without a second inference pass.

This silently replaced every grid cell's ``probs_by_dataset`` -- computed
correctly during the actual sweep -- with data scored against the trainer's
own internal one-pole-scoped view (missing rival classes, wrong split),
right at the very end of ``evaluate_decoder``. Every existing decoder-eval
test in this package used ``plot=False`` and was therefore structurally
blind to this: the bug only fires during the plot-refresh pass.

Confirmed live on gender_en/pythia-70m-deduped and race_one_pole/gpt2-small
full-suite reruns: after a long, fully-correct grid sweep (visible via
per-call diagnostic logging as `dataset_class_col='label_class'`,
`class_counts={'F': N, 'M': N}`), the plot-refresh pass at the end
overwrote every cell with `dataset_class_col='factual_id'`,
`class_counts={'F': N, 'neutral': 1000}` -- 'neutral' can only come from
the trainer's own internal fallback, never from the study-supplied frame,
which is how the substitution was detected.

The old plot-refresh pass has been removed: a cache hit with ``plot=True`` is
strictly read-only, and missing plot curves raise instead of evaluating a
checkpoint implicitly.
"""
import os
from pathlib import Path

import pandas as pd
import pytest
from unittest.mock import Mock

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer


def _row(masked, label_class, label, alternative_class, alternative, split):
    return {
        "masked": masked,
        "split": split,
        "label_class": label_class,
        "label": label,
        "alternative_class": alternative_class,
        "alternative": alternative,
    }


def test_plot_true_reuses_caller_training_like_df_not_trainer_internal_view(tmp_path):
    # Trainer's OWN training data uses split="train" only -- if the
    # plot-refresh pass ever falls back to re-deriving from this via the
    # trainer's internal _get_decoder_eval_dataframe (default split="test"),
    # it would either crash ("no data for split='test'") or silently
    # substitute a different population than the caller supplied.
    train_df = pd.DataFrame([
        _row("The nurse said that [MASK] would arrive shortly.", "F", "she", "M", "he", split="train"),
        _row("The doctor said that [MASK] would arrive shortly.", "M", "he", "F", "she", split="train"),
    ])
    config = TextPredictionConfig(
        data=train_df, target_classes=["F"], all_classes=["F", "M"],
        counterfactual_classes="all",
    )
    trainer = TextPredictionTrainer(
        model="gpt2",
        config=config,
        training_args=TrainingArguments(
            add_neutral_identity_transitions=False,
            experiment_dir=str(tmp_path),
        ),
    )
    trainer._ensure_data()

    # Caller-supplied frame: a DIFFERENT split ("validation") than the
    # trainer's own data ("train") -- if the fix regresses, this manifests
    # either as a crash (no "test" split data exists) or as silently wrong
    # coverage in probs_by_dataset.
    val_df = pd.DataFrame([
        _row("Everyone in the room turned to see what [MASK] wanted.", "F", "she", "M", "he", split="validation"),
        _row("Everyone in the room turned to see what [MASK] wanted.", "M", "he", "F", "she", split="validation"),
    ])
    neutral_df = pd.DataFrame([{"text": "The report was filed on time yesterday afternoon."}])

    original_evaluate = trainer.evaluate_base_model
    trainer.evaluate_base_model = Mock(wraps=original_evaluate)
    result = trainer.evaluate_decoder(
        target_class=["F"], summary_metrics=["F"],
        feature_factors=[1.0], lrs=[1e-5, 1e-4],
        training_like_df=val_df, neutral_df=neutral_df,
        split="validation", plot=True, show=False, use_cache=False, refine_points=0,
    )
    # Base plus two requested grid cells. Plotting must not double this to 6.
    assert trainer.evaluate_base_model.call_count == 3
    _assert_full_coverage_no_substitution(result)


def test_plot_true_with_cache_hit_also_reuses_caller_frame(tmp_path):
    """
    Same bug, second call site: `evaluate_decoder`'s cache-hit branch
    (``use_cache=True`` and a matching decoder_grid_cache.json already on
    disk) ALSO unconditionally re-derived training_like_df/neutral_df from
    the trainer's own internal state for its plot-refresh step, discarding
    the caller-supplied frame just like the non-cached path did. Same fix,
    same class of test: reuse the caller's own train-only trainer data vs.
    a validation-split caller frame to detect any silent substitution.
    """
    train_df = pd.DataFrame([
        _row("The nurse said that [MASK] would arrive shortly.", "F", "she", "M", "he", split="train"),
        _row("The doctor said that [MASK] would arrive shortly.", "M", "he", "F", "she", split="train"),
    ])
    config = TextPredictionConfig(
        data=train_df, target_classes=["F"], all_classes=["F", "M"],
        counterfactual_classes="all",
    )
    trainer = TextPredictionTrainer(
        model="gpt2",
        config=config,
        training_args=TrainingArguments(
            add_neutral_identity_transitions=False,
            experiment_dir=str(tmp_path),
        ),
    )
    trainer._ensure_data()

    val_df = pd.DataFrame([
        _row("Everyone in the room turned to see what [MASK] wanted.", "F", "she", "M", "he", split="validation"),
        _row("Everyone in the room turned to see what [MASK] wanted.", "M", "he", "F", "she", split="validation"),
    ])
    neutral_df = pd.DataFrame([{"text": "The report was filed on time yesterday afternoon."}])

    eval_kw = dict(
        target_class=["F"], summary_metrics=["F"],
        feature_factors=[1.0], lrs=[1e-5, 1e-4],
        training_like_df=val_df, neutral_df=neutral_df,
        split="validation", use_cache=True, refine_points=0,
    )
    # First call (plot=False) writes decoder_grid_cache.json.
    trainer.evaluate_decoder(plot=False, **eval_kw)
    assert (tmp_path / "decoder_grid_cache.json").is_file()

    # Second call: cache now matches, AND plot=True. Any model evaluation is
    # a regression: plotting a valid grid must be read-only.
    trainer.evaluate_base_model = Mock(
        side_effect=AssertionError("cache-hit plotting ran decoder evaluation")
    )
    result = trainer.evaluate_decoder(plot=True, show=False, **eval_kw)
    trainer.evaluate_base_model.assert_not_called()
    _assert_full_coverage_no_substitution(result)


def test_analyze_decoder_for_plotting_omitted_split_uses_evaluate_decoder_default(tmp_path):
    """
    Regression test for a bug introduced (and caught) while promoting split/
    training_like_df/neutral_df from **kwargs to explicit named parameters
    across the plot call chain: evaluate_decoder's own `split` parameter
    defaults to "test", NOT None -- `split=None` is a distinct value
    downstream (e.g. the decoder cache key treats split=None as "none", a
    different bucket than "test"). Naively forwarding an omitted (i.e. None)
    `split` from analyze_decoder_for_plotting straight into
    self.evaluate_decoder(split=split, ...) would have silently overridden
    evaluate_decoder's own "test" default with an explicit None -- exactly
    the same *class* of silent-substitution bug this whole file guards
    against, just introduced by the fix itself rather than predating it.

    This trainer's own data only has a "test" split -- if the fix regresses
    (split=None reaches evaluate_decoder), this either crashes with "no data
    for split=..." or silently scores against a different, empty population.
    """
    test_df = pd.DataFrame([
        _row("The nurse said that [MASK] would arrive shortly.", "F", "she", "M", "he", split="test"),
        _row("The doctor said that [MASK] would arrive shortly.", "M", "he", "F", "she", split="test"),
    ])
    config = TextPredictionConfig(
        data=test_df, target_classes=["F"], all_classes=["F", "M"],
        counterfactual_classes="all",
    )
    trainer = TextPredictionTrainer(
        model="gpt2",
        config=config,
        training_args=TrainingArguments(
            add_neutral_identity_transitions=False,
            experiment_dir=str(tmp_path),
        ),
    )
    trainer._ensure_data()

    # No decoder_results and no split -- forces analyze_decoder_for_plotting
    # to internally call self.evaluate_decoder(...) with whatever split it
    # resolves, which must be "test" (evaluate_decoder's own default), not
    # None, or this crashes / scores an empty frame.
    result = trainer.analyze_decoder_for_plotting(
        class_ids=["F", "M"],
        use_cache=False,
        target_class=["F"], summary_metrics=["F"],
        feature_factors=[1.0], lrs=[1e-5],
        refine_points=0, plot=False,
    )
    plotting_data = result.get("plotting_data") or {}
    assert plotting_data, "analyze_decoder_for_plotting produced no plotting data"
    for key, entry in plotting_data.items():
        pbd = entry.get("probs_by_dataset") if isinstance(entry, dict) else None
        assert pbd, f"grid entry {key} has no probs_by_dataset"
        assert set(pbd.keys()) == {"F", "M"}, (
            f"grid entry {key} probs_by_dataset panels are {sorted(pbd.keys())}, "
            "expected exactly {'F', 'M'} -- split=None must not have reached "
            "evaluate_decoder instead of its own \"test\" default."
        )


def _assert_full_coverage_no_substitution(result):
    grid = result.get("grid") or {}
    assert grid, "evaluate_decoder produced no grid entries"
    for key, entry in grid.items():
        pbd = entry.get("probs_by_dataset") if isinstance(entry, dict) else None
        assert pbd, f"grid entry {key} has no cached probs_by_dataset"
        # 'neutral' must never appear here -- it can only come from the
        # trainer's own internal fallback data, never from val_df.
        assert "neutral" not in pbd, (
            f"grid entry {key} probs_by_dataset has 'neutral' "
            f"({sorted(pbd.keys())}) -- the plot-refresh pass substituted "
            "the trainer's internal data for the caller-supplied frame."
        )
        assert set(pbd.keys()) == {"F", "M"}, (
            f"grid entry {key} probs_by_dataset panels are {sorted(pbd.keys())}, "
            "expected exactly {'F', 'M'} from the caller-supplied val_df."
        )
        for group, metrics in pbd.items():
            assert set(metrics.keys()) == {"F", "M"}, (
                f"grid entry {key} panel {group!r} metrics are {sorted(metrics.keys())}"
            )
