"""Tests for min(auc_n, auc_o) encoder selection metrics."""

import pandas as pd

from gradiend.evaluator.encoder_metrics import (
    get_encoder_metrics_from_dataframe,
    get_roc_auc_min_n_o,
)
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.stats import metric_checkpoint_rank


def _df(scores_labels):
    scores, labels = zip(*scores_labels)
    return pd.DataFrame({"encoded": list(scores), "label": list(labels), "type": ["training"] * len(scores)})


def test_min_auc_penalizes_rival_collapse_when_neutrals_separate():
    # MATCH and DISTRACTOR both score identically high; only neutrals are low →
    # pooled one-vs-rest can look good, but min(auc_n, auc_o) stays near chance.
    df = _df(
        [(1.0, 1.0)] * 20  # MATCH
        + [(1.0, -1.0)] * 20  # DISTRACTOR (same scores as MATCH)
        + [(-0.8, 0.0)] * 20  # neutral
    )
    parts = get_roc_auc_min_n_o(df)
    assert parts["roc_auc_neutral"] is not None and parts["roc_auc_neutral"] > 0.9
    assert parts["roc_auc_other"] is not None and abs(parts["roc_auc_other"] - 0.5) < 1e-9
    assert abs(parts["min_auc_n_o"] - 0.5) < 1e-9


def test_min_auc_high_when_both_splits_separate():
    df = _df(
        [(1.0, 1.0)] * 20
        + [(-1.0, -1.0)] * 20
        + [(0.0, 0.0)] * 20
    )
    parts = get_roc_auc_min_n_o(df)
    assert parts["roc_auc_neutral"] is not None and parts["roc_auc_neutral"] > 0.9
    assert parts["roc_auc_other"] is not None and parts["roc_auc_other"] > 0.9
    assert parts["min_auc_n_o"] > 0.9


def test_encoder_metrics_exposes_min_auc_fields():
    df = _df(
        [(1.0, 1.0)] * 10
        + [(-1.0, -1.0)] * 10
        + [(0.0, 0.0)] * 10
    )
    # type column needed for aggregates; neutrals usually tagged separately
    df.loc[df["label"] == 0.0, "type"] = "neutral_dataset"
    out = get_encoder_metrics_from_dataframe(df)
    assert "min_auc_n_o" in out
    assert "roc_auc_neutral" in out
    assert "roc_auc_other" in out
    assert out["min_auc_n_o"] > 0.9


def test_training_arguments_accepts_min_auc_aliases():
    args = TrainingArguments(convergent_metric="min_auc")
    assert args.convergent_metric == "min_auc_n_o"
    assert abs(float(args.convergent_score_threshold) - 0.7) < 1e-12
    assert args.convergent_mean_by_class_threshold is None


def test_metric_checkpoint_rank_treats_min_auc_like_roc_auc():
    r_low = metric_checkpoint_rank(step=10, score=0.55, metric="min_auc_n_o")
    r_high = metric_checkpoint_rank(step=20, score=0.82, metric="min_auc_n_o")
    assert r_high > r_low
