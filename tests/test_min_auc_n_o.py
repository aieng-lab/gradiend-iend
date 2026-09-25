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


def test_min_auc_self_orients_when_target_scores_below_neutral():
    # Same separation as test_min_auc_high_when_both_splits_separate, but the
    # target/rival poles saturate on the numerically LOWER side and neutral
    # sits higher. Which side an encoder happens to saturate on is arbitrary
    # (depends on random init / training dynamics, not something the trainer
    # constrains), so this must score identically to the "normal" polarity
    # above. Before self-orientation, plain roc_auc_score's naive "higher
    # score = positive" convention read roc_auc_neutral near 0.0 here (the
    # separation is real but anti-ranked under that convention), collapsing
    # min_auc_n_o to ~0 and making a well-separated checkpoint look
    # unconverged. See gradiend-sae's CLAUDE.md,
    # "class_exclusivity/specificity shared one Youden threshold...", for the
    # eval-time analogue that surfaced this (runs/gpt2-small/repetition's
    # actiend:YES: package-internal roc_auc_neutral read 0.0 there while the
    # study's own self-orienting computation read 0.9994 for the same class).
    df = _df(
        [(-1.0, 1.0)] * 20  # MATCH, now on the numerically low side
        + [(1.0, -1.0)] * 20  # DISTRACTOR, now on the numerically high side
        + [(0.0, 0.0)] * 20  # neutral, unchanged in the middle
    )
    parts = get_roc_auc_min_n_o(df)
    assert parts["roc_auc_neutral"] is not None and parts["roc_auc_neutral"] > 0.9
    assert parts["roc_auc_other"] is not None and parts["roc_auc_other"] > 0.9
    assert parts["min_auc_n_o"] > 0.9


def test_min_auc_rival_collapse_with_flipped_polarity_still_penalized():
    # Flipped-polarity counterpart of
    # test_min_auc_penalizes_rival_collapse_when_neutrals_separate: MATCH and
    # DISTRACTOR collapse on the numerically lower side, neutral sits higher.
    # Self-orientation must not paper over a genuine rival collapse -- only
    # fix the neutral-vs-target reading, not manufacture separation that
    # isn't there.
    df = _df(
        [(-1.0, 1.0)] * 20  # MATCH, collapsed
        + [(-1.0, -1.0)] * 20  # DISTRACTOR, identical scores to MATCH
        + [(0.2, 0.0)] * 20  # neutral, far away on the numerically higher side
    )
    parts = get_roc_auc_min_n_o(df)
    assert parts["roc_auc_neutral"] is not None and parts["roc_auc_neutral"] > 0.9
    assert parts["roc_auc_other"] is not None and abs(parts["roc_auc_other"] - 0.5) < 1e-9
    assert abs(parts["min_auc_n_o"] - 0.5) < 1e-9


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
    assert abs(float(args.convergent_score_threshold) - 0.9) < 1e-12
    assert args.convergent_mean_by_class_threshold is None


def test_metric_checkpoint_rank_treats_min_auc_like_roc_auc():
    r_low = metric_checkpoint_rank(
        step=10,
        score=0.55,
        metric="min_auc_n_o",
        mean_by_class={-1.0: -0.3, 1.0: 0.4},
    )
    r_high = metric_checkpoint_rank(
        step=20,
        score=0.82,
        metric="min_auc_n_o",
        mean_by_class={-1.0: -0.4, 1.0: 0.6},
    )
    assert r_high > r_low


def test_auc_checkpoint_rank_rejects_negative_positive_target_orientation():
    valid = metric_checkpoint_rank(
        step=10,
        score=0.75,
        metric="min_auc_n_o",
        mean_by_class={-1.0: -0.2, 1.0: 0.1},
    )
    invalid_but_higher_auc = metric_checkpoint_rank(
        step=20,
        score=0.99,
        metric="min_auc_n_o",
        mean_by_class={-1.0: 0.2, 1.0: -0.1},
    )

    assert valid > invalid_but_higher_auc


def test_correlation_checkpoint_rank_remains_sign_symmetric():
    negative_positive_target = metric_checkpoint_rank(
        step=20,
        score=0.9,
        metric="correlation",
        mean_by_class={-1.0: 0.2, 1.0: -0.1},
    )
    positive_positive_target = metric_checkpoint_rank(
        step=10,
        score=0.8,
        metric="correlation",
        mean_by_class={-1.0: -0.2, 1.0: 0.1},
    )

    assert negative_positive_target > positive_positive_target
