"""
Two-pole (pair/bipolar) counterpart to test_one_pole_decoder_eval_class_coverage.py.

The real production crash that started this whole investigation
(gender_en's ``probs_by_dataset['M']['F'] is absent``) was a **two-pole**
scenario (``gradiend:F-M:F``'s opposite-polarity random control), not
one-pole -- see CLAUDE.md's "Root cause found and fixed for the recurring
probs_by_dataset[X][Y] is absent crash family". A test matrix that only
covered one-pole would have missed the exact shape that actually broke in
production: a pair-trained encoder evaluated at a single non-canonical pole
(``target_class=[opposite_class]``), which is structurally different from a
one-pole trainer's `target_classes` (one-pole: single trainer target class +
CF-expanded rivals via `counterfactual_classes`; two-pole: `target_classes`
already lists every pole the trainer itself claims, no CF expansion).

Covers: a straight 2-class pair (mirrors gender_en's F/M) and a 3-class
task's single pair encoder (mirrors RAVEL/race/religion's
`pair_single`-trained pairwise encoders, which only ever claim 2 of the
task's 3 classes -- see CLAUDE.md's "3-class pair-trained (two-pole)
encoders silently dropped one class" for the *other*, already-fixed bug in
this same neighborhood). For each, both the headline call shape (both poles
evaluated together) and the opposite-polarity random-control shape
(singleton, non-canonical class) are exercised.
"""

import pandas as pd
import pytest

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer


def _row(masked, label_class, label, alternative_class, alternative, split="train"):
    return {
        "masked": masked,
        "split": split,
        "label_class": label_class,
        "label": label,
        "alternative_class": alternative_class,
        "alternative": alternative,
    }


def _build_pair_trainer(target_classes, train_rows):
    """Real gpt2-backed two-pole (pair) trainer -- no counterfactual_classes,
    which is what distinguishes a pair trainer from a one-pole trainer
    (see TextPredictionTrainer._one_pole_positive_class)."""
    train_df = pd.DataFrame(train_rows)
    config = TextPredictionConfig(data=train_df, target_classes=list(target_classes))
    trainer = TextPredictionTrainer(
        model="gpt2",
        config=config,
        training_args=TrainingArguments(add_neutral_identity_transitions=False),
    )
    trainer._ensure_data()
    assert not trainer._is_one_pole_config(), (
        "test setup bug: this must be a genuine two-pole/pair trainer, not one-pole"
    )
    return trainer


def _assert_full_coverage(probs_by_dataset, expected_classes):
    expected = set(expected_classes)
    missing_panels = expected - set(probs_by_dataset.keys())
    assert not missing_panels, (
        f"Missing probs_by_dataset panels for classes {sorted(missing_panels)}: "
        f"panels present={sorted(probs_by_dataset.keys())}. This is exactly the "
        f"production failure mode confirmed on gender_en's two-pole "
        f"opposite-polarity random control."
    )
    for group, metrics in probs_by_dataset.items():
        missing_metrics = expected - set(metrics.keys())
        assert not missing_metrics, (
            f"probs_by_dataset[{group!r}] is missing metrics {sorted(missing_metrics)} "
            f"(has {sorted(metrics.keys())})"
        )


class TestTwoClassPairDecoderCoverage:
    """2-class pair (both poles trained together) -- mirrors gender_en's F/M gradiend:F-M encoder."""

    CLASSES = ["F", "M"]

    @pytest.fixture(scope="class")
    def trainer(self):
        return _build_pair_trainer(
            target_classes=self.CLASSES,
            train_rows=[
                _row("The nurse said that [MASK] would arrive shortly.", "F", "she", "M", "he"),
                _row("The doctor said that [MASK] would arrive shortly.", "M", "he", "F", "she"),
            ],
        )

    @pytest.fixture(scope="class")
    def eval_frames(self):
        val_df = pd.DataFrame([
            _row("Everyone in the room turned to see what [MASK] wanted.", "F", "she", "M", "he", split="validation"),
            _row("Everyone in the room turned to see what [MASK] wanted.", "M", "he", "F", "she", split="validation"),
        ])
        neutral_df = pd.DataFrame([{"text": "The report was filed on time yesterday afternoon."}])
        return val_df, neutral_df

    def test_headline_both_poles_together_has_full_coverage(self, trainer, eval_frames):
        """Headline call shape: evaluate_decoder_for_classes passes the trainer's full claim list."""
        val_df, neutral_df = eval_frames
        result = trainer.evaluate_decoder(
            target_class=self.CLASSES, summary_metrics=self.CLASSES,
            feature_factors=[1.0], lrs=[1e-5],
            training_like_df=val_df, neutral_df=neutral_df,
            split="validation", plot=False, use_cache=False,
        )
        for key, entry in (result.get("grid") or {}).items():
            pbd = entry.get("probs_by_dataset") if isinstance(entry, dict) else None
            assert pbd, f"grid entry {key} has no probs_by_dataset at all"
            _assert_full_coverage(pbd, self.CLASSES)

    def test_opposite_polarity_random_control_has_full_coverage(self, trainer, eval_frames):
        """
        Exact shape of the production crash: causal_study.py's opposite-polarity
        random control calls evaluate_decoder_for_classes_refined(trainer,
        [opposite_class], feature_factors=[opp_ff], ...) -- a *singleton*
        target_class, evaluating the pole that is NOT the headline's selected
        class. This is what crashed with "probs_by_dataset['M']['F'] is absent".
        """
        val_df, neutral_df = eval_frames
        result = trainer.evaluate_decoder(
            target_class=["M"], summary_metrics=["M"],
            feature_factors=[-1.0], lrs=[1e-5],
            training_like_df=val_df, neutral_df=neutral_df,
            split="validation", plot=False, use_cache=False,
        )
        for key, entry in (result.get("grid") or {}).items():
            pbd = entry.get("probs_by_dataset") if isinstance(entry, dict) else None
            assert pbd, f"grid entry {key} has no probs_by_dataset at all"
            _assert_full_coverage(pbd, self.CLASSES)


class TestThreeClassPairSingleDecoderCoverage:
    """
    A single pair encoder trained on 2 of a 3-class task's classes (RAVEL/
    race/religion's `pair_single` ablation: e.g. a china-vs-russia encoder
    within a china/russia/united_states task). The encoder's own claim is
    only {china, russia} -- united_states never appears in this trainer's
    own data at all, matching how the study actually constructs per-pair
    trainers for pair_single tasks.
    """

    CLASSES = ["china", "russia"]

    @pytest.fixture(scope="class")
    def trainer(self):
        return _build_pair_trainer(
            target_classes=self.CLASSES,
            train_rows=[
                _row(
                    "The delegation from [MASK] arrived at the summit early.",
                    "china", "Beijing", "russia", "Moscow",
                ),
                _row(
                    "The delegation from [MASK] arrived at the summit early.",
                    "russia", "Moscow", "china", "Beijing",
                ),
            ],
        )

    @pytest.fixture(scope="class")
    def eval_frames(self):
        val_df = pd.DataFrame([
            _row(
                "Officials confirmed the ambassador from [MASK] would attend.",
                "china", "Beijing", "russia", "Moscow", split="validation",
            ),
            _row(
                "Officials confirmed the ambassador from [MASK] would attend.",
                "russia", "Moscow", "china", "Beijing", split="validation",
            ),
        ])
        neutral_df = pd.DataFrame([{"text": "The meeting concluded without any further comment."}])
        return val_df, neutral_df

    @pytest.mark.parametrize("evaluated_class,ff", [("china", 1.0), ("russia", -1.0)])
    def test_each_pole_has_full_coverage_when_evaluated_alone(
        self, trainer, eval_frames, evaluated_class, ff
    ):
        val_df, neutral_df = eval_frames
        result = trainer.evaluate_decoder(
            target_class=[evaluated_class], summary_metrics=[evaluated_class],
            feature_factors=[ff], lrs=[1e-5],
            training_like_df=val_df, neutral_df=neutral_df,
            split="validation", plot=False, use_cache=False,
        )
        for key, entry in (result.get("grid") or {}).items():
            pbd = entry.get("probs_by_dataset") if isinstance(entry, dict) else None
            assert pbd, f"grid entry {key} has no probs_by_dataset at all"
            _assert_full_coverage(pbd, self.CLASSES)
