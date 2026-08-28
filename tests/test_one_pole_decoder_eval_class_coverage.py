"""
End-to-end coverage of one-pole decoder evaluation across different class
counts (2-class, 3-class, 4-class), using a real gpt2 model rather than a
mocked trainer.

Motivation: a confirmed production bug (found while chasing live crashes on
gender_en/pronoun_number/religion_one_pole full-suite reruns) silently
dropped rows for whichever class a specific evaluate_decoder(...) call was
evaluating (or, depending on direction, the *other* classes) out of
training_like_df, before scoring ever happened -- caused by
decoder.py's ``required_datasets`` dataset-narrowing "efficiency" filter
excluding the evaluated class's own panel unconditionally. This surfaced as
a downstream ``KeyError``/``ValueError: ... probs_by_dataset[X][Y] is
absent`` in study code, for THREE different tasks with different class
counts (2-class gender_en, 2-class pronoun_number via class_merge_map,
3-class religion_one_pole) -- but every purely mock-based decoder test in
this package was blind to it, because they hand-construct their own
``evaluate_base_model`` stand-in rather than exercising the real
``evaluate_base_model`` -> ``score_probability_shift`` ->
``compute_probability_shift_score_clm`` pipeline that the bug actually lived
in.

These tests exercise that real pipeline end-to-end (real gpt2 weights, real
tokenization, real one-pole trainer construction) across a small matrix of
class counts, and assert every (group, metric) combination a caller could
legitimately need is present in ``probs_by_dataset`` -- for both the
canonical target class AND a non-canonical "opposite polarity" class (the
shape of the random-control call that originally crashed in production).
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


def _build_one_pole_trainer(target_class, all_classes, train_rows):
    """Real gpt2-backed one-pole trainer, mirroring how the study constructs one."""
    train_df = pd.DataFrame(train_rows)
    config = TextPredictionConfig(
        data=train_df,
        target_classes=[target_class],
        all_classes=list(all_classes),
        counterfactual_classes="all",
    )
    trainer = TextPredictionTrainer(
        model="gpt2",
        config=config,
        training_args=TrainingArguments(add_neutral_identity_transitions=False),
    )
    trainer._ensure_data()
    return trainer


def _assert_full_coverage(probs_by_dataset, expected_classes):
    """Every class must have its own panel, and every panel must score every class."""
    expected = set(expected_classes)
    missing_panels = expected - set(probs_by_dataset.keys())
    assert not missing_panels, (
        f"Missing probs_by_dataset panels for classes {sorted(missing_panels)}: "
        f"panels present={sorted(probs_by_dataset.keys())}. This is exactly the "
        f"production failure mode (rows for a class silently dropped from "
        f"training_like_df before scoring)."
    )
    for group, metrics in probs_by_dataset.items():
        missing_metrics = expected - set(metrics.keys())
        assert not missing_metrics, (
            f"probs_by_dataset[{group!r}] is missing metrics {sorted(missing_metrics)} "
            f"(has {sorted(metrics.keys())})"
        )


class TestTwoClassOnePoleDecoderCoverage:
    """2-class one-pole (target + 1 rival) -- mirrors gender_en (F/M)."""

    CLASSES = ["F", "M"]

    @pytest.fixture(scope="class")
    def trainer(self):
        return _build_one_pole_trainer(
            target_class="F",
            all_classes=self.CLASSES,
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

    def test_canonical_target_class_has_full_coverage(self, trainer, eval_frames):
        """evaluate_decoder(target_class=['F']) -- the trained pole itself."""
        val_df, neutral_df = eval_frames
        result = trainer.evaluate_decoder(
            target_class=["F"], summary_metrics=["F"],
            feature_factors=[1.0], lrs=[1e-5],
            training_like_df=val_df, neutral_df=neutral_df,
            split="validation", plot=False, use_cache=False,
        )
        for key, entry in (result.get("grid") or {}).items():
            pbd = entry.get("probs_by_dataset") if isinstance(entry, dict) else None
            assert pbd, f"grid entry {key} has no probs_by_dataset at all"
            _assert_full_coverage(pbd, self.CLASSES)

    def test_opposite_polarity_class_has_full_coverage(self, trainer, eval_frames):
        """
        evaluate_decoder(target_class=['M']) -- the *non-canonical* class, i.e.
        the shape of an opposite-polarity random control. This is the exact
        call pattern that crashed in production with
        "probs_by_dataset['M']['F'] is absent": M's own panel used to be
        silently dropped from training_like_df here.
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

    def test_weaken_direction_has_full_coverage(self, trainer, eval_frames):
        """
        increase_target_probabilities=False (weaken): maximize (1 - P(class) on
        class's own data). Only the strengthen direction was covered above;
        weaken exercises a different branch of decoder.py's per-direction
        logic and deserves its own coverage rather than being assumed safe
        by association.
        """
        val_df, neutral_df = eval_frames
        result = trainer.evaluate_decoder(
            target_class=["F"], summary_metrics=["F_weaken"],
            increase_target_probabilities=False,
            feature_factors=[1.0], lrs=[1e-5],
            training_like_df=val_df, neutral_df=neutral_df,
            split="validation", plot=False, use_cache=False,
        )
        for key, entry in (result.get("grid") or {}).items():
            pbd = entry.get("probs_by_dataset") if isinstance(entry, dict) else None
            assert pbd, f"grid entry {key} has no probs_by_dataset at all"
            _assert_full_coverage(pbd, self.CLASSES)


class TestThreeClassOnePoleDecoderCoverage:
    """3-class one-pole (target + 2 rivals) -- mirrors religion_one_pole (christian/muslim/jewish)."""

    CLASSES = ["christian", "muslim", "jewish"]

    @pytest.fixture(scope="class")
    def trainer(self):
        return _build_one_pole_trainer(
            target_class="christian",
            all_classes=self.CLASSES,
            train_rows=[
                _row(
                    "Every sunday morning without fail he would walk down the road to worship at the local [MASK]",
                    "christian", "church", "muslim", "mosque",
                ),
                _row(
                    "Every friday afternoon without fail he would walk down the road to worship at the local [MASK]",
                    "muslim", "mosque", "jewish", "synagogue",
                ),
                _row(
                    "Every saturday morning without fail he would walk down the road to worship at the local [MASK]",
                    "jewish", "synagogue", "christian", "church",
                ),
            ],
        )

    @pytest.fixture(scope="class")
    def eval_frames(self):
        val_df = pd.DataFrame([
            _row(
                "During the holiday season many families gathered together to celebrate at the nearby [MASK]",
                "christian", "church", "muslim", "mosque", split="validation",
            ),
            _row(
                "During the holiday season many families gathered together to celebrate at the nearby [MASK]",
                "muslim", "mosque", "jewish", "synagogue", split="validation",
            ),
            _row(
                "During the holiday season many families gathered together to celebrate at the nearby [MASK]",
                "jewish", "synagogue", "christian", "church", split="validation",
            ),
        ])
        neutral_df = pd.DataFrame([{"text": "The weather today is quite pleasant for a walk outside."}])
        return val_df, neutral_df

    @pytest.mark.parametrize("evaluated_class,ff", [
        ("christian", 1.0),
        ("muslim", -1.0),
        ("jewish", -1.0),
    ])
    def test_every_class_has_full_coverage_regardless_of_which_is_evaluated(
        self, trainer, eval_frames, evaluated_class, ff
    ):
        """
        Covers both the canonical target ('christian') and non-canonical
        classes ('muslim', 'jewish') -- the latter mirroring an
        opposite-polarity random control on a 3-class one-pole task, the
        exact shape that crashed on religion_one_pole in production.
        """
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


class TestFourClassOnePoleDecoderCoverage:
    """4-class one-pole (target + 3 rivals) -- generalization beyond the 2-/3-class cases already hit."""

    CLASSES = ["red", "blue", "green", "yellow"]

    @pytest.fixture(scope="class")
    def trainer(self):
        return _build_one_pole_trainer(
            target_class="red",
            all_classes=self.CLASSES,
            train_rows=[
                _row("The artist dipped the brush into the can of [MASK] paint.", "red", "red", "blue", "blue"),
                _row("The artist dipped the brush into the can of [MASK] paint.", "blue", "blue", "green", "green"),
                _row("The artist dipped the brush into the can of [MASK] paint.", "green", "green", "yellow", "yellow"),
                _row("The artist dipped the brush into the can of [MASK] paint.", "yellow", "yellow", "red", "red"),
            ],
        )

    @pytest.fixture(scope="class")
    def eval_frames(self):
        val_df = pd.DataFrame([
            _row("For the mural the students chose a bright shade of [MASK].", c, c,
                 self.CLASSES[(i + 1) % 4], self.CLASSES[(i + 1) % 4], split="validation")
            for i, c in enumerate(self.CLASSES)
        ])
        neutral_df = pd.DataFrame([{"text": "The train arrived at the station a few minutes late."}])
        return val_df, neutral_df

    @pytest.mark.parametrize("evaluated_class", ["red", "blue", "green", "yellow"])
    def test_every_class_has_full_coverage_regardless_of_which_is_evaluated(
        self, trainer, eval_frames, evaluated_class
    ):
        val_df, neutral_df = eval_frames
        ff = 1.0 if evaluated_class == "red" else -1.0
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
