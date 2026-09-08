"""
Tests that decoder evaluation uses the correct probability on the correct dataset
for strengthening and weakening.

- Strengthen class X: maximize P(X) on the *other* class's factual dataset (label_class);
  result key = target class X, value = P(X) on other's rows.
- Weaken class X: maximize (1 - P(X) on X's dataset) → eval on X's data,
  result key = "X_weaken", value = 1 - P(X) on X data.

These assertions are about which dataset the *selection value* is computed
from -- e.g. "in dataset_classes" -- not about which classes' rows may be
present in training_like_df alongside it. An earlier version of several of
these tests asserted row-level exclusivity (`dataset_classes == ["3PL"]`,
i.e. 3SG's own rows must be *absent*) as if that were required for
correctness. It wasn't: that exclusivity was itself produced by a decoder.py
pre-filtering "efficiency" optimization that silently dropped a class's own
dataset panel, which is exactly what caused real `probs_by_dataset[X][Y] is
absent` production crashes (gender_en/pronoun_number/religion_one_pole) --
same-panel weaken-selection and cross-panel callers (the opposite-polarity
random control) need that panel too. The optimization has since been
removed entirely (decoder.py no longer pre-filters training_like_df at
all); these tests now only assert the required dataset is present, not that
it's the only one.
"""

from contextlib import contextmanager

import pytest
import pandas as pd
from unittest.mock import MagicMock

from gradiend.evaluator.decoder import DecoderEvaluator, derive_feature_factor_for_class
from tests.testing_mocks import MockTokenizer, bind_trainer_cache_resolver


class MockModelWithGradiend3SG3PL:
    """Mock ModelWithGradiend with 3SG/3PL classes for decoder tests."""

    def __init__(self):
        self.name_or_path = "mock-model"
        self.base_model = MagicMock()
        self.tokenizer = MockTokenizer()
        self.source = "factual"
        self.target = "diff"
        self.feature_class_encoding_direction = {"3SG": 1.0, "3PL": -1.0}

    def rewrite_base_model(self, **kwargs):
        return self

    @contextmanager
    def intervene(self, **kwargs):
        yield self.base_model


class TrainerForStrengthenWeakenTest:
    """
    Mock trainer that records which training_like_df was passed to evaluate_base_model
    and returns controlled probs/probs_factual so we can assert correct dataset and metric.
    """

    def __init__(self):
        self._training_args = MagicMock()
        self._training_args.use_cache = False
        self._training_args.decoder_eval_max_size_training_like = 50
        self._training_args.decoder_eval_max_size_neutral = 50
        self._training_args.source = "factual"
        self._training_args.target = "diff"
        self.experiment_dir = None
        self.run_id = None
        self.target_classes = ["3SG", "3PL"]
        self._model = MockModelWithGradiend3SG3PL()
        self._evaluate_base_model_calls = []
        bind_trainer_cache_resolver(self)

    def _default_from_training_args(self, value, name, fallback=None):
        if value is not None:
            return value
        return getattr(self._training_args, name, fallback)

    def get_model(self):
        return self._model

    def get_target_feature_classes(self):
        return self.target_classes

    def _model_for_decoder_eval(self, model_with_gradiend):
        return model_with_gradiend

    def _get_decoder_eval_dataframe(self, tokenizer, **kwargs):
        """Full dataframe with both 3SG and 3PL so decoder can restrict by direction."""
        training_like_df = pd.DataFrame([
            {"masked": "[MASK] a", "label_class": "3SG", "label": "he"},
            {"masked": "[MASK] b", "label_class": "3PL", "label": "they"},
        ])
        neutral_df = pd.DataFrame([{"text": "neutral"}])
        return training_like_df, neutral_df

    def _resolve_decoder_eval_targets(self, training_like_df=None):
        return ({"3SG": ["he"], "3PL": ["they"]}, False)

    def evaluate_base_model(self, base_model, tokenizer, *, training_like_df=None, **kwargs):
        """Record which dataset classes were in training_like_df; return probs for that dataset."""
        if training_like_df is not None and hasattr(training_like_df, "columns"):
            col = "label_class" if "label_class" in training_like_df.columns else "factual_id"
            if col in training_like_df.columns:
                dataset_classes = sorted(training_like_df[col].dropna().astype(str).unique().tolist())
            else:
                dataset_classes = []
        else:
            dataset_classes = []
        self._evaluate_base_model_calls.append({"dataset_classes": dataset_classes})

        # probs[target] = P(target) on the other class's factual dataset rows.
        probs = {}
        probs_factual = {}
        for c in dataset_classes:
            others = [x for x in self.target_classes if x != c]
            if others:
                target = others[0]
                probs[target] = 0.9 if c == "3PL" else 0.2
            probs_factual[c] = 0.1 if c == "3SG" else 0.8

        result = {
            "lms": {"lms": 0.99},
            "probs": probs,
            "probs_factual": probs_factual,
            "id": "base",
        }
        # Grid entries get id from decoder; base is overwritten to "base"
        return result


class TrainerForStrengthenWeakenTestGrid:
    """
    Like above but returns different values for base vs grid so the selector picks a grid candidate.
    Also records dataset_classes per call so we can assert strengthen uses only 3PL and weaken only 3SG.
    """

    def __init__(self):
        self._training_args = MagicMock()
        self._training_args.use_cache = False
        self._training_args.decoder_eval_max_size_training_like = 50
        self._training_args.decoder_eval_max_size_neutral = 50
        self._training_args.source = "factual"
        self._training_args.target = "diff"
        self.experiment_dir = None
        self.run_id = None
        self.target_classes = ["3SG", "3PL"]
        self._model = MockModelWithGradiend3SG3PL()
        self._evaluate_base_model_calls = []
        self._call_count = 0
        bind_trainer_cache_resolver(self)

    def _default_from_training_args(self, value, name, fallback=None):
        if value is not None:
            return value
        return getattr(self._training_args, name, fallback)

    def get_model(self):
        return self._model

    def get_target_feature_classes(self):
        return self.target_classes

    def _model_for_decoder_eval(self, model_with_gradiend):
        return model_with_gradiend

    def _get_decoder_eval_dataframe(self, tokenizer, **kwargs):
        training_like_df = pd.DataFrame([
            {"masked": "[MASK] a", "label_class": "3SG", "label": "he"},
            {"masked": "[MASK] b", "label_class": "3PL", "label": "they"},
        ])
        neutral_df = pd.DataFrame([{"text": "neutral"}])
        return training_like_df, neutral_df

    def _resolve_decoder_eval_targets(self, training_like_df=None):
        return ({"3SG": ["he"], "3PL": ["they"]}, False)

    def evaluate_base_model(self, base_model, tokenizer, *, training_like_df=None, **kwargs):
        if training_like_df is not None and hasattr(training_like_df, "columns"):
            col = "label_class" if "label_class" in training_like_df.columns else "factual_id"
            if col in training_like_df.columns:
                dataset_classes = sorted(training_like_df[col].dropna().astype(str).unique().tolist())
            else:
                dataset_classes = []
        else:
            dataset_classes = []
        self._evaluate_base_model_calls.append({"dataset_classes": dataset_classes})
        self._call_count += 1
        is_base = self._call_count == 1

        probs = {}
        probs_factual = {}
        for c in dataset_classes:
            others = [x for x in self.target_classes if x != c]
            if others:
                target = others[0]
                probs[target] = 0.15 if is_base else 0.85
            probs_factual[c] = 0.2 if c == "3SG" else 0.7
            if c == "3SG":
                probs_factual[c] = 0.1 if is_base else 0.05  # weaken = 1 - factual → high for modified

        result = {
            "lms": {"lms": 0.99},
            "probs": probs,
            "probs_factual": probs_factual,
        }
        return result


class TrainerForSamePanelStrengthenTest(TrainerForStrengthenWeakenTest):
    """One-pole IO: strengthen on own factual panel (decoder_eval_prob_on_other_class=False)."""

    def __init__(self):
        super().__init__()
        self.target_classes = ["IO", "SUBJECT"]
        self._model.feature_class_encoding_direction = {"IO": 1.0, "SUBJECT": -1.0}
        self.config = MagicMock()
        self.config.decoder_eval_prob_on_other_class = False
        self._training_args.decoder_eval_prob_on_other_class = False

    def _get_decoder_eval_dataframe(self, tokenizer, **kwargs):
        training_like_df = pd.DataFrame([
            {"masked": "[MASK] a", "label_class": "IO", "label": "Alice"},
            {"masked": "[MASK] b", "label_class": "SUBJECT", "label": "Bob"},
        ])
        neutral_df = pd.DataFrame([{"text": "neutral"}])
        return training_like_df, neutral_df

    def _resolve_decoder_eval_targets(self, training_like_df=None):
        return (None, True)

    def evaluate_base_model(self, base_model, tokenizer, *, training_like_df=None, **kwargs):
        if training_like_df is not None and hasattr(training_like_df, "columns"):
            col = "label_class" if "label_class" in training_like_df.columns else "factual_id"
            if col in training_like_df.columns:
                dataset_classes = sorted(training_like_df[col].dropna().astype(str).unique().tolist())
            else:
                dataset_classes = []
        else:
            dataset_classes = []
        self._evaluate_base_model_calls.append({"dataset_classes": dataset_classes})

        # Same-panel strengthen: scalar lives in probs_factual[IO]; probs may be stale/legacy.
        probs = {"neutral": 0.01}
        probs_factual = {"IO": 0.85, "neutral": 0.99}
        return {
            "lms": {"lms": 0.99},
            "probs": probs,
            "probs_factual": probs_factual,
        }


class TestDecoderStrengthenWeakenDataset:
    """Ensure correct probability on correct dataset for strengthen and weaken."""

    def test_strengthen_uses_other_class_dataset(self):
        """Strengthen class X must evaluate on the *other* class's dataset (P(X) on other's data)."""
        evaluator = DecoderEvaluator()
        trainer = TrainerForStrengthenWeakenTest()
        result = evaluator.evaluate_decoder(
            trainer,
            target_class="3SG",
            increase_target_probabilities=True,
            feature_factors=[-1.0],
            lrs=[1e-2],
            plot=False,
        )
        # Decoder must keep at least the other class's dataset (3PL) -- that's
        # what P(3SG) is actually maximized on -- but must also keep 3SG's own
        # panel: evaluate_base_model's same-call weaken-selection reads
        # probs_by_dataset[class][class], and other callers (e.g. the
        # opposite-polarity random control) query cross-panel combinations
        # this local "which direction needs which dataset" filter can't
        # predict. Dropping a class's own rows here silently produced
        # `probs_by_dataset[X][Y] is absent` crashes in production (confirmed
        # on gender_en/pronoun_number/religion_one_pole) -- see decoder.py's
        # required_datasets computation.
        assert len(trainer._evaluate_base_model_calls) >= 1
        for call in trainer._evaluate_base_model_calls:
            assert "3PL" in call["dataset_classes"], (
                "Strengthen 3SG must use the other class's dataset (3PL): "
                "we maximize P(3SG) on 3PL data."
            )
        # Summary for 3SG should be present (selection key is the target class)
        assert "3SG" in result
        assert "value" in result["3SG"]
        assert result["3SG"]["value"] == 0.9

    def test_strengthen_row_wise_mode_uses_target_class_summary_key(self):
        """Row-wise scoring mode must still summarize under the requested target class."""
        evaluator = DecoderEvaluator()
        trainer = TrainerForStrengthenWeakenTestGrid()
        trainer._resolve_decoder_eval_targets = lambda training_like_df=None: (None, True)
        result = evaluator.evaluate_decoder(
            trainer,
            target_class="3SG",
            increase_target_probabilities=True,
            feature_factors=[-1.0],
            lrs=[1e-2],
            plot=False,
        )
        assert "3SG" in result
        assert result["3SG"]["feature_factor"] == -1.0
        assert result["3SG"]["value"] == 0.85
        assert "3PL" not in result

    def test_strengthen_summary_value_is_p_target_on_other_dataset(self):
        """Selected summary value for strengthen 3SG is the max P(3SG) on 3PL across candidates."""
        evaluator = DecoderEvaluator()
        trainer = TrainerForStrengthenWeakenTestGrid()
        result = evaluator.evaluate_decoder(
            trainer,
            target_class="3SG",
            increase_target_probabilities=True,
            feature_factors=[-1.0],
            lrs=[1e-2],
            plot=False,
        )
        for call in trainer._evaluate_base_model_calls:
            assert "3PL" in call["dataset_classes"], (
                "Strengthen must evaluate on other class's dataset (3PL)."
            )
        # Selector should pick the candidate with higher P(3SG) on 3PL (our grid returns 0.85 vs base 0.15)
        assert "3SG" in result
        assert result["3SG"]["value"] == 0.85

    def test_weaken_uses_own_class_dataset(self):
        """Weaken class X must evaluate on X's own dataset (P(X) on X data; we maximize 1 - P(X))."""
        evaluator = DecoderEvaluator()
        trainer = TrainerForStrengthenWeakenTest()
        trainer._evaluate_base_model_calls = []
        result = evaluator.evaluate_decoder(
            trainer,
            target_class="3SG",
            increase_target_probabilities=False,
            feature_factors=[1.0],
            lrs=[1e-2],
            plot=False,
        )
        for call in trainer._evaluate_base_model_calls:
            assert "3SG" in call["dataset_classes"], (
                "Weaken 3SG must use the class's own dataset (3SG). "
                "We maximize (1 - P(3SG) on 3SG data)."
            )
        assert "3SG_weaken" in result
        assert "value" in result["3SG_weaken"]
        # Mock returns probs_factual["3SG"] = 0.1 → 3SG_weaken = 0.9
        assert result["3SG_weaken"]["value"] == pytest.approx(0.9, abs=1e-5)

    def test_weaken_summary_value_is_one_minus_p_on_own_dataset(self):
        """Selected summary for weaken 3SG is 1 - P(3SG) on 3SG data; selector picks max."""
        evaluator = DecoderEvaluator()
        trainer = TrainerForStrengthenWeakenTestGrid()
        trainer._call_count = 0
        result = evaluator.evaluate_decoder(
            trainer,
            target_class="3SG",
            increase_target_probabilities=False,
            feature_factors=[1.0],
            lrs=[1e-2],
            plot=False,
        )
        for call in trainer._evaluate_base_model_calls:
            assert "3SG" in call["dataset_classes"], (
                "Weaken must evaluate on class's own dataset (3SG)."
            )
        # We return factual 3SG: base 0.1 (weaken 0.9), modified 0.05 (weaken 0.95) → selector picks 0.95
        assert "3SG_weaken" in result
        assert result["3SG_weaken"]["value"] == pytest.approx(0.95, abs=1e-5)

    def test_pairwise_weaken_derives_the_rival_feature_factor(self):
        """3SG weakening must steer toward 3PL, never reuse 3SG strengthen."""
        evaluator = DecoderEvaluator()
        trainer = TrainerForStrengthenWeakenTestGrid()
        result = evaluator.evaluate_decoder(
            trainer,
            target_class="3SG",
            increase_target_probabilities=False,
            lrs=[1e-2],
            refine_points=0,
            plot=False,
        )
        rival_ff = derive_feature_factor_for_class(
            trainer, trainer.get_model(), "3PL"
        )
        own_ff = derive_feature_factor_for_class(
            trainer, trainer.get_model(), "3SG"
        )
        assert rival_ff != own_ff
        assert result["3SG_weaken"]["feature_factor"] == rival_ff

    def test_same_panel_strengthen_uses_target_dataset(self):
        """One-pole strengthen must score on IO rows, not SUBJECT (other class)."""
        evaluator = DecoderEvaluator()
        trainer = TrainerForSamePanelStrengthenTest()
        result = evaluator.evaluate_decoder(
            trainer,
            target_class="IO",
            increase_target_probabilities=True,
            feature_factors=[-1.0],
            lrs=[1e-2],
            plot=False,
        )
        for call in trainer._evaluate_base_model_calls:
            assert "IO" in call["dataset_classes"], (
                "Same-panel strengthen IO must use IO dataset rows."
            )
        assert "IO" in result
        assert result["IO"]["value"] == 0.85
        assert "IO_weaken" not in result

    def test_evaluated_class_own_panel_survives_dataset_narrowing(self):
        """
        Regression test for a confirmed production bug (gender_en/pronoun_number/
        religion_one_pole causal reruns): required_datasets used to be exactly
        ``tcs - {evaluated_class}`` -- it *always* excluded the evaluated class's
        own dataset panel, even though same-call weaken-selection
        (probs_by_dataset[class][class]) and other callers such as the
        opposite-polarity random control (causal_study.py querying
        probs_by_dataset[opposite_class][target_metric]) need that panel too.

        This is exactly what happened live: evaluate_decoder(target_class=['M'])
        (the opposite-polarity control on a gender_en 'F' one-pole feature)
        silently dropped every 'M' row from training_like_df -- because the
        pre-filtering optimization assumed only the *other* class's data was
        ever needed -- leaving `probs_by_dataset['M']` entirely absent and
        crashing downstream with "probs_by_dataset['M']['F'] is absent".

        Reproduces the same shape here: evaluating target_class='3PL' (the
        class currently under test, analogous to the opposite-polarity control)
        must still receive 3PL's own rows in training_like_df, not just 3SG's.
        """
        evaluator = DecoderEvaluator()
        trainer = TrainerForStrengthenWeakenTest()
        evaluator.evaluate_decoder(
            trainer,
            target_class="3PL",
            increase_target_probabilities=True,
            feature_factors=[-1.0],
            lrs=[1e-2],
            plot=False,
        )
        assert len(trainer._evaluate_base_model_calls) >= 1
        for call in trainer._evaluate_base_model_calls:
            assert "3PL" in call["dataset_classes"], (
                "Evaluating target_class='3PL' must not drop 3PL's own rows from "
                "training_like_df, even though P(3PL) is maximized on 3SG's data -- "
                "3PL's own panel is still needed for same-call weaken-selection "
                "and cross-panel callers like the opposite-polarity random control."
            )
            assert "3SG" in call["dataset_classes"], (
                "Must still keep 3SG's data (what P(3PL) is actually maximized on)."
            )
