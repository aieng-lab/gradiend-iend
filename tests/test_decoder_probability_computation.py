"""
Tests for decoder probability computation in evaluate_base_model.

All data is built in-memory. The tests intentionally avoid external datasets or
files so they remain stable across local and CI environments.
"""

import pandas as pd
import pytest

from gradiend.trainer.text.prediction.prediction_objective import PredictionObjective
from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer
from tests.testing_mocks import MockTokenizer, SimpleMockModel


def _with_required_splits(rows):
    """Replicate tiny rows over train/validation/test, as current trainer validation requires all three."""
    out = []
    for split in ("train", "validation", "test"):
        for row in rows:
            copied = dict(row)
            copied["split"] = split
            out.append(copied)
    return pd.DataFrame(out)


def _two_class_data(left="3SG", right="3PL", left_token="he", right_token="they"):
    return _with_required_splits(
        [
            {
                "masked": "[MASK] here",
                "label_class": left,
                "label": left_token,
                "alternative_class": right,
                "alternative": right_token,
            },
            {
                "masked": "[MASK] there",
                "label_class": right,
                "label": right_token,
                "alternative_class": left,
                "alternative": left_token,
            },
        ]
    )


def _trainer(config):
    trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
    trainer._ensure_data()
    return trainer


class TestEvaluateBaseModelProbabilityComputation:
    def test_evaluate_base_model_returns_probs_dict(self, monkeypatch):
        config = TextPredictionConfig(
            data=_two_class_data(),
            target_classes=["3SG", "3PL"],
            decoder_eval_targets={"3SG": ["he", "He"], "3PL": ["they", "They"]},
            masked_col="masked",
        )
        trainer = _trainer(config)

        calls = {}

        def fake_score(self, *args, **kwargs):
            calls["score_kwargs"] = kwargs
            return {
                "3PL": {"3SG": 0.8, "3PL": 0.2},
                "3SG": {"3SG": 0.6, "3PL": 0.4},
            }

        monkeypatch.setattr(PredictionObjective, "score_probability_shift", fake_score)
        monkeypatch.setattr(PredictionObjective, "compute_lms", lambda *args, **kwargs: {"lms": 0.5})

        result = trainer.evaluate_base_model(
            model=SimpleMockModel(),
            tokenizer=MockTokenizer(),
            training_like_df=pd.DataFrame(
                [
                    {"masked": "[MASK] here", "label_class": "3SG", "alternative_id": "3PL"},
                    {"masked": "[MASK] there", "label_class": "3PL", "alternative_id": "3SG"},
                ]
            ),
            neutral_df=pd.DataFrame([{"text": "neutral text"}]),
            use_cache=False,
        )

        assert result["probs"] == {"3SG": 0.8, "3PL": 0.4}
        assert result["lms"] == {"lms": 0.5}
        assert calls["score_kwargs"]["dataset_class_col"] == "label_class"

    def test_evaluate_base_model_counterfactual_evaluation(self, monkeypatch):
        config = TextPredictionConfig(
            data=_two_class_data(),
            target_classes=["3SG", "3PL"],
            decoder_eval_targets={"3SG": ["he"], "3PL": ["they"]},
            decoder_eval_prob_on_other_class=True,
            masked_col="masked",
        )
        trainer = _trainer(config)

        monkeypatch.setattr(
            PredictionObjective,
            "score_probability_shift",
            lambda *args, **kwargs: {"3PL": {"3SG": 0.8, "3PL": 0.2}, "3SG": {"3SG": 0.6, "3PL": 0.4}},
        )
        monkeypatch.setattr(PredictionObjective, "compute_lms", lambda *args, **kwargs: {"lms": 0.5})

        result = trainer.evaluate_base_model(
            model=SimpleMockModel(),
            tokenizer=MockTokenizer(),
            training_like_df=pd.DataFrame(
                [
                    {"masked": "[MASK] here", "label_class": "3SG", "alternative_id": "3PL"},
                    {"masked": "[MASK] there", "label_class": "3PL", "alternative_id": "3SG"},
                ]
            ),
            neutral_df=pd.DataFrame([{"text": "neutral_data"}]),
            use_cache=False,
        )

        assert result["probs"]["3SG"] == 0.8
        assert result["probs"]["3PL"] == 0.4

    def test_evaluate_base_model_multiple_tokens_per_class(self, monkeypatch):
        config = TextPredictionConfig(
            data=_two_class_data("masc_nom", "fem_nom", "der", "die"),
            target_classes=["masc_nom", "fem_nom"],
            decoder_eval_targets={"masc_nom": ["der", "Der"], "fem_nom": ["die", "Die"]},
            masked_col="masked",
        )
        trainer = _trainer(config)
        calls = {}

        def fake_score(self, *args, **kwargs):
            calls["targets"] = kwargs["targets"]
            return {
                "fem_nom": {"masc_nom": 0.8, "fem_nom": 0.2},
                "masc_nom": {"masc_nom": 0.2, "fem_nom": 0.8},
            }

        monkeypatch.setattr(PredictionObjective, "score_probability_shift", fake_score)
        monkeypatch.setattr(PredictionObjective, "compute_lms", lambda *args, **kwargs: {"lms": 0.5})

        result = trainer.evaluate_base_model(
            model=SimpleMockModel(),
            tokenizer=MockTokenizer(),
            training_like_df=pd.DataFrame(
                [
                    {"masked": "[MASK] here", "label_class": "masc_nom", "alternative_id": "fem_nom"},
                    {"masked": "[MASK] there", "label_class": "fem_nom", "alternative_id": "masc_nom"},
                ]
            ),
            neutral_df=pd.DataFrame([{"text": "neutral_data"}]),
            use_cache=False,
        )

        assert calls["targets"] == {"masc_nom": ["der", "Der"], "fem_nom": ["die", "Die"]}
        assert result["probs"] == {"masc_nom": 0.8, "fem_nom": 0.8}

    def test_explicit_decoder_targets_with_overlap_warns_not_errors(self):
        config = TextPredictionConfig(
            data=_two_class_data("C1", "C2", "x", "y"),
            target_classes=["C1", "C2"],
            decoder_eval_targets={"C1": ["x", "+"], "C2": ["+", "y"]},
            masked_col="masked",
        )
        trainer = _trainer(config)

        normalized = trainer._validate_explicit_decoder_eval_targets(config.decoder_eval_targets)

        assert normalized["C1"] == ["x", "+"]
        assert normalized["C2"] == ["+", "y"]

    def test_explicit_decoder_targets_invalid_class_key_raises(self):
        config = TextPredictionConfig(
            data=_two_class_data("REAL", "OTHER", "x", "z"),
            target_classes=["REAL", "OTHER"],
            decoder_eval_targets={"REAL": ["x"], "TYPO": ["y"]},
            masked_col="masked",
        )
        trainer = _trainer(config)

        with pytest.raises(ValueError) as exc_info:
            trainer._validate_explicit_decoder_eval_targets(config.decoder_eval_targets)

        assert "TYPO" in str(exc_info.value)
        assert "REAL" in str(exc_info.value) or "known classes" in str(exc_info.value)

    def test_resolve_decoder_eval_targets_returns_tuple(self):
        config = TextPredictionConfig(
            data=_two_class_data("C1", "C2", "x", "y"),
            target_classes=["C1", "C2"],
            decoder_eval_targets={"C1": ["t1"], "C2": ["t2"]},
            masked_col="masked",
        )
        trainer = _trainer(config)

        targets, use_row_wise = trainer._resolve_decoder_eval_targets(training_like_df=None)

        assert use_row_wise is False
        assert targets == {"C1": ["t1"], "C2": ["t2"]}

    def test_decoder_eval_targets_label_uses_row_wise_mode(self):
        config = TextPredictionConfig(
            data=_two_class_data("commutative", "non-commutative", "=", "!="),
            target_classes=["commutative", "non-commutative"],
            decoder_eval_targets="label",
            masked_col="masked",
        )
        trainer = _trainer(config)

        targets, use_row_wise = trainer._resolve_decoder_eval_targets(training_like_df=None)

        assert targets is None
        assert use_row_wise is True

    def test_evaluate_base_model_row_wise_normalizes_counterfactual_probs(self, monkeypatch):
        """Row-wise scoring must still expose probs[T]=P(T) on the other class's dataset."""
        config = TextPredictionConfig(
            data=_two_class_data("M", "F", "he", "she"),
            target_classes=["M", "F"],
            decoder_eval_targets="label",
            decoder_eval_prob_on_other_class=True,
            masked_col="masked",
        )
        trainer = _trainer(config)

        def fake_score(self, *args, **kwargs):
            assert kwargs.get("use_row_wise") is True
            return {
                "F": {"M": 0.81, "F": 0.19},
                "M": {"M": 0.55, "F": 0.45},
            }

        monkeypatch.setattr(PredictionObjective, "score_probability_shift", fake_score)
        monkeypatch.setattr(PredictionObjective, "compute_lms", lambda *args, **kwargs: {"lms": 0.5})

        result = trainer.evaluate_base_model(
            model=SimpleMockModel(),
            tokenizer=MockTokenizer(),
            training_like_df=pd.DataFrame(
                [
                    {
                        "masked": "[MASK] here",
                        "label_class": "M",
                        "label": "he",
                        "alternative": "she",
                        "alternative_id": "F",
                    },
                    {
                        "masked": "[MASK] there",
                        "label_class": "F",
                        "label": "she",
                        "alternative": "he",
                        "alternative_id": "M",
                    },
                ]
            ),
            neutral_df=pd.DataFrame([{"text": "neutral_data"}]),
            use_cache=False,
        )

        assert result["probs"]["M"] == 0.81
        assert result["probs"]["F"] == 0.45
        assert result["probs_factual"]["M"] == 0.55
        assert result["probs_factual"]["F"] == 0.19

    def test_evaluate_base_model_same_panel_strengthen_ignores_neutral_panel(self, monkeypatch):
        """One-pole / same-panel strengthen must expose probs[IO], not neutral from first panel."""
        config = TextPredictionConfig(
            data=_two_class_data("IO", "SUBJECT", "Alice", "Bob"),
            target_classes=["IO", "SUBJECT"],
            decoder_eval_targets="label",
            decoder_eval_prob_on_other_class=False,
            masked_col="masked",
        )
        trainer = _trainer(config)

        monkeypatch.setattr(
            PredictionObjective,
            "score_probability_shift",
            lambda *args, **kwargs: {
                "neutral": {"neutral": 0.99, "IO": 0.01, "SUBJECT": 0.01},
                "IO": {"IO": 0.72, "SUBJECT": 0.28},
            },
        )
        monkeypatch.setattr(PredictionObjective, "compute_lms", lambda *args, **kwargs: {"lms": 0.5})

        result = trainer.evaluate_base_model(
            model=SimpleMockModel(),
            tokenizer=MockTokenizer(),
            training_like_df=pd.DataFrame(
                [
                    {"masked": "[MASK] x", "label_class": "neutral", "label": "neutral", "alternative": "x"},
                    {"masked": "[MASK] y", "label_class": "IO", "label": "Alice", "alternative": "Bob"},
                ]
            ),
            neutral_df=pd.DataFrame([{"text": "neutral_data"}]),
            use_cache=False,
        )

        assert result["probs"]["IO"] == 0.72
        assert "neutral" not in result["probs"]
        assert result["probs_factual"]["IO"] == 0.72

    def test_decoder_eval_dataframe_caps_training_like_per_class_and_neutral_separately(self):
        """Decoder plots need both factual panels; max_size is a per-class cap, not a global first-N cap."""
        rows = []
        for split in ("train", "validation", "test"):
            for index in range(5):
                rows.append(
                    {
                        "masked": f"[MASK] m_{split}_{index}",
                        "label_class": "M",
                        "label": "he",
                        "alternative_class": "F",
                        "alternative": "she",
                        "split": split,
                    }
                )
                rows.append(
                    {
                        "masked": f"[MASK] f_{split}_{index}",
                        "label_class": "F",
                        "label": "she",
                        "alternative_class": "M",
                        "alternative": "he",
                        "split": split,
                    }
                )
        neutral_rows = pd.DataFrame(
            [
                {"text": f"neutral {split} {index}", "split": split}
                for split in ("train", "validation", "test")
                for index in range(5)
            ]
        )
        config = TextPredictionConfig(
            data=pd.DataFrame(rows),
            target_classes=["M", "F"],
            decoder_eval_targets={"M": ["he"], "F": ["she"]},
            masked_col="masked",
            neutral_data=neutral_rows,
        )
        trainer = _trainer(config)

        training_like_df, neutral_df = trainer._get_decoder_eval_dataframe(
            MockTokenizer(),
            max_size_training_like=2,
            max_size_neutral=2,
            split="test",
        )

        dataset_class_col = "label_class" if "label_class" in training_like_df.columns else "factual_id"
        assert training_like_df[dataset_class_col].value_counts().to_dict() == {"F": 2, "M": 2}
        assert set(training_like_df["split"]) == {"test"}
        assert len(neutral_df) == 2
        assert set(neutral_df["split"]) == {"test"}
