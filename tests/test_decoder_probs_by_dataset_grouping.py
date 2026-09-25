"""
Contract: probs_by_dataset panel keys are factual label_class / factual_id only.

Each panel shows P(class) evaluated on rows with that factual class — never alternative_id.
Strengthen 3SG → selection metric probs["3SG"] = probs_by_dataset["3PL"]["3SG"].
"""

from __future__ import annotations

import pandas as pd
import pytest

from gradiend.trainer.text.prediction.prediction_objective import PredictionObjective
from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer
from tests.testing_mocks import MockTokenizer, SimpleMockModel


def _trainer():
    config = TextPredictionConfig(
        data=pd.DataFrame([
            {
                "masked": "[MASK] here",
                "split": "train",
                "label_class": "3SG",
                "label": "he",
                "alternative_class": "3PL",
                "alternative": "they",
            },
            {
                "masked": "[MASK] there",
                "split": "train",
                "label_class": "3PL",
                "label": "they",
                "alternative_class": "3SG",
                "alternative": "he",
            },
        ]),
        target_classes=["3SG", "3PL"],
        decoder_eval_targets={"3SG": ["he"], "3PL": ["they"]},
        decoder_eval_prob_on_other_class=True,
        masked_col="masked",
    )
    trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
    trainer._ensure_data()
    return trainer


class TestProbsByDatasetGrouping:
    def test_evaluate_groups_by_label_class_not_alternative_id(self, monkeypatch):
        trainer = _trainer()
        calls = {}

        def fake_score(self, *args, **kwargs):
            calls["dataset_class_col"] = kwargs.get("dataset_class_col")
            return {
                "3PL": {"3SG": 0.82, "3PL": 0.18},
                "3SG": {"3SG": 0.91, "3PL": 0.09},
            }

        monkeypatch.setattr(PredictionObjective, "score_probability_shift", fake_score)
        monkeypatch.setattr(PredictionObjective, "compute_lms", lambda *args, **kwargs: {"lms": 0.5})

        df = pd.DataFrame([
            {
                "masked": "[MASK] a",
                "label_class": "3SG",
                "alternative_id": "3PL",
            },
            {
                "masked": "[MASK] b",
                "label_class": "3PL",
                "alternative_id": "3SG",
            },
        ])
        result = trainer.evaluate_base_model(
            model=SimpleMockModel(),
            tokenizer=MockTokenizer(),
            training_like_df=df,
            neutral_df=pd.DataFrame([{"text": "n"}]),
            use_cache=False,
        )

        assert calls["dataset_class_col"] == "label_class"
        assert result["probs_by_dataset"]["3PL"]["3SG"] == pytest.approx(0.82)
        assert result["probs_by_dataset"]["3SG"]["3SG"] == pytest.approx(0.91)
        assert result["probs"]["3SG"] == pytest.approx(0.82)
        assert result["probs"]["3PL"] == pytest.approx(0.09)
        assert result["_probs_by_dataset_grouping"] == "label_class"
