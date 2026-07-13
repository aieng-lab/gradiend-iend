from types import SimpleNamespace

import pytest
import torch

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.suite.collection import TrainerCollection
from tests.test_trainer_model import MockTrainerForTest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def demo():
    import experiments.multilingual_gradiend_demo as mod

    return mod


def test_sentiment_suite_defaults_to_full_sentiment_only(demo, monkeypatch):
    full_trainer = MockTrainerForTest(
        model="mock-base",
        run_id="sentiment_positive_negative",
        args=TrainingArguments(experiment_dir=None),
    )
    good_bad_trainer = MockTrainerForTest(
        model="mock-base",
        run_id="sentiment_good_bad",
        args=TrainingArguments(experiment_dir=None),
    )
    full_suite = TrainerCollection(full_trainer, retain_models_in_memory=False)

    monkeypatch.setattr(
        demo,
        "_build_sentiment_full_lexicon_suite",
        lambda config, *, retain_models_in_memory, cached_run_ids=None: full_suite,
    )
    monkeypatch.setattr(
        demo,
        "_build_sentiment_good_bad_trainer",
        lambda config: good_bad_trainer,
    )

    suite = demo.build_sentiment_suite(
        SimpleNamespace(),
        retain_models_in_memory=False,
    )

    assert demo.SENTIMENT_GOOD_BAD_PAIR == ("good", "bad")
    assert demo.SENTIMENT_PRE_PRUNE_TOPK == 0.01
    assert demo.SENTIMENT_POST_PRUNE_TOPK == 0.01
    assert demo.SENTIMENT_TORCH_DTYPE is torch.bfloat16
    assert list(suite.trainers) == [
        "sentiment_positive_negative",
    ]


def test_sentiment_suite_includes_good_bad_only_when_opted_in(demo, monkeypatch):
    full_trainer = MockTrainerForTest(
        model="mock-base",
        run_id="sentiment_positive_negative",
        args=TrainingArguments(experiment_dir=None),
    )
    good_bad_trainer = MockTrainerForTest(
        model="mock-base",
        run_id="sentiment_good_bad",
        args=TrainingArguments(experiment_dir=None),
    )
    full_suite = TrainerCollection(full_trainer, retain_models_in_memory=False)

    monkeypatch.setattr(
        demo,
        "_build_sentiment_full_lexicon_suite",
        lambda config, *, retain_models_in_memory, cached_run_ids=None: full_suite,
    )
    monkeypatch.setattr(
        demo,
        "_build_sentiment_good_bad_trainer",
        lambda config: good_bad_trainer,
    )

    suite = demo.build_sentiment_suite(
        SimpleNamespace(include_sentiment_good_bad=True),
        retain_models_in_memory=False,
    )

    assert list(suite.trainers) == [
        "sentiment_positive_negative",
        "sentiment_good_bad",
    ]


def test_sent_good_bad_flag_requires_decoder_mode(demo, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["multilingual_gradiend_demo.py", "--sent-good-bad"],
    )
    cli = demo.parse_args()

    with pytest.raises(ValueError, match="--sent-good-bad is only supported"):
        demo.build_experiment_config(cli)


def test_sent_good_bad_flag_sets_decoder_config_opt_in(demo, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "multilingual_gradiend_demo.py",
            "--decoder-eval-mode",
            "left_context",
            "--sent-good-bad",
        ],
    )
    cli = demo.parse_args()
    config = demo.build_experiment_config(cli)

    assert config.include_sentiment_good_bad is True


def test_encoder_analysis_excludes_good_bad_auxiliary_sentiment(demo):
    trainers = {
        "sentiment_positive_negative": object(),
        "sentiment_good_bad": object(),
        "race_white_black": object(),
    }

    assert list(demo._filter_encoder_analysis_trainers(trainers)) == [
        "sentiment_positive_negative",
        "race_white_black",
    ]


def test_cross_task_probe_trainers_include_sentiment_pool(demo, monkeypatch):
    import pandas as pd

    created = {}

    class _ProbeTrainer:
        def __init__(self, **kwargs):
            created[kwargs["run_id"]] = kwargs

    monkeypatch.setattr(demo, "TextPredictionTrainer", _ProbeTrainer)
    monkeypatch.setattr(demo, "_pronoun_data_paths", lambda: ("pronoun.csv", "pronoun_neutral.csv"))
    monkeypatch.setattr(demo, "_sentiment_data_paths", lambda: ("sentiment.csv", "sentiment_neutral.csv"))
    monkeypatch.setattr(
        "gradiend.examples.train_sentiment.load_and_split_sentiment_training_data",
        lambda path, seed=0: pd.DataFrame(
            {
                "masked": ["[MASK] good", "[MASK] bad"],
                "split": ["test", "test"],
                "factual": ["good", "bad"],
                "alternative": ["bad", "good"],
                "factual_class": ["positive", "negative"],
                "alternative_class": ["negative", "positive"],
            }
        ),
    )

    config = demo.ExperimentConfig(
        model_name="Qwen/Qwen2.5-0.5B",
        decoder_eval_mode=demo.DecoderEvalMode.LEFT_CONTEXT,
        mlm_head_scope=demo.MlmHeadScope.PER_RUN,
        args=demo.TrainingArguments(experiment_dir=None, seed=123),
        mlm_head_args={},
    )

    probes = demo._cross_task_probe_trainers(config)

    assert set(probes) == {"pronoun_probe_pool", "sentiment_probe_pool"}
    assert created["sentiment_probe_pool"]["all_classes"] == demo.SENTIMENT_CLASSES
    assert created["sentiment_probe_pool"]["target_classes"] == ("positive", "negative")
    assert list(created["sentiment_probe_pool"]["data"]["factual_class"]) == [
        "positive",
        "negative",
    ]
