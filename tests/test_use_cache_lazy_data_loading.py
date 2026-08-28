from types import SimpleNamespace

import pandas as pd

from gradiend import TextPredictionTrainer, TrainingArguments
from gradiend.trainer.core.arguments import TrainingArguments as CoreTrainingArguments
from gradiend.trainer.text.classification.config import TextClassificationConfig
from gradiend.trainer.text.classification.trainer import TextClassificationTrainer
import gradiend.trainer.trainer as trainer_module


def _patch_training_cache_hit(monkeypatch) -> None:
    """Train() checks should_reuse_training_cache, not trainer.has_saved_model directly."""
    monkeypatch.setattr(
        trainer_module,
        "should_reuse_training_cache",
        lambda *args, **kwargs: True,
    )


class _DummyPredictionTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 1
    pad_token = "[PAD]"
    pad_token_id = 0
    eos_token = "[EOS]"


class _DummyClassificationTokenizer:
    pad_token_id = 0


def _prediction_data():
    return {
        "3SG": pd.DataFrame(
            {
                "masked": ["[MASK] went home", "[MASK] is home", "[MASK] stayed home"],
                "split": ["train", "validation", "test"],
                "3SG": ["he", "he", "he"],
            }
        ),
        "3PL": pd.DataFrame(
            {
                "masked": ["[MASK] went home", "[MASK] are home", "[MASK] stayed home"],
                "split": ["train", "validation", "test"],
                "3PL": ["they", "they", "they"],
            }
        ),
    }


def _neutral_data():
    """Shared neutral pool: add_neutral_identity_transitions defaults True whenever
    TrainingArguments is present, and now hard-requires TextPredictionConfig.neutral_data
    (see TextPredictionTrainer._resolve_shared_neutral_dataframe).

    Pre-masked rows (masked + label) so _neutral_identity_rows uses them
    directly instead of remasking raw text via tokenizer.tokenize(), which
    _DummyPredictionTokenizer (used across this file) does not implement.
    """
    return pd.DataFrame(
        {
            "masked": ["[MASK] went home", "[MASK] is home", "[MASK] stayed home"],
            "label": ["someone", "someone", "someone"],
            "split": ["train", "validation", "test"],
        }
    )


def _classification_data():
    return pd.DataFrame(
        [
            {"text": "he went home", "label": "3SG", "split": "train"},
            {"text": "they went home", "label": "3PL", "split": "train"},
            {"text": "he stayed home", "label": "3SG", "split": "test"},
            {"text": "they stayed home", "label": "3PL", "split": "test"},
        ]
    )


def _make_prediction_trainer(experiment_dir: str) -> TextPredictionTrainer:
    args = TrainingArguments(
        experiment_dir=experiment_dir,
        use_cache=True,
        train_batch_size=2,
        max_steps=2,
    )
    return TextPredictionTrainer(
        model="bert-base-uncased",
        data=_prediction_data(),
        target_classes=["3SG", "3PL"],
        args=args,
        use_class_names_as_columns=True,
        neutral_data=_neutral_data(),
    )


def _make_classification_trainer(experiment_dir: str) -> TextClassificationTrainer:
    args = CoreTrainingArguments(
        experiment_dir=experiment_dir,
        use_cache=True,
        train_batch_size=2,
        max_steps=2,
    )
    config = TextClassificationConfig(
        data=_classification_data(),
        target_classes=["3SG", "3PL"],
    )
    return TextClassificationTrainer(
        model="bert-base-uncased",
        args=args,
        config=config,
    )


def test_prediction_cached_train_defers_data_loading(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_prediction_trainer("prediction-cache-exp")

    trainer.train()

    assert trainer._last_train_used_cache is True
    assert trainer._data_loaded is False


def test_prediction_evaluate_encoder_keeps_data_unloaded_when_encoder_df_is_provided(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_prediction_trainer("prediction-cache-exp")
    trainer.train()
    trainer._evaluator = SimpleNamespace(
        evaluate_encoder=lambda **kwargs: {"n_samples": len(kwargs["encoder_df"])}
    )

    result = trainer.evaluate_encoder(
        encoder_df=pd.DataFrame({"encoded": [0.1], "label": [1.0], "type": ["training"]}),
        use_cache=True,
    )

    assert result["n_samples"] == 1
    assert trainer._data_loaded is False


def test_prediction_evaluate_encoder_lazy_loads_data_when_needed(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_prediction_trainer("prediction-cache-exp")
    trainer.train()
    trainer._evaluator = SimpleNamespace(evaluate_encoder=lambda **kwargs: {"ok": True})
    trainer.get_model = lambda: SimpleNamespace(tokenizer=_DummyPredictionTokenizer())

    def _fake_analyze_encoder(model_with_gradiend, **kwargs):
        assert trainer._data_loaded is False
        trainer.create_training_data(
            model_with_gradiend.tokenizer,
            split="test",
            batch_size=1,
            max_size=1,
            balance_column=None,
        )
        assert trainer._data_loaded is True
        return pd.DataFrame({"encoded": [0.1], "label": [1.0], "type": ["training"]})

    trainer._analyze_encoder = _fake_analyze_encoder

    trainer.evaluate_encoder(use_cache=False)

    assert trainer._data_loaded is True


def test_prediction_evaluate_decoder_keeps_data_unloaded_when_frames_are_supplied(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_prediction_trainer("prediction-cache-exp")
    trainer.train()
    trainer.get_model = lambda: SimpleNamespace(tokenizer=_DummyPredictionTokenizer())
    trainer._evaluator = SimpleNamespace(
        evaluate_decoder=lambda **kwargs: {
            "training_rows": len(kwargs["training_like_df"]),
            "neutral_rows": len(kwargs["neutral_df"]),
        }
    )

    result = trainer.evaluate_decoder(
        training_like_df=pd.DataFrame({"masked": ["[MASK] went home"], "factual": ["he"], "alternative": ["they"]}),
        neutral_df=pd.DataFrame({"text": ["he went home"]}),
        use_cache=False,
    )

    assert result["training_rows"] == 1
    assert result["neutral_rows"] == 1
    assert trainer._data_loaded is False


def test_prediction_evaluate_decoder_lazy_loads_data_when_needed(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_prediction_trainer("prediction-cache-exp")
    trainer.train()
    trainer.get_model = lambda: SimpleNamespace(tokenizer=_DummyPredictionTokenizer())

    def _fake_evaluate_decoder(**kwargs):
        assert trainer._data_loaded is False
        training_like_df, neutral_df = trainer._get_decoder_eval_dataframe(
            _DummyPredictionTokenizer(),
            cached_training_like_df=kwargs.get("training_like_df"),
            cached_neutral_df=kwargs.get("neutral_df"),
        )
        assert trainer._data_loaded is True
        return {
            "training_rows": len(training_like_df),
            "neutral_rows": len(neutral_df),
        }

    trainer._evaluator = SimpleNamespace(evaluate_decoder=_fake_evaluate_decoder)

    result = trainer.evaluate_decoder(use_cache=False)

    assert result["training_rows"] >= 1
    assert result["neutral_rows"] >= 1
    assert trainer._data_loaded is True


def test_prediction_evaluate_decoder_uses_requested_split(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_prediction_trainer("prediction-cache-exp")
    trainer.train()
    trainer.get_model = lambda: SimpleNamespace(tokenizer=_DummyPredictionTokenizer())

    def _fake_evaluate_decoder(**kwargs):
        training_like_df, _ = trainer._get_decoder_eval_dataframe(
            _DummyPredictionTokenizer(),
            split=kwargs.get("split"),
            cached_training_like_df=kwargs.get("training_like_df"),
            cached_neutral_df=kwargs.get("neutral_df"),
        )
        return {"splits": sorted(training_like_df["split"].astype(str).unique())}

    trainer._evaluator = SimpleNamespace(evaluate_decoder=_fake_evaluate_decoder)

    result = trainer.evaluate_decoder(split="validation", use_cache=False)

    assert result["splits"] == ["validation"]


def test_prediction_decoder_plotting_analysis_uses_training_argument_caps(monkeypatch):
    trainer = _make_prediction_trainer("prediction-cache-exp")
    trainer._training_args.decoder_eval_max_size_training_like = 7
    trainer._training_args.decoder_eval_max_size_neutral = 11
    trainer._training_args.eval_batch_size = 3
    trainer.get_model = lambda: SimpleNamespace(
        base_model=SimpleNamespace(),
        tokenizer=_DummyPredictionTokenizer(),
    )

    captured = {}

    def _fake_get_decoder_eval_dataframe(tokenizer, **kwargs):
        captured["data_kwargs"] = kwargs
        return (
            pd.DataFrame({"masked": ["[MASK] went home"], "label_class": ["3SG"]}),
            pd.DataFrame({"text": ["neutral"]}),
        )

    def _fake_evaluate_base_model(model, tokenizer, **kwargs):
        captured["base_kwargs"] = kwargs
        return {"probs_by_dataset": {"3SG": {"3SG": 0.8}}}

    trainer._get_decoder_eval_dataframe = _fake_get_decoder_eval_dataframe
    trainer._resolve_decoder_eval_targets = lambda training_like_df=None: ({"3SG": ["he"]}, False)
    trainer.evaluate_base_model = _fake_evaluate_base_model

    trainer.analyze_decoder_for_plotting(
        decoder_results={"grid": {"base": {}}},
        class_ids=["3SG"],
        use_cache=True,
    )

    assert captured["data_kwargs"]["split"] == "test"
    assert captured["data_kwargs"]["max_size_training_like"] == 7
    assert captured["data_kwargs"]["max_size_neutral"] == 11
    assert captured["base_kwargs"]["max_size_training_like"] == 7
    assert captured["base_kwargs"]["max_size_neutral"] == 11
    assert captured["base_kwargs"]["eval_batch_size"] == 3


def test_analyze_decoder_for_plotting_forwards_intervention_kwargs():
    """Plot refresh must reuse the grid's token_selector / activation_gate, not hardcode encoder_direction."""
    from contextlib import contextmanager

    trainer = _make_prediction_trainer("prediction-cache-exp")
    trainer._training_args.decoder_eval_max_size_training_like = 4
    trainer._training_args.decoder_eval_max_size_neutral = 4
    trainer._training_args.eval_batch_size = 2

    captured = {}

    @contextmanager
    def _fake_intervene(**kwargs):
        captured["intervene"] = kwargs
        yield SimpleNamespace()

    model = SimpleNamespace(
        base_model=SimpleNamespace(),
        tokenizer=_DummyPredictionTokenizer(),
        intervene=_fake_intervene,
    )
    trainer.get_model = lambda: model
    trainer._get_decoder_eval_dataframe = lambda tokenizer, **kwargs: (
        pd.DataFrame({"masked": ["[MASK] went home"], "label_class": ["3SG"]}),
        pd.DataFrame({"text": ["neutral"]}),
    )
    trainer._resolve_decoder_eval_targets = lambda training_like_df=None: ({"3SG": ["he"]}, False)
    trainer.evaluate_base_model = lambda *args, **kwargs: {
        "probs_by_dataset": {"3SG": {"3SG": 0.8}},
        "_probs_by_dataset_grouping": "label_class",
    }

    trainer.analyze_decoder_for_plotting(
        decoder_results={
            "grid": {
                "base": {},
                (1.0, 10.0): {
                    "id": {"feature_factor": 1.0, "learning_rate": 10.0},
                    "probs_by_dataset": {"3SG": {"3SG": 0.5}},
                },
            },
            "intervention_kwargs": {
                "token_selector": "all",
                "activation_gate": None,
                "threshold": 0.5,
            },
        },
        class_ids=["3SG"],
        use_cache=False,
    )

    assert captured["intervene"]["token_selector"] == "all"
    assert captured["intervene"]["value"] == 10.0
    assert captured["intervene"]["feature_factor"] == 1.0
    assert "activation_gate" not in captured["intervene"] or captured["intervene"].get("activation_gate") is None


def test_classification_cached_train_defers_data_loading(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_classification_trainer("classification-cache-exp")

    trainer.train()

    assert trainer._last_train_used_cache is True
    assert trainer._combined_data is None


def test_classification_evaluate_encoder_keeps_data_unloaded_when_encoder_df_is_provided(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_classification_trainer("classification-cache-exp")
    trainer.train()
    trainer._evaluator = SimpleNamespace(
        evaluate_encoder=lambda **kwargs: {"n_samples": len(kwargs["encoder_df"])}
    )

    result = trainer.evaluate_encoder(
        encoder_df=pd.DataFrame({"encoded": [0.1], "label": [1.0], "type": ["training"]}),
        use_cache=True,
    )

    assert result["n_samples"] == 1
    assert trainer._combined_data is None


def test_classification_evaluate_encoder_lazy_loads_data_when_needed(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_classification_trainer("classification-cache-exp")
    trainer.train()
    trainer._evaluator = SimpleNamespace(evaluate_encoder=lambda **kwargs: {"ok": True})
    trainer.get_model = lambda: SimpleNamespace(tokenizer=_DummyClassificationTokenizer())

    def _fake_analyze_encoder(model_with_gradiend, **kwargs):
        assert trainer._combined_data is None
        trainer.create_training_data(model_with_gradiend.tokenizer, split="test", batch_size=1, max_size=1)
        assert trainer._combined_data is not None
        return pd.DataFrame({"encoded": [0.1], "label": [1.0], "type": ["training"]})

    trainer._analyze_encoder = _fake_analyze_encoder

    trainer.evaluate_encoder(use_cache=False)

    assert trainer._combined_data is not None


def test_classification_evaluate_decoder_keeps_data_unloaded_when_frames_are_supplied(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_classification_trainer("classification-cache-exp")
    trainer.train()
    trainer.get_model = lambda: SimpleNamespace(tokenizer=_DummyClassificationTokenizer())
    trainer._evaluator = SimpleNamespace(
        evaluate_decoder=lambda **kwargs: {
            "training_rows": len(kwargs["training_like_df"]),
            "neutral_rows": len(kwargs["neutral_df"]),
        }
    )

    result = trainer.evaluate_decoder(
        training_like_df=pd.DataFrame({"text": ["he stayed home"], "label_class": ["3SG"]}),
        neutral_df=pd.DataFrame({"text": ["they stayed home"]}),
        use_cache=False,
    )

    assert result["training_rows"] == 1
    assert result["neutral_rows"] == 1
    assert trainer._combined_data is None


def test_classification_evaluate_decoder_lazy_loads_data_when_needed(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_classification_trainer("classification-cache-exp")
    trainer.train()
    trainer.get_model = lambda: SimpleNamespace(tokenizer=_DummyClassificationTokenizer())

    def _fake_evaluate_decoder(**kwargs):
        assert trainer._combined_data is None
        training_like_df, neutral_df = trainer._get_decoder_eval_dataframe(
            _DummyClassificationTokenizer(),
            cached_training_like_df=kwargs.get("training_like_df"),
            cached_neutral_df=kwargs.get("neutral_df"),
        )
        assert trainer._combined_data is not None
        return {
            "training_rows": len(training_like_df),
            "neutral_rows": len(neutral_df),
        }

    trainer._evaluator = SimpleNamespace(evaluate_decoder=_fake_evaluate_decoder)

    result = trainer.evaluate_decoder(use_cache=False)

    assert result["training_rows"] >= 1
    assert result["neutral_rows"] >= 1
    assert trainer._combined_data is not None


def test_classification_evaluate_decoder_uses_requested_split(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_classification_trainer("classification-cache-exp")
    trainer.train()
    trainer.get_model = lambda: SimpleNamespace(tokenizer=_DummyClassificationTokenizer())

    def _fake_evaluate_decoder(**kwargs):
        training_like_df, _ = trainer._get_decoder_eval_dataframe(
            _DummyClassificationTokenizer(),
            split=kwargs.get("split"),
            cached_training_like_df=kwargs.get("training_like_df"),
            cached_neutral_df=kwargs.get("neutral_df"),
        )
        return {"splits": sorted(training_like_df["split"].astype(str).unique())}

    trainer._evaluator = SimpleNamespace(evaluate_decoder=_fake_evaluate_decoder)

    result = trainer.evaluate_decoder(split="test", use_cache=False)

    assert result["splits"] == ["test"]


def test_encoder_cache_path_does_not_trigger_hf_data_load(monkeypatch, tmp_path):
    ensure_calls = {"n": 0}
    real_ensure = TextPredictionTrainer._ensure_data

    def _tracking_ensure(self, **kwargs):
        ensure_calls["n"] += 1
        return real_ensure(self, **kwargs)

    monkeypatch.setattr(TextPredictionTrainer, "_ensure_data", _tracking_ensure)

    trainer = TextPredictionTrainer(
        model="bert-base-uncased",
        run_id="gender_de_masc_nom_masc_dat",
        data="aieng-lab/de-gender-case-articles",
        target_classes=["masc_nom", "masc_dat"],
        args=TrainingArguments(experiment_dir=str(tmp_path), use_cache=True),
    )

    cache_path = trainer._encoder_cache_path("", split="test", max_size=50)

    assert ensure_calls["n"] == 0
    assert trainer._combined_data is None
    assert cache_path is not None
    assert "encoded_values_max_size_50_split_test.csv" in cache_path.replace("\\", "/")
