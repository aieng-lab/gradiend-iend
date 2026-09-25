from pathlib import Path

import pandas as pd

from gradiend.examples import train_sentiment


def _minimal_sentiment_merged() -> pd.DataFrame:
    rows = []
    for i, (word, cls) in enumerate(
        [
            ("happy", "positive"),
            ("sad", "negative"),
            ("glad", "positive"),
            ("mad", "negative"),
            ("cool", "positive"),
            ("bad", "negative"),
            ("good", "positive"),
            ("sick", "negative"),
            ("nice", "positive"),
            ("fake", "negative"),
            ("real", "positive"),
            ("late", "negative"),
        ]
    ):
        other_cls = "negative" if cls == "positive" else "positive"
        other_word = "sad" if cls == "positive" else "happy"
        rows.append(
            {
                "masked": f"text {i} [MASK]",
                "label_class": cls,
                "label": word,
                "alternative_class": other_cls,
                "alternative": other_word,
            }
        )
    return pd.DataFrame(rows)


def test_apply_vocabulary_held_out_split_assigns_all_splits():
    split_df = train_sentiment.apply_vocabulary_held_out_split(
        _minimal_sentiment_merged(),
        seed=0,
    )
    splits = set(split_df["split"].astype(str).unique())
    assert {"train", "validation", "test"}.issubset(splits)


def test_train_sentiment_seed_controls_training_args_seed(monkeypatch):
    """``train()``'s ``seed`` must flow into the ``TrainingArguments`` used for the run.

    ``train()`` now sources data from the published HF datasets (fixed
    vocabulary-held-out ``split`` subset) rather than a local, seed-resplit
    CSV, so ``seed`` no longer controls the train/validation/test split here
    (that manual-split flow lives in ``train_multi_seed_heldout_targets``
    instead). This only verifies the seed still reaches the trainer's args.
    """
    captured = {}

    class DummyTrainer:
        def __init__(self, *, model, config, args):
            captured["model"] = model
            captured["config"] = config
            captured["args"] = args

        def train(self):
            captured["trained"] = True

        def plot_training_convergence(self):
            captured["plotted"] = True

        def get_training_stats(self):
            return {"training_stats": {}}

        def evaluate_encoder(self, *, plot):
            captured["encoder_plot"] = plot
            return {}

        def get_encoder_metrics(self, *, use_cache):
            captured["metrics_use_cache"] = use_cache
            return {}

    monkeypatch.setattr(train_sentiment, "TextPredictionTrainer", DummyTrainer)
    monkeypatch.setattr(
        train_sentiment,
        "load_english_sentiment_neutral_data",
        lambda *args, **kwargs: pd.DataFrame({"text": ["a neutral sentence"]}),
    )
    monkeypatch.setattr(train_sentiment, "_evaluate_split_stability", lambda trainer, experiment_dir: None)

    train_sentiment.train(
        experiment_dir=Path("unused-test-path/runs"),
        seed=123,
        max_steps=1,
        train_batch_size=1,
    )

    assert captured["args"].seed == 123
    assert captured["config"].split_col == "split"
    assert captured["trained"] is True
