"""A published neutral dataset with only a "train" split is used for every encoder split."""

from types import SimpleNamespace

import pandas as pd
import pytest

from gradiend.trainer.text.prediction import trainer as trainer_module
from gradiend.trainer.text.prediction.trainer import TextPredictionTrainer


def _trainer():
    trainer = object.__new__(TextPredictionTrainer)
    trainer.__dict__["config"] = SimpleNamespace(dataset_trust_remote_code=None)
    return trainer


def _resolve(trainer, split="test"):
    return trainer._resolve_neutral_source_dataframe(
        "org/neutral", split=split, max_rows=None, required=True, field_name="eval_neutral_data",
        auto_split_if_missing=False,
    )


def test_missing_split_falls_back_to_train(monkeypatch):
    calls = []

    def fake_resolve(source, split=None, max_rows=None, dataset_trust_remote_code=None):
        calls.append(split)
        if split != "train":
            raise ValueError(f'Unknown split "{split}". Should be one of [\'train\'].')
        return pd.DataFrame({"text": ["a", "b", "c"]})

    monkeypatch.setattr(trainer_module, "resolve_dataframe", fake_resolve)
    assert len(_resolve(_trainer())) == 3
    assert calls == ["test", "train"]


def test_an_available_split_is_used_directly(monkeypatch):
    calls = []

    def fake_resolve(source, split=None, max_rows=None, dataset_trust_remote_code=None):
        calls.append(split)
        return pd.DataFrame({"text": ["a", "b"]})

    monkeypatch.setattr(trainer_module, "resolve_dataframe", fake_resolve)
    assert len(_resolve(_trainer())) == 2
    assert calls == ["test"]


def test_other_loading_errors_are_not_swallowed(monkeypatch):
    def fake_resolve(source, split=None, max_rows=None, dataset_trust_remote_code=None):
        raise ValueError("the parquet file is corrupt")

    monkeypatch.setattr(trainer_module, "resolve_dataframe", fake_resolve)
    with pytest.raises(ValueError, match="corrupt"):
        _resolve(_trainer())
