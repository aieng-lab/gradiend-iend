"""Tests for English pronoun demo data completeness checks."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from gradiend.examples.create_english_pronoun_data import (
    PRONOUN_CLASSES,
    _has_current_english_pronoun_data,
    _incomplete_classes_sidecar_path,
    english_pronoun_generation_config,
    ensure_english_pronoun_data,
    pronoun_training_data_is_complete,
)


def _write_training_csv(path: Path, classes: list[str]) -> None:
    rows = []
    for class_id in classes:
        rows.append(
            {
                "masked": f"[MASK] {class_id}",
                "split": "train",
                "label_class": class_id,
                "label": class_id.lower(),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_pronoun_training_data_is_complete_requires_all_classes():
    with tempfile.TemporaryDirectory() as tmp:
        training_path = Path(tmp) / "training.csv"
        _write_training_csv(training_path, ["1SG", "3SG", "3PL"])
        assert not pronoun_training_data_is_complete(tmp)


def test_pronoun_training_data_is_complete_rejects_incomplete_sidecar():
    with tempfile.TemporaryDirectory() as tmp:
        training_path = Path(tmp) / "training.csv"
        _write_training_csv(training_path, PRONOUN_CLASSES)
        sidecar = _incomplete_classes_sidecar_path(training_path)
        pd.DataFrame([{"label_class": "1PL"}]).to_csv(sidecar, index=False)
        assert not pronoun_training_data_is_complete(tmp)


def test_pronoun_training_data_is_complete_when_all_classes_present():
    with tempfile.TemporaryDirectory() as tmp:
        training_path = Path(tmp) / "training.csv"
        _write_training_csv(training_path, PRONOUN_CLASSES)
        assert pronoun_training_data_is_complete(tmp)


def test_english_pronoun_generation_config_scans_full_source_with_publish_cap():
    config = english_pronoun_generation_config()
    assert config["base_max_size"] is None
    assert config["max_size_per_class"] == 10_000


@pytest.mark.parametrize("neutral_contents", ["", "text\n"])
def test_current_pronoun_data_rejects_empty_neutral_csv(neutral_contents):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_training_csv(root / "training.csv", PRONOUN_CLASSES)
        (root / "neutral.csv").write_text(neutral_contents, encoding="utf-8")
        (root / "generation_config.json").write_text(
            __import__("json").dumps(english_pronoun_generation_config()),
            encoding="utf-8",
        )

        assert not _has_current_english_pronoun_data(tmp)


def test_invalid_neutral_regenerates_only_neutral_when_training_is_current(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_training_csv(root / "training.csv", PRONOUN_CLASSES)
        (root / "neutral.csv").write_text("", encoding="utf-8")
        (root / "generation_config.json").write_text(
            __import__("json").dumps(english_pronoun_generation_config()),
            encoding="utf-8",
        )

        class _Creator:
            def generate_training_data(self, **_kwargs):
                raise AssertionError("valid training data must not be regenerated")

            def generate_neutral_data(self, **_kwargs):
                frame = pd.DataFrame({"text": ["A neutral sentence."]})
                frame.to_csv(root / "neutral.csv", index=False)
                return frame

        monkeypatch.setattr(
            "gradiend.examples.create_english_pronoun_data.build_english_pronoun_data_creator",
            lambda **_kwargs: _Creator(),
        )

        ensure_english_pronoun_data(output_dir=tmp)
        assert pd.read_csv(root / "neutral.csv")["text"].tolist() == ["A neutral sentence."]


@pytest.mark.integration
def test_build_pronoun_suite_requires_ten_trainers(monkeypatch):
    import argparse
    import sys

    with tempfile.TemporaryDirectory() as tmp:
        from experiments.multilingual_gradiend_demo import build_experiment_config, build_pronoun_suite, parse_args

        training_path = Path(tmp) / "training.csv"
        neutral_path = Path(tmp) / "neutral.csv"
        _write_training_csv(training_path, ["1SG", "3SG", "3PL"])
        pd.DataFrame([{"masked": "neutral", "split": "test"}]).to_csv(neutral_path, index=False)
        sidecar = _incomplete_classes_sidecar_path(training_path)
        pd.DataFrame(
            [
                {"masked": "[MASK] we", "split": "train", "label_class": "1PL", "label": "we"},
                {"masked": "[MASK] you", "split": "train", "label_class": "2SGPL", "label": "you"},
            ]
        ).to_csv(sidecar, index=False)

        monkeypatch.setattr(
            "experiments.multilingual_gradiend_demo._pronoun_data_paths",
            lambda: (training_path, neutral_path),
        )
        monkeypatch.setattr(sys, "argv", ["multilingual_gradiend_demo.py"])

        cli = parse_args()
        config = build_experiment_config(argparse.Namespace(**{**vars(cli), "model": None}))
        with pytest.raises(ValueError, match="expected 10 trainers"):
            build_pronoun_suite(config, retain_models_in_memory=False)
