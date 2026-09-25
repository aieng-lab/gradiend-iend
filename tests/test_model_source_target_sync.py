"""TrainingArguments.source/target must be stored on ModelWithGradiend."""

from __future__ import annotations

import json

import torch
import torch.nn as nn

from gradiend.evaluator.decoder import derive_default_feature_factor
from gradiend.model import ModelWithGradiend, ParamMappedGradiendModel
from gradiend.model._source_target import (
    feature_factor_from_encoding_direction,
    resolve_model_source,
    resolve_source_from_checkpoint_dir,
    sync_model_source_target_from_training_args,
)
from tests.testing_mocks import MockTokenizer


class _Args:
    source = "alternative"
    target = "diff"


class _Model:
    def __init__(self, source: str = "factual"):
        self._source = source
        self._target = "diff"
        self.feature_class_encoding_direction = {"3SG": 1.0, "3PL": -1.0}
        self.tokenizer = MockTokenizer()

    @property
    def source(self):
        return self._source

    @property
    def target(self):
        return self._target


class _Trainer:
    def __init__(self, model: _Model, args: _Args | None = None):
        self._model = model
        self._training_args = args

    def get_model(self):
        return self._model


class _TinyCheckpointModel(ModelWithGradiend):
    """Small persistence-only model for checkpoint source compatibility tests."""

    def create_gradients(self, *args, **kwargs):
        raise NotImplementedError

    def _save_model(self, save_directory, **kwargs):
        pass

    @classmethod
    def _load_model(cls, load_directory, base_model_id=None, **kwargs):
        base = nn.Linear(10, 10)
        base.name_or_path = base_model_id or "tiny-base"
        return (base,)


def _tiny_checkpoint_model(source: str) -> _TinyCheckpointModel:
    # Use enough parameters that GRADIEND's random initialization has a stable
    # positive maximum (a 2x1 toy can very rarely initialize entirely negative).
    base = nn.Linear(10, 10)
    base.name_or_path = "tiny-base"
    param_map = {
        name: {"shape": tuple(param.shape), "repr": "all"}
        for name, param in base.named_parameters()
    }
    gradiend = ParamMappedGradiendModel(
        input_dim=sum(param.numel() for param in base.parameters()),
        latent_dim=1,
        param_map=param_map,
    )
    return _TinyCheckpointModel(base, gradiend, source=source, target="diff")


def test_sync_model_source_target_from_training_args():
    model = _Model()
    sync_model_source_target_from_training_args(model, _Args(), log_mismatch=False)
    assert model.source == "alternative"
    assert model.target == "diff"


def test_sync_does_not_overwrite_checkpoint_source_when_disabled():
    model = _Model("factual")
    sync_model_source_target_from_training_args(
        model,
        _Args(),
        log_mismatch=False,
        allow_overwrite=False,
    )
    assert model.source == "factual"


def test_resolve_source_from_checkpoint_dir_prefers_training_json(tmp_path):
    ckpt = tmp_path / "model"
    ckpt.mkdir()
    (ckpt / "gradiend_context.json").write_text(
        '{"source": "factual", "target": "diff"}',
        encoding="utf-8",
    )
    (ckpt / "training.json").write_text(
        '{"training_args": {"source": "alternative", "target": "diff"}}',
        encoding="utf-8",
    )
    assert resolve_source_from_checkpoint_dir(str(ckpt)) == "alternative"


def test_legacy_checkpoint_load_and_resave_repairs_context(tmp_path):
    """An old defaulted context must not survive a normal load/save cycle."""
    legacy = tmp_path / "legacy"
    repaired = tmp_path / "repaired"
    _tiny_checkpoint_model("factual").save_pretrained(
        str(legacy),
        use_safetensors=False,
    )
    (legacy / "training.json").write_text(
        json.dumps({"training_args": {"source": "alternative", "target": "diff"}}),
        encoding="utf-8",
    )

    loaded = _TinyCheckpointModel.from_pretrained(str(legacy))
    assert loaded.source == "alternative"

    loaded.save_pretrained(str(repaired), use_safetensors=False)
    repaired_context = json.loads(
        (repaired / "gradiend_context.json").read_text(encoding="utf-8")
    )
    assert repaired_context["source"] == "alternative"


def test_checkpoint_load_accepts_an_explicit_encoder_device(tmp_path):
    """A resolved device override must not be forwarded twice to _load_model."""
    checkpoint = tmp_path / "checkpoint"
    _tiny_checkpoint_model("alternative").save_pretrained(
        str(checkpoint),
        use_safetensors=False,
    )

    loaded = _TinyCheckpointModel.from_pretrained(
        str(checkpoint),
        device_encoder="cpu",
    )

    assert loaded.gradiend.device_encoder == torch.device("cpu")


def test_checkpoint_without_training_metadata_keeps_context_source(tmp_path):
    """Manually saved/new checkpoints remain loadable without training.json."""
    checkpoint = tmp_path / "checkpoint"
    _tiny_checkpoint_model("alternative").save_pretrained(
        str(checkpoint),
        use_safetensors=False,
    )

    loaded = _TinyCheckpointModel.from_pretrained(str(checkpoint))
    assert loaded.source == "alternative"


def test_resolve_model_source_prefers_model_over_training_args():
    model = _Model("factual")
    trainer = _Trainer(model, _Args())
    assert resolve_model_source(model, trainer) == "factual"


def test_resolve_model_source_falls_back_to_training_args():
    bare = type("Bare", (), {})()
    trainer = _Trainer(_Model())
    trainer._training_args = _Args()
    assert resolve_model_source(bare, trainer) == "alternative"


def test_derive_uses_training_args_when_model_has_no_source():
    bare = type("M", (), {
        "feature_class_encoding_direction": {"3SG": 1.0, "3PL": -1.0},
        "tokenizer": MockTokenizer(),
    })()
    trainer = _Trainer(_Model())
    trainer._training_args = _Args()
    assert derive_default_feature_factor(trainer, bare, class_name="3SG") == 1.0


def test_factual_and_alternative_feature_factor_signs():
    """Documented contract in gradiend.model._source_target."""
    assert feature_factor_from_encoding_direction(1.0, "factual") == -1.0
    assert feature_factor_from_encoding_direction(1.0, "alternative") == 1.0
    assert feature_factor_from_encoding_direction(1.0, "diff") == -1.0


def test_encoding_view_sign_for_source():
    from gradiend.model._source_target import encoding_view_sign_for_source

    assert encoding_view_sign_for_source("factual", "factual") == 1.0
    assert encoding_view_sign_for_source("alternative", "factual") == -1.0
    assert encoding_view_sign_for_source("factual", "counterfactual") == -1.0
    assert encoding_view_sign_for_source("alternative", "counterfactual") == 1.0
    assert encoding_view_sign_for_source("alternative", "transition") == 1.0
    assert encoding_view_sign_for_source("both", "factual") == 1.0
    assert encoding_view_sign_for_source("both", "counterfactual") == -1.0
    assert encoding_view_sign_for_source("both", "transition") == 1.0
    assert encoding_view_sign_for_source("diff", "factual") == 1.0
