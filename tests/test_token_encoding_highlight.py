"""Tests for token-level encoded-value highlighting."""

from __future__ import annotations

import torch
import pytest


class TinyTokenizer:
    mask_token = "[MASK]"
    all_special_ids = []

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False, **kwargs):
        tokens = text.split()
        ids = list(range(1, len(tokens) + 1))
        result = {"input_ids": torch.tensor([ids])}
        if return_offsets_mapping:
            offsets = []
            cursor = 0
            for token in tokens:
                start = text.index(token, cursor)
                end = start + len(token)
                offsets.append((start, end))
                cursor = end
            result["offset_mapping"] = torch.tensor([offsets])
        return result

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return [f"tok_{token_id}" for token_id in ids]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(f"tok_{token_id}" for token_id in ids)


class Component:
    def __init__(self, component_id):
        self.id = component_id


class FakeComponentEncoder:
    def __call__(self, signal):
        return signal[1:2] * 10


class FakeComponentAccessor:
    def __getitem__(self, key):
        assert key == "activation:layer.1"
        return FakeComponentEncoder()


class FakeGradiend:
    has_component_split = True
    component_slices = (Component("activation:layer.1"),)
    device_encoder = torch.device("cpu")
    torch_dtype = torch.float32
    _component_encoders = FakeComponentAccessor()

    def _component_by_key(self, key):
        assert key == "activation:layer.1"
        return self.component_slices[0]


class FakeTextModel:
    tokenizer = TinyTokenizer()
    gradiend = FakeGradiend()
    uses_activations = False

    def create_gradients(self, masked_text, label):
        return torch.tensor([float(len(label)), float(masked_text.count("[MASK]"))])

    def encode(self, signal, return_float=False):
        return signal[0:1]


class FakeActiendModule(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self._values = torch.as_tensor(values, dtype=torch.float32)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, **kwargs):
        batch = 1 if input_ids is None else int(input_ids.shape[0])
        return self._values.unsqueeze(0).expand(batch, -1, -1).contiguous()


class FakeActiendModel:
    uses_activations = True
    activation_site_modules = ["site"]
    gradiend = FakeGradiend()
    tokenizer = TinyTokenizer()

    def __init__(self):
        self.base_model = torch.nn.Module()
        self.base_model.site = FakeActiendModule([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
        self.base_model.add_module("site", self.base_model.site)

    def encode(self, signal, return_float=False):
        return signal[0:1]


def test_compute_token_encodings_uses_original_token_as_default_label():
    from gradiend.visualizer.token_encoding import compute_token_encodings

    rows = compute_token_encodings(FakeTextModel(), "alpha beta", component=None)

    assert [row.token for row in rows] == ["alpha", "beta"]
    assert [row.label for row in rows] == ["alpha", "beta"]
    assert [row.encoded for row in rows] == [5.0, 4.0]
    assert rows[0].masked_text == "[MASK] beta"


def test_compute_token_encodings_selects_split_component_by_label():
    from gradiend.visualizer.token_encoding import compute_token_encodings

    rows = compute_token_encodings(FakeTextModel(), "alpha beta", component="layer.1")

    assert [row.encoded for row in rows] == [10.0, 10.0]
    assert rows[0].component_id == "activation:layer.1"
    assert rows[0].component_label == "layer.1"


def test_compute_token_encodings_actiend_uses_position_activations(monkeypatch):
    from gradiend.visualizer import token_encoding as token_encoding_module
    from gradiend.visualizer.token_encoding import compute_token_encodings

    model = FakeActiendModel()
    monkeypatch.setattr(
        token_encoding_module,
        "_capture_site_activations",
        lambda *_args, **_kwargs: [
            ("site", torch.tensor([[[1.0, 0.0], [0.0, 2.0]]], dtype=torch.float32)),
        ],
    )

    rows = compute_token_encodings(model, "alpha beta", component=None)

    assert [row.token for row in rows] == ["alpha", "beta"]
    assert [row.encoded for row in rows] == [1.0, 0.0]
    assert rows[0].masked_text is None
    assert rows[0].label is None


def test_compute_token_encodings_actiend_does_not_call_create_gradients(monkeypatch):
    from gradiend.visualizer import token_encoding as token_encoding_module
    from gradiend.visualizer.token_encoding import compute_token_encodings

    model = FakeActiendModel()

    def boom(*_args, **_kwargs):
        raise AssertionError("ACTIEND path must not call create_gradients")

    model.create_gradients = boom
    monkeypatch.setattr(
        token_encoding_module,
        "_capture_site_activations",
        lambda *_args, **_kwargs: [
            ("site", torch.tensor([[[1.0, 0.0], [0.0, 2.0]]], dtype=torch.float32)),
        ],
    )

    rows = compute_token_encodings(model, "alpha beta")
    assert len(rows) == 2


def test_render_token_encoding_html_escapes_tokens_and_uses_neutral_center():
    from gradiend.visualizer.token_encoding import TokenEncodingRow, render_token_encoding_html

    html = render_token_encoding_html(
        [
            TokenEncodingRow(token="<x>", encoded=-0.6, index=0),
            TokenEncodingRow(token="mid", encoded=0.4, index=1),
            TokenEncodingRow(token="hi", encoded=1.4, index=2),
        ],
        color_center="neutral",
        neutral_value=0.4,
        color_extent=1.0,
    )

    assert "&lt;x&gt;" in html
    assert "neutral = 0.4" in html
    assert "bounds [-0.6, 1.4]" in html


def test_resolve_encoding_color_norm_centers_on_neutral_value():
    from gradiend.visualizer.color_norm import resolve_encoding_color_norm

    norm = resolve_encoding_color_norm(
        [-0.6, 0.4, 1.4],
        center="neutral",
        neutral_value=0.4,
        extent=1.0,
    )

    assert norm.center == 0.4
    assert norm.vmin == -0.6
    assert norm.vmax == 1.4
    assert norm.normalize(0.4) == 0.0
    assert norm.normalize(-0.6) == -1.0
    assert norm.normalize(1.4) == 1.0


def test_trainer_resolves_neutral_encoding_baseline_from_encoder_df():
    import pandas as pd

    from gradiend.trainer.trainer import Trainer

    df = pd.DataFrame(
        {
            "encoded": [-0.8, 0.2, 0.6],
            "type": ["training", "neutral_training_masked", "neutral_dataset"],
        }
    )

    baseline = Trainer.resolve_neutral_encoding_baseline(object(), encoder_df=df)

    assert baseline == pytest.approx(0.4)


def test_trainer_resolves_component_neutral_encoding_baseline():
    import pandas as pd

    from gradiend.trainer.trainer import Trainer

    encoder_df = pd.DataFrame(
        {
            "encoded": [0.0],
            "type": ["neutral_dataset"],
        }
    )
    component_df = pd.DataFrame(
        {
            "encoded": [0.2, 0.8, -0.3],
            "type": ["neutral_dataset", "neutral_training_masked", "neutral_dataset"],
            "component_id": ["activation:left", "activation:left", "activation:right"],
            "component_label": ["left", "left", "right"],
        }
    )

    baseline = Trainer.resolve_neutral_encoding_baseline(
        object(),
        encoder_df=encoder_df,
        component_df=component_df,
        component="left",
    )

    assert baseline == pytest.approx(0.5)


def test_visualizer_resolves_neutral_center_through_trainer(monkeypatch):
    import gradiend.visualizer.visualizer as visualizer_module
    from gradiend.visualizer.visualizer import Visualizer

    captured = {}

    def fake_highlight(model, text, **kwargs):
        captured.update(kwargs)
        return "<div></div>"

    class TrainerStub:
        def get_model(self):
            return FakeTextModel()

        def resolve_neutral_encoding_baseline(self, *, component=None):
            captured["resolved_component"] = component
            return 0.4

    monkeypatch.setattr(visualizer_module, "_highlight_token_encoding", fake_highlight)

    html = Visualizer(TrainerStub()).highlight_token_encoding(
        "alpha beta",
        component="layer.1",
        color_center="neutral",
        show=False,
    )

    assert html == "<div></div>"
    assert captured["resolved_component"] == "layer.1"
    assert captured["neutral_value"] == 0.4
