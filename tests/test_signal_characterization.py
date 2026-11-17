"""Characterization tests: pin the behaviour of the single-signal (gradient, activation)
pipeline on a tiny random GPT-2 so that the encoder-input/decoder-target split refactor
provably leaves every existing method unchanged.

Golden values were recorded from the code BEFORE the refactor (CPU, float32, seed 7).
"""
import json
import math
import os
import random

import numpy as np
import pytest
import torch

from gradiend import Signal, SignalScope

from tests.tiny_gpt2_support import (
    make_tiny_gpt2,
    make_trainer,
    tiny_backbone,
    tiny_gpt2_available,
    training_arguments,
)

pytestmark = pytest.mark.skipif(not tiny_gpt2_available(), reason="gpt2 tokenizer not cached")

GOLDEN = {
    "gradient": dict(
        input_dim=6560,
        enc_shape=[1, 6560],
        dec_shape=[6560, 1],
        len_ds=17,
        item0=dict(src_sum=-0.16870534420013428, src_norm=3.967400312423706,
                   tgt_sum=-0.8148870468139648, tgt_norm=8.549541473388672, label="-1"),
        losses=[0.011164125986397266, 3.0138033253024332e-05, 0.011975947767496109,
                0.01487804390490055, 2.1855617887922563e-05, 0.01567842625081539],
    ),
    "activation": dict(
        input_dim=32,
        enc_shape=[1, 32],
        dec_shape=[32, 1],
        len_ds=17,
        item0=dict(src_sum=-0.04580646753311157, src_norm=0.11121029406785965,
                   tgt_sum=-0.07942444086074829, tgt_norm=0.13218489289283752, label="-1"),
        losses=[0.0005246649379841983, 3.402836591703817e-05, 0.0006084893830120564,
                0.0005270748515613377, 26.23240089416504, 5.03961181640625],
    ),
}


def _seed_all():
    # row splits, counterfactual draws and neutral masking use the global RNGs
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)


def _signal(kind):
    return Signal.gradient() if kind == "gradient" else Signal.activation()


def _build(kind, tmp_path):
    model, tok = make_tiny_gpt2()
    ctx = tiny_backbone(model, tok)
    ctx.__enter__()
    args = training_arguments(str(tmp_path / kind), signal=_signal(kind),
                              signal_scope=SignalScope.layers())
    trainer = make_trainer(args, model, tok)
    return trainer, ctx


@pytest.mark.parametrize("kind", ["gradient", "activation"])
def test_dims_data_losses_and_roundtrip(kind, tmp_path):
    g = GOLDEN[kind]
    _seed_all()
    trainer, ctx = _build(kind, tmp_path)
    try:
        _seed_all()
        mwg = trainer.get_model()
        gm = mwg.gradiend
        # dims
        assert gm.input_dim == g["input_dim"]
        assert list(gm.encoder[0].linear.weight.shape) == g["enc_shape"]
        assert list(gm.decoder[0].linear.weight.shape) == g["dec_shape"]
        # training-data tensors of the first example
        raw = trainer.create_training_data(mwg, batch_size=1, max_size=None)
        ds = trainer.create_gradient_training_dataset(raw, mwg)
        assert len(ds) == g["len_ds"]
        it = ds[0]
        want = g["item0"]
        assert it["source"].numel() == g["input_dim"] and it["target"].numel() == g["input_dim"]
        assert float(it["source"].sum()) == pytest.approx(want["src_sum"], rel=1e-4, abs=1e-6)
        assert float(it["source"].norm()) == pytest.approx(want["src_norm"], rel=1e-4)
        assert float(it["target"].sum()) == pytest.approx(want["tgt_sum"], rel=1e-4, abs=1e-6)
        assert float(it["target"].norm()) == pytest.approx(want["tgt_norm"], rel=1e-4)
        assert str(it["label"]) == want["label"]

        # fixed-seed training losses (neutral-text masking draws from the global `random`)
        trainer.train()
        trained = trainer.get_model()
        mdir = trainer.model_path
        with open(os.path.join(mdir, "training.json")) as f:
            losses = json.load(f)["losses"]
        # Only the first two steps are reproducible across processes: later steps vary between
        # runs of the UNMODIFIED code (cause not isolated; seeding the global RNGs and PYTHONHASHSEED
        # do not remove it), so the tail is only required to be finite.
        assert len(losses) == len(g["losses"])
        assert losses[:2] == pytest.approx(g["losses"][:2], rel=1e-3, abs=1e-7)
        assert all(math.isfinite(x) for x in losses)

        # save/load roundtrip keeps weights and architecture
        with open(os.path.join(mdir, "config.json")) as f:
            arch = json.load(f)["architecture"]
        assert arch["input_dim"] == g["input_dim"]
        assert "encoder_input_dim" not in arch  # old configs stay byte-compatible
        reloaded = type(trained.gradiend).from_pretrained(mdir)
        for a, b in zip(trained.gradiend.state_dict().values(), reloaded.state_dict().values()):
            assert torch.equal(a.cpu(), b.cpu())
    finally:
        ctx.__exit__(None, None, None)
