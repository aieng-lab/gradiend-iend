"""Native hybrid IEND: activation-valued encoder input, gradient decoder target (input_signal)."""
import json
import os

import pytest
import torch

from gradiend import Signal, SignalScope
from gradiend.trainer.core.arguments import TrainingArguments

from tests.tiny_gpt2_support import (
    HIDDEN,
    LAYERS,
    make_tiny_gpt2,
    make_trainer,
    tiny_backbone,
    tiny_gpt2_available,
    training_arguments,
)

pytestmark = pytest.mark.skipif(not tiny_gpt2_available(), reason="gpt2 tokenizer not cached")

ACT_DIM = HIDDEN * LAYERS  # SignalScope.layers(): one hidden state per layer
GRAD_DIM = 6560


def _hybrid_args(tmp_path, name="hybrid", **overrides):
    return training_arguments(
        str(tmp_path / name),
        signal=Signal.gradient(),
        signal_scope=SignalScope.layers(),
        input_signal=Signal.activation(),
        input_signal_scope=SignalScope.layers(),
        **overrides,
    )


@pytest.fixture
def tiny(tmp_path):
    model, tok = make_tiny_gpt2()
    ctx = tiny_backbone(model, tok)
    ctx.__enter__()
    try:
        yield model, tok, tmp_path
    finally:
        ctx.__exit__(None, None, None)


def test_arguments_roundtrip_and_validation():
    args = TrainingArguments(
        signal=Signal.gradient(),
        input_signal=Signal.activation(),
        input_signal_scope=SignalScope.layers(),
    )
    again = TrainingArguments.from_dict(args.to_dict())
    assert again.input_signal == args.input_signal
    assert again.input_signal_scope == args.input_signal_scope
    # absent (not null) when unused: every existing config stays byte-identical
    assert "input_signal" not in TrainingArguments().to_dict()
    assert "input_signal_scope" not in TrainingArguments().to_dict()
    with pytest.raises(ValueError):
        TrainingArguments(input_signal_scope=SignalScope.layers())
    with pytest.raises(NotImplementedError):
        TrainingArguments(signal=Signal.gradient(), input_signal=Signal.gradient())
    with pytest.raises(NotImplementedError):
        TrainingArguments(signal=Signal.activation(), input_signal=Signal.activation())


def test_dims_encoder_activation_decoder_gradient(tiny):
    model, tok, tmp_path = tiny
    trainer = make_trainer(_hybrid_args(tmp_path), model, tok)
    gm = trainer.get_model().gradiend
    assert gm.has_separate_encoder_input
    assert gm.encoder_input_dim == ACT_DIM
    assert gm.input_dim == GRAD_DIM
    assert list(gm.encoder[0].linear.weight.shape) == [1, ACT_DIM]
    assert list(gm.decoder[0].linear.weight.shape) == [GRAD_DIM, 1]
    # decoder scale follows its own (large) fan-in, not the small encoder's
    dec_bound = 1.0 / (max(10000, GRAD_DIM) ** 0.5)
    assert float(gm.decoder[0].linear.weight.abs().max()) <= dec_bound + 1e-9
    assert gm.kwargs["input_signal_space"]["kind"] == "activation"


def test_source_and_target_come_from_the_same_example(tiny):
    model, tok, tmp_path = tiny
    trainer = make_trainer(_hybrid_args(tmp_path, "h"), model, tok)
    mwg = trainer.get_model()
    raw = trainer.create_training_data(mwg, batch_size=1, max_size=None)
    ds = trainer.create_gradient_training_dataset(raw, mwg)
    extractor = ds.signal_extractor

    # Spy on the merged batch the dataset extracts from, then recompute source/target from
    # exactly that batch with the two independent extractors.
    merged = []
    original = ds._merge_batch

    def spy(indices):
        batch = original(indices)
        merged.append(batch)
        return batch

    ds._merge_batch = spy
    checked = 0
    for i in range(len(ds)):
        merged.clear()
        item = ds[i]
        if str(item["label"]) == "0":
            continue  # neutral identity rows have a zero target
        batch = merged[0]
        pf, pa = batch["factual"], batch["alternative"]
        if item["factual_token"] != batch["factual_token"]:
            pf, pa = pa, pf  # source="both" presented the other pole as factual
        # the encoder input is the activation of the FILLED example; the target is the gradient
        # of the same example (unfilled prefix + label)
        expected_source = extractor.input_extractor(factual_inputs=pf.input, requires_alternative=False).factual
        expected_target = (
            extractor.target_extractor(factual_inputs=pf.target, requires_alternative=False).factual
            - extractor.target_extractor(alternative_inputs=pa.target, requires_factual=False).alternative
        )
        torch.testing.assert_close(item["source"], expected_source.to(item["source"]), rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(item["target"], expected_target.to(item["target"]), rtol=1e-4, atol=1e-6)
        assert item["source"].numel() == ACT_DIM and item["target"].numel() == GRAD_DIM
        checked += 1
    assert checked >= 2


def test_training_runs_loss_decreases_and_roundtrips(tiny):
    model, tok, tmp_path = tiny
    args = _hybrid_args(tmp_path, max_steps=40, eval_steps=10, learning_rate=1e-2, learning_rate_decoder=1e-2)
    trainer = make_trainer(args, model, tok)
    trainer.train()
    mdir = trainer.model_path
    with open(os.path.join(mdir, "training.json")) as f:
        payload = json.load(f)
    losses = payload["losses"]
    assert all(torch.isfinite(torch.tensor(losses)))
    assert sum(losses[-10:]) < sum(losses[:10])
    assert payload["training_args"]["input_signal"]["kind"] == "activation"

    with open(os.path.join(mdir, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["architecture"]["encoder_input_dim"] == ACT_DIM
    assert cfg["architecture"]["input_dim"] == GRAD_DIM
    assert cfg["metadata"]["input_signal_space"]["kind"] == "activation"

    trained = trainer.get_model()
    reloaded = type(trained.gradiend).from_pretrained(mdir)
    assert reloaded.has_separate_encoder_input and reloaded.encoder_input_dim == ACT_DIM
    for a, b in zip(trained.gradiend.state_dict().values(), reloaded.state_dict().values()):
        assert torch.equal(a.cpu(), b.cpu())

    # encode a NEW text through the activation extractor
    text = "The nurse said that [MASK] would arrive soon."
    enc = trained.encode(text, label="she", return_float=True)
    assert isinstance(enc, float) and -1.0 <= enc <= 1.0
    assert trained.encode(text, label="she", return_float=True) == pytest.approx(enc)

    # rewrite_base_model: theta + lr * (W ff + b) in gradient space
    rewritten = trained.rewrite_base_model(learning_rate=1e-3, feature_factor=1.0)
    before = dict(trained.base_model.named_parameters())
    moved = sum(
        1
        for n, p in rewritten.named_parameters()
        if n in before and not torch.equal(p.detach().cpu(), before[n].detach().cpu())
    )
    assert moved > 0


def test_prune_acts_on_decoder_side_only(tiny):
    model, tok, tmp_path = tiny
    trainer = make_trainer(_hybrid_args(tmp_path), model, tok)
    gm = trainer.get_model().gradiend
    enc_before = gm.encoder[0].linear.weight.detach().clone()
    keep = torch.arange(0, GRAD_DIM, 2)
    pruned = gm.prune(keep_idx=keep, inplace=False)
    assert pruned.input_dim == keep.numel()
    assert pruned.encoder_input_dim == ACT_DIM
    assert torch.equal(pruned.encoder[0].linear.weight, enc_before)
    assert list(pruned.decoder[0].linear.weight.shape) == [keep.numel(), 1]
    with pytest.raises(ValueError):
        gm.get_weight_importance("encoder-weight")


def test_cache_fingerprint_separates_hybrid_from_plain_gradient():
    from gradiend.trainer.core.cache_policy import build_training_cache_fingerprint

    plain = TrainingArguments(signal=Signal.gradient())
    hybrid = TrainingArguments(
        signal=Signal.gradient(),
        input_signal=Signal.activation(),
        input_signal_scope=SignalScope.layers(),
    )
    fp_plain = build_training_cache_fingerprint(plain)
    fp_hybrid = build_training_cache_fingerprint(hybrid)
    assert "input_signal" not in fp_plain and "input_signal_scope" not in fp_plain
    assert "input_signal" in fp_hybrid and fp_hybrid != fp_plain
