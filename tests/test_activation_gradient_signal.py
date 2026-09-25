"""Edge-case + correctness tests for the ``activation_gradient`` signal (dL/dh).

The activation_gradient signal captures ``dL/dh`` at each residual site rather
than the activation value ``h``. In a causal LM the target token at position
``p`` is predicted from position ``p-1`` (HF shifts logits/labels by one), so
``dL/dh`` is identically zero at the target span itself and lives at the
"generating" positions (the target span shifted one token left). These tests
pin:

* ``_generating_positions_mask`` (single-token, multi-token span, position-0
  edge, mixed batch, dtype/shape invariance);
* that the captured gradient really is zero at the target span and nonzero at
  the generating position (the reason the selector shifts);
* finite-difference correctness of the captured ``dL/dh`` at a generating
  position, for both single- and multi-token targets;
* multi-site sequential connectivity (every hooked site gets a real gradient,
  not a zeros-fallback);
* frozen base-model params stay frozen after capture (requires_grad restored);
* the factual vs alternative contrast is nonzero at the generating position
  even though the *value* there is identical (same context) -- the whole point
  of CAGA;
* the value ``activation`` signal is byte-for-byte unchanged by all of this.

The fixture is a *pointwise* causal LM (no cross-position mixing): position
``i``'s hidden state depends only on ``input_ids[i]`` and its logits are scored
against ``labels[i+1]``. That reproduces the exact causal-LM gradient position
structure (target zero / generating nonzero) with no attention, so the
assertions are exact rather than approximate.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from gradiend.model.model_with_gradiend import ModelWithGradiend
from gradiend.trainer.core.signals import (
    ActivationSignalExtractor,
    Signal,
    SignalScope,
)


def test_activation_gradient_model_uses_package_activation_intervention_path(monkeypatch):
    class _Concrete(ModelWithGradiend):
        def create_gradients(self, *args, **kwargs):
            raise NotImplementedError

        def _load_model(self, *args, **kwargs):
            raise NotImplementedError

        def _save_model(self, *args, **kwargs):
            raise NotImplementedError

        def _activation_intervention_specs(self, update, **kwargs):
            self.seen_update = update.detach().clone()
            return ([{"module": "block1", "application": {}}], {})

        def _intervention_update_vector(self, **kwargs):
            return torch.tensor([0.5, 0.5, 0.5, 0.5])

        def _apply_gradient_intervention_delta(self, *args, **kwargs):
            raise AssertionError("activation_gradient must not rewrite weights")

    wrapped = _Concrete.__new__(_Concrete)
    nn.Module.__init__(wrapped)
    wrapped.gradiend = SimpleNamespace(
        signal_kind="activation_gradient",
        param_map={"activation:block1": {"shape": (4,), "repr": "all"}},
        input_dim=4,
    )
    wrapped.base_model = nn.Linear(4, 4)
    wrapped._source = "factual"

    assert wrapped.uses_activation_gradients is True
    assert wrapped.uses_activation_space is True
    assert wrapped._intervention_signal_kind("auto") == "activation"
    assert wrapped.capabilities.activation_interventions is True
    assert wrapped.activation_site_modules == ["block1"]

    monkeypatch.setattr(
        "gradiend.model.model_with_gradiend.register_activation_steering_hooks",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "gradiend.model.model_with_gradiend.apply_activation_steering",
        lambda model, **kwargs: model,
    )
    random_update = torch.tensor([1.0, -2.0, 3.0, -4.0])
    with wrapped.intervene(
        value=1.0,
        feature_factor=1.0,
        update_vector=random_update,
        token_selector="all",
    ) as info:
        assert info["signal"] == "activation"
        assert torch.equal(wrapped.seen_update, random_update)

    modified = wrapped.modify_model(
        learning_rate=1.0,
        feature_factor=1.0,
        token_selector="all",
    )
    assert modified is not wrapped.base_model
    assert torch.equal(wrapped.seen_update, torch.full((4,), 0.5))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
class _Out:
    def __init__(self, logits, loss=None):
        self.logits = logits
        self.loss = loss


class MiniCausalLM(nn.Module):
    """Pointwise causal LM: two residual blocks, HF-style shifted CE loss.

    No attention / cross-position mixing, so ``dL/dh_i`` depends only on the
    loss term scoring ``labels[i+1]`` -- exactly the causal-LM position
    structure the extractor's generating-position shift is built for.
    """

    def __init__(self, vocab=12, dim=4, seed=0):
        super().__init__()
        self.config = type("Cfg", (), {"model_type": "gpt2"})()
        gen = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(vocab, dim)
        self.block0 = nn.Linear(dim, dim)
        self.block1 = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, vocab)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.rand(p.shape, generator=gen) - 0.5)

    def forward(self, input_ids, attention_mask=None, labels=None):
        h = self.emb(input_ids)
        h = h + torch.tanh(self.block0(h))
        h = h + torch.tanh(self.block1(h))
        logits = self.head(h)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return _Out(logits, loss)


def _grad_extractor(model, sites=("block1",), selector="prediction"):
    return ActivationSignalExtractor(
        model,
        signal=Signal.activation_gradient(token_selector=selector),
        scope=SignalScope.from_values(activation_sites=list(sites)),
    )


def _value_extractor(model, sites=("block1",), selector="prediction"):
    return ActivationSignalExtractor(
        model,
        signal=Signal.activation(token_selector=selector),
        scope=SignalScope.from_values(activation_sites=list(sites)),
    )


# --------------------------------------------------------------------------- #
# _generating_positions_mask -- pure tensor edge cases
# --------------------------------------------------------------------------- #
def _gen(mask):
    return ActivationSignalExtractor._generating_positions_mask(mask)


def test_generating_mask_single_token_target_shifts_left_by_one():
    mask = torch.tensor([[False, False, True, False]])
    assert torch.equal(_gen(mask), torch.tensor([[False, True, False, False]]))


def test_generating_mask_multi_token_span_shifts_whole_span_left():
    # target span at [2,3] -> generating positions [1,2]
    mask = torch.tensor([[False, False, True, True, False]])
    assert torch.equal(
        _gen(mask), torch.tensor([[False, True, True, False, False]])
    )


def test_generating_mask_target_at_position_zero_is_dropped():
    # position 0 has no predicting context; its generating position would be -1
    mask = torch.tensor([[True, False, False]])
    assert torch.equal(_gen(mask), torch.tensor([[False, False, False]]))


def test_generating_mask_target_at_position_zero_keeps_rest_of_span():
    # span [0,1]: pos 0 dropped, pos 1 -> generating pos 0
    mask = torch.tensor([[True, True, False]])
    assert torch.equal(_gen(mask), torch.tensor([[True, False, False]]))


def test_generating_mask_mixed_batch_rows_independent():
    mask = torch.tensor(
        [
            [False, True, False, False],  # -> pos 0
            [False, False, False, True],  # -> pos 2
            [True, False, False, False],  # -> dropped
        ]
    )
    expected = torch.tensor(
        [
            [True, False, False, False],
            [False, False, True, False],
            [False, False, False, False],
        ]
    )
    assert torch.equal(_gen(mask), expected)


def test_generating_mask_accepts_non_bool_dtype_and_returns_bool():
    mask = torch.tensor([[0, 0, 1, 0]], dtype=torch.long)
    out = _gen(mask)
    assert out.dtype == torch.bool
    assert torch.equal(out, torch.tensor([[False, True, False, False]]))


def test_generating_mask_does_not_mutate_input():
    mask = torch.tensor([[False, True, False]])
    before = mask.clone()
    _gen(mask)
    assert torch.equal(mask, before)


def test_generating_mask_last_position_target_maps_into_range():
    mask = torch.tensor([[False, False, False, True]])
    assert torch.equal(_gen(mask), torch.tensor([[False, False, True, False]]))


# --------------------------------------------------------------------------- #
# Position structure: gradient zero at target, nonzero at generating position
# --------------------------------------------------------------------------- #
def _capture_grad_dict(model, input_ids, prediction_mask, sites=("block1",)):
    ext = _grad_extractor(model, sites=sites)
    captured, _prepared = ext._capture_gradient_signal(
        {"input_ids": input_ids, "prediction_mask": prediction_mask}
    )
    return captured


def test_gradient_is_zero_at_target_and_nonzero_at_generating_position():
    torch.manual_seed(1)
    model = MiniCausalLM()
    input_ids = torch.tensor([[3, 5, 7, 2]])
    # target at position 2 -> generating position 1
    pred = torch.tensor([[False, False, True, False]])
    g = _capture_grad_dict(model, input_ids, pred)["block1"]  # (1, seq, dim)

    target_norm = g[0, 2].norm().item()
    gen_norm = g[0, 1].norm().item()
    assert target_norm == pytest.approx(0.0, abs=1e-7)
    assert gen_norm > 1e-4


def test_gradient_multi_token_target_zero_on_span_nonzero_on_generating_span():
    torch.manual_seed(2)
    model = MiniCausalLM()
    input_ids = torch.tensor([[3, 5, 7, 9, 2]])
    # target span [2,3] -> generating positions [1,2] (span shifted one left).
    # Note position 2 is BOTH a target and a generating position (it predicts
    # target token 3), so its gradient is nonzero; only the span's LAST target
    # position (3), which generates a non-target, is exactly zero.
    pred = torch.tensor([[False, False, True, True, False]])
    g = _capture_grad_dict(model, input_ids, pred)["block1"]

    # generating positions [1, 2] carry gradient
    assert g[0, 1].norm().item() > 1e-4
    assert g[0, 2].norm().item() > 1e-4
    # position 3 (target-span end) predicts a non-target -> exactly zero
    assert g[0, 3].norm().item() == pytest.approx(0.0, abs=1e-7)
    # position 0 (context, predicts non-target position 1) -> zero
    assert g[0, 0].norm().item() == pytest.approx(0.0, abs=1e-7)


# --------------------------------------------------------------------------- #
# Finite-difference correctness of captured dL/dh at a generating position
# --------------------------------------------------------------------------- #
def _loss_with_site_perturbation(model, input_ids, labels, site, pos, delta):
    """Loss after adding ``delta`` to ``site``'s output at seq index ``pos``."""
    module = dict(model.named_modules())[site]

    def hook(_m, _a, output):
        out = output[0] if isinstance(output, tuple) else output
        out = out.clone()
        out[:, pos, :] = out[:, pos, :] + delta
        if isinstance(output, tuple):
            return (out,) + tuple(output[1:])
        return out

    handle = module.register_forward_hook(hook)
    try:
        loss = model(input_ids=input_ids, labels=labels).loss
    finally:
        handle.remove()
    return float(loss.item())


@pytest.mark.parametrize(
    "pred_mask, target_pos, gen_pos",
    [
        (torch.tensor([[False, False, True, False]]), 2, 1),  # single token
        (torch.tensor([[False, True, True, False]]), None, 0),  # span [1,2] gen [0,1]
    ],
)
def test_finite_difference_matches_captured_gradient(pred_mask, target_pos, gen_pos):
    torch.manual_seed(3)
    model = MiniCausalLM()
    for p in model.parameters():
        p.requires_grad_(False)
    input_ids = torch.tensor([[3, 5, 7, 2]])

    captured = _capture_grad_dict(model, input_ids, pred_mask)["block1"]
    g = captured[0, gen_pos]  # dL/dh at the generating position

    # labels the extractor would have built: target ids where predicted, -100 else
    labels = input_ids.masked_fill(~pred_mask.bool(), -100)

    torch.manual_seed(99)
    v = torch.randn(g.shape)
    eps = 1e-3
    lp = _loss_with_site_perturbation(model, input_ids, labels, "block1", gen_pos, eps * v)
    lm = _loss_with_site_perturbation(model, input_ids, labels, "block1", gen_pos, -eps * v)
    fd = (lp - lm) / (2 * eps)
    analytic = torch.dot(g, v).item()

    assert analytic == pytest.approx(fd, rel=2e-2, abs=1e-5)


# --------------------------------------------------------------------------- #
# Multi-site sequential connectivity + frozen params restored
# --------------------------------------------------------------------------- #
def test_multiple_sequential_sites_all_receive_real_gradients():
    torch.manual_seed(4)
    model = MiniCausalLM()
    input_ids = torch.tensor([[3, 5, 7, 2]])
    pred = torch.tensor([[False, False, True, False]])
    captured = _capture_grad_dict(model, input_ids, pred, sites=("block0", "block1"))

    # both sites present, both nonzero at the generating position (no site got
    # the zeros_like fallback from allow_unused)
    assert set(captured) == {"block0", "block1"}
    assert captured["block0"][0, 1].norm().item() > 1e-5
    assert captured["block1"][0, 1].norm().item() > 1e-5


def test_frozen_params_stay_frozen_after_capture():
    model = MiniCausalLM()
    for p in model.parameters():
        p.requires_grad_(False)
    before = [p.requires_grad for p in model.parameters()]

    _capture_grad_dict(
        model,
        torch.tensor([[3, 5, 7, 2]]),
        torch.tensor([[False, False, True, False]]),
    )

    after = [p.requires_grad for p in model.parameters()]
    assert before == after == [False] * len(before)
    # and no p.grad side effects were left behind
    assert all(p.grad is None for p in model.parameters())


def test_mixed_requires_grad_flags_are_restored_exactly():
    model = MiniCausalLM()
    params = list(model.parameters())
    pattern = [i % 2 == 0 for i in range(len(params))]
    for p, flag in zip(params, pattern):
        p.requires_grad_(flag)

    _capture_grad_dict(
        model,
        torch.tensor([[3, 5, 7, 2]]),
        torch.tensor([[False, False, True, False]]),
    )

    assert [p.requires_grad for p in model.parameters()] == pattern


# --------------------------------------------------------------------------- #
# The CAGA contrast: F vs A differ at the generating position
# --------------------------------------------------------------------------- #
def test_factual_vs_alternative_gradient_contrast_is_nonzero():
    torch.manual_seed(5)
    model = MiniCausalLM()
    ext = _grad_extractor(model)
    # same context, different filled target token at position 2
    fac = {
        "input_ids": torch.tensor([[3, 5, 7, 2]]),
        "prediction_mask": torch.tensor([[False, False, True, False]]),
    }
    alt = {
        "input_ids": torch.tensor([[3, 5, 8, 2]]),
        "prediction_mask": torch.tensor([[False, False, True, False]]),
    }
    batch = ext(factual_inputs=fac, alternative_inputs=alt)
    assert batch.diff is not None
    assert batch.diff.norm().item() > 1e-4


def test_value_at_generating_position_is_identical_but_gradient_differs():
    """Justifies capturing the GRADIENT: value@gen is context-only (identical
    for F/A), so a value signal there is degenerate while the gradient is not."""
    torch.manual_seed(6)
    model = MiniCausalLM()
    fac_ids = torch.tensor([[3, 5, 7, 2]])
    alt_ids = torch.tensor([[3, 5, 8, 2]])  # differs only at target pos 2
    pred = torch.tensor([[False, False, True, False]])

    # Value signal at the generating position (pos 1): identical context -> equal
    vext = _value_extractor(model)
    gen_mask = torch.tensor([[False, True, False, False]])
    vf = vext._extract({"input_ids": fac_ids, "prediction_mask": gen_mask})
    va = vext._extract({"input_ids": alt_ids, "prediction_mask": gen_mask})
    assert torch.allclose(vf, va, atol=1e-6)

    # Gradient signal (auto-shifts to gen pos): differs
    gext = _grad_extractor(model)
    gf = gext._extract({"input_ids": fac_ids, "prediction_mask": pred})
    ga = gext._extract({"input_ids": alt_ids, "prediction_mask": pred})
    assert not torch.allclose(gf, ga, atol=1e-4)


# --------------------------------------------------------------------------- #
# Regression: the value activation signal is unchanged by the gradient plumbing
# --------------------------------------------------------------------------- #
def test_value_prediction_selector_unchanged_no_shift():
    """A value signal with token_selector='prediction' still reads the target
    span itself (NOT shifted) -- the shift is gradient-only."""
    torch.manual_seed(7)
    model = MiniCausalLM()
    input_ids = torch.tensor([[3, 5, 7, 2]])
    pred = torch.tensor([[False, False, True, False]])

    vext = _value_extractor(model)
    got = vext._extract({"input_ids": input_ids, "prediction_mask": pred})

    # ground truth: forward, take block1 residual output at the target position 2
    captured = {}
    h = model.block1.register_forward_hook(
        lambda _m, _a, out: captured.__setitem__("h", out)
    )
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        h.remove()
    expected = captured["h"][:, 2, :].reshape(1, -1)
    assert torch.allclose(got, expected, atol=1e-6)


def test_value_signal_capture_takes_no_grad_path():
    """The value path must not enable grad or leave grad state on params."""
    model = MiniCausalLM()
    for p in model.parameters():
        p.requires_grad_(False)
    vext = _value_extractor(model)
    out = vext._extract(
        {
            "input_ids": torch.tensor([[3, 5, 7, 2]]),
            "prediction_mask": torch.tensor([[False, False, True, False]]),
        }
    )
    assert not out.requires_grad
    assert all(not p.requires_grad for p in model.parameters())


# --------------------------------------------------------------------------- #
# Error surfaces (fail-fast)
# --------------------------------------------------------------------------- #
def test_gradient_capture_requires_loss_from_model():
    class NoLossLM(MiniCausalLM):
        def forward(self, input_ids, attention_mask=None, labels=None):
            out = super().forward(input_ids)  # ignore labels -> loss stays None
            return _Out(out.logits, None)

    ext = _grad_extractor(NoLossLM())
    with pytest.raises(ValueError, match="supervised loss"):
        ext._extract(
            {
                "input_ids": torch.tensor([[3, 5, 7, 2]]),
                "prediction_mask": torch.tensor([[False, False, True, False]]),
            }
        )


def test_gradient_prediction_selector_requires_prediction_mask():
    ext = _grad_extractor(MiniCausalLM())
    with pytest.raises(ValueError, match="prediction_mask"):
        ext._extract({"input_ids": torch.tensor([[3, 5, 7, 2]])})


def test_gradient_tolerates_target_at_position_zero_no_crash():
    # A target at position 0 has no predicting context, so its generating mask is
    # all-empty. The gradient 'prediction' selector must NOT crash the batch (it used
    # to raise "requires at least one selected token in every row", which killed real
    # tasks on slurm) -- that row yields a zero dL/dh, other rows are unaffected.
    torch.manual_seed(11)
    model = MiniCausalLM()
    input_ids = torch.tensor([[3, 5, 7, 2], [4, 6, 8, 9]])
    # row 0: target at position 0 (empty generating); row 1: target at pos 2 -> gen pos 1
    pred = torch.tensor([[True, False, False, False], [False, False, True, False]])
    ext = _grad_extractor(model)
    out = ext._extract({"input_ids": input_ids, "prediction_mask": pred})  # must not raise
    assert torch.isfinite(out).all()
    # And the per-row captured grad: row 0 (pos-0 target) has zero generating signal.
    captured = _capture_grad_dict(model, input_ids, pred)["block1"]
    # row 0 generating positions all empty -> its selected mean would be zero; row 1
    # has a real generating position (pos 1).
    assert captured[1, 1].norm().item() > 1e-5
