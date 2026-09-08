"""Per-step gradient norms for the decoder and encoder.

Both Adam and SGD converge the encoder and neither converges the decoder on the
same objective, which rules out an optimizer-side explanation for the ~1%
shortfall. The remaining candidate is the objective's own conditioning: if the
loss is far less sensitive to decoder scale than to encoder direction, the
decoder's gradient is smaller in proportion and no shared learning rate closes
the distance. Weight norms cannot show that; gradient norms can.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from gradiend.trainer.core.training import (
    component_gradient_norms,
    split_encoder_decoder_parameters,
)


class _Gradiend(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 1, bias=False)
        self.decoder = nn.Linear(1, 4, bias=False)


def _with_grads(scale_enc=1.0, scale_dec=1.0):
    g = _Gradiend()
    params = list(g.parameters())
    g.encoder.weight.grad = torch.full_like(g.encoder.weight, scale_enc)
    g.decoder.weight.grad = torch.full_like(g.decoder.weight, scale_dec)
    return g, params


class TestSplitMatchesTheOptimizer:
    def test_decoder_and_other_are_disjoint_and_complete(self):
        g, params = _with_grads()
        dec, other = split_encoder_decoder_parameters(g, params)
        assert len(dec) + len(other) == len(params)
        assert not ({id(p) for p in dec} & {id(p) for p in other})

    def test_decoder_params_are_the_decoders(self):
        g, params = _with_grads()
        dec, _ = split_encoder_decoder_parameters(g, params)
        assert [id(p) for p in dec] == [id(g.decoder.weight)]

    def test_missing_decoder_puts_everything_in_other(self):
        module = nn.Linear(2, 2)
        dec, other = split_encoder_decoder_parameters(module, list(module.parameters()))
        assert dec == []
        assert len(other) == len(list(module.parameters()))


class TestNorms:
    def test_norms_are_l2_over_the_group(self):
        g, params = _with_grads(scale_enc=1.0, scale_dec=1.0)
        out = component_gradient_norms(g, params)
        # encoder grad is 4 ones -> 2.0; decoder grad is 4 ones -> 2.0
        assert out["encoder"] == pytest.approx(2.0)
        assert out["decoder"] == pytest.approx(2.0)
        assert out["decoder_over_encoder"] == pytest.approx(1.0)

    def test_ratio_detects_a_starved_decoder(self):
        """The hypothesis under test: decoder gradient orders of magnitude smaller."""
        g, params = _with_grads(scale_enc=1.0, scale_dec=1e-3)
        out = component_gradient_norms(g, params)
        assert out["decoder_over_encoder"] == pytest.approx(1e-3, rel=1e-6)

    def test_counts_are_reported_so_zero_is_unambiguous(self):
        g, params = _with_grads()
        out = component_gradient_norms(g, params)
        assert out["decoder_n"] == 1
        assert out["encoder_n"] == 1

    def test_params_without_grad_are_skipped_not_counted_as_zero(self):
        g, params = _with_grads()
        g.decoder.weight.grad = None
        out = component_gradient_norms(g, params)
        assert out["decoder_n"] == 0
        assert out["decoder"] == 0.0

    def test_zero_encoder_gradient_gives_nan_ratio_not_a_crash(self):
        g, params = _with_grads(scale_enc=0.0, scale_dec=1.0)
        out = component_gradient_norms(g, params)
        assert out["encoder"] == pytest.approx(0.0)
        import math

        assert math.isnan(out["decoder_over_encoder"])


class TestWiredIntoTraining:
    def test_capture_is_between_backward_and_step(self):
        """step() does not clear grads, but the next zero_grad() does."""
        from pathlib import Path

        import gradiend.trainer.core.training as t

        src = Path(t.__file__).read_text(encoding="utf-8")
        back = src.index("loss.backward()")
        cap = src.index("component_gradient_norms(", back)
        step = src.index("optimizer.step()", back)
        assert back < cap < step

    def test_stats_buckets_exist(self):
        from pathlib import Path

        import gradiend.trainer.core.training as t

        src = Path(t.__file__).read_text(encoding="utf-8")
        for key in ("encoder_grad_norms", "decoder_grad_norms",
                    "decoder_over_encoder_grad_ratio"):
            assert f"'{key}': {{}}" in src, key
            assert f"training_stats['{key}']" in src, key

    def test_diagnostic_failure_cannot_stop_training(self):
        from pathlib import Path

        import gradiend.trainer.core.training as t

        src = Path(t.__file__).read_text(encoding="utf-8")
        block = src[src.index("_grad_norms = component_gradient_norms") - 200 :][:1100]
        assert "except Exception as _grad_exc" in block
        assert "_grad_norms = None" in block
        # An always-failing diagnostic leaves the key present but empty, which
        # reads as "no data" rather than "broken" -- so it must say why once.
        assert "gradient-norm diagnostic disabled after error" in block
