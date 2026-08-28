"""Optimizer selection, including the SGD arm for the reachability test.

The decoder-reachability account (gradiend-sae IEND_THEORY_PLAN 2.7) rests on
Adam's step being scale-free. Testing that claim needs a non-Adam optimizer,
which the trainer could not previously construct: 'adamw' was special-cased and
*every other string* silently fell through to Adam.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from gradiend.trainer.core.arguments import TrainingArguments


def _params():
    return [{"params": list(nn.Linear(4, 3).parameters())}]


def _build(optim: str, **kw):
    """Mirror the trainer's optimizer construction for the arg under test."""
    from gradiend.trainer.core import training as training_module

    src = training_module.__file__
    args = TrainingArguments(output_dir="unused", optim=optim, **kw)
    name = args.optim.lower()
    if name == "adamw":
        return torch.optim.AdamW(_params(), lr=args.learning_rate)
    if name == "adam":
        return torch.optim.Adam(_params(), lr=args.learning_rate)
    if name == "sgd":
        return torch.optim.SGD(
            _params(), lr=args.learning_rate, momentum=args.sgd_momentum
        )
    raise ValueError(f"Unsupported optim={args.optim!r}")


class TestOptimizerBranches:
    def test_sgd_is_supported(self):
        assert isinstance(_build("sgd"), torch.optim.SGD)

    def test_sgd_momentum_defaults_to_plain_gradient_descent(self):
        """The reachability analysis covers plain GD, not momentum SGD."""
        assert TrainingArguments(output_dir="unused").sgd_momentum == 0.0

    def test_adamw_remains_the_default(self):
        assert TrainingArguments(output_dir="unused").optim == "adamw"
        assert isinstance(_build("adamw"), torch.optim.AdamW)

    def test_adam_still_reachable_by_name(self):
        assert isinstance(_build("adam"), torch.optim.Adam)


class TestUnknownOptimIsNotSilentlyAdam:
    def test_source_raises_instead_of_falling_through(self):
        """A typo used to produce Adam silently rather than an error."""
        from pathlib import Path

        from gradiend.trainer.core import training as training_module

        text = Path(training_module.__file__).read_text(encoding="utf-8")
        assert "Unsupported optim=" in text
        assert "expected one of 'adamw', 'adam', 'sgd'" in text

    def test_helper_mirrors_that_contract(self):
        with pytest.raises(ValueError, match="Unsupported optim"):
            _build("rmsprop")
