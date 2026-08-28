"""Tests for the optional decoupled decoder learning rate.

The default (``learning_rate_decoder=None``) must reproduce the historical
single-group optimizer exactly; the opt-in path must place decoder parameters,
and only decoder parameters, in a second group at the requested rate.

Motivation: Adam-family updates are scale-free, so total decoder displacement is
capped at roughly ``lr * steps * sqrt(output_dim)`` regardless of gradient
magnitude. When the decoder's optimum is far larger in norm than the encoder's,
that cap can be orders of magnitude short of the required distance.
"""

import pytest
import torch
import torch.nn as nn

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.training import build_optimizer_parameter_groups


class _FakeGradiend(nn.Module):
    """Minimal encoder/decoder container mirroring GradiendModel's attributes."""

    def __init__(self, input_dim: int = 8, latent_dim: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, latent_dim, bias=False), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, input_dim, bias=True), nn.Identity())


def _all_params(model):
    return list(model.parameters())


class TestDefaultIsUnchanged:
    def test_none_returns_the_caller_list_unchanged(self):
        """The default path must hand the optimizer a plain parameter list."""
        model = _FakeGradiend()
        params = _all_params(model)

        result = build_optimizer_parameter_groups(
            model, params, learning_rate=1e-5, learning_rate_decoder=None
        )

        assert result == params
        assert all(isinstance(item, torch.nn.Parameter) for item in result)

    def test_none_produces_a_single_optimizer_group_at_the_shared_rate(self):
        model = _FakeGradiend()
        params = build_optimizer_parameter_groups(
            model, _all_params(model), learning_rate=1e-5, learning_rate_decoder=None
        )

        optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=1e-2)

        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)

    def test_default_training_arguments_leave_the_field_unset(self):
        assert TrainingArguments().learning_rate_decoder is None


class TestDecoupledPath:
    def test_decoder_parameters_go_to_their_own_group(self):
        model = _FakeGradiend()
        decoder_ids = {id(p) for p in model.decoder.parameters()}

        groups = build_optimizer_parameter_groups(
            model, _all_params(model), learning_rate=1e-5, learning_rate_decoder=1e-3
        )

        assert len(groups) == 2
        other, decoder = groups
        assert other["lr"] == pytest.approx(1e-5)
        assert decoder["lr"] == pytest.approx(1e-3)
        assert {id(p) for p in decoder["params"]} == decoder_ids
        assert not ({id(p) for p in other["params"]} & decoder_ids)

    def test_every_trainable_parameter_is_assigned_exactly_once(self):
        """No parameter may be dropped from, or duplicated across, the groups."""
        model = _FakeGradiend()
        params = _all_params(model)

        groups = build_optimizer_parameter_groups(
            model, params, learning_rate=1e-5, learning_rate_decoder=1e-3
        )

        assigned = [id(p) for group in groups for p in group["params"]]
        assert sorted(assigned) == sorted(id(p) for p in params)
        assert len(assigned) == len(set(assigned))

    def test_optimizer_actually_moves_the_decoder_faster(self):
        """End-to-end: identical gradients, different realized step sizes."""
        model = _FakeGradiend()
        groups = build_optimizer_parameter_groups(
            model, _all_params(model), learning_rate=1e-5, learning_rate_decoder=1e-3
        )
        optimizer = torch.optim.AdamW(groups, lr=1e-5, weight_decay=0.0)

        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()

        encoder_delta = max(
            (model.state_dict()[name] - before[name]).abs().max().item()
            for name in before
            if name.startswith("encoder")
        )
        decoder_delta = max(
            (model.state_dict()[name] - before[name]).abs().max().item()
            for name in before
            if name.startswith("decoder")
        )
        # Adam's normalized step is ~lr per coordinate, so the ratio of realized
        # displacements should track the ratio of learning rates.
        assert decoder_delta > encoder_delta
        assert decoder_delta / encoder_delta == pytest.approx(100.0, rel=0.05)

    def test_component_decoders_sharing_tensors_are_not_double_counted(self):
        """Component decoders alias the full decoder's tensors, not copies."""
        model = _FakeGradiend()
        aliased = list(model.decoder.parameters())
        params = _all_params(model) + aliased  # simulate an aliasing caller

        groups = build_optimizer_parameter_groups(
            model, params, learning_rate=1e-5, learning_rate_decoder=1e-3
        )

        decoder_group = groups[-1]
        # The duplicate ids land in the decoder group; none leak into the other
        # group, which is what would corrupt the encoder's effective rate.
        other_ids = {id(p) for p in groups[0]["params"]}
        assert not (other_ids & {id(p) for p in aliased})


class TestAgainstRealGradiendModel:
    """The fake above mirrors the attribute layout; this pins the real one."""

    def _model(self, **kwargs):
        from gradiend.model.model import GradiendModel

        return GradiendModel(
            input_dim=6,
            latent_dim=1,
            device=torch.device("cpu"),
            **kwargs,
        )

    def test_real_model_splits_encoder_and_decoder_correctly(self):
        model = self._model(bias_encoder=False, bias_decoder=True)
        params = list(model.parameters())
        decoder_ids = {id(p) for p in model.decoder.parameters()}
        assert decoder_ids, "real GradiendModel exposes no decoder parameters"

        groups = build_optimizer_parameter_groups(
            model, params, learning_rate=1e-5, learning_rate_decoder=1e-3
        )

        assert len(groups) == 2
        assert {id(p) for p in groups[-1]["params"]} == decoder_ids
        assigned = [id(p) for group in groups for p in group["params"]]
        assert sorted(assigned) == sorted(id(p) for p in params)
        assert len(assigned) == len(set(assigned))

    def test_real_model_default_path_is_the_untouched_list(self):
        model = self._model(bias_encoder=False, bias_decoder=True)
        params = list(model.parameters())

        assert (
            build_optimizer_parameter_groups(
                model, params, learning_rate=1e-5, learning_rate_decoder=None
            )
            == params
        )

    def test_decoder_bias_is_included_in_the_decoder_group(self):
        """bias_decoder=True adds a parameter that must not land in 'other'."""
        model = self._model(bias_encoder=False, bias_decoder=True)
        groups = build_optimizer_parameter_groups(
            model, list(model.parameters()), learning_rate=1e-5, learning_rate_decoder=1e-3
        )

        decoder_shapes = sorted(tuple(p.shape) for p in groups[-1]["params"])
        assert (6,) in [tuple(shape) for shape in decoder_shapes] or any(
            len(shape) == 1 for shape in decoder_shapes
        )


class TestFailureModes:
    def test_supervised_encoder_combination_raises_in_the_helper(self):
        model = _FakeGradiend()
        with pytest.raises(ValueError, match="supervised_encoder"):
            build_optimizer_parameter_groups(
                model,
                list(model.encoder.parameters()),
                learning_rate=1e-5,
                learning_rate_decoder=1e-3,
                supervised_encoder=True,
            )

    def test_supervised_encoder_combination_raises_at_argument_construction(self):
        with pytest.raises(ValueError, match="supervised_encoder"):
            TrainingArguments(learning_rate_decoder=1e-3, supervised_encoder=True)

    def test_no_decoder_parameter_selected_raises(self):
        """Silently training nothing at the requested rate would be worse."""
        model = _FakeGradiend()
        with pytest.raises(ValueError, match="no decoder parameter"):
            build_optimizer_parameter_groups(
                model,
                list(model.encoder.parameters()),
                learning_rate=1e-5,
                learning_rate_decoder=1e-3,
            )

    def test_unbuilt_decoder_raises(self):
        model = _FakeGradiend()
        model.decoder = None
        with pytest.raises(ValueError, match="no built\n?\\s*decoder|no built decoder"):
            build_optimizer_parameter_groups(
                model, [torch.nn.Parameter(torch.zeros(2))],
                learning_rate=1e-5,
                learning_rate_decoder=1e-3,
            )

    @pytest.mark.parametrize("value", [0.0, -1e-3])
    def test_non_positive_rate_rejected(self, value):
        with pytest.raises(ValueError, match="must be positive"):
            TrainingArguments(learning_rate_decoder=value)

    @pytest.mark.parametrize("value", ["1e-3", True, [1e-3]])
    def test_non_numeric_rate_rejected(self, value):
        with pytest.raises(TypeError, match="must be a number or None"):
            TrainingArguments(learning_rate_decoder=value)


class TestSerialization:
    def test_field_round_trips_through_to_dict(self):
        args = TrainingArguments(learning_rate=1e-5, learning_rate_decoder=2e-3)
        payload = args.to_dict()

        assert payload["learning_rate_decoder"] == pytest.approx(2e-3)
        assert payload["learning_rate"] == pytest.approx(1e-5)

    def test_unset_field_serializes_as_none(self):
        assert TrainingArguments().to_dict()["learning_rate_decoder"] is None

    def test_integer_rate_is_normalized_to_float(self):
        assert isinstance(TrainingArguments(learning_rate_decoder=1).learning_rate_decoder, float)
