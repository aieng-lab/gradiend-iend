"""Automatic decoder LR: numerical correctness and bounded memory state."""

from __future__ import annotations

import math
from contextlib import contextmanager

import pytest
import torch
from torch import nn

from gradiend.model import GradiendModel
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.decoder_lr import AutoDecoderLearningRate
from gradiend.trainer.core.stats import load_training_stats
from gradiend.trainer.core.training import train
from tests.testing_mocks import SimpleMockModel


def _controller(model, optimizer, *, aggregation="full"):
    return AutoDecoderLearningRate(
        gradiend=model,
        optimizer=optimizer,
        initial_lr=1e-12,
        aggregation=aggregation,
        criterion=nn.MSELoss(),
    )


def _one_step(model, optimizer, controller, source, target, *, aggregation="full"):
    optimizer.zero_grad()
    loss, encoded = model.reconstruction_loss(
        source, target, criterion=nn.MSELoss(), aggregation=aggregation, return_encoded=True
    )
    loss.backward()
    controller.observe(encoded)
    optimizer.step()


def test_distance_matches_explicit_linear_decoder_optimum():
    torch.manual_seed(4)
    model = GradiendModel(
        input_dim=7,
        latent_dim=2,
        bias_decoder=True,
        activation_decoder="id",
        device=torch.device("cpu"),
    )
    groups = [
        {"params": list(model.encoder.parameters()), "lr": 1e-12},
        {"params": list(model.decoder.parameters()), "lr": 1e-12},
    ]
    optimizer = torch.optim.AdamW(groups, lr=1e-12, weight_decay=0.0)
    controller = _controller(model, optimizer)

    source = torch.randn(12, 7)
    with torch.no_grad():
        encoded = model.encoder(source)
        target_weight = torch.randn(7, 2)
        target_bias = torch.randn(7)
        target = encoded @ target_weight.T + target_bias
        current_weight = model.decoder[0].linear.weight.detach().clone()
        current_bias = model.decoder[0].linear.bias.detach().clone()
        expected = math.sqrt(
            torch.sum((target_weight - current_weight) ** 2).item()
            + torch.sum((target_bias - current_bias) ** 2).item()
        )

    for _ in range(4):
        _one_step(model, optimizer, controller, source, target)
    result = controller.estimate(step=4, total_steps=40)

    assert result["status"] == "calibrated"
    assert result["decoder_distance_to_local_optimum"] == pytest.approx(expected, rel=2e-5)
    expected_floor = expected / (36 * math.sqrt(7 * 3))
    assert result["reachability_floor_lr"] == pytest.approx(expected_floor, rel=2e-5)
    assert optimizer.param_groups[-1]["lr"] == pytest.approx(expected_floor, rel=2e-5)


@pytest.mark.parametrize("aggregation", ["mean", "sum", "size_weighted"])
def test_component_split_distance_matches_blockwise_optimum(aggregation):
    torch.manual_seed(8)
    model = GradiendModel(
        input_dim=5,
        latent_dim=1,
        bias_decoder=True,
        activation_decoder="id",
        device=torch.device("cpu"),
        component_slices=[
            {"id": "left", "start": 0, "end": 2},
            {"id": "right", "start": 2, "end": 5},
        ],
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": 1e-12},
            {"params": list(model.decoder.parameters()), "lr": 1e-12},
        ],
        weight_decay=0.0,
    )
    controller = _controller(model, optimizer, aggregation=aggregation)
    source = torch.randn(16, 5)
    with torch.no_grad():
        encoded = model._encode_components(source)
        target = torch.empty(16, 5)
        target[:, :2] = encoded[:, 0, :] @ torch.tensor([[1.3, -0.7]]) + 0.2
        target[:, 2:] = encoded[:, 1, :] @ torch.tensor([[0.4, -1.1, 2.0]]) - 0.3
        target_weight = torch.tensor([[1.3], [-0.7], [0.4], [-1.1], [2.0]])
        target_bias = torch.tensor([0.2, 0.2, -0.3, -0.3, -0.3])
        expected = torch.linalg.vector_norm(
            torch.cat(
                (
                    (target_weight - model.decoder[0].linear.weight).reshape(-1),
                    target_bias - model.decoder[0].linear.bias,
                )
            )
        ).item()

    for _ in range(4):
        _one_step(
            model,
            optimizer,
            controller,
            source,
            target,
            aggregation=aggregation,
        )
    result = controller.estimate(step=4, total_steps=40)

    assert result["status"] == "calibrated"
    assert result["decoder_distance_to_local_optimum"] == pytest.approx(expected, rel=3e-5)


def test_rank_deficient_warmup_defers_without_changing_lr():
    model = GradiendModel(input_dim=5, latent_dim=1, device=torch.device("cpu"))
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": 1e-5},
            {"params": list(model.decoder.parameters()), "lr": 1e-5},
        ],
        weight_decay=0.0,
    )
    controller = AutoDecoderLearningRate(
        gradiend=model,
        optimizer=optimizer,
        initial_lr=1e-5,
        aggregation="full",
        criterion=nn.MSELoss(),
    )
    source = torch.ones(1, 5)
    target = torch.randn(1, 5)
    _one_step(model, optimizer, controller, source, target)
    result = controller.estimate(step=1, total_steps=10)

    assert result["status"] == "waiting_for_full_rank"
    assert optimizer.param_groups[-1]["lr"] == pytest.approx(1e-5)


def test_persistent_calibration_state_does_not_scale_with_decoder_width():
    model = GradiendModel(
        input_dim=200_000,
        latent_dim=1,
        bias_decoder=True,
        device=torch.device("cpu"),
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": 1e-5},
            {"params": list(model.decoder.parameters()), "lr": 1e-5},
        ],
        weight_decay=0.0,
    )
    controller = AutoDecoderLearningRate(
        gradiend=model,
        optimizer=optimizer,
        initial_lr=1e-5,
        aggregation="full",
        criterion=nn.MSELoss(),
    )

    assert controller.summary()["decoder_parameter_count"] == 400_000
    assert controller.summary()["persistent_state_numel"] == 4
    assert sum(t.numel() for t in controller.hessian_ema) == 4

    source = torch.randn(3, 200_000)
    target = torch.randn(3, 200_000)
    _one_step(model, optimizer, controller, source, target)
    gram, _ = controller._moment_gram(controller.blocks[0])
    assert gram.shape == (2, 2)
    assert gram.numel() == 4


class _CoreModel:
    def __init__(self):
        self.gradiend = GradiendModel(input_dim=8, latent_dim=1, device=torch.device("cpu"))
        self.base_model = SimpleMockModel()
        self.name_or_path = "auto-decoder-lr-test"

    def to(self, dtype=None, torch_dtype=None):
        self.base_model.dtype = torch_dtype if torch_dtype is not None else dtype
        return self

    def save_pretrained(self, save_directory, **kwargs):
        self.gradiend.save_pretrained(save_directory, **kwargs)

    @contextmanager
    def exclusive_base_gradient_access(self):
        yield


def test_core_training_calibrates_at_first_eval_boundary_and_records_metadata(tmp_path):
    torch.manual_seed(17)
    model = _CoreModel()
    batches = [
        {"source": torch.randn(4, 8), "target": torch.randn(4, 8)}
        for _ in range(4)
    ]
    args = TrainingArguments(
        output_dir=str(tmp_path),
        learning_rate=1e-8,
        learning_rate_decoder="auto",
        max_steps=4,
        num_train_epochs=1,
        eval_steps=2,
        do_eval=False,
        save_only_best=False,
    )

    train(model_with_gradiend=model, data=batches, training_args=args)
    payload = load_training_stats(str(tmp_path))
    result = payload["training_stats"]["decoder_lr_auto"]

    assert result["status"] == "calibrated"
    assert result["calibration_step"] == 2
    assert result["observations"] == 2
    assert result["selected_lr"] >= args.learning_rate


def test_core_training_rejects_auto_when_no_steps_remain_after_calibration(tmp_path):
    model = _CoreModel()
    args = TrainingArguments(
        output_dir=str(tmp_path),
        learning_rate_decoder="auto",
        max_steps=2,
        num_train_epochs=1,
        eval_steps=2,
        do_eval=False,
    )
    batches = [
        {"source": torch.randn(2, 8), "target": torch.randn(2, 8)}
        for _ in range(2)
    ]

    with pytest.raises(ValueError, match="must exceed eval_steps"):
        train(model_with_gradiend=model, data=batches, training_args=args)


def test_component_model_supervised_decoder_calibrates_full_objective(tmp_path):
    torch.manual_seed(23)
    model = _CoreModel()
    model.gradiend = GradiendModel(
        input_dim=5,
        latent_dim=1,
        device=torch.device("cpu"),
        component_slices=[
            {"id": "left", "start": 0, "end": 2},
            {"id": "right", "start": 2, "end": 5},
        ],
    )
    batches = [
        {
            "source": torch.randn(4, 5),
            "target": torch.randn(4, 5),
            "label": [-1.0, -0.25, 0.5, 1.0],
        }
        for _ in range(4)
    ]
    args = TrainingArguments(
        output_dir=str(tmp_path),
        learning_rate=1e-8,
        learning_rate_decoder="auto",
        supervised_decoder=True,
        convergent_score_threshold=100.0,
        max_steps=4,
        num_train_epochs=1,
        eval_steps=2,
        do_eval=False,
        save_only_best=False,
    )

    train(model_with_gradiend=model, data=batches, training_args=args)
    result = load_training_stats(str(tmp_path))["training_stats"]["decoder_lr_auto"]
    assert result["status"] == "calibrated"
    assert result["calibration_step"] == 2


@pytest.mark.parametrize(
    "optimizer_factory, message",
    [
        (lambda params: torch.optim.SGD(params, lr=1e-5), "requires optim='adamw'"),
        (
            lambda params: torch.optim.Adam(params, lr=1e-5, weight_decay=1e-2),
            "requires weight_decay=0",
        ),
        (
            lambda params: torch.optim.AdamW(params, lr=1e-5, weight_decay=1e-2),
            "requires weight_decay=0",
        ),
    ],
)
def test_unsupported_optimizer_modes_fail_loudly(optimizer_factory, message):
    model = GradiendModel(input_dim=5, latent_dim=1, device=torch.device("cpu"))
    optimizer = optimizer_factory(model.parameters())
    with pytest.raises(ValueError, match=message):
        AutoDecoderLearningRate(
            gradiend=model,
            optimizer=optimizer,
            initial_lr=1e-5,
            aggregation="full",
            criterion=nn.MSELoss(),
        )


def test_non_linear_decoder_and_custom_loss_fail_loudly():
    nonlinear = GradiendModel(
        input_dim=5,
        latent_dim=1,
        activation_decoder="tanh",
        device=torch.device("cpu"),
    )
    optimizer = torch.optim.AdamW(nonlinear.parameters(), lr=1e-5)
    with pytest.raises(ValueError, match="identity/linear"):
        AutoDecoderLearningRate(
            gradiend=nonlinear,
            optimizer=optimizer,
            initial_lr=1e-5,
            aggregation="full",
            criterion=nn.MSELoss(),
        )

    linear = GradiendModel(input_dim=5, latent_dim=1, device=torch.device("cpu"))
    optimizer = torch.optim.AdamW(linear.parameters(), lr=1e-5)
    with pytest.raises(ValueError, match="requires nn.MSELoss"):
        AutoDecoderLearningRate(
            gradiend=linear,
            optimizer=optimizer,
            initial_lr=1e-5,
            aggregation="full",
            criterion=nn.L1Loss(),
        )
