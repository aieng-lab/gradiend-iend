import pytest
import torch

from gradiend.model import ParamMappedGradiendModel
from gradiend.trainer.signal_trainer import SignalPairDataset, SignalTrainer
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.signals import Signal
from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
from gradiend.trainer.trainer import Trainer


class _Tokenizer:
    name_or_path = "tiny-tokenizer"
    mask_token = "[MASK]"
    mask_token_id = 0
    eos_token = "</s>"
    pad_token = "<pad>"
    model_max_length = 16


def _signal_model(width=3, *, mapping_kind="activation"):
    base = torch.nn.Linear(1, 1)
    base.name_or_path = "tiny-base"
    gradiend = ParamMappedGradiendModel(
        width,
        latent_dim=1,
        param_map={"activation:site": {"shape": (width,), "repr": "all"}},
        mapping_kind=mapping_kind,
        torch_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    return TextPredictionModelWithGradiend(
        base,
        gradiend,
        _Tokenizer(),
        source="both",
        target="diff",
        base_model_device=torch.device("cpu"),
        device_encoder=torch.device("cpu"),
        device_decoder=torch.device("cpu"),
    )


def _args(tmp_path, *, signal=None, **overrides):
    values = dict(
        experiment_dir=str(tmp_path),
        source="both",
        target="diff",
        signal=signal or Signal.activation(),
        train_batch_size=2,
        base_gradient_batch_size=2,
        gradiend_batch_size=1,
        max_steps=2,
        num_train_epochs=2,
        learning_rate=1e-3,
        do_eval=False,
        eval_strategy="no",
        max_seeds=1,
        min_convergent_seeds=1,
        use_cache=False,
        torch_dtype=torch.float32,
    )
    values.update(overrides)
    return TrainingArguments(**values)


def test_signal_trainer_is_a_real_trainer():
    assert issubclass(SignalTrainer, Trainer)


def test_signal_pair_dataset_exposes_standard_poles():
    dataset = SignalPairDataset(
        torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
        torch.tensor([[1.0, 1.0]]),
        positive_class="present",
        negative_class="absent",
        seed=7,
    )
    assert len(dataset) == 2
    row = dataset[0]
    assert row["label"] == 1.0
    assert row["factual_id"] == "present"
    assert row["alternative_id"] == "absent"


def test_signal_dataset_uses_standard_source_both_semantics(tmp_path):
    trainer = SignalTrainer(
        _signal_model(2),
        torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
        torch.tensor([[1.0, 1.0], [0.0, 2.0]]),
        args=_args(tmp_path),
        target_classes=["present", "absent"],
    )
    raw = trainer.create_training_data(trainer._model_arg, batch_size=1)
    signals = trainer.create_gradient_training_dataset(raw, trainer._model_arg)
    factual_row = signals[0]
    alternative_row = signals[1]
    torch.testing.assert_close(factual_row["target"], factual_row["source"] - raw[0]["alternative"])
    torch.testing.assert_close(
        alternative_row["target"],
        alternative_row["source"] - raw[1]["factual"],
    )
    assert factual_row["label"] == 1.0
    assert alternative_row["label"] == -1.0


def test_signal_trainer_validates_mapping_kind_and_width(tmp_path):
    with pytest.raises(ValueError, match="input_dim"):
        SignalTrainer(_signal_model(3), torch.ones(2, 2), -torch.ones(2, 2), args=_args(tmp_path))
    model = _signal_model(3, mapping_kind="gradient")
    with pytest.raises(ValueError, match="does not match model signal kind"):
        SignalTrainer(model, torch.ones(2, 3), -torch.ones(2, 3), args=_args(tmp_path))


def test_signal_trainer_accepts_precomputed_gradient_signals(tmp_path):
    model = _signal_model(3, mapping_kind="gradient")
    trainer = SignalTrainer(
        model,
        torch.ones(2, 3),
        -torch.ones(2, 3),
        args=_args(tmp_path, signal=Signal.gradient()),
    )
    raw = trainer.create_training_data(model)
    signals = trainer.create_gradient_training_dataset(raw, model)
    assert signals[0]["source"].shape == (3,)


def test_signal_trainer_runs_inherited_training_lifecycle(tmp_path):
    model = _signal_model(3)
    trainer = SignalTrainer(
        model,
        torch.ones(4, 3),
        -torch.ones(4, 3),
        args=_args(tmp_path, learning_rate_decoder=None),
        target_classes=["present", "absent"],
    )
    output_dir = tmp_path / "model"
    result = trainer.train(output_dir=str(output_dir))
    assert result is trainer
    assert (output_dir / "config.json").is_file()
    assert trainer.get_model().base_model is model.base_model


def test_signal_trainer_multi_seed_reuses_the_shared_base_model(tmp_path):
    model = _signal_model(3)
    trainer = SignalTrainer(
        model,
        torch.ones(4, 3),
        -torch.ones(4, 3),
        args=_args(
            tmp_path,
            max_seeds=2,
            max_steps=1,
            convergent_metric="loss",
            convergent_score_threshold=100.0,
            learning_rate_decoder=None,
        ),
    )
    output_dir = tmp_path / "multi-seed-model"
    trainer.train(output_dir=str(output_dir))
    assert (output_dir / "config.json").is_file()
    assert trainer.get_model().base_model is model.base_model


def test_signal_trainer_supports_encoder_evaluation_but_not_decoder_eval(tmp_path):
    model = _signal_model(3)
    trainer = SignalTrainer(
        model,
        data={
            "train": (torch.ones(2, 3), -torch.ones(2, 3)),
            "validation": (2 * torch.ones(2, 3), -2 * torch.ones(2, 3)),
        },
        args=_args(tmp_path),
    )
    frame = trainer._analyze_encoder(model, split="validation", use_cache=False)
    assert set(frame["label"].astype(float)) == {-1.0, 1.0}
    with pytest.raises(NotImplementedError, match="modality-specific evaluator"):
        trainer.evaluate_base_model(None, None)
