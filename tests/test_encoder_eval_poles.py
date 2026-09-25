"""Encoder-eval pole expansion: two-pole encodes source only; one-pole may expand.

Regression: expanding fac+alt on every encoder eval made M/F means identical whenever
H(factual)≈H(alternative) (e.g. pre_prediction on causal LMs), even for normal
two-pole gender data where source=alternative should score only alternative texts.
"""

from __future__ import annotations

import pandas as pd
import torch

from gradiend.evaluator.encoder_metrics import get_encoder_metrics_from_dataframe
from gradiend.trainer.core.dataset import SignalTrainingDatasetBase
from gradiend.trainer.core.feature_definition import FeatureLearningDefinition
from gradiend.trainer.core.signals import Signal, SignalBatch


class _TwoPoleGenderRows:
    """Two-pole rows: M factual / F alternative and F factual / M alternative.

    Factual and alternative *signals* are identical within a row (simulates
    pre_prediction on a causal LM). Across rows they differ.
    """

    batch_size = 1

    def __init__(self):
        self.rows = [
            {
                "factual": torch.tensor([3.0, 0.0]),
                "alternative": torch.tensor([3.0, 0.0]),
                "label": 1,
                "factual_token": "she",
                "alternative_token": "he",
                "factual_id": "F",
                "alternative_id": "M",
                "is_identity_transition": False,
            },
            {
                "factual": torch.tensor([0.0, 7.0]),
                "alternative": torch.tensor([0.0, 7.0]),
                "label": -1,
                "factual_token": "he",
                "alternative_token": "she",
                "factual_id": "M",
                "alternative_id": "F",
                "is_identity_transition": False,
            },
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return dict(self.rows[idx])


class _OnePoleRows:
    batch_size = 1

    def __len__(self):
        return 1

    def __getitem__(self, _idx):
        return {
            "factual": torch.tensor([1.0, 2.0]),
            "alternative": torch.tensor([10.0, 20.0]),
            "label": 1,
            "factual_token": "IO",
            "alternative_token": "S",
            "factual_id": "IO",
            "alternative_id": "S",
            "is_identity_transition": False,
        }


class _IdentityMapExtractor:
    signal = Signal.gradient()

    def __call__(
        self,
        factual_inputs=None,
        alternative_inputs=None,
        *,
        requires_factual=True,
        requires_alternative=True,
    ):
        fac = None if factual_inputs is None else factual_inputs.detach().clone()
        alt = None if alternative_inputs is None else alternative_inputs.detach().clone()
        return SignalBatch(factual=fac, alternative=alt)


def _means_by_label(dataset: SignalTrainingDatasetBase) -> dict[int, float]:
    """Encode with probe w=[1,0] so score = first activation dim."""
    w = torch.tensor([1.0, 0.0])
    buckets: dict[int, list[float]] = {}
    for i in range(len(dataset)):
        row = dataset[i]
        score = float(torch.dot(w, row["source"]))
        lab = int(row["label"])
        buckets.setdefault(lab, []).append(score)
    return {lab: sum(vals) / len(vals) for lab, vals in buckets.items()}


def test_two_pole_alternative_encoder_eval_length_is_not_doubled():
    ds = SignalTrainingDatasetBase(
        _TwoPoleGenderRows(),
        _IdentityMapExtractor(),
        source="alternative",
        target=None,
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    assert ds.expand_encoder_eval_poles is False
    assert len(ds) == 2


def test_two_pole_alternative_encoder_eval_scores_only_alternative_pole():
    ds = SignalTrainingDatasetBase(
        _TwoPoleGenderRows(),
        _IdentityMapExtractor(),
        source="alternative",
        target=None,
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    row0 = ds[0]
    row1 = ds[1]
    assert row0["label"] == -1
    assert row1["label"] == 1
    assert torch.allclose(row0["source"], torch.tensor([3.0, 0.0]))
    assert torch.allclose(row1["source"], torch.tensor([0.0, 7.0]))


def test_two_pole_without_expand_means_differ_even_when_fac_equals_alt_h():
    """Must NOT force mean(M)=mean(F) when H_fac==H_alt within a row."""
    ds = SignalTrainingDatasetBase(
        _TwoPoleGenderRows(),
        _IdentityMapExtractor(),
        source="alternative",
        target=None,
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    means = _means_by_label(ds)
    assert set(means) == {-1, 1}
    assert means[1] != means[-1]
    assert means[-1] == 3.0
    assert means[1] == 0.0


def test_expand_flag_forces_equal_means_when_fac_equals_alt_h():
    """Shows why expand is wrong for two-pole: identical H → identical class means."""
    ds = SignalTrainingDatasetBase(
        _TwoPoleGenderRows(),
        _IdentityMapExtractor(),
        source="alternative",
        target=None,
        expand_encoder_eval_poles=True,
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    assert len(ds) == 4
    means = _means_by_label(ds)
    assert means[1] == means[-1]


def test_one_pole_expand_yields_both_label_signs_from_single_factual_class():
    ds = SignalTrainingDatasetBase(
        _OnePoleRows(),
        _IdentityMapExtractor(),
        source="factual",
        target=None,
        expand_encoder_eval_poles=True,
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    assert len(ds) == 2
    labels = [int(ds[i]["label"]) for i in range(2)]
    assert labels == [1, -1]
    assert torch.allclose(ds[0]["source"], torch.tensor([1.0, 2.0]))
    assert torch.allclose(ds[1]["source"], torch.tensor([10.0, 20.0]))


def test_one_pole_without_expand_is_unipolar_under_factual_source():
    ds = SignalTrainingDatasetBase(
        _OnePoleRows(),
        _IdentityMapExtractor(),
        source="factual",
        target=None,
        expand_encoder_eval_poles=False,
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    assert len(ds) == 1
    assert int(ds[0]["label"]) == 1


def test_create_evaluation_data_sets_expand_flag_from_one_pole_config():
    calls: dict = {}

    class _Stub(FeatureLearningDefinition):
        def __init__(self, one_pole: bool):
            super().__init__(
                target_classes=["IO"] if one_pole else ["M", "F"],
                run_id="_test_encoder_eval_poles",
            )
            self._one_pole = one_pole
            self.training_args = type(
                "A",
                (),
                {
                    "source": "alternative",
                    "encoder_eval_max_size": 8,
                    "use_cached_gradients": False,
                    "encoder_eval_balance": False,
                    "include_other_classes": False,
                },
            )()

        def _is_one_pole_config(self):
            return self._one_pole

        def create_training_data(self, *args, **kwargs):
            return _TwoPoleGenderRows()

        def create_gradient_training_dataset(self, raw, model, **kwargs):
            calls.clear()
            calls.update(kwargs)
            return SignalTrainingDatasetBase(
                raw,
                _IdentityMapExtractor(),
                source=kwargs.get("source", "alternative"),
                target=kwargs.get("target"),
                expand_encoder_eval_poles=kwargs.get("expand_encoder_eval_poles", False),
                signal=Signal.gradient(),
                device=torch.device("cpu"),
            )

        def create_gradiend(self, *a, **k):
            raise NotImplementedError

        def evaluate_base_model(self, *a, **k):
            raise NotImplementedError

        def _analyze_encoder(self, *a, **k):
            raise NotImplementedError

        def _get_decoder_eval_dataframe(self, *a, **k):
            raise NotImplementedError

        def _get_decoder_eval_targets(self, *a, **k):
            raise NotImplementedError

    two = _Stub(False)
    ds_two = two.create_eval_data(model_with_gradiend=object(), split="validation")
    assert calls.get("expand_encoder_eval_poles") is False
    assert len(ds_two) == 2

    one = _Stub(True)
    ds_one = one.create_eval_data(model_with_gradiend=object(), split="validation")
    assert calls.get("expand_encoder_eval_poles") is True
    assert len(ds_one) == 4


def test_metrics_correlation_defined_for_two_pole_source_alternative():
    ds = SignalTrainingDatasetBase(
        _TwoPoleGenderRows(),
        _IdentityMapExtractor(),
        source="alternative",
        target=None,
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    rows = []
    for i in range(len(ds)):
        row = ds[i]
        rows.append(
            {
                "encoded": float(row["source"][0]),
                "label": float(row["label"]),
                "type": "training",
                "source_id": row["factual_id"],
            }
        )
    metrics = get_encoder_metrics_from_dataframe(pd.DataFrame(rows))
    assert metrics["n_samples"] == 2
    assert abs(metrics["all_data"]["correlation"]) == 1.0
