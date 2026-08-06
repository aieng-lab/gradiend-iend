"""Regression: default feature-factor sign depends on encoder source."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gradiend.evaluator.decoder import (
    compute_metric_summaries,
    derive_default_feature_factor,
    LMSThresholdPolicy,
)
from gradiend.model._source_target import (
    activation_feature_factor_from_encoding_direction,
    feature_factor_from_encoding_direction,
    intervention_feature_factor_from_encoding_direction,
)
from tests.testing_mocks import MockTokenizer


class _MockModel:
    def __init__(self, source: str, *, target: str = "diff", mapping_kind: str = "gradient"):
        self.source = source
        self.target = target
        self.gradiend = SimpleNamespace(mapping_kind=mapping_kind)
        self.feature_class_encoding_direction = {"3SG": 1.0, "3PL": -1.0}
        self.tokenizer = MockTokenizer()


class _Trainer:
    def __init__(self, model: _MockModel):
        self._model = model

    def get_model(self):
        return self._model


@pytest.mark.parametrize(
    "source,direction,expected",
    [
        ("factual", 1.0, -1.0),
        ("diff", 1.0, -1.0),
        ("both", 1.0, -1.0),
        ("alternative", 1.0, 1.0),
        ("factual", -1.0, 1.0),
        ("both", -1.0, 1.0),
        ("alternative", -1.0, -1.0),
    ],
)
def test_feature_factor_from_encoding_direction(source, direction, expected):
    assert feature_factor_from_encoding_direction(direction, source) == expected


@pytest.mark.parametrize(
    "source,target,direction,expected",
    [
        ("factual", "diff", 1.0, 1.0),
        ("diff", "diff", 1.0, 1.0),
        ("both", "diff", 1.0, 1.0),
        ("alternative", "diff", 1.0, -1.0),
    ],
)
def test_activation_feature_factor_from_encoding_direction(source, target, direction, expected):
    assert activation_feature_factor_from_encoding_direction(direction, source, target) == expected


@pytest.mark.parametrize("target", ["factual", "alternative"])
def test_activation_feature_factor_rejects_non_diff_targets(target):
    with pytest.raises(ValueError, match="ACTIEND activation interventions currently require target='diff'"):
        activation_feature_factor_from_encoding_direction(1.0, "factual", target)


def test_intervention_feature_factor_uses_signal_kind_not_scope():
    assert (
        intervention_feature_factor_from_encoding_direction(
            1.0,
            "factual",
            "diff",
            signal_kind="gradient",
        )
        == -1.0
    )
    assert (
        intervention_feature_factor_from_encoding_direction(
            1.0,
            "factual",
            "diff",
            signal_kind="activation",
        )
        == 1.0
    )


@pytest.mark.parametrize(
    "source,expected_3sg,expected_3pl",
    [
        ("factual", -1.0, 1.0),
        ("diff", -1.0, 1.0),
        ("both", -1.0, 1.0),
        ("alternative", 1.0, -1.0),
    ],
)
def test_derive_default_feature_factor_by_source(source, expected_3sg, expected_3pl):
    model = _MockModel(source)
    trainer = _Trainer(model)
    assert derive_default_feature_factor(trainer, model, class_name="3SG") == expected_3sg
    assert derive_default_feature_factor(trainer, model, class_name="3PL") == expected_3pl


@pytest.mark.parametrize(
    "source,target,expected_3sg,expected_3pl",
    [
        ("factual", "diff", 1.0, -1.0),
        ("diff", "diff", 1.0, -1.0),
        ("alternative", "diff", -1.0, 1.0),
    ],
)
def test_derive_default_feature_factor_for_activation_space(source, target, expected_3sg, expected_3pl):
    model = _MockModel(source, target=target, mapping_kind="activation")
    trainer = _Trainer(model)
    assert derive_default_feature_factor(trainer, model, class_name="3SG") == expected_3sg
    assert derive_default_feature_factor(trainer, model, class_name="3PL") == expected_3pl


@pytest.mark.parametrize("target", ["factual", "alternative"])
def test_derive_default_feature_factor_rejects_activation_non_diff_targets(target):
    model = _MockModel("factual", target=target, mapping_kind="activation")
    trainer = _Trainer(model)
    with pytest.raises(ValueError, match="ACTIEND activation interventions currently require target='diff'"):
        derive_default_feature_factor(trainer, model, class_name="3SG")


def test_strengthen_summary_filters_to_class_feature_factor():
    """With both ff in grid, strengthen 3SG must not pick the other class's ff."""
    results = {
        "base": {"id": "base", "lms": {"lms": 1.0}, "probs": {}},
        (-1.0, 0.01): {
            "id": {"feature_factor": -1.0, "learning_rate": 0.01},
            "lms": {"lms": 1.0},
            "probs": {"3SG": 0.9, "3PL": 0.1},
        },
        (1.0, 0.01): {
            "id": {"feature_factor": 1.0, "learning_rate": 0.01},
            "lms": {"lms": 1.0},
            "probs": {"3SG": 0.2, "3PL": 0.8},
        },
    }
    class_to_ff = {"3SG": 1.0, "3PL": -1.0}
    summary = compute_metric_summaries(
        results,
        metrics=["3SG"],
        selector=LMSThresholdPolicy(ratio=0.0),
        class_to_ff=class_to_ff,
        feature_factor_from_id=lambda cid: cid["feature_factor"] if isinstance(cid, dict) else cid[0],
        lr_from_id=lambda cid: cid["learning_rate"] if isinstance(cid, dict) else cid[1],
    )
    assert summary["3SG"]["feature_factor"] == 1.0
    assert summary["3SG"]["value"] == 0.2


def test_lms_threshold_policy_fallback_handles_dict_candidate_ids():
    """LMSThresholdPolicy fallback must read lr from dict-style candidate ids."""
    results = {
        "base": {"id": "base", "lms": {"lms": 1.0}, "probs": {}},
        (-1.0, 0.01): {
            "id": {"feature_factor": -1.0, "learning_rate": 0.01},
            "lms": {"lms": 0.1},
            "probs": {"M": 0.2, "F": 0.8},
        },
        (-1.0, 0.1): {
            "id": {"feature_factor": -1.0, "learning_rate": 0.1},
            "lms": {"lms": 0.1},
            "probs": {"M": 0.3, "F": 0.7},
        },
    }
    summary = compute_metric_summaries(
        results,
        metrics=["M"],
        selector=LMSThresholdPolicy(ratio=0.99),
        class_to_ff={"M": -1.0, "F": 1.0},
        feature_factor_from_id=lambda cid: cid["feature_factor"] if isinstance(cid, dict) else cid[0],
        lr_from_id=lambda cid: cid["learning_rate"] if isinstance(cid, dict) else cid[1],
    )
    assert summary["M"]["learning_rate"] == 0.01


def test_strengthen_single_class_grid_does_not_require_other_class_ff():
    """Single-target grid with only that class's ff must summarize under the target class."""
    results = {
        "base": {"id": "base", "lms": {"lms": 1.0}, "probs": {}},
        (-1.0, 0.01): {
            "id": {"feature_factor": -1.0, "learning_rate": 0.01},
            "lms": {"lms": 0.95},
            "probs": {"M": 0.7, "F": 0.3},
        },
        (-1.0, 0.1): {
            "id": {"feature_factor": -1.0, "learning_rate": 0.1},
            "lms": {"lms": 0.9},
            "probs": {"M": 0.8, "F": 0.2},
        },
    }
    summary = compute_metric_summaries(
        results,
        metrics=["M"],
        selector=LMSThresholdPolicy(ratio=0.0),
        class_to_ff={"M": -1.0, "F": 1.0},
        feature_factor_from_id=lambda cid: cid["feature_factor"] if isinstance(cid, dict) else cid[0],
        lr_from_id=lambda cid: cid["learning_rate"] if isinstance(cid, dict) else cid[1],
    )
    assert summary["M"]["feature_factor"] == -1.0
    assert summary["M"]["value"] == 0.8


def test_strengthen_missing_feature_factor_error_lists_grid_ffs():
    results = {
        "base": {"id": "base", "lms": {"lms": 1.0}, "probs": {}},
        (-1.0, 0.01): {
            "id": {"feature_factor": -1.0, "learning_rate": 0.01},
            "lms": {"lms": 1.0},
            "probs": {"M": 0.7, "F": 0.3},
        },
    }
    with pytest.raises(ValueError, match=r"strengthen class 'F'.*required feature_factor=1\.0.*Grid feature_factors present: \[-1\.0\]"):
        compute_metric_summaries(
            results,
            metrics=["F"],
            selector=LMSThresholdPolicy(ratio=0.0),
            class_to_ff={"M": -1.0, "F": 1.0},
            feature_factor_from_id=lambda cid: cid["feature_factor"] if isinstance(cid, dict) else cid[0],
            lr_from_id=lambda cid: cid["learning_rate"] if isinstance(cid, dict) else cid[1],
        )
