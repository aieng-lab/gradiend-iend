"""Unit tests for source='both' (batch-alternating poles compiled to factual/diff)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch

from gradiend.evaluator.encoder_metrics import get_encoder_metrics_from_dataframe
from gradiend.model._source_target import (
    encoding_view_sign_for_source,
    feature_factor_from_encoding_direction,
    validate_source_target,
    validate_source_target_combination,
)
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.dataset import (
    SignalTrainingDatasetBase,
    _both_eval_base_and_side,
    _both_side_for_batch,
    _invert_numeric_label,
    _swap_factual_alternative_batch,
)
from gradiend.trainer.core.signals import Signal, SignalBatch
from gradiend.trainer.text.prediction.trainer import TextPredictionTrainer
from tests.testing_mocks import MockTokenizer


class _FixedPairRows:
    """Two identical non-identity rows so batch indices 0 and 1 share the same pair."""

    batch_size = 1

    def __init__(self, n: int = 2):
        self.n = n
        self._row = {
            "factual": torch.tensor([1.0, 2.0]),
            "alternative": torch.tensor([10.0, 20.0]),
            "label": 1,
            "factual_token": "he",
            "alternative_token": "they",
            "factual_id": "3SG",
            "alternative_id": "3PL",
            "is_identity_transition": False,
        }

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Fresh dict each time so swaps cannot mutate stored rows.
        return dict(self._row)


class _IdentityRow:
    batch_size = 1

    def __len__(self):
        return 1

    def __getitem__(self, _idx):
        return {
            "factual": torch.tensor([1.0, 2.0]),
            "alternative": torch.tensor([99.0, 100.0]),
            "label": 0,
            "factual_token": "the",
            "alternative_token": "the",
            "factual_id": "neutral",
            "alternative_id": "neutral",
            "is_identity_transition": True,
        }


class _RecordingExtractor:
    """Side-symmetric mock: output depends only on the input tensor, not extract side."""

    signal = Signal.gradient()

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        factual_inputs=None,
        alternative_inputs=None,
        *,
        requires_factual=True,
        requires_alternative=True,
    ):
        self.calls.append(
            {
                "factual_inputs": None if factual_inputs is None else factual_inputs.detach().clone(),
                "alternative_inputs": None if alternative_inputs is None else alternative_inputs.detach().clone(),
                "requires_factual": requires_factual,
                "requires_alternative": requires_alternative,
            }
        )
        # Identity map: same transform for either side so fac↔alt swap yields exact opposite diffs.
        factual = factual_inputs.detach().clone() if requires_factual and factual_inputs is not None else None
        alternative = (
            alternative_inputs.detach().clone()
            if requires_alternative and alternative_inputs is not None
            else None
        )
        return SignalBatch.from_factual_alternative(
            factual,
            alternative,
            signal_id="gradient",
        )


def test_both_side_alternates_by_batch_index():
    # Single balance group: historical even/odd rule.
    assert _both_side_for_batch(0) == "factual"
    assert _both_side_for_batch(1) == "alternative"
    assert _both_side_for_batch(2) == "factual"
    assert _both_side_for_batch(3) == "alternative"
    assert _both_side_for_batch(0, n_balance_groups=1) == "factual"
    assert _both_side_for_batch(1, n_balance_groups=1) == "alternative"


def test_both_side_orthogonal_to_balance_group_phase():
    """Feature+neutral (n=2) must not lock poles to batch parity."""
    # batch % 2 picks group; visit = batch // 2 picks pole.
    assert _both_side_for_batch(0, n_balance_groups=2) == "factual"  # group0 visit0
    assert _both_side_for_batch(1, n_balance_groups=2) == "factual"  # group1 visit0
    assert _both_side_for_batch(2, n_balance_groups=2) == "alternative"  # group0 visit1
    assert _both_side_for_batch(3, n_balance_groups=2) == "alternative"  # group1 visit1
    assert _both_side_for_batch(4, n_balance_groups=2) == "factual"

    assert _both_side_for_batch(0, n_balance_groups=3) == "factual"
    assert _both_side_for_batch(3, n_balance_groups=3) == "alternative"
    assert _both_side_for_batch(6, n_balance_groups=3) == "factual"


class _FeaturePlusNeutralBalanceRows:
    """Mimics TextBatchedDatasetBase: batch_idx % 2 cycles feature then neutral."""

    batch_size = 1
    balance_keys = [0, 2]
    n_balance_groups = 2

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        batch_idx = idx // self.batch_size
        group = self.balance_keys[batch_idx % len(self.balance_keys)]
        if group == 0:
            return {
                "factual": torch.tensor([1.0, 2.0]),
                "alternative": torch.tensor([10.0, 20.0]),
                "label": 1,
                "factual_token": "Asia",
                "alternative_token": "Africa",
                "factual_id": "asian",
                "alternative_id": "black",
                "feature_class_id": 0,
                "is_identity_transition": False,
            }
        return {
            "factual": torch.tensor([3.0, 4.0]),
            "alternative": torch.tensor([3.0, 4.0]),
            "label": 0,
            "factual_token": "the",
            "alternative_token": "the",
            "factual_id": "neutral",
            "alternative_id": "neutral",
            "feature_class_id": 2,
            "is_identity_transition": True,
        }


def test_both_with_neutral_balance_group_still_exposes_feature_minus_pole():
    """Regression: even n_balance_groups must not lock feature onto factual only."""
    rows = _FeaturePlusNeutralBalanceRows()
    dataset = SignalTrainingDatasetBase(
        rows,
        _RecordingExtractor(),
        source="both",
        target="diff",
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    assert dataset._n_training_balance_groups() == 2

    labels = [dataset[i]["label"] for i in range(4)]
    # batch0 feature factual (+1), batch1 neutral (0),
    # batch2 feature alternative (-1), batch3 neutral (0)
    assert labels == [1, 0, -1, 0], labels

    feature_poles = {
        dataset[i]["factual_id"]
        for i in (0, 2)
    }
    assert feature_poles == {"asian", "black"}
    assert dataset[0]["label"] == 1
    assert dataset[2]["label"] == -1


class _AnyInputConstantExtractor:
    """Signal extractor that accepts TextTrainingDataset tokenizer dicts."""

    signal = Signal.gradient()

    def __call__(
        self,
        factual_inputs=None,
        alternative_inputs=None,
        *,
        requires_factual=True,
        requires_alternative=True,
    ):
        factual = (
            torch.tensor([1.0, 2.0])
            if requires_factual and factual_inputs is not None
            else None
        )
        alternative = (
            torch.tensor([10.0, 20.0])
            if requires_alternative and alternative_inputs is not None
            else None
        )
        return SignalBatch.from_factual_alternative(
            factual,
            alternative,
            signal_id="gradient",
        )


def _one_pole_race_like_per_class(*, n_feature: int = 8) -> dict:
    """Asian prompts with same-row black/white CF columns (one-pole multi-CF)."""
    asian_rows = [
        {
            "masked": f"Asia prompt {i} [MASK]",
            "asian": "Asia",
            "black": "Africa",
            "white": "Europe",
            "split": "train",
        }
        for i in range(n_feature)
    ]
    return {
        "asian": pd.DataFrame(asian_rows),
        "black": pd.DataFrame(
            [
                {
                    "masked": f"Black prompt {i} [MASK]",
                    "asian": "Asia",
                    "black": "Africa",
                    "white": "Europe",
                    "split": "train",
                }
                for i in range(4)
            ]
        ),
        "white": pd.DataFrame(
            [
                {
                    "masked": f"White prompt {i} [MASK]",
                    "asian": "Asia",
                    "black": "Africa",
                    "white": "Europe",
                    "split": "train",
                }
                for i in range(4)
            ]
        ),
    }


def _build_one_pole_both_text_training_data(*, batch_size: int, n_feature: int = 8, n_neutral: int = 8):
    """Real create_training_data path: one-pole + neutrals + source=both."""
    from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer

    neutral = pd.DataFrame(
        [
            {"masked": f"neutral {i} [MASK]", "label": "the", "split": "train"}
            for i in range(n_neutral)
        ]
    )
    config = TextPredictionConfig(
        target_classes=["asian"],
        all_classes=["asian", "black", "white"],
        counterfactual_classes="all",
        data=_one_pole_race_like_per_class(n_feature=n_feature),
        masked_col="masked",
        split_col="split",
        neutral_data=neutral,
        use_class_names_as_columns=True,
    )
    args = TrainingArguments(
        source="both",
        target="diff",
        seed=0,
        add_neutral_identity_transitions=True,
        add_identity_for_other_classes=False,
    )
    trainer = TextPredictionTrainer(
        model="__test_both_e2e__",
        config=config,
        args=args,
    )
    tokenizer = MockTokenizer()
    training_data = trainer.create_training_data(
        tokenizer,
        split="train",
        batch_size=batch_size,
    )
    return trainer, training_data


def test_one_pole_feature_pole_and_name_based_pre_prune():
    """One-pole emits named feature_pole; pre-prune stratifies on factual_id names."""
    from gradiend.trainer.core.pruning import (
        PrePruneConfig,
        _resolve_pre_prune_stratification,
        _stratified_indices,
    )

    trainer, training_data = _build_one_pole_both_text_training_data(batch_size=1)
    assert trainer.get_target_feature_class_ids() == ["pos", "neg"]

    df = training_data.data
    assert "feature_pole" in df.columns
    poles = set(df["feature_pole"].astype(str).unique())
    assert poles <= {"pos", "neg", "neutral"}
    # One-pole factual filter keeps + rows; source=both synthesizes the − pole at batch time.
    assert "pos" in poles
    assert "neutral" in poles
    # Deprecated int mirror of feature_pole.
    assert set(int(v) for v in df["feature_class_id"].unique()) <= {0, 1, 2}

    names = trainer.get_target_feature_classes()
    assert names is not None
    assert names[0] == "asian"
    assert set(names) >= {"asian", "black", "white"}

    key, targets = _resolve_pre_prune_stratification(
        training_data,
        PrePruneConfig(n_samples=4, topk=0.1),
        trainer,
    )
    assert key == "factual_id"
    # After one-pole factual filter, only + class rows remain in the training frame.
    assert targets == ["asian"]
    indices = _stratified_indices(
        training_data,
        n_samples=4,
        feature_class_key=key,
        target_feature_class_ids=targets,
        seed=0,
    )
    assert len(indices) == 4


def _scalar_or_list_values(value):
    if isinstance(value, list):
        return list(value)
    return [value]


def _collect_both_batch_metadata(signal_dataset, *, n_batches: int):
    rows = []
    for i in range(n_batches):
        batch = signal_dataset[i]
        labels = _scalar_or_list_values(batch["label"])
        factual_ids = _scalar_or_list_values(batch["factual_id"])
        identity = batch.get("is_identity_transition", False)
        if isinstance(identity, list):
            is_identity = all(bool(v) for v in identity)
        else:
            is_identity = bool(identity)
        rows.append(
            {
                "index": i,
                "labels": labels,
                "factual_ids": factual_ids,
                "is_identity": is_identity,
            }
        )
    return rows


@pytest.mark.parametrize("batch_size", [1, 2])
def test_both_e2e_real_text_dataset_with_neutrals_exposes_both_feature_poles(batch_size):
    """End-to-end: real TextTrainingDataset balance groups + source=both poles.

    Uses create_training_data (one-pole + neutral identity), not a hand-rolled
    balancer mock. Feature rows must appear under both +1 and -1 poles.
    """
    _trainer, training_data = _build_one_pole_both_text_training_data(
        batch_size=batch_size,
        n_feature=max(8, batch_size * 4),
        n_neutral=max(8, batch_size * 4),
    )

    assert hasattr(training_data, "n_balance_groups")
    assert training_data.n_balance_groups >= 2, training_data.balance_keys
    assert len(training_data.balance_keys) == training_data.n_balance_groups

    df = training_data.data
    assert (df["neutral_variant"] == "neutral_identity").any()
    assert "feature_pole" in df.columns
    feature_poles = sorted(
        {
            str(v)
            for v in df.loc[df["factual_id"] != "neutral", "feature_pole"].unique()
        }
    )
    neutral_poles = sorted(
        {
            str(v)
            for v in df.loc[df["factual_id"] == "neutral", "feature_pole"].unique()
        }
    )
    assert feature_poles, "expected feature training rows"
    assert neutral_poles == ["neutral"], "expected neutral identity rows"
    assert set(feature_poles).isdisjoint(set(neutral_poles))
    assert set(feature_poles) <= {"pos", "neg"}

    signal_dataset = SignalTrainingDatasetBase(
        training_data,
        _AnyInputConstantExtractor(),
        source="both",
        target="diff",
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    assert signal_dataset._n_training_balance_groups() == training_data.n_balance_groups

    n_batches = len(signal_dataset)
    # Need at least one full visit cycle over balance groups for both poles.
    assert n_batches >= 2 * training_data.n_balance_groups, (
        f"too few gradient batches ({n_batches}) for n_groups={training_data.n_balance_groups}"
    )

    meta = _collect_both_batch_metadata(signal_dataset, n_batches=n_batches)
    feature_batches = [m for m in meta if not m["is_identity"]]
    neutral_batches = [m for m in meta if m["is_identity"]]
    assert feature_batches, "scheduler never served feature batches"
    assert neutral_batches, "scheduler never served neutral batches"

    feature_labels = {int(lab) for m in feature_batches for lab in m["labels"]}
    assert feature_labels == {1, -1}, (
        "feature batches locked to a single pole under source=both with neutrals; "
        f"got labels={sorted(feature_labels)}; "
        f"n_balance_groups={training_data.n_balance_groups}; "
        f"feature_batch_indices={[m['index'] for m in feature_batches[:12]]}"
    )

    # Neutrals stay zero under either pole (identity swap is a no-op).
    neutral_labels = {int(lab) for m in neutral_batches for lab in m["labels"]}
    assert neutral_labels == {0}, neutral_labels

    # Feature factual_ids must include the + class and at least one CF after swaps.
    feature_factual_ids = {str(fid) for m in feature_batches for fid in m["factual_ids"]}
    assert "asian" in feature_factual_ids
    assert feature_factual_ids & {"black", "white"}, feature_factual_ids


def test_both_e2e_reads_n_balance_groups_from_real_text_training_dataset():
    """Guard: SignalTrainingDatasetBase must use live TextTrainingDataset.n_balance_groups."""
    _trainer, training_data = _build_one_pole_both_text_training_data(batch_size=1)
    # If this property disappears, pole math silently falls back to n=1 and the bug returns.
    assert training_data.n_balance_groups == len(training_data.balance_keys) >= 2

    signal_dataset = SignalTrainingDatasetBase(
        training_data,
        _AnyInputConstantExtractor(),
        source="both",
        target="diff",
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    assert signal_dataset._n_training_balance_groups() == training_data.n_balance_groups

    # Old even/odd rule would disagree with orthogonal rule whenever n_groups is even.
    n = training_data.n_balance_groups
    assert n % 2 == 0, "test expects even group count (feature+neutral) to catch the lockstep bug"
    assert _both_side_for_batch(2, n_balance_groups=n) == "alternative"
    assert _both_side_for_batch(2, n_balance_groups=1) == "factual"  # naive parity (buggy w/ neutrals)
    # Live resolver must follow the orthogonal rule, not naive parity.
    _fetch, side = signal_dataset._resolve_both_index(2)
    assert side == "alternative"


def test_both_eval_base_and_side_maps_expanded_indices():
    assert _both_eval_base_and_side(0) == (0, "factual")
    assert _both_eval_base_and_side(1) == (0, "alternative")
    assert _both_eval_base_and_side(2) == (1, "factual")
    assert _both_eval_base_and_side(3) == (1, "alternative")


def test_swap_factual_alternative_batch_inverts_label_and_metadata():
    batch = {
        "factual": torch.tensor([1.0]),
        "alternative": torch.tensor([2.0]),
        "label": 1,
        "factual_token": "he",
        "alternative_token": "they",
        "factual_id": "3SG",
        "alternative_id": "3PL",
        "feature_class_id": 0,
    }
    swapped = _swap_factual_alternative_batch(batch)

    # Input mapping keys are not rewritten in place.
    assert batch["label"] == 1
    assert batch["factual_id"] == "3SG"
    assert torch.equal(batch["factual"], torch.tensor([1.0]))

    assert torch.equal(swapped["factual"], torch.tensor([2.0]))
    assert torch.equal(swapped["alternative"], torch.tensor([1.0]))
    assert swapped["label"] == -1
    assert swapped["factual_token"] == "they"
    assert swapped["alternative_token"] == "he"
    assert swapped["factual_id"] == "3PL"
    assert swapped["alternative_id"] == "3SG"
    assert swapped["feature_class_id"] == 0


def test_swap_does_not_mutate_underlying_row_object():
    rows = _FixedPairRows(n=1)
    raw = rows[0]
    _ = _swap_factual_alternative_batch(raw)
    assert raw["label"] == 1
    assert raw["factual_id"] == "3SG"
    assert torch.equal(raw["factual"], torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize(
    "target",
    ["factual", "alternative"],
)
def test_both_rejects_non_diff_targets(target):
    with pytest.raises(ValueError, match="source='both' requires target='diff'"):
        validate_source_target_combination("both", target)
    with pytest.raises(ValueError, match="source='both' requires target='diff'"):
        TrainingArguments(source="both", target=target)
    with pytest.raises(ValueError, match="source='both' requires target='diff'"):
        SignalTrainingDatasetBase(
            _FixedPairRows(),
            _RecordingExtractor(),
            source="both",
            target=target,
            signal=Signal.gradient(),
        )


def test_both_allows_target_none_for_encoder_eval():
    validate_source_target_combination("both", None)
    dataset = SignalTrainingDatasetBase(
        _FixedPairRows(),
        _RecordingExtractor(),
        source="both",
        target=None,
        signal=Signal.gradient(),
    )
    assert dataset.source == "both"
    assert dataset.target is None


def test_training_arguments_accepts_both_with_diff():
    args = TrainingArguments(source="both", target="diff")
    assert args.source == "both"
    assert args.target == "diff"
    assert validate_source_target("source", "both") == "both"


def test_both_batch0_matches_factual_diff():
    rows = _FixedPairRows(n=2)
    extractor_both = _RecordingExtractor()
    extractor_fac = _RecordingExtractor()

    both = SignalTrainingDatasetBase(
        rows,
        extractor_both,
        source="both",
        target="diff",
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    factual = SignalTrainingDatasetBase(
        rows,
        extractor_fac,
        source="factual",
        target="diff",
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )

    both_row = both[0]
    fac_row = factual[0]

    assert torch.equal(both_row["source"], fac_row["source"])
    assert torch.equal(both_row["target"], fac_row["target"])
    assert both_row["label"] == fac_row["label"] == 1
    assert both_row["factual_id"] == "3SG"
    assert both_row["alternative_id"] == "3PL"
    # Identity extractor: g_fac=[1,2], g_alt=[10,20], diff=[-9,-18]
    assert both_row["source"].tolist() == [1.0, 2.0]
    assert both_row["target"].tolist() == [-9.0, -18.0]


def test_both_batch1_uses_alternative_pole_with_signed_diff():
    rows = _FixedPairRows(n=2)
    extractor = _RecordingExtractor()
    dataset = SignalTrainingDatasetBase(
        rows,
        extractor,
        source="both",
        target="diff",
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )

    row = dataset[1]

    # After swap: encoder input is original alternative
    # g_input=[10,20], opposite=[1,2], target=[9,18]
    assert row["source"].tolist() == [10.0, 20.0]
    assert row["target"].tolist() == [9.0, 18.0]
    assert row["label"] == -1
    assert row["factual_id"] == "3PL"
    assert row["alternative_id"] == "3SG"
    assert row["factual_token"] == "they"
    assert row["alternative_token"] == "he"

    # Historical source=alternative still uses target factual-alternative (not signed).
    alt_extractor = _RecordingExtractor()
    alt_dataset = SignalTrainingDatasetBase(
        rows,
        alt_extractor,
        source="alternative",
        target="diff",
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    alt_row = alt_dataset[0]
    assert alt_row["source"].tolist() == [10.0, 20.0]
    assert alt_row["target"].tolist() == [-9.0, -18.0]  # fac-alt, unchanged convention
    assert alt_row["label"] == -1


def test_both_consecutive_batches_have_opposite_targets_for_same_pair():
    rows = _FixedPairRows(n=2)
    dataset = SignalTrainingDatasetBase(
        rows,
        _RecordingExtractor(),
        source="both",
        target="diff",
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    even = dataset[0]
    odd = dataset[1]
    assert torch.allclose(even["target"], -odd["target"])
    assert even["label"] == -odd["label"]


def test_opt_in_diff_combination_reuses_factual_storage():
    dataset = SignalTrainingDatasetBase(
        _FixedPairRows(n=1),
        _RecordingExtractor(),
        source="factual",
        target="diff",
        signal=Signal.gradient(),
        device=torch.device("cpu"),
        combine_diff_in_place=True,
    )

    row = dataset[0]

    assert row["target"].tolist() == [-9.0, -18.0]
    assert row["source"].data_ptr() == row["target"].data_ptr()


def test_both_identity_row_zero_target_and_label():
    dataset = SignalTrainingDatasetBase(
        _IdentityRow(),
        _RecordingExtractor(),
        source="both",
        target="diff",
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    row = dataset[0]
    assert row["label"] == 0
    assert row["target"].abs().sum().item() == 0.0
    assert row["source"].tolist() == [1.0, 2.0]


def test_both_encoder_only_expands_each_example_to_both_poles():
    """Encoder eval (target=None) must yield +1 and -1 even from a single base row."""
    extractor = _RecordingExtractor()
    dataset = SignalTrainingDatasetBase(
        _FixedPairRows(n=1),
        extractor,
        source="both",
        target=None,
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    assert len(dataset) == 2

    row0 = dataset[0]
    assert row0["target"] is None
    assert row0["source"].tolist() == [1.0, 2.0]
    assert row0["label"] == 1
    assert row0["factual_id"] == "3SG"

    extractor.calls.clear()
    row1 = dataset[1]
    assert row1["target"] is None
    assert row1["source"].tolist() == [10.0, 20.0]
    assert row1["label"] == -1
    assert row1["factual_id"] == "3PL"
    assert all(c["requires_alternative"] is False for c in extractor.calls)


@pytest.mark.parametrize("source", ["factual", "alternative", "diff"])
def test_encoder_eval_does_not_expand_two_pole_by_default(source):
    """Normal two-pole encoder eval encodes source only (no fac/alt expand)."""
    dataset = SignalTrainingDatasetBase(
        _FixedPairRows(n=2),
        _RecordingExtractor(),
        source=source,
        target=None,
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    assert len(dataset) == 2
    assert dataset.expand_encoder_eval_poles is False


@pytest.mark.parametrize("source", ["factual", "alternative", "diff", "both"])
def test_encoder_eval_expands_one_pole_when_flag_set(source):
    """One-pole encoder eval expands both poles when expand_encoder_eval_poles=True."""
    dataset = SignalTrainingDatasetBase(
        _FixedPairRows(n=1),
        _RecordingExtractor(),
        source=source,
        target=None,
        expand_encoder_eval_poles=True,
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    if source == "both":
        # source='both' visits both poles via its own mapping (len==2 for n=1).
        assert len(dataset) == 2
    else:
        assert len(dataset) == 2
    labels = [int(dataset[i]["label"]) for i in range(2)]
    assert set(labels) == {1, -1}


def test_factual_encoder_eval_expand_enables_correlation_from_one_pole_row():
    """Regression: source='factual' one-pole data must not crash encoder metrics."""
    dataset = SignalTrainingDatasetBase(
        _FixedPairRows(n=1),
        _RecordingExtractor(),
        source="factual",
        target=None,
        expand_encoder_eval_poles=True,
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    assert len(dataset) == 2
    labels = [int(dataset[i]["label"]) for i in range(2)]
    assert labels == [1, -1]
    metrics = get_encoder_metrics_from_dataframe(
        pd.DataFrame(
            {
                "encoded": [1.0 if lab == 1 else -1.0 for lab in labels],
                "label": [float(lab) for lab in labels],
                "type": ["training", "training"],
                "source_id": [dataset[i]["factual_id"] for i in range(2)],
            }
        )
    )
    assert metrics["n_samples"] == 2
    assert metrics["all_data"]["correlation"] == pytest.approx(1.0)


def test_factual_training_len_does_not_expand():
    dataset = SignalTrainingDatasetBase(
        _FixedPairRows(n=2),
        _RecordingExtractor(),
        source="factual",
        target="diff",
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    assert len(dataset) == 2


def test_both_encoder_only_expands_two_base_rows_to_four():
    dataset = SignalTrainingDatasetBase(
        _FixedPairRows(n=2),
        _RecordingExtractor(),
        source="both",
        target=None,
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    assert len(dataset) == 4
    labels = [dataset[i]["label"] for i in range(4)]
    assert labels == [1, -1, 1, -1]


def test_both_training_len_does_not_expand():
    dataset = SignalTrainingDatasetBase(
        _FixedPairRows(n=2),
        _RecordingExtractor(),
        source="both",
        target="diff",
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    assert len(dataset) == 2


def test_both_training_single_row_still_one_batch():
    """Training keeps one pole per batch; a single row is still length 1."""
    dataset = SignalTrainingDatasetBase(
        _FixedPairRows(n=1),
        _RecordingExtractor(),
        source="both",
        target="diff",
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    assert len(dataset) == 1
    assert dataset[0]["label"] == 1


def test_both_encoder_expand_enables_correlation_from_one_factual_row():
    """Regression for one-pole validation: expand must supply two labels for correlation.

    Without expand, a single factual eval row raises (only one non-neutral / one class).
    With target=None, the same base row yields +1 and -1 for any training source.
    """
    dataset = SignalTrainingDatasetBase(
        _FixedPairRows(n=1),
        _RecordingExtractor(),
        source="both",
        target=None,
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    assert len(dataset) == 2

    # What in-training eval used to look like with alternation only (one index):
    single_row = dataset[0]
    with pytest.raises(ValueError, match="at least 2 non-neutral"):
        get_encoder_metrics_from_dataframe(
            pd.DataFrame(
                {
                    "encoded": [0.5],
                    "label": [float(single_row["label"])],
                    "type": ["training"],
                    "source_id": [single_row["factual_id"]],
                }
            )
        )

    expanded = [dataset[i] for i in range(len(dataset))]
    labels = [int(r["label"]) for r in expanded]
    assert labels == [1, -1]
    assert {r["factual_id"] for r in expanded} == {"3SG", "3PL"}

    # Distinct encoded values aligned with labels → defined correlation.
    encoder_df = pd.DataFrame(
        {
            "encoded": [1.0 if lab == 1 else -1.0 for lab in labels],
            "label": [float(lab) for lab in labels],
            "type": ["training", "training"],
            "source_id": [r["factual_id"] for r in expanded],
            "target_id": [r["alternative_id"] for r in expanded],
        }
    )
    metrics = get_encoder_metrics_from_dataframe(encoder_df)
    assert metrics["n_samples"] == 2
    assert metrics["all_data"]["correlation"] == pytest.approx(1.0)


def test_both_encoder_expand_rejects_identical_labels_without_poles():
    """Sanity: if somehow only +1 labels are present, metrics still refuse."""
    with pytest.raises(ValueError, match="only one label class"):
        get_encoder_metrics_from_dataframe(
            pd.DataFrame(
                {
                    "encoded": [0.1, 0.2],
                    "label": [1.0, 1.0],
                    "type": ["training", "training"],
                }
            )
        )


@pytest.mark.parametrize(
    "source,direction,expected",
    [
        ("both", 1.0, -1.0),
        ("both", -1.0, 1.0),
        ("factual", 1.0, -1.0),
    ],
)
def test_feature_factor_both_matches_factual(source, direction, expected):
    assert feature_factor_from_encoding_direction(direction, source) == expected


def test_encoding_view_sign_both_matches_factual():
    assert encoding_view_sign_for_source("both", "factual") == 1.0
    assert encoding_view_sign_for_source("both", "counterfactual") == -1.0
    assert encoding_view_sign_for_source("both", "transition") == 1.0
    assert encoding_view_sign_for_source("both", "factual") == encoding_view_sign_for_source(
        "factual", "factual"
    )


def test_resolve_eval_group_both_uses_factual_id():
    trainer = TextPredictionTrainer.__new__(TextPredictionTrainer)
    trainer._training_args = TrainingArguments(source="both", target="diff")
    assert trainer._resolve_eval_group("both") == "factual_id"
    assert trainer._resolve_eval_group("factual") == "factual_id"
    assert trainer._resolve_eval_group("alternative") == "feature_class_id"


def test_create_gradient_training_dataset_passes_both():
    trainer = TextPredictionTrainer.__new__(TextPredictionTrainer)
    trainer._training_args = TrainingArguments(source="both", target="diff")

    tokenizer = MockTokenizer()
    model = MagicMock()
    model.tokenizer = tokenizer
    model.gradiend.torch_dtype = torch.float32
    model.gradiend.device_encoder = torch.device("cpu")
    model.gradient_creator = MagicMock(return_value=torch.randn(4))

    raw = MagicMock()
    raw.__len__ = MagicMock(return_value=1)
    raw.batch_size = 1

    dataset = trainer.create_gradient_training_dataset(raw, model)
    assert dataset.source == "both"
    assert dataset.target == "diff"


def test_both_applies_one_side_to_entire_multi_item_batch():
    """All rows in a batch share the same compiled pole (batch-level, not per-row)."""

    class _MultiBatchRows:
        batch_size = 2

        def __len__(self):
            return 4

        def __getitem__(self, idx):
            if idx % 2 == 0:
                fac, alt = 1.0, 10.0
            else:
                fac, alt = 2.0, 20.0
            return {
                "factual": {"x": torch.tensor([fac])},
                "alternative": {"x": torch.tensor([alt])},
                "label": 1,
                "factual_id": "3SG",
                "alternative_id": "3PL",
                "is_identity_transition": False,
            }

    class _DictExtractor:
        signal = Signal.gradient()

        def __call__(
            self,
            factual_inputs=None,
            alternative_inputs=None,
            *,
            requires_factual=True,
            requires_alternative=True,
        ):
            factual = factual_inputs["x"].reshape(-1).detach().clone() if requires_factual else None
            alternative = (
                alternative_inputs["x"].reshape(-1).detach().clone() if requires_alternative else None
            )
            return SignalBatch.from_factual_alternative(
                factual,
                alternative,
                signal_id="gradient",
            )

    dataset = SignalTrainingDatasetBase(
        _MultiBatchRows(),
        _DictExtractor(),
        source="both",
        target="diff",
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    assert len(dataset) == 2

    even = dataset[0]
    assert even["source"].tolist() == [1.0, 2.0]
    assert even["target"].tolist() == [-9.0, -18.0]
    assert even["label"] == [1, 1]
    assert even["factual_id"] == ["3SG", "3SG"]

    odd = dataset[1]
    assert odd["source"].tolist() == [10.0, 20.0]
    assert odd["target"].tolist() == [9.0, 18.0]
    assert odd["label"] == [-1, -1]
    assert odd["factual_id"] == ["3PL", "3PL"]


def test_both_repeated_access_does_not_corrupt_underlying_rows():
    rows = _FixedPairRows(n=2)
    dataset = SignalTrainingDatasetBase(
        rows,
        _RecordingExtractor(),
        source="both",
        target="diff",
        signal=Signal.gradient(),
        device=torch.device("cpu"),
    )
    first = dataset[1]
    second = dataset[1]
    assert first["label"] == second["label"] == -1
    assert first["source"].tolist() == second["source"].tolist() == [10.0, 20.0]
    # Underlying stored template row remains unswapped.
    raw = rows[0]
    assert raw["label"] == 1
    assert raw["factual_id"] == "3SG"
