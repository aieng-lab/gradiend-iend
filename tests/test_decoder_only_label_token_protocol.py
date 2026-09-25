"""One label-token rule for decoder-only items, versioned so old artifacts keep theirs.

The dataset used ``vocab[label]`` (``she``, the mid-word token) while
``create_inputs`` used ``tokenizer(" " + label)`` (``▁she``). The label sits on the
last prefix token and is predicted from the token before it, so the leading-space
variant is the natural one. ``label_token_protocol`` selects the rule per run:
``canonical`` (new runs) or ``legacy`` (everything already trained).
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.text.prediction.dataset import (
    TextTrainingDataset,
    decoder_only_label_token_ids,
)
from gradiend.trainer.text.prediction.trainer import TextPredictionTrainer
from tests.testing_mocks import MockTokenizer


class SpaceVariantTokenizer(MockTokenizer):
    """Whitespace tokenizer that, like BPE/SentencePiece, has a distinct id for ' word'."""

    def __init__(self):
        super().__init__()
        self.vocab["she"] = 500          # mid-word variant
        self.vocab["▁she"] = 501    # leading-space variant

    def __call__(self, text, *args, **kwargs):
        if kwargs.get("add_special_tokens") is False and isinstance(text, str) and text.startswith(" "):
            return {"input_ids": [self.vocab["▁" + text.strip()]]}
        return super().__call__(text, *args, **kwargs)


def _dataset(protocol=None):
    data = pd.DataFrame({
        "masked": ["token_5 token_6 [MASK]"],
        "factual": ["she"],
        "alternative": ["token_8"],
        "factual_class": ["c1"],
        "alternative_class": ["c2"],
        "factual_id": [1],
        "alternative_id": [2],
        "label": ["positive"],
        "feature_class_id": [1],
        "feature_pole": ["pos"],
    })
    kwargs = {} if protocol is None else {"label_token_protocol": protocol}
    return TextTrainingDataset(
        data=data, tokenizer=SpaceVariantTokenizer(), batch_size=1, is_decoder_only_model=True, **kwargs
    )


def _labelled_id(dataset):
    item = dataset._create_item("token_5 token_6", "she")
    labels = item["labels"]
    return labels[labels != -100].item()


def test_helper_returns_the_leading_space_variant():
    tok = SpaceVariantTokenizer()
    assert decoder_only_label_token_ids(tok, "she") == [tok.vocab["▁she"]]


def test_canonical_protocol_labels_with_the_leading_space_token():
    assert _labelled_id(_dataset("canonical")) == 501


def test_legacy_protocol_keeps_the_old_vocab_label():
    assert _labelled_id(_dataset("legacy")) == 500


def test_direct_dataset_construction_defaults_to_legacy():
    """Callers that never heard of the protocol must not change behaviour."""
    assert _labelled_id(_dataset(None)) == 500


def test_dataset_rejects_unknown_protocol():
    with pytest.raises(ValueError, match="label_token_protocol"):
        _dataset("bogus")


def test_new_training_arguments_default_to_canonical():
    assert TrainingArguments().label_token_protocol == "canonical"


def test_serialized_arguments_without_the_key_stay_legacy():
    """A checkpoint written before the protocol existed must not be re-interpreted."""
    payload = TrainingArguments().to_dict()
    payload.pop("label_token_protocol")
    assert TrainingArguments.from_dict(payload).label_token_protocol == "legacy"


@pytest.mark.parametrize("value", ["canonical", "legacy"])
def test_protocol_round_trips_through_to_dict(value):
    args = TrainingArguments(label_token_protocol=value)
    assert TrainingArguments.from_dict(args.to_dict()).label_token_protocol == value


def test_training_arguments_reject_unknown_protocol():
    with pytest.raises(ValueError, match="label_token_protocol"):
        TrainingArguments(label_token_protocol="bogus")


@pytest.mark.parametrize("value", ["canonical", "legacy"])
def test_trainer_hands_its_protocol_to_every_dataset(value):
    fake_trainer = SimpleNamespace(training_args=TrainingArguments(label_token_protocol=value))
    assert TextPredictionTrainer._label_token_protocol(fake_trainer) == value


def test_trainer_without_training_args_falls_back_to_legacy():
    assert TextPredictionTrainer._label_token_protocol(SimpleNamespace(training_args=None)) == "legacy"


def test_every_model_a_trainer_creates_is_stamped_with_its_protocol():
    """Single-row scorers read the protocol from the model; it must come from the trainer.

    Goes through the real creation path (``create_model_with_gradiend``), which
    ``get_model``, ``load_model`` and training all share.
    """

    class _Model:
        @classmethod
        def from_pretrained(cls, load_directory, **kwargs):
            return cls()

    class _Trainer(TextPredictionTrainer):
        def __init__(self, protocol):  # bypass the heavy constructor
            self._training_args = TrainingArguments(label_token_protocol=protocol)

        @property
        def model_with_gradiend_cls(self):
            return _Model

    for protocol in ("canonical", "legacy"):
        model = _Trainer(protocol).create_model_with_gradiend("ignored")
        assert model.label_token_protocol == protocol


def test_training_label_token_id_matches_the_dataset_for_both_protocols():
    from gradiend.trainer.text.prediction.dataset import training_label_token_id

    tok = SpaceVariantTokenizer()
    assert training_label_token_id(tok, "she", "canonical") == 501
    assert training_label_token_id(tok, "she", "legacy") == 500
    with pytest.raises(ValueError, match="label_token_protocol"):
        training_label_token_id(tok, "she", "bogus")
