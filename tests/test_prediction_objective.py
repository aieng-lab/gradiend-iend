import os

import pandas as pd
import torch

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.text.prediction.decoder_eval_utils import (
    _build_vocab_norm_map,
    _clm_next_token_probs_for_prefixes,
    compute_probability_shift_score_clm_sequence,
    compute_probability_shift_score_clm,
    evaluate_probability_shift_score,
)
from gradiend.trainer.text.prediction.prediction_objective import (
    DecoderMLMHeadObjective,
    resolve_prediction_objective,
)
from tests.testing_mocks import MockTokenizer, SimpleMockModel


class TinyTokenizer:
    def __init__(self, *, padding_side="right"):
        self.vocab = {"[PAD]": 0, "start": 1, "a": 2, "b": 3, "ok": 4, "bad": 5}
        self.inv = {v: k for k, v in self.vocab.items()}
        self.mask_token = None
        self.mask_token_id = None
        self.pad_token_id = 0
        self.padding_side = padding_side

    def __call__(self, text, add_special_tokens=False, **_kwargs):
        if isinstance(text, list):
            ids = [self(t, add_special_tokens=add_special_tokens)["input_ids"] for t in text]
            max_len = max(len(x) for x in ids)
            if self.padding_side == "left":
                padded = [[self.pad_token_id] * (max_len - len(x)) + x for x in ids]
                masks = [[0] * (max_len - len(x)) + [1] * len(x) for x in ids]
            else:
                padded = [x + [self.pad_token_id] * (max_len - len(x)) for x in ids]
                masks = [[1] * len(x) + [0] * (max_len - len(x)) for x in ids]
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks),
            }
        return {"input_ids": [self.vocab[tok] for tok in str(text).split() if tok]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.inv[int(i)] for i in ids)


class TinyCausalModel(torch.nn.Module):
    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None):
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 6)
        logits[..., 2] = 5.0  # P(a | start)
        logits[..., 3] = 5.0  # P(b | start)
        for b in range(input_ids.shape[0]):
            for pos in range(input_ids.shape[1]):
                current = int(input_ids[b, pos])
                if current == 2:
                    logits[b, pos, 4] = 10.0  # a -> ok
                elif current == 3:
                    logits[b, pos, 5] = 10.0  # b -> bad
        return type("Output", (), {"logits": logits})()


class SpaceAwareTokenizer:
    def __init__(self):
        self.vocab = {
            "[PAD]": 0,
            "thought": 1,
            "Ġ": 2,
            "Ġshe": 3,
            "she": 4,
            "bad": 5,
            "Ġbad": 6,
        }
        self.inv = {v: k for k, v in self.vocab.items()}
        self.mask_token = None
        self.mask_token_id = None
        self.pad_token_id = 0

    def _encode(self, text):
        text = str(text)
        if text == "thought":
            return [1]
        if text == "thought ":
            return [1, 2]
        if text == "thought she":
            return [1, 3]
        if text == "thought bad":
            return [1, 6]
        if text == " she":
            return [3]
        if text == " bad":
            return [6]
        if text == "she":
            return [4]
        if text == "bad":
            return [5]
        return [self.vocab[tok] for tok in text.split() if tok in self.vocab]

    def __call__(self, text, add_special_tokens=False, **_kwargs):
        if isinstance(text, list):
            ids = [self._encode(t) for t in text]
            max_len = max(len(x) for x in ids)
            padded = [x + [self.pad_token_id] * (max_len - len(x)) for x in ids]
            masks = [[1] * len(x) + [0] * (max_len - len(x)) for x in ids]
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks),
            }
        return {"input_ids": self._encode(text)}

    def get_vocab(self):
        return dict(self.vocab)

    def tokenize(self, text):
        return [self.inv[i] for i in self._encode(text)]

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, list):
            return [self.vocab[t] for t in tokens]
        return self.vocab[tokens]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, list):
            return [self.inv[int(i)] for i in ids]
        return self.inv[int(ids)]

    def decode(self, ids, skip_special_tokens=True):
        text = ""
        for token_id in ids:
            token = self.inv[int(token_id)]
            if token == "Ġ":
                text += " "
            elif token.startswith("Ġ"):
                text += " " + token[1:]
            elif token != "[PAD]" or not skip_special_tokens:
                text += token
        return text


class SpaceSensitiveCausalModel(torch.nn.Module):
    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None):
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 7)
        for b in range(input_ids.shape[0]):
            for pos in range(input_ids.shape[1]):
                current = int(input_ids[b, pos])
                if current == 1:
                    logits[b, pos, 3] = 10.0  # thought -> " she"
                elif current == 2:
                    logits[b, pos, 5] = 10.0  # standalone space -> bad non-space piece
        return type("Output", (), {"logits": logits})()


def test_clm_sequence_cloze_uses_rhs_context():
    df = pd.DataFrame([
        {
            "masked": "start [MASK] ok",
            "factual": "a",
            "alternative": "b",
            "factual_id": "A",
            "alternative_id": "B",
        }
    ])

    result, per_row = compute_probability_shift_score_clm_sequence(
        TinyCausalModel(),
        TinyTokenizer(),
        df,
        targets={},
        use_row_wise=True,
        return_per_row_df=True,
    )

    assert result["A"]["A"] > 0.98
    assert result["A"]["B"] < 0.02
    assert per_row.loc[0, "P_factual"] > per_row.loc[0, "P_alternative"]


def test_static_target_probability_scoring_can_return_per_sample_rows():
    df = pd.DataFrame([
        {
            "masked": "token_1 [MASK]",
            "label_class": "A",
            "factual_id": "A",
            "alternative_id": "B",
        }
    ])

    result, per_row = evaluate_probability_shift_score(
        SimpleMockModel(vocab_size=200),
        MockTokenizer(vocab_size=200),
        targets={"A": ["token_2"], "B": ["token_3"]},
        eval_data_df=df,
        key_text="masked",
        dataset_class_col="label_class",
        use_row_wise=False,
        return_per_row_df=True,
        objective="mlm_mask_token",
    )

    assert set(result["A"]) == {"A", "B"}
    assert len(per_row) == 1
    assert per_row.loc[0, "row_index"] == 0
    assert per_row.loc[0, "dataset_class"] == "A"
    assert {"p_class_A", "p_class_B"}.issubset(per_row.columns)


def test_clm_static_scoring_uses_last_non_padding_prefix_token_for_any_padding_side():
    df = pd.DataFrame([
        {"masked": "a [MASK]", "label_class": "A"},
        {"masked": "start a [MASK]", "label_class": "A"},
    ])

    for padding_side in ("right", "left"):
        result, per_row = compute_probability_shift_score_clm(
            TinyCausalModel(),
            TinyTokenizer(padding_side=padding_side),
            df,
            targets={"OK": ["ok"], "BAD": ["bad"]},
            key_text="masked",
            dataset_class_col="label_class",
            return_per_row_df=True,
        )

        assert result["A"]["OK"] > 0.98
        assert per_row["p_class_OK"].min() > 0.98
        assert per_row["p_class_BAD"].max() < 0.01


def test_clm_static_scoring_moves_trailing_prefix_space_to_target_continuation():
    df = pd.DataFrame([
        {"masked": "thought [MASK]", "label_class": "A"},
    ])

    result, per_row = compute_probability_shift_score_clm(
        SpaceSensitiveCausalModel(),
        SpaceAwareTokenizer(),
        df,
        targets={"SHE": ["she"], "BAD": ["bad"]},
        key_text="masked",
        dataset_class_col="label_class",
        return_per_row_df=True,
    )

    assert result["A"]["SHE"] > 0.98
    assert result["A"]["BAD"] < 0.01
    assert per_row.loc[0, "p_class_SHE"] > 0.98


def test_clm_next_token_verbose_prints_selected_context_and_top_predictions(capsys):
    _clm_next_token_probs_for_prefixes(
        TinyCausalModel(),
        TinyTokenizer(),
        ["a"],
        torch.device("cpu"),
        verbose=True,
        top_k=3,
    )

    out = capsys.readouterr().out
    assert "CLM next-token debug [0]" in out
    assert "prefix='a'" in out
    assert "context_token='a'" in out
    assert "top 3 next-token predictions" in out
    assert "token='ok'" in out
    assert "prob=" in out


def test_clm_next_token_verbose_shows_trimmed_model_prefix_for_trailing_space(capsys):
    _clm_next_token_probs_for_prefixes(
        SpaceSensitiveCausalModel(),
        SpaceAwareTokenizer(),
        ["thought "],
        torch.device("cpu"),
        verbose=True,
        top_k=1,
    )

    out = capsys.readouterr().out
    assert "prefix='thought '" in out
    assert "model_prefix='thought'" in out
    assert "context_token='thought'" in out
    assert "context_token='Ġ'" not in out
    assert "decoded=' she'" in out


def test_vocab_normalization_groups_gpt_and_sentencepiece_leading_space_markers():
    class Tok:
        def get_vocab(self):
            return {"he": 1, "Ġhe": 2, "▁He": 3, "she": 4}

    norm_map = _build_vocab_norm_map(Tok())

    assert set(norm_map["he"]) == {"he", "Ġhe", "▁He"}
    assert norm_map["she"] == ["she"]


def test_decoder_mlm_head_objective_scores_original_clm_head():
    original = object()

    class Wrapper:
        def to_original_model(self):
            return original

    assert DecoderMLMHeadObjective().resolve_scoring_model(Wrapper(), None, None) is original


def test_explicit_clm_mlm_head_ensures_head_training(tmp_path):
    calls = {}

    class Trainer:
        experiment_dir = str(tmp_path)
        _training_args = TrainingArguments(
            experiment_dir=str(tmp_path),
            prediction_objective="clm_mlm_head",
            decoder_mlm_head_epochs=2,
            decoder_mlm_head_batch_size=3,
            decoder_mlm_head_max_size=4,
        )

        def train_decoder_only_mlm_head(self, model, **kwargs):
            calls["model"] = model
            calls["kwargs"] = kwargs

    trainer = Trainer()
    objective = resolve_prediction_objective(trainer)
    objective.ensure_training_resources(trainer, "base-model")

    assert calls["model"] == "base-model"
    assert calls["kwargs"]["epochs"] == 2
    assert calls["kwargs"]["batch_size"] == 3
    assert calls["kwargs"]["max_size"] == 4
    assert os.path.basename(calls["kwargs"]["output"]) == "decoder_mlm_head"
