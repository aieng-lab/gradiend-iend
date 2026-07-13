"""Shared mock models/tokenizers for unit tests (importable from test modules)."""

from __future__ import annotations

import torch
import torch.nn as nn


def bind_trainer_cache_resolver(trainer_stub):
    """Attach artifact cache resolution to lightweight decoder-eval test doubles."""
    from unittest.mock import MagicMock

    from gradiend.trainer.core.feature_definition import FeatureLearningDefinition

    if not hasattr(trainer_stub, "_training_args") or trainer_stub._training_args is None:
        trainer_stub._training_args = MagicMock()
    trainer_stub._resolve_artifact_use_cache = (
        FeatureLearningDefinition._resolve_artifact_use_cache.__get__(trainer_stub)
    )
    return trainer_stub


class SimpleMockModel(nn.Module):
    """Simple mock base model with minimal parameters for testing."""

    def __init__(
        self,
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        name_or_path="mock-model",
        dtype=torch.float32,
    ):
        super().__init__()
        self.name_or_path = name_or_path
        self._dtype = dtype
        self.config = type(
            "Config",
            (),
            {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "num_hidden_layers": num_layers,
            },
        )()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.GELU(),
                )
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.cls = type("Cls", (), {"predictions": self.classifier})()
        self.to(dtype=dtype)

    @property
    def device(self):
        """Device of the first parameter (required by rewrite_base_model)."""
        params = list(self.parameters())
        return params[0].device if params else torch.device("cpu")

    @property
    def dtype(self):
        """Return the dtype of the first parameter (PyTorch convention)."""
        if len(list(self.parameters())) > 0:
            return next(self.parameters()).dtype
        return self._dtype

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if input_ids is None:
            input_ids = kwargs.get("input_ids")
        if input_ids is None:
            logits = torch.zeros(1, 10, self.config.vocab_size, dtype=self.dtype)
            loss = torch.tensor(0.0, dtype=self.dtype, requires_grad=True)
            return type("Output", (), {"logits": logits, "loss": loss})()

        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        x = self.embeddings(input_ids)
        for layer in self.encoder:
            x = layer(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
            if len(logits.shape) == 3:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            else:
                loss = torch.tensor(0.0, dtype=self.dtype, requires_grad=True)

        return type("Output", (), {"logits": logits, "loss": loss})()


class _TokenizedBatch(dict):
    """Dict-like tokenizer output that supports .to(device) like HF BatchEncoding."""

    def to(self, device):
        out = _TokenizedBatch()
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.to(device)
            else:
                out[key] = value
        return out


class MockTokenizer:
    """Simple mock tokenizer."""

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.name_or_path = "mock-tokenizer"
        self.mask_token = "[MASK]"
        self.mask_token_id = 103
        self.pad_token = "[PAD]"
        self.pad_token_id = 0
        self.eos_token = "[EOS]"
        self.eos_token_id = 102
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.vocab = {f"token_{i}": i for i in range(vocab_size)}
        self.vocab.update({"[MASK]": 103, "[PAD]": 0, "[CLS]": 101, "[SEP]": 102})
        self.all_special_ids = [0, 101, 102, 103]  # PAD, CLS, SEP, MASK

    def convert_tokens_to_ids(self, tokens):
        """Convert tokens to IDs."""
        if isinstance(tokens, str):
            return self.vocab.get(tokens, 0)
        if isinstance(tokens, list):
            return [self.vocab.get(token, 0) for token in tokens]
        return tokens

    def __call__(
        self,
        text,
        return_tensors=None,
        padding=True,
        truncation=True,
        max_length=48,
        add_special_tokens=True,
        **kwargs,
    ):
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        all_input_ids = []
        for item in texts:
            tokens = item.split()[: max_length - 2] if truncation else item.split()
            token_ids = [self.vocab.get(token, 1) for token in tokens]
            if add_special_tokens:
                token_ids = [self.vocab["[CLS]"]] + token_ids + [self.vocab["[SEP]"]]
            if padding and len(token_ids) < max_length:
                token_ids = token_ids + [self.vocab["[PAD]"]] * (max_length - len(token_ids))
            all_input_ids.append(token_ids[:max_length])
        result = {"input_ids": all_input_ids[0] if not is_batch else all_input_ids}
        if return_tensors == "pt":
            ids = result["input_ids"]
            result["input_ids"] = torch.tensor(ids if is_batch else [ids])
            mask = [
                [1 if token_id != self.vocab["[PAD]"] else 0 for token_id in row]
                for row in all_input_ids
            ]
            result["attention_mask"] = torch.tensor(mask)
            return _TokenizedBatch(result)
        return result

    def tokenize(self, text, **kwargs):
        """Return list of token strings (space-split)."""
        return text.split()

    def convert_tokens_to_string(self, tokens):
        """Join token strings back to a single string."""
        return " ".join(tokens) if isinstance(tokens, list) else str(tokens)

    def encode(self, text, add_special_tokens=False, **kwargs):
        tokens = text.split()
        return [self.vocab.get(token, 1) for token in tokens]

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        reverse_vocab = {value: key for key, value in self.vocab.items()}
        tokens = [reverse_vocab.get(token_id, f"<unk_{token_id}>") for token_id in token_ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if not (token.startswith("[") and token.endswith("]"))]
        return " ".join(tokens)
