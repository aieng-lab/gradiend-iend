"""Shared tiny real GPT-2 harness for signal/trainer tests (CPU, no model download).

A randomly initialised 2-layer GPT-2 with the cached ``gpt2`` tokenizer is plugged into the
real :class:`TextPredictionTrainer` through the same ``_load_model`` patch the workflow tests use,
so everything above the backbone (datasets, signal extractors, training loop, evaluation,
checkpointing) is the production code.  ``tiny_gpt2_available()`` lets tests skip when the
tokenizer is not in the local Hugging Face cache.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from unittest.mock import patch

import pandas as pd
import torch

from gradiend import TextFilterConfig, TextPredictionDataCreator, TextPredictionTrainer, TrainingArguments

MINI_TEXTS = [
    "The chef tasted the soup, then he added a pinch of pepper.",
    "She opened the window and watched the leaves fall.",
    "The dog ran to the door; it wanted to go outside.",
    "The mechanic wiped his hands and said the car would be ready.",
    "The gardener pruned the roses and he left the cuttings by the gate.",
    "The players huddled on the pitch before they ran back.",
    "The committee members met on Tuesday and they voted.",
    "The volunteers packed the boxes and said they would load the van.",
    "The staff members finished the inventory and they reported the count.",
    "The panel members discussed the proposal and they reached a consensus.",
    "The teacher smiled at the class and she handed out the papers.",
    "The farmers rose early and they fed the animals before sunrise.",
]

NEUTRAL_TEXTS = [
    "The report will be ready for review by the end of the week.",
    "The museum opens at nine and closes at six on weekdays.",
    "A cold wind blew across the empty field all afternoon long.",
    "The library closes early on Sundays during the summer months.",
    "Several old maps were stored in the cabinet near the window.",
    "The bridge was repaired last year after the heavy spring floods.",
    "Fresh bread is baked every morning at the small corner bakery.",
    "The train from the coast arrives shortly after noon each day.",
    "Rain fell steadily over the hills throughout the entire night.",
    "The concert hall was filled with soft light and quiet music.",
    "Two lanterns hung above the narrow door of the old tavern.",
    "The festival begins next Friday and lasts for three full days.",
]

HIDDEN = 16
LAYERS = 2


def tiny_gpt2_available() -> bool:
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        return True
    except Exception:
        return False


def make_tiny_gpt2(seed: int = 0):
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    tok.pad_token = tok.eos_token
    cfg = GPT2Config(n_layer=LAYERS, n_embd=HIDDEN, n_head=2, n_positions=1024, n_ctx=1024,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = GPT2LMHeadModel(cfg)
    model.name_or_path = "gpt2"
    model.eval()
    return model, tok


def make_data():
    creator = TextPredictionDataCreator(
        base_data=MINI_TEXTS,
        feature_targets=[
            TextFilterConfig(targets=["he", "she", "it"], id="3SG"),
            TextFilterConfig(targets=["they"], id="3PL"),
        ],
        min_left_context_words=0,
    )
    training = creator.generate_training_data(
        max_size_per_class=15, min_rows_per_class_for_split=0, val_ratio=0.0, test_ratio=0.2
    )
    neutral = pd.DataFrame(
        {
            "text": NEUTRAL_TEXTS,
            "split": ["train"] * (len(NEUTRAL_TEXTS) - 4) + ["test"] * 4,
        }
    )
    return training, neutral


def training_arguments(experiment_dir: str, **overrides) -> TrainingArguments:
    values = dict(
        train_batch_size=1,
        base_gradient_batch_size=1,
        eval_steps=3,
        num_train_epochs=1,
        max_steps=6,
        learning_rate=1e-3,
        experiment_dir=experiment_dir,
        use_cache=False,
        do_eval=False,
        source="both",
        target="diff",
        seed=7,
        max_seeds=1,
        min_convergent_seeds=1,
        torch_dtype=torch.float32,
        encoder_decoder_same_device=True,
    )
    values.update(overrides)
    return TrainingArguments(**values)


@contextmanager
def tiny_backbone(model, tokenizer):
    """Patch ``_load_model`` so every (re)load returns the same in-memory tiny backbone."""

    def _load(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):  # noqa: ANN001
        return model, tok

    tok = tokenizer
    with patch(
        "gradiend.trainer.text.common.model_base.TextModelWithGradiend._load_model", classmethod(_load)
    ):
        yield


def make_trainer(args: TrainingArguments, model, tokenizer, training=None, neutral=None):
    if training is None:
        training, neutral = make_data()
    return TextPredictionTrainer(
        model="gpt2",
        data=training,
        neutral_data=neutral,
        split_train_ratio=0.8,
        split_val_ratio=0.0,
        split_test_ratio=0.2,
        max_counterfactuals_per_sentence=1,
        args=args,
    )
