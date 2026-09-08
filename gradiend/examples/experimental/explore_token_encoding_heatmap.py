"""
Load a trained ACTIEND checkpoint and write a token-encoding heatmap HTML.

Does not retrain. Uses the existing english-pronouns activation run by default.

Run from the gradiend repo root:

    python -m gradiend.examples.experimental.explore_token_encoding_heatmap
"""

from __future__ import annotations

import os
from typing import List

from gradiend import Signal, TextPredictionConfig, TextPredictionTrainer, TrainingArguments
from gradiend.examples.english_pronoun_datasets import (
    EN_PRONOUN_HF_SPLITS,
    EN_PRONOUNS_HF_DATASET,
    load_english_pronoun_neutral_data,
)
from gradiend.trainer.core import SignalScope

MODEL_NAME = "gpt2"
EXPERIMENT_DIR = "runs/english_pronouns_activation_factual"
RUN_ID = "pronoun_3sg_3pl_activation"
CHECKPOINT_DIR = os.path.join(EXPERIMENT_DIR, RUN_ID, "model")
OUTPUT_HTML = os.path.join(EXPERIMENT_DIR, RUN_ID, "token_encoding_highlight.html")
# Use zero-center by default so explore does not trigger evaluate_encoder.
# Pass --neutral to center colors on the cached neutral encoder baseline.
COLOR_CENTER = "zero"

EXAMPLE_TEXTS: List[str] = [
    "The teacher said he would arrive soon",
    "The teacher said she would arrive soon",
    "The teachers said they would arrive soon",
    "Someone left their bag on the chair",
]


def _make_trainer() -> TextPredictionTrainer:
    neutral_data = load_english_pronoun_neutral_data()
    config = TextPredictionConfig(
        run_id=RUN_ID,
        hf_dataset=EN_PRONOUNS_HF_DATASET,
        hf_splits=EN_PRONOUN_HF_SPLITS,
        target_classes=["3SG", "3PL"],
        neutral_data=neutral_data,
    )
    args = TrainingArguments(
        experiment_dir=EXPERIMENT_DIR,
        train_batch_size=1,
        eval_batch_size=1,
        gradiend_batch_size=1,
        num_train_epochs=5,
        max_steps=1000,
        eval_steps=100,
        bias_encoder=True,
        learning_rate=1e-5,
        target="diff",
        source="factual",
        signal=Signal.activation(),
        add_neutral_identity_transitions=True,
        signal_scope=SignalScope.layer(9),
        fail_on_non_convergence=False,
        use_cache=True,
    )
    return TextPredictionTrainer(model=MODEL_NAME, config=config, args=args)


def main(color_center: str = COLOR_CENTER) -> None:
    if not os.path.isdir(CHECKPOINT_DIR):
        raise FileNotFoundError(
            f"Missing ACTIEND checkpoint at {CHECKPOINT_DIR}. "
            "Train first via train_english_pronouns_activation.py"
        )

    trainer = _make_trainer()
    model = trainer.get_model(load_directory=CHECKPOINT_DIR)
    print(f"loaded: {CHECKPOINT_DIR}")
    print(f"uses_activations: {model.uses_activations}")
    print(f"activation sites ({len(model.activation_site_modules)}): {model.activation_site_modules}")
    print(f"color_center: {color_center}")

    sections = [
        "<html><head><meta charset=\"utf-8\"><title>ACTIEND token encoding heatmap</title></head><body>",
        "<h1>ACTIEND token encoding heatmap</h1>",
        "<p>Each token is colored by the ACTIEND encoding of the activation at that position.</p>",
    ]

    for text in EXAMPLE_TEXTS:
        html, rows = trainer.visualizer.highlight_token_encoding(
            text,
            show=False,
            return_rows=True,
            color_center=color_center,
            color_extent=1.0,
            title=text,
        )
        scores = ", ".join(f"{row.token!r}:{row.encoded:.3f}" for row in rows)
        print(f"\ntext: {text!r}")
        print(f"  scores: {scores}")
        sections.append(html)
        sections.append("<hr>")

    sections.append("</body></html>")
    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as handle:
        handle.write("\n".join(sections))
    print(f"\nwrote: {OUTPUT_HTML}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--neutral",
        action="store_true",
        help="Center colors on the neutral encoder baseline (may run evaluate_encoder).",
    )
    args = parser.parse_args()
    main(color_center="neutral" if args.neutral else COLOR_CENTER)
