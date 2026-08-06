"""
T5 seq2seq GRADIEND workflow with multi-token decoder sequence cloze.

**Experimental / known non-convergent example** —
``seq2seq_decoder_sequence_cloze`` does not currently converge reliably for this
workflow and is intentionally excluded from the example smoke test suite. It is
kept as an experimental implementation example, not as a passing convergence
demonstration. Prefer ``train_seq2seq_encoder_mlm.py`` (encoder-side MLM, default
for ``prediction_objective="auto"`` on T5/BART); that workflow does converge and
is covered by the example smoke tests.

Uses the same published English pronoun data as ``train_english_pronouns.py``.

Run:
    python -m gradiend.examples.train_seq2seq_decoder_sequence
"""

from __future__ import annotations

from pathlib import Path

from gradiend import PostPruneConfig, TextPredictionTrainer, TrainingArguments
from gradiend.examples.english_pronoun_datasets import (
    EN_PRONOUN_HF_SPLITS,
    EN_PRONOUNS_HF_DATASET,
    load_english_pronoun_neutral_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / "runs" / "examples" / "t5_decoder_sequence"
TARGET_CLASSES = ("3SG", "3PL")


if __name__ == "__main__":
    neutral_data = load_english_pronoun_neutral_data()
    print(f"=== English pronoun data: using {EN_PRONOUNS_HF_DATASET} ===")

    args = TrainingArguments(
        experiment_dir=str(EXPERIMENT_DIR),
        prediction_objective="seq2seq_decoder_sequence_cloze",
        decoder_sequence_cloze_rhs_window=0,
        train_batch_size=4,
        train_max_size=1000,
        eval_steps=100,
        max_steps=1000,
        source="alternative",
        target="diff",
        learning_rate=1e-5,
        use_cache=False,
        fail_on_non_convergence=True,
        add_identity_for_other_classes=False,
        post_prune_config=PostPruneConfig(topk=0.001, part="decoder-weight"),
    )
    trainer = TextPredictionTrainer(
        model="t5-small",
        run_id="t5_3sg_3pl_sequence_cloze",
        hf_dataset=EN_PRONOUNS_HF_DATASET,
        hf_splits=EN_PRONOUN_HF_SPLITS,
        target_classes=list(TARGET_CLASSES),
        eval_neutral_data=neutral_data,
        args=args,
    )

    print("\n=== T5 pronouns: 3SG vs 3PL (seq2seq_decoder_sequence_cloze) ===")
    trainer.train()
    print(f"Model saved at {trainer.model_path}")

    trainer.plot_training_convergence()
    stats = trainer.get_training_stats() or {}
    print(f"  correlation={stats.get('training_stats', {}).get('correlation')}")

    print("\n=== Decoder evaluation ===")
    dec = trainer.evaluate_decoder(plot=True)
    for cls in TARGET_CLASSES:
        if cls in dec:
            print(f"  {cls}: {dec[cls]}")

    print("\n=== Done ===")
