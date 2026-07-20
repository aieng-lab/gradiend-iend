"""
English pronoun workflow using an activation signal instead of raw gradients.

This mirrors gradiend/examples/train_english_pronouns.py, but trains GRADIEND on
the filled prediction-span activation from BERT's default activation scope.

Requires: pip install gradiend[data]
"""

from gradiend import (
    Signal,
    TextPredictionConfig,
    TextPredictionTrainer,
    TrainingArguments, __all__,
)
from gradiend.examples.create_english_pronoun_data import ensure_english_pronoun_data
from gradiend.trainer.core import SignalScope

DATA_DIR = "data/english_pronouns"
MODEL_NAME = "bert-base-uncased"
MODEL_NAME = "gpt2"


if __name__ == "__main__":
    training_path, neutral_path = ensure_english_pronoun_data(output_dir=DATA_DIR)

    config = TextPredictionConfig(
        run_id="pronoun_3sg_3pl_activation",
        data=training_path,
        target_classes=["3SG", "3PL"],
        eval_neutral_data=neutral_path,
    )

    args = TrainingArguments(
        experiment_dir="runs/english_pronouns_activation",
        train_batch_size=1,
        eval_batch_size=1,
        gradiend_batch_size=1,
        num_train_epochs=5,
        max_steps=5000,
        eval_steps=100,
        learning_rate=5e-5,
        target="diff",
        source="factual",
        signal=Signal.activation(),
        #signal=Signal.gradient(), #todo token_selector="mask"),
        signal_scope=SignalScope.from_values(
            activation_sites=["bert.encoder.layer.10.output"]
        ),
        fail_on_non_convergence=False,
    )

    print("\n=== Training activation GRADIEND ===")
    print(f"  model: {MODEL_NAME}")
    print("  activation scope: default backbone/text tower")
    print("  token selector: prediction span")

    trainer = TextPredictionTrainer(
        model=MODEL_NAME,
        config=config,
        args=args,
    )
    trainer.train()
    trainer.plot_training_convergence(class_spread='ci95')

    stats = trainer.get_training_stats()
    ts = stats.get("training_stats", {}) if stats else {}
    print(f"  correlation={ts.get('correlation')}")
    print("\n=== Done ===")
