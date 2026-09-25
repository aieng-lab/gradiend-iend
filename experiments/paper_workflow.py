from gradiend import (
    PostPruneConfig,
    PrePruneConfig,
    TextPredictionTrainer,
    TrainingArguments,
    check_plot_environment,
)
from gradiend.examples.english_pronoun_datasets import (
    EN_PRONOUN_HF_SPLITS,
    EN_PRONOUNS_HF_DATASET,
    load_english_pronoun_neutral_data,
)

check_plot_environment()


neutral_data = load_english_pronoun_neutral_data()

model="bert-base-cased"

args = TrainingArguments(
    train_batch_size=16,
    max_steps=200,
    eval_steps=20,
    learning_rate=1e-5,
    experiment_dir=f'runs/demonstration-{model}',
    use_cache=False,
    pre_prune_config=PrePruneConfig(n_samples=8, topk=0.1),
    post_prune_config=PostPruneConfig(topk=0.01),
)
trainer = TextPredictionTrainer(
    model=model,
    hf_dataset=EN_PRONOUNS_HF_DATASET,
    hf_splits=EN_PRONOUN_HF_SPLITS,
    target_classes=["3SG", "3PL"],
    eval_neutral_data=neutral_data,
    img_format="pdf", # different from paper script, to persist the figures
    args=args,
)

trainer.train()
trainer.plot_training_convergence(class_spread="iqr")

enc_result = trainer.evaluate_encoder(plot=True, plot_kwargs={'figsize': (5, 2), 'legend_fontsize': 7})
print("Correlation:", enc_result["correlation"])
print("Mean by class:", enc_result["mean_by_feature_class"])
dec = trainer.evaluate_decoder(plot=True, target_class="3SG", use_cache=True, plot_kwargs={'figsize': (7, 4.5)}) #, lrs=[-10, -1, -0.1, -0.01, -0.001])
print(dec["3SG"]["learning_rate"])
changed_base_model = trainer.rewrite_base_model(decoder_results=dec, target_class="3SG")
